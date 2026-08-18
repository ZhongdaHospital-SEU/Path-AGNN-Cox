"""Path-AGNN-Cox model: pathway-constrained adaptive graph neural network.

Architecture:
  1. Gene embedding: per-gene expression -> node features.
  2. L adaptive GAT layers whose attention is restricted to the pathway
     adjacency (edges only among co-regulated genes in the same pathway)
     and modulated by a sample-level malignancy signal s_i:
         alpha_ij = softmax_j( LeakyReLU( a^T [W h_i || W h_j] )
                               * (1 + tanh(beta) * s_i) )
     Each gene is assigned to its primary pathway module, so the graph is a
     true block-diagonal union of pathway subgraphs; attention is computed
     per block (memory-bounded on real TCGA/GEO cohorts with thousands of
     genes) and exported as a sparse COO edge list for rewiring analysis.
  3. Pathway readout: mean-pool node features within each pathway block,
     concatenate -> MLP -> Cox risk score (single scalar).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _blocks_from_ids(ids: torch.Tensor, n_classes: int):
    """Return per-pathway node index tensors for a primary-pathway assignment."""
    return [torch.nonzero(ids == k, as_tuple=False).flatten().to(torch.int64)
            for k in range(n_classes) if (ids == k).any()]


class AdaptiveGATLayer(nn.Module):
    """Pathway-masked graph attention with sample-adaptive malignancy gating.

    Args:
        in_dim, out_dim: feature dims.
        adj: binary adjacency (N, N) built from pathway subgraphs (dense).
        blocks: list of node-index tensors, one per pathway subgraph
            (disjoint primary-pathway blocks). If None, identity propagation.
        negative_slope: LeakyReLU slope for attention logits.
        dropout: feature dropout.
        malignancy_dim: hidden dim of the malignancy MLP (per-sample scalar s).
    """

    def __init__(self, in_dim: int, out_dim: int, adj: torch.Tensor,
                 blocks: list[torch.Tensor] | None = None,
                 negative_slope: float = 0.2, dropout: float = 0.1,
                 malignancy_dim: int = 16, use_adaptive: bool = True):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.neg_slope = negative_slope
        self.use_adaptive = use_adaptive
        self.register_buffer("adj", adj)          # (N, N) float mask
        self.blocks = blocks                      # list of (n_k,) int64
        if blocks is not None:
            # node permutation grouping each pathway block into a contiguous
            # range (index_select is O(N*d) and autograd-friendly, unlike
            # per-block in-place scatter); inv_perm maps back to gene order
            n_nodes = adj.shape[0]
            ids_arr = torch.zeros(n_nodes, dtype=torch.int64)
            for bi, b in enumerate(blocks):
                ids_arr[b] = bi
            perm = torch.argsort(ids_arr, stable=True)
            self.register_buffer("perm", perm)
            self.register_buffer("inv_perm", torch.argsort(perm))
            self.block_ranges = []
            start = 0
            for b in blocks:
                cnt = b.numel()
                self.block_ranges.append((start, start + cnt))
                start += cnt
            # sparse COO edge list (src-major per block, self-loops included)
            srcs, dsts, self.block_slices = [], [], []
            off = 0
            for b in blocks:
                n = b.numel()
                ii, jj = torch.meshgrid(torch.arange(n), torch.arange(n),
                                        indexing="ij")
                srcs.append(b[ii].reshape(-1))
                dsts.append(b[jj].reshape(-1))
                self.block_slices.append((off, off + n * n))
                off += n * n
            self.register_buffer("src", torch.cat(srcs))
            self.register_buffer("dst", torch.cat(dsts))
            # fixed symmetric-normalized edge weights for static mode:
            # within a complete block of size n every edge weight is 1/n
            unif = [torch.full((b.numel() ** 2,), 1.0 / b.numel())
                    for b in blocks]
            self.register_buffer("adj_weight", torch.cat(unif))
            self.block_alpha: list[torch.Tensor] | None = None
            self.block_alpha_grad: list[torch.Tensor] | None = None
        else:
            self.block_slices = []
            self.block_ranges = []
            self.register_buffer("perm", torch.arange(adj.shape[0], dtype=torch.int64))
            self.register_buffer("inv_perm", torch.arange(adj.shape[0], dtype=torch.int64))
            self.register_buffer("src", torch.arange(adj.shape[0], dtype=torch.int64))
            self.register_buffer("dst", torch.arange(adj.shape[0], dtype=torch.int64))
            self.register_buffer("adj_weight", torch.ones(adj.shape[0]))
            self.block_alpha = None
            self.block_alpha_grad = None
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(2 * out_dim))
        nn.init.xavier_uniform_(self.a.view(1, -1))
        self.beta = nn.Parameter(torch.zeros(1))      # malignancy gate weight
        self.s_mlp = nn.Sequential(
            nn.Linear(in_dim, malignancy_dim), nn.ReLU(),
            nn.Linear(malignancy_dim, 1))
        self.dropout = nn.Dropout(dropout)
        self.attn: torch.Tensor | None = None         # last alpha (B, E)

    def sparsity_penalty(self) -> torch.Tensor | None:
        """Mean |alpha| over all pathway edges (self-loops included).

        Uses the *non-detached* attention so the penalty actually
        back-propagates into the attention parameters.
        """
        if self.block_alpha_grad is None:
            return None
        total = sum(a.abs().sum() for a in self.block_alpha_grad)
        return total / max(self.src.numel(), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, in_dim). Returns updated (B, N, out_dim)."""
        B = x.shape[0]
        h = self.W(x)                                  # (B, N, out)
        if self.blocks is None:
            return self.dropout(h) + h                 # identity propagation
        if not self.use_adaptive:
            # static pathway GNN: fixed symmetric-normalized adjacency =
            # uniform mean-pooling inside each complete pathway block
            hp = h[:, self.perm]                       # block-contiguous
            parts = [hp[:, s:e].mean(dim=1, keepdim=True).expand(-1, e - s, -1)
                     for (s, e) in self.block_ranges]
            out = torch.cat(parts, dim=1)[:, self.inv_perm]
            out = self.dropout(out)
            return out + h
        # sample-level malignancy score from node features (learnable)
        s = torch.sigmoid(self.s_mlp(x.mean(dim=1)))   # (B, 1)
        left = (h @ self.a[:self.out_dim])[:, self.perm]    # (B, N)
        right = (h @ self.a[self.out_dim:])[:, self.perm]   # (B, N)
        hp = h[:, self.perm]
        block_alpha = []
        block_alpha_grad = []
        parts = []
        for (s0, e0) in self.block_ranges:
            li = left[:, s0:e0].unsqueeze(-1)          # (B, n, 1)
            rj = right[:, s0:e0].unsqueeze(1)          # (B, 1, n)
            logits = F.leaky_relu(li + rj, self.neg_slope)   # (B, n, n)
            # Malignancy-modulated attention temperature. A *constant* shift
            # (beta*s) would cancel in softmax, so the gate must act
            # multiplicatively: scale in (0, 2) via tanh; high-malignancy
            # samples (s -> 1) with beta > 0 sharpen attention toward
            # dominant pathway interactions, beta < 0 flattens it.
            scale = 1.0 + torch.tanh(self.beta) * s.unsqueeze(-1)  # (B, 1, 1)
            logits = logits * scale
            alpha_b = F.softmax(logits, dim=-1)        # (B, n, n)
            block_alpha.append(alpha_b.detach())       # export / rewiring
            block_alpha_grad.append(alpha_b)           # sparsity penalty
            parts.append(torch.bmm(alpha_b, hp[:, s0:e0]))
        self.block_alpha = block_alpha
        self.block_alpha_grad = block_alpha_grad
        self.attn = None
        out = torch.cat(parts, dim=1)[:, self.inv_perm]  # back to gene order
        out = self.dropout(out)
        return out + h                                 # residual


class PathAGNNCox(nn.Module):
    """Pathway-constrained adaptive GNN with a Cox risk head.

    Args:
        n_genes: number of genes in the pathway-filtered feature matrix.
        adj: pathway adjacency (N, N) or None (then identity = plain MLP-like).
        pathway_ids: per-gene pathway id (N,) used for block readout; if None,
            global mean pooling is used.
        hidden: hidden dims of GNN layers.
        n_layers: number of adaptive GAT layers.
        mlp_hidden: hidden dims of the risk MLP.
        dropout: dropout rate.
    """

    def __init__(self, n_genes: int, adj: torch.Tensor | None,
                 pathway_ids: torch.Tensor | None = None,
                 hidden: int = 64, n_layers: int = 2,
                 mlp_hidden: int = 32, dropout: float = 0.1,
                 use_adaptive: bool = True):
        super().__init__()
        self.n_genes = n_genes
        self.n_layers = n_layers
        self.use_adaptive = use_adaptive
        self.embed = nn.Linear(1, hidden)
        if adj is None:
            adj = torch.eye(n_genes)
        self.register_buffer("adj", adj.float())
        self.pathway_ids = pathway_ids  # (N,) or None
        blocks = None
        if pathway_ids is not None:
            blocks = _blocks_from_ids(pathway_ids.to(torch.int64),
                                      int(pathway_ids.max().item()) + 1)
        self.blocks = blocks
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(AdaptiveGATLayer(hidden, hidden, adj, blocks,
                                                dropout=dropout,
                                                use_adaptive=use_adaptive))
        self.risk_mlp = nn.Sequential(
            nn.Linear(hidden * 2, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1))

    def forward(self, x: torch.Tensor, return_alpha: bool = False):
        """x: (B, N) expression matrix (pathway-filtered genes).

        Returns risk (B,) and optionally the last-layer attention
        (alpha_edges (B, E), src (E,), dst (E,)) in sparse COO form.
        """
        x = x.unsqueeze(-1)                       # (B, N, 1)
        h = F.relu(self.embed(x))                 # (B, N, hidden)
        for layer in self.layers:
            h = layer(h)
        # pathway block readout: mean over genes within each pathway block
        # (blocks are disjoint primary-pathway modules, so block means are
        # exact; O(B*N*d) scatter instead of O(B*N*K*d) dense einsum)
        if self.pathway_ids is not None:
            ids = self.pathway_ids.to(h.device)
            k = int(ids.max().item()) + 1
            Bn = h.shape[0]
            pooled = torch.zeros(Bn, k, h.shape[2], device=h.device,
                                 dtype=h.dtype)
            pooled.scatter_add_(1, ids.view(1, -1, 1).expand(Bn, -1, h.shape[2]),
                                h)
            counts = torch.bincount(ids, minlength=k).clamp(min=1.0)
            pooled = pooled / counts.view(1, -1, 1)
            g_path = pooled.mean(dim=1)                             # (B, d)
        else:
            g_path = h.mean(dim=1)                                  # (B, d)
        g_gene = h.mean(dim=1)                                      # (B, d)
        g = torch.cat([g_gene, g_path], dim=-1)                     # (B, 2d)
        risk = self.risk_mlp(g).squeeze(-1)        # (B,)
        if return_alpha:
            layer = self.layers[-1]
            if layer.attn is not None:
                return risk, layer.attn, layer.src, layer.dst
            if layer.block_alpha is not None:
                parts = [a.transpose(1, 2).reshape(x.shape[0], -1)
                         for a in layer.block_alpha]
                return risk, torch.cat(parts, dim=1), layer.src, layer.dst
            # static mode: uniform per-block weights are returned by caller
            return risk, None, layer.src, layer.dst
        return risk

    @torch.no_grad()
    def predict_risk(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)

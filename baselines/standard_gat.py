"""Standard GAT survival model without pathway constraints (negative control).

Same training protocol and Cox head as Path-AGNN-Cox, but message passing runs
on a k-nearest-neighbour graph with plain GAT attention (no pathway masking,
no malignancy gate). Used to test whether patient-specific attention rewiring
is specific to the pathway-constrained design.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_knn_edges(X: np.ndarray, k: int = 10, seed: int = 0):
    """k-NN graph on gene features (rows=genes, cols=samples), undirected."""
    from sklearn.neighbors import NearestNeighbors
    # genes are the nodes: build k-NN on the genes x samples matrix
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
    nn.fit(X.T)
    idx = nn.kneighbors(X.T, return_distance=False)[:, 1:]  # (n_genes, k)
    n_genes = X.shape[1]
    src = np.repeat(np.arange(n_genes), k)
    dst = idx.reshape(-1)
    undirected = np.concatenate([np.stack([src, dst], axis=0),
                                 np.stack([dst, src], axis=0)], axis=1)
    return undirected  # (2, E)


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, edge_index: torch.Tensor,
                 negative_slope: float = 0.2, dropout: float = 0.1):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.neg_slope = negative_slope
        self.register_buffer("edge_index", edge_index)  # (2, E) int64
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(2 * out_dim))
        nn.init.xavier_uniform_(self.a.view(1, -1))
        self.dropout = nn.Dropout(dropout)
        self.attn: torch.Tensor | None = None
        self.src: torch.Tensor | None = None
        self.dst: torch.Tensor | None = None

    def sparsity_penalty(self):
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        src, dst = self.edge_index
        h = self.W(x)                                   # (B, N, d)
        hi = h[:, src]                                  # (B, E, d)
        hj = h[:, dst]
        logits = F.leaky_relu((hi * self.a[:self.out_dim]).sum(-1) +
                              (hj * self.a[self.out_dim:]).sum(-1),
                              self.neg_slope)           # (B, E)
        # softmax over edges per source node
        maxs = torch.full((B, N), float("-inf"), device=x.device)
        maxs = maxs.scatter_reduce(1, src.unsqueeze(0).expand(B, -1),
                                   logits, reduce="amax", include_self=False)
        maxs = torch.where(torch.isfinite(maxs), maxs, torch.zeros_like(maxs))
        exps = torch.exp(logits - maxs.gather(1, src.unsqueeze(0).expand(B, -1)))
        sums = torch.zeros(B, N, device=x.device)
        sums = sums.scatter_add(1, src.unsqueeze(0).expand(B, -1), exps)
        alpha = exps / sums.gather(1, src.unsqueeze(0).expand(B, -1)).clamp(min=1e-8)
        self.attn = alpha.detach()
        self.src = src
        self.dst = dst
        out = torch.zeros(B, N, self.out_dim, device=x.device)
        out = out.scatter_add(1, dst.unsqueeze(0).expand(B, -1).unsqueeze(-1)
                              .expand(B, -1, self.out_dim),
                              (alpha.unsqueeze(-1) * hj))
        return self.dropout(out) + h


class StandardGAT(nn.Module):
    """GAT + global pooling + Cox MLP head (no pathway prior)."""

    def __init__(self, n_genes: int, edge_index: torch.Tensor,
                 hidden: int = 64, n_layers: int = 2,
                 mlp_hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.n_genes = n_genes
        self.embed = nn.Linear(1, hidden)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(GATLayer(hidden, hidden, edge_index, dropout=dropout))
        self.risk_mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1))

    def forward(self, x: torch.Tensor, return_alpha: bool = False):
        x = x.unsqueeze(-1)
        h = F.relu(self.embed(x))
        for layer in self.layers:
            h = layer(h)
        g = h.mean(dim=1)
        risk = self.risk_mlp(g).squeeze(-1)
        if return_alpha:
            layer = self.layers[-1]
            return risk, layer.attn, layer.src, layer.dst
        return risk

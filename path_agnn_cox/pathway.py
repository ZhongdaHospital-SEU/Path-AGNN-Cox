"""Pathway-constrained subgraph construction from KEGG/GO gene sets.

Builds a block-diagonal adjacency matrix A in which an edge exists between
two genes iff they co-occur in at least one pathway (or, optionally, a
STRING/PPI-filtered subset). Message passing in the GNN is then confined to
these biologically co-regulated gene groups.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def load_gmt(path: str, min_genes: int = 3, max_genes: int = 300) -> Dict[str, List[str]]:
    """Load a GMT file (e.g. c2.cp.kegg / GO terms) into {pathway: [genes]}.

    Args:
        path: GMT file path. First column = pathway id, second = description,
            remaining columns = gene symbols.
        min_genes / max_genes: filter pathway sizes to avoid tiny/huge sets.
    """
    out: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pid, genes = parts[0], [g.upper() for g in parts[2:]]
            genes = list(dict.fromkeys(genes))  # dedupe, keep order
            if min_genes <= len(genes) <= max_genes:
                out[pid] = genes
    return out


def pathway_gene_matrix(
    gene_list: List[str],
    pathway_dict: Dict[str, List[str]],
) -> pd.DataFrame:
    """Binary pathway membership matrix (genes x pathways)."""
    gene_set = set(g.upper() for g in gene_list)
    rows = {}
    for pid, genes in pathway_dict.items():
        hit = [g for g in genes if g in gene_set]
        if len(hit) >= 3:
            rows[pid] = hit
    genes_sorted = sorted(gene_set)
    mat = pd.DataFrame(0, index=genes_sorted, columns=list(rows.keys()), dtype=int)
    for pid, hit in rows.items():
        mat.loc[hit, pid] = 1
    # drop genes not in any pathway -> they are excluded from the graph
    keep = mat.sum(axis=1) > 0
    return mat.loc[keep]


def build_pathway_adjacency(
    gene_list: List[str],
    pathway_dict: Dict[str, List[str]],
    self_loop: bool = True,
    ppi_filter: Optional[pd.DataFrame] = None,
    overlap: str = "primary",
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Build the pathway-constrained adjacency matrix.

    Args:
        gene_list: genes present in the expression matrix.
        pathway_dict: {pathway_id: [genes]} from load_gmt.
        self_loop: add identity (self-loops) so isolated nodes keep features.
        ppi_filter: optional edge list DataFrame with columns gene1/gene2; edges
            are kept only when both endpoints share a pathway AND the pair is
            present in the PPI filter (e.g. STRING high-confidence edges).
        overlap: "primary" (default) assigns each gene to its primary pathway
            module (block-diagonal subgraphs, consistent with the model's
            pathway readout and bounded edge count on real cohorts);
            "union" connects gene pairs co-occurring in any pathway.

    Returns:
        A: binary adjacency (n x n), genes x pathways membership matrix,
        gene_order: the subset of gene_list that belongs to at least one pathway.
    """
    mem = pathway_gene_matrix(gene_list, pathway_dict)
    genes = list(mem.index)
    n = len(genes)
    pos = {g: i for i, g in enumerate(genes)}
    A = np.zeros((n, n), dtype=np.float32)
    pathway_ids = mem.columns.tolist()
    if overlap == "primary":
        groups: Dict[str, List[int]] = {}
        for g, pid in mem.idxmax(axis=1).items():
            groups.setdefault(pid, []).append(pos[g])
        for idx in groups.values():
            for i in idx:
                for j in idx:
                    if i != j:
                        A[i, j] = 1.0
    else:
        for pid in pathway_ids:
            idx = np.where(mem[pid].values == 1)[0]
            for i in idx:
                for j in idx:
                    if i != j:
                        A[i, j] = 1.0
    if ppi_filter is not None:
        keep_edges = set()
        for _, r in ppi_filter.iterrows():
            a, b = str(r["gene1"]).upper(), str(r["gene2"]).upper()
            if a in pos and b in pos:
                keep_edges.add((pos[a], pos[b]))
        for i in range(n):
            for j in range(n):
                if A[i, j] and (i, j) not in keep_edges:
                    A[i, j] = 0.0
    if self_loop:
        A += np.eye(n, dtype=np.float32)
    return A, mem, np.array(genes)


def pathway_ids_per_gene(mem: pd.DataFrame) -> np.ndarray:
    """For each gene, its primary (most-connected) pathway id, -1 if none."""
    ids = mem.idxmax(axis=1).values
    return np.array([mem.columns.get_loc(v) for v in ids], dtype=np.int64)
# -*- coding: utf-8 -*-
"""Single-block matched control: for each real pathway of size k, draw random
k-subsets from real pathway blocks of size >= k (same topology: complete
directed block), compute |Cohen's d| between risk strata, and report the
percentile of the real pathway's effect. Complements the edge-matched control
which cannot reach large block sizes. Merges results into pathway_effects.csv.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
sys.path.insert(0, str(ROOT))

from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency
from path_agnn_cox.data import load_survival_data, split_features

def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    q = pvals[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out

def cohen_d(a1, a2, n1, n2):
    sp = np.sqrt(((n1 - 1) * a1.std(ddof=1) ** 2 + (n2 - 1) * a2.std(ddof=1) ** 2) / (n1 + n2 - 2.0))
    return (a1.mean() - a2.mean()) / sp if sp > 0 else 0.0

def analyze(ds):
    out_dir = ROOT / "results" / "rewiring" / ds
    df = load_survival_data(str(ROOT / "data" / "processed" / ds / "train.csv"))
    X, time, event = split_features(df)
    pathway_dict = load_gmt(str(ROOT / "data" / "pathways" / "kegg_cancer_core.gmt"))
    adj, mem, gene_order = build_pathway_adjacency(list(X.columns), pathway_dict)
    cols = np.array([c for c in gene_order if c in X.columns])
    gene_to_pw = mem.loc[cols].idxmax(axis=1)

    alpha = np.load(out_dir / "alpha.npy")
    edges = pd.read_csv(out_dir / "edges_meta.csv")
    risk = pd.read_csv(out_dir / "risk_scores.csv")["risk_score"].to_numpy()
    src = edges["src_gene"].to_numpy()
    dst = edges["dst_gene"].to_numpy()
    nonself = src != dst
    gidx = {g: i for i, g in enumerate(cols)}
    src_i = np.array([gidx[g] for g in src])
    dst_i = np.array([gidx[g] for g in dst])
    N = len(cols)

    med = np.median(risk)
    hi = np.where(risk > med)[0]
    lo = np.where(risk <= med)[0]
    n1, n2 = len(hi), len(lo)

    # real pathway blocks (gene index lists)
    blocks = {}
    for g, pw in gene_to_pw.items():
        blocks.setdefault(pw, []).append(gidx[g])
    blocks = {pw: np.array(v, dtype=int) for pw, v in blocks.items()}

    eff = pd.read_csv(out_dir / "pathway_effects.csv")
    rng = np.random.default_rng(20260820)
    n_null = 200
    rows = []
    for _, r in eff.iterrows():
        pw = r["pathway"]
        k = int(r["n_genes"])
        real_d = abs(float(r["cohen_d"]))
        cand = [b for b in blocks.values() if len(b) >= k]
        null_dc = []
        accepted = 0
        tries = 0
        while accepted < n_null and tries < 20000:
            tries += 1
            blk = cand[rng.integers(len(cand))]
            S = rng.choice(blk, size=k, replace=False)
            member = np.zeros(N, dtype=bool)
            member[S] = True
            sel = np.flatnonzero(member[src_i] & member[dst_i] & nonself)
            if len(sel) < 5:
                continue
            sc = alpha[:, sel].mean(axis=1)
            null_dc.append(abs(cohen_d(sc[hi], sc[lo], n1, n2)))
            accepted += 1
        null_dc = np.array(null_dc)
        pct = float((np.sum(null_dc >= real_d) + 1) / (len(null_dc) + 1)) if len(null_dc) else np.nan
        rows.append({"pathway": pw, "n_null_block_sets": accepted,
                     "block_null_median_abs_d": float(np.median(null_dc)) if len(null_dc) else np.nan,
                     "block_null_p95": float(np.percentile(null_dc, 95)) if len(null_dc) else np.nan,
                     "block_null_pct": pct})
    ctrl = pd.DataFrame(rows)
    merged = eff.merge(ctrl, on="pathway", how="left")
    merged.to_csv(out_dir / "pathway_effects.csv", index=False)
    n_exceed = int((merged["block_null_pct"] >= 0.95).sum())
    med_pct = float(merged["block_null_pct"].median())
    print(f"== {ds}: block-matched null, n>=0.95: {n_exceed} (expected {0.05 * len(merged):.2f}), median pct={med_pct:.3f}", flush=True)
    sub = merged[merged["perm_q"] < 0.05][["pathway", "cohen_d", "perm_q", "null_pct", "block_null_pct"]]
    print(sub.to_string(index=False), flush=True)

if __name__ == "__main__":
    for ds in ["LUAD", "BRCA"]:
        analyze(ds)

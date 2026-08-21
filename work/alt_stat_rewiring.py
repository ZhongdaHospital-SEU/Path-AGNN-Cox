# -*- coding: utf-8 -*-
"""P0-2: alternative edge-weight statistics for between-stratum rewiring.

The mean-based pathway statistic ranks BRCA real pathways below matched random
gene sets (percentiles 0.09-0.20). This script tests whether sparse, edge-
concentrated rewiring is masked by the mean: it evaluates, per pathway,
  - mean |d|          (reference, mean-based)
  - top-decile mean |d| (sparse-rewiring statistic)
  - concentration     (share of total |d| carried by the top 10% edges)
  - max |d|
against 200 density-matched random k-subsets drawn from real pathway blocks,
and reports the percentile of each real pathway within its null.

Outputs: results/rewiring/<DS>/alt_stat_controls.csv (+ _summary.csv)
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
sys.path.insert(0, str(ROOT))
from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency
from path_agnn_cox.data import load_survival_data, split_features

N_NULL = 200
SEED = 20260821

def topdec_conc_max(vals):
    v = np.abs(np.asarray(vals, dtype=float))
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    k10 = max(1, int(np.ceil(0.1 * len(v))))
    order = np.argsort(v)[::-1][:k10]
    top = v[order]
    return float(top.mean()), float(top.sum() / max(v.sum(), 1e-300)), float(v.max())

def analyze(ds):
    out_dir = ROOT / "results" / "rewiring" / ds
    df = load_survival_data(str(ROOT / "data" / "processed" / ds / "train.csv"))
    X, _, _ = split_features(df)
    pathway_dict = load_gmt(str(ROOT / "data" / "pathways" / "kegg_cancer_core.gmt"))
    adj, mem, gene_order = build_pathway_adjacency(list(X.columns), pathway_dict)
    cols = np.array([c for c in gene_order if c in X.columns])
    gene_to_pw = mem.loc[cols].idxmax(axis=1)
    blocks = {}
    for g, pw in gene_to_pw.items():
        blocks.setdefault(pw, []).append(int(np.where(cols == g)[0][0]))
    blocks = {pw: np.asarray(v, dtype=int) for pw, v in blocks.items()}
    pw_list = sorted(blocks.keys())

    diff = pd.read_csv(out_dir / "edge_diff.csv")
    diff = diff[diff["gene_i"] != diff["gene_j"]].reset_index(drop=True)
    # per-block edge d matrices
    gidx = {g: i for i, g in enumerate(cols)}
    block_mat = {}
    for pw in pw_list:
        b = blocks[pw]
        sub = diff[diff["pathway"] == pw]
        if not len(sub):
            continue
        mat = np.full((len(b), len(b)), np.nan)
        for _, r in sub.iterrows():
            i, j = gidx.get(r["gene_i"]), gidx.get(r["gene_j"])
            if i is None or j is None:
                continue
            li = int(np.where(b == i)[0][0]); lj = int(np.where(b == j)[0][0])
            mat[li, lj] = r["d"]
        block_mat[pw] = (b, mat)
    print(ds, "blocks with edges:", len(block_mat), flush=True)

    rng = np.random.default_rng(SEED)
    rows = []
    for pw in pw_list:
        if pw not in block_mat:
            continue
        b, mat = block_mat[pw]
        k = len(b)
        ii, jj = np.triu_indices(k, 1)
        # directed pairs (both directions)
        off = np.concatenate([mat[ii, jj], mat[jj, ii]])
        off = off[np.isfinite(off)]
        if len(off) < 5:
            continue
        real_mean, real_top, real_conc, real_max = np.abs(off).mean(), *topdec_conc_max(off)
        cand = [bb for bb in blocks.values() if len(bb) >= k]
        null_mean, null_top, null_conc, null_max = [], [], [], []
        tries = 0
        while len(null_mean) < N_NULL and tries < 20000:
            tries += 1
            bb = cand[int(rng.integers(len(cand)))]
            sub = np.sort(rng.choice(bb, size=k, replace=False))
            m = block_mat.get(pw, (None, None))[1]
            # need d-matrix of the block that bb belongs to
            bpw = gene_to_pw[cols[bb[0]]]
            bm = block_mat.get(bpw)
            if bm is None:
                continue
            bmat = bm[1]
            loc = np.array([int(np.where(bm[0] == g)[0][0]) for g in sub])
            vals = bmat[np.ix_(loc, loc)]
            off2 = vals[~np.eye(k, dtype=bool)]
            off2 = off2[np.isfinite(off2)]
            if len(off2) < 5:
                continue
            null_mean.append(np.abs(off2).mean())
            t, co, mx = topdec_conc_max(off2)
            null_top.append(t); null_conc.append(co); null_max.append(mx)
        if len(null_mean) < 50:
            continue
        null_mean = np.asarray(null_mean); null_top = np.asarray(null_top)
        null_conc = np.asarray(null_conc); null_max = np.asarray(null_max)
        rows.append({
            "pathway": pw, "k": k, "n_edges": len(off),
            "mean_abs_d": real_mean, "topdec_mean_abs_d": real_top,
            "concentration": real_conc, "max_abs_d": real_max,
            "null_median_mean_abs_d": float(np.median(null_mean)),
            "pct_mean": float(np.mean(real_mean >= null_mean)) * 100.0,
            "null_median_topdec": float(np.median(null_top)),
            "pct_topdec": float(np.mean(real_top >= null_top)) * 100.0,
            "null_median_conc": float(np.median(null_conc)),
            "pct_conc": float(np.mean(real_conc >= null_conc)) * 100.0,
            "null_median_max": float(np.median(null_max)),
            "pct_max": float(np.mean(real_max >= null_max)) * 100.0,
        })
    res = pd.DataFrame(rows).sort_values("pct_topdec", ascending=False).reset_index(drop=True)
    res.to_csv(out_dir / "alt_stat_controls.csv", index=False)
    summ = pd.DataFrame([{
        "dataset": ds, "n_pathways": len(res),
        "median_pct_mean": float(res["pct_mean"].median()),
        "median_pct_topdec": float(res["pct_topdec"].median()),
        "median_pct_conc": float(res["pct_conc"].median()),
        "median_pct_max": float(res["pct_max"].median()),
        "n_above50_mean": int((res["pct_mean"] > 50).sum()),
        "n_above50_topdec": int((res["pct_topdec"] > 50).sum()),
        "n_above50_conc": int((res["pct_conc"] > 50).sum()),
        "n_above50_max": int((res["pct_max"] > 50).sum()),
        "n_above95_topdec": int((res["pct_topdec"] > 95).sum()),
    }])
    summ.to_csv(out_dir / "alt_stat_summary.csv", index=False)
    print(ds, "pathways:", len(res), "median pct mean/topdec/conc/max:",
          round(float(res["pct_mean"].median()), 1), round(float(res["pct_topdec"].median()), 1),
          round(float(res["pct_conc"].median()), 1), round(float(res["pct_max"].median()), 1), flush=True)

if __name__ == "__main__":
    for ds in ["BRCA", "LUAD"]:
        analyze(ds)
    print("ALT_STAT_DONE", flush=True)

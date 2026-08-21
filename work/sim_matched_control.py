# -*- coding: utf-8 -*-
"""P0-2 decisive experiment: distribution of the density-matched control
percentile (block_null_pct) under a PURE NULL (no between-stratum signal),
with row-normalized log-normal attention on the REAL LUAD/BRCA block layout.

Observed block_null_pct medians: LUAD 0.687, BRCA 0.199 (P(null >= real),
so LOW = real effect larger than matched random gene sets).
If the null simulation reproduces ~0.5 medians, the observed BRCA value is
genuine enrichment above matched nulls; if it reproduces ~0.2, the below-0.5
values are a structural property of row-normalized attention and matched
k-subsets, not signal.

Outputs: results/simulation/matched_control_null_<DS>.csv (pathway-level pcts
per dataset) + _summary.csv (dataset-level medians).
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
sys.path.insert(0, str(ROOT))
from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency
from path_agnn_cox.data import load_survival_data, split_features

B = 300
N_CTL = 100
N_SIM = 24
SEED = 20260822

def cohen_d(a1, a2, n1, n2):
    sp = np.sqrt(((n1 - 1) * a1.std(ddof=1) ** 2 + (n2 - 1) * a2.std(ddof=1) ** 2) / (n1 + n2 - 2.0))
    return (a1.mean() - a2.mean()) / sp if sp > 0 else 0.0

def build_layout(ds):
    df = load_survival_data(str(ROOT / "data" / "processed" / ds / "train.csv"))
    X, _, _ = split_features(df)
    pathway_dict = load_gmt(str(ROOT / "data" / "pathways" / "kegg_cancer_core.gmt"))
    adj, mem, gene_order = build_pathway_adjacency(list(X.columns), pathway_dict)
    cols = np.array([c for c in gene_order if c in X.columns])
    blocks = {}
    for g, pw in mem.loc[cols].idxmax(axis=1).items():
        blocks.setdefault(pw, []).append(int(np.where(cols == g)[0][0]))
    blocks = {pw: np.asarray(v, dtype=int) for pw, v in blocks.items()}
    # per-block edge arrays (global edge list)
    src_all, dst_all, pw_of_edge = [], [], []
    for pw in sorted(blocks):
        g = blocks[pw]
        for i in g:
            for j in g:
                if i != j:
                    src_all.append(i); dst_all.append(j); pw_of_edge.append(pw)
    src_all = np.asarray(src_all, dtype=np.int64)
    dst_all = np.asarray(dst_all, dtype=np.int64)
    pw_of_edge = np.asarray(pw_of_edge, dtype=object)
    N = len(cols)
    return blocks, src_all, dst_all, pw_of_edge, N

def sim_alpha_null(blocks, src_all, dst_all, N, rng):
    """Row-normalized log-normal attention, no between-stratum signal."""
    E = len(src_all)
    alpha = np.empty((B, E), dtype=np.float64)
    e = 0
    for pw in sorted(blocks):
        g = blocks[pw]
        k = len(g)
        W = rng.lognormal(mean=0.0, sigma=0.3, size=(B, k, k))
        np.fill_diagonal(W[0], 0.0)  # placeholder; diagonal per sample below
        for b in range(B):
            np.fill_diagonal(W[b], 0.0)
        row_sum = W.sum(axis=2, keepdims=True)
        A = W / np.maximum(row_sum, 1e-12)
        for i in range(k):
            for j in range(k):
                if i != j:
                    alpha[:, e] = A[:, i, j]
                    e += 1
    return alpha

def run_ds(ds, seed):
    blocks, src_all, dst_all, pw_of_edge, N = build_layout(ds)
    pw_list = sorted(blocks.keys())
    rng = np.random.default_rng(seed)
    n1 = n2 = B // 2
    rows = []
    for s in range(N_SIM):
        alpha = sim_alpha_null(blocks, src_all, dst_all, N, rng)
        perm = rng.permutation(B)
        hi, lo = perm[: n1], perm[n1:]
        for pw in pw_list:
            g = blocks[pw]
            k = len(g)
            sel = np.flatnonzero((pw_of_edge == pw) & (src_all != dst_all))
            real = abs(cohen_d(alpha[hi][:, sel].mean(axis=1), alpha[lo][:, sel].mean(axis=1), n1, n2))
            cand = [b for b in blocks.values() if len(b) >= k]
            null_d = []
            tries = 0
            while len(null_d) < N_CTL and tries < 3000:
                tries += 1
                blk = cand[int(rng.integers(len(cand)))]
                bpw = None
                for pw2, bb in blocks.items():
                    if np.array_equal(blk, bb):
                        bpw = pw2; break
                S = rng.choice(blk, size=k, replace=False)
                member = np.zeros(N, dtype=bool)
                member[S] = True
                sel2 = np.flatnonzero(member[src_all] & member[dst_all] & (src_all != dst_all) & (pw_of_edge == bpw))
                if len(sel2) < 5:
                    continue
                sc = alpha[hi][:, sel2].mean(axis=1)
                sc2 = alpha[lo][:, sel2].mean(axis=1)
                null_d.append(abs(cohen_d(sc, sc2, n1, n2)))
            if len(null_d) < 20:
                continue
            null_d = np.array(null_d)
            pct = float((np.sum(null_d >= real) + 1) / (len(null_d) + 1))
            rows.append({"dataset": ds, "sim": s, "pathway": pw, "k": k,
                         "block_null_pct": pct, "real_abs_d": real,
                         "null_median_abs_d": float(np.median(null_d))})
        if (s + 1) % 6 == 0:
            sim_pcts = [r["block_null_pct"] for r in rows if r["sim"] == s]
            print(ds, "sim", s + 1, "median pct so far:", round(float(np.median(sim_pcts)), 3), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "simulation" / ("matched_control_null_%s.csv" % ds), index=False)
    meds = out.groupby("sim")["block_null_pct"].median()
    summ = pd.DataFrame([{"dataset": ds, "n_sim": N_SIM,
                          "null_median_of_medians": float(meds.median()),
                          "null_mean_of_medians": float(meds.mean()),
                          "null_p05_medians": float(np.percentile(meds, 5)),
                          "null_p95_medians": float(np.percentile(meds, 95)),
                          "pooled_pct_median": float(out["block_null_pct"].median()),
                          "frac_pct_below_0.25": float((out["block_null_pct"] < 0.25).mean())}])
    summ.to_csv(ROOT / "results" / "simulation" / ("matched_control_null_%s_summary.csv" % ds), index=False)
    print(ds, "DONE median-of-medians:", round(float(meds.median()), 3),
          "5-95:", round(float(np.percentile(meds, 5)), 3), round(float(np.percentile(meds, 95)), 3), flush=True)
    return summ

if __name__ == "__main__":
    os.makedirs(ROOT / "results" / "simulation", exist_ok=True)
    with Pool(2) as pool:
        results = pool.starmap(run_ds, [("LUAD", SEED), ("BRCA", SEED + 1)])
    pd.concat(results).to_csv(ROOT / "results" / "simulation" / "matched_control_null_summary_all.csv", index=False)
    print("MC_NULL_DONE", flush=True)

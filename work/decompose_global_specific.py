# -*- coding: utf-8 -*-
"""Global vs pathway-specific decomposition.
Question: are the many significant pathways just a projection of a global
between-stratum shift in overall attention level?
For each cohort: (1) global test on per-sample mean of ALL edges;
(2) per-pathway tests on pathway score MINUS the global per-sample mean.
Also report direction consistency of pathway effects."""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
sys.path.insert(0, str(ROOT))
from benchmark.rewiring_analysis import bh_fdr
from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency
from path_agnn_cox.data import load_survival_data, split_features

RW = ROOT / "results" / "rewiring"
gmt = load_gmt(str(ROOT / "data" / "pathways" / "kegg_cancer_core.gmt"))
out_rows = []
for d in ["LUAD", "BRCA", "KIRC"]:
    df = load_survival_data(str(ROOT / "data" / "processed" / d / "train.csv"))
    X, _, _ = split_features(df)
    adj, mem, gene_order = build_pathway_adjacency(list(X.columns), gmt)
    cols = np.array([c for c in gene_order if c in X.columns])
    alpha = np.load(RW / d / "alpha.npy")
    risk = pd.read_csv(RW / d / "risk_scores.csv")["risk_score"].to_numpy()
    med = np.median(risk)
    hi = np.flatnonzero(risk > med); lo = np.flatnonzero(risk <= med)
    # edge list
    pid = mem.loc[cols].idxmax(axis=1)
    blocks = {}
    for g, pw in pid.items():
        blocks.setdefault(pw, []).append(int(np.where(cols == g)[0][0]))
    src_a, dst_a, pw_e = [], [], []
    for pw, gs in blocks.items():
        for i in gs:
            for j in gs:
                if i != j:
                    src_a.append(i); dst_a.append(j); pw_e.append(pw)
    src_a = np.array(src_a); dst_a = np.array(dst_a); pw_e = np.array(pw_e, dtype=object)
    # global per-sample mean over all edges
    global_score = alpha.mean(axis=1)
    u, p_global = mannwhitneyu(global_score[hi], global_score[lo], alternative="two-sided")
    d_global = global_score[hi].mean() - global_score[lo].mean()
    n_pos = 0; n_neg = 0; rows = []
    for pw in pd.unique(pw_e):
        sel = np.flatnonzero((pw_e == pw) & (src_a != dst_a))
        if len(sel) < 5:
            continue
        sc = alpha[:, sel].mean(axis=1)
        d_raw = sc[hi].mean() - sc[lo].mean()
        if d_raw > 0: n_pos += 1
        else: n_neg += 1
        # pathway-specific: residual after removing the global shift
        sc_adj = sc - global_score
        u2, p2 = mannwhitneyu(sc_adj[hi], sc_adj[lo], alternative="two-sided")
        rows.append({"pathway": pw, "d_raw": d_raw, "d_adj": sc_adj[hi].mean() - sc_adj[lo].mean(), "p_adj": p2})
    r = pd.DataFrame(rows)
    r["q_adj"] = bh_fdr(r["p_adj"].to_numpy())
    n_sig_adj = int((r["q_adj"] < 0.05).sum())
    r.to_csv(RW / d / "pathway_specific_decomposition.csv", index=False)
    out_rows.append({"dataset": d, "n_hi": len(hi), "n_lo": len(lo),
                     "global_d": d_global, "global_p": p_global,
                     "n_pathways": len(r), "n_direction_pos": n_pos, "n_direction_neg": n_neg,
                     "n_sig_raw_q005": int((r['p_adj'] <= 1).sum()),  # placeholder
                     "n_sig_specific_q005": n_sig_adj})
    print(d, "global d=%.4f p=%.3g | pos %d neg %d | specific-sig %d/%d" %
          (d_global, p_global, n_pos, n_neg, n_sig_adj, len(r)), flush=True)
pd.DataFrame(out_rows).to_csv(RW / "global_vs_specific_summary.csv", index=False)
print("DECOMP_DONE")

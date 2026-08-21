# -*- coding: utf-8 -*-
"""Global vs pathway-specific decomposition (v2, using saved edge list).
For each cohort: (1) global test on per-sample mean over all non-self edges;
(2) per-pathway tests on pathway score MINUS the per-sample global mean.
Sanity-check: pathway score means must reproduce pathway_effects.csv."""
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
    gidx = {g: i for i, g in enumerate(cols)}
    alpha = np.load(RW / d / "alpha.npy")
    risk = pd.read_csv(RW / d / "risk_scores.csv")["risk_score"].to_numpy()
    assert alpha.shape[0] == len(risk) == len(df)
    edges = pd.read_csv(RW / d / "edges_meta.csv")
    src_i = np.array([gidx[g] for g in edges["src_gene"]])
    dst_i = np.array([gidx[g] for g in edges["dst_gene"]])
    nonself = src_i != dst_i
    gene_to_pw = mem.loc[cols].idxmax(axis=1)
    pw_of_edge = gene_to_pw.reindex(edges["src_gene"]).to_numpy()
    med = np.median(risk)
    hi = np.flatnonzero(risk > med); lo = np.flatnonzero(risk <= med)
    n1, n2 = len(hi), len(lo)
    # sanity: reproduce pathway_effects m1/m2
    pe = pd.read_csv(RW / d / "pathway_effects.csv").set_index("pathway")
    global_score = alpha[:, nonself].mean(axis=1)
    u, p_global = mannwhitneyu(global_score[hi], global_score[lo], alternative="two-sided")
    d_global = global_score[hi].mean() - global_score[lo].mean()
    n_pos = n_neg = 0
    rows = []
    max_dev = 0.0
    for pw in pd.unique(pw_of_edge):
        sel = np.flatnonzero((pw_of_edge == pw) & nonself)
        if len(sel) < 5:
            continue
        sc = alpha[:, sel].mean(axis=1)
        m1, m2 = float(sc[hi].mean()), float(sc[lo].mean())
        if pw in pe.index:
            dev = abs((m1 - m2) - (float(pe.loc[pw, "mean_hi"]) - float(pe.loc[pw, "mean_lo"])))
            max_dev = max(max_dev, dev)
        if m1 - m2 > 0: n_pos += 1
        else: n_neg += 1
        sc_adj = sc - global_score
        u2, p2 = mannwhitneyu(sc_adj[hi], sc_adj[lo], alternative="two-sided")
        rows.append({"pathway": pw, "n_edges": len(sel),
                     "d_raw": m1 - m2, "d_adj": sc_adj[hi].mean() - sc_adj[lo].mean(),
                     "p_raw": np.nan, "p_adj": p2})
    r = pd.DataFrame(rows)
    r["q_adj"] = bh_fdr(r["p_adj"].to_numpy())
    n_sig_adj = int((r["q_adj"] < 0.05).sum())
    r.to_csv(RW / d / "pathway_specific_decomposition.csv", index=False)
    out_rows.append({"dataset": d, "n_hi": n1, "n_lo": n2,
                     "global_d": d_global, "global_p": p_global,
                     "n_pathways": len(r), "n_direction_pos": n_pos, "n_direction_neg": n_neg,
                     "n_sig_specific_q005": n_sig_adj,
                     "max_dev_from_pathway_effects": max_dev})
    print(d, "global d=%.2e p=%.3g | pos %d neg %d | specific-sig %d/%d | maxdev %.2e" %
          (d_global, p_global, n_pos, n_neg, n_sig_adj, len(r), max_dev), flush=True)
pd.DataFrame(out_rows).to_csv(RW / "global_vs_specific_summary.csv", index=False)
print("DECOMP2_DONE")

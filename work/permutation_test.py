# -*- coding: utf-8 -*-
"""Permutation null for rewiring: shuffle risk labels, recompute pathway-level
Mann-Whitney tests, compare significant-pathway count to the observed value.
Saves results/rewiring/<DS>/permutation_test.csv (LUAD + BRCA)."""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
sys.path.insert(0, str(ROOT))
from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency
from path_agnn_cox.data import load_survival_data, split_features, standardize
from benchmark.rewiring_analysis import pathway_level_test, bh_fdr

N_PERM = int(os.environ.get("PERM_N", "200"))
SEED = 0
rng = np.random.default_rng(SEED)

def run_ds(ds):
    out_dir = ROOT / "results" / "rewiring" / ds
    alpha = np.load(out_dir / "alpha.npy")
    risk = pd.read_csv(out_dir / "risk_scores.csv")["risk_score"].to_numpy()
    meta = pd.read_csv(out_dir / "edges_meta.csv")
    src = meta["src_gene"].to_numpy()
    dst = meta["dst_gene"].to_numpy()
    df = load_survival_data(str(ROOT / "data" / "processed" / ds / "train.csv"))
    X, time, event = split_features(df)
    pathway_dict = load_gmt(str(ROOT / "data" / "pathways" / "kegg_cancer_core.gmt"))
    adj, mem, gene_order = build_pathway_adjacency(list(X.columns), pathway_dict)
    cols = np.array([c for c in gene_order if c in X.columns])
    # map edge gene names -> integer node ids
    pos = {g: i for i, g in enumerate(cols)}
    src = np.array([pos[g] for g in src], dtype=int)
    dst = np.array([pos[g] for g in dst], dtype=int)
    pid = mem.idxmax(axis=1)
    # observed
    med = np.median(risk)
    hi = np.where(risk > med)[0]; lo = np.where(risk <= med)[0]
    obs = pathway_level_test(alpha, hi, lo, src, dst, cols, mem.loc[cols])
    obs_n = int((obs["q"] < 0.05).sum()) if len(obs) else 0
    n_sig = []
    n_path = []
    for k in range(N_PERM):
        perm = rng.permutation(len(risk))
        r_perm = risk[perm]
        medp = np.median(r_perm)
        hi_p = np.where(r_perm > medp)[0]; lo_p = np.where(r_perm <= medp)[0]
        # pathway tests on the same alpha but permuted groups
        res = pathway_level_test(alpha, hi_p, lo_p, src, dst, cols, mem.loc[cols])
        n_sig.append(int((res["q"] < 0.05).sum()) if len(res) else 0)
        n_path.append(len(res))
        if (k + 1) % 50 == 0:
            print(ds, "perm", k + 1, "sig so far:", np.mean(n_sig), flush=True)
    n_sig = np.array(n_sig)
    pval = (1 + int((n_sig >= obs_n).sum())) / (1 + N_PERM)
    out = pd.DataFrame({
        "observed_sig": [obs_n],
        "n_perm": [N_PERM],
        "null_mean_sig": [float(n_sig.mean())],
        "null_max_sig": [int(n_sig.max())],
        "null_frac_ge_1": [float((n_sig >= 1).mean())],
        "perm_p": [pval],
        "n_pathways_observed": [len(obs)],
        "n_pathways_null_mean": [float(np.mean(n_path))],
    })
    out.to_csv(out_dir / "permutation_test.csv", index=False)
    print(ds, "DONE observed=", obs_n, "null mean=", round(float(n_sig.mean()), 2),
          "max=", int(n_sig.max()), "perm_p=", pval, flush=True)

if __name__ == "__main__":
    for ds in ["LUAD", "BRCA"]:
        run_ds(ds)
    print("ALL_PERM_DONE", flush=True)

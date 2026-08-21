# -*- coding: utf-8 -*-
"""P1-1: pathway-level rewiring replication in external LUAD cohorts.

Trains the LUAD model with the exact code path of benchmark/rewiring_analysis.py
(epochs=80, seed 0, benchmark config), transfers to GSE31210 / GSE50081 /
GSE68465, computes per-pathway between-stratum Mann-Whitney tests (median-risk
split within each cohort), and meta-analyzes each pathway across cohorts with
Stouffer's method (unweighted and sample-size weighted) plus direction
consistency and sample-size weighted Cohen's d.

Outputs: results/ext_pathway_meta.csv (+ _focus.csv for HR / DNA replication)
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
sys.path.insert(0, str(ROOT))
import torch
from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency
from path_agnn_cox.data import load_survival_data, split_features, standardize
from path_agnn_cox.models import PathAGNNCox
from path_agnn_cox.train import train_model, predict_with_alpha
from benchmark.dataset_manifest import load_benchmark_config
from benchmark.rewiring_analysis import bh_fdr

DS = "LUAD"
EXT = ["GSE31210", "GSE50081", "GSE68465"]
EPOCHS = int(os.environ.get("EXT_META_EPOCHS", "80"))
SEED = 0

t0 = time.time()
# ---- train on TCGA-LUAD (same code path as rewiring_analysis.py) ----
df = load_survival_data(str(ROOT / "data" / "processed" / DS / "train.csv"))
X, time_, event = split_features(df)
pathway_dict = load_gmt(str(ROOT / "data" / "pathways" / "kegg_cancer_core.gmt"))
adj, mem, gene_order = build_pathway_adjacency(list(X.columns), pathway_dict)
cols = np.array([c for c in gene_order if c in X.columns])
Xs, _ = standardize(X[cols])
Xn = Xs.to_numpy(dtype=float)
cfg = load_benchmark_config()["models"]["path_agnn_cox"]
mcfg = dict(cfg)
mcfg["batch_size"] = int(os.environ.get("PATH_AGNN_BATCH_SIZE", mcfg["batch_size"]))
ids = torch.tensor([list(mem.columns).index(mem.loc[c].idxmax()) for c in cols])
adj_t = torch.tensor(adj[:len(cols), :len(cols)])
torch.manual_seed(SEED)
model = PathAGNNCox(n_genes=len(cols), adj=adj_t, pathway_ids=ids,
                    hidden=mcfg["hidden"], n_layers=mcfg["n_layers"],
                    mlp_hidden=mcfg["mlp_hidden"], dropout=mcfg["dropout"])
train_model(model, Xn, time_, event, epochs=EPOCHS, lr=mcfg["lr"],
            batch_size=mcfg["batch_size"], l2=mcfg["l2"],
            lambda_sparse=mcfg["lambda_sparse"], lambda_consist=mcfg["lambda_consist"])
torch.save(model.state_dict(), str(ROOT / "work" / "models" / "luad_ext_meta.pt"))
print("trained in %.1f min" % ((time.time() - t0) / 60.0), flush=True)

pos = {g: i for i, g in enumerate(cols)}
pid = mem.loc[cols].idxmax(axis=1)
pw_list = sorted(pid.unique())
extdir = ROOT / "data" / "processed" / DS / "external"
cohort_rows = []
for cohort in EXT:
    ext = pd.read_csv(extdir / (cohort + ".csv"))
    Ex = ext.drop(columns=["sample_id", "OS_time", "OS_event"], errors="ignore")
    Exm = Ex.reindex(columns=cols).fillna(0.0)
    _, Xe = standardize(X[cols], Exm)
    Xe = Xe.to_numpy(dtype=float)
    risk, alpha, src, dst = predict_with_alpha(model, Xe)
    src_i = np.array([pos[g] for g in cols[src]], dtype=int)
    dst_i = np.array([pos[g] for g in cols[dst]], dtype=int)
    nonself = src_i != dst_i
    pw_of_edge = pid.iloc[src_i].to_numpy()
    med = np.median(risk)
    hi = np.where(risk > med)[0]; lo = np.where(risk <= med)[0]
    n1, n2 = len(hi), len(lo)
    for pw in pw_list:
        sel = np.where((pw_of_edge == pw) & nonself)[0]
        if len(sel) < 5:
            continue
        scores = alpha[:, sel].mean(axis=1)
        a_hi, a_lo = scores[hi], scores[lo]
        try:
            u, p = mannwhitneyu(a_hi, a_lo, alternative="two-sided")
        except ValueError:
            continue
        mu = n1 * n2 / 2.0
        sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        z = (u - mu) / max(sd, 1e-12)
        d = a_hi.mean() - a_lo.mean()
        sp = np.sqrt(((n1 - 1) * a_hi.std(ddof=1) ** 2 + (n2 - 1) * a_lo.std(ddof=1) ** 2) / (n1 + n2 - 2.0))
        cd = d / sp if sp > 0 else 0.0
        cohort_rows.append({"cohort": cohort, "n": len(risk), "pathway": pw,
                            "n_edges": int(len(sel)), "z": float(z), "p": float(p),
                            "d": float(d), "cohen_d": float(cd)})
    print(cohort, "n", len(risk), "events", int(ext["OS_event"].sum()), "done", flush=True)

cdf = pd.DataFrame(cohort_rows)
cdf.to_csv(ROOT / "results" / "ext_pathway_cohorts.csv", index=False)

# ---- meta-analysis per pathway ----
rows = []
for pw, grp in cdf.groupby("pathway"):
    k = len(grp)
    zs = grp["z"].to_numpy()
    ns = grp["n"].to_numpy()
    ds = grp["cohen_d"].to_numpy()
    z_un = float(zs.sum() / np.sqrt(k))
    w = np.sqrt(ns)
    z_w = float((w * zs).sum() / np.sqrt((w ** 2).sum()))
    rows.append({
        "pathway": pw, "n_cohorts": k,
        "z_unweighted": z_un, "p_unweighted": float(2 * norm.sf(abs(z_un))),
        "z_weighted": z_w, "p_weighted": float(2 * norm.sf(abs(z_w))),
        "d_weighted_mean": float((ns * ds).sum() / ns.sum()),
        "n_direction_consistent": int((np.sign(zs) == np.sign(zs[0])).sum()),
        "cohort_z": ";".join("%.2f" % z for z in zs),
    })
meta = pd.DataFrame(rows)
meta["q_unweighted"] = bh_fdr(meta["p_unweighted"].to_numpy())
meta["q_weighted"] = bh_fdr(meta["p_weighted"].to_numpy())
meta = meta.sort_values("p_unweighted").reset_index(drop=True)
meta.to_csv(ROOT / "results" / "ext_pathway_meta.csv", index=False)
focus = meta[meta["pathway"].isin(["Homologous recombination", "DNA replication"])]
focus.to_csv(ROOT / "results" / "ext_pathway_meta_focus.csv", index=False)
print("META", flush=True)
print(focus[["pathway", "z_unweighted", "p_unweighted", "q_unweighted", "z_weighted", "p_weighted", "d_weighted_mean", "n_direction_consistent"]].to_string(index=False), flush=True)
print("EXT_PATHWAY_META_DONE", flush=True)

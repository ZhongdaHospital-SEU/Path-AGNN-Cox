# -*- coding: utf-8 -*-
"""Finish rewiring outputs for LUAD by reusing saved alpha.npy / risk_scores.csv:
computes H3 clinical correlations and the H4 static negative control (trains the
static model only, ~10 min), then prints the same summary as rewiring_analysis."""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
sys.path.insert(0, str(ROOT))
import torch
from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency
from path_agnn_cox.data import load_survival_data, split_features, standardize
from path_agnn_cox.models import PathAGNNCox
from path_agnn_cox.train import train_model, predict_with_alpha
from path_agnn_cox.evaluate import c_index
from benchmark.rewiring_analysis import rewiring_vs_clinical, static_null
from benchmark.dataset_manifest import load_benchmark_config

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--dataset", default="LUAD")
ds = _ap.parse_args().dataset
root = ROOT
out_dir = root / "results" / "rewiring" / ds
df = load_survival_data(str(root / "data" / "processed" / ds / "train.csv"))
X, time, event = split_features(df)
pathway_dict = load_gmt(str(root / "data" / "pathways" / "kegg_cancer_core.gmt"))
adj, mem, gene_order = build_pathway_adjacency(list(X.columns), pathway_dict)
cols = np.array([c for c in gene_order if c in X.columns])
Xs, _ = standardize(X[cols])
Xn = Xs.to_numpy(dtype=float)
ids = torch.tensor([list(mem.columns).index(mem.loc[c].idxmax()) for c in cols])
adj_t = torch.tensor(adj[:len(cols), :len(cols)])

alpha = np.load(out_dir / "alpha.npy")
risk = pd.read_csv(out_dir / "risk_scores.csv")["risk_score"].to_numpy()

# H3 clinical correlation (aligned 12-char barcodes)
clin_df = pd.read_csv(root / "data" / "processed" / "rewiring" / f"clinical_{ds}.csv")
sid = df["sample_id"].astype(str).str[:12].str.upper()
cid = clin_df["sample_id"].astype(str).str[:12].str.upper()
clin_df = clin_df.assign(_cid=cid).drop_duplicates(subset="_cid", keep="first")
clin_df = clin_df.set_index("_cid").reindex(sid)
clin_df = clin_df.drop(columns=["sample_id"], errors="ignore").reset_index()
clin_df["sample_id"] = sid.to_numpy()
if "MKI67" in X.columns:
    clin_df["ki67"] = X["MKI67"].to_numpy()
clin_res, rew = rewiring_vs_clinical(alpha, clin_df)
clin_res.to_csv(out_dir / "clinical_corr.csv", index=False)
print("H3 clinical corr:\n", clin_res.round(3).to_string(index=False), flush=True)

# H4 static negative control
cfg = load_benchmark_config()["models"]["path_agnn_cox"]
mcfg = dict(cfg); mcfg["batch_size"] = int(os.environ.get("PATH_AGNN_BATCH_SIZE", 64))
torch.manual_seed(0)
model_static = PathAGNNCox(n_genes=len(cols), adj=adj_t, pathway_ids=ids,
                           hidden=mcfg["hidden"], n_layers=mcfg["n_layers"],
                           mlp_hidden=mcfg["mlp_hidden"], dropout=mcfg["dropout"],
                           use_adaptive=False)
train_model(model_static, Xn, time, event, epochs=80, lr=mcfg["lr"],
            batch_size=mcfg["batch_size"], l2=mcfg["l2"], lambda_sparse=0.0, lambda_consist=0.0)
_, alpha_static, src_s, dst_s = predict_with_alpha(model_static, Xn)
null = static_null(alpha_static, src_s, dst_s)
pd.Series(null).to_csv(out_dir / "static_null.csv")
print("H4 static null:", null, flush=True)
ci = c_index(risk, time, event)
print(f"== {ds}: n={len(df)}, genes={len(cols)}, C-index={ci:.3f}", flush=True)
print("FINISH_DONE", flush=True)

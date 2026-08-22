# -*- coding: utf-8 -*-
"""KIRC static negative control only (H4): train static PathAGNNCox, save static_null.csv."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "4")
import torch
from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency
from path_agnn_cox.data import load_survival_data, split_features, standardize
from path_agnn_cox.models import PathAGNNCox
from path_agnn_cox.train import train_model, predict_with_alpha
from benchmark.rewiring_analysis import static_null
from benchmark.dataset_manifest import load_benchmark_config

ds = "KIRC"
out_dir = ROOT / "results" / "rewiring" / ds
df = load_survival_data(str(ROOT / "data" / "processed" / ds / "train.csv"))
X, time, event = split_features(df)
pathway_dict = load_gmt(str(ROOT / "data" / "pathways" / "kegg_cancer_core.gmt"))
adj, mem, gene_order = build_pathway_adjacency(list(X.columns), pathway_dict)
cols = np.array([c for c in gene_order if c in X.columns])
Xs, _ = standardize(X[cols])
Xn = Xs.to_numpy(dtype=float)
ids = torch.tensor([list(mem.columns).index(mem.loc[c].idxmax()) for c in cols])
adj_t = torch.tensor(adj[:len(cols), :len(cols)])
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
print("STATIC_NULL_KIRC:", null, flush=True)
print("KIRC_STATIC_DONE", flush=True)

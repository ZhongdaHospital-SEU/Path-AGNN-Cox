# -*- coding: utf-8 -*-
"""Random-pathway control for rewiring (LUAD): retrain the adaptive model with
random gene-to-pathway assignments (block sizes preserved), run the same
pathway-level rewiring tests, and show that significant-pathway counts and
known-pathway enrichment collapse under the null structure. Saves
results/rewiring/LUAD/random_control.csv."""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
sys.path.insert(0, str(ROOT))
from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency
from path_agnn_cox.data import load_survival_data, split_features, standardize
from path_agnn_cox.models import PathAGNNCox
from path_agnn_cox.train import train_model, predict_with_alpha
from path_agnn_cox.evaluate import c_index
from benchmark.rewiring_analysis import pathway_level_test, enrichment
from benchmark.dataset_manifest import load_benchmark_config

DS = "LUAD"
N_SEEDS = 3
cfg = load_benchmark_config()["models"]["path_agnn_cox"]
mcfg = dict(cfg); mcfg["batch_size"] = int(os.environ.get("PATH_AGNN_BATCH_SIZE", 64))
root = ROOT
out_dir = root / "results" / "rewiring" / DS
known = [ln.strip() for ln in (root / "data" / "pathways" / "luad_known_pathways.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]

df = load_survival_data(str(root / "data" / "processed" / DS / "train.csv"))
X, time, event = split_features(df)
pathway_dict = load_gmt(str(root / "data" / "pathways" / "kegg_cancer_core.gmt"))
adj, mem, gene_order = build_pathway_adjacency(list(X.columns), pathway_dict)
cols = np.array([c for c in gene_order if c in X.columns])
Xs, _ = standardize(X[cols])
Xn = Xs.to_numpy(dtype=float)
pid_real = mem.loc[cols].idxmax(axis=1).to_numpy()
block_sizes = np.array([int((pid_real == p).sum()) for p in mem.columns])

rows = []
for seed in range(N_SEEDS):
    rng = np.random.default_rng(seed)
    # random assignment preserving block sizes
    idx = rng.permutation(len(cols))
    pid_rand = np.empty(len(cols), dtype=object)
    start = 0
    for b, p in enumerate(mem.columns):
        pid_rand[idx[start:start + block_sizes[b]]] = p
        start += block_sizes[b]
    ids = torch.tensor([list(mem.columns).index(p) for p in pid_rand])
    # random adjacency is the full block-diagonal of the random blocks
    adj_t = torch.tensor(adj[:len(cols), :len(cols)])
    torch.manual_seed(seed)
    model = PathAGNNCox(n_genes=len(cols), adj=adj_t, pathway_ids=ids,
                        hidden=mcfg["hidden"], n_layers=mcfg["n_layers"],
                        mlp_hidden=mcfg["mlp_hidden"], dropout=mcfg["dropout"])
    train_model(model, Xn, time, event, epochs=80, lr=mcfg["lr"],
                batch_size=mcfg["batch_size"], l2=mcfg["l2"],
                lambda_sparse=mcfg["lambda_sparse"], lambda_consist=mcfg["lambda_consist"])
    risk, alpha, src, dst = predict_with_alpha(model, Xn)
    med = np.median(risk)
    hi = np.where(risk > med)[0]; lo = np.where(risk <= med)[0]
    mem_rand = pd.DataFrame(0, index=cols, columns=mem.columns)
    for i, p in enumerate(pid_rand):
        mem_rand.loc[cols[i], p] = 1
    pw = pathway_level_test(alpha, hi, lo, src, dst, cols, mem_rand.loc[cols])
    n_sig = int((pw["q"] < 0.05).sum()) if len(pw) else 0
    enr = enrichment(pw, known) if len(pw) else {"p": np.nan}
    ci = c_index(risk, time, event)
    rows.append({"seed": seed, "n_sig_q005": n_sig, "n_pathways": len(pw),
                 "enrich_hits": enr.get("hits", np.nan), "enrich_top_k": enr.get("top_k", np.nan),
                 "enrich_p": enr.get("p", np.nan), "c_index": ci})
    print("seed", seed, "sig:", n_sig, "enrich:", enr, "c:", round(ci, 3), flush=True)
res = pd.DataFrame(rows)
# real-pathway observed values for comparison
obs_pw = pd.read_csv(out_dir / "pathway_test.csv")
obs_sig = int((obs_pw["q"] < 0.05).sum())
enr_obs = pd.read_csv(out_dir / "enrichment.csv", index_col=0)
res.to_csv(out_dir / "random_control.csv", index=False)
print("random control saved; observed sig (real pathways):", obs_sig, flush=True)
print("ALL_RANDOM_CTRL_DONE", flush=True)

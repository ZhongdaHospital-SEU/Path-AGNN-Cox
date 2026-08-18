# -*- coding: utf-8 -*-
"""LUAD fold-1 validation of FIXED adaptive model (config A vs B)."""
import os, sys, time, json
import numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedKFold
torch.set_num_threads(6)
ROOT = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
sys.path.insert(0, ROOT)
from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency, pathway_gene_matrix
from path_agnn_cox.data import load_survival_data, split_features, standardize
from path_agnn_cox.models import PathAGNNCox
from path_agnn_cox.train import train_model, predict_risk, predict_with_alpha
from path_agnn_cox.evaluate import c_index

df = load_survival_data(os.path.join(ROOT, "data", "processed", "LUAD", "train.csv"))
X, stime, event = split_features(df)
pathway_dict = load_gmt(os.path.join(ROOT, "data", "pathways", "kegg_cancer_core.gmt"))
col_upper = {c.upper(): c for c in X.columns}
mem = pathway_gene_matrix(X.columns.to_numpy(), pathway_dict)
keep = [col_upper[g] for g in mem.index if g in col_upper]
X = X[keep]
gene_order = X.columns.to_numpy()
print("genes in pathway set:", X.shape[1], flush=True)
adj, mem2, genes = build_pathway_adjacency(gene_order, pathway_dict)
cols = [c for c in genes if c in X.columns]
adj_a = adj[:len(cols), :len(cols)]
ids = torch.tensor(mem2.loc[cols].idxmax(axis=1).map(lambda v: list(mem2.columns).index(v)).to_numpy())

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tr_idx, va_idx = next(iter(skf.split(X, event)))
Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
ttr, etr = stime[tr_idx], event[tr_idx]
tva, eva = stime[va_idx], event[va_idx]
Xtr_s, Xva_s = standardize(Xtr, Xva)
Xtr_a = Xtr_s[cols].to_numpy(dtype=float)
Xva_a = Xva_s[cols].to_numpy(dtype=float)
print("fold1 train/val:", Xtr_a.shape, Xva_a.shape, "events:", int(etr.sum()), int(eva.sum()), flush=True)

def analyze(m, Xa, n_samples=16):
    r, alpha, src, dst = predict_with_alpha(m, Xa[:n_samples])
    eps = 1e-12
    ent = -(alpha * np.log(alpha + eps)).sum(axis=1)
    with torch.no_grad():
        x = torch.tensor(Xa[:n_samples], dtype=torch.float32)
        h = torch.relu(m.embed(x.unsqueeze(-1)))
        s = torch.sigmoid(m.layers[-1].s_mlp(h.mean(dim=1))).numpy().ravel()
    corr = float(np.corrcoef(s, ent)[0, 1]) if n_samples > 2 else float("nan")
    return corr, float(s.min()), float(s.max()), float(ent.min()), float(ent.max())

def run(name, cfg):
    t0 = time.time()
    m = PathAGNNCox(n_genes=len(cols), adj=torch.tensor(adj_a), pathway_ids=ids,
                    hidden=cfg["hidden"], n_layers=cfg["n_layers"], mlp_hidden=cfg["mlp_hidden"],
                    dropout=cfg["dropout"], use_adaptive=True)
    m = train_model(m, Xtr_a, ttr, etr, Xva_a, tva, eva,
                    epochs=cfg["epochs"], lr=cfg["lr"], batch_size=cfg["batch_size"],
                    l2=cfg["l2"], lambda_sparse=cfg["lambda_sparse"],
                    lambda_consist=cfg["lambda_consist"], patience=cfg["patience"])
    risk = predict_risk(m, Xva_a)
    ci = c_index(risk, tva, eva)
    beta = float(m.layers[-1].beta.item())
    corr, smin, smax, emin, emax = analyze(m, Xva_a)
    os.makedirs(os.path.join(ROOT, "work", "models"), exist_ok=True)
    torch.save(m.state_dict(), os.path.join(ROOT, "work", "models", "val_%s.pt" % name))
    out = dict(name=name, c_index=round(float(ci), 4), beta=round(beta, 4),
               corr_s_entropy=corr, s_min=smin, s_max=smax,
               ent_min=emin, ent_max=emax, time_s=round(time.time() - t0, 1))
    print(json.dumps(out), flush=True)
    return out

cfgA = dict(hidden=32, n_layers=2, mlp_hidden=32, dropout=0.1, epochs=100, lr=1e-3,
            batch_size=128, l2=1e-4, lambda_sparse=1e-3, lambda_consist=0.1, patience=15)
cfgB = dict(hidden=64, n_layers=2, mlp_hidden=64, dropout=0.2, epochs=200, lr=5e-4,
            batch_size=64, l2=1e-4, lambda_sparse=1e-3, lambda_consist=0.1, patience=30)
res = [run("A_current", cfgA), run("B_tuned", cfgB)]
with open(os.path.join(ROOT, "work", "validate_results.json"), "w") as fh:
    json.dump(res, fh, indent=2)
print("DONE", flush=True)

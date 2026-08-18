# -*- coding: utf-8 -*-
"""Standard GAT negative control: no pathway constraint, same pipeline."""
import sys, os
sys.path.insert(0, r"D:/TT paper/Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
import numpy as np, pandas as pd, torch
from path_agnn_cox.data import load_survival_data, split_features, standardize
from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency
from path_agnn_cox.train import train_model
from baselines.standard_gat import StandardGAT, build_knn_edges
from benchmark.rewiring_analysis import pathway_level_test, bh_fdr

root = r"D:/TT paper/Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
ds = sys.argv[1] if len(sys.argv) > 1 else "LUAD"
knn_k = int(os.environ.get("STDGAT_K", "10"))
epochs = int(os.environ.get("STDGAT_EPOCHS", "80"))
df = load_survival_data(os.path.join(root, "data", "processed", ds, "train.csv"))
X, time, event = split_features(df)
pathway_dict = load_gmt(os.path.join(root, "data", "pathways", "kegg_cancer_core.gmt"))
adj, mem, gene_order = build_pathway_adjacency(list(X.columns), pathway_dict)
cols = np.array([c for c in gene_order if c in X.columns])
Xs, _ = standardize(X[cols])
Xn = Xs.to_numpy(dtype=float)
edge = build_knn_edges(Xn, k=knn_k)
print(ds, "genes", len(cols), "knn edges", edge.shape[1], flush=True)
torch.manual_seed(0); np.random.seed(0)
model = StandardGAT(n_genes=len(cols), edge_index=torch.tensor(edge, dtype=torch.int64),
                    hidden=64, n_layers=2, mlp_hidden=32, dropout=0.1)
train_model(model, Xn, time, event, epochs=epochs, lr=1e-3, batch_size=64,
            l2=1e-4, lambda_sparse=0.0, lambda_consist=0.0, seed=0)
def predict_chunked(model, Xn, batch=64):
    """Full-cohort forward in chunks to bound peak memory (h[:, src] is (B,E,d))."""
    model.eval()
    risks, alphas = [], []
    srcs = dsts = None
    with torch.no_grad():
        for i in range(0, len(Xn), batch):
            xb = torch.tensor(Xn[i:i + batch], dtype=torch.float32)
            r, a, s, d = model(xb, return_alpha=True)
            risks.append(r.numpy()); alphas.append(a.numpy())
            srcs = s.numpy(); dsts = d.numpy()
    return np.concatenate(risks), np.concatenate(alphas), srcs, dsts


risk, alpha, src, dst = predict_chunked(model, Xn)
risk = risk.ravel()
med = np.median(risk)
hi, lo = np.where(risk > med)[0], np.where(risk <= med)[0]
memc = mem.loc[cols]
pw = pathway_level_test(alpha, hi, lo, src, dst, cols, memc)
outdir = os.path.join(root, "results", "rewiring", ds)
os.makedirs(outdir, exist_ok=True)
pw.to_csv(os.path.join(outdir, "stdgat_pathway_test.csv"), index=False)
np.save(os.path.join(outdir, "stdgat_alpha.npy"), alpha)
print(ds, "stdgat pathways tested", len(pw), "sig q<0.05", int((pw["q"] < 0.05).sum()) if len(pw) else 0, flush=True)
print("STDGAT_DONE_" + ds, flush=True)


import sys, os
sys.path.insert(0, r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
import numpy as np, torch
from path_agnn_cox.data import load_survival_data, split_features, standardize
from baselines.standard_gat import StandardGAT, build_knn_edges
from path_agnn_cox.train import train_model

df = load_survival_data(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis/data/processed/LUAD/train.csv")
X, time, event = split_features(df)
cols = list(X.columns[:2000])
Xs, _ = standardize(X[cols])
Xn = Xs.to_numpy(float)
edge = build_knn_edges(Xn, k=8)
print("edges:", edge.shape)
model = StandardGAT(n_genes=len(cols), edge_index=torch.tensor(edge, dtype=torch.int64),
                    hidden=32, n_layers=1, mlp_hidden=16, dropout=0.1)
train_model(model, Xn, time, event, epochs=5, lr=1e-3, batch_size=64, verbose=True)
risk, alpha, src, dst = model(torch.tensor(Xn, dtype=torch.float32), return_alpha=True)
print("risk:", risk.shape, "alpha:", alpha.shape, "src:", src.shape)
print("SMOKE_OK")

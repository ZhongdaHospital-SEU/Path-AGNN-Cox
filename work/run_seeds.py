# -*- coding: utf-8 -*-
"""3-seed internal CV for Path-AGNN-Cox on one dataset (replicates benchmark folds)."""
import sys, os
sys.path.insert(0, r"D:/TT paper/Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
import numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedKFold
from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency, pathway_gene_matrix
from path_agnn_cox.data import load_survival_data, split_features, standardize
from path_agnn_cox.models import PathAGNNCox
from path_agnn_cox.train import train_model, predict_risk
from path_agnn_cox.evaluate import full_report
from benchmark.dataset_manifest import load_benchmark_config, load_datasets

root = r"D:/TT paper/Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
ds = sys.argv[1] if len(sys.argv) > 1 else "LUAD"
seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
cfg = load_benchmark_config()
mcfg = cfg["models"]["path_agnn_cox"]
folds = mcfg.get("cv_folds", 5)

df = load_survival_data(os.path.join(root, "data", "processed", ds, "train.csv"))
X, time, event = split_features(df)
pathway_dict = load_gmt(os.path.join(root, "data", "pathways", "kegg_cancer_core.gmt"))
col_upper = {c.upper(): c for c in X.columns}
mem = pathway_gene_matrix(X.columns.to_numpy(), pathway_dict)
keep = [col_upper[g] for g in mem.index if g in col_upper]
X = X[keep]
gene_order = X.columns.to_numpy()
adj, memm, genes = build_pathway_adjacency(gene_order, pathway_dict)
cols = [c for c in genes if c in X.columns]
adj_a = adj[:len(cols), :len(cols)]
mem_a = memm.loc[cols]
ids = torch.tensor(mem_a.idxmax(axis=1).map(lambda v: list(mem_a.columns).index(v)).to_numpy())

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg["evaluation"]["seed"])
folds_ci = []
rows = []
for seed in seeds:
    torch.manual_seed(seed); np.random.seed(seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, event)):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ttr, etr = time[tr_idx], event[tr_idx]
        tva, eva = time[va_idx], event[va_idx]
        Xtr_a, Xva_a = standardize(Xtr, Xva)
        Xtr_n = Xtr_a[cols].to_numpy(dtype=float)
        Xva_n = Xva_a[cols].to_numpy(dtype=float)
        m = PathAGNNCox(n_genes=len(cols), adj=torch.tensor(adj_a), pathway_ids=ids,
                        hidden=mcfg["hidden"], n_layers=mcfg["n_layers"],
                        mlp_hidden=mcfg["mlp_hidden"], dropout=mcfg["dropout"])
        m = train_model(m, Xtr_n, ttr, etr, Xva_n, tva, eva,
                        epochs=mcfg["epochs"], lr=mcfg["lr"], batch_size=mcfg["batch_size"],
                        l2=mcfg["l2"], lambda_sparse=mcfg["lambda_sparse"],
                        lambda_consist=mcfg["lambda_consist"], patience=mcfg["patience"],
                        seed=seed)
        ci = full_report(predict_risk(m, Xva_n), tva, eva)["c_index"]
        folds_ci.append(ci)
        rows.append({"dataset": ds, "seed": seed, "fold": fold, "c_index": ci})
        print(ds, "seed", seed, "fold", fold, "c_index %.3f" % ci, flush=True)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(root, "results", "seed_analysis_%s.csv" % ds), index=False)
print(ds, "MEAN %.3f SD %.3f over %d (seed x fold) runs" %
      (np.mean(folds_ci), np.std(folds_ci), len(folds_ci)), flush=True)
print("SEEDS_DONE_" + ds, flush=True)
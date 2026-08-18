# -*- coding: utf-8 -*-
"""External rewiring replication: train on TCGA (LUAD/BRCA), transfer to GEO cohorts."""
import sys, os
sys.path.insert(0, r"D:/TT paper/Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
import numpy as np, pandas as pd, torch
from path_agnn_cox.data import load_survival_data, split_features, standardize
from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency
from path_agnn_cox.models import PathAGNNCox
from path_agnn_cox.train import train_model
from benchmark.dataset_manifest import load_benchmark_config

root = r"D:/TT paper/Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
ds = sys.argv[1] if len(sys.argv) > 1 else "LUAD"
epochs = int(os.environ.get("EXT_RW_EPOCHS", "80"))
outbase = os.path.join(root, "results", "rewiring_external", ds)
os.makedirs(outbase, exist_ok=True)

# ---- train on TCGA ----
df = load_survival_data(os.path.join(root, "data", "processed", ds, "train.csv"))
X, time, event = split_features(df)
pathway_dict = load_gmt(os.path.join(root, "data", "pathways", "kegg_cancer_core.gmt"))
adj, mem, gene_order = build_pathway_adjacency(list(X.columns), pathway_dict)
cols = np.array([c for c in gene_order if c in X.columns])
Xs, _ = standardize(X[cols])
Xn = Xs.to_numpy(dtype=float)
cfg = load_benchmark_config()["models"]["path_agnn_cox"]
import torch as _t
ids = _t.tensor([list(mem.columns).index(mem.loc[c].idxmax()) for c in cols])
adj_t = _t.tensor(adj[:len(cols), :len(cols)])
_t.manual_seed(0)
model = PathAGNNCox(n_genes=len(cols), adj=adj_t, pathway_ids=ids,
                    hidden=cfg["hidden"], n_layers=cfg["n_layers"],
                    mlp_hidden=cfg["mlp_hidden"], dropout=cfg["dropout"])
train_model(model, Xn, time, event, epochs=epochs, lr=cfg["lr"],
            batch_size=int(os.environ.get("PATH_AGNN_BATCH_SIZE", cfg["batch_size"])),
            l2=cfg["l2"], lambda_sparse=cfg["lambda_sparse"],
            lambda_consist=cfg["lambda_consist"], seed=0)
print(ds, "trained, genes", len(cols), flush=True)

# ---- transfer to external cohorts ----
extdir = os.path.join(root, "data", "processed", ds, "external")
if not os.path.isdir(extdir):
    print("no external dir"); sys.exit(0)
for fname in sorted(os.listdir(extdir)):
    if not fname.endswith(".csv"):
        continue
    cohort = fname[:-4]
    ext = pd.read_csv(os.path.join(extdir, fname))
    Ex = ext.drop(columns=["sample_id", "OS_time", "OS_event"])
    genes_in = [c for c in cols if c in Ex.columns]
    Exm = Ex.reindex(columns=cols).fillna(0.0)
    _, Xe = standardize(X[cols], Exm)
    Xe = Xe.to_numpy(dtype=float)
    model.eval()
    with torch.no_grad():
        risk, alpha, src, dst = model(torch.tensor(Xe, dtype=torch.float32),
                                     return_alpha=True)
    risk = risk.numpy(); alpha = alpha.numpy()
    mu = alpha.mean(axis=0)
    rew = np.abs(alpha - mu).sum(axis=1)
    res = pd.DataFrame({"sample_id": ext["sample_id"], "risk_score": risk,
                        "rewiring_magnitude": rew,
                        "OS_time": pd.to_numeric(ext["OS_time"], errors="coerce"),
                        "OS_event": pd.to_numeric(ext["OS_event"], errors="coerce")})
    med = np.median(rew)
    res["group"] = (rew >= med).astype(int)
    # external survival association (Cox HR of high vs low rewiring)
    hr = np.nan; p = np.nan; lo = np.nan; hi = np.nan
    try:
        from lifelines import CoxPHFitter
        c = res[["OS_time", "OS_event", "group"]].rename(columns={"OS_time": "T", "OS_event": "E"})
        c = c.dropna(subset=["T", "E", "group"])
        if c["E"].sum() >= 5 and len(c) >= 20:
            cf = CoxPHFitter().fit(c, duration_col="T", event_col="E")
            # lifelines 0.30: hazard_ratios_ is exp-scale; confidence_intervals_ is log-scale
            hr = float(cf.hazard_ratios_["group"])
            ci = cf.confidence_intervals_.loc["group"]
            lo = float(np.exp(ci.iloc[0])); hi = float(np.exp(ci.iloc[1]))
            p = float(cf.summary["p"]["group"])
    except Exception as e:
        print("cox err", cohort, e, flush=True)
    res.to_csv(os.path.join(outbase, cohort + ".csv"), index=False)
    summ = pd.DataFrame([{"cohort": cohort, "n": len(c), "events": int(c["E"].sum()),
                          "hr_high_vs_low": hr, "hr_lo": lo, "hr_hi": hi, "p": p,
                          "risk_rew_corr": float(np.corrcoef(risk, rew)[0, 1])}])
    summ.to_csv(os.path.join(outbase, cohort + "_summary.csv"), index=False)
    print(cohort, "n", len(res), "HR", round(hr, 3), "P", round(p, 4), flush=True)
print("EXT_REWIRING_DONE_" + ds, flush=True)

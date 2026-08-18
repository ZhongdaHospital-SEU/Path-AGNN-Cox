# -*- coding: utf-8 -*-
"""Config-B diagnostic on LUAD: 5-fold CV for 4 deep models with
hidden=64 / epochs=200 / patience=30 (vs benchmark config A: hidden=32 / 100 / 15).
Reuses benchmark.run_benchmark._run_model for identical data/standardization."""
import os, sys
from pathlib import Path
ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
sys.path.insert(0, str(ROOT))
import numpy as np
from benchmark.run_benchmark import _run_model, _ArrayPredictor, MODEL_FACTORY
from benchmark.dataset_manifest import load_benchmark_config, load_datasets
from path_agnn_cox.pathway import load_gmt, pathway_gene_matrix
from path_agnn_cox.data import load_survival_data, split_features

def main():
    dsname = sys.argv[1] if len(sys.argv) > 1 else "LUAD"
    models = sys.argv[2].split(",") if len(sys.argv) > 2 else ["path_agnn_cox"]
    cfg = load_benchmark_config()
    # config B override for deep GNN family
    import os as _os
    _bs = int(_os.environ.get("PATH_AGNN_BATCH_SIZE", 128))
    for m in ["path_agnn_cox", "path_agnn_cox_static", "path_agnn_cox_noreg"]:
        cfg["models"][m].update(hidden=64, mlp_hidden=64, epochs=200, patience=30,
                                batch_size=_bs, lr=1e-3, l2=1e-4,
                                lambda_sparse=0.001, lambda_consist=0.1)
    ds = [d for d in load_datasets() if d["name"] == dsname][0]
    p = cfg["paths"]
    train_csv = ROOT / p["processed"] / dsname / "train.csv"
    gmt_path = ROOT / p["pathway_gmt"]
    pathway_dict = load_gmt(str(gmt_path))
    df = load_survival_data(str(train_csv))
    X, time, event = split_features(df)
    col_upper = {c.upper(): c for c in X.columns}
    mem = pathway_gene_matrix(X.columns.to_numpy(), pathway_dict)
    keep = [col_upper[g] for g in mem.index if g in col_upper]
    X = X[keep]
    gene_order = X.columns.to_numpy()
    print(f"== {dsname}: {len(df)} samples, {X.shape[1]} genes ==", flush=True)
    for model_name in models:
        rows = _run_model(dsname, ds, model_name, cfg, X, time, event,
                          gene_order, pathway_dict, ROOT / p["processed"], 5)
        fc = [r["c_index"] for r in rows if r["split"] == "cv"]
        print(f"{model_name:20s} configB folds={np.round(fc,4)} mean={np.mean(fc):.4f} sd={np.std(fc):.4f}", flush=True)
    print("DIAG_DONE", flush=True)

if __name__ == "__main__":
    main()

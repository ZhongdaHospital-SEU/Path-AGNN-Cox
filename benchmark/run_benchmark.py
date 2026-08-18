"""Run the full benchmark: internal CV on each TCGA cohort + external GEO validation.

Usage:
    python -m benchmark.run_benchmark --datasets LUAD,BRCA --models path_agnn_cox,lasso_cox
    python -m benchmark.run_benchmark                      # all datasets, all models
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency, pathway_gene_matrix
from path_agnn_cox.data import load_survival_data, split_features, standardize
from path_agnn_cox.models import PathAGNNCox
from path_agnn_cox.train import train_model, predict_risk
from path_agnn_cox.evaluate import full_report
from benchmark.dataset_manifest import load_datasets, load_benchmark_config

MODEL_FACTORY = {}


def _register(name):
    def deco(fn):
        MODEL_FACTORY[name] = fn
        return fn
    return deco


@_register("path_agnn_cox")
def _fit_agnn(cfg, Xtr, ttr, etr, Xva, tva, eva, gene_order, pathway_dict):
    adj, mem, genes = build_pathway_adjacency(gene_order, pathway_dict)
    import torch
    import pandas as pd
    # align X columns to gene order present in matrix
    cols = [c for c in genes if c in Xtr.columns]
    Xtr_a = Xtr[cols].to_numpy(dtype=float)
    Xva_a = Xva[cols].to_numpy(dtype=float) if Xva is not None else None
    adj_a = adj[:len(cols), :len(cols)]
    mem_a = mem.loc[cols]
    ids = torch.tensor(mem_a.idxmax(axis=1).map(lambda v: list(mem_a.columns).index(v)).to_numpy())
    m = PathAGNNCox(n_genes=len(cols), adj=torch.tensor(adj_a),
                    pathway_ids=ids, hidden=cfg["hidden"],
                    n_layers=cfg["n_layers"], mlp_hidden=cfg["mlp_hidden"],
                    dropout=cfg["dropout"],
                    use_adaptive=cfg.get("use_adaptive", True))
    if Xva_a is not None:
        m = train_model(m, Xtr_a, ttr, etr, Xva_a, tva, eva,
                        epochs=cfg["epochs"], lr=cfg["lr"], batch_size=cfg["batch_size"],
                        l2=cfg["l2"], lambda_sparse=cfg["lambda_sparse"],
                        lambda_consist=cfg["lambda_consist"], patience=cfg["patience"])
    else:
        m = train_model(m, Xtr_a, ttr, etr, epochs=cfg["epochs"], lr=cfg["lr"],
                        batch_size=cfg["batch_size"], l2=cfg["l2"],
                        lambda_sparse=cfg["lambda_sparse"],
                        lambda_consist=cfg["lambda_consist"], patience=cfg["patience"])
    return _ArrayPredictor(lambda X: predict_risk(m, X), prep=lambda X: X[cols].to_numpy(dtype=float))


@_register("path_agnn_cox_static")
def _fit_agnn_static(cfg, *a):
    """Ablation -Adaptive: fixed normalized adjacency (cfg.use_adaptive=false)."""
    return _fit_agnn(cfg, *a)


@_register("path_agnn_cox_noreg")
def _fit_agnn_noreg(cfg, *a):
    """Ablation -Regularization: no sparse/consistency terms (cfg lambdas = 0)."""
    return _fit_agnn(cfg, *a)


@_register("lasso_cox")
def _fit_lasso(cfg, Xtr, ttr, etr, *a):
    from baselines.cox_penalized import lasso_cox
    return _ArrayPredictor(lasso_cox(Xtr.to_numpy(), ttr, etr, **cfg).predict_risk)


@_register("ridge_cox")
def _fit_ridge(cfg, Xtr, ttr, etr, *a):
    from baselines.cox_penalized import ridge_cox
    return _ArrayPredictor(ridge_cox(Xtr.to_numpy(), ttr, etr, **cfg).predict_risk)


@_register("elastic_net")
def _fit_en(cfg, Xtr, ttr, etr, *a):
    from baselines.cox_penalized import elastic_net_cox
    return _ArrayPredictor(elastic_net_cox(Xtr.to_numpy(), ttr, etr, **cfg).predict_risk)


@_register("rsf")
def _fit_rsf(cfg, Xtr, ttr, etr, *a):
    from baselines.rsf import RSFModel
    return _ArrayPredictor(RSFModel(**cfg).fit(Xtr.to_numpy(), ttr, etr).predict_risk)


@_register("deepsurv")
def _fit_ds(cfg, Xtr, ttr, etr, *a):
    from baselines.deepsurv import deepsurv
    return _ArrayPredictor(deepsurv(Xtr.to_numpy(), ttr, etr, **cfg).predict_risk)


@_register("cox_nnet")
def _fit_cn(cfg, Xtr, ttr, etr, *a):
    from baselines.deepsurv import cox_nnet
    return _ArrayPredictor(cox_nnet(Xtr.to_numpy(), ttr, etr, **cfg).predict_risk)


@_register("plain_gnn")
def _fit_pgnn(cfg, Xtr, ttr, etr, Xva, tva, eva, *a):
    from baselines.plain_gnn import plain_gnn_survival
    Xv = Xva.to_numpy() if Xva is not None else None
    p = plain_gnn_survival(Xtr.to_numpy(), ttr, etr, Xv, tva if Xv is not None else None,
                           eva if Xv is not None else None, adj_mode=cfg.get("adj_mode", "identity"))
    return _ArrayPredictor(p.predict_risk)


class _ArrayPredictor:
    def __init__(self, fn, prep=None):
        self.fn = fn
        self.prep = prep

    def predict_risk(self, X):
        if self.prep is not None:
            X = self.prep(X)
        elif hasattr(X, "to_numpy"):
            X = X.to_numpy(dtype=float)
        return np.asarray(self.fn(X), dtype=float)


def _report_row(dataset, split, model, cohort, report, n):
    row = {"dataset": dataset, "split": split, "model": model, "cohort": cohort,
           "n": n, "c_index": report.get("c_index", np.nan),
           "auc_mean": report.get("auc_mean", np.nan),
           "calib_slope": report.get("slope", np.nan)}
    if report.get("auc_times") is not None:
        at = report["auc_times"]
        row["auc_t1"], row["auc_t2"], row["auc_t3"] = list(at["auc"])[:3]
    return row


def _save_incremental(rows, out_path):
    """Append a chunk of rows to the results CSV (header only if new)."""
    import os
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    df.to_csv(out_path, mode="a", header=write_header, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None, help="comma list, default all")
    ap.add_argument("--models", default=None, help="comma list, default all")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--external-only", action="store_true",
                    help="skip internal CV folds; only full-data training + external cohorts")
    args = ap.parse_args()

    datasets = load_datasets()
    cfg = load_benchmark_config()
    p = cfg["paths"]
    root = ROOT
    processed = root / p["processed"]
    results_dir = Path(args.out) if args.out else root / p["results"]
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(exist_ok=True)

    if args.datasets:
        names = [s.strip() for s in args.datasets.split(",")]
        datasets = [d for d in datasets if d["name"] in names]
    models = ([s.strip() for s in args.models.split(",")] if args.models
              else list(MODEL_FACTORY.keys()))

    gmt_path = root / p["pathway_gmt"]
    pathway_dict = load_gmt(str(gmt_path)) if gmt_path.exists() else {}

    folds = args.folds or cfg["models"]["path_agnn_cox"].get("cv_folds", 5)
    rows = []

    for ds in datasets:
        name = ds["name"]
        train_csv = processed / name / "train.csv"
        if not train_csv.exists():
            print(f"[skip] {name}: no processed train.csv at {train_csv}")
            continue
        df = load_survival_data(str(train_csv))
        X, time, event = split_features(df)
        if pathway_dict:
            # Fair comparison: all models (GNN and baselines) use the same
            # pathway-mapped gene set; also avoids OOM on 59k x 59k matrices.
            # Case-insensitive match: GMT genes are uppercase while expression
            # columns may be mixed-case (e.g. C8orf44-SGK3).
            col_upper = {c.upper(): c for c in X.columns}
            mem = pathway_gene_matrix(X.columns.to_numpy(), pathway_dict)
            keep = [col_upper[g] for g in mem.index if g in col_upper]
            X = X[keep]
        gene_order = X.columns.to_numpy()
        print(f"== {name}: {len(df)} samples, {X.shape[1]} genes (pathway set) ==")

        for model_name in models:
            if model_name not in MODEL_FACTORY:
                continue
            try:
                self_rows = _run_model(name, ds, model_name, cfg, X, time, event,
                                       gene_order, pathway_dict, processed,
                                       folds, external_only=args.external_only)
                rows.extend(self_rows)
                _save_incremental(self_rows, results_dir / "benchmark_results.csv")
            except Exception as exc:  # noqa: BLE001 - keep benchmark alive
                print(f"  [ERROR] {model_name} on {name}: {exc}")
                with open(results_dir / "benchmark_errors.log", "a") as fh:
                    fh.write(f"{name}\t{model_name}\t{exc}\n")

    out_path = results_dir / "benchmark_results.csv"
    _save_incremental(rows, out_path)
    out_df = pd.DataFrame(rows)
    print(f"\nSaved ({len(rows)} rows): {out_path}")


def _run_model(name, ds, model_name, cfg, X, time, event, gene_order,
               pathway_dict, processed, folds, external_only=False):
    """Run one model on one dataset: internal CV + external validation."""
    from sklearn.model_selection import StratifiedKFold
    mcfg = cfg["models"].get(model_name, {})
    rows = []
    fold_c = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg["evaluation"]["seed"])
    if external_only:
        print(f"  {model_name:12s} external-only mode")
    for tr_idx, va_idx in skf.split(X, event) if not external_only else []:
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ttr, etr = time[tr_idx], event[tr_idx]
        tva, eva = time[va_idx], event[va_idx]
        Xtr, Xva = standardize(Xtr, Xva)
        pred = MODEL_FACTORY[model_name](mcfg, Xtr, ttr, etr, Xva, tva, eva,
                                         gene_order, pathway_dict)
        rep = full_report(pred.predict_risk(Xva), tva, eva)
        fold_c.append(rep.get("c_index", np.nan))
        rows.append(_report_row(name, "cv", model_name, f"fold{len(fold_c)}",
                                rep, len(va_idx)))
    print(f"  {model_name:12s} internal C-index: {np.mean(fold_c):.3f} +- {np.std(fold_c):.3f}")

    mu = X.mean()
    sd = X.std(ddof=0).replace(0, 1.0)
    Xtr = (X - mu) / sd
    pred = MODEL_FACTORY[model_name](mcfg, Xtr, time, event, None, None, None,
                                     gene_order, pathway_dict)
    ext_dir = processed / name / "external"
    for gse in ds.get("external", []):
        ext_csv = ext_dir / f"{gse}.csv"
        if not ext_csv.exists():
            continue
        edf = load_survival_data(str(ext_csv))
        Xe, te, ee = split_features(edf)
        Xte = Xe.reindex(columns=X.columns)
        Xte = (Xte - mu) / sd
        Xte = Xte.fillna(0.0)
        risk = pred.predict_risk(Xte)
        rep = full_report(risk, te, ee)
        rows.append(_report_row(name, "external", model_name, gse, rep, len(edf)))
        print(f"  {model_name:12s} external {gse}: C-index {rep.get('c_index', np.nan):.3f}")
    return rows


if __name__ == "__main__":
    main()

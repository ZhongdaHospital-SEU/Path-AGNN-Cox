"""Summarize benchmark_results.csv into paper-ready tables and figures.

Outputs (results/):
  internal_cindex_table.csv / external_cindex_table.csv   (dataset x model)
  paper_main_table.md                                     (model x dataset + paired tests)
  paper_ablation_table.md                                 (ablation variants)
  summary_long.csv                                        (long format for plotting)
  figures/cindex_comparison.png
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.dataset_manifest import load_benchmark_config

MAIN_MODELS = ["path_agnn_cox", "lasso_cox", "ridge_cox", "elastic_net",
               "rsf", "deepsurv", "cox_nnet", "plain_gnn"]
ABLATION = ["path_agnn_cox", "path_agnn_cox_static", "path_agnn_cox_noreg", "plain_gnn"]
LABELS = {
    "path_agnn_cox": "Path-AGNN-Cox",
    "path_agnn_cox_static": "Path-AGNN-Cox (-Adaptive)",
    "path_agnn_cox_noreg": "Path-AGNN-Cox (-Regularization)",
    "plain_gnn": "Plain GNN (-Pathway)",
    "lasso_cox": "LASSO-Cox", "ridge_cox": "Ridge-Cox", "elastic_net": "EN-Cox",
    "rsf": "RSF", "deepsurv": "DeepSurv", "cox_nnet": "Cox-nnet",
}


def _fmt_p(v) -> str:
    """P-value house rule: P>=0.001 -> 3 decimals; P<0.001 -> 'P<0.001'."""
    if v is None or (isinstance(v, float) and (v != v)):
        return "NA"
    if isinstance(v, str):
        return v
    if v < 0.001:
        return "P<0.001"
    return "P=%.3f" % v


def _fmt_mean_sd(x: pd.Series) -> str:
    m = x["mean"] if "mean" in x else x
    s = x["std"] if "std" in x else float("nan")
    return f"{m:.2f}±{s:.2f}"


def _paired_wilcoxon(cv: pd.DataFrame, reference: str, model: str) -> float | str:
    """Paired Wilcoxon signed-rank on per-dataset mean C-index (internal CV)."""
    ref = cv.xs(reference, level="model")["mean"].reindex(cv.xs(model, level="model").index)
    alt = cv.xs(model, level="model")["mean"]
    common = alt.index.intersection(ref.index)
    if len(common) < 3:
        return "NA"
    d = alt.loc[common].to_numpy() - ref.loc[common].to_numpy()
    if np.allclose(d, 0):
        return "NA"
    try:
        return float(wilcoxon(d).pvalue)
    except ValueError:
        return "NA"


def _paper_table(cv: pd.DataFrame, ext: pd.DataFrame, models: list[str]) -> str:
    lines = []
    lines.append("| Model | " + " | ".join(
        [f"{d}" for d in cv.index.get_level_values("dataset").unique()] +
        ["Internal mean\u00b1SD", "External mean\u00b1SD", "Wilcoxon p (vs full)"]) + " |")
    lines.append("|---" + "|---" * (len(cv.index.get_level_values("dataset").unique()) + 3) + "|")
    for m in models:
        if m not in cv.index.get_level_values("model"):
            continue
        row = [LABELS.get(m, m)]
        for d in cv.index.get_level_values("dataset").unique():
            try:
                row.append(_fmt_mean_sd(cv.loc[(d, m)]))
            except KeyError:
                row.append("-")
        try:
            row.append(_fmt_mean_sd(cv.xs(m, level="model")["mean"].agg(["mean", "std"])))
        except Exception:
            row.append("-")
        try:
            row.append(_fmt_mean_sd(ext.xs(m, level="model")["mean"].agg(["mean", "std"])))
        except Exception:
            row.append("-")
        if m == models[0]:
            row.append("ref")
        else:
            row.append(_fmt_p(_paired_wilcoxon(cv, models[0], m)))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_benchmark_config()
    root = ROOT
    res_path = Path(args.results) if args.results else root / cfg["paths"]["results"] / "benchmark_results.csv"
    out_dir = Path(args.out) if args.out else root / cfg["paths"]["results"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    df = pd.read_csv(res_path)

    cv = df[df["split"] == "cv"].groupby(["dataset", "model"])["c_index"].agg(["mean", "std", "count"])
    cv_table = cv.pivot_table(index="dataset", columns="model", values="mean")
    cv_table.to_csv(out_dir / "internal_cindex_table.csv")
    print("Internal C-index table:\n", cv_table.round(3))

    ext = df[df["split"] == "external"]
    ext_sum = ext.groupby(["dataset", "model"])["c_index"].agg(["mean", "std", "count"])
    ext_table = ext_sum.pivot_table(index="dataset", columns="model", values="mean")
    ext_table.to_csv(out_dir / "external_cindex_table.csv")
    print("\nExternal C-index table (mean over GEO cohorts):\n", ext_table.round(3))

    # per-cohort external breakdown (for supplementary)
    ext_long = ext.pivot_table(index="cohort", columns="model", values="c_index")
    ext_long.to_csv(out_dir / "external_per_cohort_table.csv")
    print("\nExternal per-cohort table:\n", ext_long.round(3))

    cv_long = cv.reset_index().rename(columns={"mean": "c_index", "std": "se"})
    cv_long["split"] = "internal"
    ext_long2 = ext_sum.reset_index().rename(columns={"mean": "c_index", "std": "se"})
    ext_long2["split"] = "external"
    summary = pd.concat([cv_long, ext_long2])[["dataset", "model", "c_index", "se", "split"]]
    summary.to_csv(out_dir / "summary_long.csv", index=False)

    present = [m for m in MAIN_MODELS if m in df["model"].unique()]
    main_tab = _paper_table(cv, ext_sum, present)
    (out_dir / "paper_main_table.md").write_text(main_tab, encoding="utf-8")
    print("\nPaper main table:\n", main_tab)

    abl_present = [m for m in ABLATION if m in df["model"].unique()]
    if len(abl_present) > 1:
        abl_tab = _paper_table(cv, ext_sum, abl_present)
        (out_dir / "paper_ablation_table.md").write_text(abl_tab, encoding="utf-8")
        print("\nPaper ablation table:\n", abl_tab)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6))
        datasets = sorted(summary["dataset"].unique())
        for model in present:
            sub = summary[summary["model"] == model]
            internal = sub[sub["split"] == "internal"].set_index("dataset")["c_index"].reindex(datasets)
            external = sub[sub["split"] == "external"].set_index("dataset")["c_index"].reindex(datasets)
            x = np.arange(len(datasets))
            ax.plot(x, internal, "o-", label=LABELS.get(model, model), lw=1.2, ms=4)
            ax.plot(x, external, "s--", lw=1.0, ms=3.5, alpha=0.65,
                    color=ax.lines[-1].get_color() if len(ax.lines) else None)
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set_xticks(np.arange(len(datasets)))
        ax.set_xticklabels(datasets, rotation=45)
        ax.set_ylabel("C-index (solid: internal CV, dashed: external GEO)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "figures" / "cindex_comparison.svg", format="svg")
        fig.savefig(out_dir / "figures" / "cindex_comparison.png", dpi=200)
        print("\nFigures saved:", out_dir / "figures" / "cindex_comparison.svg|png")
    except Exception as e:  # noqa: BLE001
        print("figure skipped:", e)


if __name__ == "__main__":
    main()

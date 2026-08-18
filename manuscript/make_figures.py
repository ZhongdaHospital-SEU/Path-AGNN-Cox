# -*- coding: utf-8 -*-
"""Generate editable SVG figures for the manuscript, numbered by first
appearance in the template (dynamic, not hardcoded to 5).

Usage:  python manuscript/make_figures.py
Outputs: results/figures/Figure<N>_<NAME>.svg (+ .png previews) and
         results/figures/figure_manifest.json
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"] = "none"  # keep text editable in SVG

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
TEMPLATE = ROOT / "manuscript" / "Path-AGNN-Cox_manuscript_template.md"
FIGDIR = ROOT / "results" / "figures"
CSV = ROOT / "results" / "benchmark_results.csv"
FIGDIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["LUAD", "LUSC", "BRCA", "COAD", "STAD", "LIHC", "KIRC", "HNSC", "BLCA", "OV", "GBM"]
CANCER_NAMES = {
    "LUAD": "LUAD", "LUSC": "LUSC", "BRCA": "BRCA", "COAD": "COAD", "STAD": "STAD",
    "LIHC": "LIHC", "KIRC": "KIRC", "HNSC": "HNSC", "BLCA": "BLCA", "OV": "OV", "GBM": "GBM"}
MAIN_MODELS = ["path_agnn_cox", "lasso_cox", "ridge_cox", "elastic_net",
               "rsf", "deepsurv", "cox_nnet", "plain_gnn"]
ABL_MODELS = ["path_agnn_cox", "path_agnn_cox_static", "path_agnn_cox_noreg", "plain_gnn"]
LABELS = {
    "path_agnn_cox": "Path-AGNN-Cox",
    "path_agnn_cox_static": "\u2212Adaptive (static)",
    "path_agnn_cox_noreg": "\u2212Regularization",
    "plain_gnn": "\u2212Pathway (plain GNN)",
    "lasso_cox": "LASSO-Cox", "ridge_cox": "Ridge-Cox", "elastic_net": "EN-Cox",
    "rsf": "RSF", "deepsurv": "DeepSurv", "cox_nnet": "Cox-nnet"}
C_FULL = "#C0392B"
C_ABL = {"path_agnn_cox": "#C0392B", "path_agnn_cox_static": "#2980B9",
         "path_agnn_cox_noreg": "#27AE60", "plain_gnn": "#7F8C8D"}

def mean_cv(df, model):
    g = df[(df["split"] == "cv") & (df["model"] == model)].groupby("dataset")["c_index"].mean()
    return g.reindex(DATASETS)
def mean_ext(df, model):
    g = df[(df["split"] == "external") & (df["model"] == model)].groupby("dataset")["c_index"].mean()
    return g.reindex(DATASETS)
def mean_auc(df, model):
    g = df[(df["split"] == "cv") & (df["model"] == model)].groupby("dataset")["auc_mean"].mean()
    return g.reindex(DATASETS)

def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.7)
    ax.set_ylim(0.35, 0.85)
    ax.tick_params(labelsize=8)
    ax.set_ylabel("C-index / AUC", fontsize=9)

def _dot_plot(ax, series_dict, xlabels):
    """series_dict: model -> pd.Series over DATASETS. Full model highlighted."""
    x = np.arange(len(DATASETS))
    for m, s in series_dict.items():
        if m == "path_agnn_cox":
            ax.plot(x, [s.get(d, np.nan) for d in DATASETS], "-o", color=C_FULL,
                    lw=1.6, ms=4, zorder=5, label=LABELS[m])
        else:
            ax.plot(x, [s.get(d, np.nan) for d in DATASETS], ".", color="#B0B0B0",
                    ms=5, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=7)
    _style_ax(ax)
    ax.legend(fontsize=7, loc="lower left", frameon=False)

# ---------------- figure generators ----------------
def fig_method():
    fig = plt.figure(figsize=(10.5, 6.2))
    # Panel A: pipeline
    ax = fig.add_axes([0.03, 0.62, 0.94, 0.32]); ax.axis("off")
    steps = ["TCGA/GEO\nExpression\n(N \u00d7 G)", "KEGG pathway\nmapping\n(57 pathways)",
             "Pathway-block\nadjacency", "Adaptive GAT\n\u00d7 L layers",
             "Pathway readout\n+ MLP", "Cox risk\nscore", "Rewiring\nstatistics"]
    xs = np.linspace(0.02, 0.94, len(steps))
    w, h = 0.115, 0.72
    for i, (x, s) in enumerate(zip(xs, steps)):
        rect = plt.Rectangle((x, 0.14), w, h, facecolor="#F4F6F7" if i != 5 else "#FDEBD0",
                             edgecolor="#34495E", lw=1.0)
        ax.add_patch(rect)
        ax.text(x + w / 2, 0.5, s, ha="center", va="center", fontsize=7.5, linespacing=1.3)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + 0.006, 0.5), xytext=(x + w - 0.004, 0.5),
                        arrowprops=dict(arrowstyle="->", color="#34495E", lw=1.0))
    ax.text(0.0, 1.0, "A", fontsize=13, fontweight="bold", va="top")
    ax.set_title("End-to-end pipeline", fontsize=9, pad=2, loc="left")
    # Panel B: block-diagonal adjacency
    axb = fig.add_axes([0.03, 0.06, 0.30, 0.45]); axb.axis("off")
    rng = np.random.default_rng(0)
    K = 6
    blocks = []
    n = 60
    sizes = [12, 9, 15, 8, 10, 6]
    pos = {}
    for k, s in enumerate(sizes):
        for j in range(s):
            blocks.append(k)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if blocks[i] == blocks[j]:
                M[i, j] = 1
    im = axb.imshow(M, cmap="Blues", aspect="auto")
    axb.set_xticks([]); axb.set_yticks([])
    axb.set_title("Block-diagonal adjacency: edges only within pathway",
                  fontsize=8, pad=2)
    axb.text(-0.12, 1.02, "B", transform=axb.transAxes, fontsize=13, fontweight="bold", va="top")
    # Panel C: attention formula
    axc = fig.add_axes([0.36, 0.06, 0.30, 0.45]); axc.axis("off")
    axc.text(0.0, 0.92, "C  Sample-adaptive attention", fontsize=9, fontweight="bold")
    axc.text(0.0, 0.55, r"$\alpha_{ij}^{(l)} = \mathrm{softmax}_j\left( \mathrm{LeakyReLU}\left( \mathbf{a}^\top [\mathbf{W} h_i \| \mathbf{W} h_j] \right) \cdot (1 + \tanh(\beta)\, m_s) \right)$",
             fontsize=8.5, va="center", wrap=True)
    axc.text(0.0, 0.18, "edges only for genes sharing a KEGG pathway;  \u03b2 learnable;  m_s = per-sample malignancy score",
             fontsize=7.5, color="#555555", wrap=True)
    # Panel D: loss
    axd = fig.add_axes([0.69, 0.06, 0.28, 0.45]); axd.axis("off")
    axd.text(0.0, 0.92, "D  Survival objective with dual regularization", fontsize=9, fontweight="bold")
    axd.text(0.0, 0.5, r"$\mathcal{L} = -\!\!\!\sum_{i:\delta_i=1}\! \left[ \hat{y}_i - \log\!\!\sum_{j \in R(t_i)} e^{\hat{y}_j} \right] + \lambda_1 \sum_k \| \alpha_k \|_1 + \lambda_2 \cdot \mathrm{Consistency}$",
             fontsize=8.5, va="center", wrap=True)
    axd.text(0.0, 0.12, "Cox partial likelihood  +  intra-pathway sparsity  +  dropout-view consistency",
             fontsize=7.5, color="#555555", wrap=True)
    fig.savefig(FIGDIR / "Figure1_method.svg", format="svg")
    fig.savefig(FIGDIR / "Figure1_method.png", dpi=300)
    plt.close(fig)
    return "Figure1_method.svg", ["A", "B", "C", "D"]

def fig_benchmark(df):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    cv = {m: mean_cv(df, m) for m in MAIN_MODELS}
    ext = {m: mean_ext(df, m) for m in MAIN_MODELS}
    auc = {m: mean_auc(df, m) for m in MAIN_MODELS}
    _dot_plot(axes[0], cv, DATASETS); axes[0].set_title("A  Internal 5-fold CV C-index", fontsize=9)
    _dot_plot(axes[1], ext, DATASETS); axes[1].set_title("B  External C-index (mean over GEO cohorts)", fontsize=9)
    _dot_plot(axes[2], auc, DATASETS); axes[2].set_title("C  Mean time-dependent AUC (internal CV)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "Figure2_benchmark.svg", format="svg")
    fig.savefig(FIGDIR / "Figure2_benchmark.png", dpi=300)
    plt.close(fig)
    return "Figure2_benchmark.svg", ["A", "B", "C"]

def fig_ablation(df):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    cv = {m: mean_cv(df, m) for m in ABL_MODELS}
    ext = {m: mean_ext(df, m) for m in ABL_MODELS}
    x = np.arange(len(DATASETS))
    for m in ABL_MODELS:
        axes[0].plot(x, [cv[m].get(d, np.nan) for d in DATASETS], "-o", ms=3.5, lw=1.3,
                     color=C_ABL[m], label=LABELS[m])
        axes[1].plot(x, [ext[m].get(d, np.nan) for d in DATASETS], "-o", ms=3.5, lw=1.3,
                     color=C_ABL[m], label=LABELS[m])
    for ax in [axes[0], axes[1]]:
        ax.set_xticks(x); ax.set_xticklabels(DATASETS, rotation=40, ha="right", fontsize=7)
        _style_ax(ax)
    axes[0].set_title("A  Internal C-index per variant", fontsize=9)
    axes[1].set_title("B  External C-index per variant", fontsize=9)
    axes[0].legend(fontsize=6.5, frameon=False, loc="lower left")
    # C: mean drop
    full = np.array([cv["path_agnn_cox"].get(d, np.nan) for d in DATASETS])
    names, drops, sds = [], [], []
    for m in ABL_MODELS[1:]:
        v = np.array([cv[m].get(d, np.nan) for d in DATASETS])
        d = full - v
        names.append(LABELS[m]); drops.append(np.nanmean(d)); sds.append(np.nanstd(d))
    y = np.arange(len(names))
    axes[2].barh(y, drops, xerr=sds, color=[C_ABL[m] for m in ABL_MODELS[1:]], alpha=0.85,
                 error_kw=dict(lw=1, capsize=2))
    axes[2].set_yticks(y); axes[2].set_yticklabels(names, fontsize=7.5)
    axes[2].axvline(0, color="black", lw=0.8)
    axes[2].set_xlabel("Mean internal drop (full \u2212 variant)", fontsize=8)
    axes[2].set_title("C  Ablation contribution", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "Figure3_ablation.svg", format="svg")
    fig.savefig(FIGDIR / "Figure3_ablation.png", dpi=300)
    plt.close(fig)
    return "Figure3_ablation.svg", ["A", "B", "C"]

def fig_external(df):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ext = df[(df["split"] == "external")]
    colors = plt.cm.tab20(np.linspace(0, 1, len(DATASETS)))
    cmap = dict(zip(DATASETS, colors))
    for ds in DATASETS:
        s = ext[(ext["dataset"] == ds)]
        for cohort in sorted(s["cohort"].unique()):
            sub = s[s["cohort"] == cohort]
            fr = sub[sub["model"] == "path_agnn_cox"]["c_index"]
            if not len(fr):
                continue
            fc = float(fr.iloc[0])
            br = sub[sub["model"].isin(MAIN_MODELS[1:])]
            bc = float(br["c_index"].max()) if len(br) else np.nan
            ax.plot(bc, fc, "o", color=cmap[ds], ms=6, alpha=0.85,
                    markeredgecolor="white", markeredgewidth=0.5)
            if abs(fc - bc) > 0.12:
                ax.annotate(cohort, (bc, fc), textcoords="offset points", xytext=(4, 4),
                            fontsize=6.5, color="#444444")
    ax.plot([0.3, 0.9], [0.3, 0.9], ls="--", color="grey", lw=1)
    ax.axhline(0.5, color="grey", lw=0.7, ls=":")
    ax.axvline(0.5, color="grey", lw=0.7, ls=":")
    ax.set_xlabel("Best baseline C-index (per cohort)", fontsize=9)
    ax.set_ylabel("Path-AGNN-Cox C-index", fontsize=9)
    ax.set_xlim(0.3, 0.9); ax.set_ylim(0.3, 0.9)
    ax.set_title("External validation: Path-AGNN-Cox vs best baseline (25 GEO cohorts)",
                 fontsize=9)
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap[ds], ms=6, label=ds)
               for ds in DATASETS]
    ax.legend(handles=handles, fontsize=6.5, ncol=3, frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "Figure4_external.svg", format="svg")
    fig.savefig(FIGDIR / "Figure4_external.png", dpi=300)
    plt.close(fig)
    return "Figure4_external.svg", ["A"]

def fig_rewiring(df=None):
    rw = ROOT / "results" / "rewiring" / "LUAD"
    panels = []
    fig = plt.figure(figsize=(11.5, 6.5))
    # A: top pathways by |z|
    ax = fig.add_axes([0.07, 0.56, 0.40, 0.36])
    pw = rw / "pathway_test.csv"
    if pw.exists():
        t = pd.read_csv(pw).head(10).iloc[::-1]
        colors = ["#C0392B" if d > 0 else "#2980B9" for d in t["d"]]
        ax.barh(np.arange(len(t)), t["z"].abs(), color=colors, alpha=0.85)
        ax.set_yticks(np.arange(len(t))); ax.set_yticklabels(t["pathway"], fontsize=6.8)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlabel("|standardized z|", fontsize=8)
        ax.set_title("A  Top rewired pathways (LUAD)", fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        panels.append("A")
    else:
        ax.text(0.5, 0.5, "pending", ha="center", va="center"); ax.axis("off")
    # B: mean edge weight hi vs lo for top-5
    ax = fig.add_axes([0.52, 0.56, 0.44, 0.36])
    if pw.exists():
        t5 = pd.read_csv(pw).head(5).iloc[::-1]
        x = np.arange(len(t5)); w = 0.36
        ax.bar(x - w / 2, t5["mean_hi"], w, label="High risk", color="#C0392B", alpha=0.8)
        ax.bar(x + w / 2, t5["mean_lo"], w, label="Low risk", color="#2980B9", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(t5["pathway"], rotation=30, ha="right", fontsize=6.8)
        ax.set_ylabel("Mean edge weight", fontsize=8)
        ax.set_title("B  Pathway attention strength by risk stratum", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        panels.append("B")
    else:
        ax.text(0.5, 0.5, "pending", ha="center", va="center"); ax.axis("off")
    # C: label-permutation null vs observed
    ax = fig.add_axes([0.07, 0.08, 0.28, 0.36])
    perm_files = [("LUAD", rw / "permutation_test.csv"),
                  ("BRCA", ROOT / "results" / "rewiring" / "BRCA" / "permutation_test.csv")]
    if all(p.exists() for _, p in perm_files):
        obs, null_mean, null_max, perm_p = [], [], [], []
        for _, pf in perm_files:
            q = pd.read_csv(pf).iloc[0]
            obs.append(int(q["observed_sig"]))
            null_mean.append(float(q["null_mean_sig"]))
            null_max.append(int(q["null_max_sig"]))
            perm_p.append(float(q["perm_p"]))
        grp = np.array([0.0, 1.7]); wd = 0.55
        ax.bar(grp - wd / 2, obs, wd, color=["#C0392B", "#8E44AD"], alpha=0.9, label="Observed")
        ax.bar(grp + wd / 2, null_mean, wd, color="#BDC3C7", alpha=0.95,
               label="Permutation null (mean)")
        ax.scatter(grp + wd / 2, null_max, marker="_", s=150, color="#2C3E50", zorder=5,
                   label="Null maximum")
        for i, (x, o, nm, pv) in enumerate(zip(grp, obs, null_mean, perm_p)):
            ax.annotate(str(o), (x - wd / 2, o), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7)
            ax.annotate(f"{nm:.2f}", (x + wd / 2, nm), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7)
            pv_txt = "P<0.001" if pv < 0.001 else "P=%.3f" % pv
            ax.text(x, max(o, null_max[i]) * 1.06 + 1.5, pv_txt, ha="center",
                    fontsize=7, color="#555555")
        ax.set_xticks(grp); ax.set_xticklabels(["LUAD", "BRCA"])
        ax.set_ylabel("Significant pathways (q<0.05)", fontsize=8)
        ax.set_ylim(0, max(obs) * 1.28)
        ax.set_title("C  Rewiring vs label-permutation null (1,000 perms)", fontsize=9)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        panels.append("C")
    else:
        ax.text(0.5, 0.5, "pending", ha="center", va="center"); ax.axis("off")
    # D: clinical correlation
    ax = fig.add_axes([0.40, 0.08, 0.28, 0.36])
    cc = rw / "clinical_corr.csv"
    if cc.exists() and cc.stat().st_size > 1:
        c = pd.read_csv(cc)
        if len(c):
            c = c.reindex(c["rho"].abs().sort_values(ascending=False).index)
            cols = c["clinical"].tolist()
            rhos = c["rho"].tolist()
            ax.barh(np.arange(len(c)), rhos, color="#27AE60", alpha=0.85)
            ax.set_yticks(np.arange(len(c))); ax.set_yticklabels(cols, fontsize=8)
            ax.axvline(0, color="black", lw=0.6)
            for i, (r, p) in enumerate(zip(c["rho"], c["p"])):
                ax.text(r + 0.01, i, "P<0.001" if p < 0.001 else "P=%.3f" % p,
                        fontsize=6.5, va="center")
            ax.set_xlabel("Spearman \u03c1 (rewiring magnitude)", fontsize=8)
            ax.set_title("D  Clinical correlation", fontsize=9)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            panels.append("D")
    else:
        ax.text(0.5, 0.5, "pending", ha="center", va="center"); ax.axis("off")
    # E: BRCA top pathways (replication)
    ax = fig.add_axes([0.73, 0.08, 0.24, 0.36])
    br = ROOT / "results" / "rewiring" / "BRCA" / "pathway_test.csv"
    if br.exists():
        t = pd.read_csv(br).head(8).iloc[::-1]
        ax.barh(np.arange(len(t)), t["z"].abs(), color="#8E44AD", alpha=0.8)
        ax.set_yticks(np.arange(len(t))); ax.set_yticklabels(t["pathway"], fontsize=6)
        ax.set_title("E  Top rewired pathways (BRCA)", fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        panels.append("E")
    else:
        ax.text(0.5, 0.5, "pending", ha="center", va="center"); ax.axis("off")
    fig.savefig(FIGDIR / "Figure5_rewiring.svg", format="svg")
    fig.savefig(FIGDIR / "Figure5_rewiring.png", dpi=300)
    plt.close(fig)
    return "Figure5_rewiring.svg", panels


def fig_immune_drug(df=None):
    """Figure: immune infiltration (A) + BRCA predicted drug sensitivity (B, C)."""
    imm = ROOT / "results" / "immune"
    panels = []
    fig = plt.figure(figsize=(11.5, 6.8))
    # A: immune features, signed -log10 P (high vs low rewiring) for LUAD & BRCA
    ax = fig.add_axes([0.30, 0.10, 0.66, 0.80])
    feats = []
    luad_rows, brca_rows = None, None
    lp = imm / "LUAD" / "immune_stats.csv"
    bp = imm / "BRCA" / "immune_stats.csv"
    if lp.exists() and bp.exists():
        luad = pd.read_csv(lp)
        brca = pd.read_csv(bp)
        luad = luad.set_index("feature")
        brca = brca.set_index("feature")
        feats = list(luad.index)
        y = np.arange(len(feats))
        lx = np.where(luad["high_median"] > luad["low_median"], -np.log10(luad["wilcox_P"]), np.log10(luad["wilcox_P"]))
        bx = np.where(brca["high_median"] > brca["low_median"], -np.log10(brca["wilcox_P"]), np.log10(brca["wilcox_P"]))
        ax.scatter(bx, y + 0.18, marker="s", s=26, color="#2980B9", alpha=0.9, label="BRCA", zorder=3)
        ax.scatter(lx, y - 0.18, marker="o", s=30, color="#C0392B", alpha=0.9, label="LUAD", zorder=3)
        ax.axvline(0, color="black", lw=0.6)
        ax.axvline(np.log10(0.05), color="grey", lw=0.7, ls="--")
        ax.axvline(-np.log10(0.05), color="grey", lw=0.7, ls="--")
        ax.set_yticks(y)
        lbl = [_feat_label_plot(f) for f in feats]
        ax.set_yticklabels(lbl, fontsize=6.5)
        ax.set_xlabel("Signed -log10 P (high vs low rewiring; + = higher in high-rewiring)", fontsize=8)
        ax.set_title("A  Immune features vs rewiring magnitude", fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_xlim(-2.1, 2.1)
        panels.append("A")
    else:
        ax.text(0.5, 0.5, "pending", ha="center", va="center"); ax.axis("off")
    # B: BRCA drug IC50 (log2) high vs low for top-8 nominal drugs
    ax = fig.add_axes([0.05, 0.56, 0.22, 0.34])
    dp = imm / "BRCA" / "drug_pred_ic50_BRCA.csv"
    sp = imm / "BRCA" / "drug_stats_BRCA.csv"
    if dp.exists() and sp.exists():
        pred = pd.read_csv(dp, index_col=0)
        stats = pd.read_csv(sp).sort_values("wilcox_P").head(8)
        mag = pd.read_csv(ROOT / "results" / "rewiring" / "BRCA" / "rewiring_magnitude.csv")
        mag["sample_id"] = mag["sample_id"].str[:12]
        common = [c for c in pred.columns if c in set(mag["sample_id"])]
        pred = pred[common]
        med = mag.set_index("sample_id").loc[common, "rewiring_magnitude"].median()
        grp = np.where(mag.set_index("sample_id").loc[common, "rewiring_magnitude"] >= med, "High", "Low")
        x = np.arange(len(stats))
        w = 0.38
        hi_data, lo_data = [], []
        for _, r in stats.iterrows():
            base = str(r["drug"])
            cols = [c for c in pred.index if c.startswith(base + "_")]
            if cols:
                vals = pred.loc[cols].mean(axis=0)
            else:
                vals = pred.loc[base]
            hi_data.append(np.log2(vals[grp == "High"] + 1e-6))
            lo_data.append(np.log2(vals[grp == "Low"] + 1e-6))
        bp_hi = ax.boxplot(hi_data, positions=x - w / 2, widths=w * 0.9, patch_artist=True,
                           showfliers=False, medianprops=dict(color="white"))
        bp_lo = ax.boxplot(lo_data, positions=x + w / 2, widths=w * 0.9, patch_artist=True,
                           showfliers=False, medianprops=dict(color="white"))
        for b in bp_hi["boxes"]: b.set_facecolor("#C0392B"); b.set_alpha(0.8)
        for b in bp_lo["boxes"]: b.set_facecolor("#2980B9"); b.set_alpha(0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(stats["drug"], rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("log2 predicted IC50", fontsize=8)
        ax.set_title("B  BRCA: top-8 drugs by nominal P", fontsize=9)
        ax.legend([bp_hi["boxes"][0], bp_lo["boxes"][0]], ["High rewiring", "Low rewiring"],
                  fontsize=6.5, frameon=False)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        panels.append("B")
    else:
        ax.text(0.5, 0.5, "pending", ha="center", va="center"); ax.axis("off")
    # C: BRCA Spearman rho across 17 drugs
    ax = fig.add_axes([0.05, 0.08, 0.22, 0.36])
    if sp.exists():
        stats = pd.read_csv(sp).sort_values("spearman_P")
        stats = stats.reindex(stats["spearman_rho"].abs().sort_values(ascending=False).index)
        yy = np.arange(len(stats))
        colors = ["#C0392B" if q < 0.05 else "#95A5A6" for q in stats["spearman_q"]]
        ax.barh(yy, stats["spearman_rho"], color=colors, alpha=0.85)
        ax.set_yticks(yy)
        ax.set_yticklabels(stats["drug"], fontsize=6)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlabel("Spearman ρ (rewiring magnitude)", fontsize=8)
        ax.set_title("C  BRCA: IC50 ρ, FDR q<0.05 in red", fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        panels.append("C")
    else:
        ax.text(0.5, 0.5, "pending", ha="center", va="center"); ax.axis("off")
    fig.savefig(FIGDIR / "Figure7_immunedrug.svg", format="svg")
    fig.savefig(FIGDIR / "Figure7_immunedrug.png", dpi=300)
    plt.close(fig)
    return "Figure7_immunedrug.svg", panels

def fig_imvigor(df=None):
    """Figure: IMvigor210 anti-PD-L1 cohort (exploratory; panels A-C)."""
    rdir = ROOT / "results" / "rewiring" / "IMvigor210"
    panels = []
    fig = plt.figure(figsize=(12.0, 3.9))
    # A: rewiring magnitude responders (CR/PR) vs non-responders (SD/PD)
    ax = fig.add_axes([0.06, 0.16, 0.27, 0.72])
    has_a = False
    if (rdir / "alpha.npy").exists() and (rdir / "risk_scores.csv").exists():
        alpha = np.load(rdir / "alpha.npy")
        rew = np.abs(alpha - alpha.mean(axis=0)).sum(axis=1)
        risk = pd.read_csv(rdir / "risk_scores.csv")
        clin = pd.read_csv(ROOT / "data" / "processed" / "IMvigor210" / "clinical.csv")
        dfm = pd.DataFrame({"sample_id": risk["sample_id"], "rew": rew}).merge(clin, on="sample_id", how="left")
        sub = dfm.dropna(subset=["response"])
        hi = sub.loc[sub["response"] == "CR/PR", "rew"]
        lo = sub.loc[sub["response"] == "SD/PD", "rew"]
        bp = ax.boxplot([hi, lo], widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color="white"),
                        tick_labels=["CR/PR (n=%d)" % len(hi), "SD/PD (n=%d)" % len(lo)])
        for b, c in zip(bp["boxes"], ["#C0392B", "#2980B9"]):
            b.set_facecolor(c); b.set_alpha(0.75)
        rng = np.random.default_rng(0)
        for i, x in enumerate([hi, lo]):
            y = x.to_numpy(float)
            ax.scatter(np.full(len(y), i + 1) + rng.uniform(-0.12, 0.12, len(y)), y,
                       s=7, color="black", alpha=0.25, linewidths=0)
        from scipy.stats import mannwhitneyu
        _, pv = mannwhitneyu(hi, lo)
        pstr = "P<0.001" if pv < 0.001 else "P=%.3f" % pv
        ax.set_title("A  Rewiring magnitude by response", fontsize=9)
        ax.set_ylabel("Rewiring magnitude", fontsize=8)
        ax.text(0.03, 0.95, "Wilcoxon %s" % pstr, transform=ax.transAxes, fontsize=8, va="top")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        panels.append("A"); has_a = True
    if not has_a:
        ax.text(0.5, 0.5, "pending", ha="center", va="center"); ax.axis("off")
    # B: OS by high vs low rewiring (median split)
    ax = fig.add_axes([0.42, 0.16, 0.27, 0.72])
    has_b = False
    if has_a:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
        sub2 = dfm.dropna(subset=["os_months"])
        med = sub2["rew"].median()
        sub2 = sub2.assign(grp=(sub2["rew"] >= med).astype(int))
        kmf = KaplanMeierFitter()
        for g, c, lab in [(1, "#C0392B", "High rewiring"), (0, "#2980B9", "Low rewiring")]:
            d = sub2[sub2["grp"] == g]
            kmf.fit(d["os_months"], event_observed=d["os_event"], label=lab)
            kmf.plot_survival_function(ax=ax, ci_show=False, color=c, lw=1.6)
        lr = logrank_test(sub2.loc[sub2["grp"] == 1, "os_months"], sub2.loc[sub2["grp"] == 0, "os_months"],
                          event_observed_A=sub2.loc[sub2["grp"] == 1, "os_event"],
                          event_observed_B=sub2.loc[sub2["grp"] == 0, "os_event"])
        pstr = "P<0.001" if lr.p_value < 0.001 else "P=%.3f" % lr.p_value
        ax.set_title("B  OS: high vs low rewiring", fontsize=9)
        ax.set_xlabel("Months", fontsize=8); ax.set_ylabel("Overall survival", fontsize=8)
        ax.text(0.03, 0.95, "logrank %s" % pstr, transform=ax.transAxes, fontsize=8, va="top")
        ax.legend(fontsize=7, frameon=False)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        panels.append("B"); has_b = True
    if not has_b:
        ax.text(0.5, 0.5, "pending", ha="center", va="center"); ax.axis("off")
    # C: Ki-67 expression vs rewiring magnitude
    ax = fig.add_axes([0.78, 0.16, 0.20, 0.72])
    has_c = False
    trp = ROOT / "data" / "processed" / "IMvigor210" / "train.csv"
    if has_a and trp.exists():
        tr = pd.read_csv(trp, usecols=["sample_id", "MKI67"])
        km = dfm.merge(tr, on="sample_id", how="left").dropna(subset=["MKI67", "rew"])
        from scipy.stats import spearmanr
        rho, pv = spearmanr(km["rew"], km["MKI67"])
        pstr = "P<0.001" if pv < 0.001 else "P=%.3f" % pv
        ax.scatter(km["MKI67"], km["rew"], s=8, color="#7F8C8D", alpha=0.55, linewidths=0)
        ax.set_title("C  Ki-67 vs rewiring", fontsize=9)
        ax.set_xlabel("Ki-67 (vst)", fontsize=8); ax.set_ylabel("Rewiring magnitude", fontsize=8)
        ax.text(0.03, 0.95, "rho=%.2f\n%s" % (rho, pstr), transform=ax.transAxes, fontsize=8, va="top")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        panels.append("C"); has_c = True
    if not has_c:
        ax.text(0.5, 0.5, "pending", ha="center", va="center"); ax.axis("off")
    fig.savefig(FIGDIR / "Figure6_imvigor.svg", format="svg")
    fig.savefig(FIGDIR / "Figure6_imvigor.png", dpi=300)
    plt.close(fig)
    return "Figure6_imvigor.svg", panels


def _feat_label_plot(f):
    f = str(f)
    if f.startswith("ssGSEA_"):
        return f.replace("ssGSEA_", "ssGSEA-").replace("_", "-")
    return f

GENERATORS = {
    "METHOD": fig_method, "BENCHMARK": fig_benchmark, "ABLATION": fig_ablation,
    "EXTERNAL": fig_external, "REWIRING": fig_rewiring, "IMV": fig_imvigor,
    "IMMUNEDRUG": fig_immune_drug,
}

def main():
    src = TEMPLATE.read_text(encoding="utf-8-sig")
    pat = re.compile(r"\{\{(?:FREF|FDEF):([A-Z]+)(?:\|[A-Z])?\}\}")
    order = []
    for m in pat.finditer(src):
        if m.group(1) not in order:
            order.append(m.group(1))
    df = pd.read_csv(CSV)
    manifest = {}
    for i, name in enumerate(order, start=1):
        if name in GENERATORS:
            fname, panels = GENERATORS[name](df) if name != "METHOD" else GENERATORS[name]()
            manifest[f"Figure{i}"] = {"token": name, "file": fname, "panels": panels}
            print(f"Figure {i}: {name} -> {fname} ({', '.join(panels)})")
        else:
            print(f"WARNING: no generator for figure token '{name}' (Figure {i})")
    (FIGDIR / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    print("manifest:", FIGDIR / "figure_manifest.json")

if __name__ == "__main__":
    main()

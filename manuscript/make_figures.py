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


# ---------------- panel splitting ----------------
def _crop_svg_text(svg_text, x0, y0, w, h):
    """Return a standalone SVG whose viewBox shows only the cropped region.
    Content is wrapped in a translated group, so text remains editable."""
    m = re.search(r"<svg[^>]*>", svg_text)
    if not m:
        return None
    inner = svg_text[m.end(): svg_text.rfind("</svg>")]
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        'width="%.2f" height="%.2f" viewBox="%.2f %.2f %.2f %.2f">\n'
        '<g transform="translate(%.2f %.2f)">%s</g>\n</svg>\n'
        % (w, h, x0, y0, w, h, -x0, -y0, inner)
    )

def _save_panels(fig, stem, pairs, svg_path, png_path, only=None):
    """Crop the composite SVG into one standalone SVG per panel.

    pairs: list of (Axes, panel letter) in figure order.
    only: optional iterable of panel letters to emit, e.g. panels with data.
    Returns {panel letter: relative panel SVG filename}."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    svg_text = svg_path.read_text(encoding="utf-8")
    vm = re.search(r'viewBox="([\d.eE+-]+) ([\d.eE+-]+) ([\d.eE+-]+) ([\d.eE+-]+)"', svg_text)
    if not vm:
        return {}
    svg_w, svg_h = float(vm.group(3)), float(vm.group(4))
    dw, dh = fig.canvas.get_width_height()
    sx, sy = svg_w / dw, svg_h / dh
    try:
        from PIL import Image
        png = Image.open(png_path) if png_path.exists() else None
        pxs = (png.width / dw, png.height / dh) if png is not None else None
    except Exception:
        png = None; pxs = None
    pad = 5.0
    out = {}
    for ax, lab in pairs:
        if only is not None and lab not in only:
            continue
        bb = ax.get_tightbbox(renderer)
        if bb is None:
            continue
        x0 = max(0.0, (bb.x0 - pad) * sx)
        y0 = max(0.0, (bb.y0 - pad) * sy)
        x1 = min(svg_w, (bb.x1 + pad) * sx)
        y1 = min(svg_h, (bb.y1 + pad) * sy)
        rel = "%s%s.svg" % (stem, lab)
        crop = _crop_svg_text(svg_text, x0, y0, x1 - x0, y1 - y0)
        if crop is not None:
            (FIGDIR / rel).write_text(crop, encoding="utf-8")
        if png is not None:
            bx0 = int(max(0.0, (bb.x0 - pad) * pxs[0])); by0 = int(max(0.0, (bb.y0 - pad) * pxs[1]))
            bx1 = int(min(png.width, (bb.x1 + pad) * pxs[0])); by1 = int(min(png.height, (bb.y1 + pad) * pxs[1]))
            png.crop((bx0, by0, bx1, by1)).save(FIGDIR / ("%s%s.png" % (stem, lab)), dpi=(300, 300))
        out[lab] = rel
    return out

# ---------------- figure generators ----------------
def fig_method(panel_stem="Figure1"):
    fig = plt.figure(figsize=(10.5, 6.2))
    pairs = []
    # Panel A: pipeline
    ax = fig.add_axes([0.03, 0.62, 0.94, 0.32]); ax.axis("off")
    pairs.append((ax, "A"))
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
    pairs.append((axb, "B"))
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
    pairs.append((axc, "C"))
    axc.text(0.0, 0.92, "C  Sample-adaptive attention", fontsize=9, fontweight="bold")
    axc.text(0.0, 0.55, r"$\alpha_{ij}^{(l)} = \mathrm{softmax}_j\left( \mathrm{LeakyReLU}\left( \mathbf{a}^\top [\mathbf{W} h_i \| \mathbf{W} h_j] \right) \cdot (1 + \tanh(\beta)\, m_s) \right)$",
             fontsize=8.5, va="center", wrap=True)
    axc.text(0.0, 0.18, "edges only for genes sharing a KEGG pathway;  \u03b2 learnable;  m_s = per-sample malignancy score",
             fontsize=7.5, color="#555555", wrap=True)
    # Panel D: loss
    axd = fig.add_axes([0.69, 0.06, 0.28, 0.45]); axd.axis("off")
    pairs.append((axd, "D"))
    axd.text(0.0, 0.92, "D  Survival objective with dual regularization", fontsize=9, fontweight="bold")
    axd.text(0.0, 0.5, r"$\mathcal{L} = -\!\!\!\sum_{i:\delta_i=1}\! \left[ \hat{y}_i - \log\!\!\sum_{j \in R(t_i)} e^{\hat{y}_j} \right] + \lambda_1 \sum_k \| \alpha_k \|_1 + \lambda_2 \cdot \mathrm{Consistency}$",
             fontsize=8.5, va="center", wrap=True)
    axd.text(0.0, 0.12, "Cox partial likelihood  +  intra-pathway sparsity  +  dropout-view consistency",
             fontsize=7.5, color="#555555", wrap=True)
    fig.savefig(FIGDIR / "Figure1_method.svg", format="svg")
    fig.savefig(FIGDIR / "Figure1_method.png", dpi=300)
    panel_files = _save_panels(fig, panel_stem, pairs,
                               FIGDIR / "Figure1_method.svg", FIGDIR / "Figure1_method.png",
                               only=["A", "B", "C", "D"])
    plt.close(fig)
    return "Figure1_method.svg", ["A", "B", "C", "D"], panel_files

def fig_benchmark(df, panel_stem="Figure2"):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    pairs = [(axes[0], "A"), (axes[1], "B"), (axes[2], "C")]
    cv = {m: mean_cv(df, m) for m in MAIN_MODELS}
    ext = {m: mean_ext(df, m) for m in MAIN_MODELS}
    auc = {m: mean_auc(df, m) for m in MAIN_MODELS}
    _dot_plot(axes[0], cv, DATASETS); axes[0].set_title("A  Internal 5-fold CV C-index", fontsize=9)
    _dot_plot(axes[1], ext, DATASETS); axes[1].set_title("B  External C-index (mean over GEO cohorts)", fontsize=9)
    _dot_plot(axes[2], auc, DATASETS); axes[2].set_title("C  Mean time-dependent AUC (internal CV)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "Figure2_benchmark.svg", format="svg")
    fig.savefig(FIGDIR / "Figure2_benchmark.png", dpi=300)
    panel_files = _save_panels(fig, panel_stem, pairs,
                               FIGDIR / "Figure2_benchmark.svg", FIGDIR / "Figure2_benchmark.png",
                               only=["A", "B", "C"])
    plt.close(fig)
    return "Figure2_benchmark.svg", ["A", "B", "C"], panel_files

def fig_ablation(df, panel_stem="Figure3"):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    pairs = [(axes[0], "A"), (axes[1], "B"), (axes[2], "C")]
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
    panel_files = _save_panels(fig, panel_stem, pairs,
                               FIGDIR / "Figure3_ablation.svg", FIGDIR / "Figure3_ablation.png",
                               only=["A", "B", "C"])
    plt.close(fig)
    return "Figure3_ablation.svg", ["A", "B", "C"], panel_files

def fig_external(df, panel_stem="Figure4"):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    pairs = [(ax, "A")]
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
    panel_files = _save_panels(fig, panel_stem, pairs,
                               FIGDIR / "Figure4_external.svg", FIGDIR / "Figure4_external.png",
                               only=["A"])
    plt.close(fig)
    return "Figure4_external.svg", ["A"], panel_files

def fig_rewiring(df=None, panel_stem="Figure5"):
    """Figure 5: between-stratum effect sizes with permutation-calibrated
    significance (A/B), cohort-level label-permutation null (C), clinical
    correlation (D), matched random-set controls (E)."""
    panels = []
    fig = plt.figure(figsize=(11.5, 6.5))
    pairs = []
    # A/B: Cohen's d forest plots (all pathways)
    for k, (ds, pos, title) in enumerate([
            ("LUAD", [0.07, 0.56, 0.44, 0.36], "A  Between-stratum effect sizes (LUAD)"),
            ("BRCA", [0.55, 0.56, 0.42, 0.36], "B  Between-stratum effect sizes (BRCA)")]):
        ef = ROOT / "results" / "rewiring" / ds / "pathway_effects.csv"
        ax = fig.add_axes(pos)
        pairs.append((ax, "A" if k == 0 else "B"))
        if ef.exists():
            d = pd.read_csv(ef).sort_values("cohen_d")
            sig = (d["perm_q"] < 0.05).to_numpy()
            y = np.arange(len(d))
            ax.errorbar(d["cohen_d"], y,
                        xerr=[d["cohen_d"] - d["d_ci_lo"], d["d_ci_hi"] - d["cohen_d"]],
                        fmt="none", ecolor="#95A5A6", elinewidth=0.6, zorder=1)
            cols = np.where(sig, "#C0392B", "#7F8C8D")
            ax.scatter(d["cohen_d"], y, s=8, c=cols, zorder=2)
            ax.axvline(0, color="black", lw=0.6)
            ax.set_yticks(y)
            ax.set_yticklabels([p if len(p) <= 24 else p[:23] + "..." for p in d["pathway"]],
                               fontsize=5.2)
            ax.set_xlabel("Cohen's d (high vs. low risk)", fontsize=8)
            ax.set_title(title, fontsize=9)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.text(0.98, 0.02, "%d significant after permutation FDR" % int(sig.sum()),
                    transform=ax.transAxes, ha="right", fontsize=6.5, color="#C0392B")
            panels.append("A" if k == 0 else "B")
        else:
            ax.text(0.5, 0.5, "pending", ha="center", va="center"); ax.axis("off")
    # C: label-permutation null vs observed
    ax = fig.add_axes([0.07, 0.08, 0.28, 0.36])
    pairs.append((ax, "C"))
    perm_files = [("LUAD", ROOT / "results" / "rewiring" / "LUAD" / "permutation_test.csv"),
                  ("BRCA", ROOT / "results" / "rewiring" / "BRCA" / "permutation_test.csv"),
                  ("KIRC", ROOT / "results" / "rewiring" / "KIRC" / "permutation_test.csv")]
    if all(p.exists() for _, p in perm_files):
        obs, null_mean, null_max, perm_p = [], [], [], []
        for _, pf in perm_files:
            q = pd.read_csv(pf).iloc[0]
            obs.append(int(q["observed_sig"]))
            null_mean.append(float(q["null_mean_sig"]))
            null_max.append(int(q["null_max_sig"]))
            perm_p.append(float(q["perm_p"]))
        grp = np.array([0.0, 1.7, 3.4]); wd = 0.55
        ax.bar(grp - wd / 2, obs, wd, color=["#C0392B", "#8E44AD", "#16A085"], alpha=0.9, label="Observed")
        ax.bar(grp + wd / 2, null_mean, wd, color="#BDC3C7", alpha=0.95,
               label="Permutation null (mean)")
        ax.scatter(grp + wd / 2, null_max, marker="_", s=150, color="#2C3E50", zorder=5,
                   label="Null maximum")
        for i, (x, o, nm, pv) in enumerate(zip(grp, obs, null_mean, perm_p)):
            ax.annotate(str(o), (x - wd / 2, o), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7)
            ax.annotate("%.2f" % nm, (x + wd / 2, nm), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7)
            pv_txt = "P<0.001" if pv < 0.001 else "P=%.3f" % pv
            ax.text(x, max(o, null_max[i]) * 1.06 + 1.5, pv_txt, ha="center",
                    fontsize=7, color="#555555")
        ax.set_xticks(grp); ax.set_xticklabels(["LUAD", "BRCA", "KIRC"])
        ax.set_ylabel("Significant pathways (q<0.05)", fontsize=8)
        ax.set_ylim(0, max(obs) * 1.28)
        ax.set_title("C  Rewiring vs label-permutation null", fontsize=9)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        panels.append("C")
    else:
        ax.text(0.5, 0.5, "pending", ha="center", va="center"); ax.axis("off")
    # D: clinical correlation
    ax = fig.add_axes([0.40, 0.08, 0.28, 0.36])
    pairs.append((ax, "D"))
    cc = ROOT / "results" / "rewiring" / "LUAD" / "clinical_corr.csv"
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
            ax.set_xlabel("Spearman rho (rewiring magnitude)", fontsize=8)
            ax.set_title("D  Clinical correlation", fontsize=9)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            panels.append("D")
    else:
        ax.text(0.5, 0.5, "pending", ha="center", va="center"); ax.axis("off")
    # E: matched random-set controls (percentiles of real pathways)
    ax = fig.add_axes([0.73, 0.08, 0.24, 0.36])
    pairs.append((ax, "E"))
    data, positions, labels = [], [], []
    for i, ds in enumerate(["LUAD", "BRCA", "KIRC"]):
        ef = ROOT / "results" / "rewiring" / ds / "pathway_effects.csv"
        if not ef.exists():
            continue
        d = pd.read_csv(ef)
        for j, col in enumerate(["null_pct", "block_null_pct"]):
            v = d[col].dropna().to_numpy()
            data.append(v)
            positions.append(i * 1.5 + j * 0.32)
            labels.append("%s %s" % (ds, "edge" if j == 0 else "density"))
    if data:
        bp = ax.boxplot(data, positions=positions, widths=0.26, patch_artist=True,
                        showfliers=False, manage_ticks=False)
        for patch in bp["boxes"]:
            patch.set_facecolor("#AED6F1"); patch.set_alpha(0.8)
        ax.axhline(0.5, color="#2C3E50", ls="--", lw=0.9)
        ax.axhline(0.95, color="#E67E22", ls=":", lw=0.9)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=6, rotation=25, ha="right")
        ax.set_ylabel("P(null effect >= real effect)", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title("E  Matched random-set controls", fontsize=9)
        ax.text(0.02, 0.97, "uniform 0.50", transform=ax.transAxes, fontsize=6, color="#2C3E50")
        ax.text(0.02, 0.89, "structural-null median (LUAD/BRCA): 1.00", transform=ax.transAxes,
                fontsize=5.6, color="#E67E22")
        ax.text(0.02, 0.905, "95th pct", transform=ax.transAxes, fontsize=6, color="#E67E22")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        panels.append("E")
    else:
        ax.text(0.5, 0.5, "pending", ha="center", va="center"); ax.axis("off")
    fig.savefig(FIGDIR / "Figure5_rewiring.svg", format="svg")
    fig.savefig(FIGDIR / "Figure5_rewiring.png", dpi=300)
    panel_files = _save_panels(fig, panel_stem, pairs,
                               FIGDIR / "Figure5_rewiring.svg", FIGDIR / "Figure5_rewiring.png",
                               only=panels)
    plt.close(fig)
    return "Figure5_rewiring.svg", panels, panel_files

def fig_immune_drug(df=None, panel_stem="Figure8"):
    """Figure: immune infiltration (A) + BRCA predicted drug sensitivity (B, C)."""
    imm = ROOT / "results" / "immune"
    panels = []
    fig = plt.figure(figsize=(11.5, 6.8))
    pairs = []
    # A: immune features, signed -log10 P (high vs low rewiring) for LUAD & BRCA
    ax = fig.add_axes([0.30, 0.10, 0.66, 0.80])
    pairs.append((ax, "A"))
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
    # B: BRCA median predicted IC50 (log2) high vs low for top-8 nominal drugs
    ax = fig.add_axes([0.05, 0.56, 0.22, 0.34])
    pairs.append((ax, "B"))
    sp = imm / "BRCA" / "drug_stats_BRCA.csv"
    if sp.exists():
        stats = pd.read_csv(sp).sort_values("wilcox_P").head(8)
        x = np.arange(len(stats))
        w = 0.36
        hi = np.log2(stats["high_median_IC50"].to_numpy() + 1e-6)
        lo = np.log2(stats["low_median_IC50"].to_numpy() + 1e-6)
        ax.bar(x - w / 2, hi, width=w, color="#C0392B", alpha=0.85, label="High rewiring")
        ax.bar(x + w / 2, lo, width=w, color="#2980B9", alpha=0.85, label="Low rewiring")
        ax.set_xticks(x)
        ax.set_xticklabels(stats["drug"], rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("log2 median predicted IC50", fontsize=8)
        ax.set_title("B  BRCA: top-8 drugs by nominal P", fontsize=9)
        ax.legend(fontsize=6.5, frameon=False)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        panels.append("B")
    else:
        ax.text(0.5, 0.5, "pending", ha="center", va="center"); ax.axis("off")
    # C: BRCA Spearman rho across 17 drugs
    ax = fig.add_axes([0.05, 0.08, 0.22, 0.36])
    pairs.append((ax, "C"))
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
    panel_files = _save_panels(fig, panel_stem, pairs,
                               FIGDIR / "Figure7_immunedrug.svg", FIGDIR / "Figure7_immunedrug.png",
                               only=panels)
    plt.close(fig)
    return "Figure7_immunedrug.svg", panels, panel_files

def fig_imvigor(df=None, panel_stem="Figure6"):
    """Figure: IMvigor210 anti-PD-L1 cohort (exploratory) + template transfer (panels A-D)."""
    rdir = ROOT / "results" / "rewiring" / "IMvigor210"
    panels = []
    fig = plt.figure(figsize=(15.5, 3.9))
    pairs = []
    # A: rewiring magnitude responders (CR/PR) vs non-responders (SD/PD)
    ax = fig.add_axes([0.05, 0.16, 0.20, 0.72])
    pairs.append((ax, "A"))
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
    ax = fig.add_axes([0.28, 0.16, 0.20, 0.72])
    pairs.append((ax, "B"))
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
    ax = fig.add_axes([0.51, 0.16, 0.17, 0.72])
    pairs.append((ax, "C"))
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
    # D: template-transfer OS validation forest plot (BRCA/KIRC/LUAD GEO cohorts)
    ax = fig.add_axes([0.71, 0.16, 0.26, 0.72])
    pairs.append((ax, "D"))
    te = ROOT / "results" / "template_external" / "template_external_os.csv"
    tm = ROOT / "results" / "template_external" / "template_external_meta.csv"
    has_d = False
    if te.exists() and tm.exists():
        osd = pd.read_csv(te)
        meta = pd.read_csv(tm).set_index("dataset")
        ds_colors = {"BRCA": "#2980B9", "KIRC": "#16A085", "LUAD": "#C0392B"}
        rows, labs = [], []
        for ds in ["BRCA", "KIRC", "LUAD"]:
            subd = osd[osd["dataset"] == ds]
            if not len(subd):
                continue
            for r in subd.itertuples():
                rows.append((float(r.hr_high_vs_low), float(r.hr_lo), float(r.hr_hi), ds_colors[ds], False))
                labs.append((r.cohort, ds))
            m = meta.loc[ds]
            rows.append((float(m["hr_fe"]), float(m["hr_lo_fe"]), float(m["hr_hi_fe"]), ds_colors[ds], True))
            labs.append(("Fixed effect", ds))
        yv = np.arange(len(rows))[::-1]
        for (hr, lo, hi, col, is_fe), yy in zip(rows, yv):
            if is_fe:
                ax.plot([lo, hi], [yy, yy], color=col, lw=1.8, zorder=2)
                ax.scatter([hr], [yy], marker="D", s=30, color=col, zorder=3,
                           edgecolor="white", linewidths=0.4)
            else:
                ax.plot([lo, hi], [yy, yy], color="#95A5A6", lw=1.2, zorder=1)
                ax.scatter([hr], [yy], s=24, color=col, zorder=3,
                           edgecolor="white", linewidths=0.4)
        # cancer-type separators
        ymax = len(rows)
        for yy in (ymax - 4.5, ymax - 6.5):
            ax.axhline(yy, color="#D5D8DC", lw=0.8, zorder=0)
        ax.axvline(1.0, color="grey", lw=0.8, ls="--", zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(0.35, 4.6)
        ax.set_xticks([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
        ax.set_xticklabels(["0.5", "0.75", "1.0", "1.5", "2.0", "3.0"], fontsize=6.5)
        ax.set_yticks(yv)
        ax.set_yticklabels([l[0] for l in labs], fontsize=6)
        for tk, (_, ds) in zip(ax.get_yticklabels(), labs):
            tk.set_color(ds_colors[ds])
        ax.set_xlabel("HR (high vs. low template score)", fontsize=8)
        ax.set_title("D  Template transfer: OS association", fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#2980B9", ms=5,
                                  label="Cohort HR"),
                           Line2D([], [], marker="D", ls="", color="#2980B9", ms=5,
                                  label="Fixed effect")],
                  fontsize=6, frameon=False, loc="lower right")
        panels.append("D"); has_d = True
    if not has_d:
        ax.text(0.5, 0.5, "pending", ha="center", va="center"); ax.axis("off")
    fig.savefig(FIGDIR / "Figure6_imvigor.svg", format="svg")
    fig.savefig(FIGDIR / "Figure6_imvigor.png", dpi=300)
    panel_files = _save_panels(fig, panel_stem, pairs,
                               FIGDIR / "Figure6_imvigor.svg", FIGDIR / "Figure6_imvigor.png",
                               only=panels)
    plt.close(fig)
    return "Figure6_imvigor.svg", panels, panel_files


def _feat_label_plot(f):
    f = str(f)
    if f.startswith("ssGSEA_"):
        return f.replace("ssGSEA_", "ssGSEA-").replace("_", "-")
    return f


def fig_dca(df=None, panel_stem="Figure7"):
    """Figure: IPCW decision-curve analysis at 3 years (panels A-C)."""
    dca = pd.read_csv(ROOT / "results" / "rewiring" / "dca_results.csv")
    d3 = dca[dca["horizon"] == "3y"]
    cohorts = ["LUAD", "BRCA", "KIRC"]
    styles = {
        "clinical": dict(color="#7F8C8D", ls="--", lw=1.4, label="Clinical (age+stage)"),
        "clinical+risk": dict(color="#C0392B", ls="-", lw=1.8, label="Clinical + risk score"),
        "risk": dict(color="#2980B9", ls="-", lw=1.4, label="Risk score"),
        "treat_all": dict(color="#555555", ls=":", lw=1.2, label="Treat all"),
    }
    fig = plt.figure(figsize=(11.4, 3.4))
    pairs = []
    panels = []
    for i, co in enumerate(cohorts):
        ax = fig.add_axes([0.065 + 0.315 * i, 0.17, 0.27, 0.72])
        pairs.append((ax, chr(65 + i)))
        for model, st in styles.items():
            sub = d3[(d3["dataset"] == co) & (d3["model"] == model)].sort_values("threshold")
            if model == "treat_all":
                pass
            ax.plot(sub["threshold"], sub["net_benefit"], **st)
        ax.axhline(0.0, color="#999999", lw=0.9, ls="-", label="Treat none")
        ax.set_title(f"{chr(65 + i)}  {co}", fontsize=10)
        ax.set_xlabel("Threshold probability", fontsize=8)
        ax.set_ylabel("Net benefit (3-year)", fontsize=8)
        ax.set_ylim(-0.02, 0.15)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        panels.append(chr(65 + i))
    fig.legend(loc="lower center", ncol=5, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.savefig(FIGDIR / "FigureDCA_dca.svg", format="svg")
    fig.savefig(FIGDIR / "FigureDCA_dca.png", dpi=300)
    panel_files = _save_panels(fig, panel_stem, pairs,
                               FIGDIR / "FigureDCA_dca.svg", FIGDIR / "FigureDCA_dca.png",
                               only=panels)
    plt.close(fig)
    return "FigureDCA_dca.svg", panels, panel_files


GENERATORS = {
    "METHOD": fig_method, "BENCHMARK": fig_benchmark, "ABLATION": fig_ablation,
    "EXTERNAL": fig_external, "REWIRING": fig_rewiring, "IMV": fig_imvigor,
    "IMMUNEDRUG": fig_immune_drug, "DCA": fig_dca,
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
            fname, panels, panel_files = (GENERATORS[name](df, f"Figure{i}")
                                           if name != "METHOD" else GENERATORS[name](f"Figure{i}"))
            manifest[f"Figure{i}"] = {"token": name, "file": fname, "panels": panels,
                                      "panel_files": panel_files}
            print(f"Figure {i}: {name} -> {fname} ({', '.join(panels)}) "
                  f"panels: {', '.join(sorted(panel_files))}")
        else:
            print(f"WARNING: no generator for figure token '{name}' (Figure {i})")
    (FIGDIR / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    print("manifest:", FIGDIR / "figure_manifest.json")

if __name__ == "__main__":
    main()

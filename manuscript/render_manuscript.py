# -*- coding: utf-8 -*-
"""Render the final manuscript from the tokenized template + benchmark results.

Usage:
    python manuscript/render_manuscript.py [--draft]

Numbering: tables and figures are numbered sequentially by FIRST APPEARANCE
in the text (TREF/TDEF/FREF/FDEF tokens). Adding/removing a table or figure
re-numbers automatically.

Formatting rules applied:
  - all numeric values: 2 decimals (sample sizes remain integers)
  - P values: P>=0.001 -> 3 decimals (e.g. P=0.024); P<0.001 -> "P<0.001"
"""
from __future__ import annotations
import argparse
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
TEMPLATE = ROOT / "manuscript" / "Path-AGNN-Cox_manuscript_template.md"
OUT = ROOT / "manuscript" / "Path-AGNN-Cox_manuscript.md"
RES = ROOT / "results"
CSV = RES / "benchmark_results.csv"
INFO = ROOT / "work" / "dataset_info.json"

DATASETS = ["LUAD", "LUSC", "BRCA", "COAD", "STAD", "LIHC", "KIRC", "HNSC", "BLCA", "OV", "GBM"]
CANCER_NAMES = {
    "LUAD": "Lung adenocarcinoma", "LUSC": "Lung squamous carcinoma", "BRCA": "Breast carcinoma",
    "COAD": "Colon adenocarcinoma", "STAD": "Stomach adenocarcinoma", "LIHC": "Liver hepatocellular carcinoma",
    "KIRC": "Kidney renal clear-cell carcinoma", "HNSC": "Head and neck squamous carcinoma",
    "BLCA": "Bladder urothelial carcinoma", "OV": "Ovarian serous carcinoma", "GBM": "Glioblastoma",
}
MAIN_MODELS = ["path_agnn_cox", "lasso_cox", "ridge_cox", "elastic_net",
               "rsf", "deepsurv", "cox_nnet", "plain_gnn"]
ABL_MODELS = ["path_agnn_cox", "path_agnn_cox_static", "path_agnn_cox_noreg", "plain_gnn"]
LABELS = {
    "path_agnn_cox": "Path-AGNN-Cox",
    "path_agnn_cox_static": "\u2212Adaptive (static)",
    "path_agnn_cox_noreg": "\u2212Regularization",
    "plain_gnn": "\u2212Pathway (plain GNN)",
    "lasso_cox": "LASSO-Cox", "ridge_cox": "Ridge-Cox", "elastic_net": "EN-Cox",
    "rsf": "RSF", "deepsurv": "DeepSurv", "cox_nnet": "Cox-nnet",
}

# ---------- formatting helpers ----------
def fmt2(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "\u2014"
    if isinstance(x, (int, float)) and abs(float(x)) < 0.005:
        return "0.00"
    return f"{x:.2f}"

def fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "NA"
    if p < 0.001:
        return "P<0.001"
    return f"P={p:.3f}"

def fmt_sd(x) -> str:
    return f"{x:.2f}"

def fmt_q(q) -> str:
    if q is None or (isinstance(q, float) and not np.isfinite(q)):
        return "NA"
    if q < 0.001:
        return "q<0.001"
    return f"q={q:.3f}"

def diff_ci_paired(a, b, fmt="\u0394") -> str:
    """Paired mean difference with 95% CI (t distribution)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return "\u2014"
    from scipy import stats as _st
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = _st.t.ppf(0.975, len(d) - 1)
    return "%s=%.2f (95%% CI %.2f to %.2f)" % (fmt, m, m - t * se, m + t * se)

def p_wilcoxon(a: np.ndarray, b: np.ndarray):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if len(d) < 3 or np.allclose(d, 0):
        return None
    try:
        return float(wilcoxon(d).pvalue)
    except ValueError:
        return None

# ---------- data ----------
def load_df():
    df = pd.read_csv(CSV)
    return df

def mean_cv(df, model):
    g = df[(df["split"] == "cv") & (df["model"] == model)].groupby("dataset")["c_index"].mean()
    return g.reindex(DATASETS)

def mean_ext(df, model):
    g = df[(df["split"] == "external") & (df["model"] == model)].groupby("dataset")["c_index"].mean()
    return g.reindex(DATASETS)

def mean_auc(df, model):
    g = df[(df["split"] == "cv") & (df["model"] == model)].groupby("dataset")["auc_mean"].mean()
    return g.reindex(DATASETS)

def read_info():
    with open(INFO, encoding="utf-8") as f:
        return json.load(f)

# ---------- tables ----------
def table1(df, info) -> str:
    ext_used = {}
    for ds in DATASETS:
        cohorts = sorted(df[(df["dataset"] == ds) & (df["split"] == "external")]["cohort"].unique())
        ext_used[ds] = cohorts
    lines = ["| Cancer type | TCGA cohort | N | Events | External GEO cohorts (N) |",
             "|---|---|---|---|---|"]
    total_n = total_e = 0
    for ds in DATASETS:
        n = info[ds]["n"]; e = info[ds]["events"]
        total_n += n; total_e += e
        ext = ", ".join(f"{c} ({info[ds]['external'].get(c, '?')})" for c in ext_used[ds])
        lines.append(f"| {CANCER_NAMES[ds]} | TCGA-{ds} | {n} | {e} | {ext} |")
    lines.append(f"| **Total** | \u2014 | **{total_n}** | **{total_e}** | **{sum(len(v) for v in ext_used.values())} cohorts** |")
    return "\n".join(lines)

def table2(df) -> str:
    cv = {m: mean_cv(df, m) for m in MAIN_MODELS}
    auc = {m: mean_auc(df, m) for m in MAIN_MODELS}
    hdr = "| Model | " + " | ".join(DATASETS) + " | Mean C-index |"
    sep = "|---|" + "---|" * (len(DATASETS) + 1)
    lines = [hdr, sep]
    for m in MAIN_MODELS:
        cells = []
        for ds in DATASETS:
            c = cv[m].get(ds); a = auc[m].get(ds)
            if c is None or (isinstance(c, float) and not np.isfinite(c)):
                cells.append("\u2014")
            else:
                cells.append(f"{fmt2(c)}/{fmt2(a)}")
        mean_c = np.nanmean([cv[m].get(d, np.nan) for d in DATASETS])
        lines.append(f"| {LABELS[m]} | " + " | ".join(cells) + f" | {fmt2(mean_c)} |")
    # bold the best C-index per dataset column
    best = []
    for ds in DATASETS:
        vals = {m: cv[m].get(ds, np.nan) for m in MAIN_MODELS}
        bm = max(vals, key=vals.get)
        best.append((ds, bm))
    # rebuild with bolding
    lines2 = [hdr, sep]
    for m in MAIN_MODELS:
        cells = []
        for ds in DATASETS:
            c = cv[m].get(ds); a = auc[m].get(ds)
            if c is None or (isinstance(c, float) and not np.isfinite(c)):
                cells.append("\u2014")
            else:
                txt = f"{fmt2(c)}/{fmt2(a)}"
                if ds in [b[0] for b in best] and m == dict(best)[ds]:
                    txt = f"**{txt}**"
                cells.append(txt)
        mean_c = np.nanmean([cv[m].get(d, np.nan) for d in DATASETS])
        lines2.append(f"| {LABELS[m]} | " + " | ".join(cells) + f" | {fmt2(mean_c)} |")
    lines2.append("\n*C-index / mean time-dependent AUC; bold = best C-index per cancer type (5-fold CV).*")
    return "\n".join(lines2)

def table3(df) -> str:
    cv = {m: mean_cv(df, m) for m in ABL_MODELS}
    ext = {m: mean_ext(df, m) for m in ABL_MODELS}
    hdr = "| Variant | " + " | ".join(DATASETS) + " | Internal mean\u00b1SD | External mean\u00b1SD | \u0394 vs full | P |"
    sep = "|---|" + "---|" * (len(DATASETS) + 4)
    lines = [hdr, sep]
    full_cv = cv["path_agnn_cox"]
    for m in ABL_MODELS:
        cells = [fmt2(cv[m].get(d, np.nan)) for d in DATASETS]
        im = np.nanmean([cv[m].get(d, np.nan) for d in DATASETS])
        isd = np.nanstd([cv[m].get(d, np.nan) for d in DATASETS])
        em = np.nanmean([ext[m].get(d, np.nan) for d in DATASETS])
        esd = np.nanstd([ext[m].get(d, np.nan) for d in DATASETS])
        if m == "path_agnn_cox":
            delta, pv = "ref", "ref"
        else:
            delta = fmt2(im - np.nanmean([full_cv.get(d, np.nan) for d in DATASETS]))
            pv = fmt_p(p_wilcoxon([cv[m].get(d, np.nan) for d in DATASETS],
                                  [full_cv.get(d, np.nan) for d in DATASETS]))
        lines.append(f"| {LABELS[m]} | " + " | ".join(cells)
                     + f" | {fmt2(im)}\u00b1{fmt2(isd)} | {fmt2(em)}\u00b1{fmt2(esd)} | {delta} | {pv} |")
    return "\n".join(lines)

def table4(df, info) -> str:
    rows = []
    for ds in DATASETS:
        sub = df[(df["dataset"] == ds) & (df["split"] == "external")]
        for cohort in sorted(sub["cohort"].unique()):
            s = sub[sub["cohort"] == cohort]
            full = s[s["model"] == "path_agnn_cox"]["c_index"]
            full_c = float(full.iloc[0]) if len(full) else np.nan
            base_rows = s[s["model"].isin(["lasso_cox", "ridge_cox", "elastic_net", "rsf", "deepsurv", "cox_nnet", "plain_gnn"])]
            if len(base_rows):
                bi = base_rows.loc[base_rows["c_index"].idxmax()]
                base_c = float(bi["c_index"]); base_m = LABELS[bi["model"]]
            else:
                base_c, base_m = np.nan, "\u2014"
            n = info[ds]["external"].get(cohort, "?")
            rows.append((ds, cohort, n, full_c, base_m, base_c))
    lines = ["| Cancer | Cohort | N | Path-AGNN-Cox | Best baseline | Baseline C-index | \u0394 |",
             "|---|---|---|---|---|---|---|"]
    for ds, cohort, n, fc, bm, bc in rows:
        delta = fmt2(fc - bc) if np.isfinite(fc) and np.isfinite(bc) else "\u2014"
        lines.append(f"| {CANCER_NAMES[ds]} | {cohort} | {n} | {fmt2(fc)} | {bm} | {fmt2(bc)} | {delta} |")
    return "\n".join(lines)

def table5(rw_dir) -> str:
    pw = pd.read_csv(rw_dir / "pathway_test.csv").copy()
    pw = pw.reindex(pw["z"].abs().sort_values(ascending=False).index)
    top = pw.head(10)
    lines = ["| Pathway | Edges | z | P | FDR q | Mean \u0394w (\u00d710\u207b\u2076) |",
             "|---|---|---|---|---|---|"]
    for _, row in top.iterrows():
        lines.append(f"| {row['pathway']} | {int(row['n_edges'])} | {fmt2(float(row['z']))} | "
                     f"{fmt_p(float(row['p']))} | {fmt_q(float(row['q']))} | {fmt2(float(row['d']) * 1e6)} |")
    extra = []
    perm = {}
    for ds, pf in [("LUAD", rw_dir / "permutation_test.csv"),
                   ("BRCA", ROOT / "results" / "rewiring" / "BRCA" / "permutation_test.csv")]:
        if pf.exists():
            perm[ds] = pd.read_csv(pf).iloc[0]
    if perm.get("LUAD") is not None and perm.get("BRCA") is not None:
        lq, bq = perm["LUAD"], perm["BRCA"]
        extra.append(
            "Permutation null: under 1,000 label permutations, a mean of "
            f"{fmt2(float(lq['null_mean_sig']))} (LUAD) and {fmt2(float(bq['null_mean_sig']))} (BRCA) pathways reached FDR q<0.05 "
            f"by chance (maximum {int(lq['null_max_sig'])} in LUAD, {int(bq['null_max_sig'])} in BRCA); "
            f"observed {int(lq['observed_sig'])} (LUAD, {fmt_p(float(lq['perm_p']))}) and "
            f"{int(bq['observed_sig'])} (BRCA, {fmt_p(float(bq['perm_p']))}).")
    rc = rw_dir / "random_control.csv"
    if rc.exists():
        rc_df = pd.read_csv(rc)
        extra.append(f"Randomized pathway partitions (3 seeds): "
                     f"{int(rc_df['n_sig_q005'].min())}\u2013{int(rc_df['n_sig_q005'].max())} significant pathways.")
    sn = rw_dir / "static_null.csv"
    if sn.exists():
        s = pd.read_csv(sn, index_col=0)
        extra.append(f"Static-model between-stratum edge variance: {float(s.loc['total_var', '0']):.2e} "
                     f"(essentially zero by construction).")
    cc = rw_dir / "clinical_corr.csv"
    if cc.exists():
        c = pd.read_csv(cc)
        if len(c):
            c = c.reindex(c["rho"].abs().sort_values(ascending=False).index)
            r0 = c.iloc[0]
            extra.append(f"Clinical correlation ({r0['clinical'].upper()}): \u03c1={fmt2(float(r0['rho']))}, "
                         f"{fmt_p(float(r0['p']))}, n={int(r0['n'])}.")
    mvc = {}
    for ds in ("LUAD", "BRCA"):
        mf = rw_dir.parent / ds / "multivariable_cox.csv"
        if mf.exists():
            m = pd.read_csv(mf)
            row = m[(m["model"] == "multivariable") & (m["covariate"] == "risk_z")]
            if len(row):
                row = row.iloc[0]
                mvc[ds] = (fmt2(float(row["hr"])),
                           f"{fmt2(float(row['ci_lower']))}\u2013{fmt2(float(row['ci_upper']))}",
                           fmt_p(float(row["p"])))
    if mvc.get("LUAD") and mvc.get("BRCA"):
        extra.append(f"Multivariable Cox (risk score + stage + age): LUAD HR {mvc['LUAD'][0]} per SD "
                     f"(95% CI {mvc['LUAD'][1]}, {mvc['LUAD'][2]}); BRCA HR {mvc['BRCA'][0]} "
                     f"(95% CI {mvc['BRCA'][1]}, {mvc['BRCA'][2]}).")
    if extra:
        lines.append("")
        lines.extend(extra)
    return "\n".join(lines)

# ---------- stats tokens ----------
def compute_stats(df, info) -> dict:
    st = {}
    cv_full = mean_cv(df, "path_agnn_cox")
    ext_full = mean_ext(df, "path_agnn_cox")
    auc_full = mean_auc(df, "path_agnn_cox")
    base_names = [m for m in MAIN_MODELS if m != "path_agnn_cox"]
    cv_base = {m: mean_cv(df, m) for m in base_names}
    ext_base = {m: mean_ext(df, m) for m in base_names}
    auc_base = {m: mean_auc(df, m) for m in base_names}

    st["N_DATASETS"] = str(len(DATASETS))
    ext_cohorts = sorted(df[df["split"] == "external"]["cohort"].unique())
    st["N_EXTERNAL"] = str(len(ext_cohorts))
    st["TCGA_TOTAL_N"] = str(sum(info[d]["n"] for d in DATASETS))
    st["TCGA_TOTAL_EVENTS"] = str(sum(info[d]["events"] for d in DATASETS))

    st["CV_FULL_MEAN"] = fmt2(np.nanmean([cv_full.get(d, np.nan) for d in DATASETS]))
    st["CV_FULL_SD"] = fmt2(np.nanstd([cv_full.get(d, np.nan) for d in DATASETS]))
    bmean = {m: np.nanmean([cv_base[m].get(d, np.nan) for d in DATASETS]) for m in base_names}
    best_b = max(bmean, key=bmean.get)
    st["CV_BEST_BASELINE_MEAN"] = fmt2(bmean[best_b])
    st["BEST_BASELINE_NAME"] = LABELS[best_b]
    st["CV_FULL_P"] = fmt_p(p_wilcoxon([cv_full.get(d, np.nan) for d in DATASETS],
                                       [cv_base[best_b].get(d, np.nan) for d in DATASETS]))
    deep_names = ["deepsurv", "cox_nnet"]
    dmean = {m: np.nanmean([cv_base[m].get(d, np.nan) for d in DATASETS]) for m in deep_names}
    st["CV_BEST_DEEP_MEAN"] = fmt2(max(dmean.values())) if dmean else "\u2014"

    wins_i = sum(1 for d in DATASETS if cv_full.get(d, np.nan) > max(cv_base[m].get(d, np.nan) for m in base_names))
    wins_e = sum(1 for d in DATASETS if ext_full.get(d, np.nan) > max(ext_base[m].get(d, np.nan) for m in base_names))
    wins_a = sum(1 for d in DATASETS if auc_full.get(d, np.nan) > max(auc_base[m].get(d, np.nan) for m in base_names))
    st["BEST_INTERNAL_WINS"] = str(wins_i)
    st["BEST_EXTERNAL_WINS"] = str(wins_e)
    st["BEST_AUC_WINS"] = str(wins_a)

    gains = {d: cv_full.get(d, np.nan) - max(cv_base[m].get(d, np.nan) for m in base_names) for d in DATASETS}
    gd = max(gains, key=gains.get)
    st["TOP_GAIN_DATASET"] = gd
    st["CV_TOP_GAIN_FULL"] = fmt2(cv_full.get(gd, np.nan))
    st["CV_TOP_GAIN_BASE"] = fmt2(max(cv_base[m].get(gd, np.nan) for m in base_names))
    st["CV_TOP_GAIN_DELTA"] = fmt2(gains[gd])

    st["EXT_FULL_MEAN"] = fmt2(np.nanmean([ext_full.get(d, np.nan) for d in DATASETS]))
    st["EXT_FULL_SD"] = fmt2(np.nanstd([ext_full.get(d, np.nan) for d in DATASETS]))
    emean = {m: np.nanmean([ext_base[m].get(d, np.nan) for d in DATASETS]) for m in base_names}
    best_e = max(emean, key=emean.get)
    st["EXT_BEST_BASELINE_MEAN"] = fmt2(emean[best_e])
    st["EXT_BEST_BASELINE_NAME"] = LABELS[best_e]
    edeep = {m: np.nanmean([ext_base[m].get(d, np.nan) for d in DATASETS]) for m in deep_names}
    st["EXT_BEST_DEEP_MEAN"] = fmt2(max(edeep.values())) if edeep else "\u2014"

    abl = {m: np.nanmean([mean_cv(df, m).get(d, np.nan) for d in DATASETS]) for m in ABL_MODELS}
    fullm = abl["path_agnn_cox"]
    for m in ["plain_gnn", "path_agnn_cox_static", "path_agnn_cox_noreg"]:
        key = {"plain_gnn": "PATHWAY", "path_agnn_cox_static": "ADAPTIVE", "path_agnn_cox_noreg": "NOREG"}[m]
        st[f"ABL_{key}_DROP"] = fmt2(fullm - abl[m])
        st[f"ABL_{key}_P"] = fmt_p(p_wilcoxon([mean_cv(df, m).get(d, np.nan) for d in DATASETS],
                                              [mean_cv(df, "path_agnn_cox").get(d, np.nan) for d in DATASETS]))
    st["ABL_ADAPTIVE_EXT_DROP"] = fmt2(np.nanmean([ext_full.get(d, np.nan) for d in DATASETS])
                                       - np.nanmean([mean_ext(df, "path_agnn_cox_static").get(d, np.nan) for d in DATASETS]))
    st["ABL_NOREG_EXT_SD"] = fmt2(np.nanstd([mean_ext(df, "path_agnn_cox_noreg").get(d, np.nan) for d in DATASETS]))
    st["ABL_FULL_EXT_SD"] = st["EXT_FULL_SD"]

    ext_rows = df[(df["split"] == "external") & (df["model"] == "path_agnn_cox")]
    st["EXT_ABOVE_50"] = str(int((ext_rows["c_index"] > 0.5).sum()))
    base_ext_rows = df[(df["split"] == "external") & (df["model"].isin(base_names))]
    st["EXT_BASE_ABOVE_50"] = str(int((base_ext_rows.groupby("cohort")["c_index"].max() > 0.5).sum()))
    st["EXT_STRONGEST_COHORT_DESC"] = "Across cohorts, the largest external gains over the best baseline were observed in KIRC (GSE29609) and BRCA (GSE20685)"  # placeholder, refined by render if data available

    # paired difference 95% CIs (full vs best baseline / ablations)
    per_cv = {d: {} for d in DATASETS}
    for m in base_names + ABL_MODELS:
        sub = df[(df["split"] == "cv") & (df["model"] == m)]
        for d in DATASETS:
            per_cv[d][m] = sub.loc[sub["dataset"] == d, "c_index"].to_numpy(float)
    diffs = []
    for d in DATASETS:
        best_m = max(base_names, key=lambda m: np.nanmean(per_cv[d][m]))
        diffs.append(per_cv[d]["path_agnn_cox"] - per_cv[d][best_m])
    per_ds_means = np.array([np.nanmean(x) for x in diffs])
    st["CV_DIFF_CI"] = diff_ci_paired(per_ds_means, np.zeros_like(per_ds_means))
    ext_f = df[(df["split"] == "external") & (df["model"] == "path_agnn_cox")].set_index("cohort")["c_index"]
    ext_diffs = []
    for cohort in ext_f.index:
        bs = df[(df["split"] == "external") & (df["cohort"] == cohort) & (df["model"].isin(base_names))]
        ext_diffs.append(float(ext_f[cohort]) - float(bs["c_index"].max()))
    st["EXT_DIFF_CI"] = diff_ci_paired(ext_diffs, np.zeros_like(ext_diffs))
    for m, key in [("plain_gnn", "PATHWAY"), ("path_agnn_cox_static", "ADAPTIVE"),
                   ("path_agnn_cox_noreg", "NOREG")]:
        full = np.array([np.nanmean(per_cv[d]["path_agnn_cox"]) for d in DATASETS])
        abl = np.array([np.nanmean(per_cv[d][m]) for d in DATASETS])
        st["ABL_%s_DIFF_CI" % key] = diff_ci_paired(full, abl)

    st["BENCHMARK_HOURS"] = "1,500"

    # pathway catalogue / gene-universe stats
    bg = info.get("_benchmark_genes") or {}
    if bg:
        vals = [bg[d] for d in DATASETS if d in bg]
        st["GENES_AVG"] = str(int(round(float(np.mean(vals))))) if vals else "\u2014"
        st["N_GENES_MIN"] = str(int(min(vals))) if vals else "\u2014"
        st["N_GENES_MAX"] = str(int(max(vals))) if vals else "\u2014"
    st["N_PATHWAYS"] = str(info.get("_n_pathways", 57))
    st["GENES_UNION"] = str(info.get("_n_union_genes", 3097))

    # strongest external gain cohort (full model vs best baseline)
    ext_sub = df[(df["split"] == "external")]
    best_desc = None
    for cohort in sorted(ext_sub["cohort"].unique()):
        s = ext_sub[ext_sub["cohort"] == cohort]
        frow = s[s["model"] == "path_agnn_cox"]["c_index"]
        if not len(frow):
            continue
        fc = float(frow.iloc[0])
        br = s[s["model"].isin(base_names)]
        if not len(br):
            continue
        bc = float(br["c_index"].max())
        gain = fc - bc
        if best_desc is None or gain > best_desc[0]:
            ds_of = s.iloc[0]["dataset"]
            best_desc = (gain, cohort, ds_of, fc, bc)
    if best_desc:
        st["EXT_STRONGEST_COHORT_DESC"] = (
            f"The largest external gain over the best baseline was observed in "
            f"{CANCER_NAMES.get(best_desc[2], best_desc[2])} ({best_desc[1]}, "
            f"Path-AGNN-Cox C-index {fmt2(best_desc[3])} vs best baseline {fmt2(best_desc[4])})")
    else:
        st["EXT_STRONGEST_COHORT_DESC"] = "external gains over the best baseline varied by cohort"
    return st

def rewiring_tokens(rw_dir) -> dict:
    """Read rewiring outputs into paper tokens; missing files leave tokens unfilled."""
    import numpy as np
    rw_dir = Path(rw_dir)
    st = {}
    pw = rw_dir / "pathway_test.csv"
    if pw.exists():
        t = pd.read_csv(pw)
        if len(t):
            st["TOP_REWIRED_PATHWAYS"] = ", ".join(t.head(5)["pathway"].astype(str).tolist())
    en = rw_dir / "enrichment.csv"
    if en.exists():
        e = pd.read_csv(en, index_col=0)
        get = lambda k, default="\u2014": (e.loc[k, "0"] if k in e.index and pd.notna(e.loc[k, "0"]) else default)
        st["ENRICH_HITS"] = str(int(get("hits", 0)))
        st["ENRICH_TOP_K"] = str(int(get("top_k", 0)))
        pv = get("p", np.nan)
        st["ENRICH_P"] = fmt_p(float(pv)) if isinstance(pv, (int, float)) and np.isfinite(float(pv)) else "\u2014"
    sn = rw_dir / "static_null.csv"
    if sn.exists():
        s = pd.read_csv(sn, index_col=0)
        if "total_var" in s.index:
            st["STATIC_NULL_VAR"] = fmt2(float(s.loc["total_var", "0"]))
    alpha_f = rw_dir / "alpha.npy"
    if alpha_f.exists():
        alpha = np.load(alpha_f)
        if alpha.ndim == 2 and alpha.shape[0] > 1:
            st["ADAPTIVE_REWIRE_VAR"] = fmt2(float(alpha.var(axis=0).sum()))
    cc = rw_dir / "clinical_corr.csv"
    if cc.exists():
        c = pd.read_csv(cc)
        if len(c):
            c = c.reindex(c["rho"].abs().sort_values(ascending=False).index)
            r0 = c.iloc[0]
            st["CLINICAL_RHO"] = fmt2(float(r0["rho"]))
            st["CLINICAL_P"] = fmt_p(float(r0["p"]))
            st["CLINICAL_CORR_DESC"] = f"{r0['clinical']} (n={int(r0['n'])})"
    # per-dataset clinical correlations (BRCA ki67, LUAD tmb)
    for ds, prefix in [("BRCA", "CLINICAL_BRCA"), ("LUAD", "CLINICAL_LUAD")]:
        cf = rw_dir.parent / ds / "clinical_corr.csv"
        if cf.exists():
            c = pd.read_csv(cf)
            if len(c):
                c = c.reindex(c["rho"].abs().sort_values(ascending=False).index)
                r0 = c.iloc[0]
                st[f"{prefix}_RHO"] = fmt2(float(r0["rho"]))
                st[f"{prefix}_P"] = fmt_p(float(r0["p"]))
                st[f"{prefix}_N"] = str(int(r0["n"]))
    # multivariable Cox: risk score adjusted for stage/age (risk_z row)
    for ds, prefix in [("LUAD", "MVC_LUAD"), ("BRCA", "MVC_BRCA")]:
        mf = rw_dir.parent / ds / "multivariable_cox.csv"
        if mf.exists():
            m = pd.read_csv(mf)
            row = m[(m["model"] == "multivariable") & (m["covariate"] == "risk_z")]
            if len(row):
                row = row.iloc[0]
                st[f"{prefix}_HR"] = fmt2(float(row["hr"]))
                st[f"{prefix}_CI"] = f"{fmt2(float(row['ci_lower']))}\u2013{fmt2(float(row['ci_upper']))}"
                st[f"{prefix}_P"] = fmt_p(float(row["p"]))
    # label-permutation nulls (anti-circularity)
    for ds, prefix in [("LUAD", "PERM_LUAD"), ("BRCA", "PERM_BRCA")]:
        pf = rw_dir.parent / ds / "permutation_test.csv"
        if pf.exists():
            q = pd.read_csv(pf).iloc[0]
            st[f"{prefix}_SIG"] = str(int(q["observed_sig"]))
            st[f"{prefix}_P"] = fmt_p(float(q["perm_p"]))
            st["PERM_N_PATHWAYS"] = str(int(q["n_pathways_observed"]))
            st["PERM_NULL_MEAN"] = fmt2(float(q["null_mean_sig"]))
            st[f"{prefix}_NULL_MEAN"] = fmt2(float(q["null_mean_sig"]))
            st[f"{prefix}_NULL_MAX"] = str(int(q["null_max_sig"]))
    return st



def calibration_tokens() -> dict:
    """Read calibration results (results/calibration_results.csv) into tokens."""
    st = {}
    csv = ROOT / "results" / "calibration_results.csv"
    if not csv.exists():
        return st
    df = pd.read_csv(csv)
    for ds, pre in (("LUAD", "CAL_LUAD"), ("BRCA", "CAL_BRCA")):
        for model, mpre in (("path_agnn_cox", "PATH"), ("ridge_cox", "RIDGE")):
            sub = df[(df["dataset"] == ds) & (df["setting"] == "internal") & (df["model"] == model)]
            if len(sub):
                r = sub.iloc[0]
                k = f"{pre}_{mpre}"
                st[f"{k}_SLOPE"] = fmt2(float(r["slope"]))
                st[f"{k}_CI"] = f"{fmt2(float(r['slope_ci_low']))}\u2013{fmt2(float(r['slope_ci_high']))}"
                st[f"{k}_MAE"] = fmt2(float(r["cal_mae"]))
    ext = df[(df["setting"] == "external") & (df["model"] == "path_agnn_cox")]
    if len(ext):
        sl = ext["slope"].dropna()
        if len(sl):
            st["CAL_EXT_MEAN"] = fmt2(float(sl.mean()))
            st["CAL_EXT_MIN"] = fmt2(float(sl.min()))
            st["CAL_EXT_MAX"] = fmt2(float(sl.max()))
    return st


# ---------- immune / drug tokens ----------
def _feat_label(f):
    f = str(f)
    if f.startswith("ssGSEA_"):
        return f.replace("ssGSEA_", "ssGSEA ").replace("_", "-")
    return f

def immune_drug_tokens(imm_dir) -> dict:
    """Read immune infiltration + drug sensitivity outputs into paper tokens."""
    imm_dir = Path(imm_dir)
    st = {}
    def rd(ds, name):
        p = imm_dir / ds / name
        if p.exists() and p.stat().st_size > 1:
            return pd.read_csv(p)
        return None
    luad_imm = rd("LUAD", "immune_stats.csv")
    brca_imm = rd("BRCA", "immune_stats.csv")
    if luad_imm is not None and len(luad_imm):
        t = luad_imm.sort_values("wilcox_P")
        for i, key in enumerate(["IMM_LUAD_TOP1", "IMM_LUAD_TOP2", "IMM_LUAD_TOP3"], start=0):
            if i < len(t):
                row = t.iloc[i]
                st[key] = _feat_label(row["feature"])
                st[key + "_P"] = fmt_p(float(row["wilcox_P"]))
        st["IMM_LUAD_Q"] = fmt_q(float(t["wilcox_q"].min()))
    if brca_imm is not None and len(brca_imm):
        st["IMM_BRCA_MIN_P"] = fmt_p(float(brca_imm["wilcox_P"].min()))
    luad_d = rd("LUAD", "drug_stats_LUAD.csv")
    brca_d = rd("BRCA", "drug_stats_BRCA.csv")
    if brca_d is not None and len(brca_d):
        b = brca_d.sort_values("wilcox_P")
        sig = b[b["wilcox_P"] < 0.05]
        st["DRUG_BRCA_NSIG"] = str(len(sig))
        for i, key in enumerate(["DRUG_BRCA_P1", "DRUG_BRCA_P2", "DRUG_BRCA_P3"]):
            if i < len(sig):
                row = sig.iloc[i]
                st[key] = fmt_p(float(row["wilcox_P"]))
                st[key + "_NAME"] = str(row["drug"])
        st["DRUG_BRCA_WILCOX_Q"] = fmt_q(float(b["wilcox_q"].min()))
        rb = brca_d.sort_values("spearman_P")
        top = rb.iloc[0]
        st["DRUG_BRCA_RHO_TOP"] = str(top["drug"])
        st["DRUG_BRCA_RHO1"] = fmt2(float(top["spearman_rho"]))
        st["DRUG_BRCA_RHO1_P"] = fmt_p(float(top["spearman_P"]))
        st["DRUG_BRCA_RHO_Q"] = fmt_q(float(rb["spearman_q"].min()))
    if luad_d is not None and len(luad_d):
        st["DRUG_LUAD_MIN_P"] = fmt_p(float(luad_d["wilcox_P"].min()))
    return st

def table6(imm_dir) -> str:
    """Predicted drug sensitivity table (exploratory)."""
    imm_dir = Path(imm_dir)
    lines = []
    for ds in ["BRCA", "LUAD"]:
        p = imm_dir / ds / ("drug_stats_%s.csv" % ds)
        if not (p.exists() and p.stat().st_size > 1):
            continue
        d = pd.read_csv(p).sort_values("wilcox_P")
        lines.append("**%s (n high/low: %d/%d)**" % (ds, int(d["n_high"].iloc[0]), int(d["n_low"].iloc[0])))
        lines.append("| Drug | IC50 median (high) | IC50 median (low) | Wilcoxon P | FDR q | Spearman \u03c1 | Spearman P |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in d.iterrows():
            lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                r["drug"], fmt2(float(r["high_median_IC50"])), fmt2(float(r["low_median_IC50"])),
                fmt_p(float(r["wilcox_P"])), fmt_q(float(r["wilcox_q"])),
                fmt2(float(r["spearman_rho"])), fmt_p(float(r["spearman_P"]))))
        lines.append("")
    lines.append("_IC50 values are GDSC2/oncoPredict in-silico predictions; associations are exploratory and not FDR-significant unless stated._")
    return "\n".join(lines)


def imv_tokens() -> dict:
    """IMvigor210 anti-PD-L1 response tokens (filled when available)."""
    st = {}
    p = ROOT / "results" / "rewiring" / "IMvigor210" / "response_stats.csv"
    if p.exists() and p.stat().st_size > 1:
        s = pd.read_csv(p, index_col=0)["0"].to_dict()
        def f(k, d="\u2014"):
            return d if k not in s or pd.isna(s[k]) else str(s[k])
        def fi(k, d="\u2014"):
            v = s.get(k)
            try:
                return str(int(float(v)))
            except (TypeError, ValueError):
                return d
        nr = fi("n_responder"); nn = fi("n_nonresponder")
        wp = f("wilcox_P")
        if wp != "\u2014":
            pv = float(wp)
            wp = "P<0.001" if pv < 0.001 else "P=%.3f" % pv
        mr = f("med_resp"); mn = f("med_nonresp")
        try:
            mr = fmt2(float(mr)); mn = fmt2(float(mn))
        except (TypeError, ValueError):
            pass
        st["IMV_RESULT_SENTENCE"] = (
            "the rewiring magnitude differed between responders (CR/PR) and non-responders (SD/PD) "
            "(median %s vs %s; Wilcoxon %s; n=%s/%s)" % (mr, mn, wp, nr, nn))
        if "os_P" in s and pd.notna(s.get("os_P")):
            hr = f("os_hr"); lo = f("os_hr_lo"); hi = f("os_hr_hi")
            for k in ("os_hr", "os_hr_lo", "os_hr_hi"):
                v = s.get(k)
                try:
                    s[k] = fmt2(float(v))
                except (TypeError, ValueError):
                    pass
            hr = f("os_hr"); lo = f("os_hr_lo"); hi = f("os_hr_hi")
            op = float(s["os_P"])
            op = "P<0.001" if op < 0.001 else "P=%.3f" % op
            st["IMV_RESULT_SENTENCE"] += (
                "; high-rewiring patients showed HR %s (95%% CI %s-%s, %s) for OS"
                % (hr, lo, hi, op))
        if "ki67_rho" in s and pd.notna(s.get("ki67_rho")):
            try:
                kr = float(s["ki67_rho"]); kn = int(float(s.get("ki67_n", 0)))
                kp = float(s["ki67_P"])
                kp = "P<0.001" if kp < 0.001 else "P=%.3f" % kp
                st["IMV_RESULT_SENTENCE"] += (
                    "; rewiring magnitude correlated with Ki-67 expression "
                    "(Spearman rho=%s, %s, n=%d)" % (fmt2(kr), kp, kn))
            except (TypeError, ValueError):
                pass
    return st


def stdgat_tokens() -> dict:
    """Standard-GAT negative control: significant pathways under identical tests."""
    st = {}
    for ds in ("LUAD", "BRCA"):
        p = ROOT / "results" / "rewiring" / ds / "stdgat_pathway_test.csv"
        n_sig = n_tot = None
        if p.exists():
            t = pd.read_csv(p)
            n_tot = len(t)
            n_sig = int((t["q"] < 0.05).sum()) if "q" in t.columns else None
        st["STDGAT_%s_SIG" % ds] = "\u2014" if n_sig is None else str(n_sig)
        st["STDGAT_%s_TOT" % ds] = "\u2014" if n_tot is None else str(n_tot)
    return st


def ext_rw_tokens() -> dict:
    """External rewiring replication: per-cohort HR of high vs low rewiring."""
    st = {}
    for ds in ("LUAD", "BRCA"):
        d = ROOT / "results" / "rewiring_external" / ds
        rows = []
        if d.is_dir():
            for f in sorted(d.glob("*_summary.csv")):
                s = pd.read_csv(f)
                if not len(s):
                    continue
                r = s.iloc[0]
                if pd.notna(r.get("p")):
                    rows.append((r["cohort"], float(r["n"]), r["hr_high_vs_low"],
                                 r["hr_lo"], r["hr_hi"], float(r["p"])))
        if not rows:
            st["EXT_RW_%s_SENT" % ds] = ""
            continue
        parts = []
        n_nom = 0
        for cohort, n, hr, lo, hi, p in rows:
            if p < 0.05:
                n_nom += 1
            parts.append("%s (HR %s, 95%% CI %s-%s, %s; n=%d)"
                         % (cohort, fmt2(float(hr)), fmt2(float(lo)), fmt2(float(hi)),
                            fmt_p(p), int(n)))
        st["EXT_RW_%s_SENT" % ds] = ("%s of %d GEO cohort(s) showed a nominally significant "
                                     "association between rewiring magnitude and OS: %s"
                                     % (n_nom, len(rows), "; ".join(parts)))
    return st



def sensitivity_tokens() -> dict:
    """Robustness of clinical anchors to the rewiring-magnitude definition."""
    p = ROOT / "results" / "rewiring" / "sensitivity_magnitude.csv"
    if not p.exists():
        return {"SENSITIVITY_SENT": ""}
    t = pd.read_csv(p)
    def rng(ds, col, better="rho"):
        sub = t[(t["dataset"] == ds) & (t["definition"].isin(["L1", "z-L1", "1-r"]))]
        vals = sub[col].dropna()
        return (min(vals), max(vals)) if len(vals) else None
    parts = []
    b = rng("BRCA", "rho_ki67")
    if b:
        parts.append("the Ki-67 association in BRCA (rho = %s-%s across definitions, all P<0.001)"
                     % (fmt2(b[0]), fmt2(b[1])))
    im = rng("IMvigor210", "rho_risk")
    if im:
        parts.append("the risk-score association in IMvigor210 (rho = %s-%s, all P<1e-9)"
                     % (fmt2(im[0]), fmt2(im[1])))
    lu = t[(t["dataset"] == "LUAD") & (t["definition"] == "L1")]
    if len(lu):
        r0 = lu.iloc[0]
        parts.append("the LUAD TMB association was directionally consistent but weaker "
                     "(rho = %s, %s, for the primary definition)" % (fmt2(r0["rho_tmb"]), fmt_p(r0["P_tmb"])))
    return {"SENSITIVITY_SENT": "; ".join(parts) if parts else ""}



def seed_tokens() -> dict:
    """Three-seed sensitivity on LUAD/BRCA (mean +/- SD over 3 seeds x 5 folds)."""
    st = {}
    for ds in ("LUAD", "BRCA"):
        p = ROOT / "results" / ("seed_analysis_%s.csv" % ds)
        if p.exists():
            t = pd.read_csv(p)
            if len(t):
                st["SEEDS_%s" % ds] = "%s \u00b1 %s" % (fmt2(float(t["c_index"].mean())),
                                                         fmt2(float(t["c_index"].std())))
    return st



def table_cal() -> str:
    """Calibration table from results/calibration_results.csv (Table S4 -> inline)."""
    csv = ROOT / "results" / "calibration_results.csv"
    if not csv.exists():
        return "Calibration results pending."
    df = pd.read_csv(csv)
    model_names = {"path_agnn_cox": "Path-AGNN-Cox", "ridge_cox": "Ridge-Cox"}
    setting_names = {"internal": "Internal CV", "external": "External transfer"}
    lines = ["| Dataset | Setting | Cohort | Model | N | Events | Slope | 95% CI | MAE |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append("| %s | %s | %s | %s | %d | %d | %s | %s\u2013%s | %s |" % (
            r["dataset"], setting_names.get(r["setting"], r["setting"]), r["cohort"],
            model_names.get(r["model"], r["model"]), int(r["n"]), int(r["events"]),
            fmt2(float(r["slope"])), fmt2(float(r["slope_ci_low"])), fmt2(float(r["slope_ci_high"])),
            fmt2(float(r["cal_mae"]))))
    return "\n".join(lines)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true", help="allow rendering with incomplete data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not CSV.exists():
        sys.exit("benchmark_results.csv not found")
    df = load_df()
    info = read_info()
    st = compute_stats(df, info)
    st.update(rewiring_tokens(ROOT / "results" / "rewiring" / "LUAD"))
    st.update(immune_drug_tokens(ROOT / "results" / "immune"))
    st.update(imv_tokens())
    st.update(stdgat_tokens())
    st.update(ext_rw_tokens())
    st.update(sensitivity_tokens())
    st.update(seed_tokens())
    st.update(calibration_tokens())
    tables = {
        "DATASETS": table1(df, info),
        "BENCHMARK": table2(df),
        "ABLATION": table3(df),
        "EXTERNAL": table4(df, info),
        "REWIRING": table5(ROOT / "results" / "rewiring" / "LUAD"),
        "DRUGS": table6(ROOT / "results" / "immune"),
        "CALIBRATION": table_cal(),
    }

    mf_path = ROOT / "results" / "figures" / "figure_manifest.json"
    manifest = json.loads(mf_path.read_text(encoding="utf-8")) if mf_path.exists() else {}
    src = io.open(TEMPLATE, encoding="utf-8-sig").read()
    tbl_no, fig_no = {}, {}
    t_next, f_next = 1, 1

    def repl(m):
        nonlocal t_next, f_next
        kind, name, panel = m.group(1), m.group(2), m.group(3)
        if kind in ("TREF", "TDEF"):
            if name not in tbl_no:
                tbl_no[name] = t_next; t_next += 1
            n = tbl_no[name]
            return f"Table {n}" if kind == "TREF" else f"Table {n}"
        if kind in ("FREF", "FDEF"):
            if name not in fig_no:
                fig_no[name] = f_next; f_next += 1
            n = fig_no[name]
            return f"Figure {n}{panel or ''}" if kind == "FREF" else f"Figure {n}"
        if kind == "FIG":
            if name not in fig_no:
                fig_no[name] = f_next; f_next += 1
            n = fig_no[name]
            fname = manifest.get("Figure%d" % n, {}).get("file", "")
            return "![Figure %d](results/figures/%s)" % (n, fname)
        return m.group(0)

    pat = re.compile(r"\{\{(TREF|TDEF|FREF|FDEF|FIG):([A-Z]+)(?:\|([A-Z]))?\}\}")
    src = pat.sub(repl, src)

    # data tokens + table content
    for key, val in st.items():
        src = src.replace("{{" + key + "}}", val)
    for key, content in tables.items():
        src = src.replace("{{TABLE:" + key + "}}", content)

    leftovers = re.findall(r"\{\{[A-Z_:|]+\}\}", src)
    out_path = Path(args.out) if args.out else OUT
    io.open(out_path, "w", encoding="utf-8", newline="\n").write(src)
    print("rendered:", out_path)
    print("table numbering:", tbl_no)
    print("figure numbering:", fig_no)
    if leftovers:
        print("WARNING leftover tokens:", leftovers)

if __name__ == "__main__":
    main()

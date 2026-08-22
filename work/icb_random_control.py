# -*- coding: utf-8 -*-
"""Random gene-set control for ICB rewiring-score analysis.
For each cohort: rebuild the rewiring template with equal-size random gene sets
(50 reps per pathway), project onto cohort expression, test response difference,
and compare the real effect size against the random null distribution.
"""
import gzip, io, os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
RAW = ROOT / "data/raw/ICB"
GMT = ROOT / "data/pathways/kegg_cancer_core.gmt"
OUT = ROOT / "results/icb"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260822)
N_RAND = 50

def load_gmt():
    gmt = {}
    for ln in io.open(GMT, encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        gmt[p[0]] = p[2:]
    return gmt

def zscore_rows(X):
    mu = X.mean(axis=0); sd = X.std(axis=0, ddof=1); sd[sd == 0] = 1.0
    return (X - mu) / sd

def corr_matrix(X):
    Z = zscore_rows(X)
    return np.nan_to_num(Z.T @ Z / (Z.shape[0] - 1), nan=0.0)

def build_template_from_tcga(tcga, gmt):
    keep = [g for g in tcga.columns if g not in ("sample_id", "OS_time", "OS_event")]
    if "risk_score" in tcga.columns:
        med = tcga["risk_score"].median()
        hi = tcga[tcga["risk_score"] >= med].drop(columns=["sample_id", "risk_score"])
        lo = tcga[tcga["risk_score"] < med].drop(columns=["sample_id", "risk_score"])
    else:
        sub = tcga.dropna(subset=["OS_event"])
        sub = sub[sub["OS_event"].isin([0, 1])]
        hi = sub[sub["OS_event"] == 1].drop(columns=["sample_id", "OS_time", "OS_event"])
        lo = sub[sub["OS_event"] == 0].drop(columns=["sample_id", "OS_time", "OS_event"])
    templates = {}
    for pw, genes in gmt.items():
        genes = [g for g in genes if g in keep]
        if len(genes) < 3:
            continue
        templates[pw] = (genes, corr_matrix(hi[genes].to_numpy(dtype=float)) - corr_matrix(lo[genes].to_numpy(dtype=float)))
    return templates, keep

def score_samples(expr, templates):
    total = np.zeros(len(expr))
    n_pw = 0
    for pw, (genes, D) in templates.items():
        genes = [g for g in genes if g in expr.columns]
        if len(genes) < 3:
            continue
        Xc = expr[genes].to_numpy(dtype=float)
        idx = [genes.index(g) for g in genes]
        Ds = D[np.ix_(idx, idx)]
        s = np.einsum("ij,jk,ik->i", Xc, Ds, Xc)
        total += np.nan_to_num(s, nan=0.0)
        n_pw += 1
    return total / max(n_pw, 1)

def run_random_null(tcga, gmt, cohort_expr, y, n_rand=50):
    """cohort_expr: samples x genes; y: binary response (nan excluded)."""
    m = ~np.isnan(y)
    X = cohort_expr.loc[m]
    yb = y[m].astype(int)
    pool = [g for g in tcga.columns if g not in ("sample_id", "OS_time", "OS_event", "risk_score")]
    size_map = {pw: len([g for g in gmt[pw] if g in pool]) for pw in gmt}
    gs = []
    for pw in gmt:
        sz = size_map[pw]
        if sz < 3:
            continue
        for ri in range(n_rand):
            rset = list(rng.choice(pool, size=sz, replace=False))
            rset = [g for g in rset if g in X.columns]
            if len(rset) < 3:
                continue
            Ch = corr_matrix(tcga.loc[tcga["OS_event"] == 1, rset].to_numpy(dtype=float))
            Cl = corr_matrix(tcga.loc[tcga["OS_event"] == 0, rset].to_numpy(dtype=float))
            D = Ch - Cl
            Xc = X[rset].to_numpy(dtype=float)
            idx = [rset.index(g) for g in rset]
            Ds = D[np.ix_(idx, idx)]
            s = np.einsum("ij,jk,ik->i", Xc, Ds, Xc)
            s = (s - s.mean()) / s.std(ddof=1)
            a = s[yb == 1]; b = s[yb == 0]
            if len(a) >= 3 and len(b) >= 3 and a.std(ddof=1) + b.std(ddof=1) > 0:
                u, p = stats.mannwhitneyu(a, b)
                gs.append((pw, ri, a.mean() - b.mean(), p))
    return pd.DataFrame(gs, columns=["pathway", "rep", "mean_diff", "P"])

if __name__ == "__main__":
    gmt = load_gmt()
    blca = ROOT / "data/processed/BLCA/train.csv"
    # IMvigor210
    imv_expr = ROOT / "data/processed/IMvigor210/train.csv"
    imv_clin = ROOT / "data/processed/IMvigor210/clinical.csv"
    tcga_blca = pd.read_csv(blca)
    expr = pd.read_csv(imv_expr).set_index("sample_id")
    clin = pd.read_csv(imv_clin).set_index("sample_id")
    common = sorted(set(expr.index) & set(clin.index))
    y = np.where(clin.loc[common, "response"] == "CR/PR", 1,
                 np.where(clin.loc[common, "response"] == "SD/PD", 0, np.nan))
    null = run_random_null(tcga_blca, gmt, expr.loc[common], y)
    null.to_csv(OUT / "imvigor210_random_null.csv", index=False)
    print("IMvigor210 random null done:", len(null), "rows; mean_diff quantiles:",
          null["mean_diff"].quantile([0.025, 0.5, 0.975]).round(3).to_dict())

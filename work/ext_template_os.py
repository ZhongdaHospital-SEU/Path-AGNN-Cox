# -*- coding: utf-8 -*-
"""Template-transfer OS validation: TCGA correlation-difference templates
(BRCA/KIRC/LUAD) transferred to GEO cohorts; Cox HR of high-vs-low template
score (median split) + continuous-score z; fixed-effect meta per cancer."""
import io, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
GMT = ROOT / "data/pathways/kegg_cancer_core.gmt"
OUT = ROOT / "results" / "template_external"
OUT.mkdir(parents=True, exist_ok=True)

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

def build_template(train_csv, gmt):
    tcga = pd.read_csv(train_csv)
    keep = [g for g in tcga.columns if g not in ("sample_id", "OS_time", "OS_event")]
    sub = tcga.dropna(subset=["OS_event"]); sub = sub[sub["OS_event"].isin([0, 1])]
    hi = sub[sub["OS_event"] == 1].drop(columns=["sample_id", "OS_time", "OS_event"])
    lo = sub[sub["OS_event"] == 0].drop(columns=["sample_id", "OS_time", "OS_event"])
    tpl = {}
    for pw, genes in gmt.items():
        genes = [g for g in genes if g in keep]
        if len(genes) < 3:
            continue
        tpl[pw] = (genes, corr_matrix(hi[genes].to_numpy(dtype=float)) - corr_matrix(lo[genes].to_numpy(dtype=float)))
    return tpl

def score_samples(expr, tpl):
    total = np.zeros(len(expr))
    n = 0
    for pw, (genes, D) in tpl.items():
        genes = [g for g in genes if g in expr.columns]
        if len(genes) < 3:
            continue
        Xc = expr[genes].to_numpy(dtype=float)
        idx = [genes.index(g) for g in genes]
        Ds = D[np.ix_(idx, idx)]
        s = np.einsum("ij,jk,ik->i", Xc, Ds, Xc)
        total += np.nan_to_num(s, nan=0.0)
        n += 1
    return total / max(n, 1)

def cox_hr(df):
    """df: T, E, g -> (hr, lo, hi, p, n_events) via lifelines; fallback logrank."""
    from lifelines import CoxPHFitter
    d = pd.DataFrame({"T": df["T"].to_numpy(float), "E": df["E"].to_numpy(float),
                      "g": df["g"].to_numpy(float)})
    cf = CoxPHFitter().fit(d, duration_col="T", event_col="E")
    s = cf.summary
    hr = float(s["exp(coef)"]["g"])
    lo = float(s["exp(coef) lower 95%"]["g"]); hi = float(s["exp(coef) upper 95%"]["g"])
    p = float(s["p"]["g"])
    return hr, lo, hi, p, int(d["E"].sum())

def cox_cont(df):
    """continuous score z and P from Cox on standardized score."""
    from lifelines import CoxPHFitter
    d = pd.DataFrame({"T": df["T"].to_numpy(float), "E": df["E"].to_numpy(float),
                      "s": (df["s"] - df["s"].mean()) / df["s"].std(ddof=1)})
    cf = CoxPHFitter().fit(d, duration_col="T", event_col="E")
    s = cf.summary
    return float(s["z"]["s"]), float(s["p"]["s"])

gmt = load_gmt()
COHORTS = {
    "BRCA": ["GSE20685", "GSE21653", "GSE7390"],
    "KIRC": ["GSE29609"],
    "LUAD": ["GSE31210", "GSE50081", "GSE68465"],
}
rows = []
for ds, cohorts in COHORTS.items():
    tpl = build_template(ROOT / "data/processed" / ds / "train.csv", gmt)
    print(ds, "template pathways:", len(tpl), flush=True)
    for co in cohorts:
        ext = pd.read_csv(ROOT / "data/processed" / ds / "external" / (co + ".csv"))
        ext = ext.dropna(subset=["OS_time", "OS_event"])
        ext = ext[ext["OS_event"].isin([0, 1])]
        Ex = ext.drop(columns=["sample_id", "OS_time", "OS_event"])
        sc = score_samples(Ex, tpl)
        sc = (sc - sc.mean()) / sc.std(ddof=1)
        med = np.median(sc)
        grp = (sc >= med).astype(int)
        d = pd.DataFrame({"T": ext["OS_time"].to_numpy(float), "E": ext["OS_event"].to_numpy(float),
                          "g": grp, "s": sc})
        hr, lo, hi, p, ne = cox_hr(d)
        z, pc = cox_cont(d)
        rows.append({"dataset": ds, "cohort": co, "n": len(d), "events": ne,
                     "hr_high_vs_low": hr, "hr_lo": lo, "hr_hi": hi, "p_median": p,
                     "z_cont": z, "p_cont": pc})
        print(co, "n=%d ev=%d HR=%.2f (%.2f-%.2f) P=%.3f z_cont=%.2f P=%.3f"
              % (len(d), ne, hr, lo, hi, p, z, pc), flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "template_external_os.csv", index=False)

# fixed-effect meta per cancer on continuous-score z (Stouffer, sample-weight + plain)
meta = []
for ds in COHORTS:
    sub = res[res["dataset"] == ds]
    if len(sub) == 0:
        continue
    zs = sub["z_cont"].to_numpy()
    ns = sub["n"].to_numpy()
    z_unw = zs.sum() / np.sqrt(len(zs))
    p_unw = 2 * stats.norm.sf(abs(z_unw))
    z_w = (zs * ns).sum() / np.sqrt((ns ** 2).sum())
    p_w = 2 * stats.norm.sf(abs(z_w))
    # inverse-variance fixed effect on logHR (median split)
    se_log = np.log(sub["hr_hi"] / sub["hr_lo"]) / 3.92
    loghr = np.log(sub["hr_high_vs_low"])
    w = 1 / se_log ** 2
    lh = (loghr * w).sum() / w.sum()
    se_lh = np.sqrt(1 / w.sum())
    z_hr = lh / se_lh
    p_hr = 2 * stats.norm.sf(abs(z_hr))
    meta.append({"dataset": ds, "n_cohorts": len(sub), "z_unw": z_unw, "p_unw": p_unw,
                 "z_w": z_w, "p_w": p_w, "hr_fe": float(np.exp(lh)), "hr_lo_fe": float(np.exp(lh - 1.96 * se_lh)),
                 "hr_hi_fe": float(np.exp(lh + 1.96 * se_lh)), "z_hr": z_hr, "p_hr": p_hr})
    print(ds, "meta: z_unw=%.2f P=%.3f | z_w=%.2f P=%.3f | HR_fe=%.2f P=%.3f"
          % (z_unw, p_unw, z_w, p_w, np.exp(lh), p_hr), flush=True)
pd.DataFrame(meta).to_csv(OUT / "template_external_meta.csv", index=False)
print("DONE")

# -*- coding: utf-8 -*-
"""Plan B main analysis: rewiring score vs ICB response, per cohort + meta.
Templates: same-cancer TCGA (SKCM for melanoma cohorts; BLCA for IMvigor210),
OS-event stratified correlation-difference. Cohort-internal z-score scoring.
"""
import io
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
GMT = ROOT / "data/pathways/kegg_cancer_core.gmt"
RES = ROOT / "results/icb"
RES.mkdir(parents=True, exist_ok=True)

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
    return templates

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

def hedges_g(x, y):
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if sp == 0:
        return 0.0, 1.0
    g = (x.mean() - y.mean()) / sp
    j = 1 - 3 / (4 * (nx + ny) - 9)
    v = (nx + ny) / (nx * ny) + g * g / (2 * (nx + ny))
    return g * j, v

def random_effects_meta(rows):
    gs = np.array([r[1] for r in rows]); vs = np.array([r[2] for r in rows])
    w = 1 / vs
    gbar = (w * gs).sum() / w.sum()
    q = float((w * (gs - gbar) ** 2).sum())
    k = len(rows)
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    ws = 1 / (vs + tau2)
    g_pool = float((ws * gs).sum() / ws.sum())
    se = float(np.sqrt(1 / ws.sum()))
    z = g_pool / se
    p = 2 * stats.norm.sf(abs(z))
    i2 = float(max(0.0, 100 * (q - (k - 1)) / q)) if q > 0 else 0.0
    return {"n_cohorts": k, "g": g_pool, "se": se, "z": z, "P": p, "I2": i2, "tau2": tau2,
            "g_lo": g_pool - 1.96 * se, "g_hi": g_pool + 1.96 * se}

def cohort_analysis(name, expr, clin, tpl):
    common = sorted(set(expr.index) & set(clin["sample"]))
    if len(common) < 8:
        return None
    X = expr.loc[common]
    c = clin.set_index("sample").loc[common]
    sc = score_samples(X, tpl)
    sc = (sc - sc.mean()) / sc.std(ddof=1)
    y = pd.to_numeric(c["resp_bin"], errors="coerce").to_numpy()
    m = ~np.isnan(y)
    if m.sum() < 8 or (y[m] == 1).sum() < 3 or (y[m] == 0).sum() < 3:
        return None
    a = sc[m][y[m] == 1]; b = sc[m][y[m] == 0]
    g, v = hedges_g(a, b)
    u, p = stats.mannwhitneyu(a, b)
    osrow = None
    if "os_time" in c.columns and "os_event" in c.columns:
        c2 = c.dropna(subset=["os_time", "os_event"])
        c2 = c2[c2["os_event"].isin([0, 1])]
        if len(c2) >= 10 and c2["os_event"].sum() >= 5:
            pos = np.isin(common, c2.index)
            med = np.median(sc[pos])
            grp = (sc[pos] >= med).astype(int)
            try:
                from lifelines import CoxPHFitter
                d = pd.DataFrame({"T": c2["os_time"].to_numpy(float), "E": c2["os_event"].to_numpy(float), "g": grp.to_numpy()})
                cf = CoxPHFitter().fit(d, duration_col="T", event_col="E")
                osrow = {"os_hr": float(cf.hazard_ratios_["g"]), "os_P": float(cf.summary["p"]["g"])}
            except Exception:
                osrow = None
    return {"cohort": name, "n": int(m.sum()), "n_resp": int((y[m] == 1).sum()),
            "n_nonresp": int((y[m] == 0).sum()), "g": float(g), "v": float(v),
            "wilcox_P": float(p), "os_hr": osrow["os_hr"] if osrow else np.nan,
            "os_P": osrow["os_P"] if osrow else np.nan}

def main():
    gmt = load_gmt()
    skcm_tpl = None
    skcm_train = ROOT / "data/processed/SKCM/train.csv"
    if skcm_train.exists():
        skcm_tpl = build_template(skcm_train, gmt)
        print("SKCM template ready")
    blca_tpl = build_template(ROOT / "data/processed/BLCA/train.csv", gmt)
    rows = []
    # IMvigor210
    imv_expr = pd.read_csv(ROOT / "data/processed/IMvigor210/train.csv").set_index("sample_id")
    imv_clin = pd.read_csv(ROOT / "data/processed/IMvigor210/clinical.csv")
    imv_clin["sample"] = imv_clin["sample_id"]
    imv_clin["resp_bin"] = imv_clin["response"].map({"CR/PR": 1, "SD/PD": 0})
    r = cohort_analysis("IMvigor210", imv_expr, imv_clin, blca_tpl)
    if r: rows.append(r); print(r)
    # GSE cohorts with SKCM template
    if skcm_tpl is not None:
        for name, ep_name in [("gse91061", "gse91061_expr_pre.csv"),
                              ("gse78220", "gse78220_expr.csv"),
                              ("gse100797", "gse100797_expr.csv")]:
            ep = RES / ep_name
            cp = RES / f"{name}_clinical.csv"
            if not (ep.exists() and cp.exists()):
                continue
            expr = pd.read_csv(ep).set_index("sample")
            clin = pd.read_csv(cp)
            r = cohort_analysis(name.upper(), expr, clin, skcm_tpl)
            if r: rows.append(r); print(r)
    pd.DataFrame(rows).to_csv(RES / "cohort_results.csv", index=False)
    if len(rows) >= 2:
        meta = random_effects_meta([(r["cohort"], r["g"], r["v"]) for r in rows])
        print("META:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in meta.items()})
        pd.Series(meta).to_csv(RES / "meta_results.csv")
    print("DONE")

if __name__ == "__main__":
    main()

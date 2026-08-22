# -*- coding: utf-8 -*-
"""Plan B: ICB cohort rewiring-score vs response meta-analysis.
Cohorts: IMvigor210 (local), GSE91061, GSE78220, GSE100797.
Template: same-cancer TCGA high-vs-low risk correlation-difference (SKCM for melanoma
cohorts; BLCA for IMvigor210). Score = pathway-blocked x^T D x; cohort z-score;
Hedges g per cohort; random-effects meta.
"""
import gzip, io, os, re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
RAW = ROOT / "data/raw/ICB"
GMT = ROOT / "data/pathways/kegg_cancer_core.gmt"
OUT = ROOT / "results/icb"
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

def parse_series(path):
    """Return DataFrame: geo_accession x characteristics dict."""
    rows = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("!Sample_"):
                continue
            key, _, vals = line.partition("\t")
            key = key.strip()
            if key not in ("!Sample_geo_accession", "!Sample_title", "!Sample_characteristics_ch1"):
                continue
            import csv as _csv
            vals = list(_csv.reader([vals.strip()], delimiter="\t"))[0]
            for i, v in enumerate(vals):
                v = v.strip('"')
                rows.setdefault(i, {})[key] = v
    df = pd.DataFrame(rows).T
    df["geo"] = df["!Sample_geo_accession"]
    ch = {}
    for _, r in df.iterrows():
        d = {}
        c = r.get("!Sample_characteristics_ch1")
        if isinstance(c, str):
            for part in c.split("|"):
                if ":" in part:
                    k, _, v = part.partition(":")
                    d[k.strip()] = v.strip()
        ch[r["geo"]] = d
    return df, ch

def build_template(cancer, train_csv, gmt, sig_only=False):
    """High-vs-low risk correlation-difference template from TCGA train.csv.
    Prefers model risk_score when available; otherwise OS event stratification
    (OS_event==1 as high-risk, OS_event==0 as low-risk)."""
    tcga = pd.read_csv(train_csv)
    keep = [g for g in tcga.columns if g not in ("sample_id", "OS_time", "OS_event")]
    if "risk_score" in tcga.columns:
        med = tcga["risk_score"].median()
        hi = tcga[tcga["risk_score"] >= med].drop(columns=["sample_id", "risk_score"])
        lo = tcga[tcga["risk_score"] < med].drop(columns=["sample_id", "risk_score"])
    else:
        sub = tcga.dropna(subset=["OS_event"])
        sub = sub[sub["OS_event"].isin([0, 1])]
        if sub["OS_event"].sum() < 10 or (sub["OS_event"] == 0).sum() < 10:
            raise ValueError("not enough OS events for stratification")
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
    """expr: samples x genes DataFrame (symbols). Returns per-sample total score."""
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
    """rows: list of (name, g, v). Returns dict with pooled g, z, P, I2."""
    gs = np.array([r[1] for r in rows]); vs = np.array([r[2] for r in rows])
    w = 1 / vs
    q = float((w * (gs - (w * gs).sum() / w.sum()) ** 2).sum())
    k = len(rows)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())) if q > k - 1 else 0.0
    ws = 1 / (vs + tau2)
    g_pool = float((ws * gs).sum() / ws.sum())
    se = float(np.sqrt(1 / ws.sum()))
    z = g_pool / se
    p = 2 * stats.norm.sf(abs(z))
    i2 = float(max(0.0, 100 * (q - (k - 1)) / q)) if q > 0 else 0.0
    return {"n_cohorts": k, "g": g_pool, "se": se, "z": z, "P": p, "I2": i2, "tau2": tau2}

def main():
    gmt = load_gmt()
    # SKCM template
    skcm_train = ROOT / "data/processed/SKCM/train.csv"
    # IMvigor210 template via BLCA
    blca_train = ROOT / "data/processed/BLCA/train.csv"
    # IMvigor210 expression + response (already processed)
    imv_expr = ROOT / "data/processed/IMvigor210/train.csv"
    imv_clin = ROOT / "data/processed/IMvigor210/clinical.csv"
    cohorts = []
    # --- IMvigor210 (BLCA template) ---
    if blca_train.exists() and imv_expr.exists():
        tpl = build_template("BLCA", blca_train, gmt)
        expr = pd.read_csv(imv_expr).set_index("sample_id")
        clin = pd.read_csv(imv_clin)
        common = sorted(set(expr.index) & set(clin["sample_id"]))
        expr = expr.loc[common]
        clin = clin.set_index("sample_id").loc[common]
        sc = score_samples(expr, tpl)
        sc = (sc - sc.mean()) / sc.std(ddof=1)
        resp = np.where(clin["response"] == "CR/PR", 1, np.where(clin["response"] == "SD/PD", 0, np.nan))
        m = ~np.isnan(resp)
        if m.sum() >= 10:
            g, v = hedges_g(sc[m][resp[m] == 1], sc[m][resp[m] == 0])
            u, p = stats.mannwhitneyu(sc[m][resp[m] == 1], sc[m][resp[m] == 0])
            cohorts.append(("IMvigor210", g, v, float(p), int(m.sum()), int((resp[m] == 1).sum()), int((resp[m] == 0).sum())))
            print("IMvigor210 n=%d resp=%d nonresp=%d g=%.3f wilcox_P=%.3f" % (int(m.sum()), int((resp[m] == 1).sum()), int((resp[m] == 0).sum()), g, p))
    # --- GSE cohorts (SKCM template) ---
    if skcm_train.exists():
        tpl = build_template("SKCM", skcm_train, gmt)
        # GSE91061
        sm = RAW / "GSE91061_series_matrix.txt.gz"
        fpkm = RAW / "GSE91061_BMS038109Sample.hg19KnownGene.fpkm.csv.gz"
        if sm.exists() and fpkm.exists():
            df, ch = parse_series(sm)
            expr = pd.read_csv(fpkm, index_col=0)  # rows Entrez, cols GSM
            expr = expr.T
            expr.index = [str(i) for i in expr.index]
            keep_geo = [g for g in expr.index if g in ch and ch[g].get("visit") == "Pre"]
            expr = expr.loc[keep_geo]
            resp = pd.Series({g: ch[g].get("response", "NA") for g in expr.index})
            ok = resp.isin(["PRCR", "PR", "CR"]).astype(int)
            non = resp.isin(["SD", "PD"]).astype(int)
            y = np.where(ok == 1, 1, np.where(non == 1, 0, np.nan))
            m = ~np.isnan(y)
            if m.sum() >= 8:
                sc = score_samples(expr.loc[m], tpl)
                sc = (sc - sc.mean()) / sc.std(ddof=1)
                g, v = hedges_g(sc[y[m] == 1], sc[y[m] == 0])
                u, p = stats.mannwhitneyu(sc[y[m] == 1], sc[y[m] == 0])
                cohorts.append(("GSE91061", g, v, float(p), int(m.sum()), int((y[m] == 1).sum()), int((y[m] == 0).sum())))
                print("GSE91061 n=%d resp=%d nonresp=%d g=%.3f wilcox_P=%.3f" % (int(m.sum()), int((y[m] == 1).sum()), int((y[m] == 0).sum()), g, p))
        # GSE78220
        sm2 = RAW / "GSE78220_series_matrix.txt.gz"
        xlsx = RAW / "GSE78220_PatientFPKM.xlsx"
        if sm2.exists() and xlsx.exists():
            df2, ch2 = parse_series(sm2)
            ex2 = pd.read_excel(xlsx, index_col=0)
            ex2 = ex2.T
            ex2.index = [str(i) for i in ex2.index]
            resp = pd.Series({g: ch2.get(g, {}).get("response", "NA") for g in ex2.index})
            ok = resp.str.upper().isin(["PR", "CR", "RESPONDER", "R"]).astype(int)
            non = resp.str.upper().isin(["SD", "PD", "NONRESPONDER", "NR"]).astype(int)
            y = np.where(ok == 1, 1, np.where(non == 1, 0, np.nan))
            m = ~np.isnan(y)
            if m.sum() >= 8:
                sc = score_samples(ex2.loc[m], tpl)
                sc = (sc - sc.mean()) / sc.std(ddof=1)
                g, v = hedges_g(sc[y[m] == 1], sc[y[m] == 0])
                u, p = stats.mannwhitneyu(sc[y[m] == 1], sc[y[m] == 0])
                cohorts.append(("GSE78220", g, v, float(p), int(m.sum()), int((y[m] == 1).sum()), int((y[m] == 0).sum())))
                print("GSE78220 n=%d resp=%d nonresp=%d g=%.3f wilcox_P=%.3f" % (int(m.sum()), int((y[m] == 1).sum()), int((y[m] == 0).sum()), g, p))
        # GSE100797
        sm3 = RAW / "GSE100797_series_matrix.txt.gz"
        prc = RAW / "GSE100797_ProcessedData.txt.gz"
        if sm3.exists() and prc.exists():
            df3, ch3 = parse_series(sm3)
            ex3 = pd.read_csv(prc, compression="gzip", sep="\t", index_col=0)
            ex3 = ex3.T
            ex3.index = [str(i) for i in ex3.index]
            resp = pd.Series({g: ch3.get(g, {}).get("response", "NA") for g in ex3.index})
            ok = resp.str.upper().isin(["PR", "CR", "RESPONDER", "R"]).astype(int)
            non = resp.str.upper().isin(["SD", "PD", "NONRESPONDER", "NR"]).astype(int)
            y = np.where(ok == 1, 1, np.where(non == 1, 0, np.nan))
            m = ~np.isnan(y)
            if m.sum() >= 8:
                sc = score_samples(ex3.loc[m], tpl)
                sc = (sc - sc.mean()) / sc.std(ddof=1)
                g, v = hedges_g(sc[y[m] == 1], sc[y[m] == 0])
                u, p = stats.mannwhitneyu(sc[y[m] == 1], sc[y[m] == 0])
                cohorts.append(("GSE100797", g, v, float(p), int(m.sum()), int((y[m] == 1).sum()), int((y[m] == 0).sum())))
                print("GSE100797 n=%d resp=%d nonresp=%d g=%.3f wilcox_P=%.3f" % (int(m.sum()), int((y[m] == 1).sum()), int((y[m] == 0).sum()), g, p))
    # meta
    if len(cohorts) >= 2:
        meta = random_effects_meta([(c[0], c[1], c[2]) for c in cohorts])
        print("META:", meta)
    pd.DataFrame([c for c in cohorts], columns=["cohort", "g", "v", "wilcox_P", "n", "n_resp", "n_nonresp"]).to_csv(OUT / "cohort_results.csv", index=False)
    print("DONE")

if __name__ == "__main__":
    main()

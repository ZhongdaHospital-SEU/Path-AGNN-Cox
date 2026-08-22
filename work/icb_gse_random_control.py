# -*- coding: utf-8 -*-
"""Random gene-set control for the three melanoma ICB cohorts (SKCM template)."""
import io
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
GMT = ROOT / "data/pathways/kegg_cancer_core.gmt"
RES = ROOT / "results/icb"
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

tcga = pd.read_csv(ROOT / "data/processed/SKCM/train.csv")
sub = tcga.dropna(subset=["OS_event"])
sub = sub[sub["OS_event"].isin([0, 1])]
hi = sub[sub["OS_event"] == 1].drop(columns=["sample_id", "OS_time", "OS_event"])
lo = sub[sub["OS_event"] == 0].drop(columns=["sample_id", "OS_time", "OS_event"])
pool = [g for g in tcga.columns if g not in ("sample_id", "OS_time", "OS_event")]
gmt = load_gmt()

for cohort in ["gse91061", "gse78220", "gse100797"]:
    ep = RES / f"{cohort}_expr.csv"
    cp = RES / f"{cohort}_clinical.csv"
    if not (ep.exists() and cp.exists()):
        ep = RES / "gse91061_expr_pre.csv" if cohort == "gse91061" else ep
    expr = pd.read_csv(ep).set_index("sample")
    clin = pd.read_csv(cp)
    common = sorted(set(expr.index) & set(clin["sample"]))
    X = expr.loc[common]
    c = clin.set_index("sample").loc[common]
    y = pd.to_numeric(c["resp_bin"], errors="coerce").to_numpy()
    m = ~np.isnan(y)
    X = X[m]; yb = y[m].astype(int)
    rows = []
    for pw, genes in gmt.items():
        genes = [g for g in genes if g in pool]
        sz = len(genes)
        if sz < 3:
            continue
        for ri in range(N_RAND):
            rset = list(rng.choice(pool, size=sz, replace=False))
            rset = [g for g in rset if g in X.columns]
            if len(rset) < 3:
                continue
            Ch = corr_matrix(hi[rset].to_numpy(dtype=float))
            Cl = corr_matrix(lo[rset].to_numpy(dtype=float))
            D = Ch - Cl
            Xc = X[rset].to_numpy(dtype=float)
            idx = [rset.index(g) for g in rset]
            Ds = D[np.ix_(idx, idx)]
            s = np.einsum("ij,jk,ik->i", Xc, Ds, Xc)
            s = (s - s.mean()) / s.std(ddof=1)
            a = s[yb == 1]; b = s[yb == 0]
            if len(a) >= 3 and len(b) >= 3 and a.std(ddof=1) + b.std(ddof=1) > 0:
                rows.append((pw, ri, a.mean() - b.mean()))
    null = pd.DataFrame(rows, columns=["pathway", "rep", "mean_diff"])
    null.to_csv(RES / f"{cohort}_random_null.csv", index=False)
    nd = null["mean_diff"]
    print("%s random null: n=%d median=%.3f sd=%.3f p2.5=%.3f p97.5=%.3f" %
          (cohort, len(nd), nd.median(), nd.std(), nd.quantile(0.025), nd.quantile(0.975)))
print("DONE")

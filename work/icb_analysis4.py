# -*- coding: utf-8 -*-
"""GSE298296 (perioperative TURBT, anti-PD-1/PD-L1?) BLCA-template ICB analysis.
Replicates icb_analysis2/3 logic + random gene-set control + meta update."""
import io, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
GMT = ROOT / "data/pathways/kegg_cancer_core.gmt"
RES = ROOT / "results/icb"
RES.mkdir(parents=True, exist_ok=True)
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

# ---------- BLCA template (same as icb_analysis2/3: OS-event stratified) ----------
tcga = pd.read_csv(ROOT / "data/processed/BLCA/train.csv")
keep = [g for g in tcga.columns if g not in ("sample_id", "OS_time", "OS_event")]
sub = tcga.dropna(subset=["OS_event"]); sub = sub[sub["OS_event"].isin([0, 1])]
hi = sub[sub["OS_event"] == 1].drop(columns=["sample_id", "OS_time", "OS_event"])
lo = sub[sub["OS_event"] == 0].drop(columns=["sample_id", "OS_time", "OS_event"])
pool = keep
gmt = load_gmt()
tpl = {}
for pw, genes in gmt.items():
    genes = [g for g in genes if g in keep]
    if len(genes) < 3:
        continue
    tpl[pw] = (genes, corr_matrix(hi[genes].to_numpy(dtype=float)) - corr_matrix(lo[genes].to_numpy(dtype=float)))
print("BLCA template pathways:", len(tpl), flush=True)

# ---------- GSE298296 ----------
expr = pd.read_csv(ROOT / "work/_gse298296_expr.csv", index_col=0)  # 102 x genes
expr.index = [int(re.match(r"GSM\d+_(\d+)\.CEL", c).group(1)) for c in expr.index]
expr.index.name = "sample"
resp = pd.read_csv(ROOT / "work/_gse298296_resp_map.csv")
resp["resp_bin"] = resp["response"].map({"CR": 1, "PR": 1, "NR": 0})
common = sorted(set(expr.index) & set(resp["sample"]))
print("GSE298296 common:", len(common), "resp:", resp.set_index("sample").loc[common, "resp_bin"].sum().astype(int), "nonresp:", (resp.set_index("sample").loc[common, "resp_bin"] == 0).sum(), flush=True)
X = expr.loc[common]
y = resp.set_index("sample").loc[common, "resp_bin"].to_numpy(float)

def score_all(X, tpl):
    total = np.zeros(len(X))
    n = 0
    for pw, (genes, D) in tpl.items():
        genes = [g for g in genes if g in X.columns]
        if len(genes) < 3:
            continue
        Xc = X[genes].to_numpy(dtype=float)
        idx = [genes.index(g) for g in genes]
        Ds = D[np.ix_(idx, idx)]
        s = np.einsum("ij,jk,ik->i", Xc, Ds, Xc)
        total += np.nan_to_num(s, nan=0.0)
        n += 1
    return total / max(n, 1)

sc = score_all(X, tpl)
sc = (sc - sc.mean()) / sc.std(ddof=1)
a = sc[y == 1]; b = sc[y == 0]
g, v = hedges_g(a, b)
u, p = stats.mannwhitneyu(a, b)
print("real: g=%.4f v=%.4f wilcox_P=%.4f n_resp=%d n_nonresp=%d" % (g, v, p, len(a), len(b)), flush=True)

# ---------- random gene-set control (BLCA template hi/lo) ----------
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
        aa = s[y == 1]; bb = s[y == 0]
        if len(aa) >= 3 and len(bb) >= 3 and aa.std(ddof=1) + bb.std(ddof=1) > 0:
            rows.append((pw, ri, aa.mean() - bb.mean()))
null = pd.DataFrame(rows, columns=["pathway", "rep", "mean_diff"])
null.to_csv(RES / "gse298296_random_null.csv", index=False)
pct = float((null["mean_diff"] >= (a.mean() - b.mean())).mean())
print("random null n=%d median=%.3f pct(real>=null)=%.4f" % (len(null), null["mean_diff"].median(), pct), flush=True)

# ---------- meta: BLCA template cohorts ----------
prev = pd.read_csv(RES / "cohort_results_v3.csv")
blca = prev[prev["cohort"].isin(["IMvigor210", "GSE176307", "GSE225066"])]
rows_all = [(r.cohort, r.g, r.v) for r in blca.itertuples()] + [("GSE298296", g, v)]
meta = random_effects_meta(rows_all)
print("BLCA 4-cohort meta:", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in meta.items()}, flush=True)
pd.Series(meta).to_csv(RES / "meta_blca_v4.csv")

# single-cohort P list for FDR (all ICB cohort-level tests, incl. new)
allrows = [(r.cohort, r.wilcox_P) for r in pd.concat([prev, pd.DataFrame([{"cohort": "GSE298296", "wilcox_P": p}])]).itertuples()]
ps = sorted([x[1] for x in allrows])
print("all cohort-level P:", [round(x, 4) for x in ps], flush=True)
out = pd.DataFrame([{"cohort": c, "P": pp} for c, pp in allrows])
out.to_csv(RES / "cohort_p_values_v4.csv", index=False)
print("DONE")

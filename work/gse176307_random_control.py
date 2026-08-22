# -*- coding: utf-8 -*-
"""Random gene-set control for GSE176307 (BLCA template)."""
import io
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
root = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
sys = __import__("sys")
sys.path.insert(0, str(root))
import importlib.util
spec = importlib.util.spec_from_file_location("icb", root / "work/icb_analysis2.py")
icb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(icb)
gmt = icb.load_gmt()
rng = np.random.default_rng(20260822)
N_RAND = 50

tcga = pd.read_csv(root / "data/processed/BLCA/train.csv")
sub = tcga.dropna(subset=["OS_event"])
sub = sub[sub["OS_event"].isin([0, 1])]
hi = sub[sub["OS_event"] == 1].drop(columns=["sample_id", "OS_time", "OS_event"])
lo = sub[sub["OS_event"] == 0].drop(columns=["sample_id", "OS_time", "OS_event"])
pool = [g for g in tcga.columns if g not in ("sample_id", "OS_time", "OS_event")]

expr = pd.read_csv(root / "results/icb/gse176307_expr.csv").set_index("sample")
clin = pd.read_csv(root / "results/icb/gse176307_clinical.csv")
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
        Ch = icb.corr_matrix(hi[rset].to_numpy(dtype=float))
        Cl = icb.corr_matrix(lo[rset].to_numpy(dtype=float))
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
null.to_csv(root / "results/icb/gse176307_random_null.csv", index=False)
nd = null["mean_diff"]
print("GSE176307 random null: n=%d median=%.3f sd=%.3f p2.5=%.3f p97.5=%.3f" %
      (len(nd), nd.median(), nd.std(), nd.quantile(0.025), nd.quantile(0.975)))
# real
tpl = icb.build_template(root / "data/processed/BLCA/train.csv", gmt)
sc = icb.score_samples(X, tpl)
sc = (sc - sc.mean()) / sc.std(ddof=1)
a = sc[yb == 1]; b = sc[yb == 0]
md = a.mean() - b.mean()
z = (md - nd.mean()) / nd.std()
pct = (nd < md).mean()
print("real mean_diff=%.3f z=%.2f percentile=%.3f" % (md, z, pct))

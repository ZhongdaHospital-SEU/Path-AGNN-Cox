# -*- coding: utf-8 -*-
"""GSE225066 (NEODURVARIB) BLCA-template validation.
- Pre-treatment TURBT samples (SCRNEO, n=16) scored with the TCGA-BLCA template.
- Endpoint: pathological response (Responder = pathological complete response or no
  residual disease; Non-responder = residual disease / progression), mapped from the
  NEODURVARIB supplementary file S5 (StudySubjectID -> Genetic Studies ID -> R-{N} SCR).
- Random gene-set control mirrors gse176307_random_control.py.
- Meta: BLCA-template 3-cohort (IMvigor210 + GSE176307 + GSE225066), random-effects.
"""
import io, sys
from pathlib import Path
import numpy as np, pandas as pd, gzip
from scipy import stats
root = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
sys.path.insert(0, str(root))
import importlib.util
spec = importlib.util.spec_from_file_location("icb", root / "work/icb_analysis2.py")
icb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(icb)
RES = root / "results/icb"
gmt = icb.load_gmt()

# ---------- 1. load counts, keep SCRNEO (pre-treatment) ----------
raw = pd.read_csv(root / "data/raw/ICB/GSE225066_samples.normalizedCounts.txt.gz", sep="\t", compression="gzip")
raw = raw.rename(columns={raw.columns[0]: "gene"})
raw = raw.drop_duplicates("gene").set_index("gene")
scr_cols = [c for c in raw.columns if c.startswith("SCRNEO")]
expr = raw[scr_cols].T
expr.index.name = "sample"
print("SCRNEO samples:", len(scr_cols), "| genes:", expr.shape[1])

# ---------- 2. clinical mapping (from NEODURVARIB suppl S5) ----------
# Genetic Studies ID -> response (R=Responder/clinical benefit, NR=Non responder)
id2resp = {1:"R",2:"R",3:"NR",4:"NR",5:"NR",8:"R",10:"R",11:"NR",14:"R",15:"R",
           16:"NR",18:"R",21:"R",22:"NR",24:"R",26:"R"}
clin = pd.DataFrame({"sample": scr_cols})
clin["gs_id"] = clin["sample"].str.replace("SCRNEO", "").astype(int)
clin["response"] = clin["gs_id"].map(id2resp)
clin["resp_bin"] = clin["response"].map({"R": 1, "NR": 0})
print("response counts:", clin["resp_bin"].value_counts().to_dict())
clin.to_csv(RES / "gse225066_clinical.csv", index=False)
expr.to_csv(RES / "gse225066_expr.csv")

# ---------- 3. score with BLCA template ----------
tpl = icb.build_template(root / "data/processed/BLCA/train.csv", gmt)
common = sorted(set(expr.index) & set(clin["sample"]))
X = expr.loc[common]
c = clin.set_index("sample").loc[common]
y = pd.to_numeric(c["resp_bin"], errors="coerce").to_numpy()
msk = ~np.isnan(y)
Xc = X.to_numpy(dtype=float)
sc = icb.score_samples(X, tpl)
sc = (sc - sc.mean()) / sc.std(ddof=1)
a = sc[msk][y[msk] == 1]; b = sc[msk][y[msk] == 0]
g, v = icb.hedges_g(a, b)
u, p = stats.mannwhitneyu(a, b)
print(f"[GSE225066] n={int(msk.sum())} resp={int((y[msk]==1).sum())} nonresp={int((y[msk]==0).sum())} g={g:.3f} P={p:.3f}")

# ---------- 4. random gene-set control ----------
rng = np.random.default_rng(20260823)
N_RAND = 50
tcga = pd.read_csv(root / "data/processed/BLCA/train.csv")
sub = tcga.dropna(subset=["OS_event"]); sub = sub[sub["OS_event"].isin([0, 1])]
hi = sub[sub["OS_event"] == 1].drop(columns=["sample_id", "OS_time", "OS_event"])
lo = sub[sub["OS_event"] == 0].drop(columns=["sample_id", "OS_time", "OS_event"])
pool = [g for g in tcga.columns if g not in ("sample_id", "OS_time", "OS_event")]
null_rows = []
for pw, genes in gmt.items():
    genes = [g for g in genes if g in pool]
    sz = len(genes)
    if sz < 3: continue
    for ri in range(N_RAND):
        rset = list(rng.choice(pool, size=sz, replace=False))
        rset = [g for g in rset if g in expr.columns]
        if len(rset) < 3: continue
        D = icb.corr_matrix(hi[rset].to_numpy(dtype=float)) - icb.corr_matrix(lo[rset].to_numpy(dtype=float))
        Xm = Xc[:, [expr.columns.get_loc(g) for g in rset]]
        idx = [rset.index(g) for g in rset]
        s = np.einsum("ij,jk,ik->i", Xm, D[np.ix_(idx, idx)], Xm)
        s = (s - s.mean()) / s.std(ddof=1)
        aa = s[msk][y[msk] == 1]; bb = s[msk][y[msk] == 0]
        if len(aa) >= 3 and len(bb) >= 3 and aa.std(ddof=1) + bb.std(ddof=1) > 0:
            null_rows.append((pw, ri, aa.mean() - bb.mean()))
null = pd.DataFrame(null_rows, columns=["pathway", "rep", "mean_diff"])
null.to_csv(RES / "gse225066_random_null.csv", index=False)
nd = null["mean_diff"]
md = a.mean() - b.mean()
pct = float((nd < md).mean())
print(f"random null: n={len(nd)} median={nd.median():.3f} sd={nd.std():.3f} real md={md:.3f} percentile={pct:.3f}")

# ---------- 5. summary row + 3-cohort BLCA meta ----------
row = {"cohort": "GSE225066", "n": int(msk.sum()), "n_resp": int((y[msk] == 1).sum()),
       "n_nonresp": int((y[msk] == 0).sum()), "g": float(g), "v": float(v), "wilcox_P": float(p),
       "random_null_percentile": pct, "random_null_median": float(nd.median())}
pd.DataFrame([row]).to_csv(RES / "gse225066_results.csv", index=False)

prev = pd.read_csv(RES / "cohort_results_v2.csv")
prev = prev[prev["cohort"] != "GSE225066"]
combined = pd.concat([prev, pd.DataFrame([{k: row.get(k) for k in ["cohort","n","n_resp","n_nonresp","g","v","wilcox_P"]}])], ignore_index=True)
combined.to_csv(RES / "cohort_results_v3.csv", index=False)
bl = combined[combined["cohort"].isin(["IMvigor210", "GSE176307", "GSE225066"])]
meta3 = icb.random_effects_meta([(r["cohort"], r["g"], r["v"]) for r in bl.to_dict("records")])
print("\nBLCA 3-cohort meta:", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in meta3.items()})
pd.Series(meta3).to_csv(RES / "meta_blca_v3.csv")
# 2-cohort RECIST meta unchanged
bl2 = combined[combined["cohort"].isin(["IMvigor210", "GSE176307"])]
meta2 = icb.random_effects_meta([(r["cohort"], r["g"], r["v"]) for r in bl2.to_dict("records")])
print("BLCA 2-cohort RECIST meta:", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in meta2.items()})
print("DONE")


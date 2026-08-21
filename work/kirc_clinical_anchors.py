# -*- coding: utf-8 -*-
"""KIRC clinical anchors: stage/grade correlations + anchor-stratified rewiring."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
sys.path.insert(0, str(ROOT))
from benchmark.rewiring_analysis import bh_fdr, pathway_level_test
from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency
from path_agnn_cox.data import load_survival_data, split_features

RW = ROOT / "results" / "rewiring"
d = "KIRC"
df = load_survival_data(str(ROOT / "data" / "processed" / d / "train.csv"))
X, _, _ = split_features(df)
gmt = load_gmt(str(ROOT / "data" / "pathways" / "kegg_cancer_core.gmt"))
adj, mem, gene_order = build_pathway_adjacency(list(X.columns), gmt)
cols = np.array([c for c in gene_order if c in X.columns])
alpha = np.load(RW / d / "alpha.npy")
risk = pd.read_csv(RW / d / "risk_scores.csv")
sid = df["sample_id"].astype(str).str[:12].str.upper().to_numpy()

# clinical matrix (old Xena, 945 rows) + GDC (stage) merged
cm = pd.read_csv(ROOT / "data" / "raw" / "TCGA-xena" / "KIRC_clinicalMatrix", sep="\t", low_memory=False)
cm = cm.assign(sid=cm["sampleID"].astype(str).str[:12].str.upper()).drop_duplicates("sid").set_index("sid")
gd = pd.read_csv(ROOT / "data" / "processed" / d / "clinical.csv")
gd = gd.assign(sid=gd["sample_id"].astype(str).str[:12].str.upper()).drop_duplicates("sid").set_index("sid")
def to_num(s):
    if s is None: return np.nan
    m = pd.Series(s).astype(str).str.extract(r"([IV]+)").iloc[0,0]
    return {"I":1,"II":2,"III":3,"IV":4}.get(m, np.nan) if pd.notna(m) else np.nan
stage_gdc = gd["stage_num"]
stage_xena = cm["pathologic_stage"].map(lambda s: to_num(s) if pd.notna(s) else np.nan)
grade = cm["neoplasm_histologic_grade"].astype(str).str.extract(r"(G[1-4])")[0].str[1].astype(float)
stage = stage_gdc.combine_first(stage_xena)
age = cm["age_at_initial_pathologic_diagnosis"]
print("stage n:", stage.notna().sum(), "grade n:", grade.notna().sum(), "age n:", age.notna().sum())

# rewiring magnitude (L1 from cohort mean, same definition as pipeline)
rwm = np.abs(alpha - alpha.mean(axis=0)).sum(axis=1)
rows = []
for name, vals in [("stage", stage), ("grade", grade), ("age", age)]:
    v = vals.reindex(pd.Index(sid)).to_numpy(dtype=float)
    mask = ~np.isnan(v) & (np.isfinite(rwm))
    if mask.sum() < 30: continue
    rho, p = spearmanr(rwm[mask], v[mask])
    rows.append({"clinical": name, "n": int(mask.sum()), "rho": rho, "p": p})
print(pd.DataFrame(rows).to_string(index=False))
pd.DataFrame(rows).to_csv(RW / d / "clinical_corr.csv", index=False)

# anchor-stratified rewiring tests (stage I-II vs III-IV; grade G1-2 vs G3-4)
src_all, dst_all, pw_e = [], [], []
pid = mem.loc[cols].idxmax(axis=1)
blocks = {}
for g, pw in pid.items():
    blocks.setdefault(pw, []).append(int(np.where(cols == g)[0][0]))
for pw, gs in blocks.items():
    for i in gs:
        for j in gs:
            if i != j:
                src_all.append(i); dst_all.append(j); pw_e.append(pw)
src_all = np.array(src_all); dst_all = np.array(dst_all)
for name, fn in [("stage", lambda v: np.where(v <= 2, 1, 0)),
                 ("grade", lambda v: np.where(v <= 2, 1, 0))]:
    v = (stage if name == "stage" else grade).reindex(pd.Index(sid)).to_numpy(dtype=float)
    lab = fn(v)
    hi = np.flatnonzero(lab == 1); lo = np.flatnonzero(lab == 0)
    if min(len(hi), len(lo)) < 40: 
        print(name, "insufficient", len(hi), len(lo)); continue
    res = pathway_level_test(alpha, hi, lo, src_all, dst_all, cols, mem.loc[cols])
    n_sig = int((res["q"] < 0.05).sum())
    print(name, "hi/lo:", len(hi), len(lo), "sig:", n_sig, "top:", res.iloc[0]["pathway"])
    res.to_csv(RW / d / ("anchor_%s_pathway_test.csv" % name), index=False)
print("KIRC_CLIN_DONE")

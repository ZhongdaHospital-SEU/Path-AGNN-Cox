# -*- coding: utf-8 -*-
"""P0-1/P0-2: cross-cohort overlap test + independent-anchor rewiring test.
A) Hypergeometric overlap of permutation-significant pathways across LUAD/BRCA/KIRC
   plus correlation of Cohen's d between cohorts.
B) Rewiring test stratified by anchors that do NOT use the model risk score:
   stage (I-II vs III-IV), TMB median, MKI67 expression median.
Outputs under results/rewiring/."""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, hypergeom, pearsonr

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
sys.path.insert(0, str(ROOT))
from benchmark.rewiring_analysis import bh_fdr, pathway_level_test
from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency

RW = ROOT / "results" / "rewiring"
DS = ["LUAD", "BRCA", "KIRC"]

# ---------------- A) cross-cohort overlap ----------------
eff = {d: pd.read_csv(RW / d / "pathway_effects.csv") for d in DS}
sig = {d: set(eff[d][eff[d]["perm_q"] < 0.05]["pathway"]) for d in DS}
pool = 53
rows = []
for a in DS:
    for b in DS:
        if a >= b:
            continue
        ov = len(sig[a] & sig[b])
        na, nb = len(sig[a]), len(sig[b])
        # P(overlap >= ov | random subsets of size na,nb from pool)
        p = hypergeom.sf(ov - 1, pool, na, nb)
        d = eff[a].merge(eff[b], on="pathway", suffixes=("_A", "_B"))
        rho, rp = spearmanr(d["cohen_d_A"], d["cohen_d_B"])
        pr, pp = pearsonr(d["cohen_d_A"], d["cohen_d_B"])
        rows.append({"cohort_A": a, "cohort_B": b, "n_sig_A": na, "n_sig_B": nb,
                     "overlap": ov, "hypergeom_p": p,
                     "spearman_d": rho, "spearman_p": rp,
                     "pearson_d": pr, "pearson_p": pp,
                     "n_pathways_both": len(d)})
ov = pd.DataFrame(rows)
ov.to_csv(RW / "cross_cohort_overlap.csv", index=False)
print(ov.to_string(index=False))

# ---------------- B) independent-anchor rewiring ----------------
tmb = pd.read_csv(ROOT / "data" / "processed" / "rewiring" / "tmb_by_sample.csv")
tmb = tmb.assign(sid=tmb["sample_id"].str[:12].str.upper()).drop_duplicates("sid")
tmb = tmb.set_index("sid")["tmb_nonsyn"]

gmt = load_gmt(str(ROOT / "data" / "pathways" / "kegg_cancer_core.gmt"))
summ_rows = []
for d in DS:
    train = pd.read_csv(ROOT / "data" / "processed" / d / "train.csv")
    X, _, _ = (lambda df: (df.drop(columns=["sample_id", "OS_time", "OS_event"]), None, None))(train)
    adj, mem, gene_order = build_pathway_adjacency(list(X.columns), gmt)
    cols = np.array([c for c in gene_order if c in X.columns])
    alpha = np.load(RW / d / "alpha.npy")
    risk = pd.read_csv(RW / d / "risk_scores.csv")
    sid = train["sample_id"].astype(str).str[:12].str.upper().to_numpy()
    assert len(sid) == alpha.shape[0], (d, len(sid), alpha.shape[0])
    # anchors
    anchors = {}
    clin = ROOT / "data" / "processed" / d / "clinical.csv"
    if clin.exists():
        c = pd.read_csv(clin)
        c = c.assign(sid=c["sample_id"].astype(str).str[:12].str.upper()).drop_duplicates("sid").set_index("sid")
        st = c["stage"].reindex(sid)
        anchors["stage"] = np.where(st.isna(), np.nan, np.where(st <= 2, 1, 0))  # 1=I-II, 0=III-IV
    tm = tmb.reindex(sid)
    med_tmb = tm.median()
    anchors["tmb"] = np.where(tm.isna(), np.nan, np.where(tm >= med_tmb, 1, 0))
    mk = X["MKI67"].to_numpy() if "MKI67" in X.columns else None
    if mk is not None:
        med_mk = np.median(mk)
        anchors["mki67"] = np.where(mk >= med_mk, 1, 0)
    for an, lab in anchors.items():
        hi = np.flatnonzero(lab == 1)
        lo = np.flatnonzero(lab == 0)
        if min(len(hi), len(lo)) < 40:
            summ_rows.append({"dataset": d, "anchor": an, "n_hi": len(hi), "n_lo": len(lo),
                              "n_sig_q005": np.nan, "top_pathway": np.nan})
            continue
        src_all, dst_all = np.array([], dtype=int), np.array([], dtype=int)
        # rebuild global edge list from pathway blocks (same layout as pipeline)
        blocks = {}
        for g, pw in mem.loc[cols].idxmax(axis=1).items():
            blocks.setdefault(pw, []).append(int(np.where(cols == g)[0][0]))
        for gs in blocks.values():
            for i in gs:
                for j in gs:
                    if i != j:
                        src_all = np.append(src_all, i); dst_all = np.append(dst_all, j)
        res = pathway_level_test(alpha, hi, lo, src_all, dst_all, cols, mem.loc[cols])
        n_sig = int((res["q"] < 0.05).sum())
        top = res.sort_values("p").iloc[0]["pathway"] if len(res) else np.nan
        summ_rows.append({"dataset": d, "anchor": an, "n_hi": int(len(hi)), "n_lo": int(len(lo)),
                          "n_sig_q005": n_sig, "top_pathway": top})
        res.to_csv(RW / d / ("anchor_%s_pathway_test.csv" % an), index=False)
        print(d, an, "hi/lo:", len(hi), len(lo), "sig:", n_sig, "top:", top, flush=True)
summ = pd.DataFrame(summ_rows)
summ.to_csv(RW / "anchor_rewiring_summary.csv", index=False)
print(summ.to_string(index=False))
print("ANCHOR_DONE")

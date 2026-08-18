# -*- coding: utf-8 -*-
"""Sensitivity: rewiring-magnitude definition robustness (L1 vs z-L1 vs 1-corr)."""
import sys, os
sys.path.insert(0, r"D:/TT paper/Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
root = r"D:/TT paper/Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
out_rows = []
for ds in ["LUAD", "BRCA", "IMvigor210"]:
    rdir = os.path.join(root, "results", "rewiring", ds)
    ap = os.path.join(rdir, "alpha.npy")
    rp = os.path.join(rdir, "risk_scores.csv")
    if not (os.path.exists(ap) and os.path.exists(rp)):
        print(ds, "skip (no alpha/risk)"); continue
    alpha = np.load(ap).astype(np.float64)          # (n, E)
    risk = pd.read_csv(rp)
    n = alpha.shape[0]
    mu = alpha.mean(axis=0)
    # def1: L1 distance (current)
    d1 = np.abs(alpha - mu).sum(axis=1)
    # def2: per-edge z-scored L1
    sd = alpha.std(axis=0); sd[sd == 0] = 1e-12
    z = (alpha - mu) / sd
    d2 = np.abs(z).sum(axis=1)
    # def3: correlation distance to cohort-mean profile
    c3 = np.array([1.0 - np.corrcoef(alpha[i], mu)[0, 1] for i in range(n)])
    defs = {"L1": d1, "z-L1": d2, "1-r": c3}
    # clinical anchors
    clin = None
    if ds != "IMvigor210":
        cp = os.path.join(root, "work", "results" if False else "", "results")
        # rewiring clinical files were built at data/processed/rewiring/clinical_<DS>.csv
        cp = os.path.join(root, "data", "processed", "rewiring", "clinical_%s.csv" % ds)
        if os.path.exists(cp):
            clin = pd.read_csv(cp)
    else:
        cp = os.path.join(root, "data", "processed", "IMvigor210", "clinical.csv")
        clin = pd.read_csv(cp)
    # risk scores order matches alpha rows (as used in rewiring analyses)
    risk = risk.iloc[:n].reset_index(drop=True)
    med = np.median(risk["risk_score"])
    hi, lo = np.where(risk["risk_score"] > med)[0], np.where(risk["risk_score"] <= med)[0]
    for name, dv in defs.items():
        rho_r, p_r = spearmanr(dv, risk["risk_score"])
        u, p_w = mannwhitneyu(dv[hi], dv[lo])
        row = {"dataset": ds, "definition": name,
               "rho_risk": rho_r, "P_risk": p_r,
               "wilcox_hi_vs_lo": p_w,
               "med_hi": float(np.median(dv[hi])), "med_lo": float(np.median(dv[lo]))}
        # anchors
        if clin is not None:
            sid = risk["sample_id"] if "sample_id" in risk.columns else None
            if sid is not None:
                c = clin.copy()
                c["sample_id"] = c["sample_id"].astype(str)
                if ds == "IMvigor210":
                    key = "sample_id"
                    r2 = pd.DataFrame({"sample_id": sid.astype(str), "dv": dv})
                else:
                    r2 = pd.DataFrame({"sample_id": sid.str[:15], "dv": dv})
                    c["sample_id"] = c["sample_id"].str[:15]
                m = r2.merge(c, on="sample_id", how="left")
                for anchor in ["ki67", "tmb"]:
                    if anchor in m.columns:
                        sub = m.dropna(subset=[anchor])
                        if len(sub) >= 20:
                            rho, p = spearmanr(sub["dv"], sub[anchor])
                            row["rho_%s" % anchor] = rho
                            row["P_%s" % anchor] = p
                            row["n_%s" % anchor] = len(sub)
        out_rows.append(row)
    print(ds, "done", flush=True)
out = pd.DataFrame(out_rows)
out.to_csv(os.path.join(root, "results", "rewiring", "sensitivity_magnitude.csv"), index=False)
print(out.to_string())
print("SENSITIVITY_DONE")
# -*- coding: utf-8 -*-
"""Recompute external-rewiring summaries: numeric high/low group, robust event counts,
exp-scale CIs (lifelines 0.30: hazard_ratios_ exp-scale, confidence_intervals_ log-scale)."""
import sys, os
sys.path.insert(0, r"D:/TT paper/Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
root = r"D:/TT paper/Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
ds = sys.argv[1] if len(sys.argv) > 1 else "LUAD"
outbase = os.path.join(root, "results", "rewiring_external", ds)
for fname in sorted(os.listdir(outbase)):
    if not fname.endswith(".csv") or fname.endswith("_summary.csv"):
        continue
    cohort = fname[:-4]
    res = pd.read_csv(os.path.join(outbase, fname))
    res["OS_time"] = pd.to_numeric(res["OS_time"], errors="coerce")
    res["OS_event"] = pd.to_numeric(res["OS_event"], errors="coerce")
    c = res[["OS_time", "OS_event", "rewiring_magnitude"]].rename(columns={"OS_time": "T", "OS_event": "E"}).dropna(subset=["T", "E", "rewiring_magnitude"])
    med = c["rewiring_magnitude"].median()
    c = c.assign(grp=(c["rewiring_magnitude"] >= med).astype(int))
    hr = np.nan; p = np.nan; lo = np.nan; hi = np.nan
    if c["E"].sum() >= 5 and len(c) >= 20:
        cf = CoxPHFitter().fit(c[["T", "E", "grp"]], duration_col="T", event_col="E")
        hr = float(cf.hazard_ratios_["grp"])
        ci = cf.confidence_intervals_.loc["grp"]
        lo = float(np.exp(ci[0])); hi = float(np.exp(ci[1]))
        p = float(cf.summary["p"]["grp"])
    summ = pd.DataFrame([{"cohort": cohort, "n": len(c), "events": int(c["E"].sum()),
                          "hr_high_vs_low": hr, "hr_lo": lo, "hr_hi": hi, "p": p,
                          "risk_rew_corr": float(np.corrcoef(res["risk_score"], res["rewiring_magnitude"])[0, 1])}])
    summ.to_csv(os.path.join(outbase, cohort + "_summary.csv"), index=False)
    print(cohort, "n", len(c), "events", int(c["E"].sum()), "HR", round(hr, 3), "P", round(p, 4), flush=True)
print("RECOMPUTE_DONE_" + ds, flush=True)
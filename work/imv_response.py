# -*- coding: utf-8 -*-
"""IMvigor210: rewiring magnitude vs anti-PD-L1 response and OS."""
import sys, os
sys.path.insert(0, r"D:/TT paper/Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
root = r"D:/TT paper/Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
rdir = os.path.join(root, "results", "rewiring", "IMvigor210")
alpha = np.load(os.path.join(rdir, "alpha.npy"))
mu = alpha.mean(axis=0)
rew = np.abs(alpha - mu).sum(axis=1)
risk = pd.read_csv(os.path.join(rdir, "risk_scores.csv"))
clin = pd.read_csv(os.path.join(root, "data", "processed", "IMvigor210", "clinical.csv"))
df = pd.DataFrame({"sample_id": risk["sample_id"], "rewiring_magnitude": rew,
                   "risk_score": risk["risk_score"]})
df = df.merge(clin, on="sample_id", how="left")
df["resp_bin"] = np.where(df["response"] == "CR/PR", 1,
                          np.where(df["response"] == "SD/PD", 0, np.nan))
out = {}
sub = df.dropna(subset=["resp_bin"])
out["n_responder"] = int((sub["resp_bin"] == 1).sum())
out["n_nonresponder"] = int((sub["resp_bin"] == 0).sum())
if len(sub) >= 10:
    hi = sub.loc[sub["resp_bin"] == 1, "rewiring_magnitude"]
    lo = sub.loc[sub["resp_bin"] == 0, "rewiring_magnitude"]
    u, p = mannwhitneyu(hi, lo, alternative="two-sided")
    out["wilcox_P"] = float(p)
    out["med_resp"] = float(hi.median())
    out["med_nonresp"] = float(lo.median())
tmb = df.dropna(subset=["tmb"])
if len(tmb) >= 10:
    rho, p = spearmanr(tmb["rewiring_magnitude"], tmb["tmb"])
    out["tmb_rho"] = float(rho); out["tmb_P"] = float(p); out["tmb_n"] = len(tmb)
# OS association: median split on rewiring magnitude
sub2 = df.dropna(subset=["os_months"])
med = sub2["rewiring_magnitude"].median()
sub2 = sub2.assign(grp=(sub2["rewiring_magnitude"] >= med).astype(int))
if len(sub2) >= 10 and sub2["os_event"].sum() >= 5:
    try:
        from lifelines import CoxPHFitter
        c = sub2[["os_months", "os_event", "grp"]].rename(columns={"os_months": "T", "os_event": "E"})
        cf = CoxPHFitter().fit(c, duration_col="T", event_col="E")
        out["os_hr"] = float(cf.hazard_ratios_["grp"])
        ci = cf.confidence_intervals_.loc["grp"]
        out["os_hr_lo"] = float(np.exp(ci[0])); out["os_hr_hi"] = float(np.exp(ci[1]))
        out["os_P"] = float(cf.summary["p"]["grp"])
    except Exception as e:
        out["os_err"] = str(e)
# Ki-67 replication (from rewiring clinical_corr, independent cohort anchor)
cc = os.path.join(rdir, "clinical_corr.csv")
if os.path.exists(cc):
    ccd = pd.read_csv(cc)
    k = ccd[ccd["clinical"] == "ki67"]
    if len(k):
        out["ki67_rho"] = float(k.iloc[0]["rho"]); out["ki67_P"] = float(k.iloc[0]["p"])
        out["ki67_n"] = int(k.iloc[0]["n"])
try:
    from lifelines.statistics import logrank_test
    t = sub2["os_months"].to_numpy(float); e = sub2["os_event"].to_numpy(float); g = sub2["grp"].to_numpy()
    lr = logrank_test(t[g == 1], t[g == 0], event_observed_A=e[g == 1], event_observed_B=e[g == 0])
    out["os_logrank_P"] = float(lr.p_value)
except Exception as ex:
    out["os_err"] = str(ex)
pd.Series(out).to_csv(os.path.join(rdir, "response_stats.csv"))
print(out)
print("RESPONSE_DONE")

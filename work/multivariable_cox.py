# -*- coding: utf-8 -*-
"""Multivariable Cox: risk score adjusted for stage and age (LUAD/BRCA).

Usage: python work/multivariable_cox.py LUAD
Output: results/rewiring/<DS>/multivariable_cox.csv
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import CoxPHFitter

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
DS = sys.argv[1]
proc = ROOT / "data" / "processed" / DS
rw = ROOT / "results" / "rewiring" / DS

def norm(df, col):
    df = df.copy()
    df[col] = df[col].str[:12]          # patient-level TCGA barcode
    df = df.drop_duplicates(subset=col, keep="first")
    return df

train = pd.read_csv(proc / "train.csv", usecols=["sample_id", "OS_time", "OS_event"])
clin = pd.read_csv(ROOT / "data" / "processed" / "rewiring" / f"clinical_{DS}.csv")
risk = pd.read_csv(rw / "risk_scores.csv")

train = norm(train, "sample_id")
clin = norm(clin, "sample_id")
risk = norm(risk, "sample_id")

df = train.merge(risk, on="sample_id", how="inner").merge(
    clin[["sample_id", "stage", "age"]], on="sample_id", how="left")
df = df.dropna(subset=["OS_time", "OS_event"])
df = df[df["OS_time"] > 0]
df["risk_z"] = (df["risk_score"] - df["risk_score"].mean()) / df["risk_score"].std()

rows = []
def fit(cols, label):
    sub = df.dropna(subset=cols)
    if len(sub) < 30 or sub["OS_event"].sum() < 10:
        return
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(sub[cols + ["OS_time", "OS_event"]], duration_col="OS_time",
            event_col="OS_event", fit_options={"step_size": 0.5, "max_steps": 200})
    ci = np.exp(cph.confidence_intervals_)
    for cov in cols:
        rows.append({"model": label, "covariate": cov,
                     "hr": float(cph.hazard_ratios_[cov]),
                     "ci_lower": float(ci.loc[cov, "95% lower-bound"]),
                     "ci_upper": float(ci.loc[cov, "95% upper-bound"]),
                     "p": float(cph.summary.loc[cov, "p"]),
                     "n": len(sub), "events": int(sub["OS_event"].sum())})

fit(["risk_z"], "univariable")
fit(["risk_z", "stage", "age"], "multivariable")

out = pd.DataFrame(rows)
out.to_csv(rw / "multivariable_cox.csv", index=False)
print(DS, "-> rows:", len(out))
print(out.to_string(index=False) if len(out) else "no rows")

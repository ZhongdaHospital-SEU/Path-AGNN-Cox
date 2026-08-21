# -*- coding: utf-8 -*-
"""DCA v3 with IPCW (censoring-corrected) net benefit, Vickers formula.
Horizons 1/3/5 y = 365/1095/1825 days. OS_time in DAYS."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
RW = ROOT / "results" / "rewiring"
HOR = [("1y", 365), ("3y", 1095), ("5y", 1825)]
THRS = np.arange(0.05, 0.56, 0.05)

def ipcw_weights(time, event, t):
    """G(u)=P(C>u) via KM on censoring indicator; weights for IPCW NB at horizon t."""
    cens = 1 - event  # 1 if censored
    kmc = KaplanMeierFitter().fit(time, event_observed=cens)
    G_t = kmc.predict(t)
    G_T = kmc.predict(time)
    w_tp = np.where(event & (time <= t), 1.0 / np.clip(G_T, 1e-6, None), 0.0)
    w_fp = np.where(time > t, 1.0 / np.clip(G_t, 1e-6, None), 0.0)
    return w_tp, w_fp

OUT = []
SUMM = []
for d in ["LUAD", "BRCA", "KIRC"]:
    df = pd.read_csv(ROOT / "data" / "processed" / d / "train.csv", usecols=["sample_id", "OS_time", "OS_event"])
    risk = pd.read_csv(RW / d / "risk_scores.csv")
    df["risk_z"] = df["sample_id"].map(risk.set_index("sample_id")["risk_score"].to_dict())
    clin = ROOT / "data" / "processed" / d / "clinical.csv"
    if clin.exists():
        c = pd.read_csv(clin)
        sid_u = df["sample_id"].astype(str).str[:12].str.upper()
        c2 = c.assign(sid=c["sample_id"].astype(str).str[:12].str.upper()).drop_duplicates("sid").set_index("sid")
        df["age"] = sid_u.map(c2["age"])
        if "stage_num" in c.columns:
            df["stage"] = sid_u.map(c2["stage_num"])
        else:
            df["stage"] = sid_u.map(c2["stage"])
        df["stage"] = pd.to_numeric(df["stage"], errors="coerce")
    df = df.dropna(subset=["risk_z", "age", "stage"]).copy()
    df["risk_z"] = (df["risk_z"] - df["risk_z"].mean()) / df["risk_z"].std()
    if len(df) < 100:
        print(d, "too few:", len(df)); continue
    res = {}
    for name, cols in [("clinical", ["age", "stage"]),
                       ("clinical+risk", ["age", "stage", "risk_z"]),
                       ("risk", ["risk_z"])]:
        sub = df[["OS_time", "OS_event"] + cols].copy()
        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(sub, duration_col="OS_time", event_col="OS_event")
        except Exception as e:
            print(d, name, "fit fail:", repr(e)); continue
        for label, t in HOR:
            surv = cph.predict_survival_function(sub, times=[t]).T.iloc[:, 0].to_numpy()
            p_risk = 1.0 - surv
            km = KaplanMeierFitter().fit(sub["OS_time"], sub["OS_event"])
            p_t = float(1 - km.predict(t))
            w_tp, w_fp = ipcw_weights(sub["OS_time"].to_numpy(), sub["OS_event"].to_numpy(), t)
            n = len(sub)
            nb = []
            for thr in THRS:
                treat = p_risk > thr
                tp = float(np.sum(w_tp * treat))
                fp = float(np.sum(w_fp * treat))
                nb.append(tp / n - fp / n * (p_t / max(1 - p_t, 1e-9)))
            res[(name, label)] = np.array(nb)
    for label, t in HOR:
        km = KaplanMeierFitter().fit(df["OS_time"], df["OS_event"])
        p_t = float(1 - km.predict(t))
        ref_all = p_t - (1 - p_t) * THRS / np.maximum(1 - THRS, 1e-9)
        for thr, v in zip(THRS, ref_all):
            OUT.append({"dataset": d, "model": "treat_all", "horizon": label, "threshold": thr, "net_benefit": v})
        if ("clinical", label) in res and ("clinical+risk", label) in res:
            nb_c = res[("clinical", label)]; nb_cr = res[("clinical+risk", label)]
            diff = nb_cr - nb_c
            best_i = int(np.argmax(diff))
            SUMM.append({"dataset": d, "horizon": label, "max_diff": diff.max(),
                         "thr_at_max": float(THRS[best_i]), "n_pos": int((diff > 0).sum()),
                         "n_thr": len(THRS), "clinical_max": nb_c.max(), "risk_max": nb_cr.max(),
                         "mean_diff": diff.mean()})
            for thr, v in zip(THRS, nb_c):
                OUT.append({"dataset": d, "model": "clinical", "horizon": label, "threshold": thr, "net_benefit": v})
            for thr, v in zip(THRS, nb_cr):
                OUT.append({"dataset": d, "model": "clinical+risk", "horizon": label, "threshold": thr, "net_benefit": v})
        if ("risk", label) in res:
            for thr, v in zip(THRS, res[("risk", label)]):
                OUT.append({"dataset": d, "model": "risk", "horizon": label, "threshold": thr, "net_benefit": v})
pd.DataFrame(OUT).to_csv(RW / "dca_results.csv", index=False)
pd.DataFrame(SUMM).to_csv(RW / "dca_summary.csv", index=False)
print(pd.DataFrame(SUMM).round(4).to_string(index=False))
print("DCA_V3_DONE rows:", len(OUT))

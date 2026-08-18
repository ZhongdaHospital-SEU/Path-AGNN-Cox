# -*- coding: utf-8 -*-
"""Calibration analysis: Path-AGNN-Cox vs Ridge-Cox, internal CV + external transfer.

Outputs results/calibration_results.csv with rows:
  dataset, setting (internal/external), cohort, model, n, events,
  slope, slope_ci_low, slope_ci_high, cal_mae
Usage: python work/run_calibration.py LUAD
"""
import sys, os
sys.path.insert(0, r"D:/TT paper/Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd, torch
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.model_selection import StratifiedKFold
from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency
from path_agnn_cox.data import load_survival_data, split_features, standardize
from path_agnn_cox.models import PathAGNNCox
from path_agnn_cox.train import train_model, predict_risk
from benchmark.dataset_manifest import load_benchmark_config
from baselines.cox_penalized import ridge_cox

ROOT = r"D:/TT paper/Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
ds = sys.argv[1] if len(sys.argv) > 1 else "LUAD"
CSV_OUT = os.path.join(ROOT, "results", "calibration_results.csv")
done = set()
if os.path.exists(CSV_OUT):
    for _, r in pd.read_csv(CSV_OUT).iterrows():
        done.add((r["dataset"], r["setting"], r["cohort"], r["model"]))
epochs_full = int(os.environ.get("CAL_EPOCHS", "80"))
cfg = load_benchmark_config()["models"]["path_agnn_cox"]
out_rows = []

def make_model(X):
    pathway_dict = load_gmt(os.path.join(ROOT, "data", "pathways", "kegg_cancer_core.gmt"))
    adj, mem, gene_order = build_pathway_adjacency(list(X.columns), pathway_dict)
    cols = np.array([c for c in gene_order if c in X.columns])
    ids = torch.tensor([list(mem.columns).index(mem.loc[c].idxmax()) for c in cols])
    adj_t = torch.tensor(adj[:len(cols), :len(cols)])
    m = PathAGNNCox(n_genes=len(cols), adj=adj_t, pathway_ids=ids,
                    hidden=cfg["hidden"], n_layers=cfg["n_layers"],
                    mlp_hidden=cfg["mlp_hidden"], dropout=cfg["dropout"])
    return m, cols

def calib_metrics(risk, time, event):
    z = (risk - risk.mean()) / (risk.std(ddof=0) + 1e-12)
    df = pd.DataFrame({"risk": z, "time": np.asarray(time, float), "event": np.asarray(event, int)})
    cph = CoxPHFitter(penalizer=1e-6)
    cph.fit(df, duration_col="time", event_col="event")
    ci = cph.confidence_intervals_.loc["risk"]
    slope = float(cph.params_["risk"]); lo = float(ci.iloc[0]); hi = float(ci.iloc[1])
    H0 = cph.baseline_cumulative_hazard_["baseline cumulative hazard"]
    lp = cph.params_["risk"] * z
    terts = pd.qcut(z, 3, labels=False, duplicates="drop")
    qs = np.quantile(time, [0.25, 0.5, 0.75])
    devs = []
    for tau in qs:
        h0v = H0.loc[H0.index <= tau]
        h0v = h0v.iloc[-1] if len(h0v) else float(H0.iloc[0])
        sp = np.exp(-h0v * np.exp(lp))
        for g in range(3):
            m = (terts == g) & np.isfinite(sp)
            if m.sum() < 5:
                continue
            k = KaplanMeierFitter().fit(time[m], event[m]).predict(tau)
            devs.append(abs(float(sp[m].mean()) - float(k)))
    return slope, lo, hi, (float(np.mean(devs)) if devs else np.nan)

def emit(dataset, setting, cohort, model, n, n_ev, risk, time, event):
    key = (dataset, setting, cohort, model)
    if key in done:
        print("skip existing", key, flush=True)
        return
    try:
        slope, lo, hi, mae = calib_metrics(risk, time, event)
    except Exception as e:
        print("  calib failed", cohort, model, e, flush=True)
        slope = lo = hi = mae = np.nan
    row = {"dataset": dataset, "setting": setting, "cohort": cohort, "model": model,
           "n": int(n), "events": int(n_ev),
           "slope": slope, "slope_ci_low": lo, "slope_ci_high": hi, "cal_mae": mae}
    out_rows.append(row)
    pd.DataFrame([row]).to_csv(CSV_OUT, index=False, mode="a", header=not os.path.exists(CSV_OUT))
    print(cohort, model, "slope %.2f (%.2f-%.2f) mae %.3f" % (slope, lo, hi, mae), flush=True)

# ---------- internal CV ----------
df = load_survival_data(os.path.join(ROOT, "data", "processed", ds, "train.csv"))
X, time, event = split_features(df)
time = np.asarray(time, float); event = np.asarray(event, int)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_a = np.zeros(len(df)); oof_r = np.zeros(len(df))
for fold, (tr, va) in enumerate(skf.split(X, event)):
    Xtr, Xva = X.iloc[tr], X.iloc[va]
    ttr, etr = time[tr], event[tr]; tva, eva = time[va], event[va]
    Xtr_a, Xva_a = standardize(Xtr, Xva)
    m, cols = make_model(Xtr_a)
    Xtr_n = Xtr_a[cols].to_numpy(dtype=float)
    Xva_n = Xva_a[cols].to_numpy(dtype=float)
    m = train_model(m, Xtr_n, ttr, etr, Xva_n, tva, eva,
                    epochs=cfg["epochs"], lr=cfg["lr"], batch_size=cfg["batch_size"],
                    l2=cfg["l2"], lambda_sparse=cfg["lambda_sparse"],
                    lambda_consist=cfg["lambda_consist"], patience=cfg["patience"], seed=0)
    oof_a[va] = predict_risk(m, Xva_n)
    r = ridge_cox(Xtr_a[cols].to_numpy(), ttr, etr, penalizer=0.1)
    oof_r[va] = r.predict_risk(Xva_a[cols].to_numpy())
    print(ds, "fold", fold, "done", flush=True)
emit(ds, "internal", ds, "path_agnn_cox", len(df), int(event.sum()), oof_a, time, event)
emit(ds, "internal", ds, "ridge_cox", len(df), int(event.sum()), oof_r, time, event)

# ---------- external transfer ----------
Xs, _ = standardize(X)
m_full, cols = make_model(Xs)
Xn = Xs[cols].to_numpy(dtype=float)
torch.manual_seed(0); np.random.seed(0)
m_full = train_model(m_full, Xn, time, event, epochs=epochs_full, lr=cfg["lr"],
                     batch_size=cfg["batch_size"], l2=cfg["l2"],
                     lambda_sparse=cfg["lambda_sparse"], lambda_consist=cfg["lambda_consist"], seed=0)
r_full = ridge_cox(Xs[cols].to_numpy(), time, event, penalizer=0.1)
extdir = os.path.join(ROOT, "data", "processed", ds, "external")
if os.path.isdir(extdir):
    for fname in sorted(os.listdir(extdir)):
        if not fname.endswith(".csv"):
            continue
        cohort = fname[:-4]
        ext = pd.read_csv(os.path.join(extdir, fname))
        Ex = ext.drop(columns=["sample_id", "OS_time", "OS_event"])
        Exm = Ex.reindex(columns=cols).fillna(0.0)
        _, Xe = standardize(Xs[cols], Exm)
        Xe = Xe.to_numpy(dtype=float)
        te = ext["OS_time"].to_numpy(float); ee = ext["OS_event"].to_numpy(int)
        m_full.eval()
        with torch.no_grad():
            ra = predict_risk(m_full, Xe)
        emit(ds, "external", cohort, "path_agnn_cox", len(ext), int(ee.sum()), ra, te, ee)
        rr = r_full.predict_risk(Xe)
        emit(ds, "external", cohort, "ridge_cox", len(ext), int(ee.sum()), rr, te, ee)

print("DONE", ds, "rows:", len(out_rows))
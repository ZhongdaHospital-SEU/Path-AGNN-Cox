# -*- coding: utf-8 -*-
"""P0-1: Calibration (type I error) and power simulation of the pathway-level
between-stratum rewiring test (Mann-Whitney on per-sample mean edge weight,
BH-FDR across pathways). Uses the real LUAD pathway block topology.

Outputs (results/simulation/):
  typeI_summary.csv      per-pathway marginal rejection at p<0.05 + FWER
  power_summary.csv      detection at q<0.05 vs injected effect (fraction of
                         pathway mean), plus matched-block-control percentile
  control_calibration.csv pooled percentile distribution under the null
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
sys.path.insert(0, str(ROOT))
from benchmark.rewiring_analysis import bh_fdr
from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency
from path_agnn_cox.data import load_survival_data, split_features

OUT = ROOT / "results" / "simulation"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- real LUAD topology ----------------
df = load_survival_data(str(ROOT / "data" / "processed" / "LUAD" / "train.csv"))
X, _, _ = split_features(df)
pathway_dict = load_gmt(str(ROOT / "data" / "pathways" / "kegg_cancer_core.gmt"))
adj, mem, gene_order = build_pathway_adjacency(list(X.columns), pathway_dict)
cols = np.array([c for c in gene_order if c in X.columns])
gene_to_pw = mem.loc[cols].idxmax(axis=1)
blocks = {}
for g, pw in gene_to_pw.items():
    blocks.setdefault(pw, []).append(int(np.where(cols == g)[0][0]))
blocks = {pw: np.asarray(v, dtype=int) for pw, v in blocks.items()}
pw_list = sorted(blocks.keys())
PW_EDGES = {pw: len(v) * (len(v) - 1) for pw, v in blocks.items()}

# global edge index: (src, dst, pw_label)
src_all, dst_all, pw_all = [], [], []
for pw in pw_list:
    g = blocks[pw]
    for i in g:
        for j in g:
            if i != j:
                src_all.append(i); dst_all.append(j); pw_all.append(pw)
src_all = np.asarray(src_all, dtype=np.int64)
dst_all = np.asarray(dst_all, dtype=np.int64)
pw_all = np.asarray(pw_all, dtype=object)
E = len(src_all)
pw_label_unique = np.unique(pw_all)
edge_sel = {pw: np.where(pw_all == pw)[0] for pw in pw_label_unique}
# per-block edge columns (for matched random k-subsets)
block_edges = {}
for pw in pw_list:
    g = blocks[pw]
    sel = np.where(pw_all == pw)[0]
    s_arr, d_arr = src_all[sel], dst_all[sel]
    block_edges[pw] = (sel, s_arr, d_arr)
print("topology:", len(pw_list), "pathways,", E, "edges", flush=True)

RNG_SEED = 20260820
B = 300  # simulated cohort size (150/150 split)

def sim_alpha(rng, inject=None, f=0.0, n_hi=None):
    """Log-normal attention weights (positive, mean ~0.008). Optionally scale
    edges of selected pathways by (1+f) in the hi stratum only."""
    alpha = 0.008 * np.exp(0.3 * rng.standard_normal((B, E)))
    if inject and f > 0:
        hi = np.arange(n_hi)
        for pw in inject:
            sel = edge_sel[pw]
            alpha[np.ix_(hi, sel)] *= (1.0 + f)
    return alpha

def pathway_test(alpha, hi, lo):
    rows = []
    for pw in pw_list:
        sel = edge_sel[pw]
        scores = alpha[:, sel].mean(axis=1)
        a_hi, a_lo = scores[hi], scores[lo]
        try:
            u, p = mannwhitneyu(a_hi, a_lo, alternative="two-sided")
        except ValueError:
            continue
        n1, n2 = len(a_hi), len(a_lo)
        mu = n1 * n2 / 2.0
        sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        z = (u - mu) / max(sd, 1e-12)
        d = a_hi.mean() - a_lo.mean()
        sp = np.sqrt(((n1 - 1) * a_hi.std(ddof=1) ** 2 + (n2 - 1) * a_lo.std(ddof=1) ** 2) / (n1 + n2 - 2.0))
        cd = d / sp if sp > 0 else 0.0
        rows.append({"pathway": pw, "z": float(z), "p": float(p), "d": float(d), "cohen_d": float(cd)})
    res = pd.DataFrame(rows)
    if len(res):
        res["q"] = bh_fdr(res["p"].to_numpy())
    return res

def block_control_percentile(alpha, hi, lo, pw, n_null=30, rng=None, res=None):
    """Percentile of the real pathway's |Cohen's d| among matched random
    k-subsets drawn from real blocks of size >= k (density-matched)."""
    k = len(blocks[pw])
    cand = [b for b in blocks.values() if len(b) >= k]
    if not cand:
        return np.nan
    if res is None:
        res = pathway_test(alpha, hi, lo)
    real_d = abs(float(res.loc[res["pathway"] == pw, "cohen_d"].iloc[0]))
    n1, n2 = len(hi), len(lo)
    null_ds = []
    tries = 0
    while len(null_ds) < n_null and tries < 2000:
        tries += 1
        b = cand[int(rng.integers(len(cand)))]
        bpw = gene_to_pw[cols[b[0]]]
        sel, s_arr, d_arr = block_edges[bpw]
        sub = np.sort(rng.choice(b, size=k, replace=False))
        m = np.isin(s_arr, sub) & np.isin(d_arr, sub) & (s_arr != d_arr)
        if int(m.sum()) < 5:
            continue
        sub_sel = sel[m]
        s_hi = alpha[np.ix_(hi, sub_sel)].mean(axis=1)
        s_lo = alpha[np.ix_(lo, sub_sel)].mean(axis=1)
        sp = np.sqrt(((n1 - 1) * s_hi.std(ddof=1) ** 2 + (n2 - 1) * s_lo.std(ddof=1) ** 2) / (n1 + n2 - 2.0))
        cd = (s_hi.mean() - s_lo.mean()) / sp if sp > 0 else 0.0
        null_ds.append(abs(cd))
    if len(null_ds) < 10:
        return np.nan
    return float(np.mean(real_d >= np.array(null_ds))) * 100.0

def run_typeI(rng, n_sim=200, n_perm_pw=None):
    marginal = {pw: 0 for pw in pw_list}
    n_sig_q = []
    n_paths = []
    for s in range(n_sim):
        alpha = sim_alpha(rng)
        perm = rng.permutation(B)
        hi, lo = perm[: B // 2], perm[B // 2:]
        res = pathway_test(alpha, hi, lo)
        for _, r in res.iterrows():
            if r["p"] < 0.05:
                marginal[r["pathway"]] += 1
        n_sig_q.append(int((res["q"] < 0.05).sum()))
        n_paths.append(len(res))
        if (s + 1) % 50 == 0:
            print("typeI", s + 1, "FWER so far:", np.mean(np.array(n_sig_q) > 0), flush=True)
    n_sim = float(n_sim)
    rows = [{"pathway": pw, "reject_p005": marginal[pw] / n_sim} for pw in pw_list]
    n_sig_q = np.asarray(n_sig_q)
    rows.append({"pathway": "ALL", "reject_p005": float((n_sig_q > 0).mean()),
                 "null_mean_sig_q": float(n_sig_q.mean()), "null_p95_sig_q": float(np.percentile(n_sig_q, 95))})
    return pd.DataFrame(rows)

def run_power(rng, inject, fracs, n_sim=60, n_ctl=30):
    rows = []
    for f in fracs:
        det1 = det2 = either = 0
        ctl_pct = []
        cohens = []
        for s in range(n_sim):
            alpha = sim_alpha(rng, inject=[inject[0], inject[1]], f=f, n_hi=B // 2)
            hi = np.arange(B // 2); lo = np.arange(B // 2, B)
            res = pathway_test(alpha, hi, lo)
            sig = set(res.loc[res["q"] < 0.05, "pathway"])
            det1 += int(inject[0] in sig); det2 += int(inject[1] in sig)
            either += int(bool(sig & set(inject)))
            cd = res.loc[res["pathway"].isin(inject), "cohen_d"].abs().mean()
            cohens.append(float(cd))
            pct = block_control_percentile(alpha, hi, lo, inject[0], n_null=n_ctl, rng=rng, res=res)
            if not np.isnan(pct):
                ctl_pct.append(pct)
        rows.append({
            "frac": f,
            "n_sim": n_sim,
            "power_pw1": det1 / n_sim,
            "power_pw2": det2 / n_sim,
            "power_either": either / n_sim,
            "mean_cohen_d_injected": float(np.mean(cohens)),
            "mean_ctl_pct": float(np.mean(ctl_pct)) if ctl_pct else np.nan,
        })
        print("power f=", f, "either=", either / n_sim, flush=True)
    return pd.DataFrame(rows)

def run_control_calibration(rng, n_sim=20, n_ctl=50):
    pcts = []
    for s in range(n_sim):
        alpha = sim_alpha(rng)
        perm = rng.permutation(B)
        hi, lo = perm[: B // 2], perm[B // 2:]
        for pw in pw_list:
            pct = block_control_percentile(alpha, hi, lo, pw, n_null=n_ctl, rng=rng)
            if not np.isnan(pct):
                pcts.append(pct)
        if (s + 1) % 5 == 0:
            print("calib", s + 1, flush=True)
    pcts = np.asarray(pcts)
    return pd.DataFrame([{
        "n_datasets": n_sim, "n_sets_per_pathway": n_ctl, "n_pathway_obs": len(pcts),
        "pct_mean": float(pcts.mean()), "pct_median": float(np.median(pcts)),
        "pct_p05": float(np.percentile(pcts, 5)), "pct_p95": float(np.percentile(pcts, 95)),
        "frac_below_5": float((pcts < 5).mean()), "frac_above_95": float((pcts > 95).mean()),
    }])

if __name__ == "__main__":
    rng = np.random.default_rng(RNG_SEED)
    t1 = run_typeI(rng, n_sim=200)
    t1.to_csv(OUT / "typeI_summary.csv", index=False)
    print("typeI done; FWER =", t1.iloc[-1]["reject_p005"], flush=True)

    pw1, pw2 = "Cell cycle", "p53 signaling pathway"
    fracs = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
    pw = run_power(rng, [pw1, pw2], fracs, n_sim=60, n_ctl=30)
    pw.to_csv(OUT / "power_summary.csv", index=False)
    print("power done", flush=True)

    cc = run_control_calibration(rng, n_sim=20, n_ctl=50)
    cc.to_csv(OUT / "control_calibration.csv", index=False)
    print("calibration done", cc.to_dict("records"), flush=True)
    print("SIM_ALL_DONE", flush=True)

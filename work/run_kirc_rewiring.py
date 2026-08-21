# -*- coding: utf-8 -*-
"""P2: full rewiring pipeline for a third cancer type (KIRC).

Replicates the LUAD/BRCA pipeline: train (epochs 80, seed 0, benchmark config),
per-pathway between-stratum tests with Cohen's d, 1000 label permutations,
BH-FDR, edge-matched (internal-edge-count matched) and density-matched
(block-subset) random gene-set controls, plus the cohort-level permutation test.

Outputs: results/rewiring/KIRC/{alpha.npy, risk_scores.csv, edges_meta.csv,
pathway_test.csv, pathway_effects.csv, permutation_test.csv}
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
sys.path.insert(0, str(ROOT))
import torch
from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency
from path_agnn_cox.data import load_survival_data, split_features, standardize
from path_agnn_cox.models import PathAGNNCox
from path_agnn_cox.train import train_model, predict_with_alpha
from benchmark.dataset_manifest import load_benchmark_config
from benchmark.rewiring_analysis import bh_fdr, pathway_level_test

DS = "KIRC"
EPOCHS = int(os.environ.get("KIRC_EPOCHS", "80"))
N_PERM = int(os.environ.get("KIRC_N_PERM", "1000"))
N_NULL = 200
OUT = ROOT / "results" / "rewiring" / DS
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
df = load_survival_data(str(ROOT / "data" / "processed" / DS / "train.csv"))
X, time_, event = split_features(df)
print(DS, "n =", len(df), "events =", int(event.sum()), flush=True)
pathway_dict = load_gmt(str(ROOT / "data" / "pathways" / "kegg_cancer_core.gmt"))
adj, mem, gene_order = build_pathway_adjacency(list(X.columns), pathway_dict)
cols = np.array([c for c in gene_order if c in X.columns])
Xs, _ = standardize(X[cols])
Xn = Xs.to_numpy(dtype=float)
cfg = load_benchmark_config()["models"]["path_agnn_cox"]
mcfg = dict(cfg)
mcfg["batch_size"] = int(os.environ.get("PATH_AGNN_BATCH_SIZE", mcfg["batch_size"]))
ids = torch.tensor([list(mem.columns).index(mem.loc[c].idxmax()) for c in cols])
adj_t = torch.tensor(adj[:len(cols), :len(cols)])
torch.manual_seed(0)
model = PathAGNNCox(n_genes=len(cols), adj=adj_t, pathway_ids=ids,
                    hidden=mcfg["hidden"], n_layers=mcfg["n_layers"],
                    mlp_hidden=mcfg["mlp_hidden"], dropout=mcfg["dropout"])
train_model(model, Xn, time_, event, epochs=EPOCHS, lr=mcfg["lr"],
            batch_size=mcfg["batch_size"], l2=mcfg["l2"],
            lambda_sparse=mcfg["lambda_sparse"], lambda_consist=mcfg["lambda_consist"])
torch.save(model.state_dict(), str(ROOT / "work" / "models" / "kirc_rewiring.pt"))
print("trained %.1f min" % ((time.time() - t0) / 60.0), flush=True)

risk, alpha, src, dst = predict_with_alpha(model, Xn)
np.save(OUT / "alpha.npy", alpha)
pd.DataFrame({"src_gene": cols[src], "dst_gene": cols[dst]}).to_csv(OUT / "edges_meta.csv", index=False)
pd.DataFrame({"sample_id": df["sample_id"], "risk_score": risk}).to_csv(OUT / "risk_scores.csv", index=False)
med = np.median(risk)
hi = np.where(risk > med)[0]; lo = np.where(risk <= med)[0]
n1, n2 = len(hi), len(lo)
print("hi/lo:", n1, n2, flush=True)

pw_test = pathway_level_test(alpha, hi, lo, src, dst, cols, mem.loc[cols])
pw_test.to_csv(OUT / "pathway_test.csv", index=False)
print("pathway tests:", len(pw_test), flush=True)

# ---- pathway_effects.csv: cohen_d + CI + mw_p + 1000-perm + both controls ----
edges = pd.read_csv(OUT / "edges_meta.csv")
src_g = edges["src_gene"].to_numpy(); dst_g = edges["dst_gene"].to_numpy()
gene_to_pw = mem.loc[cols].idxmax(axis=1)
pw_of_edge = gene_to_pw.reindex(src_g).to_numpy()
nonself = src_g != dst_g
gidx = {g: i for i, g in enumerate(cols)}
src_i = np.array([gidx[g] for g in src_g]); dst_i = np.array([gidx[g] for g in dst_g])
N = len(cols)

def cohen_ci(d, a, b):
    var = (a + b) / (a * b) + d * d / (2.0 * (a + b))
    se = np.sqrt(var)
    return d - 1.96 * se, d + 1.96 * se

rows = []
rng = np.random.default_rng(20260819)
for pw in pd.unique(pw_of_edge):
    e_sel = np.flatnonzero((pw_of_edge == pw) & nonself)
    if len(e_sel) < 5:
        continue
    scores = alpha[:, e_sel].mean(axis=1)
    a_hi, a_lo = scores[hi], scores[lo]
    m1, m2 = float(a_hi.mean()), float(a_lo.mean())
    s1, s2 = float(a_hi.std(ddof=1)), float(a_lo.std(ddof=1))
    sp = np.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2.0))
    d = m1 - m2
    dc = d / sp if sp > 0 else 0.0
    cilo, cihi = cohen_ci(dc, n1, n2)
    u, p_mw = mannwhitneyu(a_hi, a_lo, alternative="two-sided")
    y = np.concatenate([a_hi, a_lo])
    lbl = np.concatenate([np.ones(n1), np.zeros(n2)]).astype(bool)
    obs = abs(dc)
    cnt = 0
    for _ in range(N_PERM):
        perm = rng.permutation(lbl)
        m1p, m2p = y[perm].mean(), y[~perm].mean()
        s1p, s2p = y[perm].std(ddof=1), y[~perm].std(ddof=1)
        spp = np.sqrt(((n1 - 1) * s1p * s1p + (n2 - 1) * s2p * s2p) / (n1 + n2 - 2.0))
        dcp = (m1p - m2p) / spp if spp > 0 else 0.0
        if abs(dcp) >= obs:
            cnt += 1
    perm_p = (1 + cnt) / (N_PERM + 1.0)
    rows.append({"pathway": pw, "n_genes": int(gene_to_pw.eq(pw).sum()), "n_edges": len(e_sel),
                 "mean_hi": m1, "mean_lo": m2, "d": d, "cohen_d": dc,
                 "d_ci_lo": cilo, "d_ci_hi": cihi, "mw_p": p_mw, "perm_p": perm_p})
res = pd.DataFrame(rows)
res["perm_q"] = bh_fdr(res["perm_p"].to_numpy())
print("permutation tests done; sig q<0.05:", int((res["perm_q"] < 0.05).sum()), flush=True)

# edge-matched control (internal edge count matched, random gene sets)
neigh = {}
for s, d, ns in zip(src_g, dst_g, nonself):
    if ns:
        neigh.setdefault(s, set()).add(d)
genes_all = list(cols)
rng2 = np.random.default_rng(42)
null_rows = []
for _, r in res.iterrows():
    pw = r["pathway"]
    k = int(r["n_genes"])
    m_target = int(r["n_edges"])
    null_dc = []
    accepted = 0
    tries = 0
    while accepted < N_NULL and tries < 6000:
        tries += 1
        S = set(rng2.choice(genes_all, size=k, replace=False).tolist())
        m_s = sum(len(neigh.get(g, ())) for g in S) // 2
        if not (0.5 * m_target <= m_s <= 2.0 * m_target):
            continue
        member = np.zeros(N, dtype=bool)
        for g in S:
            member[gidx[g]] = True
        sel = np.flatnonzero(member[src_i] & member[dst_i] & nonself)
        if len(sel) < 5:
            continue
        sc = alpha[:, sel].mean(axis=1)
        a1, a2 = sc[hi], sc[lo]
        sp_ = np.sqrt(((n1 - 1) * a1.std(ddof=1) ** 2 + (n2 - 1) * a2.std(ddof=1) ** 2) / (n1 + n2 - 2.0))
        dc_ = (a1.mean() - a2.mean()) / sp_ if sp_ > 0 else 0.0
        null_dc.append(abs(dc_))
        accepted += 1
    null_dc = np.array(null_dc)
    real_d = abs(float(r["cohen_d"]))
    pct = float((np.sum(null_dc >= real_d) + 1) / (len(null_dc) + 1)) if len(null_dc) else np.nan
    null_rows.append({"pathway": pw, "n_null_edge_sets": len(null_dc),
                      "null_median_abs_d": float(np.median(null_dc)) if len(null_dc) else np.nan,
                      "null_pct": pct})
    if accepted % 100 == 0:
        print("edge-ctrl", pw, "accepted", accepted, flush=True)
ctrl = pd.DataFrame(null_rows)
res = res.merge(ctrl, on="pathway", how="left")
print("edge-matched control done", flush=True)

# block-matched control (density-matched k-subsets from real blocks)
blocks = {}
for g, pw in gene_to_pw.items():
    blocks.setdefault(pw, []).append(gidx[g])
blocks = {pw: np.asarray(v, dtype=int) for pw, v in blocks.items()}
rng3 = np.random.default_rng(20260820)
block_rows = []
for _, r in res.iterrows():
    pw = r["pathway"]
    k = int(r["n_genes"])
    real_d = abs(float(r["cohen_d"]))
    cand = [b for b in blocks.values() if len(b) >= k]
    null_dc = []
    accepted = 0
    tries = 0
    while accepted < N_NULL and tries < 20000:
        tries += 1
        blk = cand[int(rng3.integers(len(cand)))]
        bpw = [p for p, b in blocks.items() if np.array_equal(blk, b)][0]
        S = rng3.choice(blk, size=k, replace=False)
        member = np.zeros(N, dtype=bool)
        member[S] = True
        sel = np.flatnonzero(member[src_i] & member[dst_i] & nonself & (pw_of_edge == bpw))
        if len(sel) < 5:
            continue
        sc = alpha[:, sel].mean(axis=1)
        a1, a2 = sc[hi], sc[lo]
        sp_ = np.sqrt(((n1 - 1) * a1.std(ddof=1) ** 2 + (n2 - 1) * a2.std(ddof=1) ** 2) / (n1 + n2 - 2.0))
        dc_ = (a1.mean() - a2.mean()) / sp_ if sp_ > 0 else 0.0
        null_dc.append(abs(dc_))
        accepted += 1
    null_dc = np.array(null_dc)
    pct = float((np.sum(null_dc >= real_d) + 1) / (len(null_dc) + 1)) if len(null_dc) else np.nan
    block_rows.append({"pathway": pw, "n_null_block_sets": len(null_dc),
                       "block_null_median_abs_d": float(np.median(null_dc)) if len(null_dc) else np.nan,
                       "block_null_pct": pct})
bctrl = pd.DataFrame(block_rows)
res = res.merge(bctrl, on="pathway", how="left")
print("block-matched control done", flush=True)
res.to_csv(OUT / "pathway_effects.csv", index=False)

# cohort-level permutation test
rng4 = np.random.default_rng(0)
N_PERM2 = 200
obs_n = int((res["perm_q"] < 0.05).sum())
n_sig = []
for k in range(N_PERM2):
    perm = rng4.permutation(len(risk))
    r_perm = risk[perm]
    medp = np.median(r_perm)
    hi_p = np.where(r_perm > medp)[0]; lo_p = np.where(r_perm <= medp)[0]
    r2 = pathway_level_test(alpha, hi_p, lo_p, src, dst, cols, mem.loc[cols])
    n_sig.append(int((r2["q"] < 0.05).sum()) if len(r2) else 0)
n_sig = np.array(n_sig)
pval = (1 + int((n_sig >= obs_n).sum())) / (1 + N_PERM2)
pd.DataFrame([{"observed_sig": obs_n, "n_perm": N_PERM2, "null_mean_sig": float(n_sig.mean()),
               "null_max_sig": int(n_sig.max()), "perm_p": pval}]).to_csv(OUT / "permutation_test.csv", index=False)
print("cohort perm: observed", obs_n, "null mean", round(float(n_sig.mean()), 2), "perm_p", round(pval, 4), flush=True)
print("KIRC_REWIRING_DONE", flush=True)

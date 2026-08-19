# -*- coding: utf-8 -*-
"""Pathway-specificity correction (Route A): effect-size + permutation-calibrated
per-pathway tests and size-matched random gene-set controls, computed from the
saved per-sample attention weights (results/rewiring/<DS>/alpha.npy) without
retraining.

Outputs:
  results/rewiring/<DS>/pathway_effects.csv
  results/rewiring/<DS>/matched_control_summary.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
sys.path.insert(0, str(ROOT))

from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency
from path_agnn_cox.data import load_survival_data, split_features
from scipy.stats import mannwhitneyu, hypergeom


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    q = pvals[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def cohen_d_ci(d, n1, n2):
    var = (n1 + n2) / (n1 * n2) + d * d / (2.0 * (n1 + n2))
    se = np.sqrt(var)
    return d - 1.96 * se, d + 1.96 * se


def analyze(ds):
    out_dir = ROOT / "results" / "rewiring" / ds
    df = load_survival_data(str(ROOT / "data" / "processed" / ds / "train.csv"))
    X, time, event = split_features(df)
    pathway_dict = load_gmt(str(ROOT / "data" / "pathways" / "kegg_cancer_core.gmt"))
    adj, mem, gene_order = build_pathway_adjacency(list(X.columns), pathway_dict)
    cols = np.array([c for c in gene_order if c in X.columns])
    gene_to_pw = mem.loc[cols].idxmax(axis=1)

    alpha = np.load(out_dir / "alpha.npy")
    edges = pd.read_csv(out_dir / "edges_meta.csv")
    risk = pd.read_csv(out_dir / "risk_scores.csv")["risk_score"].to_numpy()
    assert alpha.shape[0] == len(risk) == len(df)
    assert alpha.shape[1] == len(edges)
    src = edges["src_gene"].to_numpy()
    dst = edges["dst_gene"].to_numpy()
    pw_of_edge = gene_to_pw.reindex(src).to_numpy()
    assert not pd.isna(pw_of_edge).any()
    nonself = src != dst

    med = np.median(risk)
    hi = np.where(risk > med)[0]
    lo = np.where(risk <= med)[0]
    n1, n2 = len(hi), len(lo)

    gidx = {g: i for i, g in enumerate(cols)}
    src_i = np.array([gidx[g] for g in src])
    dst_i = np.array([gidx[g] for g in dst])
    N = len(cols)
    neigh = {}
    for s, d, ns in zip(src, dst, nonself):
        if ns:
            neigh.setdefault(s, set()).add(d)

    rows = []
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
        cilo, cihi = cohen_d_ci(dc, n1, n2)
        u, p_mw = mannwhitneyu(a_hi, a_lo, alternative="two-sided")
        rng = np.random.default_rng(20260819)
        y = np.concatenate([a_hi, a_lo])
        lbl = np.concatenate([np.ones(n1), np.zeros(n2)]).astype(bool)
        n_perm = 1000
        cnt = 0
        obs = abs(dc)
        for _ in range(n_perm):
            perm = rng.permutation(lbl)
            m1p = y[perm].mean()
            m2p = y[~perm].mean()
            s1p = y[perm].std(ddof=1)
            s2p = y[~perm].std(ddof=1)
            spp = np.sqrt(((n1 - 1) * s1p * s1p + (n2 - 1) * s2p * s2p) / (n1 + n2 - 2.0))
            dcp = (m1p - m2p) / spp if spp > 0 else 0.0
            if abs(dcp) >= obs:
                cnt += 1
        perm_p = (1 + cnt) / (n_perm + 1.0)
        rows.append({"pathway": pw, "n_genes": int(gene_to_pw.eq(pw).sum()),
                     "n_edges": len(e_sel), "mean_hi": m1, "mean_lo": m2,
                     "d": d, "cohen_d": dc, "d_ci_lo": cilo, "d_ci_hi": cihi,
                     "mw_p": p_mw, "perm_p": perm_p})
    res = pd.DataFrame(rows)
    res["perm_q"] = bh_fdr(res["perm_p"].to_numpy())

    rng2 = np.random.default_rng(42)
    n_null = 200
    summary = []
    genes_all = list(cols)
    for _, r in res.iterrows():
        pw = r["pathway"]
        genes_p = list(gene_to_pw[gene_to_pw.eq(pw)].index)
        k = len(genes_p)
        m_target = int(r["n_edges"])
        null_dc = []
        accepted = 0
        tries = 0
        while accepted < n_null and tries < 6000:
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
        pct = float((np.sum(null_dc >= abs(r["cohen_d"])) + 1) / (len(null_dc) + 1)) if len(null_dc) else np.nan
        summary.append({"pathway": pw, "n_null_sets": accepted,
                        "null_median_abs_d": float(np.median(null_dc)) if len(null_dc) else np.nan,
                        "null_p95": float(np.percentile(null_dc, 95)) if len(null_dc) else np.nan,
                        "null_pct": pct})
    ctrl = pd.DataFrame(summary)
    merged = res.merge(ctrl, on="pathway")
    merged.to_csv(out_dir / "pathway_effects.csv", index=False)

    n_path = len(merged)
    n_sig_perm = int((merged["perm_q"] < 0.05).sum())
    n_exceed = int((merged["null_pct"] >= 0.95).sum())
    med_pct = float(merged["null_pct"].median())
    print(f"== {ds}: pathways={n_path}, perm_q<0.05: {n_sig_perm}, "
          f"null_pct>=0.95: {n_exceed} (expected {0.05 * n_path:.2f}), median null_pct={med_pct:.3f}", flush=True)
    known_file = ROOT / "data" / "pathways" / "luad_known_pathways.txt"
    if known_file.exists():
        known = [ln.strip() for ln in known_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        n_known = len(set(known) & set(merged["pathway"]))
        top = set(merged.sort_values("perm_p").head(20)["pathway"])
        hits = len(top & set(known))
        p_enr = hypergeom.sf(hits - 1, n_path, n_known, min(20, n_path)) if n_known > 0 else np.nan
        print(f"   known-pathway enrichment (top-20 by perm_p): hits={hits}, known_total={n_known}, P={p_enr:.3f}", flush=True)


if __name__ == "__main__":
    for ds in ["LUAD", "BRCA"]:
        analyze(ds)


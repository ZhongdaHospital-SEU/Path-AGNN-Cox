"""Rewiring analysis: prove that learned sample-specific edge weights are
biologically meaningful (H1-H5 from rewiring_analysis_plan.md).

Usage:
    python -m benchmark.rewiring_analysis --dataset LUAD \
        --train-csv data/processed/LUAD/train.csv \
        --gmt data/pathways/kegg_cancer_core.gmt \
        --known-pathways-file data/pathways/luad_known_pathways.txt \
        --clinical-csv data/processed/LUAD/clinical.csv \
        --out results/rewiring/LUAD
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency
from path_agnn_cox.data import load_survival_data, split_features, standardize
from path_agnn_cox.models import PathAGNNCox
from path_agnn_cox.train import train_model, predict_with_alpha
from path_agnn_cox.evaluate import c_index
from benchmark.dataset_manifest import load_benchmark_config


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def edge_diff_test(alpha: np.ndarray, hi_idx: np.ndarray, lo_idx: np.ndarray,
                   src: np.ndarray, dst: np.ndarray,
                   gene_order: np.ndarray, mem: pd.DataFrame,
                   exclude_self: bool = True):
    """H1: per-edge weight differences between risk groups (sparse edge list).

    alpha: (B, E) sample-specific attention weights; src/dst: (E,) node ids.
    """
    from scipy.stats import mannwhitneyu
    pid = mem.idxmax(axis=1)
    rows = []
    for e in range(len(src)):
        i, j = int(src[e]), int(dst[e])
        if exclude_self and i == j:
            continue
        a_hi, a_lo = alpha[hi_idx, e], alpha[lo_idx, e]
        if a_hi.std() == 0 and a_lo.std() == 0:
            continue
        try:
            _, p = mannwhitneyu(a_hi, a_lo, alternative="two-sided")
        except ValueError:
            continue
        rows.append({"gene_i": gene_order[i], "gene_j": gene_order[j],
                     "pathway": pid.iloc[i], "mean_hi": float(a_hi.mean()),
                     "mean_lo": float(a_lo.mean()),
                     "d": float(a_hi.mean() - a_lo.mean()), "p": float(p)})
    res = pd.DataFrame(rows)
    if len(res):
        res["q"] = bh_fdr(res["p"].to_numpy())
    return res


def pathway_level_test(alpha: np.ndarray, hi_idx: np.ndarray, lo_idx: np.ndarray,
                       src: np.ndarray, dst: np.ndarray,
                       gene_order: np.ndarray, mem: pd.DataFrame):
    """H1': pathway-level per-sample rewiring score and Mann-Whitney test.

    For each pathway, per-sample mean edge attention over its (non-self)
    edges; compares high- vs low-risk groups. Returns a ranked table with
    BH-FDR q-values and a standardized U-statistic z for enrichment ranking.
    """
    from scipy.stats import mannwhitneyu, norm
    pid = mem.idxmax(axis=1)
    pw_of_edge = pid.iloc[src].to_numpy()          # pathway label per edge
    mask = src != dst
    rows = []
    for pw in pd.unique(pw_of_edge):
        e_sel = np.where((pw_of_edge == pw) & mask)[0]
        if len(e_sel) < 5:
            continue
        scores = alpha[:, e_sel].mean(axis=1)      # per-sample pathway score
        a_hi, a_lo = scores[hi_idx], scores[lo_idx]
        try:
            u, p = mannwhitneyu(a_hi, a_lo, alternative="two-sided")
        except ValueError:
            continue
        n1, n2 = len(a_hi), len(a_lo)
        mu = n1 * n2 / 2.0
        sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        z = (u - mu) / max(sd, 1e-12)
        d = a_hi.mean() - a_lo.mean()
        rows.append({"pathway": pw, "n_edges": len(e_sel),
                     "mean_hi": float(a_hi.mean()), "mean_lo": float(a_lo.mean()),
                     "d": float(d), "z": float(z), "p": float(p)})
    res = pd.DataFrame(rows)
    if len(res):
        res["q"] = bh_fdr(res["p"].to_numpy())
    return res.sort_values("z", key=np.abs, ascending=False).reset_index(drop=True)


def pathway_aggregate(diff: pd.DataFrame) -> pd.DataFrame:
    """Pathway-level rewiring score from per-edge tests (legacy H1 output)."""
    if not len(diff):
        return pd.DataFrame()
    agg = diff.groupby("pathway").agg(
        n_edges=("d", "size"),
        mean_abs_d=("d", lambda x: np.abs(x).mean()),
        n_sig=("q", lambda x: int((x < 0.05).sum())),
        mean_d=("d", "mean")).reset_index()
    agg["frac_sig"] = agg["n_sig"] / agg["n_edges"].clip(lower=1)
    return agg.sort_values("mean_abs_d", ascending=False)


def enrichment(ranked: pd.DataFrame, known_pathways: list[str], top_k: int = 20):
    """H2: hypergeometric enrichment of known driver pathways in top-K rewired."""
    from scipy.stats import hypergeom
    n_total = len(ranked)
    n_known_total = len(set(known_pathways) & set(ranked["pathway"]))
    top = set(ranked.head(top_k)["pathway"])
    n_hit = len(top & set(known_pathways))
    if n_known_total == 0 or (n_top := min(top_k, n_total)) == 0:
        return {"hits": n_hit, "top_k": top_k, "p": np.nan}
    p = hypergeom.sf(n_hit - 1, n_total, n_known_total, n_top)
    return {"hits": int(n_hit), "top_k": int(n_top), "known_in_agg": int(n_known_total),
            "p": float(p), "hit_pathways": sorted(top & set(known_pathways))}


def rewiring_vs_clinical(alpha: np.ndarray, clin: pd.DataFrame,
                         clinical_cols=("stage", "grade", "ki67", "tmb", "purity")):
    """H3: rewiring magnitude (L1 distance to mean weight) vs clinical indicators."""
    from scipy.stats import spearmanr
    mu = alpha.mean(axis=0)
    rew = np.abs(alpha - mu).sum(axis=1)               # (n,) over sparse edges
    rows = []
    for col in clinical_cols:
        if col not in clin.columns:
            continue
        vals = pd.to_numeric(clin[col], errors="coerce")
        mask = vals.notna()
        if mask.sum() < 10:
            continue
        rho, p = spearmanr(rew[mask], vals[mask])
        rows.append({"clinical": col, "n": int(mask.sum()), "rho": rho, "p": p})
    return pd.DataFrame(rows), rew


def static_null(alpha_static: np.ndarray, src: np.ndarray, dst: np.ndarray):
    """H4: static model edge weights have zero sample-level variance."""
    var_sum = 0.0
    n_edges = 0
    for e in range(len(src)):
        if src[e] == dst[e]:
            continue
        var_sum += float(alpha_static[:, e].var())
        n_edges += 1
    return {"n_edges": n_edges, "total_var": float(var_sum),
            "mean_edge_var": float(var_sum / max(n_edges, 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--gmt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--known-pathways", nargs="*", default=[])
    ap.add_argument("--known-pathways-file", default=None,
                    help="text file with one known pathway name per line")
    ap.add_argument("--clinical-csv", default=None,
                    help="optional clinical table (sample_id + stage/grade/...) for H3")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_benchmark_config()
    mcfg = dict(cfg["models"]["path_agnn_cox"])
    import os as _os
    mcfg["batch_size"] = int(_os.environ.get("PATH_AGNN_BATCH_SIZE", mcfg["batch_size"]))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_survival_data(args.train_csv)
    X, time, event = split_features(df)
    pathway_dict = load_gmt(args.gmt)
    adj, mem, gene_order = build_pathway_adjacency(list(X.columns), pathway_dict)
    cols = np.array([c for c in gene_order if c in X.columns])
    Xs, _ = standardize(X[cols])
    Xn = Xs.to_numpy(dtype=float)

    import torch
    ids = torch.tensor([list(mem.columns).index(mem.loc[c].idxmax()) for c in cols])
    adj_t = torch.tensor(adj[:len(cols), :len(cols)])

    # train adaptive model
    torch.manual_seed(args.seed)
    model = PathAGNNCox(n_genes=len(cols), adj=adj_t, pathway_ids=ids,
                        hidden=mcfg["hidden"], n_layers=mcfg["n_layers"],
                        mlp_hidden=mcfg["mlp_hidden"], dropout=mcfg["dropout"])
    train_model(model, Xn, time, event, epochs=args.epochs, lr=mcfg["lr"],
                batch_size=mcfg["batch_size"], l2=mcfg["l2"],
                lambda_sparse=mcfg["lambda_sparse"], lambda_consist=mcfg["lambda_consist"])
    risk, alpha, src, dst = predict_with_alpha(model, Xn)

    # save model outputs for downstream / supplementary analyses
    np.save(out_dir / "alpha.npy", alpha)
    pd.DataFrame({"src_gene": cols[src], "dst_gene": cols[dst]}).to_csv(
        out_dir / "edges_meta.csv", index=False)
    pd.DataFrame({"sample_id": df["sample_id"], "risk_score": risk}).to_csv(
        out_dir / "risk_scores.csv", index=False)

    # risk groups (median split)
    med = np.median(risk)
    hi, lo = np.where(risk > med)[0], np.where(risk <= med)[0]

    # H1 per-edge tests + pathway aggregation (legacy, heavy on real cohorts)
    diff = edge_diff_test(alpha, hi, lo, src, dst, cols, mem.loc[cols])
    diff.to_csv(out_dir / "edge_diff.csv", index=False)
    agg = pathway_aggregate(diff)
    agg.to_csv(out_dir / "pathway_rewiring.csv", index=False)

    # H1' pathway-level per-sample tests (primary ranking for the paper)
    pw_test = pathway_level_test(alpha, hi, lo, src, dst, cols, mem.loc[cols])
    pw_test.to_csv(out_dir / "pathway_test.csv", index=False)
    print(f"H1' pathway tests: {len(pw_test)}, significant (q<0.05): "
          f"{(pw_test['q'] < 0.05).sum() if len(pw_test) else 0}")

    # H2 enrichment on the pathway-level |z| ranking
    known = list(args.known_pathways)
    if args.known_pathways_file and Path(args.known_pathways_file).exists():
        known += [ln.strip() for ln in
                  Path(args.known_pathways_file).read_text(encoding="utf-8").splitlines()
                  if ln.strip()]
    enrich = enrichment(pw_test, known) if len(pw_test) else {"p": np.nan}
    pd.Series(enrich).to_csv(out_dir / "enrichment.csv")

    # H3 clinical correlation (stage / age / ki67 proxy from expression)
    # clinical tables use 12-char TCGA barcodes; align to the train cohort order
    clin_df = pd.read_csv(args.clinical_csv) if args.clinical_csv and Path(args.clinical_csv).exists() else None
    if clin_df is not None and len(clin_df):
        sid = df["sample_id"].astype(str).str[:12].str.upper()
        cid = clin_df["sample_id"].astype(str).str[:12].str.upper()
        clin_df = clin_df.assign(_cid=cid).drop_duplicates(subset="_cid", keep="first")
        clin_df = clin_df.set_index("_cid").reindex(sid)
        clin_df = clin_df.drop(columns=["sample_id"], errors="ignore").reset_index()
        clin_df["sample_id"] = sid.to_numpy()
    if clin_df is not None and len(clin_df) == len(df) and "MKI67" in X.columns:
        clin_df["ki67"] = X["MKI67"].to_numpy()
    clin_res, rew = rewiring_vs_clinical(alpha, clin_df) if clin_df is not None else (pd.DataFrame(), np.zeros(len(df)))
    if len(clin_res):
        clin_res.to_csv(out_dir / "clinical_corr.csv", index=False)

    # H4 static negative control
    model_static = PathAGNNCox(n_genes=len(cols), adj=adj_t, pathway_ids=ids,
                               hidden=mcfg["hidden"], n_layers=mcfg["n_layers"],
                               mlp_hidden=mcfg["mlp_hidden"], dropout=mcfg["dropout"],
                               use_adaptive=False)
    train_model(model_static, Xn, time, event, epochs=args.epochs, lr=mcfg["lr"],
                batch_size=mcfg["batch_size"], l2=mcfg["l2"],
                lambda_sparse=0.0, lambda_consist=0.0)
    _, alpha_static, src_s, dst_s = predict_with_alpha(model_static, Xn)
    null = static_null(alpha_static, src_s, dst_s)
    pd.Series(null).to_csv(out_dir / "static_null.csv")

    # summary print
    ci = c_index(risk, time, event)
    print(f"== {args.dataset}: n={len(df)}, genes={len(cols)}, C-index={ci:.3f}")
    print(f"H1 edges tested: {len(diff)}, significant (q<0.05): "
          f"{(diff['q'] < 0.05).sum() if len(diff) else 0}")
    print(f"H1' top-5 rewired pathways:\n{pw_test.head(5)[['pathway','n_edges','d','z','q']].round(4) if len(pw_test) else 'NA'}")
    print(f"H2 enrichment: {enrich}")
    print(f"H4 static null: {null}")
    if len(clin_res):
        print(f"H3 clinical corr:\n{clin_res.round(3)}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Random-gene-set control for the GDSC2 validation (Plan A).
For each significant pathway, N=50 equal-size random gene sets are run through the
identical TCGA-template -> GDSC2-projection -> IC50-association pipeline.
"""
import csv, io
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
GDSC_IC50 = ROOT / "work/pkg/GDSC2/DataFiles/DataFiles/GLDS/GDSCv2/complete_matrix_output GDSCv2.txt"
GDSC_EXPR = ROOT / "work/gdsc2_expr.csv"
GMT = ROOT / "data/pathways/kegg_cancer_core.gmt"
TCGA_TRAIN = ROOT / "data/processed/BRCA/train.csv"
RISK = ROOT / "results/rewiring/BRCA/risk_scores.csv"
PTEST = ROOT / "results/rewiring/BRCA/pathway_test.csv"
OUT = ROOT / "results/gdsc"
N_RAND = 50
rng = np.random.default_rng(20260822)

CURATED = {
    "MK-1775": "MK-1775_1179", "Paclitaxel": "Paclitaxel_1080", "Gefitinib": "Gefitinib_1010",
    "Gemcitabine": "Gemcitabine_1190", "5-Fluorouracil": "5-Fluorouracil_1073",
    "Palbociclib": "Palbociclib_1054", "Docetaxel": "Docetaxel_1007", "Cisplatin": "Cisplatin_1005",
    "AZD7762": "AZD7762_1022", "AZD6738": "AZD6738_1917", "Talazoparib": "Talazoparib_1259",
    "Niraparib": "Niraparib_1177", "Olaparib": "Olaparib_1017", "Ribociclib": "Ribociclib_1632",
    "Trametinib": "Trametinib_1372", "Erlotinib": "Erlotinib_1168", "Selumetinib": "Selumetinib_1736",
}

def load_ic50():
    with open(GDSC_IC50, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter=" ", quotechar='"'))
    df = pd.DataFrame({r[0]: [float(v) for v in r[1:]] for r in rows[1:]}).T
    df.columns = rows[0]
    return df

def load_gmt():
    gmt = {}
    for ln in io.open(GMT, encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        gmt[p[0]] = p[2:]
    return gmt

def zscore_rows(X):
    mu = X.mean(axis=0); sd = X.std(axis=0, ddof=1); sd[sd == 0] = 1.0
    return (X - mu) / sd

def corr_matrix(X):
    Z = zscore_rows(X)
    return np.nan_to_num(Z.T @ Z / (Z.shape[0] - 1), nan=0.0)

ic50 = load_ic50()[[CURATED[k] for k in CURATED]]
ic50.columns = list(CURATED)
expr = pd.read_csv(GDSC_EXPR, index_col=0)
expr_z = pd.DataFrame(zscore_rows(expr.T).T, index=expr.index, columns=expr.columns)
gmt = load_gmt()
ptest = pd.read_csv(PTEST)
sig_pw = ptest[ptest["q"] < 0.05]["pathway"].tolist()

with open(TCGA_TRAIN, encoding="utf-8") as f:
    hdr = set(f.readline().rstrip("\n").split(","))
keep = [g for g in sorted(set(g for pw in sig_pw for g in gmt[pw])) if g in hdr]
tcga = pd.read_csv(TCGA_TRAIN, usecols=["sample_id"] + keep).merge(pd.read_csv(RISK), on="sample_id", how="inner")
med = tcga["risk_score"].median()
hi = tcga[tcga["risk_score"] >= med].drop(columns=["sample_id", "risk_score"])
lo = tcga[tcga["risk_score"] < med].drop(columns=["sample_id", "risk_score"])
print("TCGA:", len(tcga), "keep genes:", len(keep))

# GDSC cell matrix (cells x genes), z-scored
gdsc_cells = expr_z.T
pool = [g for g in keep if g in gdsc_cells.columns]
print("pool genes:", len(pool))
gdsc_arr = gdsc_cells[pool].to_numpy()
pool_idx = {g: i for i, g in enumerate(pool)}
drug_cols = {d: ic50[d].to_numpy() for d in ic50.columns}

def run_gene_set(genes):
    """Return dict drug -> (rho, P) for one gene set through the full pipeline."""
    genes = [g for g in genes if g in hdr and g in gdsc_cells.columns]
    if len(genes) < 3:
        return None
    Ch = corr_matrix(hi[genes].to_numpy(dtype=float))
    Cl = corr_matrix(lo[genes].to_numpy(dtype=float))
    D = Ch - Cl
    idx = [pool_idx[g] for g in genes]
    Xc = gdsc_arr[:, idx]
    Ds = D[np.ix_([genes.index(g) for g in genes], [genes.index(g) for g in genes])]
    s = np.einsum("ij,jk,ik->i", Xc, Ds, Xc)
    out = {}
    for drug, y in drug_cols.items():
        m = np.isfinite(s) & np.isfinite(y)
        if m.sum() < 30:
            continue
        rho, p = stats.spearmanr(s[m], y[m])
        out[drug] = (rho, p)
    return out

# real pathways
real_rows = []
for pw in sig_pw:
    r = run_gene_set(gmt[pw])
    if r:
        for drug, (rho, p) in r.items():
            real_rows.append((pw, drug, rho, p))
real = pd.DataFrame(real_rows, columns=["pathway", "drug", "rho", "P"])
real["q"] = stats.false_discovery_control(real["P"].to_numpy())
print("real tests:", len(real), "q<0.05:", int((real["q"] < 0.05).sum()))

# random sets
rand_rows = []
size_map = {pw: len([g for g in gmt[pw] if g in pool]) for pw in sig_pw}
for pw in sig_pw:
    m = size_map[pw]
    for ri in range(N_RAND):
        rset = list(rng.choice(pool, size=m, replace=False))
        r = run_gene_set(rset)
        if not r:
            continue
        for drug, (rho, p) in r.items():
            rand_rows.append((pw, ri, drug, rho, p))
rand = pd.DataFrame(rand_rows, columns=["pathway", "rep", "drug", "rho", "P"])
rand["q"] = stats.false_discovery_control(rand["P"].to_numpy())
print("random tests:", len(rand), "q<0.05:", int((rand["q"] < 0.05).sum()))

# per-drug comparison: real |rho| vs random 95th percentile
summary = []
for drug in ic50.columns:
    rr = real[real["drug"] == drug]["rho"].abs()
    rn = rand[rand["drug"] == drug]["rho"].abs()
    p95 = np.percentile(rn, 95)
    exceed = int((rr > p95).sum())
    summary.append((drug, len(rr), round(float(rr.median()), 3), round(float(rn.median()), 3),
                    round(float(p95), 3), exceed))
summ = pd.DataFrame(summary, columns=["drug", "n_real", "median_abs_rho_real", "median_abs_rho_random", "p95_random", "n_real_exceed_p95"])
summ.to_csv(OUT / "random_control_real_ic50.csv", index=False)
print(summ.to_string(index=False))

# overall: real vs random q<0.05 rates per pathway (paired)
pw_rate = []
for pw in sig_pw:
    rq = real[real["pathway"] == pw]
    rnq = rand[rand["pathway"] == pw]
    rate_r = (rq["q"] < 0.05).mean() if len(rq) else np.nan
    rate_n = (rnq["q"] < 0.05).mean() if len(rnq) else np.nan
    pw_rate.append((pw, round(rate_r, 3), round(rate_n, 3), int(len(rq))))
pwdf = pd.DataFrame(pw_rate, columns=["pathway", "real_q005_rate", "random_q005_rate", "n_drugs"])
pwdf.to_csv(OUT / "random_control_per_pathway_real_ic50.csv", index=False)
print("pathways where real rate > random rate:", int((pwdf["real_q005_rate"] > pwdf["random_q005_rate"]).sum()), "/", len(pwdf))

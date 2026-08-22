# -*- coding: utf-8 -*-
"""Diagnose tissue-type confounding in the GDSC2 validation:
1) paired real-vs-random |rho| tests per drug;
2) tissue-centered expression rescoring (removes tissue-mean effects);
3) breast-cancer-cell-line-only analysis.
"""
import csv, io
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
GDSC_IC50 = ROOT / "work/pkg/GDSC2/DataFiles/DataFiles/GLDS/GDSCv2/complete_matrix_output GDSCv2.txt"
GDSC_EXPR = ROOT / "work/gdsc2_expr.csv"
CELLS = ROOT / "work/pkg/GDSC2/DataFiles/DataFiles/GLDS/GDSCv2/Cell_Lines_Details.xlsx"
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

# cell line tissue annotation
cells = pd.read_excel(CELLS, sheet_name="Cell line details")
cells["COSMIC"] = cells["COSMIC identifier"].astype("Int64").astype(str)
tissue = dict(zip(cells["COSMIC"], cells["Cancer Type\n(matching TCGA label)"]))
tissue2 = dict(zip(cells["COSMIC"], cells["GDSC\nTissue descriptor 1"]))
cell_ids = [str(c).removeprefix("COSMIC_") for c in expr_z.columns]
tiss = [tissue.get(c, "NA") for c in cell_ids]
print("tissue coverage:", sum(1 for t in tiss if t != "NA"), "/", len(cell_ids))
print("tissue distribution:", pd.Series(tiss).value_counts().head(8).to_dict())

with open(TCGA_TRAIN, encoding="utf-8") as f:
    hdr = set(f.readline().rstrip("\n").split(","))
keep = [g for g in sorted(set(g for pw in sig_pw for g in gmt[pw])) if g in hdr]
tcga = pd.read_csv(TCGA_TRAIN, usecols=["sample_id"] + keep).merge(pd.read_csv(RISK), on="sample_id", how="inner")
med = tcga["risk_score"].median()
hi = tcga[tcga["risk_score"] >= med].drop(columns=["sample_id", "risk_score"])
lo = tcga[tcga["risk_score"] < med].drop(columns=["sample_id", "risk_score"])

# templates
templates = {}
for pw in sig_pw:
    genes = [g for g in gmt[pw] if g in keep]
    if len(genes) < 3:
        continue
    templates[pw] = (genes, corr_matrix(hi[genes].to_numpy(dtype=float)) - corr_matrix(lo[genes].to_numpy(dtype=float)))

drug_arr = {d: ic50[d].to_numpy() for d in ic50.columns}

def score_and_test(expr_mat, cells_subset=None, label=""):
    """expr_mat: cells x genes DataFrame (z-scored, optional tissue-centered).
    Returns per (pathway, drug) rho/P for cells_subset."""
    rows = []
    for pw, (genes, D) in templates.items():
        genes = [g for g in genes if g in expr_mat.columns]
        if len(genes) < 3:
            continue
        Xc = expr_mat.loc[cells_subset, genes].to_numpy() if cells_subset is not None else expr_mat[genes].to_numpy()
        idx = [genes.index(g) for g in genes]
        Ds = D[np.ix_(idx, idx)]
        s = np.einsum("ij,jk,ik->i", Xc, Ds, Xc)
        for d, y in drug_arr.items():
            m = np.isfinite(s) & np.isfinite(y)
            if m.sum() < 30:
                continue
            rho, p = stats.spearmanr(s[m], y[m])
            rows.append((pw, d, int(m.sum()), rho, p))
    df = pd.DataFrame(rows, columns=["pathway", "drug", "n", "rho", "P"])
    df["q"] = stats.false_discovery_control(df["P"].to_numpy())
    return df

# 1) raw (all cells)
raw = score_and_test(expr_z.T)
print("RAW q<0.05:", int((raw["q"] < 0.05).sum()), "/", len(raw))

# 2) tissue-centered expression
expr_c = expr_z.T.copy()
tiss_arr = np.array(tiss)
for t in np.unique(tiss_arr):
    m = tiss_arr == t
    if m.sum() >= 5:
        expr_c.loc[m] = expr_c.loc[m] - expr_c.loc[m].mean(axis=0)
tc = score_and_test(expr_c)
print("TISSUE-CENTERED q<0.05:", int((tc["q"] < 0.05).sum()), "/", len(tc))

# 3) breast cell lines only (raw expression)
brca_cells = [c for c, t in zip(cell_ids, tiss) if t == "BRCA"]
print("BRCA cell lines:", len(brca_cells))
if len(brca_cells) >= 20:
    br = score_and_test(expr_z.T, cells_subset=brca_cells)
    print("BRCA-ONLY q<0.05:", int((br["q"] < 0.05).sum()), "/", len(br))
    br.to_csv(OUT / "brca_cell_lines_real_ic50.csv", index=False)

raw.to_csv(OUT / "pathway_drug_real_ic50.csv", index=False)
tc.to_csv(OUT / "pathway_drug_tissue_centered.csv", index=False)
# paired real-vs-random per drug using stored random control
rand = pd.read_csv(OUT / "random_control_real_ic50.csv") if False else None

# paired test per drug: real |rho| vs the 50 random |rho| of the same pathway
rand_full = None
rp = OUT / "random_full_real_ic50.csv"
if rp.exists():
    rand_full = pd.read_csv(rp)
if rand_full is None:
    # rebuild random rho table quickly (pathway, rep, drug, rho)
    pool = [g for g in keep if g in expr_z.index]
    pool_idx = {g: i for i, g in enumerate(pool)}
    gdsc_arr = expr_z.T[pool].to_numpy()
    rand_rows = []
    size_map = {pw: len([g for g in gmt[pw] if g in pool]) for pw in sig_pw}
    for pw in sig_pw:
        m = size_map[pw]
        for ri in range(N_RAND):
            rset = list(rng.choice(pool, size=m, replace=False))
            rset = [g for g in rset if g in pool]
            Ch = corr_matrix(hi[rset].to_numpy(dtype=float))
            Cl = corr_matrix(lo[rset].to_numpy(dtype=float))
            D = Ch - Cl
            idx = [pool_idx[g] for g in rset]
            Xc = gdsc_arr[:, idx]
            Ds = D[np.ix_([rset.index(g) for g in rset], [rset.index(g) for g in rset])]
            s = np.einsum("ij,jk,ik->i", Xc, Ds, Xc)
            for d, y in drug_arr.items():
                msk = np.isfinite(s) & np.isfinite(y)
                if msk.sum() < 30:
                    continue
                rho, p = stats.spearmanr(s[msk], y[msk])
                rand_rows.append((pw, ri, d, rho))
    rand_full = pd.DataFrame(rand_rows, columns=["pathway", "rep", "drug", "rho"])
    rand_full.to_csv(rp, index=False)

paired = []
for d in ic50.columns:
    diffs = []
    for pw in sig_pw:
        rv = raw[(raw["pathway"] == pw) & (raw["drug"] == d)]["rho"].abs()
        rnv = rand_full[(rand_full["pathway"] == pw) & (rand_full["drug"] == d)]["rho"].abs()
        if len(rv) and len(rnv):
            diffs.append(rv.iloc[0] - rnv.mean())
    if len(diffs) >= 5:
        w, p = stats.wilcoxon(diffs)
        paired.append((d, round(float(np.mean(diffs)), 4), len(diffs), round(float(p), 4)))
pddf = pd.DataFrame(paired, columns=["drug", "mean_diff_real_minus_random", "n_pathways", "Wilcoxon_P"])
pddf.to_csv(OUT / "paired_real_vs_random_real_ic50.csv", index=False)
print(pddf.to_string(index=False))

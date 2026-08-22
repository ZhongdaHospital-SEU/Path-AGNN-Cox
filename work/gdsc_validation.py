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
OUT.mkdir(parents=True, exist_ok=True)

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

ic50 = load_ic50()
ic50 = ic50[[CURATED[k] for k in CURATED]]
ic50.columns = list(CURATED)
print("IC50:", ic50.shape)

expr = pd.read_csv(GDSC_EXPR, index_col=0)
expr_z = pd.DataFrame(zscore_rows(expr.T).T, index=expr.index, columns=expr.columns)
print("GDSC expr:", expr.shape)

gmt = load_gmt()
ptest = pd.read_csv(PTEST)
sig_pw = ptest[ptest["q"] < 0.05]["pathway"].tolist()
all_genes = sorted(set(g for pw in sig_pw for g in gmt[pw]))
print("significant pathways:", len(sig_pw), "| gene union:", len(all_genes))

# TCGA columns actually present
with open(TCGA_TRAIL if False else TCGA_TRAIN, encoding="utf-8") as f:
    header = f.readline().rstrip("\n").split(",")
hdr_set = set(header)
keep = [g for g in all_genes if g in hdr_set]
print("TCGA present pathway genes:", len(keep))
tcga = pd.read_csv(TCGA_TRAIN, usecols=["sample_id"] + keep)
risk = pd.read_csv(RISK)
tcga = tcga.merge(risk, on="sample_id", how="inner")
med = tcga["risk_score"].median()
hi = tcga[tcga["risk_score"] >= med].drop(columns=["sample_id", "risk_score"])
lo = tcga[tcga["risk_score"] < med].drop(columns=["sample_id", "risk_score"])
print("TCGA samples:", len(tcga), "hi/lo:", len(hi), len(lo))

def corr_matrix(X):
    Z = zscore_rows(X)
    C = Z.T @ Z / (Z.shape[0] - 1)
    return np.nan_to_num(C, nan=0.0)

templates = {}
for pw in sig_pw:
    genes = [g for g in gmt[pw] if g in keep]
    if len(genes) < 3:
        continue
    Ch = corr_matrix(hi[genes].to_numpy(dtype=float))
    Cl = corr_matrix(lo[genes].to_numpy(dtype=float))
    templates[pw] = (genes, Ch - Cl)
print("templates:", len(templates))

gdsc_genes = [g for g in expr_z.index if g in keep]
print("GDSC pathway genes present:", len(gdsc_genes))
expr_arr = expr_z.T  # cells x genes
scores = {}
for pw, (genes, D) in templates.items():
    genes = [g for g in genes if g in expr_arr.columns]
    if len(genes) < 3:
        continue
    Xc = expr_arr[genes].to_numpy()
    Ds = D[np.ix_([g for g in gmt[pw] if g in keep].index if False else [genes.index(g) for g in genes], [genes.index(g) for g in genes])]
    scores[pw] = np.einsum("ij,jk,ik->i", Xc, Ds, Xc)
score_df = pd.DataFrame(scores, index=expr_arr.index)
print("score matrix:", score_df.shape)

rows = []
for pw in score_df.columns:
    s = score_df[pw]
    for drug in ic50.columns:
        y = ic50[drug]
        mask = y.notna() & s.notna()
        if mask.sum() < 30:
            continue
        rho, p = stats.spearmanr(s[mask], y[mask])
        rows.append((pw, drug, int(mask.sum()), rho, p))
res = pd.DataFrame(rows, columns=["pathway", "drug", "n", "rho", "P"])
res["q"] = stats.false_discovery_control(res["P"].to_numpy())
res.to_csv(OUT / "pathway_drug_real_ic50.csv", index=False)
print("tests:", len(res), "| q<0.05:", int((res["q"] < 0.05).sum()))
print(res.sort_values("q").head(12).to_string(index=False))

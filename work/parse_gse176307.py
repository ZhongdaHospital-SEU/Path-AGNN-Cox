# -*- coding: utf-8 -*-
"""Parse GSE176307 (BACI urothelial ICB): salmon TPM + series matrix clinical + BLCA-template scoring."""
import gzip, csv
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
RAW = ROOT / "data/raw/ICB"
OUT = ROOT / "results/icb"
OUT.mkdir(parents=True, exist_ok=True)

def parse_series(path):
    rows = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("!Sample_"):
                continue
            key, _, vals = line.partition("\t")
            key = key.strip()
            if key not in ("!Sample_title", "!Sample_characteristics_ch1"):
                continue
            vals = list(csv.reader([vals.strip()], delimiter="\t"))[0]
            for i, v in enumerate(vals):
                v = v.strip('"')
                rows.setdefault(i, {}).setdefault(key, []).append(v)
    ch = {}
    for i, r in rows.items():
        title = r.get("!Sample_title", [""])[0]
        d = {}
        for c in r.get("!Sample_characteristics_ch1", []):
            for part in c.split("|"):
                if ":" in part:
                    k, _, v = part.partition(":")
                    d[k.strip()] = v.strip()
        ch[title] = d
    return ch

ch = parse_series(RAW / "GSE176307_series_matrix.txt.gz")
tpm = pd.read_csv(RAW / "GSE176307_salmon_tpm_gene.matrix.tsv.gz", compression="gzip", sep="\t")
print("tpm shape:", tpm.shape, "cols[:3]:", list(tpm.columns[:3]))
# gene column: first column (Ensembl or symbol?) and sample cols RS-*
key = pd.read_csv(RAW / "GSE176307_BACI_Omniseq_Sample_Name_Key_submitted_GEO_v2.csv.gz", compression="gzip")
rs2baci = {}
for _, r in key.iterrows():
    for rs in str(r["Omniseq_RS_ID (RNAseq)"]).split(","):
        rs2baci[rs.strip()] = str(r["Sample ID"]).strip()
gene_col = tpm.columns[0]
expr = tpm.set_index(gene_col)
sample_cols = [c for c in expr.columns]
print("sample cols sample:", sample_cols[:3], "n=", len(sample_cols))
expr = expr[sample_cols]
# log2(x+1)
expr = expr.apply(lambda c: np.log2(c.astype(float) + 1))
expr = expr.T
expr.index.name = "sample"
# map RS -> BACI
expr["baci"] = [rs2baci.get(str(i), str(i)) for i in expr.index]
expr = expr[~expr["baci"].duplicated(keep="first")]
expr = expr.set_index("baci")
print("expr after mapping:", expr.shape)
# clinical from series matrix: title "Patient sample BACI107" -> BACI107
clin_rows = []
for title, d in ch.items():
    baci = title.replace("Patient sample ", "").strip()
    clin_rows.append({"sample": baci, "response": d.get("io.response", "NA"),
                      "therapy": d.get("io.therapy", "NA"),
                      "tmb": pd.to_numeric(d.get("tmb", np.nan), errors="coerce"),
                      "pfs": pd.to_numeric(d.get("pfs", np.nan), errors="coerce"),
                      "progressed": d.get("progressed", "NA"),
                      "os": pd.to_numeric(d.get("overall survival", np.nan), errors="coerce"),
                      "alive": d.get("alive", "NA")})
clin = pd.DataFrame(clin_rows)
clin["resp_bin"] = clin["response"].map({"CR": 1, "PR": 1, "SD": 0, "PD": 0})
print("clinical n:", len(clin), "| response:", clin["resp_bin"].value_counts(dropna=False).to_dict())
common = sorted(set(expr.index) & set(clin["sample"]))
print("common:", len(common))
expr.loc[common].rename_axis("sample").to_csv(OUT / "gse176307_expr.csv")
clin.set_index("sample").loc[common].to_csv(OUT / "gse176307_clinical.csv")
print("wrote gse176307")

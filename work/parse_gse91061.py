# -*- coding: utf-8 -*-
"""Parse GSE91061: FPKM (Entrez x samples) + series matrix (response) -> pre-treatment matrix + clinical."""
import gzip, csv, subprocess, tempfile, os
from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
RAW = ROOT / "data/raw/ICB"
OUT = ROOT / "results/icb"
OUT.mkdir(parents=True, exist_ok=True)
RSCRIPT = r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe"

# 1) expression
expr = pd.read_csv(RAW / "GSE91061_BMS038109Sample.hg19KnownGene.fpkm.csv.gz", compression="gzip")
expr = expr.rename(columns={expr.columns[0]: "entrez"})
expr["entrez"] = expr["entrez"].astype(str)
pre_cols = [c for c in expr.columns if "_Pre_" in c]
print("samples:", len(expr.columns) - 1, "pre:", len(pre_cols))
X = expr[["entrez"] + pre_cols]

# 2) Entrez -> symbol via R org.Hs.eg.db
ids = X["entrez"].tolist()
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, dir=str(ROOT / "work")) as f:
    f.write("\n".join(ids))
    idf = f.name
outf = str(ROOT / "work" / "gse91061_entrez_map.csv")
subprocess.run([RSCRIPT, "--vanilla", str(ROOT / "work/map_entrez_symbol.R"), idf, outf], check=True)
os.unlink(idf)
m = pd.read_csv(outf)
m = m[m["SYMBOL"].notna() & (m["SYMBOL"] != "")]
m["ENTREZID"] = m["ENTREZID"].astype(str)
m = m.drop_duplicates("ENTREZID")
X = X.merge(m, left_on="entrez", right_on="ENTREZID", how="inner")
X = X.drop(columns=["entrez", "ENTREZID"]).drop_duplicates("SYMBOL")
X = X.set_index("SYMBOL")
print("mapped genes:", X.shape[0])

# 3) series matrix response
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

ch = parse_series(RAW / "GSE91061_series_matrix.txt.gz")
info = {t: ch.get(t, {}) for t in pre_cols}
clin = pd.DataFrame({
    "sample": pre_cols,
    "response": [info[t].get("response", "NA") for t in pre_cols],
    "visit": [info[t].get("visit", "NA") for t in pre_cols],
})
clin["resp_bin"] = clin["response"].map({"PRCR": 1, "PR": 1, "CR": 1, "SD": 0, "PD": 0})
print("clinical:", clin["response"].value_counts().to_dict())
X = X.T
X.index.name = "sample"
X.to_csv(OUT / "gse91061_expr_pre.csv")
clin.to_csv(OUT / "gse91061_clinical.csv", index=False)
print("wrote", X.shape, "samples x genes")

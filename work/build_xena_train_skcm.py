# -*- coding: utf-8 -*-
"""Build real TCGA-SKCM train.csv from Xena GDC star_tpm + survival."""
import gzip, os, re, time
import pandas as pd

ROOT = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
RAW = os.path.join(ROOT, "data", "raw", "TCGA-xena")
PROC = os.path.join(ROOT, "data", "processed")
GTF = os.path.join(ROOT, "data", "raw", "gencode.v23.annotation.gtf.gz")

ens2sym = {}
with gzip.open(GTF, "rt") as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        p = line.split("\t")
        if len(p) < 9 or p[2] != "gene":
            continue
        m = re.search(r'gene_id "([^"]+)"', p[8])
        n = re.search(r'gene_name "([^"]+)"', p[8])
        if m and n:
            ens2sym[m.group(1).split(".")[0]] = n.group(1)
print("ens2sym=%d" % len(ens2sym))

c = "SKCM"
TUMOR_RE = re.compile(r"-01[0-9A-Z]*$")
tpm = os.path.join(RAW, "TCGA-%s.star_tpm.tsv.gz" % c)
X = pd.read_csv(tpm, sep="\t", compression="gzip")
sample_cols = [col for col in X.columns if col != "Ensembl_ID"]
tumor_cols = [col for col in sample_cols if TUMOR_RE.search(col)]
print("samples=%d tumor=%d" % (len(sample_cols), len(tumor_cols)))
X = X[["Ensembl_ID"] + tumor_cols]
X["ens"] = X["Ensembl_ID"].str.split(".").str[0]
X["symbol"] = X["ens"].map(ens2sym)
X = X[X["symbol"].notna() & (X["symbol"] != "")]
sym_cols = [col for col in X.columns if col not in ("Ensembl_ID", "ens", "symbol")]
X["mean"] = X[sym_cols].mean(axis=1)
X = X.sort_values("mean", ascending=False).drop_duplicates("symbol")
X = X.set_index("symbol")[sym_cols].T
X.index.name = "sample_id"

f = os.path.join(RAW, "TCGA-%s.survival.tsv.gz" % c)
df = pd.read_csv(f, sep="\t", compression="gzip")
df = df[df["sample"].str.contains(TUMOR_RE)]
df = df.rename(columns={"sample": "sample_id", "OS.time": "OS_time", "OS": "OS_event"})
df = df[["sample_id", "OS_time", "OS_event"]].drop_duplicates("sample_id")
out = df.merge(X, left_on="sample_id", right_index=True, how="inner")
out = out.sort_values("sample_id").reset_index(drop=True)
outp = os.path.join(PROC, c, "train.csv")
os.makedirs(os.path.dirname(outp), exist_ok=True)
out.to_csv(outp, index=False)
print("wrote %s : %d samples x %d genes" % (outp, out.shape[0], out.shape[1] - 3))

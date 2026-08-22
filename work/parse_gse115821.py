# -*- coding: utf-8 -*-
"""Parse GSE115821 (IMPRES MGH cohort): counts + SOFT sample characteristics."""
import gzip
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
RAW = ROOT / "data/raw/ICB"
OUT = ROOT / "results/icb"
OUT.mkdir(parents=True, exist_ok=True)

# 1) SOFT sample characteristics
samples = []
cur = None
with gzip.open(RAW / "GSE115821_family.soft.gz", "rt", encoding="utf-8", errors="replace") as f:
    for ln in f:
        if ln.startswith("^SAMPLE"):
            cur = {"geo": ln.split("=")[1].strip()}
            samples.append(cur)
        elif ln.startswith("!Sample_title") and cur is not None:
            cur["title"] = ln.split("=", 1)[1].strip()
        elif ln.startswith("!Sample_characteristics_ch1") and cur is not None:
            v = ln.split("=", 1)[1].strip().strip('"')
            if ": " in v:
                k, _, val = v.partition(": ")
                cur[k] = val
            elif ":" in v:
                k, _, val = v.partition(":")
                cur[k] = val.strip()
meta = pd.DataFrame(samples)
print("samples:", len(meta), "| response:", meta["response"].value_counts(dropna=False).to_dict(),
      "| antibody:", meta["antibody"].value_counts(dropna=False).to_dict(),
      "| treatment:", meta["treatment state"].value_counts(dropna=False).to_dict())
meta.to_csv(OUT / "gse115821_meta.csv", index=False)

# 2) counts
cnt = pd.read_csv(RAW / "GSE115821_MGH_counts.csv.gz", compression="gzip")
gene_col = cnt.columns[0]
X = cnt.set_index(gene_col).drop(columns=["Chr", "Start", "End", "Strand", "Length"], errors="ignore")
X = X.apply(lambda c: np.log2(c.astype(float) + 1))
X = X.T
X.index.name = "sample"
# keep pre-treatment samples matching meta
pre = meta[meta["treatment state"].str.contains("PRE", na=False)]
keep = [s for s in X.index if s in set(pre["title"])]
print("pre samples in expr:", len(keep))
X = X.loc[keep]
resp_map = {"R": 1, "NR": 0}
clin = pre.set_index("title").loc[keep].copy()
clin["resp_bin"] = clin["response"].map(resp_map)
clin = clin[["response", "resp_bin", "antibody", "patient id"]]
print("analyzable:", clin["resp_bin"].value_counts(dropna=False).to_dict())
X.to_csv(OUT / "gse115821_expr.csv")
clin.to_csv(OUT / "gse115821_clinical.csv")
print("wrote", X.shape, clin.shape)

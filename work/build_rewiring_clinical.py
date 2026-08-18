
import os
import pandas as pd
root = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
proc = os.path.join(root, "data", "processed")
outd = os.path.join(proc, "rewiring")
os.makedirs(outd, exist_ok=True)
tmb = pd.read_csv(os.path.join(outd, "tmb_by_sample.csv"))
tmb = tmb[["sample_id", "tmb_nonsyn"]].rename(columns={"tmb_nonsyn": "tmb"})
cancers = ["LUAD", "BRCA", "BLCA", "COAD", "GBM", "HNSC", "KIRC", "LIHC", "LUSC", "OV", "STAD"]
for c in cancers:
    tr = pd.read_csv(os.path.join(proc, c, "train.csv"), usecols=lambda col: col in ("sample_id", "MKI67", "MKI67.1"))
    tr = tr.rename(columns={"MKI67": "ki67"})
    if "MKI67.1" in tr.columns and "ki67" not in tr.columns:
        tr = tr.rename(columns={"MKI67.1": "ki67"})
    tr["sample_id"] = tr["sample_id"].str[:15]
    out = tr.merge(tmb, on="sample_id", how="left")
    cl = os.path.join(proc, c, "clinical.csv")
    if os.path.exists(cl):
        cdf = pd.read_csv(cl)
        cdf["sample_id"] = cdf["sample_id"].str[:15]
        for col in ("stage", "age"):
            if col in cdf.columns:
                out = out.merge(cdf[["sample_id", col]], on="sample_id", how="left")
    out.to_csv(os.path.join(outd, "clinical_%s.csv" % c), index=False)
    print(c, out.shape, "ki67_n=%d tmb_n=%d" % (out["ki67"].notna().sum(), out["tmb"].notna().sum()))

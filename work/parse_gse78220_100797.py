# -*- coding: utf-8 -*-
"""Parse GSE78220 and GSE100797: expression + clinical (response, OS where available)."""
import gzip, csv
from pathlib import Path
import pandas as pd

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

# ---- GSE78220 ----
ch = parse_series(RAW / "GSE78220_series_matrix.txt.gz")
xl = pd.read_excel(RAW / "GSE78220_PatientFPKM.xlsx")
xl = xl.rename(columns={xl.columns[0]: "gene"}).set_index("gene")
xl = xl.T
xl.index.name = "sample"
xl = xl[~xl.index.duplicated(keep="first")]
# map PtN.baseline -> PtN
xl["pt"] = xl.index.str.replace(".baseline", "", regex=False)
info = {pt: ch.get(pt, {}) for pt in xl["pt"]}
resp_map = {"Complete Response": 1, "Partial Response": 1, "Progressive Disease": 0, "Stable Disease": 0}
clin = pd.DataFrame({
    "sample": xl.index,
    "pt": xl["pt"],
    "response": [info[p].get("anti-pd-1 response", "NA") for p in xl["pt"]],
})
clin["resp_bin"] = clin["response"].map(resp_map)
clin["os_days"] = pd.to_numeric([info[p].get("overall survival (days)", "NA") for p in xl["pt"]], errors="coerce")
clin["os_event"] = [1 if info[p].get("vital status") == "Dead" else (0 if info[p].get("vital status") == "Alive" else None) for p in xl["pt"]]
print("GSE78220 n=%d resp=%s" % (len(clin), clin["resp_bin"].value_counts().to_dict()))
xl.drop(columns=["pt"]).to_csv(OUT / "gse78220_expr.csv")
clin.to_csv(OUT / "gse78220_clinical.csv", index=False)

# ---- GSE100797 ----
ch2 = parse_series(RAW / "GSE100797_series_matrix.txt.gz")
expr = pd.read_csv(RAW / "GSE100797_ProcessedData.txt.gz", compression="gzip", sep="\t", index_col=0)
expr = expr.T
expr.index.name = "sample"
expr = expr[~expr.index.duplicated(keep="first")]
info2 = {s: ch2.get(s, {}) for s in expr.index}
recist_map = {"CR": 1, "PR": 1, "SD": 0, "PD": 0}
clin2 = pd.DataFrame({
    "sample": expr.index,
    "response": [info2[s].get("recist", "NA") for s in expr.index],
})
clin2["resp_bin"] = clin2["response"].map(recist_map)
clin2["pfs_time"] = pd.to_numeric([info2[s].get("pfs.time", "NA") for s in expr.index], errors="coerce")
clin2["pfs_event"] = pd.to_numeric([info2[s].get("pfs.event", "NA") for s in expr.index], errors="coerce")
clin2["os_time"] = pd.to_numeric([info2[s].get("os.time", "NA") for s in expr.index], errors="coerce")
clin2["os_event"] = pd.to_numeric([info2[s].get("os.event", "NA") for s in expr.index], errors="coerce")
print("GSE100797 n=%d resp=%s" % (len(clin2), clin2["resp_bin"].value_counts().to_dict()))
expr.to_csv(OUT / "gse100797_expr.csv")
clin2.to_csv(OUT / "gse100797_clinical.csv", index=False)
print("DONE")

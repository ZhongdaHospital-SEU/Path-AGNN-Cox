# -*- coding: utf-8 -*-
"""Extend Plan B: add GSE176307 (BACI urothelial ICB) to BLCA-template cohorts and recompute meta."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
root = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
sys.path.insert(0, str(root))
import importlib.util
spec = importlib.util.spec_from_file_location("icb2", root / "work/icb_analysis2.py")
icb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(icb)

gmt = icb.load_gmt()
blca_tpl = icb.build_template(root / "data/processed/BLCA/train.csv", gmt)
res = root / "results/icb"

rows = []
# IMvigor210 (BLCA template)
imv_expr = pd.read_csv(root / "data/processed/IMvigor210/train.csv").set_index("sample_id")
imv_clin = pd.read_csv(root / "data/processed/IMvigor210/clinical.csv")
imv_clin["sample"] = imv_clin["sample_id"]
imv_clin["resp_bin"] = imv_clin["response"].map({"CR/PR": 1, "SD/PD": 0})
r = icb.cohort_analysis("IMvigor210", imv_expr, imv_clin, blca_tpl)
if r: rows.append(r); print(r)
# GSE176307 (BLCA template)
expr = pd.read_csv(res / "gse176307_expr.csv").set_index("sample")
clin = pd.read_csv(res / "gse176307_clinical.csv")
r = icb.cohort_analysis("GSE176307", expr, clin, blca_tpl)
if r: rows.append(r); print(r)
# melanoma cohorts (SKCM template)
skcm_tpl = icb.build_template(root / "data/processed/SKCM/train.csv", gmt)
for name, ep_name in [("gse91061", "gse91061_expr_pre.csv"), ("gse78220", "gse78220_expr.csv"), ("gse100797", "gse100797_expr.csv")]:
    ep = res / ep_name
    cp = res / f"{name}_clinical.csv"
    if not (ep.exists() and cp.exists()):
        continue
    ex = pd.read_csv(ep).set_index("sample")
    cl = pd.read_csv(cp)
    r = icb.cohort_analysis(name.upper(), ex, cl, skcm_tpl)
    if r: rows.append(r); print(r)

df = pd.DataFrame(rows)
df.to_csv(res / "cohort_results_v2.csv", index=False)
print("\n=== cohort table ===")
print(df[["cohort", "n", "n_resp", "n_nonresp", "g", "wilcox_P"]].round(3).to_string(index=False))

# meta: BLCA-template cohorts (IMvigor210 + GSE176307)
bl = [r for r in rows if r["cohort"] in ("IMvigor210", "GSE176307")]
if len(bl) >= 2:
    meta_bl = icb.random_effects_meta([(r["cohort"], r["g"], r["v"]) for r in bl])
    print("\nBLCA-template meta:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in meta_bl.items()})
    pd.Series(meta_bl).to_csv(res / "meta_blca_v2.csv")
# meta: anti-PD-1/PD-L1 all (BLCA + SKCM PD-1 cohorts)
pd1 = [r for r in rows if r["cohort"] in ("IMvigor210", "GSE176307", "GSE91061", "GSE78220")]
if len(pd1) >= 2:
    meta_pd1 = icb.random_effects_meta([(r["cohort"], r["g"], r["v"]) for r in pd1])
    print("\nanti-PD-1/PD-L1 meta:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in meta_pd1.items()})
    pd.Series(meta_pd1).to_csv(res / "meta_pd1_v2.csv")
# all 5
if len(rows) >= 2:
    meta_all = icb.random_effects_meta([(r["cohort"], r["g"], r["v"]) for r in rows])
    print("\nall-5 meta:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in meta_all.items()})
    pd.Series(meta_all).to_csv(res / "meta_all_v2.csv")
print("DONE")

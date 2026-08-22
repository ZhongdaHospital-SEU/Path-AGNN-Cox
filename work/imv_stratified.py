# -*- coding: utf-8 -*-
"""IMvigor210: rewiring template score vs response, stratified by TMB and immune-cell IC."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
root = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
sys.path.insert(0, str(root))
import importlib.util
spec = importlib.util.spec_from_file_location("icb", root / "work/icb_analysis2.py")
icb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(icb)

gmt = icb.load_gmt()
tpl = icb.build_template(root / "data/processed/BLCA/train.csv", gmt)
expr = pd.read_csv(root / "data/processed/IMvigor210/train.csv").set_index("sample_id")
clin = pd.read_csv(root / "data/processed/IMvigor210/clinical.csv").set_index("sample_id")
common = sorted(set(expr.index) & set(clin.index))
expr = expr.loc[common]; clin = clin.loc[common]
sc = icb.score_samples(expr, tpl)
sc = (sc - sc.mean()) / sc.std(ddof=1)
resp = np.where(clin["response"] == "CR/PR", 1, np.where(clin["response"] == "SD/PD", 0, np.nan))

rows = []
def test(name, mask, resp_arr, sc_arr):
    m = mask & ~np.isnan(resp_arr)
    if m.sum() < 8 or (resp_arr[m] == 1).sum() < 3 or (resp_arr[m] == 0).sum() < 3:
        rows.append((name, int(m.sum()), int((resp_arr[m] == 1).sum()), int((resp_arr[m] == 0).sum()), np.nan, np.nan))
        return
    a = sc_arr[m][resp_arr[m] == 1]; b = sc_arr[m][resp_arr[m] == 0]
    u, p = stats.mannwhitneyu(a, b)
    rows.append((name, int(m.sum()), int((resp_arr[m] == 1).sum()), int((resp_arr[m] == 0).sum()),
                 round(float(a.mean() - b.mean()), 3), round(float(p), 3)))
    print("%-18s n=%3d resp=%2d nonresp=%3d mean_diff=%.3f P=%.3f" %
          (name, int(m.sum()), int((resp_arr[m] == 1).sum()), int((resp_arr[m] == 0).sum()), a.mean() - b.mean(), p))

sc_arr = np.asarray(sc); resp_arr = resp
tmb = pd.to_numeric(clin["tmb"], errors="coerce").to_numpy()
tmb_med = np.nanmedian(tmb)
ic = clin["ic"].to_numpy()
test("all", np.ones(len(sc_arr), bool), resp_arr, sc_arr)
test("TMB high", tmb >= tmb_med, resp_arr, sc_arr)
test("TMB low", tmb < tmb_med, resp_arr, sc_arr)
test("IC0", ic == "IC0", resp_arr, sc_arr)
test("IC1", ic == "IC1", resp_arr, sc_arr)
test("IC2", ic == "IC2", resp_arr, sc_arr)
test("IC1+2", np.isin(ic, ["IC1", "IC2"]), resp_arr, sc_arr)
# combined: high rewiring AND high TMB
high_rw = sc_arr >= np.median(sc_arr)
high_tmb = tmb >= tmb_med
m = ~np.isnan(resp_arr) & ~np.isnan(tmb)
grp = np.where(high_rw & high_tmb, 1, 0)
a = sc_arr[m][grp[m] == 1] if False else None
# response rate by quadrant
sub = pd.DataFrame({"rw": high_rw, "tmb_hi": high_tmb, "resp": resp_arr})
sub = sub.dropna(subset=["resp"])
for (rw, th), g in sub.groupby(["rw", "tmb_hi"]):
    rr = g["resp"].mean()
    rows.append(("quad rw=%d tmb=%d" % (rw, th), len(g), int(g["resp"].sum()), int((1 - g["resp"]).sum()),
                 round(float(rr), 3), np.nan))
    print("quad rw=%d tmb=%d n=%d resp_rate=%.3f" % (rw, th, len(g), rr))
out = pd.DataFrame(rows, columns=["stratum", "n", "n_resp", "n_nonresp", "mean_diff", "P"])
out.to_csv(root / "results/icb/imvigor210_stratified.csv", index=False)
print("DONE")

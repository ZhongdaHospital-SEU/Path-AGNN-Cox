# -*- coding: utf-8 -*-
"""GSE91061 On-treatment timepoint analysis (careful handling).
- On samples carry no response field in GEO; BOR is patient-level and mapped from Pre rows.
- Cross-sectional On: SKCM-template score, Wilcoxon + Hedges g + random gene-set control.
- Paired Pre->On delta: z-scored across the combined Pre+On pool; group-difference design
  cancels common timepoint/batch shifts; random-control percentile included.
- Sensitivities: drop duplicate Pt109 On sample; paired-subset cross-section.
"""
import io, sys
from pathlib import Path
import numpy as np, pandas as pd, re, gzip, csv
from scipy import stats
root = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
sys.path.insert(0, str(root))
import importlib.util
spec = importlib.util.spec_from_file_location("icb", root / "work/icb_analysis2.py")
icb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(icb)
RSCRIPT = r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe"
RAW = root / "data/raw/ICB"
RES = root / "results/icb"
gmt = icb.load_gmt()

# ---------- 1. parse On expression ----------
expr = pd.read_csv(RAW / "GSE91061_BMS038109Sample.hg19KnownGene.fpkm.csv.gz", compression="gzip")
expr = expr.rename(columns={expr.columns[0]: "entrez"})
expr["entrez"] = expr["entrez"].astype(str)
on_cols = [c for c in expr.columns if "_On_" in c]
X = expr[["entrez"] + on_cols]
m = pd.read_csv(root / "work/gse91061_entrez_map.csv")
m = m[m["SYMBOL"].notna() & (m["SYMBOL"] != "")]
m["ENTREZID"] = m["ENTREZID"].astype(str)
m = m.drop_duplicates("ENTREZID")
X = X.merge(m, left_on="entrez", right_on="ENTREZID", how="inner").drop(columns=["entrez", "ENTREZID"]).drop_duplicates("SYMBOL").set_index("SYMBOL")
X_on = X.T
X_on.index.name = "sample"

# ---------- 2. patient-level response from Pre rows ----------
clin_pre = pd.read_csv(RES / "gse91061_clinical.csv")
clin_pre["patient"] = clin_pre["sample"].str.extract(r"(Pt\d+)")[0]
pt2resp = {}
for _, r in clin_pre.drop_duplicates("patient").iterrows():
    pt2resp[r["patient"]] = (r["response"], r["resp_bin"])
def pt_of(s):
    mt = re.match(r"(Pt\d+)_", s)
    return mt.group(1) if mt else None
clin_on = pd.DataFrame({"sample": on_cols})
clin_on["patient"] = clin_on["sample"].map(pt_of)
clin_on["response"] = clin_on["patient"].map(lambda p: pt2resp.get(p, ("NA", np.nan))[0])
clin_on["resp_bin"] = clin_on["patient"].map(lambda p: pt2resp.get(p, ("NA", np.nan))[1])
print("On samples:", len(on_cols), "| with patient-level BOR:", int(clin_on["resp_bin"].notna().sum()))
X_on.to_csv(RES / "gse91061_on_expr.csv")
clin_on.to_csv(RES / "gse91061_on_clinical.csv", index=False)

# ---------- 3. cross-sectional On (SKCM template) ----------
tpl = icb.build_template(root / "data/processed/SKCM/train.csv", gmt)
def cross_sec(X, clin, label):
    common = sorted(set(X.index) & set(clin["sample"]))
    c = clin.set_index("sample").loc[common]
    y = pd.to_numeric(c["resp_bin"], errors="coerce").to_numpy()
    msk = ~np.isnan(y)
    Xc = X.loc[common].to_numpy(dtype=float)
    sc = icb.score_samples(X.loc[common], tpl)
    sc = (sc - sc.mean()) / sc.std(ddof=1)
    a = sc[msk][y[msk] == 1]; b = sc[msk][y[msk] == 0]
    g, v = icb.hedges_g(a, b)
    u, p = stats.mannwhitneyu(a, b)
    print(f"[{label}] n={int(msk.sum())} resp={int((y[msk]==1).sum())} nonresp={int((y[msk]==0).sum())} g={g:.3f} P={p:.3f}")
    return {"label": label, "n": int(msk.sum()), "n_resp": int((y[msk]==1).sum()),
            "n_nonresp": int((y[msk]==0).sum()), "g": g, "v": v, "wilcox_P": p,
            "mean_resp": a.mean(), "mean_nonresp": b.mean(), "sd_resp": a.std(ddof=1), "sd_nonresp": b.std(ddof=1)}

res_rows = []
res_rows.append(cross_sec(X_on, clin_on, "On_all"))
# sensitivity: drop duplicate Pt109 second On sample
dup = [s for s in on_cols if s.startswith("Pt109_On_")]
keep = [s for s in on_cols if s not in dup[1:]]
res_rows.append(cross_sec(X_on.loc[keep], clin_on[clin_on["sample"].isin(keep)], "On_no_dup"))
# sensitivity: paired subset only
paired = sorted(set(clin_pre["patient"]) & set(clin_on["patient"]))
Xp_on = X_on.loc[[s for s in on_cols if pt_of(s) in paired]]
cp_on = clin_on[clin_on["sample"].isin(Xp_on.index)]
res_rows.append(cross_sec(Xp_on, cp_on, "On_paired_subset"))

# ---------- 4. random gene-set control (cross-sectional On) ----------
rng = np.random.default_rng(20260822)
N_RAND = 50
tcga = pd.read_csv(root / "data/processed/SKCM/train.csv")
sub = tcga.dropna(subset=["OS_event"]); sub = sub[sub["OS_event"].isin([0, 1])]
hi = sub[sub["OS_event"] == 1].drop(columns=["sample_id", "OS_time", "OS_event"])
lo = sub[sub["OS_event"] == 0].drop(columns=["sample_id", "OS_time", "OS_event"])
pool = [g for g in tcga.columns if g not in ("sample_id", "OS_time", "OS_event")]
common = sorted(set(X_on.index) & set(clin_on["sample"]))
c = clin_on.set_index("sample").loc[common]
y = pd.to_numeric(c["resp_bin"], errors="coerce").to_numpy()
msk = ~np.isnan(y)
Xc = X_on.loc[common].to_numpy(dtype=float); yb = y[msk].astype(int)
null_rows = []
for pw, genes in gmt.items():
    genes = [g for g in genes if g in pool]
    sz = len(genes)
    if sz < 3: continue
    for ri in range(N_RAND):
        rset = list(rng.choice(pool, size=sz, replace=False))
        rset = [g for g in rset if g in X_on.columns]
        if len(rset) < 3: continue
        D = icb.corr_matrix(hi[rset].to_numpy(dtype=float)) - icb.corr_matrix(lo[rset].to_numpy(dtype=float))
        Xm = Xc[:, [X_on.columns.get_loc(g) for g in rset]]
        idx = [rset.index(g) for g in rset]
        s = np.einsum("ij,jk,ik->i", Xm, D[np.ix_(idx, idx)], Xm)
        s = (s - s.mean()) / s.std(ddof=1)
        a = s[msk][yb == 1]; b = s[msk][yb == 0]
        if len(a) >= 3 and len(b) >= 3 and a.std(ddof=1) + b.std(ddof=1) > 0:
            null_rows.append((pw, ri, a.mean() - b.mean()))
null = pd.DataFrame(null_rows, columns=["pathway", "rep", "mean_diff"])
null.to_csv(RES / "gse91061_on_random_null.csv", index=False)
nd = null["mean_diff"]
md = res_rows[0]["mean_resp"] - res_rows[0]["mean_nonresp"]
pct = float((nd < md).mean())
print(f"On random null: n={len(nd)} median={nd.median():.3f} sd={nd.std():.3f} real md={md:.3f} percentile={pct:.3f}")
res_rows[0]["random_null_percentile"] = pct
res_rows[0]["random_null_median"] = float(nd.median())

# ---------- 5. paired Pre->On delta ----------
X_pre = pd.read_csv(RES / "gse91061_expr_pre.csv").set_index("sample")
assert set(X_pre.columns) == set(X_on.columns), "gene set mismatch"
X_pre = X_pre[X_on.columns]
sc_pre = icb.score_samples(X_pre, tpl)
sc_on = icb.score_samples(X_on, tpl)
zall = np.concatenate([sc_pre, sc_on]); mu, sd = zall.mean(), zall.std(ddof=1)
z_pre = (sc_pre - mu) / sd; z_on = (sc_on - mu) / sd
pre_map = {}
for _s, _z in zip(X_pre.index, z_pre):
    _p = pt_of(_s)
    if _p is not None and _p not in pre_map:
        pre_map[_p] = _z
on_map = {s: z for s, z in zip(X_on.index, z_on)}
# one On sample per patient (drop duplicate Pt109)
on_first = {}
for s in on_cols:
    p = pt_of(s)
    if p not in on_first: on_first[p] = s
delta_rows = []
for p in paired:
    if p not in pre_map or p not in on_first: continue
    resp = pt2resp.get(p, ("NA", np.nan))
    delta_rows.append({"patient": p, "response": resp[0], "resp_bin": resp[1],
                       "z_pre": pre_map[p], "z_on": on_map[on_first[p]],
                       "delta": on_map[on_first[p]] - pre_map[p]})
delta = pd.DataFrame(delta_rows)
delta.to_csv(RES / "gse91061_paired_delta.csv", index=False)
d = delta.dropna(subset=["resp_bin"])
da = d[d["resp_bin"] == 1]["delta"].to_numpy(); db = d[d["resp_bin"] == 0]["delta"].to_numpy()
if len(da) >= 3 and len(db) >= 3:
    gd, vd = icb.hedges_g(da, db)
    ud, pd_ = stats.mannwhitneyu(da, db)
    rho, rp = stats.spearmanr(d["delta"], d["resp_bin"])
    print(f"[paired delta] n={len(d)} resp={len(da)} nonresp={len(db)} g={gd:.3f} P={pd_:.3f} spearman={rho:.3f} P={rp:.3f}")
    res_rows.append({"label": "paired_delta", "n": int(len(d)), "n_resp": int(len(da)),
                     "n_nonresp": int(len(db)), "g": gd, "v": vd, "wilcox_P": float(pd_),
                     "spearman": float(rho), "spearman_P": float(rp),
                     "mean_resp": da.mean(), "mean_nonresp": db.mean(),
                     "sd_resp": da.std(ddof=1), "sd_nonresp": db.std(ddof=1)})
    # random control for delta
    nd2 = []
    for pw, genes in gmt.items():
        genes = [g for g in genes if g in pool]
        sz = len(genes)
        if sz < 3: continue
        for ri in range(N_RAND):
            rset = list(rng.choice(pool, size=sz, replace=False))
            rset = [g for g in rset if g in X_on.columns]
            if len(rset) < 3: continue
            D = icb.corr_matrix(hi[rset].to_numpy(dtype=float)) - icb.corr_matrix(lo[rset].to_numpy(dtype=float))
            iX = [rset.index(g) for g in rset]
            s_pre = np.einsum("ij,jk,ik->i", X_pre[rset].to_numpy(dtype=float), D[np.ix_(iX, iX)], X_pre[rset].to_numpy(dtype=float))
            s_on = np.einsum("ij,jk,ik->i", X_on[rset].to_numpy(dtype=float), D[np.ix_(iX, iX)], X_on[rset].to_numpy(dtype=float))
            zc = np.concatenate([s_pre, s_on]); mu2, sd2 = zc.mean(), zc.std(ddof=1)
            zpre = (s_pre - mu2) / sd2; zon = (s_on - mu2) / sd2
            pre_pat, on_pat = {}, {}
            for _s, _z in zip(X_pre.index, zpre):
                _p = pt_of(_s)
                if _p is not None and _p not in pre_pat:
                    pre_pat[_p] = _z
            for _s, _z in zip(X_on.index, zon):
                _p = pt_of(_s)
                if _p is not None and _p not in on_pat:
                    on_pat[_p] = _z
            dd = pd.DataFrame({"patient": sorted(set(pre_pat) & set(on_pat))})
            dd["delta"] = dd["patient"].map(lambda p: on_pat[p] - pre_pat[p])
            dv = dd.merge(delta[["patient", "resp_bin"]], on="patient").dropna()
            if len(dv) >= 8 and dv["resp_bin"].nunique() == 2:
                x = dv[dv["resp_bin"] == 1]["delta"].to_numpy(); y2 = dv[dv["resp_bin"] == 0]["delta"].to_numpy()
                if len(x) >= 3 and len(y2) >= 3:
                    nd2.append(x.mean() - y2.mean())
    nd2 = np.array(nd2)
    pct2 = float((nd2 < gd).mean()) if len(nd2) else np.nan
    print(f"delta random null: n={len(nd2)} median={np.nanmedian(nd2):.3f} real g={gd:.3f} percentile={pct2:.3f}")
    res_rows[-1]["random_null_percentile"] = pct2
    res_rows[-1]["random_null_median"] = float(np.nanmedian(nd2)) if len(nd2) else np.nan

df = pd.DataFrame(res_rows)
df.to_csv(RES / "gse91061_on_results.csv", index=False)
print("\n=== summary ===")
print(df[["label", "n", "n_resp", "n_nonresp", "g", "wilcox_P"]].round(3).to_string(index=False))
print("DONE")




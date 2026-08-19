# -*- coding: utf-8 -*-
"""Route B: patch render_manuscript.py - replace table5() and rewiring_tokens()."""
import io, re
from pathlib import Path

P = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis") / "manuscript" / "render_manuscript.py"
t = io.open(P, encoding="utf-8-sig").read()

def replace_func(src, name, new_body):
    pat = re.compile(r"^def " + name + r"\(.*?(?=^def )", re.S | re.M)
    m = pat.search(src)
    assert m, f"function {name} not found"
    return src[:m.start()] + new_body + "\n\n" + src[m.end():]

table5_new = '''def table5(rw_dir) -> str:
    """Framework-validation table for patient-specific rewiring (LUAD/BRCA)."""
    from scipy.stats import hypergeom
    def load_eff(ds):
        p = ROOT / "results" / "rewiring" / ds / "pathway_effects.csv"
        return pd.read_csv(p) if p.exists() else None
    lu, br = load_eff("LUAD"), load_eff("BRCA")
    lines = ["| Check | LUAD | BRCA |", "|---|---|---|"]
    if lu is not None and br is not None:
        n_path = len(lu)
        perm = {}
        for ds in ("LUAD", "BRCA"):
            pf = ROOT / "results" / "rewiring" / ds / "permutation_test.csv"
            if pf.exists():
                perm[ds] = pd.read_csv(pf).iloc[0]
        if len(perm) == 2:
            lq, bq = perm["LUAD"], perm["BRCA"]
            lines.append(f"| Pathways tested | {int(lq['n_pathways_observed'])} | {int(bq['n_pathways_observed'])} |")
            lines.append(f"| Significant pathways, BH-FDR on the unadjusted test | {int(lq['observed_sig'])} | {int(bq['observed_sig'])} |")
            lines.append(f"| Per-pathway permutation-calibrated pathways (FDR q<0.05) | {int((lu['perm_q'] < 0.05).sum())} | {int((br['perm_q'] < 0.05).sum())} |")
            lines.append(f"| Cohort-level label-permutation null, mean | {fmt2(float(lq['null_mean_sig']))} | {fmt2(float(bq['null_mean_sig']))} |")
            lines.append(f"| Cohort-level label-permutation null, maximum | {int(lq['null_max_sig'])} | {int(bq['null_max_sig'])} |")
            lines.append(f"| Cohort-level permutation P | {fmt_p(float(lq['perm_p']))} | {fmt_p(float(bq['perm_p']))} |")
        for col, label in (("null_pct", "edge-matched"), ("block_null_pct", "density-matched")):
            lines.append(f"| Median percentile of real pathways, {label} null | {fmt2(float(lu[col].median()))} | {fmt2(float(br[col].median()))} |")
            lines.append(f"| Pathways above the 95th percentile, {label} null | {int((lu[col] >= 0.95).sum())} | {int((br[col] >= 0.95).sum())} |")
        lines.append(f"| Expected above the 95th percentile by chance | {fmt2(0.05 * n_path)} | {fmt2(0.05 * n_path)} |")
        known_file = ROOT / "data" / "pathways" / "luad_known_pathways.txt"
        if known_file.exists():
            known = [ln.strip() for ln in known_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            n_known = len(set(known) & set(lu["pathway"]))
            top = set(lu.sort_values("perm_p").head(20)["pathway"])
            hits = len(top & set(known))
            p_enr = hypergeom.sf(hits - 1, n_path, n_known, min(20, n_path)) if n_known > 0 else np.nan
            lines.append(f"| Known-pathway enrichment, top-20 by permutation P | {hits} hits, {fmt_p(float(p_enr))} | n.a. |")
        sn_rows = {}
        for ds in ("LUAD", "BRCA"):
            sf = ROOT / "results" / "rewiring" / ds / "static_null.csv"
            if sf.exists():
                s = pd.read_csv(sf, index_col=0)
                if "total_var" in s.index:
                    sn_rows[ds] = float(s.loc["total_var", "0"])
        ev = ROOT / "results" / "rewiring" / "edge_var.csv"
        if len(sn_rows) == 2 and ev.exists():
            evd = pd.read_csv(ev).set_index("dataset")
            lines.append(f"| Static-model total edge variance | {sn_rows['LUAD']:.2e} | {sn_rows['BRCA']:.2e} |")
            lines.append(f"| Adaptive-model total edge variance | {float(evd.loc['LUAD', 'total_var']):.2e} | {float(evd.loc['BRCA', 'total_var']):.2e} |")
        rc = rw_dir / "random_control.csv"
        if rc.exists():
            rc_df = pd.read_csv(rc)
            lines.append(f"| Randomized-partition control, significant pathways (3 seeds) | {int(rc_df['n_sig_q005'].min())}\u2013{int(rc_df['n_sig_q005'].max())} | n.a. |")
    hg = []
    if lu is not None:
        for pw in ("Homologous recombination", "DNA replication"):
            r = lu[lu["pathway"] == pw]
            if len(r):
                r = r.iloc[0]
                hg.append(("LUAD", pw, fmt2(float(r["cohen_d"])),
                           f"{fmt2(float(r['d_ci_lo']))}\u2013{fmt2(float(r['d_ci_hi']))}",
                           fmt_q(float(r["perm_q"])), f"{100 * float(r['block_null_pct']):.0f}"))
    if br is not None:
        r = br[br["pathway"] == "MAPK signaling pathway"]
        if len(r):
            r = r.iloc[0]
            hg.append(("BRCA", "MAPK signaling pathway", fmt2(float(r["cohen_d"])),
                       f"{fmt2(float(r['d_ci_lo']))}\u2013{fmt2(float(r['d_ci_hi']))}",
                       fmt_q(float(r["perm_q"])), f"{100 * float(r['block_null_pct']):.0f}"))
    if hg:
        lines.append("")
        lines.append("Hypothesis-generating pathways that exceeded the per-pathway permutation null and the density-matched control:")
        lines.append("")
        lines.append("| Cancer | Pathway | Cohen's d (95% CI) | Permutation q | Density-matched percentile |")
        lines.append("|---|---|---|---|---|")
        for row in hg:
            lines.append("| %s | %s | %s (%s) | %s | %s |" % row)
    return "\\n".join(lines)'''

rt_new = '''def rewiring_tokens(rw_dir) -> dict:
    """Read rewiring outputs into paper tokens; missing files leave tokens unfilled."""
    rw_dir = Path(rw_dir)
    st = {}
    en = rw_dir / "enrichment.csv"
    if en.exists():
        e = pd.read_csv(en, index_col=0)
        get = lambda k, default="\u2014": (e.loc[k, "0"] if k in e.index and pd.notna(e.loc[k, "0"]) else default)
        st["ENRICH_HITS"] = str(int(get("hits", 0)))
        st["ENRICH_TOP_K"] = str(int(get("top_k", 0)))
        pv = get("p", np.nan)
        st["ENRICH_P"] = fmt_p(float(pv)) if isinstance(pv, (int, float)) and np.isfinite(float(pv)) else "\u2014"
    sn = rw_dir / "static_null.csv"
    if sn.exists():
        s = pd.read_csv(sn, index_col=0)
        if "total_var" in s.index:
            st["STATIC_NULL_VAR"] = fmt2(float(s.loc["total_var", "0"]))
    ev = ROOT / "results" / "rewiring" / "edge_var.csv"
    if ev.exists():
        d = pd.read_csv(ev).set_index("dataset")
        if "LUAD" in d.index:
            st["ADAPTIVE_REWIRE_VAR"] = fmt2(float(d.loc["LUAD", "total_var"]))
    cc = rw_dir / "clinical_corr.csv"
    if cc.exists():
        c = pd.read_csv(cc)
        if len(c):
            c = c.reindex(c["rho"].abs().sort_values(ascending=False).index)
            r0 = c.iloc[0]
            st["CLINICAL_RHO"] = fmt2(float(r0["rho"]))
            st["CLINICAL_P"] = fmt_p(float(r0["p"]))
            st["CLINICAL_CORR_DESC"] = f"{r0['clinical']} (n={int(r0['n'])})"
    for ds, prefix in [("BRCA", "CLINICAL_BRCA"), ("LUAD", "CLINICAL_LUAD")]:
        cf = rw_dir.parent / ds / "clinical_corr.csv"
        if cf.exists():
            c = pd.read_csv(cf)
            if len(c):
                c = c.reindex(c["rho"].abs().sort_values(ascending=False).index)
                r0 = c.iloc[0]
                st[f"{prefix}_RHO"] = fmt2(float(r0["rho"]))
                st[f"{prefix}_P"] = fmt_p(float(r0["p"]))
                st[f"{prefix}_N"] = str(int(r0["n"]))
    for ds, prefix in [("LUAD", "MVC_LUAD"), ("BRCA", "MVC_BRCA")]:
        mf = rw_dir.parent / ds / "multivariable_cox.csv"
        if mf.exists():
            m = pd.read_csv(mf)
            row = m[(m["model"] == "multivariable") & (m["covariate"] == "risk_z")]
            if len(row):
                row = row.iloc[0]
                st[f"{prefix}_HR"] = fmt2(float(row["hr"]))
                st[f"{prefix}_CI"] = f"{fmt2(float(row['ci_lower']))}\u2013{fmt2(float(row['ci_upper']))}"
                st[f"{prefix}_P"] = fmt_p(float(row["p"]))
    for ds, prefix in [("LUAD", "PERM_LUAD"), ("BRCA", "PERM_BRCA")]:
        pf = rw_dir.parent / ds / "permutation_test.csv"
        if pf.exists():
            q = pd.read_csv(pf).iloc[0]
            st[f"{prefix}_SIG"] = str(int(q["observed_sig"]))
            st[f"{prefix}_P"] = fmt_p(float(q["perm_p"]))
            st["PERM_N_PATHWAYS"] = str(int(q["n_pathways_observed"]))
            st["PERM_NULL_MEAN"] = fmt2(float(q["null_mean_sig"]))
            st[f"{prefix}_NULL_MEAN"] = fmt2(float(q["null_mean_sig"]))
            st[f"{prefix}_NULL_MAX"] = str(int(q["null_max_sig"]))
    eff = {}
    for ds in ("LUAD", "BRCA"):
        pf = rw_dir.parent / ds / "pathway_effects.csv"
        if pf.exists():
            eff[ds] = pd.read_csv(pf)
    if "LUAD" in eff and "BRCA" in eff:
        lu, br = eff["LUAD"], eff["BRCA"]
        st["PWP_LUAD_SIG"] = str(int((lu["perm_q"] < 0.05).sum()))
        st["PWP_BRCA_SIG"] = str(int((br["perm_q"] < 0.05).sum()))
        st["MATCHED_LUAD_MED"] = fmt2(float(lu["null_pct"].median()))
        st["MATCHED_BRCA_MED"] = fmt2(float(br["null_pct"].median()))
        st["MATCHED_LUAD_EXCEED"] = str(int((lu["null_pct"] >= 0.95).sum()))
        st["MATCHED_BRCA_EXCEED"] = str(int((br["null_pct"] >= 0.95).sum()))
        st["MATCHED_BLOCK_LUAD_MED"] = fmt2(float(lu["block_null_pct"].median()))
        st["MATCHED_BLOCK_BRCA_MED"] = fmt2(float(br["block_null_pct"].median()))
        st["MATCHED_EXPECTED"] = fmt2(0.05 * len(lu))
        def d_tokens(df, pw, prefix):
            r = df[df["pathway"] == pw]
            if len(r):
                r = r.iloc[0]
                st[prefix] = fmt2(float(r["cohen_d"]))
                st[prefix + "_CI"] = f"{fmt2(float(r['d_ci_lo']))}\u2013{fmt2(float(r['d_ci_hi']))}"
        d_tokens(lu, "Homologous recombination", "D_LUAD_HR")
        d_tokens(lu, "DNA replication", "D_LUAD_DNA")
        d_tokens(br, "MAPK signaling pathway", "D_BRCA_MAPK")
        from scipy.stats import hypergeom
        known_file = ROOT / "data" / "pathways" / "luad_known_pathways.txt"
        if known_file.exists():
            known = [ln.strip() for ln in known_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            n_known = len(set(known) & set(lu["pathway"]))
            top = set(lu.sort_values("perm_p").head(20)["pathway"])
            hits = len(top & set(known))
            p_enr = hypergeom.sf(hits - 1, len(lu), n_known, min(20, len(lu))) if n_known > 0 else np.nan
            st["ENRICH_LUAD_HITS"] = str(int(hits))
            st["ENRICH_LUAD_P"] = fmt_p(float(p_enr))
    return st'''

t = replace_func(t, "table5", table5_new)
t = replace_func(t, "rewiring_tokens", rt_new)
io.open(P, "w", encoding="utf-8", newline="\n").write(t)
print("RENDER PATCH OK, length", len(t))
import ast
ast.parse(t)
print("SYNTAX OK")

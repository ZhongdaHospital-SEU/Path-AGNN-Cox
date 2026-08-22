import io
p = r"manuscript/render_manuscript.py"
t = io.open(p, encoding="utf-8").read()
old = '''def table6(imm_dir) -> str:
    """Predicted drug sensitivity table (exploratory)."""
    imm_dir = Path(imm_dir)
    lines = []
    for ds in ["BRCA", "LUAD"]:
        p = imm_dir / ds / ("drug_stats_%s.csv" % ds)
        if not (p.exists() and p.stat().st_size > 1):
            continue
        d = pd.read_csv(p).sort_values("wilcox_P")
        lines.append("**%s (n high/low: %d/%d)**" % (ds, int(d["n_high"].iloc[0]), int(d["n_low"].iloc[0])))
        lines.append("| Drug | IC50 median (high) | IC50 median (low) | Wilcoxon P | FDR q | Spearman \u03c1 | Spearman P |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in d.iterrows():
            lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                r["drug"], fmt2(float(r["high_median_IC50"])), fmt2(float(r["low_median_IC50"])),
                fmt_p(float(r["wilcox_P"])), fmt_q(float(r["wilcox_q"])),
                fmt2(float(r["spearman_rho"])), fmt_p(float(r["spearman_P"]))))
        lines.append("")
    lines.append("_IC50 values are GDSC2/oncoPredict in-silico predictions; associations are exploratory and not FDR-significant unless stated._")
    return "\\n".join(lines)'''
new = '''def table6(imm_dir) -> str:
    """Predicted drug sensitivity table (exploratory), single table with cohort rows."""
    imm_dir = Path(imm_dir)
    lines = ["| Cohort | Drug | IC50 median (high) | IC50 median (low) | Wilcoxon P | FDR q | Spearman \\u03c1 | Spearman P |",
             "|---|---|---|---|---|---|---|---|"]
    for ds in ["BRCA", "LUAD"]:
        p = imm_dir / ds / ("drug_stats_%s.csv" % ds)
        if not (p.exists() and p.stat().st_size > 1):
            continue
        d = pd.read_csv(p).sort_values("wilcox_P")
        lines.append("| **%s (n high/low: %d/%d)** | | | | | | | |" % (ds, int(d["n_high"].iloc[0]), int(d["n_low"].iloc[0])))
        for _, r in d.iterrows():
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                "", r["drug"], fmt2(float(r["high_median_IC50"])), fmt2(float(r["low_median_IC50"])),
                fmt_p(float(r["wilcox_P"])), fmt_q(float(r["wilcox_q"])),
                fmt2(float(r["spearman_rho"])), fmt_p(float(r["spearman_P"]))))
    lines.append("_IC50 values are GDSC2/oncoPredict in-silico predictions; associations are exploratory and not FDR-significant unless stated._")
    return "\\n".join(lines)'''
assert t.count(old) == 1
t = t.replace(old, new)
io.open(p, "w", encoding="utf-8").write(t)
print("table6 patched to single table")

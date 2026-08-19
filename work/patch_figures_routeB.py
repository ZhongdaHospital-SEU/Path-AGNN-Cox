# -*- coding: utf-8 -*-
"""Route B: patch make_figures.py - replace fig_rewiring with effect-size panels."""
import io, re
from pathlib import Path

P = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis") / "manuscript" / "make_figures.py"
t = io.open(P, encoding="utf-8-sig").read()

pat = re.compile(r"\ndef fig_rewiring\(df=None\):.*?(?=\ndef )", re.S)
m = pat.search(t)
assert m, "fig_rewiring not found"

new = '''def fig_rewiring(df=None):
    """Figure 5: between-stratum effect sizes with permutation-calibrated
    significance (A/B), cohort-level label-permutation null (C), clinical
    correlation (D), matched random-set controls (E)."""
    panels = []
    fig = plt.figure(figsize=(11.5, 6.5))
    # A/B: Cohen's d forest plots (all pathways)
    for k, (ds, pos, title) in enumerate([
            ("LUAD", [0.07, 0.56, 0.44, 0.36], "A  Between-stratum effect sizes (LUAD)"),
            ("BRCA", [0.55, 0.56, 0.42, 0.36], "B  Between-stratum effect sizes (BRCA)")]):
        ef = ROOT / "results" / "rewiring" / ds / "pathway_effects.csv"
        ax = fig.add_axes(pos)
        if ef.exists():
            d = pd.read_csv(ef).sort_values("cohen_d")
            sig = (d["perm_q"] < 0.05).to_numpy()
            y = np.arange(len(d))
            ax.errorbar(d["cohen_d"], y,
                        xerr=[d["cohen_d"] - d["d_ci_lo"], d["d_ci_hi"] - d["cohen_d"]],
                        fmt="none", ecolor="#95A5A6", elinewidth=0.6, zorder=1)
            cols = np.where(sig, "#C0392B", "#7F8C8D")
            ax.scatter(d["cohen_d"], y, s=8, c=cols, zorder=2)
            ax.axvline(0, color="black", lw=0.6)
            ax.set_yticks(y)
            ax.set_yticklabels([p if len(p) <= 24 else p[:23] + "..." for p in d["pathway"]],
                               fontsize=5.2)
            ax.set_xlabel("Cohen's d (high vs. low risk)", fontsize=8)
            ax.set_title(title, fontsize=9)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.text(0.98, 0.02, "%d significant after permutation FDR" % int(sig.sum()),
                    transform=ax.transAxes, ha="right", fontsize=6.5, color="#C0392B")
            panels.append("A" if k == 0 else "B")
        else:
            ax.text(0.5, 0.5, "pending", ha="center", va="center"); ax.axis("off")
    # C: label-permutation null vs observed
    ax = fig.add_axes([0.07, 0.08, 0.28, 0.36])
    perm_files = [("LUAD", ROOT / "results" / "rewiring" / "LUAD" / "permutation_test.csv"),
                  ("BRCA", ROOT / "results" / "rewiring" / "BRCA" / "permutation_test.csv")]
    if all(p.exists() for _, p in perm_files):
        obs, null_mean, null_max, perm_p = [], [], [], []
        for _, pf in perm_files:
            q = pd.read_csv(pf).iloc[0]
            obs.append(int(q["observed_sig"]))
            null_mean.append(float(q["null_mean_sig"]))
            null_max.append(int(q["null_max_sig"]))
            perm_p.append(float(q["perm_p"]))
        grp = np.array([0.0, 1.7]); wd = 0.55
        ax.bar(grp - wd / 2, obs, wd, color=["#C0392B", "#8E44AD"], alpha=0.9, label="Observed")
        ax.bar(grp + wd / 2, null_mean, wd, color="#BDC3C7", alpha=0.95,
               label="Permutation null (mean)")
        ax.scatter(grp + wd / 2, null_max, marker="_", s=150, color="#2C3E50", zorder=5,
                   label="Null maximum")
        for i, (x, o, nm, pv) in enumerate(zip(grp, obs, null_mean, perm_p)):
            ax.annotate(str(o), (x - wd / 2, o), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7)
            ax.annotate("%.2f" % nm, (x + wd / 2, nm), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7)
            pv_txt = "P<0.001" if pv < 0.001 else "P=%.3f" % pv
            ax.text(x, max(o, null_max[i]) * 1.06 + 1.5, pv_txt, ha="center",
                    fontsize=7, color="#555555")
        ax.set_xticks(grp); ax.set_xticklabels(["LUAD", "BRCA"])
        ax.set_ylabel("Significant pathways (q<0.05)", fontsize=8)
        ax.set_ylim(0, max(obs) * 1.28)
        ax.set_title("C  Rewiring vs label-permutation null (1,000 perms)", fontsize=9)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        panels.append("C")
    else:
        ax.text(0.5, 0.5, "pending", ha="center", va="center"); ax.axis("off")
    # D: clinical correlation
    ax = fig.add_axes([0.40, 0.08, 0.28, 0.36])
    cc = ROOT / "results" / "rewiring" / "LUAD" / "clinical_corr.csv"
    if cc.exists() and cc.stat().st_size > 1:
        c = pd.read_csv(cc)
        if len(c):
            c = c.reindex(c["rho"].abs().sort_values(ascending=False).index)
            cols = c["clinical"].tolist()
            rhos = c["rho"].tolist()
            ax.barh(np.arange(len(c)), rhos, color="#27AE60", alpha=0.85)
            ax.set_yticks(np.arange(len(c))); ax.set_yticklabels(cols, fontsize=8)
            ax.axvline(0, color="black", lw=0.6)
            for i, (r, p) in enumerate(zip(c["rho"], c["p"])):
                ax.text(r + 0.01, i, "P<0.001" if p < 0.001 else "P=%.3f" % p,
                        fontsize=6.5, va="center")
            ax.set_xlabel("Spearman rho (rewiring magnitude)", fontsize=8)
            ax.set_title("D  Clinical correlation", fontsize=9)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            panels.append("D")
    else:
        ax.text(0.5, 0.5, "pending", ha="center", va="center"); ax.axis("off")
    # E: matched random-set controls (percentiles of real pathways)
    ax = fig.add_axes([0.73, 0.08, 0.24, 0.36])
    data, positions, labels = [], [], []
    for i, ds in enumerate(["LUAD", "BRCA"]):
        ef = ROOT / "results" / "rewiring" / ds / "pathway_effects.csv"
        if not ef.exists():
            continue
        d = pd.read_csv(ef)
        for j, col in enumerate(["null_pct", "block_null_pct"]):
            v = d[col].dropna().to_numpy()
            data.append(v)
            positions.append(i * 1.5 + j * 0.32)
            labels.append("%s %s" % (ds, "edge" if j == 0 else "density"))
    if data:
        bp = ax.boxplot(data, positions=positions, widths=0.26, patch_artist=True,
                        showfliers=False, manage_ticks=False)
        for patch in bp["boxes"]:
            patch.set_facecolor("#AED6F1"); patch.set_alpha(0.8)
        ax.axhline(0.5, color="#2C3E50", ls="--", lw=0.9)
        ax.axhline(0.95, color="#E67E22", ls=":", lw=0.9)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=6, rotation=25, ha="right")
        ax.set_ylabel("Percentile of real pathway effect", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title("E  Matched random-set controls", fontsize=9)
        ax.text(0.02, 0.97, "chance 0.50", transform=ax.transAxes, fontsize=6, color="#2C3E50")
        ax.text(0.02, 0.905, "95th pct", transform=ax.transAxes, fontsize=6, color="#E67E22")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        panels.append("E")
    else:
        ax.text(0.5, 0.5, "pending", ha="center", va="center"); ax.axis("off")
    fig.savefig(FIGDIR / "Figure5_rewiring.svg", format="svg")
    fig.savefig(FIGDIR / "Figure5_rewiring.png", dpi=300)
    plt.close(fig)
    return "Figure5_rewiring.svg", panels
'''

t = t[:m.start()] + "\n" + new + t[m.end():]
io.open(P, "w", encoding="utf-8", newline="\n").write(t)
import ast
ast.parse(t)
print("FIGURES PATCH OK, length", len(t))

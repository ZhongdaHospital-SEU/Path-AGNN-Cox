# -*- coding: utf-8 -*-
"""Remove the incorrect MAPK claim (it is at the bottom of the density-matched null)."""
import io, ast
root = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"

# 1) template
p = root + r"\manuscript\Path-AGNN-Cox_manuscript_template.md"
t = io.open(p, encoding="utf-8-sig").read()
old = "The two LUAD pathways that survived the permutation-calibrated test exceeded 94%\u201396% of the density-matched null sets, and the MAPK signaling pathway in BRCA exceeded all 200 matched sets (d = {{D_BRCA_MAPK}}, 95% CI {{D_BRCA_MAPK_CI}}); these observations are hypothesis-generating rather than evidence of global pathway-specific rewiring."
new = "The two LUAD pathways that survived the permutation-calibrated test exceeded 94%\u201396% of the density-matched null sets, whereas no BRCA pathway exceeded the 95th percentile of that null; these observations are hypothesis-generating rather than evidence of global pathway-specific rewiring."
n = t.count(old)
print("template occurrences:", n)
assert n == 1, n
t = t.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(t)

# 2) render table5: drop MAPK block
p2 = root + r"\manuscript\render_manuscript.py"
r = io.open(p2, encoding="utf-8").read()
old2 = """    if br is not None:
        r = br[br["pathway"] == "MAPK signaling pathway"]
        if len(r):
            r = r.iloc[0]
            hg.append(("BRCA", "MAPK signaling pathway", fmt2(float(r["cohen_d"])),
                       f"{fmt2(float(r['d_ci_lo']))}\u2013{fmt2(float(r['d_ci_hi']))}",
                       fmt_q(float(r["perm_q"])), f"{100 * (1 - float(r['block_null_pct'])):.1f}"))
"""
n2 = r.count(old2)
print("render occurrences:", n2)
assert n2 == 1, n2
r = r.replace(old2, "")
old3 = '        d_tokens(br, "MAPK signaling pathway", "D_BRCA_MAPK")\n'
n3 = r.count(old3)
print("token occurrences:", n3)
assert n3 == 1, n3
r = r.replace(old3, "")
io.open(p2, "w", encoding="utf-8", newline="\n").write(r)
ast.parse(r)
print("ALL FIXED")

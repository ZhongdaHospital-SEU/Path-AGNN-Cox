# -*- coding: utf-8 -*-
import io, ast
root = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
p = root + r"\manuscript\render_manuscript.py"
t = io.open(p, encoding="utf-8").read()
old = "f\"{100 * float(r['block_null_pct']):.0f}\"))"
new = "f\"{100 * (1 - float(r['block_null_pct'])):.1f}\"))"
n = t.count(old)
print("occurrences:", n)
assert n == 2, n
t = t.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(t)
ast.parse(t)
print("FIXED")

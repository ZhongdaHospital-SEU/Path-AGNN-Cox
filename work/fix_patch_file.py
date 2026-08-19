# -*- coding: utf-8 -*-
import io
p = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis\work\patch_render_routeB.py"
t = io.open(p, encoding="utf-8").read()
old = 'return "\\n".join(lines)'
new = 'return "\\\\n".join(lines)'
n = t.count(old)
print("occurrences:", n)
assert n == 1
t = t.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(t)
print("patch file fixed")

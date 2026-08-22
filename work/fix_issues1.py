# -*- coding: utf-8 -*-
import io, re
# A) move REWIRING table definition before 3.4.1
p = "manuscript/Path-AGNN-Cox_manuscript_template.md"
s = io.open(p, encoding="utf-8").read()
blk_old = "\n### {{TDEF:REWIRING}}. Framework validation of patient-specific pathway rewiring in LUAD, BRCA and KIRC.\n{{TABLE:REWIRING}}\n"
assert blk_old in s
s = s.replace(blk_old, "\n", 1)
anchor = "\n**3.4.1. Between-stratum rewiring"
assert anchor in s
s = s.replace(anchor, "\n### {{TDEF:REWIRING}}. Framework validation of patient-specific pathway rewiring in LUAD, BRCA and KIRC.\n{{TABLE:REWIRING}}\n" + anchor, 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("A done")

# B) hyperparam table 1e notation
p2 = "manuscript/render_manuscript.py"
r = io.open(p2, encoding="utf-8").read()
old = '"hidden 32, layers 2, mlp 32, dropout 0.1, epochs 100, lr 0.001, batch 128, patience 15, L2 0.0001, lambda_sparse 0.001, lambda_consist 0.1"'
new = '"hidden 32, layers 2, mlp 32, dropout 0.1, epochs 100, lr 1e-3, batch 128, patience 15, L2 1e-4, lambda_sparse 1e-3, lambda_consist 0.1"'
assert old in r
r = r.replace(old, new)
io.open(p2, "w", encoding="utf-8", newline="\n").write(r)
print("B done")

# C) check_formatting: skip References section for decimal check
p3 = "manuscript/check_formatting.py"
c = io.open(p3, encoding="utf-8").read()
old = 'lines = MD.read_text(encoding="utf-8").splitlines()\ntext = "\\n".join(lines)'
new = 'lines = MD.read_text(encoding="utf-8").splitlines()\ntext = "\\n".join(lines)\n# exclude the reference list from content-format checks\nref_i = next((i for i, ln in enumerate(lines) if ln.strip() == "## References"), None)\nif ref_i is not None:\n    body_lines = lines[:ref_i]\n    body_text = "\\n".join(body_lines)\nelse:\n    body_lines, body_text = lines, text'
assert old in c
c = c.replace(old, new)
# decimal-check loop should scan body_lines; find the relevant loop
old2 = '# 5. numbers: 2 decimals; P values: >=0.001 -> 3 decimals; <0.001 -> P<0.001\n'
if old2 not in c:
    old2 = '# 5.'
print("C marker found:", old2 in c)
# locate the numbers section and swap loop source
m = re.search(r"for i, ln in enumerate\(lines, 1\):\n    for mm in re\.finditer\(r\"\\d+\\.\\d{3,}\"", c)
if m:
    c = c[:m.start()] + "for i, ln in enumerate(body_lines, 1):\n    for mm in re.finditer(r\"\\d+\\.\\d{3,}\"" + c[m.end():]
    print("C swapped loop")
else:
    print("C loop pattern not found; manual check needed")
io.open(p3, "w", encoding="utf-8", newline="\n").write(c)
print("C done")

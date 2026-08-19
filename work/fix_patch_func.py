# -*- coding: utf-8 -*-
import io
p = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis\work\patch_render_routeB.py"
t = io.open(p, encoding="utf-8").read()
old = '''def replace_func(src, name, new_body):
    pat = re.compile(r"\\ndef " + name + r"\\(.*?\\) -> str:.*?(?=\\ndef |\\Z)", re.S)
    m = pat.search(src)
    assert m, f"function {name} not found"
    return src[:m.start()] + "\\n" + new_body + src[m.end():]'''
new = '''def replace_func(src, name, new_body):
    pat = re.compile(r"^def " + name + r"\\(.*?(?=^def )", re.S | re.M)
    m = pat.search(src)
    assert m, f"function {name} not found"
    return src[:m.start()] + new_body + "\\n\\n" + src[m.end():]'''
assert t.count(old) == 1, t.count(old)
t = t.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(t)
print("replace_func fixed")

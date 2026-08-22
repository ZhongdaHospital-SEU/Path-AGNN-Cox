# -*- coding: utf-8 -*-
import io
p = "manuscript/md2docx.py"
s = io.open(p, encoding="utf-8").read()

# 1) add booktabs helper after imports/helpers
anchor = "def main():"
helper = '''from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def _border(el, tag, sz):
    e = OxmlElement(tag)
    e.set(qn("w:val"), "single")
    e.set(qn("w:sz"), str(sz))
    e.set(qn("w:space"), "0")
    e.set(qn("w:color"), "000000")
    el.append(e)


def style_booktabs(tbl):
    """Three-line table: top rule, header bottom rule, bottom rule; no vertical lines."""
    nrow = len(tbl.rows)
    for ri, row in enumerate(tbl.rows):
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            tcB = tcPr.find(qn("w:tcBorders"))
            if tcB is None:
                tcB = OxmlElement("w:tcBorders")
                tcPr.append(tcB)
            for tag in ("w:top", "w:left", "w:bottom", "w:right", "w:insideH", "w:insideV"):
                el = tcB.find(qn(tag))
                if el is not None:
                    tcB.remove(el)
            if ri == 0:
                _border(tcB, "w:top", 12)
                _border(tcB, "w:bottom", 6)
            if ri == nrow - 1:
                _border(tcB, "w:bottom", 12)


def main():'''
assert anchor in s
s = s.replace(anchor, helper, 1)

# 2) replace table style usage
old = """                tbl = doc.add_table(rows=len(rows), cols=ncol)
                tbl.style = "Table Grid"
                for ri, r in enumerate(rows):"""
new = """                tbl = doc.add_table(rows=len(rows), cols=ncol)
                tbl.style = "Normal Table"
                style_booktabs(tbl)
                for ri, r in enumerate(rows):"""
assert old in s
s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("md2docx.py patched")

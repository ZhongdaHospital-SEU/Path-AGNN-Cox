# -*- coding: utf-8 -*-
"""Convert the rendered manuscript Markdown to DOCX (python-docx).

Usage:  python manuscript/md2docx.py
Output: manuscript/Path-AGNN-Cox_manuscript.docx

- headings -> Heading 1/2/3
- markdown tables -> Table Grid
- bullet lines -> List Bullet
- **bold** and *italic* inline formatting
- figure PNGs inserted (centered) above their legend lines in the
  'Figure legends' section, using results/figures/figure_manifest.json
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
MD = ROOT / "manuscript" / "Path-AGNN-Cox_manuscript.md"
OUT = ROOT / "manuscript" / "Path-AGNN-Cox_manuscript.docx"
MANIFEST = ROOT / "results" / "figures" / "figure_manifest.json"

BOLD = re.compile(r"\*\*(.+?)\*\*")
ITAL = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")

def add_runs(par, text, base_bold=False):
    """Add text to paragraph, honoring **bold** and *italic* markers."""
    pos = 0
    for m in BOLD.finditer(text):
        if m.start() > pos:
            add_runs_plain(par, text[pos:m.start()], base_bold)
        r = par.add_run(m.group(1)); r.bold = True
        pos = m.end()
    if pos < len(text):
        add_runs_plain(par, text[pos:], base_bold)

def add_runs_plain(par, text, base_bold=False):
    pos = 0
    for m in ITAL.finditer(text):
        if m.start() > pos:
            r = par.add_run(text[pos:m.start()]); r.bold = base_bold
        r = par.add_run(m.group(1)); r.italic = True; r.bold = base_bold
        pos = m.end()
    if pos < len(text):
        r = par.add_run(text[pos:]); r.bold = base_bold

def parse_table(block):
    rows = []
    for ln in block.strip().splitlines():
        ln = ln.strip()
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        rows.append(cells)
    return rows

def main():
    text = MD.read_text(encoding="utf-8")
    lines = text.splitlines()
    man = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    png_by_num = {}
    for fname, meta in man.items():
        m = re.match(r"Figure(\d+)", fname)
        if m:
            png = ROOT / "results" / "figures" / (Path(meta["file"]).stem + ".png")
            if png.exists():
                png_by_num[int(m.group(1))] = png
    doc = Document()
    # base style
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(10.5)
    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
            continue
        if s == "---":
            i += 1
            continue
        # tables: detect a block of consecutive lines starting with |
        if s.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            rows = parse_table("\n".join(lines[i:j]))
            if rows:
                ncol = max(len(r) for r in rows)
                tbl = doc.add_table(rows=len(rows), cols=ncol)
                tbl.style = "Table Grid"
                for ri, r in enumerate(rows):
                    for ci in range(ncol):
                        cell = tbl.cell(ri, ci)
                        cell.text = ""
                        par = cell.paragraphs[0]
                        add_runs(par, r[ci] if ci < len(r) else "")
                i = j
            else:
                i += 1
            continue
        # headings
        m = re.match(r"^(#{1,3})\s+(.*)$", s)
        if m:
            lvl = len(m.group(1))
            doc.add_heading(m.group(2), level=lvl)
            i += 1
            continue
        # inline figure image line -> centered PNG (SVG is the editable source)
        mi = re.match(r"^!\[Figure\s*(\d+)\]\(([^)]+)\)$", s)
        if mi:
            num = int(mi.group(1))
            if num in png_by_num:
                par = doc.add_paragraph()
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = par.add_run()
                run.add_picture(str(png_by_num[num]), width=Inches(6.0))
            i += 1
            continue
        # figure legend line -> paragraph with bold lead
        mf = re.match(r"^- \*\*Figure\s+(\d+)([A-Z]?)\.\s+(.*)$", s)
        if mf:
            num = int(mf.group(1))
            par = doc.add_paragraph()
            r = par.add_run("Figure %d%s. " % (num, mf.group(2))); r.bold = True
            add_runs(par, mf.group(3))
            par.paragraph_format.space_after = Pt(10)
            i += 1
            continue
        # bullet
        if s.startswith("- "):
            par = doc.add_paragraph(style="List Bullet")
            add_runs(par, s[2:])
            i += 1
            continue
        # normal paragraph
        par = doc.add_paragraph()
        add_runs(par, s)
        i += 1
    doc.save(OUT)
    print("saved:", OUT)
    print("figures embedded:", len(png_by_num))

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""Formatting audit for the rendered manuscript.

Checks (per user rules):
  1. No unfilled {{TOKEN}} placeholders.
  2. No doubled "P=P" artifacts, no infinity-ratio artifacts.
  3. Tables numbered 1..N by first appearance, contiguous, headings titled.
  4. Figures numbered 1..M by first appearance, contiguous; manifest panels A.. contiguous.
  5. Data numbers carry 2 decimals; P/q values: >=0.001 -> exactly 3 decimals,
     <0.001 -> written "P<0.001"; no "P=0.000".
  6. No leftover placeholders (your-github, Table S?, TODO, XXX, FIXME, 'placeholder').

Usage: python manuscript/check_formatting.py [manuscript.md]
Exit code 0 = clean, 1 = issues found.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
MD = ROOT / "manuscript" / "Path-AGNN-Cox_manuscript.md"
MANIFEST = ROOT / "results" / "figures" / "figure_manifest.json"

issues = []

def report(lineno, msg):
    issues.append(f"L{lineno}: {msg}")

lines = MD.read_text(encoding="utf-8").splitlines()
text = "\n".join(lines)

# 1. unfilled tokens
for i, ln in enumerate(lines, 1):
    for m in re.finditer(r"\{\{[^}]*\}\}", ln):
        report(i, f"unfilled token {m.group(0)}")

# 2. artifacts
for i, ln in enumerate(lines, 1):
    if re.search(r"P\s*=\s*P", ln):
        report(i, "doubled 'P = P'")
    if re.search(r"[\u221e\u221d]", ln):
        report(i, "infinity symbol present")
    if re.search(r"\d+(\.\d+)?\s*\u00d7\s*larger", ln):
        report(i, "ratio 'x larger' phrasing")

# 3. tables by first appearance
seen_t = []
for i, ln in enumerate(lines, 1):
    for m in re.finditer(r"\bTable\s+(\d+)\b", ln):
        n = int(m.group(1))
        if n not in [x[0] for x in seen_t]:
            seen_t.append((n, i))
if seen_t != [(k + 1, i) for k, (_, i) in enumerate(seen_t)]:
    got = [n for n, _ in seen_t]
    report(0, f"Table numbers not contiguous 1..N by first appearance: {got}")
else:
    print(f"[ok] Tables referenced in order: {[n for n, _ in seen_t]}")
tdefs = re.findall(r"^### Table\s+(\d+)\.\s+(.+)$", text, re.M)
if tdefs:
    nums = [int(a) for a, _ in tdefs]
    if nums != list(range(1, len(nums) + 1)):
        report(0, f"Table headings not contiguous: {nums}")
    for a, title in tdefs:
        if len(title.strip()) < 5:
            report(0, f"Table {a} heading too short: {title!r}")
    print(f"[ok] Table headings: {nums}")

# 4. figures by first appearance
seen_f = []
for i, ln in enumerate(lines, 1):
    for m in re.finditer(r"\bFigure\s+(\d+)([A-Z])?\b", ln):
        n = int(m.group(1))
        if n not in [x[0] for x in seen_f]:
            seen_f.append((n, i))
if seen_f != [(k + 1, i) for k, (_, i) in enumerate(seen_f)]:
    got = [n for n, _ in seen_f]
    report(0, f"Figure numbers not contiguous 1..M by first appearance: {got}")
else:
    print(f"[ok] Figures referenced in order: {[n for n, _ in seen_f]}")
if MANIFEST.exists():
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for fname, meta in man.items():
        panels = meta.get("panels", [])
        expect = [chr(ord("A") + k) for k in range(len(panels))]
        if panels != expect:
            report(0, f"{fname}: panels {panels} not contiguous A..")
        else:
            print(f"[ok] {fname}: panels {''.join(panels)}")

# 7. inline figure images present, in order; no trailing tables/legends sections
img_nums = [int(m.group(1)) for m in re.finditer(r"^!\[Figure\s*(\d+)\]\(results/figures/.*\.svg\)$", text, re.M)]
expected = list(range(1, len(man) + 1)) if MANIFEST.exists() and "man" in dir() else []
if img_nums != expected:
    report(0, f"inline figure image lines missing/out-of-order: {img_nums} (expected {expected})")
else:
    print(f"[ok] Inline figure images in order: {img_nums}")
for bad in ("## Tables", "## Figure legends"):
    if re.search(rf"^{bad}$", text, re.M):
        report(0, f"trailing section '{bad}' must be removed (tables/figures must be inline)")
    else:
        print(f"[ok] No trailing '{bad}' section")

# 5. numeric formatting (arXiv ids and DOIs scrubbed first)
for i, ln in enumerate(lines, 1):
    ln2 = re.sub(r"arXiv\s*:\s*\d+\.\d+", "arXiv:XX", ln, flags=re.I)
    ln2 = re.sub(r"\b10\.\d{4,}/[0-9A-Za-z.\-]+", "DOI", ln2)
    ln2 = re.sub(r"doi\.org/[0-9A-Za-z./\-]+", "URL", ln2, flags=re.I)
    # 5a. numbers with >2 decimals outside P=/q= context
    for m in re.finditer(r"(?<![\w=<>])\d+\.\d{3,}\d*(?![\d])", ln2):
        s = m.group(0)
        pre = ln2[:m.start()]
        if re.search(r"(?:P|q)\s*=\s*$", pre):
            continue
        report(i, f"number with >2 decimals: {s}")
    # 5b. P values
    for m in re.finditer(r"P\s*=\s*(\d+\.\d+)", ln2):
        v = float(m.group(1))
        frac = m.group(1).split(".")[1]
        if v >= 0.001 and len(frac) != 3:
            report(i, f"P value {m.group(0)} should have 3 decimals")
        if v == 0.0:
            report(i, f"P value {m.group(0)} is zero")
    if re.search(r"P\s*=\s*<", ln2):
        report(i, "malformed 'P = <'")
    if re.search(r"P\s*=\s*0\.000", ln2):
        report(i, "'P=0.000' should be 'P<0.001'")
    # 5c. q values same rule
    for m in re.finditer(r"\bq\s*=\s*(\d+\.\d+)", ln2):
        v = float(m.group(1))
        frac = m.group(1).split(".")[1]
        if v >= 0.001 and len(frac) != 3:
            report(i, f"q value {m.group(0)} should have 3 decimals")

# 6. leftover placeholders
for i, ln in enumerate(lines, 1):
    for pat in (r"your-github", r"Table\s+S\?", r"\bTODO\b", r"\bXXX\b", r"\bFIXME\b", r"placeholder"):
        if re.search(pat, ln, re.I):
            report(i, f"placeholder artifact: {pat}")

print()
if issues:
    print(f"ISSUES FOUND: {len(issues)}")
    for it in issues:
        print(" -", it)
    sys.exit(1)
print("ALL CHECKS PASSED")

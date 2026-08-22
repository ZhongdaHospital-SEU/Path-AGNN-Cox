# -*- coding: utf-8 -*-
"""Part C: apply section replacements, targeted edits, citation numbering, references."""
import re, json, io
from rewrite_part_a import FRONT, INTRO
from rewrite_part_b import DISCUSSION, END_SECTIONS

TEMPLATE = r"manuscript/Path-AGNN-Cox_manuscript_template.md"
REFJSON = r"work/refs_final.json"
s = io.open(TEMPLATE, encoding="utf-8-sig").read()
old_tokens = set(re.findall(r"\{\{[A-Z0-9_:|]+\}\}", s))

def rep(old, new):
    global s
    assert old in s, "NOT FOUND: " + old[:90]
    s = s.replace(old, new, 1)

# 1. front matter + abstract
a, b = s.index("## Title"), s.index("## 1. Introduction")
s = s[:a] + FRONT + "\n" + s[b:]

# 2. introduction
a, b = s.index("## 1. Introduction"), s.index("## 2. Materials and methods")
s = s[:a] + INTRO + "\n" + s[b:]

# 3. discussion
a, b = s.index("## 4. Discussion"), s.index("## 5. Conclusion")
s = s[:a] + DISCUSSION + "\n" + s[b:]

# 4. targeted edits
rep("where each patient was represented by RNA-sequencing expression (UCSC Xena TPM, log2-transformed and z-scored within training folds) and overall-survival annotation.",
    "where each patient was represented by RNA-sequencing expression from UCSC Xena [xena2020], log2-transformed and z-scored within training folds, and overall-survival annotation extracted with TCGAbiolinks [tcga_biolinks2016].")
rep("we additionally compiled {{N_EXTERNAL}} GEO microarray cohorts (Affymetrix;",
    "we additionally compiled {{N_EXTERNAL}} GEO microarray cohorts [geo2013,geoquery2007] (Affymetrix;")
rep("using the platform annotation tables downloaded from NCBI GEO (e.g., GPL570, GPL96, GPL6480)",
    "using the platform annotation tables downloaded from NCBI GEO [geo2013] (e.g., GPL570, GPL96, GPL6480)")
rep("Expression matrices were first mapped to the KEGG cancer-core pathway catalogue",
    "Expression matrices were first mapped to the KEGG cancer-core pathway catalogue [kegg2023]")
rep("Model parameters are optimized by the negative Cox partial likelihood with Breslow tie handling,",
    "Model parameters are optimized by the negative Cox partial likelihood with Breslow tie handling [cox1972,therneau2000],")
rep("The concordance index (C-index) served as the primary discrimination metric; the time-dependent AUC (mean over the 0.25/0.50/0.75 quantile times) was used as a secondary metric.",
    "The concordance index [harrell1996,uno2011] served as the primary discrimination metric; the time-dependent AUC [heagerty2005] (mean over the 0.25/0.50/0.75 quantile times) was used as a secondary metric.")
rep("Decision-curve analysis was used to assess the clinical value of the risk score:",
    "Decision-curve analysis [vickers2006] was used to assess the clinical value of the risk score:")
rep("the false discovery rate was controlled with the Benjamini\u2013Hochberg procedure.",
    "the false discovery rate was controlled with the Benjamini\u2013Hochberg procedure [bh1995].")
rep("For external validation, each model was retrained on the full TCGA cohort and evaluated on the GEO cohorts without any fine-tuning.",
    "For external validation, each model was retrained on the full TCGA cohort and evaluated on the GEO cohorts without any fine-tuning, following current recommendations for validating prognostic models [altman2009].")
rep("Calibration of the risk score was assessed with the slope of a univariate Cox regression",
    "Calibration of the risk score [crowson2016] was assessed with the slope of a univariate Cox regression")
rep("Path-AGNN-Cox was implemented in Python 3.10 with PyTorch 2.4 (CPU), and all\nsurvival statistics were computed with lifelines 0.30.",
    "Path-AGNN-Cox was implemented in Python 3.10 with PyTorch [pytorch2019] on CPU, and all survival statistics were computed with lifelines [lifelines2019]; cross-validation splits and tree baselines were implemented with scikit-learn [sklearn2011].")
rep("Model hyperparameters are listed in the footnote of {{TREF:BENCHMARK}} (see config/benchmark.yaml).",
    "Model hyperparameters and baseline configurations are summarized in {{TREF:HYPERPARAM}}.")
rep("**Algorithm 1** Training and rewiring testing.",
    "### {{TDEF:HYPERPARAM}}. Model hyperparameters and baseline configurations.\n{{TABLE:HYPERPARAM}}\n\n**Algorithm 1** Training and rewiring testing.")
rep("PathMoG reported external validation in a single breast-cancer cohort (METABRIC) [PathMoG]",
    "PathMoG reported external validation in a single breast-cancer cohort, METABRIC [pathmog2026,metabric2012]")
rep("and in LUAD with tumor mutational burden (\u03c1 = {{CLINICAL_LUAD_RHO}}, {{CLINICAL_LUAD_P}}, n = {{CLINICAL_LUAD_N}})",
    "and in LUAD with tumor mutational burden [tmb2017] (\u03c1 = {{CLINICAL_LUAD_RHO}}, {{CLINICAL_LUAD_P}}, n = {{CLINICAL_LUAD_N}})")
rep("In the independent IMvigor210 anti-PD-L1 cohort, {{IMV_RESULT_SENTENCE}}",
    "In the IMvigor210 anti-PD-L1 cohort [imvigor2018], {{IMV_RESULT_SENTENCE}}")
rep("Tumor purity was not used as an additional anchor because ESTIMATE-based purity",
    "Tumor purity was not used as an additional anchor because ESTIMATE-based purity [estimate2013]")
rep("The associations were robust to the definition of rewiring magnitude ({{SENSITIVITY_SENT}}).\n\n{{FIG:IMV}}",
    "The associations were robust to the definition of rewiring magnitude: {{SENSITIVITY_SENT}}.\n\n### {{TDEF:SENSITIVITY}}. Sensitivity of clinical anchors to the rewiring-magnitude definition.\n{{TABLE:SENSITIVITY}}\n\n{{FIG:IMV}}")
rep("we computed MCP-counter immune cell abundance estimates and ssGSEA scores of nine immune Hallmark gene sets per patient",
    "we computed MCP-counter immune cell abundance estimates [mcpcounter2016] and ssGSEA scores of nine immune Hallmark gene sets [gsva2013,hallmark2015] per patient")
rep("We further predicted drug sensitivity with the oncoPredict model trained on GDSC2 cell-line pharmacogenomic data (198 compounds)",
    "We further predicted drug sensitivity with the oncoPredict model [oncopredict2021] trained on GDSC2 cell-line pharmacogenomic data [gdsc2013] (198 compounds)")

# 5. end sections before References
assert "## References" in s
s = s.replace("\n## References", "\n" + END_SECTIONS + "\n## References", 1)

# 6. drop old references
i = s.index("## References")
s = s[:i]

# 7. citation numbering
pat = re.compile(r"\[([a-z][a-z0-9_]*(?:,[a-z][a-z0-9_]*)*)\]")
num, order = {}, []
for m in pat.finditer(s):
    for k in m.group(1).split(","):
        if k not in num:
            num[k] = len(order) + 1
            order.append(k)
refs = json.load(io.open(REFJSON, encoding="utf-8"))
missing = set(num) - set(refs)
assert not missing, missing
extra = set(refs) - set(num)
assert not extra, ("uncited refs: %s" % extra)
assert len(order) == 46, len(order)
def repl(m):
    ns = sorted(num[k] for k in m.group(1).split(","))
    return "[" + ",".join(str(n) for n in ns) + "]"
s2 = pat.sub(repl, s)
refs_block = "## References\n\n" + "\n".join("%d. %s" % (num[k], refs[k]["text"]) for k in order) + "\n"
s2 = s2 + refs_block

# 8. write + token diff
new_tokens = set(re.findall(r"\{\{[A-Z0-9_:|]+\}\}", s2))
gone = old_tokens - new_tokens
print("tokens removed:", sorted(gone))
NEW_EXPECTED = {"{{TABLE:HYPERPARAM}}", "{{TDEF:SENSITIVITY}}", "{{TREF:HYPERPARAM}}", "{{TABLE:SENSITIVITY}}", "{{TDEF:HYPERPARAM}}"}
assert not (new_tokens - old_tokens - NEW_EXPECTED), new_tokens - old_tokens - NEW_EXPECTED
io.open(TEMPLATE, "w", encoding="utf-8", newline="\n").write(s2)
print("TEMPLATE REWRITTEN; citations:", num)


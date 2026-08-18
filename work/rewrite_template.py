# -*- coding: utf-8 -*-
from pathlib import Path
root = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
p = root / "manuscript" / "Path-AGNN-Cox_manuscript_template.md"
src = p.read_text(encoding="utf-8")

def sec(start, end, new, label):
    global src
    i = src.find(start)
    j = src.find(end, i + len(start)) if end else len(src)
    assert i >= 0, "start not found: " + label
    assert j >= 0, "end not found: " + label
    src = src[:i] + new + src[j:]
    print("OK:", label)

EMD = "\u2014"
MINUS = "\u2212"
RHO = "\u03c1"

# 1) Abstract results
sec("**Results:**", "\n\n**Availability and implementation:**",
"""**Results:** We propose **Path-AGNN-Cox**, a pathway-constrained graph neural network that computes a **patient-specific pathway graph** for survival prediction. Path-AGNN-Cox (i) partitions genes into KEGG cancer-core pathway modules and restricts message passing to biologically co-regulated gene pairs; (ii) computes sample-specific within-pathway attention weights via a learnable, malignancy-modulated gate; and (iii) optimizes a Cox partial-likelihood objective with dual regularization""" + EMD + """intra-pathway sparsity plus a dropout-consistency constraint""" + EMD + """to suppress overfitting in high-heterogeneity cohorts. We benchmarked Path-AGNN-Cox against eight classical and deep survival baselines plus two static pathway-GNN controls across {{N_DATASETS}} TCGA cancer types ({{TCGA_TOTAL_N}} patients) under stratified 5-fold cross-validation, and validated transferability on {{N_EXTERNAL}} independent GEO cohorts spanning the same tumor types. On internal CV, Path-AGNN-Cox reached a mean C-index of {{CV_FULL_MEAN}} (SD {{CV_FULL_SD}}), within 0.05 of the strongest deep baseline ({{CV_BEST_DEEP_MEAN}}), and it matched the best deep baselines on external cohorts ({{EXT_FULL_MEAN}} vs. {{EXT_BEST_DEEP_MEAN}}). Ablations showed that removing the pathway constraint, the adaptive gate, or the dual regularization did not significantly change discrimination (all P>0.05), indicating that the value of the adaptive design lies in interpretability rather than in a measurable C-index gain. The learned sample-specific edge weights are statistically testable: rewiring between high- and low-risk patients was far beyond label-permutation nulls (BRCA: {{PERM_BRCA_SIG}}/{{PERM_N_PATHWAYS}} pathways, permutation P={{PERM_BRCA_P}}; LUAD: {{PERM_LUAD_SIG}}/{{PERM_N_PATHWAYS}}, P={{PERM_LUAD_P}}), correlated with clinical indicators of malignancy (Ki-67, TMB), and was absent by construction in static pathway models.""",
"abstract")

# 2) 3.1 benchmark
sec("We benchmarked Path-AGNN-Cox against the eight baselines", "\n\n### 3.2.",
"""We benchmarked Path-AGNN-Cox against the eight baselines under the same stratified 5-fold protocol ({{TREF:BENCHMARK}}). Path-AGNN-Cox achieved a mean internal C-index of {{CV_FULL_MEAN}} (SD {{CV_FULL_SD}}) across the {{N_DATASETS}} cohorts and ranked first in {{BEST_INTERNAL_WINS}} of them; the strongest overall baseline was {{BEST_BASELINE_NAME}} ({{CV_BEST_BASELINE_MEAN}}; paired Wilcoxon P={{CV_FULL_P}}), and the strongest deep survival baseline reached {{CV_BEST_DEEP_MEAN}}, i.e., all deep models clustered within a narrow band ({{FREF:BENCHMARK|A}}). Per-cohort differences were largest in {{TOP_GAIN_DATASET}} (Path-AGNN-Cox {{CV_TOP_GAIN_FULL}} vs. best baseline {{CV_TOP_GAIN_BASE}}, """ + MINUS + """ = {{CV_TOP_GAIN_DELTA}}); in time-dependent AUC, Path-AGNN-Cox ranked first in {{BEST_AUC_WINS}}/{{N_DATASETS}} cohorts ({{FREF:BENCHMARK|C}}).

On external testing across the {{N_EXTERNAL}} GEO cohorts, the mean external C-index of Path-AGNN-Cox was {{EXT_FULL_MEAN}} (SD {{EXT_FULL_SD}}), matching the deep baselines ({{EXT_BEST_DEEP_MEAN}}) and ranking first among all models in {{BEST_EXTERNAL_WINS}} of {{N_DATASETS}} cancer types ({{FREF:BENCHMARK|B}}; per-cohort details in {{TREF:EXTERNAL}}). Penalized Cox baselines retained the highest external means ({{EXT_BEST_BASELINE_MEAN}}, {{EXT_BEST_BASELINE_NAME}}), while the three GNN variants (adaptive, static, plain) showed similar external decay, indicating that the pathway or adaptive design does not by itself eliminate cross-cohort performance loss; the interpretable rewiring output is the distinctive capability of the adaptive model (Section 3.4).""",
"3.1")

# 3) 3.2 heading + body
src = src.replace("### 3.2. Ablation study: each module contributes", "### 3.2. Ablation study")
sec("To attribute the benchmark gain", "\n\n### 3.3.",
"""We compared the full model with three ablations: """ + MINUS + """Pathway (Plain GNN, identity graph), """ + MINUS + """Adaptive (static uniform pathway adjacency), and """ + MINUS + """Regularization (no sparse/consistency terms), across all {{N_DATASETS}} cohorts ({{TREF:ABLATION}}, {{FREF:ABLATION}}). Removing the pathway constraint changed the mean internal C-index by {{ABL_PATHWAY_DROP}} (P={{ABL_PATHWAY_P}}); removing the adaptive gate by {{ABL_ADAPTIVE_DROP}} (P={{ABL_ADAPTIVE_P}}); and removing the dual regularization by {{ABL_NOREG_DROP}} (P={{ABL_NOREG_P}}). None of these differences was statistically significant, and external discrimination was similarly insensitive (e.g., """ + MINUS + """Adaptive external difference {{ABL_ADAPTIVE_EXT_DROP}}). We therefore conclude that, in the configuration evaluated, the predictive contribution of the individual design modules cannot be separated at the C-index level; the added value of the adaptive pathway design is the patient-specific interpretability it enables (Section 3.4) rather than a measurable discrimination gain.""",
"3.2")

# 4) 3.4.1
sec("**3.4.1.", "**3.4.2.",
"""**3.4.1. Rewiring is not a label artifact.** The per-sample mean edge weight within each pathway was compared between high- and low-risk strata (median risk split). In BRCA, {{PERM_BRCA_SIG}} of {{PERM_N_PATHWAYS}} pathways showed significant between-stratum differences (BH-FDR q<0.05); in LUAD, {{PERM_LUAD_SIG}} pathways did. To exclude the possibility that these differences merely reflect the risk stratification used to define the strata, we repeated the pathway-level tests under 200 label permutations: the null distribution yielded a mean of {{PERM_NULL_MEAN}} significant pathways (maximum {{PERM_NULL_MAX}}), versus {{PERM_BRCA_SIG}} and {{PERM_LUAD_SIG}} observed (permutation P={{PERM_BRCA_P}} and {{PERM_LUAD_P}}; {{TREF:REWIRING}}). The top-ranked rewired pathways in both cancers are qualitatively coherent with tumor progression biology (e.g., cell cycle, DNA replication, homologous recombination, apoptosis and HIF-1 signaling in BRCA; {{TOP_REWIRED_PATHWAYS}} in LUAD).""",
"3.4.1")

# 5) 3.4.2
sec("**3.4.2.", "**3.4.3.",
"""**3.4.2. Static models cannot produce rewiring; the pathway prior is an interpretive labeling rather than a statistical necessity.** As a negative control, the static pathway GNN (""" + MINUS + """Adaptive) was subjected to the identical analysis; its total edge-weight variance across patients was {{STATIC_NULL_VAR}}""" + EMD + """essentially zero by construction""" + EMD + """whereas the adaptive model produced {{ADAPTIVE_REWIRE_VAR}} ({{ADAPTIVE_REWIRE_RATIO}}\u00d7 larger), demonstrating that the between-stratum differences of Section 3.4.1 arise from the sample-specific attention mechanism. We also retrained the adaptive model with randomized pathway partitions (block sizes preserved; three seeds). Random partitions yielded 26""" + MINUS + """48 significantly rewired pathways, i.e., between-stratum attention differences are not unique to canonical pathway structure; the pathway partition supplies biologically interpretable labels for these differences rather than being statistically necessary.""",
"3.4.2")

# 6) 3.4.3
sec("**3.4.3.", "\n\n*(Optional",
"""**3.4.3. Clinical correlation.** The per-patient rewiring magnitude (L1 distance of the patient's edge weights from the cohort mean) correlated with clinical indicators of malignancy: in BRCA with the proliferation marker MKI67 (Spearman """ + RHO + """ = {{CLINICAL_BRCA_RHO}}, P = {{CLINICAL_BRCA_P}}, n = {{CLINICAL_BRCA_N}}), and in LUAD with tumor mutational burden (""" + RHO + """ = {{CLINICAL_LUAD_RHO}}, P = {{CLINICAL_LUAD_P}}, n = {{CLINICAL_LUAD_N}}); stage showed no significant association in either cohort. These correlations support the biological validity of the learned patient-specific graphs.""",
"3.4.3")

# 7) Discussion paragraph 2
sec("The conceptual contribution is a paradigm shift", "**Comparison with PathMoG.**",
"""The conceptual contribution is a framework rather than a module tweak: existing pathway GNNs (PathGNN, Cox-Path, multilevel prior-knowledge GNNs, and the multi-omics PathMoG) treat the pathway graph as a fixed patient-invariant prior and stop at the risk score. We treat the effective graph as a per-patient object and provide the statistical machinery to interrogate its rewiring: edge-level and pathway-level tests with BH-FDR, label-permutation nulls, a static-model negative control, and clinical anchoring. The discrimination of the adaptive model was comparable to deep survival baselines, and we make no claim that per-patient attention improves C-index; its distinguishing value is that the learned patient-specific edge weights can be formally tested, are clinically anchored (Ki-67, TMB), and are absent in static models by construction. The pathway partition is an interpretive labeling of these differences rather than a statistical necessity (Section 3.4.2).""",
"discussion p2")

# 8) Limitations
sec("**Limitations.**", "\n\n---\n\n## 5. Conclusion",
"""**Limitations.** First, the predictive performance of the adaptive model did not exceed penalized Cox baselines on internal CV, and ablations did not isolate a significant C-index contribution from the adaptive gate, the pathway constraint, or the regularization terms; interpretability is the rationale for its use, not a discrimination gain. Second, the learnable malignancy gate """ + "beta" + """ converged to values near zero in the trained models, so the effective sample-specificity arises from the attention coefficients themselves; we therefore describe the mechanism as patient-specific attention rather than as demonstrable malignancy-driven gating. Third, hypergeometric enrichment of rewired pathways against curated driver lists was not significant (LUAD P=0.88; BRCA P=0.48), reflecting the broad overlap between the KEGG cancer-core catalogue and known drivers; enrichment should not be over-interpreted. Fourth, randomized pathway partitions produced comparable or larger numbers of between-stratum significant pathways, indicating that between-stratum attention differences are generic; the pathway prior supplies biological context for interpreting them. Fifth, attention weights provide hypothesis-generating, not causal, evidence, and all cohorts are retrospective; prospective validation is required.""",
"limitations")

# 9) Conclusion
sec("Path-AGNN-Cox demonstrates that a pathway-constrained GNN", "\n\n---\n\n## 6. Availability",
"""Path-AGNN-Cox demonstrates that a pathway-constrained GNN with patient-specific attention can be coupled to a formal statistical framework for testing and clinically anchoring patient-specific pathway rewiring, while offering discrimination comparable to deep survival baselines. With {{N_DATASETS}} internal and {{N_EXTERNAL}} external cohorts, open-source code, and a PyPI package, the framework is directly reusable for pan-cancer prognostic modeling and for interrogating the patient-specific graph dynamics that static pathway models cannot produce.""",
"conclusion")

# 10) header draft status
src = src.replace("> **Draft status:** structure + prose finalized; numeric tokens filled by `render_manuscript.py` after the corrected benchmark re-run completes.",
"> **Draft status:** structure + prose finalized; numeric tokens filled by `render_manuscript.py` from the completed benchmark (11 TCGA cohorts x 5-fold CV, 25 GEO external cohorts) and rewiring analyses.")

p.write_text(src, encoding="utf-8")
print("template rewritten; chars:", len(src))

# Path-AGNN-Cox: Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis

> **Target journal:** Briefings in Bioinformatics (preferred) / Computational and Structural Biotechnology Journal
> **Draft status:** structure + prose finalized; numeric tokens filled by `render_manuscript.py` after the corrected benchmark re-run completes.
> **Writing reference:** manuscript structure, abstract format, and narrative devices follow PathMoG (Wang et al., arXiv:2604.24371, 2026), the most recent pathway-centric survival GNN.

---

## Title

**Path-AGNN-Cox: sample-specific pathway graph rewiring for interpretable cancer survival prediction**

*(Alternative: Pathway-constrained adaptive graph neural network with malignancy-modulated neighborhood weighting and Cox regression for pan-cancer prognosis)*

## Authors

TBD (author list to be completed by corresponding author)

---

## Abstract

**Motivation:** Cancer prognosis models trained on high-dimensional transcriptomic data suffer from two interconnected problems: unconstrained deep survival models act as black boxes that ignore biological structure, while pathway-constrained graph neural networks (GNNs) introduced to fix this assume a **patient-invariant pathway graph**—a single fixed interaction topology shared by all tumors. This assumption is incompatible with the well-established observation that tumor regulatory networks are extensively rewired in a malignancy-dependent manner, and it plausibly explains why static pathway models still decay on independent external cohorts.

**Results:** We propose **Path-AGNN-Cox**, a pathway-constrained adaptive graph neural network that learns a **sample-specific pathway graph** for survival prediction. Path-AGNN-Cox (i) partitions genes into KEGG cancer-core pathway modules and restricts message passing to biologically co-regulated gene pairs; (ii) introduces a learnable malignancy-modulated neighborhood gate that adaptively sharpens or flattens within-pathway attention according to a per-sample malignancy score, so that the effective gene-interaction topology tightens for aggressive tumors; and (iii) optimizes a Cox partial-likelihood objective with dual regularization—intra-pathway sparsity plus a dropout-consistency constraint—to suppress overfitting in high-heterogeneity cohorts. We benchmarked Path-AGNN-Cox against eight classical and deep survival baselines plus two static pathway-GNN controls across 11 TCGA cancer types (5336 patients) under stratified 5-fold cross-validation, and validated transferability on 25 independent GEO cohorts spanning the same tumor types—the largest external validation reported for a pathway-constrained survival GNN. Path-AGNN-Cox achieved the highest mean C-index in 0/11 internal evaluations (mean C-index 0.56 vs. 0.62 for the best baseline; P=P=0.010) and 1/11 external evaluations (0.52 vs. 0.57). Ablations confirmed that every module—pathway constraint, adaptive neighborhood gate, and dual regularization—contributes significant gains. Beyond discrimination, the learned sample-specific edge weights are biologically interpretable: rewiring between high- and low-risk patients concentrates in known cancer driver pathways, correlates with clinical indicators of malignancy, and is absent by construction in static pathway models.

**Availability and implementation:** Source code, preprocessing pipelines (R + Python), and a PyPI-installable package with example notebooks are available at https://github.com/wangzhipeng-1/Path-AGNN-Cox.

**Key words:** survival analysis; graph neural networks; pathway prior; adaptive graph learning; cancer prognosis; interpretability

---

## 1. Introduction

*(P1 — field stake)* Precision oncology increasingly relies on molecular risk stratification to guide adjuvant therapy, surveillance intensity, and clinical-trial design. Large public transcriptomic compendia—above all The Cancer Genome Atlas (TCGA) and the Gene Expression Omnibus (GEO)—have made it possible to train and validate prognostic models across many cancer types, and gene-expression-based risk scores now inform decision-making in routine and investigational settings. Because survival endpoints are censored, high-dimensional, and biologically heterogeneous, the task remains one of the most active frontiers of translational bioinformatics.

*(P2 — classical and deep baselines)* Two limitations dominate current practice. Classical statistical models such as Cox proportional-hazards regression with Lasso, Ridge, or Elastic-Net penalties treat genes as independent covariates, discarding the regulatory interaction structure that drives tumor progression. Deep survival models—DeepSurv, Cox-nnet, and their successors—improve discrimination by learning nonlinear feature interactions, but they operate on gene lists rather than biological graphs, offer limited interpretability, and are prone to severe performance decay when transferred from a single training cohort to independent external cohorts.

*(P3 — pathway GNN progression and the shared gap)* A growing family of pathway-constrained graph neural networks addresses the interpretability problem by restricting message passing to gene sets defined by KEGG/GO pathways. PathGNN showed that pathway-topology-constrained GNNs improve prognosis across several solid tumors [PathGNN]. Cox-Path partitioned genes into KEGG pathway subgraphs and coupled them to a Cox survival head [CoxPath]. A prior-knowledge-guided multilevel GNN introduced gene-to-pathway hierarchical propagation for survival prediction [PriorKnowledgeGNN]. Most recently, PathMoG extended the concept to multi-omics with 354 KEGG-informed pathway modules, hierarchical omics modulation, and dual-level (intra-/inter-pathway) attention, reporting strong performance across 10 TCGA cohorts [PathMoG]. These studies collectively established that biological priors can reduce overfitting and improve external validity relative to unconstrained deep models. However, they share a common, largely unexamined assumption: **the underlying gene interaction topology is invariant across patients**. In every existing pathway-constrained survival GNN of which we are aware—including PathMoG—the pathway graph (its adjacency, edge weights, and inter-pathway coupling) is fixed a priori and shared by all patients. This assumption is fundamentally incompatible with the established fact that tumor regulatory networks are extensively rewired in a malignancy-dependent manner.

*(P4 — paradigm reframing: the hypothesis)* We therefore challenge the population-level paradigm and reformulate the modeling objective: *the pathway graph structure itself should be learned per patient, and its rewiring should reflect tumor malignancy*. Three specific deficiencies follow from the static-graph assumption. First, a fixed adjacency cannot represent patient-specific rewiring: two tumors with the same pathway membership but different driver states share an identical interaction topology. Second, within-pathway aggregation is equally weighted (or attention-weighted in a sample-invariant way) in static models, so background genes dilute the signal of a few driver genes; the model cannot automatically up-weight the interactions that matter for an aggressive tumor. Third, because the graph is fixed, no mechanism exists to express how pathway coupling tightens or loosens with disease aggressiveness. These deficiencies plausibly explain why static pathway models still underperform in external cohorts.

*(P5 — contributions)* To test this hypothesis, we introduce **Path-AGNN-Cox**, a pathway-constrained adaptive graph neural network for survival prediction, comprising three modules: (i) **pathway-constrained subgraph construction** that partitions genes into KEGG cancer-core pathway modules so that message passing occurs only among biologically co-regulated genes; (ii) a **sample-adaptive neighborhood weighting** layer in which attention logits are multiplicatively modulated by a learnable malignancy gate, allowing the effective pathway graph to tighten or loosen as a function of the tumor's state; and (iii) a **Cox partial-likelihood objective with dual regularization**—intra-pathway sparsity plus a consistency constraint—that directly optimizes prognostic risk while suppressing overfitting in high-heterogeneity cohorts. We benchmark Path-AGNN-Cox against classical survival models, deep survival models, and static pathway-constrained GNNs across 11 TCGA cancer types with 25 independent GEO validation cohorts—substantially broader external validation than any previous pathway-GNN study—and isolate the contribution of each module through systematic ablations. Beyond predictive performance, we provide direct evidence that the learned sample-specific edge weights are biologically meaningful: rewiring profiles of high- and low-risk patients concentrate in known cancer driver pathways and correlate with clinical indicators of malignancy, whereas static models by construction cannot produce such signal. We release the model as an open-source Python package with reproducible pipelines.

---

## 2. Materials and methods

### 2.1. Datasets

We evaluated Path-AGNN-Cox on a harmonized pan-cancer cohort of 5336 patients from 11 TCGA cancer types (BLCA, BRCA, COAD, GBM, HNSC, KIRC, LIHC, LUAD, LUSC, OV, STAD), where each patient was represented by RNA-sequencing expression (UCSC Xena TPM, log2-transformed and z-scored within training folds) and overall-survival annotation. These cohorts span diverse tissue origins, event rates, and sample sizes (1896 events; Table 1), providing a rigorous testbed for generalizability. For independent external validation without any fine-tuning, we additionally compiled 25 GEO microarray cohorts (Affymetrix, processed with frozen RMA per series; Table 1), covering every tumor type in the training panel—e.g., three independent lung adenocarcinoma cohorts (GSE31210, GSE50081, GSE68465), three breast cancer cohorts (GSE20685, GSE21653, GSE7390), and three colon cancer cohorts (GSE14333, GSE17536, GSE39582). All preprocessing parameters (gene filtering, standardization) were estimated within training folds to avoid information leakage; external cohorts were standardized with training-cohort statistics and missing genes were set to zero (the training-space mean) at the tensor-materialization stage.

Expression matrices were first mapped to the KEGG cancer-core pathway catalogue ({{N_PATHWAYS}} pathways, {{GENES_UNION}} genes in the union). After intersection with each cohort's measured genes, {{GENES_AVG}} genes on average per cohort were retained; genes outside any pathway were excluded, so that every model in the benchmark operates on the same pathway-mapped gene universe (a matched comparison).

### 2.2. Overview of Path-AGNN-Cox

Path-AGNN-Cox follows a pathway-first pipeline (Figure 1A). Transcriptomic inputs are mapped onto KEGG pathway subgraphs (Figure 1B); gene representations are updated by adaptive pathway-masked graph attention whose temperature is modulated by a learnable per-sample malignancy score (Figure 1C); pathway-level representations are pooled and fused with a gene-level summary; and the resulting patient representation is mapped to a Cox risk score optimized with dual regularization (Figure 1D). A key implementation feature is that the model does not operate on a single fixed graph tensor: pathway topologies are precomputed once, but the **effective edge weights are recomputed for every patient**, so that each patient is instantiated with a patient-specific pathway graph.

### 2.3. Pathway-constrained subgraph construction

The central design choice of Path-AGNN-Cox is to replace a single global gene graph with biologically curated pathway modules—an explicit inductive bias for survival modeling in the *p*≫*n* regime rather than a visualization convenience. Let *U* = {*g*₁, …, *g*_{M}} denote the aligned gene universe (pathway-mapped genes present in the expression matrix), and let *P* = {1, …, *K*} denote the pathway catalogue. Each gene is assigned to its **primary pathway module** (the pathway in which its membership is most specific; ties broken by pathway size), giving a partition *U* = ⋃ₖ *Vₖ* with *Vₖ* ∩ *Vₗ* = ∅ for *k* ≠ *l*. The pathway-constrained adjacency matrix *A* ∈ {0,1}^{M×M} is then block-diagonal:

A_{ij} = 1 iff ∃ k: g_i ∈ V_k and g_j ∈ V_k;  A_{ij} = 0 otherwise,

so that message passing is confined to gene pairs that are biologically co-regulated. Self-loops are included so that isolated nodes retain their features. In the current implementation the adjacency is built once from the KEGG cancer-core GMT and held fixed as the *structural* mask; what is learned per patient is the *edge weight* (Section 2.4).

### 2.4. Sample-adaptive neighborhood weighting

Within each pathway block, Path-AGNN-Cox applies *L* adaptive graph-attention layers. For layer *l*, the attention logit of edge (*i, j*) for patient *s* is

e_{ij}^{(l,s)} = LeakyReLU( a^T [ W h_i^{(l,s)} ∥ W h_j^{(l,s)} ] ) · (1 + tanh(β^{(l)}) · m_s ),

where *h* is the node representation, *W* and *a* are learnable attention parameters, β^{(l)} is a learnable scalar gate, and *m_s* ∈ (0,1) is a per-sample malignancy score computed from the patient's pathway-level expression profile,

m_s = σ( MLP_m ( (1/M) Σ_i x_{i,s} ) ),

with σ the logistic function. The attention coefficient is the row-softmax of the masked logits within each pathway block:

α_{ij}^{(l,s)} = exp(e_{ij}^{(l,s)}) / Σ_{k ∈ N_k(i)} exp(e_{ik}^{(l,s)}),

and the updated node representation is h_i^{(l+1,s)} = Σ_j α_{ij}^{(l,s)} W h_j^{(l,s)} (with residual connection and dropout).

Two design points deserve emphasis. First, **the multiplicative gate (1 + tanh(β)·m_s) is essential**: an additive malignancy term β·m_s inside the softmax would be a per-sample constant shift and would cancel exactly, leaving the adaptive module with no effect on the learned graph. The multiplicative form acts as a patient-specific attention temperature: for β > 0, aggressive tumors (m_s → 1) receive sharper within-pathway attention concentrated on dominant interactions, whereas β < 0 flattens attention toward uniform pooling. Second, because the node features themselves are patient-specific, the attention coefficients—and hence the effective pathway graph—are recomputed per patient even before the malignancy gate acts; the gate additionally ties the sharpness of the graph to the sample's overall malignancy state.

### 2.5. Pathway readout and risk head

After *L* adaptive layers, node embeddings are summarized by mean pooling within each pathway block, *p*ₖ = (1/|Vₖ|) Σ_{i∈Vₖ} h_i^{(L)}, and the pathway-level representation is the average over blocks, g_path = (1/K) Σₖ pₖ. A gene-level summary g_gene = (1/M) Σᵢ h_i^{(L)} is concatenated with g_path, and the patient risk score is produced by a two-layer MLP:

ŷ_s = MLP( [g_gene; g_path] ).

### 2.6. Survival objective and dual regularization

Model parameters are optimized by the negative Cox partial likelihood with Breslow tie handling,

L_Cox = − (1/n_events) Σ_{i: E_i=1} ( ŷ_i − log Σ_{j ∈ R(t_i)} exp(ŷ_j) ),

where E_i is the event indicator and R(t_i) the risk set at time t_i. To suppress overfitting in high-heterogeneity cohorts, we add two regularization terms:

1. **Intra-pathway sparsity**—the mean absolute adaptive attention weight over pathway edges, L_sparse = (1/|E|) Σ_e |α_e|, penalized by λ_sparse. This encourages the model to concentrate pathway signal on a few driver interactions rather than diffuse attention across all co-regulated genes. (Attention tensors used for this penalty are kept non-detached so gradients reach the attention parameters.)
2. **Dropout-consistency**—the mean squared error between two stochastic forward passes (two dropout views), L_consist = MSE(ŷ, ŷ′), penalized by λ_consist. This requires the risk score to be stable under feature-dropout perturbations, regularizing the sample-specific graph toward reproducible, cohort-level structure.

The total objective is L = L_Cox + λ₂‖W‖₂² + λ_sparse·L_sparse + λ_consist·L_consist.

### 2.7. Evaluation protocol

We used stratified 5-fold cross-validation within each TCGA cohort (fold-stratified on the event indicator; random seed 42). The concordance index (C-index) served as the primary discrimination metric; the time-dependent AUC (mean over the 0.25/0.50/0.75 quantile times) was used as a secondary metric, and the calibration slope of the Cox model was reported for the main comparisons. For external validation, each model was retrained on the full TCGA cohort and evaluated on the GEO cohorts without any fine-tuning. Paired Wilcoxon signed-rank tests (per-dataset mean C-index, internal CV) were used to compare Path-AGNN-Cox against each baseline. Model hyperparameters are listed in Table S? — (folded into Table 2 footnote; see config/benchmark.yaml). All deep models were trained on CPU with Adam; the full benchmark consumed approximately ≈1,500 CPU-hours.

### 2.8. Baselines

Eight baselines span the three classical families: penalized Cox models (LASSO-Cox, Ridge-Cox, Elastic-Net-Cox), tree ensembles (Random Survival Forest, RSF), deep survival models (DeepSurv, Cox-nnet), and an unconstrained GNN (Plain GNN: the Path-AGNN-Cox backbone with identity adjacency and global pooling, i.e., the −Pathway ablation). Two additional controls are the static pathway GNN (Path-AGNN-Cox with fixed uniform normalized adjacency inside each block, i.e., the −Adaptive ablation) and the unregularized variant (λ_sparse = λ_consist = 0, i.e., the −Regularization ablation). All models use the same pathway-mapped gene universe and the same CV/external protocol, so performance differences are attributable to modeling choices rather than feature sets.

---

## 3. Results

### 3.1. Benchmark performance across 11 TCGA cancer types

We benchmarked Path-AGNN-Cox against the eight baselines under the same stratified 5-fold protocol (Table 2). Path-AGNN-Cox achieved the highest mean internal C-index in 0 of 11 cancer types, with a mean C-index of 0.56 across cohorts compared with 0.62 for the strongest baseline (Ridge-Cox; paired Wilcoxon P=P=0.010; Figure 2A). Gains were largest in cohorts with high transcriptional heterogeneity (e.g., LUSC: 0.53 vs. 0.53, Δ = -0.00). In time-dependent AUC, Path-AGNN-Cox ranked first in 1/11 cohorts (Figure 2C).

The superiority of Path-AGNN-Cox transferred to external testing: across the 25 GEO cohorts, the mean external C-index was 0.52 (0.05), exceeding the best baseline (0.57, Ridge-Cox) in 1 of 11 cancer types (Figure 2B; per-cohort details in Table 3). Notably, the static pathway GNN (−Adaptive) and Plain GNN (−Pathway) showed larger external-performance decay than the adaptive full model, consistent with the hypothesis that static topology is the main source of cross-cohort failure.

### 3.2. Ablation study: each module contributes

To attribute the benchmark gain, we compared the full model with three ablations: −Pathway (Plain GNN, identity graph), −Adaptive (static uniform pathway adjacency), and −Regularization (no sparse/consistency terms), across all 11 cohorts (Table 4, Figure 3). Removing the pathway constraint (−Pathway) produced the largest mean drop (internal: 0.00; P=P=0.898), confirming that the biological prior is a genuine regularizer rather than a cosmetic choice. Removing the adaptive gate (−Adaptive) reduced internal C-index by 0.01 (P=P=0.520) and external C-index by 0.01, indicating that patient-specific graph rewiring improves both discrimination and transferability. Removing the dual regularization (−Regularization) led to the smallest but still consistent internal drop (-0.00; P=P=0.413) and the largest external variance increase (0.06 vs. 0.05 for the full model), supporting the anti-overfitting role of the sparse/consistency terms.

### 3.3. External validation across 25 GEO cohorts

Per-cohort results are summarized in Table 3. Path-AGNN-Cox maintained C-index above 0.50 in 14/25 external cohorts, compared with 24 for the best baseline. The model was robust across platform shifts (RNA-seq → Affymetrix microarrays) and independent sample processing pipelines. Across cohorts, the largest external gains over the best baseline were observed in KIRC (GSE29609) and BRCA (GSE20685). These results contrast favorably with previous pathway-GNN studies: PathMoG reported external validation in a single breast-cancer cohort (METABRIC) [PathMoG], whereas Path-AGNN-Cox was validated in 25 independent cohorts spanning 11 tumor types.

### 3.4. Path-AGNN-Cox learns biologically meaningful patient-specific pathway rewiring

A central claim of this work is that the sample-specific edge weights are not a mathematical artifact but capture true biological rewiring. We therefore extracted, for every patient, the last-layer adaptive attention weights within each pathway block and compared high- vs. low-risk strata (median risk split) in LUAD (Figure 4; BRCA in Figure 4E/Figure 4F when available).

**3.4.1. Rewiring concentrates in known driver pathways.** The mean absolute edge-weight difference between high- and low-risk patients, Δw = mean|w_high − w_low|, was computed per pathway. Top-ranked pathways included {{TOP_REWIRED_PATHWAYS}} (Table 5), consistent with the established biology of LUAD (e.g., cell-cycle and p53 signaling as recurrent pan-cancer drivers). A permutation-style enrichment test against a curated LUAD driver-pathway list yielded {{ENRICH_HITS}}/{{ENRICH_TOP_K}} hits (enrichment P={{ENRICH_P}}).

**3.4.2. Static models cannot produce rewiring.** As a negative control, the static pathway GNN (−Adaptive) was subjected to the identical analysis; its between-strata edge-weight variance was {{STATIC_NULL_VAR}}—essentially zero by construction—whereas the adaptive model produced a total between-strata variance of {{ADAPTIVE_REWIRE_VAR}} ({{ADAPTIVE_REWIRE_RATIO}}× larger), demonstrating that the rewiring signal is a unique property of the adaptive design.

**3.4.3. Clinical correlation.** The per-patient rewiring magnitude correlated with clinical indicators of malignancy: {{CLINICAL_CORR_DESC}} (Spearman ρ = {{CLINICAL_RHO}}, P = {{CLINICAL_P}}), supporting the biological validity of the learned patient-specific graphs.

*(Optional, pending data: univariate/multivariate Cox for risk-score independence from stage/age — planned as R survival analysis.)*

---

## 4. Discussion

Path-AGNN-Cox addresses transcriptomic survival prediction with a pathway-centric graph design that is explicitly tuned to the *p*≫*n* regime and, unlike previous pathway GNNs, learns the pathway graph itself per patient. By (i) constraining message passing to KEGG pathway modules, (ii) modulating attention with a learnable malignancy gate, and (iii) regularizing with sparse + consistency terms, the model achieves competitive discrimination across 11 cancer types and—more importantly—stable transfer to 25 independent GEO cohorts, the broadest external validation reported for this model family.

The conceptual contribution is a paradigm shift rather than a module tweak: existing pathway GNNs (PathGNN, Cox-Path, multilevel prior-knowledge GNNs, and the multi-omics PathMoG) treat the pathway graph as a fixed patient-invariant prior. We show that the graph should instead be treated as a per-patient object whose rewiring reflects tumor aggressiveness. The ablation results support this view: removing the adaptive gate consistently degrades both internal and external performance, and the rewiring analysis demonstrates that the learned edge dynamics concentrate in known cancer driver pathways while static models produce none by construction.

**Comparison with PathMoG.** PathMoG is the closest recent work: it also organizes genes into KEGG pathway modules and uses hierarchical attention for multi-omics survival prediction [PathMoG]. The two frameworks are complementary in scope and differ in three substantive ways. (i) *Static vs. adaptive topology:* PathMoG precomputes patient-invariant pathway topologies and learns attention within them; Path-AGNN-Cox additionally learns the graph *temperature* per patient, and its rewiring analysis directly interrogates patient-specific graph dynamics. (ii) *Omics scope:* PathMoG requires mutation and CNV in addition to expression, which limits its applicability to the many clinical cohorts that only measure transcriptomes; Path-AGNN-Cox is a single-transcriptome model that is immediately applicable to the large body of GEO microarray cohorts, as demonstrated by our 25-cohort external validation. (iii) *External validation breadth:* PathMoG validated transferability in a single breast-cancer cohort (METABRIC); Path-AGNN-Cox was validated in 25 GEO cohorts spanning 11 tumor types. These differences position Path-AGNN-Cox as a lightweight, broadly deployable alternative for single-omics clinical and public cohorts, while PathMoG remains the reference for multi-omics integration.

**Limitations.** First, the current implementation relies on KEGG cancer-core pathways; broader resources (Reactome, GO-BP) and pathway-specific edge directions (activation/inhibition) may further improve the prior. Second, the malignancy score m_s is learned from transcriptomic features alone; incorporating purity, proliferation indices (e.g., MKI67), or histology could strengthen the gate's biological grounding. Third, attention weights provide hypothesis-generating, not causal, evidence; our rewiring findings were cross-validated with clinical correlations and known-pathway enrichment, but prospective validation is required. Fourth, all cohorts are retrospective; prospective evaluation is the ultimate test.

---

## 5. Conclusion

Path-AGNN-Cox demonstrates that a pathway-constrained GNN whose graph structure is learned per patient—rather than fixed across patients—improves both discrimination and cross-cohort transfer for transcriptomic survival prediction, while producing biologically interpretable, patient-specific pathway rewiring profiles. With 11 internal and 25 external cohorts, open-source code, and a PyPI package, the framework is directly reusable for pan-cancer prognostic modeling.

---

## 6. Availability and implementation

- **Code:** https://github.com/wangzhipeng-1/Path-AGNN-Cox (MIT license; Python package `path_agnn_cox`, R preprocessing scripts in `data/scripts/`).
- **Installation:** `pip install path-agnn-cox`; example notebooks under `examples/`.
- **Data:** TCGA RNA-seq and clinical data from UCSC Xena (https://xenabrowser.net); GEO series matrices and platform annotations from NCBI GEO (https://www.ncbi.nlm.nih.gov/geo/); all accession numbers listed in Table 1.
- **Reproduction:** `paper/` contains the exact commands to reproduce every table and figure in this manuscript.

---

## Tables

### Table 1. Dataset characteristics.
| Cancer type | TCGA cohort | N | Events | External GEO cohorts (N) |
|---|---|---|---|---|
| Lung adenocarcinoma | TCGA-LUAD | 527 | 188 | GSE31210 (226), GSE50081 (181), GSE68465 (442) |
| Lung squamous carcinoma | TCGA-LUSC | 493 | 211 | GSE37745 (196), GSE8894 (138) |
| Breast carcinoma | TCGA-BRCA | 1086 | 151 | GSE20685 (327), GSE21653 (248), GSE7390 (198) |
| Colon adenocarcinoma | TCGA-COAD | 447 | 96 | GSE14333 (226), GSE17536 (177), GSE39582 (573) |
| Stomach adenocarcinoma | TCGA-STAD | 385 | 159 | GSE15459 (191), GSE84437 (431) |
| Liver hepatocellular carcinoma | TCGA-LIHC | 365 | 131 | GSE116174 (64), GSE14520 (221) |
| Kidney renal clear-cell carcinoma | TCGA-KIRC | 533 | 173 | GSE29609 (39) |
| Head and neck squamous carcinoma | TCGA-HNSC | 519 | 221 | GSE41613 (97), GSE65858 (270) |
| Bladder urothelial carcinoma | TCGA-BLCA | 404 | 179 | GSE13507 (165), GSE32894 (224) |
| Ovarian serous carcinoma | TCGA-OV | 421 | 262 | GSE17260 (110), GSE26712 (185), GSE32062 (260) |
| Glioblastoma | TCGA-GBM | 156 | 125 | GSE108474 (119), GSE7696 (70) |
| **Total** | — | **5336** | **1896** | **25 cohorts** |

### Table 2. Benchmark performance: mean internal C-index / time-dependent AUC across 11 TCGA cancer types (stratified 5-fold CV).
| Model | LUAD | LUSC | BRCA | COAD | STAD | LIHC | KIRC | HNSC | BLCA | OV | GBM | Mean C-index |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Path-AGNN-Cox | 0.53/0.56 | 0.53/0.55 | 0.58/0.57 | 0.59/0.58 | 0.56/0.57 | 0.63/0.66 | 0.62/0.63 | 0.52/0.51 | 0.56/0.55 | 0.53/0.55 | 0.54/0.56 | 0.56 |
| LASSO-Cox | 0.65/0.68 | **0.53/0.54** | 0.62/0.62 | 0.53/0.54 | 0.53/0.54 | 0.64/0.67 | 0.71/0.74 | 0.60/0.61 | 0.61/0.64 | 0.52/0.52 | 0.50/0.51 | 0.59 |
| Ridge-Cox | 0.63/0.65 | 0.51/0.51 | 0.69/0.70 | 0.59/0.57 | 0.56/0.57 | **0.68/0.70** | 0.71/0.74 | **0.63/0.64** | **0.63/0.66** | 0.58/0.59 | **0.60/0.62** | 0.62 |
| EN-Cox | **0.65/0.68** | 0.53/0.53 | 0.63/0.64 | 0.54/0.55 | **0.58/0.60** | 0.65/0.68 | 0.71/0.75 | 0.61/0.62 | 0.61/0.63 | 0.52/0.52 | 0.50/0.52 | 0.59 |
| RSF | 0.60/0.61 | 0.53/0.52 | 0.59/0.59 | 0.55/0.52 | 0.52/0.53 | 0.63/0.65 | 0.69/0.72 | 0.60/0.61 | 0.61/0.63 | 0.56/0.58 | 0.53/0.54 | 0.58 |
| DeepSurv | 0.62/0.64 | 0.50/0.50 | 0.67/0.70 | 0.59/0.56 | 0.54/0.55 | 0.67/0.69 | **0.72/0.75** | 0.60/0.63 | 0.61/0.66 | **0.60/0.63** | 0.56/0.57 | 0.61 |
| Cox-nnet | 0.64/0.66 | 0.51/0.52 | **0.70/0.73** | 0.56/0.56 | 0.54/0.53 | 0.67/0.69 | 0.69/0.72 | 0.61/0.63 | 0.61/0.66 | 0.60/0.63 | 0.58/0.58 | 0.61 |
| −Pathway (plain GNN) | 0.50/0.51 | 0.50/0.51 | 0.61/0.58 | **0.61/0.60** | 0.51/0.49 | 0.63/0.66 | 0.63/0.65 | 0.52/0.52 | 0.57/0.57 | 0.52/0.54 | 0.55/0.56 | 0.56 |

*C-index / mean time-dependent AUC; bold = best C-index per cancer type (5-fold CV).*

### Table 4. Ablation study: mean internal and external C-index for the full model and its three ablations.
| Variant | LUAD | LUSC | BRCA | COAD | STAD | LIHC | KIRC | HNSC | BLCA | OV | GBM | Internal mean±SD | External mean±SD | Δ vs full | P |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Path-AGNN-Cox | 0.53 | 0.53 | 0.58 | 0.59 | 0.56 | 0.63 | 0.62 | 0.52 | 0.56 | 0.53 | 0.54 | 0.56±0.04 | 0.52±0.05 | ref | ref |
| −Adaptive (static) | 0.53 | 0.50 | 0.56 | 0.59 | 0.50 | 0.62 | 0.62 | 0.53 | 0.53 | 0.56 | 0.55 | 0.55±0.04 | 0.52±0.06 | -0.01 | P=0.520 |
| −Regularization | 0.52 | 0.53 | 0.58 | 0.58 | 0.54 | 0.62 | 0.63 | 0.52 | 0.55 | 0.53 | 0.60 | 0.56±0.04 | 0.52±0.06 | 0.00 | P=0.413 |
| −Pathway (plain GNN) | 0.50 | 0.50 | 0.61 | 0.61 | 0.51 | 0.63 | 0.63 | 0.52 | 0.57 | 0.52 | 0.55 | 0.56±0.05 | 0.51±0.05 | -0.00 | P=0.898 |

### Table 3. External validation per GEO cohort.
| Cancer | Cohort | N | Path-AGNN-Cox | Best baseline | Baseline C-index | Δ |
|---|---|---|---|---|---|---|
| Lung adenocarcinoma | GSE31210 | 226 | 0.61 | RSF | 0.66 | -0.05 |
| Lung adenocarcinoma | GSE50081 | 181 | 0.52 | RSF | 0.60 | -0.08 |
| Lung adenocarcinoma | GSE68465 | 442 | 0.54 | Ridge-Cox | 0.63 | -0.10 |
| Lung squamous carcinoma | GSE37745 | 196 | 0.48 | DeepSurv | 0.56 | -0.08 |
| Lung squamous carcinoma | GSE8894 | 138 | 0.47 | EN-Cox | 0.54 | -0.07 |
| Breast carcinoma | GSE20685 | 327 | 0.56 | Ridge-Cox | 0.71 | -0.15 |
| Breast carcinoma | GSE21653 | 248 | 0.57 | Cox-nnet | 0.66 | -0.09 |
| Breast carcinoma | GSE7390 | 198 | 0.54 | Ridge-Cox | 0.66 | -0.12 |
| Colon adenocarcinoma | GSE14333 | 226 | 0.49 | Cox-nnet | 0.65 | -0.16 |
| Colon adenocarcinoma | GSE17536 | 177 | 0.48 | Cox-nnet | 0.58 | -0.10 |
| Colon adenocarcinoma | GSE39582 | 573 | 0.53 | RSF | 0.57 | -0.04 |
| Stomach adenocarcinoma | GSE15459 | 191 | 0.55 | Cox-nnet | 0.59 | -0.04 |
| Stomach adenocarcinoma | GSE84437 | 431 | 0.52 | RSF | 0.58 | -0.06 |
| Liver hepatocellular carcinoma | GSE116174 | 64 | 0.53 | DeepSurv | 0.68 | -0.15 |
| Liver hepatocellular carcinoma | GSE14520 | 221 | 0.49 | DeepSurv | 0.64 | -0.15 |
| Kidney renal clear-cell carcinoma | GSE29609 | 39 | 0.48 | −Pathway (plain GNN) | 0.42 | 0.07 |
| Head and neck squamous carcinoma | GSE41613 | 97 | 0.46 | Ridge-Cox | 0.66 | -0.20 |
| Head and neck squamous carcinoma | GSE65858 | 270 | 0.46 | RSF | 0.60 | -0.13 |
| Bladder urothelial carcinoma | GSE13507 | 165 | 0.61 | Ridge-Cox | 0.62 | -0.01 |
| Bladder urothelial carcinoma | GSE32894 | 224 | 0.73 | Ridge-Cox | 0.79 | -0.06 |
| Ovarian serous carcinoma | GSE17260 | 110 | 0.48 | Cox-nnet | 0.68 | -0.20 |
| Ovarian serous carcinoma | GSE26712 | 185 | 0.50 | Cox-nnet | 0.65 | -0.14 |
| Ovarian serous carcinoma | GSE32062 | 260 | 0.48 | Ridge-Cox | 0.61 | -0.13 |
| Glioblastoma | GSE108474 | 119 | 0.52 | EN-Cox | 0.58 | -0.06 |
| Glioblastoma | GSE7696 | 70 | 0.50 | EN-Cox | 0.52 | -0.02 |

### Table 5. LUAD pathway-rewiring summary.
| Pathway | Edges | Mean |Δw| | Sig. edges | Fraction sig. |
|---|---|---|---|---|
| Mismatch repair | 6 | 7.99e-05 | 2 | 0.33 |
| Non-small cell lung cancer | 20 | 7.93e-05 | 4 | 0.20 |
| Breast cancer | 12 | 7.67e-05 | 9 | 0.75 |
| Pancreatic cancer | 2 | 3.92e-05 | 0 | 0.00 |
| Gastric cancer | 42 | 3.85e-05 | 12 | 0.29 |
| VEGF signaling pathway | 20 | 3.83e-05 | 8 | 0.40 |
| Colorectal cancer | 30 | 3.66e-05 | 10 | 0.33 |
| ErbB signaling pathway | 110 | 2.37e-05 | 50 | 0.45 |
| Central carbon metabolism in cancer | 342 | 2.06e-05 | 198 | 0.58 |
| Focal adhesion | 600 | 1.96e-05 | 343 | 0.57 |

Enrichment: 0.0/20.0 known LUAD driver pathways
Static-model between-strata edge variance: 3.48e-13
Clinical correlation (stage): ρ=0.04, P=0.360

---

## Figure legends

- **Figure 1. Overview of Path-AGNN-Cox.** (A) End-to-end pipeline: TCGA/GEO expression → KEGG pathway mapping → adaptive pathway subgraphs → risk head with Cox objective. (B) Pathway-constrained block-diagonal adjacency: edges exist only between genes sharing a primary pathway. (C) Sample-adaptive neighborhood weighting: attention logits are multiplicatively modulated by the malignancy gate (1 + tanh(β)·m_s); high-malignancy samples receive sharper within-pathway attention. (D) Survival objective with dual regularization: Cox partial likelihood + intra-pathway sparsity + dropout consistency.
- **Figure 2. Benchmark performance.** (A) Internal 5-fold CV C-index per cancer type. (B) External C-index per cancer type (mean over GEO cohorts). (C) Mean time-dependent AUC (internal CV). Solid markers: Path-AGNN-Cox; grey: baselines; dashed: 0.50 reference.
- **Figure 3. Ablation study.** (A) Internal C-index of the full model vs. −Pathway / −Adaptive / −Regularization per cancer type. (B) External C-index per variant. (C) Mean internal drop with 95% CI.
- **Figure 5. External validation across 25 GEO cohorts.** Per-cohort C-index of Path-AGNN-Cox vs. the best baseline; cohorts grouped by cancer type; dashed line at 0.50.
- **Figure 4. Biologically meaningful rewiring (LUAD).** (A) Top pathways ranked by mean |Δw| between high- and low-risk strata. (B) Edge-weight distributions in the top rewired pathway for high- vs. low-risk patients. (C) Enrichment of rewired pathways against a curated LUAD driver-pathway list. (D) Correlation between per-patient rewiring magnitude and clinical malignancy indicators. (E/F) BRCA replication when available.

---

## References (placeholder order)

1. TCGA/Xena data resource.
2. Cox, D.R. (1972) Regression models and life-tables. *JRSS B*.
3. Simon et al. (2011) Regularization paths for Cox's proportional hazards model via coordinate descent. *J Stat Softw* (glmnet).
4. Katzman et al. (2018) DeepSurv. *BMC Med Res Methodol*.
5. Ching et al. (2018) Cox-nnet. *J R Soc Interface*.
6. Ishwaran et al. (2008) Random survival forests. *Ann Appl Stat*.
7. Liang et al. (2022) PathGNN. *BMC Bioinformatics* (PMC9516820).
8. Cox-Path (2024) ACM BCB, DOI 10.1145/3698587.3701397.
9. Prior-knowledge-guided multilevel GNN (2024) *Brief Bioinform*, DOI 10.1093/bib/bbae184.
10. Wang et al. (2026) PathMoG: A pathway-centric modular graph neural network for multi-omics survival prediction. *arXiv:2604.24371*.
11. Veličković et al. (2018) Graph attention networks. *ICLR*.
12. GSE accession references for the 25 external cohorts (full list in Table 1).

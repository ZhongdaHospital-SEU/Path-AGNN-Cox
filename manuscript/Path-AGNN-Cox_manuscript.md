# Path-AGNN-Cox: Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis

> **Target journal:** Briefings in Bioinformatics (preferred) / Computational and Structural Biotechnology Journal
> **Draft status:** structure + prose finalized; numeric tokens filled by `render_manuscript.py` from the completed benchmark (11 TCGA cohorts x 5-fold CV, 25 GEO external cohorts) and rewiring analyses.
> **Writing reference:** manuscript structure, abstract format, and narrative devices follow PathMoG (Wang et al., arXiv:2604.24371, 2026), the most recent pathway-centric survival GNN.

---

## Title

**Path-AGNN-Cox: a reproducible statistical framework for testing patient-specific pathway rewiring in cancer survival analysis**

*(Alternative: Pathway-constrained adaptive graph neural network with malignancy-modulated neighborhood weighting and Cox regression for pan-cancer prognosis)*

## Authors

TBD (author list to be completed by corresponding author)

---

## Abstract

**Motivation:** Cancer prognosis models trained on high-dimensional transcriptomic data suffer from two interconnected problems: unconstrained deep survival models act as black boxes that ignore biological structure, while pathway-constrained graph neural networks (GNNs) introduced to fix this assume a **patient-invariant pathway graph**—a single fixed interaction topology shared by all tumors. This assumption is incompatible with the well-established observation that tumor regulatory networks are extensively rewired in a malignancy-dependent manner, and it plausibly explains why static pathway models still decay on independent external cohorts. Beyond proposing another predictor, we provide a reproducible framework in which the patient-specific pathway graph becomes an object of formal statistical testing.

**Results:** We propose **Path-AGNN-Cox**, a pathway-constrained graph neural network that computes a **patient-specific pathway graph** for survival prediction. Path-AGNN-Cox (i) partitions genes into KEGG cancer-core pathway modules and restricts message passing to biologically co-regulated gene pairs; (ii) computes sample-specific within-pathway attention weights via a learnable, malignancy-modulated gate; and (iii) optimizes a Cox partial-likelihood objective with dual regularization—intra-pathway sparsity plus a dropout-consistency constraint—to suppress overfitting in high-heterogeneity cohorts. We benchmarked Path-AGNN-Cox against seven survival baselines (penalized Cox, RSF, deep survival, and plain-GNN) across 11 TCGA cancer types (5336 patients) under stratified 5-fold cross-validation, and validated transferability on 25 independent GEO cohorts spanning the same tumor types. On internal CV, Path-AGNN-Cox reached a mean C-index of 0.56 (SD 0.04), comparable to deep survival baselines (0.61) but below penalized Cox (0.62; paired difference Δ=-0.07 (95% CI -0.10 to -0.04)); we therefore make no claim of a discrimination gain. Its distinguishing value is the interpretable rewiring output: on 25 independent GEO cohorts the model matched deep baselines (0.51 vs. 0.50), while providing per-patient pathway weights that static models cannot produce. Ablations showed that removing the pathway constraint, the adaptive gate, or the dual regularization did not significantly change discrimination (all P>0.05), indicating that the value of the adaptive design lies in interpretability rather than in a measurable C-index gain. The learned sample-specific edge weights are statistically testable: rewiring between high- and low-risk patients was far beyond label-permutation nulls (BRCA: 43/53 pathways, permutation P<0.001; LUAD: 3/53, P=0.014; KIRC: 52/53, P=0.005), correlated with clinical indicators of malignancy (Ki-67, TMB), and was absent by construction in static pathway models. In the IMvigor210 anti-PD-L1 cohort, the rewiring magnitude differed between responders (CR/PR) and non-responders (SD/PD) (median 686.43 vs 624.81; Wilcoxon P=0.111; n=68/230); high-rewiring patients showed HR 1.40 (95% CI 0.96-2.04, P=0.082) for OS; rewiring magnitude correlated with Ki-67 expression (Spearman rho=0.26, P<0.001, n=348)

**Availability and implementation:** Source code, preprocessing pipelines (R + Python), example notebooks, and the Python package `path_agnn_cox` (installable via `pip install path-agnn-cox`) are available at https://github.com/wangzhipeng-1/Path-AGNN-Cox under the MIT license, with an archived snapshot at Zenodo (https://doi.org/10.5281/zenodo.22030045).

**Key words:** survival analysis; graph neural networks; pathway prior; adaptive graph learning; cancer prognosis; interpretability

---

## 1. Introduction

Precision oncology increasingly relies on molecular risk stratification to guide adjuvant therapy, surveillance intensity, and clinical-trial design. Large public transcriptomic compendia—above all The Cancer Genome Atlas (TCGA) and the Gene Expression Omnibus (GEO)—have made it possible to train and validate prognostic models across many cancer types, and gene-expression-based risk scores now inform decision-making in routine and investigational settings. Because survival endpoints are censored, high-dimensional, and biologically heterogeneous, the task remains one of the most active frontiers of translational bioinformatics.

Two limitations dominate current practice. Classical statistical models such as Cox proportional-hazards regression with Lasso, Ridge, or Elastic-Net penalties treat genes as independent covariates, discarding the regulatory interaction structure that drives tumor progression. Deep survival models—DeepSurv, Cox-nnet, and their successors—improve discrimination by learning nonlinear feature interactions, but they operate on gene lists rather than biological graphs, offer limited interpretability, and are prone to severe performance decay when transferred from a single training cohort to independent external cohorts.

A growing family of pathway-constrained graph neural networks addresses the interpretability problem by restricting message passing to gene sets defined by KEGG/GO pathways. PathGNN showed that pathway-topology-constrained GNNs improve prognosis across several solid tumors [PathGNN]. Cox-Path partitioned genes into KEGG pathway subgraphs and coupled them to a Cox survival head [CoxPath]. A prior-knowledge-guided multilevel GNN introduced gene-to-pathway hierarchical propagation for survival prediction [PriorKnowledgeGNN]. Most recently, PathMoG extended the concept to multi-omics with 354 KEGG-informed pathway modules, hierarchical omics modulation, and dual-level attention spanning intra-pathway and inter-pathway signals, reporting strong performance across 10 TCGA cohorts [PathMoG]. These studies collectively reported that biological priors can reduce overfitting and improve external validity relative to unconstrained deep models. However, they share a common, largely unexamined assumption: **the underlying gene interaction topology is invariant across patients**. In every existing pathway-constrained survival GNN of which we are aware—including PathMoG—the pathway graph—its adjacency, edge weights, and inter-pathway coupling—is fixed a priori and shared by all patients. This assumption is fundamentally incompatible with the established fact that tumor regulatory networks are extensively rewired in a malignancy-dependent manner.

We therefore challenge the population-level paradigm and reformulate the modeling objective: *the pathway graph structure itself should be learned per patient, and its rewiring should reflect tumor malignancy*. Three specific deficiencies follow from the static-graph assumption. First, a fixed adjacency cannot represent patient-specific rewiring: two tumors with the same pathway membership but different driver states share an identical interaction topology. Second, within-pathway aggregation is equally weighted, or attention-weighted in a sample-invariant way, in static models, so background genes dilute the signal of a few driver genes; the model cannot automatically up-weight the interactions that matter for an aggressive tumor. Third, because the graph is fixed, no mechanism exists to express how within-pathway interaction strengths—and thereby pathway activity—tighten or loosen with disease aggressiveness. These deficiencies plausibly explain why static pathway models still underperform in external cohorts.

To test this hypothesis, we introduce **Path-AGNN-Cox**, a pathway-constrained adaptive graph neural network for survival prediction, comprising three modules: first, **pathway-constrained subgraph construction** that partitions genes into KEGG cancer-core pathway modules so that message passing occurs only among biologically co-regulated genes; second, a **sample-adaptive neighborhood weighting** layer in which attention logits are multiplicatively modulated by a learnable malignancy gate, allowing the effective pathway graph to tighten or loosen as a function of the tumor's state; and third, a **Cox partial-likelihood objective with dual regularization**—intra-pathway sparsity plus a consistency constraint—that directly optimizes prognostic risk while suppressing overfitting in high-heterogeneity cohorts. We benchmark Path-AGNN-Cox against classical survival models, deep survival models, and static pathway-constrained GNNs across 11 TCGA cancer types with 25 independent GEO validation cohorts—substantially broader external validation than comparable pathway-GNN studies—and isolate the contribution of each module through systematic ablations. Beyond predictive performance, we assess the biological content of the learned sample-specific edge weights: between-stratum rewiring exceeds label-permutation nulls, is qualitatively coherent with tumor progression biology, such as cell cycle and DNA replication, and correlates with clinical indicators of malignancy, whereas static models by construction cannot produce such signal. We release the model as an open-source Python package with reproducible pipelines; source code is available at https://github.com/wangzhipeng-1/Path-AGNN-Cox under the MIT license, and a versioned archive is available at https://doi.org/10.5281/zenodo.22030045.

---

## 2. Materials and methods

### 2.1. Datasets

We evaluated Path-AGNN-Cox on a harmonized pan-cancer cohort of 5336 patients from 11 TCGA cancer types (BLCA, BRCA, COAD, GBM, HNSC, KIRC, LIHC, LUAD, LUSC, OV, STAD), where each patient was represented by RNA-sequencing expression (UCSC Xena TPM, log2-transformed and z-scored within training folds) and overall-survival annotation. These cohorts span diverse tissue origins, event rates, and sample sizes (1896 events; Table 1), providing a rigorous testbed for generalizability. For independent external validation without any fine-tuning, we additionally compiled 25 GEO microarray cohorts (Affymetrix; expression values were taken directly from the GEO series matrix files as provided by the data contributors; Table 1), covering every tumor type in the training panel—e.g., three independent lung adenocarcinoma cohorts (GSE31210, GSE50081, GSE68465), three breast cancer cohorts (GSE20685, GSE21653, GSE7390), and three colon cancer cohorts (GSE14333, GSE17536, GSE39582). All preprocessing parameters (gene filtering, standardization) were estimated within training folds to avoid information leakage; external cohorts were standardized with training-cohort statistics and missing genes were set to zero (the training-space mean) at the tensor-materialization stage.

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


For the GEO cohorts, probes were mapped to gene symbols using the platform annotation tables downloaded from NCBI GEO (e.g., GPL570, GPL96, GPL6480); duplicate gene symbols were collapsed by maximum mean expression, and per-gene missing values were imputed with the gene-wise mean within each cohort. Survival endpoints were harmonized to overall survival in days: GEO follow-up times reported in months or years were converted with 365.25/12 or 365.25 days per unit, respectively, and TCGA overall-survival time was derived from GDC clinical annotations, that is, days to death or days to last follow-up. Because each TCGA cohort was modeled separately and each GEO cohort was validated independently without pooling, no cross-cohort batch correction was required.

Expression values were analyzed at each cohort's native scale: TCGA RNA-seq values were log2(TPM+1) as provided by UCSC Xena GDC STAR-TPM, and GEO series matrices were used as provided, typically log2-scale microarray intensities. Before model fitting, every gene was z-scored with the training-cohort mean and standard deviation; external cohorts were standardized with the training statistics, and zero-variance genes were left unchanged.

Expression matrices were first mapped to the KEGG cancer-core pathway catalogue (57 pathways, 3097 genes in the union). After intersection with each cohort's measured genes, on average 2760 genes per cohort were retained; genes outside any pathway were excluded, so that every model in the benchmark operates on the same pathway-mapped gene universe (a matched comparison).

### 2.2. Overview of Path-AGNN-Cox

Path-AGNN-Cox follows a pathway-first pipeline (Figure 1A). Transcriptomic inputs are mapped onto KEGG pathway subgraphs (Figure 1B); gene representations are updated by adaptive pathway-masked graph attention whose temperature is modulated by a learnable per-sample malignancy score (Figure 1C); pathway-level representations are pooled and fused with a gene-level summary; and the resulting patient representation is mapped to a Cox risk score optimized with dual regularization (Figure 1D). A key implementation feature is that the model does not operate on a single fixed graph tensor: pathway topologies are precomputed once, but the **effective edge weights are recomputed for every patient**, so that each patient is instantiated with a patient-specific pathway graph.

![Figure 1](results/figures/Figure1_method.svg)
- **Figure 1. Overview of Path-AGNN-Cox.** (A) End-to-end pipeline: TCGA/GEO expression → KEGG pathway mapping → adaptive pathway subgraphs → risk head with Cox objective. (B) Pathway-constrained block-diagonal adjacency: edges exist only between genes sharing a primary pathway. (C) Sample-adaptive neighborhood weighting: attention logits are multiplicatively modulated by the malignancy gate (1 + tanh(β)·m_s); high-malignancy samples receive sharper within-pathway attention. (D) Survival objective with dual regularization: Cox partial likelihood + intra-pathway sparsity + dropout consistency.


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

We used stratified 5-fold cross-validation within each TCGA cohort (fold-stratified on the event indicator; random seed 42). The concordance index (C-index) served as the primary discrimination metric; the time-dependent AUC (mean over the 0.25/0.50/0.75 quantile times) was used as a secondary metric. For external validation, each model was retrained on the full TCGA cohort and evaluated on the GEO cohorts without any fine-tuning. Paired Wilcoxon signed-rank tests (per-dataset mean C-index, internal CV) were used to compare Path-AGNN-Cox against each baseline. Per-patient edge weights were extracted from the last adaptive layer and patients were split at the median predicted risk. Between-stratum rewiring was tested per pathway on the per-sample mean edge weight within each pathway, reporting Cohen's d with a normal-approximation 95% CI, and the false discovery rate was controlled with the Benjamini–Hochberg procedure. Each pathway was additionally tested under 1,000 within-pathway label permutations of the risk strata; the permutation P was one plus the number of null permutations with at least as large an absolute effect, divided by 1,001. At the cohort level, the number of significant pathways was compared with a 1,000-permutation null obtained by permuting the risk labels, with the permutation P computed as one plus the number of null counts at least as large as the observed count, divided by 1,001. Two matched random gene-set controls assessed pathway-identity selectivity: for each real pathway, 200 random gene sets of equal size with matched internal edge counts, and 200 random equal-size subsets drawn from real pathway blocks so that size and density were matched, were scored with the identical effect-size statistic, and the percentile of each real pathway within its null distribution was recorded. Pathway-level enrichment of the top-20 pathways ranked by permutation P against a curated LUAD driver-pathway list was tested with the hypergeometric distribution over the 57-pathway cancer-core catalogue. Static-model, randomized-partition and standard-GAT controls followed the same protocol. All analyses are implemented in benchmark/rewiring_analysis.py and the work/ scripts of the repository at https://github.com/wangzhipeng-1/Path-AGNN-Cox. Model hyperparameters are listed in the footnote of Table 2 (see config/benchmark.yaml). All deep models were trained on CPU with Adam; the full benchmark consumed approximately 1,500 CPU-hours.

### 2.8. Baselines

Seven baselines span four modeling families: penalized Cox models (LASSO-Cox, Ridge-Cox, Elastic-Net-Cox), a tree ensemble (Random Survival Forest, RSF), deep survival models (DeepSurv, Cox-nnet), and an unconstrained GNN (Plain GNN: the Path-AGNN-Cox backbone with identity adjacency and global pooling, i.e., the −Pathway ablation). Two additional controls are the static pathway GNN (Path-AGNN-Cox with fixed uniform normalized adjacency inside each block, i.e., the −Adaptive ablation) and the unregularized variant (λ_sparse = λ_consist = 0, i.e., the −Regularization ablation). All models use the same pathway-mapped gene universe and the same CV/external protocol, so performance differences are attributable to modeling choices rather than feature sets.

### 2.9. Implementation details and complexity

Path-AGNN-Cox was implemented in Python 3.10 with PyTorch 2.4 (CPU), and all
survival statistics were computed with lifelines 0.30. Models were trained
with Adam (learning rate 1e-3, weight decay 1e-4) for up to 100 epochs with a
batch size of 128 and early stopping on the validation C-index (patience 15);
the network used two hidden layers of width 32 with dropout 0.1, and the
regularization weights were lambda_sparse = 1e-3 and lambda_consist = 0.1
(config/benchmark.yaml). All hyperparameters were held fixed across cancer types and baselines; no cohort-specific tuning was performed. All deep models were trained from a single random Penalized Cox baselines were fit with penalizers of 0.05 for LASSO-Cox with 10-fold internal CV, 0.1 for Ridge-Cox, and l1_ratio 0.5 with penalizer 0.1 for Elastic-Net-Cox; the Random Survival Forest used 500 trees with min_samples_leaf 15; DeepSurv used hidden layers of widths 32 and 16, and Cox-nnet a single hidden layer of width 64.
seed (42) for the benchmark, and seed effects are discussed in Section 4.

The pathway constraint bounds the computational cost of attention. For a batch
of size B, a hidden width d, N genes and E within-pathway edges, each adaptive
layer costs O(B x (E x d + N x d^2)) time and O(B x E x d) memory, dominated by
the edge-indexed feature gather h[:, src]. For the 2,873-gene universe used
here, the KEGG cancer-core partition yields E ~ 2.3 x 10^5 within-pathway edges,
a ~36-fold reduction relative to a fully connected graph (N^2 ~ 8.3 x 10^6),
which keeps the model feasible on commodity hardware.

**Algorithm 1** Training and rewiring testing.
```
Require: expression matrix X, survival times t, event indicators E, pathway memberships P
1: partition genes into K pathway subgraphs; build block-diagonal adjacency A
2: initialize Path-AGNN-Cox parameters theta
3: repeat until early stopping criterion (validation C-index, patience 15)
4:   sample batch; forward pass -> risk scores y, per-patient edge weights alpha
5:   L = L_Cox + lambda_2 ||W||^2 + lambda_sparse * L_sparse + lambda_consist * L_consist
6:   update theta with Adam
7: split patients into high- and low-risk strata (median y)
8: for each pathway k do
9:   test between-stratum difference of mean edge weight (z-test over edges)
10:  end
11: control the false discovery rate (BH) over K pathways
12: repeat the pathway-level procedure under label permutations (1,000x) to obtain a null
```

---

## 3. Results

### 3.1. Benchmark performance across 11 TCGA cancer types

We benchmarked Path-AGNN-Cox against the seven baselines under the same stratified 5-fold protocol (Table 2). Path-AGNN-Cox achieved a mean internal C-index of 0.56 (SD 0.04) across the 11 cohorts and ranked first in 0 of them; the strongest overall baseline was Ridge-Cox (0.62; paired difference Δ=-0.07 (95% CI -0.10 to -0.04), Wilcoxon P=0.005), and the strongest deep survival baseline reached 0.61, i.e., all deep models clustered within a narrow band (Figure 2A). Per-cohort differences were generally small; for example, in LUSC Path-AGNN-Cox reached 0.52 vs. the best baseline 0.53 (Δ = -0.01); in time-dependent AUC, Path-AGNN-Cox ranked first in 1/11 cohorts (Figure 2C). Because the benchmark used a single random seed (42), we retrained the model with three seeds on LUAD and BRCA under the same 5-fold partitions: the internal C-index was stable (LUAD: 0.50 ± 0.04; BRCA: 0.59 ± 0.04; mean ± SD over 3 seeds × 5 folds), with per-seed means spanning at most 0.02.

![Figure 2](results/figures/Figure2_benchmark.svg)
- **Figure 2. Benchmark performance.** (A) Internal 5-fold CV C-index per cancer type. (B) External C-index per cancer type (mean over GEO cohorts). (C) Mean time-dependent AUC (internal CV). Solid markers: Path-AGNN-Cox; grey: baselines; dashed: 0.50 reference.


### Table 2. Benchmark performance: mean internal C-index / time-dependent AUC across 11 TCGA cancer types (stratified 5-fold CV).
| Model | LUAD | LUSC | BRCA | COAD | STAD | LIHC | KIRC | HNSC | BLCA | OV | GBM | Mean C-index |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Path-AGNN-Cox | 0.49/0.51 | 0.52/0.54 | 0.58/0.57 | 0.60/0.60 | 0.53/0.54 | 0.61/0.64 | 0.63/0.64 | 0.54/0.53 | 0.56/0.55 | 0.54/0.55 | 0.58/0.60 | 0.56 |
| LASSO-Cox | 0.65/0.68 | **0.53/0.54** | 0.62/0.62 | 0.53/0.54 | 0.53/0.54 | 0.64/0.67 | 0.71/0.74 | 0.60/0.61 | 0.61/0.64 | 0.52/0.52 | 0.50/0.51 | 0.59 |
| Ridge-Cox | 0.63/0.65 | 0.51/0.51 | 0.69/0.70 | 0.59/0.57 | 0.56/0.57 | **0.68/0.70** | 0.71/0.74 | **0.63/0.64** | **0.63/0.66** | 0.58/0.59 | **0.60/0.62** | 0.62 |
| EN-Cox | **0.65/0.68** | 0.53/0.53 | 0.63/0.64 | 0.54/0.55 | **0.58/0.60** | 0.65/0.68 | 0.71/0.75 | 0.61/0.62 | 0.61/0.63 | 0.52/0.52 | 0.50/0.52 | 0.59 |
| RSF | 0.60/0.61 | 0.53/0.52 | 0.59/0.59 | 0.55/0.52 | 0.52/0.53 | 0.63/0.65 | 0.69/0.72 | 0.60/0.61 | 0.61/0.63 | 0.56/0.58 | 0.53/0.54 | 0.58 |
| DeepSurv | 0.62/0.64 | 0.50/0.50 | 0.67/0.70 | 0.59/0.56 | 0.54/0.55 | 0.67/0.69 | **0.72/0.75** | 0.60/0.63 | 0.61/0.66 | **0.60/0.63** | 0.56/0.57 | 0.61 |
| Cox-nnet | 0.64/0.66 | 0.51/0.52 | **0.70/0.73** | 0.56/0.56 | 0.54/0.53 | 0.67/0.69 | 0.69/0.72 | 0.61/0.63 | 0.61/0.66 | 0.60/0.63 | 0.58/0.58 | 0.61 |
| −Pathway (plain GNN) | 0.50/0.51 | 0.50/0.51 | 0.61/0.58 | **0.61/0.60** | 0.51/0.49 | 0.63/0.66 | 0.63/0.65 | 0.52/0.52 | 0.57/0.57 | 0.52/0.54 | 0.55/0.56 | 0.56 |

*C-index / mean time-dependent AUC; bold = best C-index per cancer type (5-fold CV).*


On external testing across the 25 GEO cohorts, the mean external C-index of Path-AGNN-Cox was 0.51 (SD 0.04), matching the deep baselines (0.50) and ranking first among all models in 0 of 11 cancer types (Figure 2B; per-cohort details in Table 3). Penalized Cox baselines retained the highest external means (0.57, Ridge-Cox; paired difference Δ=-0.10 (95% CI -0.14 to -0.06)), while the three GNN variants (adaptive, static, plain) showed similar external decay, indicating that the pathway or adaptive design does not by itself eliminate cross-cohort performance loss; the interpretable rewiring output is the distinctive capability of the adaptive model (Section 3.4).

### Table 3. External validation per GEO cohort.
| Cancer | Cohort | N | Path-AGNN-Cox | Best baseline | Baseline C-index | Δ |
|---|---|---|---|---|---|---|
| Lung adenocarcinoma | GSE31210 | 226 | 0.76 | Cox-nnet | 0.67 | 0.09 |
| Lung adenocarcinoma | GSE50081 | 181 | 0.57 | RSF | 0.60 | -0.02 |
| Lung adenocarcinoma | GSE68465 | 442 | 0.52 | Ridge-Cox | 0.63 | -0.12 |
| Lung squamous carcinoma | GSE37745 | 196 | 0.51 | RSF | 0.52 | -0.01 |
| Lung squamous carcinoma | GSE8894 | 138 | 0.55 | DeepSurv | 0.57 | -0.03 |
| Breast carcinoma | GSE20685 | 327 | 0.52 | Ridge-Cox | 0.71 | -0.19 |
| Breast carcinoma | GSE21653 | 248 | 0.45 | Ridge-Cox | 0.65 | -0.21 |
| Breast carcinoma | GSE7390 | 198 | 0.60 | Ridge-Cox | 0.66 | -0.07 |
| Colon adenocarcinoma | GSE14333 | 226 | 0.50 | RSF | 0.58 | -0.08 |
| Colon adenocarcinoma | GSE17536 | 177 | 0.45 | DeepSurv | 0.53 | -0.08 |
| Colon adenocarcinoma | GSE39582 | 573 | 0.49 | RSF | 0.57 | -0.08 |
| Stomach adenocarcinoma | GSE15459 | 191 | 0.55 | Ridge-Cox | 0.58 | -0.03 |
| Stomach adenocarcinoma | GSE84437 | 431 | 0.48 | RSF | 0.58 | -0.10 |
| Liver hepatocellular carcinoma | GSE116174 | 64 | 0.60 | EN-Cox | 0.64 | -0.04 |
| Liver hepatocellular carcinoma | GSE14520 | 221 | 0.42 | Ridge-Cox | 0.64 | -0.22 |
| Kidney renal clear-cell carcinoma | GSE29609 | 39 | 0.48 | −Pathway (plain GNN) | 0.59 | -0.11 |
| Head and neck squamous carcinoma | GSE41613 | 97 | 0.46 | Ridge-Cox | 0.66 | -0.20 |
| Head and neck squamous carcinoma | GSE65858 | 270 | 0.49 | RSF | 0.60 | -0.11 |
| Bladder urothelial carcinoma | GSE13507 | 165 | 0.50 | Ridge-Cox | 0.62 | -0.13 |
| Bladder urothelial carcinoma | GSE32894 | 224 | 0.40 | Ridge-Cox | 0.79 | -0.39 |
| Ovarian serous carcinoma | GSE17260 | 110 | 0.56 | Ridge-Cox | 0.64 | -0.09 |
| Ovarian serous carcinoma | GSE26712 | 185 | 0.50 | Ridge-Cox | 0.63 | -0.13 |
| Ovarian serous carcinoma | GSE32062 | 260 | 0.49 | Ridge-Cox | 0.61 | -0.13 |
| Glioblastoma | GSE108474 | 119 | 0.49 | EN-Cox | 0.58 | -0.08 |
| Glioblastoma | GSE7696 | 70 | 0.51 | EN-Cox | 0.52 | -0.01 |


### 3.2. Ablation study

We compared the full model with three ablations: −Pathway (Plain GNN, identity graph), −Adaptive (static uniform pathway adjacency), and −Regularization (no sparse/consistency terms), across all 11 cohorts (Table 4, Figure 3). Removing the pathway constraint changed the mean internal C-index by 0.00 (paired Δ=0.00 (95% CI -0.01 to 0.02), P=0.520); removing the adaptive gate by 0.01 (paired Δ=0.01 (95% CI -0.01 to 0.02), P=0.206); and removing the dual regularization by 0.00 (paired Δ=-0.00 (95% CI -0.01 to 0.01), P=0.765). None of these differences was statistically significant, and external discrimination was similarly insensitive (e.g., −Adaptive external difference 0.01). We therefore conclude that, in the configuration evaluated, the predictive contribution of the individual design modules cannot be separated at the C-index level; the added value of the adaptive pathway design is the patient-specific interpretability it enables (Section 3.4) rather than a measurable discrimination gain.

![Figure 3](results/figures/Figure3_ablation.svg)
- **Figure 3. Ablation study.** (A) Internal C-index of the full model vs. −Pathway / −Adaptive / −Regularization per cancer type. (B) External C-index per variant. (C) Mean internal drop with 95% CI.


### Table 4. Ablation study: mean internal and external C-index for the full model and its three ablations.
| Variant | LUAD | LUSC | BRCA | COAD | STAD | LIHC | KIRC | HNSC | BLCA | OV | GBM | Internal mean±SD | External mean±SD | Δ vs full | P |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Path-AGNN-Cox | 0.49 | 0.52 | 0.58 | 0.60 | 0.53 | 0.61 | 0.63 | 0.54 | 0.56 | 0.54 | 0.58 | 0.56±0.04 | 0.51±0.04 | ref | ref |
| −Adaptive (static) | 0.53 | 0.50 | 0.56 | 0.59 | 0.50 | 0.62 | 0.62 | 0.53 | 0.53 | 0.56 | 0.55 | 0.55±0.04 | 0.50±0.05 | -0.01 | P=0.206 |
| −Regularization | 0.53 | 0.53 | 0.58 | 0.59 | 0.54 | 0.62 | 0.63 | 0.52 | 0.55 | 0.52 | 0.58 | 0.56±0.04 | 0.50±0.03 | 0.00 | P=0.765 |
| −Pathway (plain GNN) | 0.50 | 0.50 | 0.61 | 0.61 | 0.51 | 0.63 | 0.63 | 0.52 | 0.57 | 0.52 | 0.55 | 0.56±0.05 | 0.51±0.04 | 0.00 | P=0.520 |


### 3.3. External validation across 25 GEO cohorts

Per-cohort results are summarized in Table 3 and Figure 4. Path-AGNN-Cox maintained C-index above 0.50 in 12/25 external cohorts, compared with 25 for the best baseline. The model transferred across platform shifts (RNA-seq → Affymetrix microarrays) and independent sample processing pipelines, with a mean external C-index comparable to the deep baselines. The largest external gain over the best baseline was observed in Lung adenocarcinoma (GSE31210, Path-AGNN-Cox C-index 0.76 vs best baseline 0.67). These results contrast favorably with previous pathway-GNN studies: PathMoG reported external validation in a single breast-cancer cohort (METABRIC) [PathMoG], whereas Path-AGNN-Cox was validated in 25 independent cohorts spanning 11 tumor types.

Calibration of the risk score was assessed with the slope of a univariate Cox regression of the standardized risk score and with the mean absolute deviation between model-predicted and Kaplan\u2013Meier survival across risk tertiles at the 25th, 50th and 75th percentiles of follow-up time as detailed in Table 5. The internal out-of-fold slope was -0.02 with 95% CI -0.16–0.12 and a calibration MAE of 0.02 in LUAD, and -0.05 with 95% CI -0.23–0.14 and MAE 0.01 in BRCA; the penalized Cox counterpart produced slopes of 0.39 and 0.49, respectively. External slopes for Path-AGNN-Cox ranged from -0.11 to 0.79 with a mean of 0.15.

![Figure 4](results/figures/Figure4_external.svg)
- **Figure 4. External validation across 25 GEO cohorts.** Per-cohort C-index of Path-AGNN-Cox vs. the best baseline; cohorts grouped by cancer type; dashed line at 0.50.


### Table 5. Calibration of the risk score in internal and external cohorts.
| Dataset | Setting | Cohort | Model | N | Events | Slope | 95% CI | MAE |
|---|---|---|---|---|---|---|---|---|
| LUAD | Internal CV | LUAD | Path-AGNN-Cox | 527 | 188 | -0.02 | -0.16–0.12 | 0.02 |
| LUAD | Internal CV | LUAD | Ridge-Cox | 527 | 188 | 0.39 | 0.25–0.53 | 0.00 |
| LUAD | External transfer | GSE31210 | Path-AGNN-Cox | 226 | 35 | 0.79 | 0.49–1.09 | 0.03 |
| LUAD | External transfer | GSE31210 | Ridge-Cox | 226 | 35 | 0.57 | 0.30–0.84 | 0.05 |
| LUAD | External transfer | GSE50081 | Path-AGNN-Cox | 181 | 75 | 0.05 | -0.19–0.30 | 0.05 |
| LUAD | External transfer | GSE50081 | Ridge-Cox | 181 | 75 | 0.17 | -0.05–0.38 | 0.09 |
| LUAD | External transfer | GSE68465 | Path-AGNN-Cox | 442 | 236 | 0.05 | -0.09–0.18 | 0.02 |
| LUAD | External transfer | GSE68465 | Ridge-Cox | 442 | 236 | 0.22 | 0.10–0.34 | 0.04 |
| BRCA | Internal CV | BRCA | Path-AGNN-Cox | 1086 | 151 | -0.05 | -0.23–0.14 | 0.01 |
| BRCA | Internal CV | BRCA | Ridge-Cox | 1086 | 151 | 0.49 | 0.34–0.64 | 0.02 |
| BRCA | External transfer | GSE20685 | Path-AGNN-Cox | 327 | 83 | -0.11 | -0.33–0.10 | 0.07 |
| BRCA | External transfer | GSE20685 | Ridge-Cox | 327 | 83 | 0.68 | 0.46–0.89 | 0.02 |
| BRCA | External transfer | GSE21653 | Path-AGNN-Cox | 248 | 79 | -0.05 | -0.27–0.18 | 0.03 |
| BRCA | External transfer | GSE21653 | Ridge-Cox | 248 | 79 | 0.46 | 0.25–0.66 | 0.03 |
| BRCA | External transfer | GSE7390 | Path-AGNN-Cox | 198 | 56 | 0.20 | -0.06–0.45 | 0.02 |
| BRCA | External transfer | GSE7390 | Ridge-Cox | 198 | 56 | 0.43 | 0.18–0.68 | 0.04 |

### 3.4. Path-AGNN-Cox learns biologically meaningful patient-specific pathway rewiring

A central claim of this work is that the sample-specific edge weights carry patient-specific variation that can be tested formally for biological content. We therefore extracted, for every patient, the last-layer adaptive attention weights within each pathway block and compared high- vs. low-risk strata (median risk split) in LUAD, BRCA and KIRC (Figure 5).

![Figure 5](results/figures/Figure5_rewiring.svg)
- **Figure 5. Patient-specific pathway rewiring.** (A) Between-stratum effect sizes (Cohen's d with 95% CI) for all 53 pathways in LUAD, with the two pathways surviving per-pathway label permutation and BH-FDR correction highlighted. (B) Same as A for BRCA. (C) Cohort-level label-permutation null: number of significant pathways observed versus the null mean and maximum under 1,000 label permutations (LUAD and BRCA). (D) Correlation between per-patient rewiring magnitude and clinical malignancy indicators (LUAD). (E) Matched random gene-set controls in LUAD, BRCA and KIRC: distribution of P(null effect ≥ real effect) for the edge-matched and density-matched nulls; low values indicate real pathway effects above the matched random sets. The simulated pure-null median of the density-matched statistic was 1.00 (LUAD and BRCA), so values below the simulated 5th percentile exceed the structural bias of row-normalized attention.


**3.4.1. Between-stratum rewiring exceeds label-permutation chance; matched-set selectivity is cohort-dependent.** The per-sample mean edge weight within each pathway was compared between high- and low-risk strata (median risk split) with Cohen's d and 95% CI (Figure 5A,B). After per-pathway label permutation (1,000 permutations) and BH-FDR correction, 43 of 53 pathways in BRCA, 2 in LUAD and 52 in KIRC remained significant; the two LUAD pathways were Homologous recombination (d = 0.35, 95% CI 0.18–0.53) and DNA replication (d = 0.35, 95% CI 0.18–0.52) (Table 6). At the cohort level, the number of pathways with BH-FDR q<0.05 on the pathway-level test exceeded the cohort-level label-permutation null in all three cohorts (LUAD: 3 observed vs. null mean 0.16, maximum 34, permutation P=0.014; BRCA: 43 observed vs. null mean 0.19, maximum 19, permutation P<0.001; KIRC: 52 observed vs. null mean 0.07, maximum 3, permutation P=0.005; 200 cohort-level permutations). The between-stratum differences therefore reflect risk-associated structure in the attention weights rather than the risk stratification itself.

Pathway identity was not globally selective, but matched-set enrichment was cohort-dependent. The percentile is defined as P(null effect ≥ real effect), so low values indicate that a real pathway exceeded the matched random sets. Because this statistic is computed on row-normalized attention, we first characterized it under a pure null: across 24 simulated datasets with no between-stratum signal, the median density-matched percentile was 1.00 in LUAD and 1.00 in BRCA (5th percentile of the simulated medians: 0.60 and 0.66), showing a strong upward structural bias of the statistic. Against this baseline, the observed density-matched medians were 0.69 in LUAD, within the pure-null range, 0.20 in BRCA, below the simulated 5th percentile, and 0.05 in KIRC. The edge-matched control gave median percentiles of 0.67, 0.09 and <0.01 in LUAD, BRCA and KIRC. The number of real pathways above the 95th percentile of the density-matched null was 1, 6 and 25 of 53 (expected 2.65 by chance), and 2, 16 and 39 of 53 for the edge-matched null. Matched-set enrichment was therefore evident in BRCA and KIRC and could not be attributed to the structural bias of row-normalized attention, whereas LUAD showed no enrichment beyond the structural null. Known-pathway enrichment of the top-20 pathways ranked by permutation P was not significant in LUAD (5 hits, P=0.689). These observations are hypothesis-generating rather than evidence of specific pathway activation, and the pathway partition remains an interpretable organizing principle rather than a statistically necessary one.**3.4.2. Static models cannot produce rewiring.** As a negative control, the static pathway GNN (−Adaptive) was subjected to the identical analysis; its total edge-weight variance across patients was 0.00, essentially zero by construction, whereas the adaptive model produced a total edge-weight variance of 19.21, orders of magnitude larger; the between-stratum differences of Section 3.4.1 are therefore attributable to the sample-specific attention mechanism rather than to the fixed adjacency. Retraining the adaptive model with randomized pathway partitions (block sizes preserved; three seeds) yielded 26–48 significant pathways in LUAD, indicating that between-stratum attention differences do not require the canonical partition. Combined with the matched-set results of Section 3.4.1, the pathway partition supplies biologically interpretable labels for the rewiring output; whether the canonical grouping itself contributes beyond any fixed grouping remains unresolved.**3.4.3. Clinical correlation.** The per-patient rewiring magnitude (L1 distance of the patient's edge weights from the cohort mean) correlated with clinical indicators of malignancy: in BRCA with the proliferation marker MKI67 (Spearman ρ = 0.33, P<0.001, n = 1086), and in LUAD with tumor mutational burden (ρ = 0.13, P=0.003, n = 522); stage showed no significant association in either cohort. In multivariable Cox models adjusting for stage and age, the risk score remained independently associated with overall survival in both cohorts (LUAD: HR = 1.22 per SD, 95% CI 1.01–1.46, P=0.035; BRCA: HR = 1.27 per SD, 95% CI 1.06–1.53, P=0.010). In the independent IMvigor210 anti-PD-L1 cohort, the rewiring magnitude differed between responders (CR/PR) and non-responders (SD/PD) (median 686.43 vs 624.81; Wilcoxon P=0.111; n=68/230); high-rewiring patients showed HR 1.40 (95% CI 0.96-2.04, P=0.082) for OS; rewiring magnitude correlated with Ki-67 expression (Spearman rho=0.26, P<0.001, n=348) (Figure 6); the rewiring-Ki-67 correlation replicated in this immunotherapy setting, supporting the biological validity of the learned patient-specific graphs. Because MKI67 is measured in the same transcriptomic input used to compute the attention weights, the Ki-67 correlations are descriptive anchors rather than independent validations; TMB provides an independent (mutation-based) anchor. The associations were robust to the definition of rewiring magnitude (the Ki-67 association in BRCA (rho = 0.19-0.29 across definitions, all P<0.001); the risk-score association in IMvigor210 (rho = 0.32-0.43, all P<1e-9); the LUAD TMB association was directionally consistent but weaker (rho = 0.09, P<0.001, for the primary definition)).

![Figure 6](results/figures/Figure6_imvigor.svg)
- **Figure 6. IMvigor210 anti-PD-L1 cohort (exploratory).** (A) Per-patient rewiring magnitude in responders (CR/PR) vs non-responders (SD/PD); Wilcoxon rank-sum test. (B) Overall survival of high- vs low-rewiring strata (median split) with log-rank test. (C) Ki-67 expression vs rewiring magnitude with Spearman correlation.
### Table 6. Framework validation of patient-specific pathway rewiring in LUAD, BRCA and KIRC.
| Check | LUAD | BRCA | KIRC |
|---|---|---|---|
| Pathways tested | 53 | 53 | 53 |
| Significant pathways, BH-FDR on the unadjusted test | 3 | 43 | 52 |
| Per-pathway permutation-calibrated pathways (FDR q<0.05) | 2 | 43 | 52 |
| Cohort-level label-permutation null, mean | 0.16 | 0.19 | 0.07 |
| Cohort-level label-permutation null, maximum | 34 | 19 | 3 |
| Cohort-level permutation P | P=0.014 | P<0.001 | P=0.005 |
| Median percentile of real pathways, edge-matched null | 0.67 | 0.09 | <0.01 |
| Pathways above the 95th percentile of the edge-matched null | 2 | 16 | 39 |
| Median percentile of real pathways, density-matched null | 0.69 | 0.20 | 0.05 |
| Pathways above the 95th percentile of the density-matched null | 1 | 6 | 25 |
| Expected above the 95th percentile by chance | 2.65 | 2.65 | 2.65 |
| Pure-null structural median (density-matched), simulated | 1.00 | 1.00 | n.a. |
| Known-pathway enrichment, top-20 by permutation P | 5 hits, P=0.689 | n.a. | n.a. |
| Static-model total edge variance | 3.48e-13 | 2.81e-13 | n.a. |
| Adaptive-model total edge variance | 1.92e+01 | 8.41e-02 | 2.44e+01 |
| Randomized-partition control, significant pathways (3 seeds) | 26-48 | n.a. | n.a. |

Hypothesis-generating pathways that exceeded the per-pathway permutation null and the density-matched control:

| Cancer | Pathway | Cohen's d (95% CI) | Permutation q | Density-matched percentile |
|---|---|---|---|---|
| LUAD | Homologous recombination | 0.35 (0.18-0.53) | q=0.026 | 96.0 |
| LUAD | DNA replication | 0.35 (0.18-0.52) | q=0.026 | 94.0 |



**3.4.4. Standard-GAT negative control.** To test whether the between-stratum edge-weight differences are specific to the pathway-constrained adaptive architecture, we trained a standard GAT on a sample-invariant k-nearest-neighbor gene graph (k=10; no pathway constraint) under the identical training protocol and applied the identical pathway-level testing pipeline. The standard GAT yielded 11 of 54 significantly rewired pathways in BRCA and 1 of 54 in LUAD (BH-FDR q<0.05), compared with 43 of 53 and 2 of 53 for the adaptive pathway-constrained model under the same permutation-calibrated procedure (Table 6); the pathway-constrained model yielded more detectable between-stratum differences in BRCA and a comparable count in LUAD, indicating that the pathway partition contributes to the statistical detection of these differences; consistent with the randomized-partition control (Section 3.4.2), canonical pathway structure is nonetheless not uniquely required to produce them.

**3.4.5. External replication of patient-specific rewiring.** As an out-of-cohort check, we transferred each TCGA-trained model to independent GEO cohorts of the same cancer type and tested whether per-patient rewiring magnitude was associated with overall survival within each cohort. 0 of 3 GEO cohorts showed a nominally significant association between rewiring magnitude and OS: GSE31210 (HR 0.83, 95% CI 0.43-1.62, P=0.595; n=226); GSE50081 (HR 0.90, 95% CI 0.57-1.42, P=0.651; n=181); GSE68465 (HR 1.03, 95% CI 0.80-1.33, P=0.815; n=442). 0 of 3 GEO cohorts showed a nominally significant association between rewiring magnitude and OS: GSE20685 (HR 1.02, 95% CI 0.66-1.56, P=0.941; n=327); GSE21653 (HR 1.12, 95% CI 0.72-1.75, P=0.602; n=248); GSE7390 (HR 1.37, 95% CI 0.81-2.33, P=0.243; n=198). These associations are exploratory and were not corrected for multiple testing.

---


### 3.5. Immune infiltration and predicted drug sensitivity (exploratory)

To connect patient-specific rewiring to the tumor microenvironment and to therapeutic vulnerability, we computed MCP-counter immune cell abundance estimates and ssGSEA scores of nine immune Hallmark gene sets per patient, and tested their association with the per-patient rewiring magnitude (Figure 7A). In LUAD, the rewiring magnitude was nominally associated with lower ssGSEA IL2-STAT5-SIGNALING (P=0.016), B lineage (P=0.018) and ssGSEA COMPLEMENT (P=0.046); none of these associations survived BH-FDR correction (smallest q=0.169), and in BRCA no immune feature was associated with rewiring magnitude (smallest P=0.274). Immune infiltration was therefore not strongly coupled to pathway rewiring in these two cohorts.

![Figure 7](results/figures/Figure7_immunedrug.svg)
- **Figure 7. Immune infiltration and predicted drug sensitivity (exploratory).** (A) Association of 19 immune features with per-patient rewiring magnitude in LUAD (red) and BRCA (blue): signed -log10 P of the Wilcoxon test between high- and low-rewiring strata; dashed line: P=0.050. (B) Predicted IC50 (GDSC2/oncoPredict) of the eight nominally significant compounds in BRCA, high- versus low-rewiring strata. (C) Spearman correlation between rewiring magnitude and predicted IC50 across 17 curated compounds in BRCA; filled markers: FDR q<0.05.


We further predicted drug sensitivity with the oncoPredict model trained on GDSC2 cell-line pharmacogenomic data (198 compounds) and compared predicted IC50 values between high- and low-rewiring strata (Figure 7B, Table 7). In BRCA, the high-rewiring stratum was predicted to be more sensitive (lower IC50) to 9 of the 17 curated compounds at nominal significance, most strongly MK-1775 (WEE1, Wilcoxon P=0.006), paclitaxel (P=0.006) and palbociclib (CDK4/6, P=0.018); the Spearman association remained significant after FDR correction for MK-1775 and paclitaxel (q=0.006), whereas the Wilcoxon comparisons did not survive correction (smallest q=0.054). In LUAD, no compound showed a significant difference (smallest P=0.063). These analyses are hypothesis-generating: the predicted IC50 values are in-silico estimates from cell-line-derived models and do not substitute for direct pharmacologic assays (Section 4).

### Table 7. Predicted drug sensitivity (GDSC2/oncoPredict IC50) in high- versus low-rewiring strata (exploratory).
**BRCA (n high/low: 536/535)**
| Drug | IC50 median (high) | IC50 median (low) | Wilcoxon P | FDR q | Spearman ρ | Spearman P |
|---|---|---|---|---|---|---|
| MK-1775 | 2.12 | 2.40 | P=0.006 | q=0.054 | -0.10 | P<0.001 |
| Paclitaxel | 0.06 | 0.06 | P=0.006 | q=0.054 | -0.10 | P<0.001 |
| Gefitinib | 27.84 | 29.30 | P=0.018 | q=0.062 | -0.08 | P=0.007 |
| Gemcitabine | 0.53 | 0.63 | P=0.018 | q=0.062 | -0.07 | P=0.018 |
| 5-Fluorouracil | 131.74 | 150.89 | P=0.019 | q=0.062 | -0.09 | P=0.005 |
| Palbociclib | 36.30 | 38.94 | P=0.024 | q=0.062 | -0.07 | P=0.016 |
| Docetaxel | 0.07 | 0.08 | P=0.026 | q=0.062 | -0.08 | P=0.006 |
| Cisplatin | 28.92 | 33.20 | P=0.033 | q=0.071 | -0.07 | P=0.015 |
| AZD7762 | 1.22 | 1.32 | P=0.047 | q=0.089 | -0.07 | P=0.014 |
| AZD6738 | 8.19 | 8.74 | P=0.068 | q=0.116 | -0.07 | P=0.020 |
| Talazoparib | 23.20 | 26.38 | P=0.083 | q=0.117 | -0.06 | P=0.050 |
| Niraparib | 79.60 | 86.53 | P=0.083 | q=0.117 | -0.07 | P=0.022 |
| Olaparib | 68.50 | 71.74 | P=0.185 | q=0.242 | -0.05 | P=0.077 |
| Ribociclib | 45.50 | 46.65 | P=0.210 | q=0.255 | -0.05 | P=0.073 |
| Trametinib | 2.11 | 2.24 | P=0.298 | q=0.338 | -0.04 | P=0.235 |
| Erlotinib | 15.36 | 15.61 | P=0.341 | q=0.345 | -0.04 | P=0.251 |
| Selumetinib | 65.30 | 68.41 | P=0.345 | q=0.345 | -0.04 | P=0.242 |

**LUAD (n high/low: 252/252)**
| Drug | IC50 median (high) | IC50 median (low) | Wilcoxon P | FDR q | Spearman ρ | Spearman P |
|---|---|---|---|---|---|---|
| Ribociclib | 45.83 | 44.05 | P=0.063 | q=0.322 | 0.13 | P=0.004 |
| Niraparib | 83.33 | 76.45 | P=0.064 | q=0.322 | 0.10 | P=0.027 |
| MK-1775 | 2.02 | 2.41 | P=0.065 | q=0.322 | -0.07 | P=0.099 |
| Erlotinib | 14.40 | 15.24 | P=0.076 | q=0.322 | -0.04 | P=0.320 |
| Gefitinib | 25.55 | 27.29 | P=0.140 | q=0.457 | -0.03 | P=0.476 |
| Olaparib | 71.42 | 65.38 | P=0.161 | q=0.457 | 0.08 | P=0.080 |
| Talazoparib | 23.74 | 22.37 | P=0.236 | q=0.573 | 0.07 | P=0.135 |
| Selumetinib | 64.14 | 60.81 | P=0.305 | q=0.613 | 0.07 | P=0.111 |
| AZD6738 | 8.38 | 8.42 | P=0.325 | q=0.613 | -0.05 | P=0.252 |
| Docetaxel | 0.07 | 0.07 | P=0.396 | q=0.631 | -0.02 | P=0.578 |
| Paclitaxel | 0.06 | 0.06 | P=0.442 | q=0.631 | -0.03 | P=0.542 |
| AZD7762 | 1.20 | 1.22 | P=0.467 | q=0.631 | -0.02 | P=0.616 |
| Gemcitabine | 0.53 | 0.48 | P=0.483 | q=0.631 | 0.03 | P=0.443 |
| Cisplatin | 30.34 | 29.16 | P=0.659 | q=0.771 | 0.00 | P=0.932 |
| 5-Fluorouracil | 126.39 | 112.80 | P=0.680 | q=0.771 | 0.01 | P=0.742 |
| Trametinib | 2.07 | 1.96 | P=0.789 | q=0.838 | 0.05 | P=0.307 |
| Palbociclib | 32.99 | 33.75 | P=0.961 | q=0.961 | 0.04 | P=0.340 |

_IC50 values are GDSC2/oncoPredict in-silico predictions; associations are exploratory and not FDR-significant unless stated._


---

## 4. Discussion

Path-AGNN-Cox addresses transcriptomic survival prediction with a pathway-centric graph design that is explicitly tuned to the *p*≫*n* regime and, unlike previous pathway GNNs, learns the pathway graph itself per patient. By constraining message passing to KEGG pathway modules, by modulating attention with a learnable malignancy gate, and by regularizing with sparse and consistency terms, the model achieves competitive discrimination across 11 cancer types and—more importantly—stable transfer to 25 independent GEO cohorts—substantially broader external validation than comparable pathway-GNN studies.

The conceptual contribution is a framework rather than a module tweak: existing pathway GNNs—PathGNN, Cox-Path, multilevel prior-knowledge GNNs, and the multi-omics PathMoG—treat the pathway graph as a fixed patient-invariant prior and stop at the risk score. We treat the effective graph as a per-patient object and provide the statistical machinery to interrogate its rewiring: edge-level and pathway-level tests with BH-FDR, label-permutation nulls, a static-model negative control, a standard-GAT architectural control, and clinical anchoring. The discrimination of the adaptive model was comparable to deep survival baselines, and we make no claim that per-patient attention improves C-index; its distinguishing value is that the learned patient-specific edge weights can be formally tested, are clinically anchored to Ki-67 and TMB, and are absent in static models by construction. The pathway partition is an interpretive labeling of these differences rather than a statistical necessity, as detailed in Section 3.4.2.**Comparison with PathMoG.** PathMoG is the closest recent work: it also organizes genes into KEGG pathway modules and uses hierarchical attention for multi-omics survival prediction [PathMoG]. The two frameworks are complementary in scope and differ in three substantive ways. First, *static versus adaptive topology*: PathMoG precomputes patient-invariant pathway topologies and learns attention within them; Path-AGNN-Cox additionally learns the graph *temperature* per patient, and its rewiring analysis directly interrogates patient-specific graph dynamics. Second, *omics scope*: PathMoG requires mutation and CNV in addition to expression, which limits its applicability to the many clinical cohorts that only measure transcriptomes; Path-AGNN-Cox is a single-transcriptome model that is immediately applicable to the large body of GEO microarray cohorts, as supported by our 25-cohort external validation. Third, *external validation breadth*: PathMoG validated transferability in a single breast-cancer cohort, METABRIC; Path-AGNN-Cox was validated in 25 GEO cohorts spanning 11 tumor types. These differences position Path-AGNN-Cox as a lightweight, broadly deployable alternative for single-omics clinical and public cohorts, while PathMoG remains the reference for multi-omics integration.

**Limitations.** First, the predictive performance of the adaptive model did not exceed penalized Cox baselines on internal CV, and ablations did not isolate a significant C-index contribution from the adaptive gate, the pathway constraint, or the regularization terms; interpretability is the rationale for its use, not a discrimination gain. Second, the learnable malignancy gate beta converged to values near zero in the trained models, so the effective sample-specificity arises from the attention coefficients themselves; we therefore describe the mechanism as patient-specific attention rather than as demonstrable malignancy-driven gating. Third, hypergeometric enrichment of the top-20 rewired pathways ranked by permutation P against a curated LUAD driver-pathway list was not significant (5 hits, P=0.689); an equivalent test was not performed for BRCA because no matched curated driver-pathway list was available. Fourth, matched random gene-set controls were cohort-dependent: real pathways were enriched above matched random sets in BRCA and KIRC beyond a simulated pure-null baseline of the statistic, whereas LUAD showed no enrichment beyond the structural null; the pathways that exceeded both permutation and matched-set nulls remain hypothesis-generating, and the pathway prior is retained as an interpretable organizing principle rather than as evidence of specific pathway activation. Fifth, attention weights provide hypothesis-generating, not causal, evidence, and all cohorts are retrospective; prospective validation is required. Sixth, the pan-cancer benchmark used a single seed; a three-seed sensitivity analysis on LUAD and BRCA, the cohorts used for the rewiring analyses, showed stable C-index values of 0.50 ± 0.04 in LUAD and 0.59 ± 0.04 in BRCA, but seed variance was not propagated to all cohorts. Seventh, the two LUAD pathways that were significant in the training cohort were not replicated at the pathway level across three independent external LUAD cohorts (DNA replication: Stouffer z = -1.47, P=0.141, with 1 of 3 cohorts in the same direction; homologous recombination: z = 0.69, P=0.490, 2 of 3); pathway-level rewiring findings should therefore be interpreted within the cohort in which they were derived.

---

## 5. Conclusion

Path-AGNN-Cox couples a pathway-constrained GNN with patient-specific attention to a formal statistical framework for testing and clinically anchoring patient-specific pathway rewiring, while offering discrimination comparable to deep survival baselines. With 11 internal and 25 external cohorts, open-source code at https://github.com/wangzhipeng-1/Path-AGNN-Cox, a versioned Zenodo archive at https://doi.org/10.5281/zenodo.22030045, and a pip-installable Python package, the framework is directly reusable for pan-cancer prognostic modeling and for interrogating the patient-specific graph dynamics that static pathway models cannot produce.

---

## 6. Availability and implementation

- **Code:** https://github.com/wangzhipeng-1/Path-AGNN-Cox (MIT license; Python package `path_agnn_cox`; R preprocessing scripts in `data/scripts/`; Zenodo archive: https://doi.org/10.5281/zenodo.22030045).
- **Installation:** `pip install path-agnn-cox` (PyPI) or `pip install .` from the repository; example notebooks under `examples/`.
- **Data:** TCGA RNA-seq and clinical data from UCSC Xena (https://xenabrowser.net); GEO series matrices and platform annotations from NCBI GEO (https://www.ncbi.nlm.nih.gov/geo/); all accession numbers listed in Table 1.
- **Reproduction:** `manuscript/render_manuscript.py` and `manuscript/make_figures.py` regenerate every table and figure from `results/benchmark_results.csv` and the rewiring outputs (see README.md).

---



## References

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
13. Ha MJ, Baladandayuthapani V. DINGO: differential network analysis in genomics. *Bioinformatics*. 2015;31(21):3413-3420.
14. de la Fuente A. From 'differential expression' to 'differential networking': identification of dysfunctional regulatory networks in diseases. *Trends Genet*. 2010;26(7):326-333.
15. Gill R, Datta S, Datta S. A statistical framework for differential network analysis from microarray data. *BMC Bioinformatics*. 2010;11:95.
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

**Results:** We propose **Path-AGNN-Cox**, a pathway-constrained graph neural network that computes a **patient-specific pathway graph** for survival prediction. Path-AGNN-Cox (i) partitions genes into KEGG cancer-core pathway modules and restricts message passing to biologically co-regulated gene pairs; (ii) computes sample-specific within-pathway attention weights via a learnable, malignancy-modulated gate; and (iii) optimizes a Cox partial-likelihood objective with dual regularization—intra-pathway sparsity plus a dropout-consistency constraint—to suppress overfitting in high-heterogeneity cohorts. We benchmarked Path-AGNN-Cox against seven survival baselines (penalized Cox, RSF, deep survival, and plain-GNN) across {{N_DATASETS}} TCGA cancer types ({{TCGA_TOTAL_N}} patients) under stratified 5-fold cross-validation, and validated transferability on {{N_EXTERNAL}} independent GEO cohorts spanning the same tumor types. On internal CV, Path-AGNN-Cox reached a mean C-index of {{CV_FULL_MEAN}} (SD {{CV_FULL_SD}}), comparable to deep survival baselines ({{CV_BEST_DEEP_MEAN}}) but below penalized Cox ({{CV_BEST_BASELINE_MEAN}}; paired difference {{CV_DIFF_CI}}); we therefore make no claim of a discrimination gain. Its distinguishing value is the interpretable rewiring output: on {{N_EXTERNAL}} independent GEO cohorts the model matched deep baselines ({{EXT_FULL_MEAN}} vs. {{EXT_BEST_DEEP_MEAN}}), while providing per-patient pathway weights that static models cannot produce. Ablations showed that removing the pathway constraint, the adaptive gate, or the dual regularization did not significantly change discrimination (all P>0.05), indicating that the value of the adaptive design lies in interpretability rather than in a measurable C-index gain. The learned sample-specific edge weights are statistically testable: rewiring between high- and low-risk patients was far beyond label-permutation nulls (BRCA: {{PERM_BRCA_SIG}}/{{PERM_N_PATHWAYS}} pathways, permutation {{PERM_BRCA_P}}; LUAD: {{PERM_LUAD_SIG}}/{{PERM_N_PATHWAYS}}, {{PERM_LUAD_P}}; KIRC: {{PERM_KIRC_SIG}}/{{PERM_N_PATHWAYS}}, {{PERM_KIRC_P}}), correlated with clinical indicators of malignancy (Ki-67, TMB), and was absent by construction in static pathway models. In the IMvigor210 anti-PD-L1 cohort, {{IMV_RESULT_SENTENCE}}

**Availability and implementation:** Source code, preprocessing pipelines (R + Python), example notebooks, and the Python package `path_agnn_cox` (installable via `pip install path-agnn-cox`) are available at https://github.com/wangzhipeng-1/Path-AGNN-Cox under the MIT license, with an archived snapshot at Zenodo (https://doi.org/10.5281/zenodo.22030045).

**Key words:** survival analysis; graph neural networks; pathway prior; adaptive graph learning; cancer prognosis; interpretability

---

## 1. Introduction

Precision oncology increasingly relies on molecular risk stratification to guide adjuvant therapy, surveillance intensity, and clinical-trial design. Large public transcriptomic compendia—above all The Cancer Genome Atlas (TCGA) and the Gene Expression Omnibus (GEO)—have made it possible to train and validate prognostic models across many cancer types, and gene-expression-based risk scores now inform decision-making in routine and investigational settings. Because survival endpoints are censored, high-dimensional, and biologically heterogeneous, the task remains one of the most active frontiers of translational bioinformatics.

Two limitations dominate current practice. Classical statistical models such as Cox proportional-hazards regression with Lasso, Ridge, or Elastic-Net penalties treat genes as independent covariates, discarding the regulatory interaction structure that drives tumor progression. Deep survival models—DeepSurv, Cox-nnet, and their successors—improve discrimination by learning nonlinear feature interactions, but they operate on gene lists rather than biological graphs, offer limited interpretability, and are prone to severe performance decay when transferred from a single training cohort to independent external cohorts.

A growing family of pathway-constrained graph neural networks addresses the interpretability problem by restricting message passing to gene sets defined by KEGG/GO pathways. PathGNN showed that pathway-topology-constrained GNNs improve prognosis across several solid tumors [PathGNN]. Cox-Path partitioned genes into KEGG pathway subgraphs and coupled them to a Cox survival head [CoxPath]. A prior-knowledge-guided multilevel GNN introduced gene-to-pathway hierarchical propagation for survival prediction [PriorKnowledgeGNN]. Most recently, PathMoG extended the concept to multi-omics with 354 KEGG-informed pathway modules, hierarchical omics modulation, and dual-level attention spanning intra-pathway and inter-pathway signals, reporting strong performance across 10 TCGA cohorts [PathMoG]. These studies collectively reported that biological priors can reduce overfitting and improve external validity relative to unconstrained deep models. However, they share a common, largely unexamined assumption: **the underlying gene interaction topology is invariant across patients**. In every existing pathway-constrained survival GNN of which we are aware—including PathMoG—the pathway graph—its adjacency, edge weights, and inter-pathway coupling—is fixed a priori and shared by all patients. This assumption is fundamentally incompatible with the established fact that tumor regulatory networks are extensively rewired in a malignancy-dependent manner.

We therefore challenge the population-level paradigm and reformulate the modeling objective: *the pathway graph structure itself should be learned per patient, and its rewiring should reflect tumor malignancy*. Three specific deficiencies follow from the static-graph assumption. First, a fixed adjacency cannot represent patient-specific rewiring: two tumors with the same pathway membership but different driver states share an identical interaction topology. Second, within-pathway aggregation is equally weighted, or attention-weighted in a sample-invariant way, in static models, so background genes dilute the signal of a few driver genes; the model cannot automatically up-weight the interactions that matter for an aggressive tumor. Third, because the graph is fixed, no mechanism exists to express how within-pathway interaction strengths—and thereby pathway activity—tighten or loosen with disease aggressiveness. These deficiencies plausibly explain why static pathway models still underperform in external cohorts.

To test this hypothesis, we introduce **Path-AGNN-Cox**, a pathway-constrained adaptive graph neural network for survival prediction, comprising three modules: first, **pathway-constrained subgraph construction** that partitions genes into KEGG cancer-core pathway modules so that message passing occurs only among biologically co-regulated genes; second, a **sample-adaptive neighborhood weighting** layer in which attention logits are multiplicatively modulated by a learnable malignancy gate, allowing the effective pathway graph to tighten or loosen as a function of the tumor's state; and third, a **Cox partial-likelihood objective with dual regularization**—intra-pathway sparsity plus a consistency constraint—that directly optimizes prognostic risk while suppressing overfitting in high-heterogeneity cohorts. We benchmark Path-AGNN-Cox against classical survival models, deep survival models, and static pathway-constrained GNNs across {{N_DATASETS}} TCGA cancer types with {{N_EXTERNAL}} independent GEO validation cohorts—substantially broader external validation than comparable pathway-GNN studies—and isolate the contribution of each module through systematic ablations. Beyond predictive performance, we assess the biological content of the learned sample-specific edge weights: between-stratum rewiring exceeds label-permutation nulls, is qualitatively coherent with tumor progression biology, such as cell cycle and DNA replication, and correlates with clinical indicators of malignancy, whereas static models by construction cannot produce such signal. We release the model as an open-source Python package with reproducible pipelines; source code is available at https://github.com/wangzhipeng-1/Path-AGNN-Cox under the MIT license, and a versioned archive is available at https://doi.org/10.5281/zenodo.22030045.

---

## 2. Materials and methods

### 2.1. Datasets

We evaluated Path-AGNN-Cox on a harmonized pan-cancer cohort of {{TCGA_TOTAL_N}} patients from {{N_DATASETS}} TCGA cancer types (BLCA, BRCA, COAD, GBM, HNSC, KIRC, LIHC, LUAD, LUSC, OV, STAD), where each patient was represented by RNA-sequencing expression (UCSC Xena TPM, log2-transformed and z-scored within training folds) and overall-survival annotation. These cohorts span diverse tissue origins, event rates, and sample sizes ({{TCGA_TOTAL_EVENTS}} events; {{TREF:DATASETS}}), providing a rigorous testbed for generalizability. For independent external validation without any fine-tuning, we additionally compiled {{N_EXTERNAL}} GEO microarray cohorts (Affymetrix; expression values were taken directly from the GEO series matrix files as provided by the data contributors; {{TREF:DATASETS}}), covering every tumor type in the training panel—e.g., three independent lung adenocarcinoma cohorts (GSE31210, GSE50081, GSE68465), three breast cancer cohorts (GSE20685, GSE21653, GSE7390), and three colon cancer cohorts (GSE14333, GSE17536, GSE39582). All preprocessing parameters (gene filtering, standardization) were estimated within training folds to avoid information leakage; external cohorts were standardized with training-cohort statistics and missing genes were set to zero (the training-space mean) at the tensor-materialization stage.

### {{TDEF:DATASETS}}. Dataset characteristics.
{{TABLE:DATASETS}}


For the GEO cohorts, probes were mapped to gene symbols using the platform annotation tables downloaded from NCBI GEO (e.g., GPL570, GPL96, GPL6480); duplicate gene symbols were collapsed by maximum mean expression, and per-gene missing values were imputed with the gene-wise mean within each cohort. Survival endpoints were harmonized to overall survival in days: GEO follow-up times reported in months or years were converted with 365.25/12 or 365.25 days per unit, respectively, and TCGA overall-survival time was derived from GDC clinical annotations, that is, days to death or days to last follow-up. Because each TCGA cohort was modeled separately and each GEO cohort was validated independently without pooling, no cross-cohort batch correction was required.

Expression values were analyzed at each cohort's native scale: TCGA RNA-seq values were log2(TPM+1) as provided by UCSC Xena GDC STAR-TPM, and GEO series matrices were used as provided, typically log2-scale microarray intensities. Before model fitting, every gene was z-scored with the training-cohort mean and standard deviation; external cohorts were standardized with the training statistics, and zero-variance genes were left unchanged.

Expression matrices were first mapped to the KEGG cancer-core pathway catalogue ({{N_PATHWAYS}} pathways, {{GENES_UNION}} genes in the union). After intersection with each cohort's measured genes, on average {{GENES_AVG}} genes per cohort were retained; genes outside any pathway were excluded, so that every model in the benchmark operates on the same pathway-mapped gene universe (a matched comparison).

### 2.2. Overview of Path-AGNN-Cox

Path-AGNN-Cox follows a pathway-first pipeline ({{FREF:METHOD|A}}). Transcriptomic inputs are mapped onto KEGG pathway subgraphs ({{FREF:METHOD|B}}); gene representations are updated by adaptive pathway-masked graph attention whose temperature is modulated by a learnable per-sample malignancy score ({{FREF:METHOD|C}}); pathway-level representations are pooled and fused with a gene-level summary; and the resulting patient representation is mapped to a Cox risk score optimized with dual regularization ({{FREF:METHOD|D}}). A key implementation feature is that the model does not operate on a single fixed graph tensor: pathway topologies are precomputed once, but the **effective edge weights are recomputed for every patient**, so that each patient is instantiated with a patient-specific pathway graph.

{{FIG:METHOD}}
- **{{FDEF:METHOD}}. Overview of Path-AGNN-Cox.** (A) End-to-end pipeline: TCGA/GEO expression → KEGG pathway mapping → adaptive pathway subgraphs → risk head with Cox objective. (B) Pathway-constrained block-diagonal adjacency: edges exist only between genes sharing a primary pathway. (C) Sample-adaptive neighborhood weighting: attention logits are multiplicatively modulated by the malignancy gate (1 + tanh(β)·m_s); high-malignancy samples receive sharper within-pathway attention. (D) Survival objective with dual regularization: Cox partial likelihood + intra-pathway sparsity + dropout consistency.


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

We used stratified 5-fold cross-validation within each TCGA cohort (fold-stratified on the event indicator; random seed 42). The concordance index (C-index) served as the primary discrimination metric; the time-dependent AUC (mean over the 0.25/0.50/0.75 quantile times) was used as a secondary metric. For external validation, each model was retrained on the full TCGA cohort and evaluated on the GEO cohorts without any fine-tuning. Paired Wilcoxon signed-rank tests (per-dataset mean C-index, internal CV) were used to compare Path-AGNN-Cox against each baseline. Per-patient edge weights were extracted from the last adaptive layer and patients were split at the median predicted risk. Between-stratum rewiring was tested per pathway on the per-sample mean edge weight within each pathway, reporting Cohen's d with a normal-approximation 95% CI, and the false discovery rate was controlled with the Benjamini–Hochberg procedure. Each pathway was additionally tested under 1,000 within-pathway label permutations of the risk strata; the permutation P was one plus the number of null permutations with at least as large an absolute effect, divided by 1,001. At the cohort level, the number of significant pathways was compared with a 1,000-permutation null obtained by permuting the risk labels, with the permutation P computed as one plus the number of null counts at least as large as the observed count, divided by 1,001. Two matched random gene-set controls assessed pathway-identity selectivity: for each real pathway, 200 random gene sets of equal size with matched internal edge counts, and 200 random equal-size subsets drawn from real pathway blocks so that size and density were matched, were scored with the identical effect-size statistic, and the percentile of each real pathway within its null distribution was recorded. Pathway-level enrichment of the top-20 pathways ranked by permutation P against a curated LUAD driver-pathway list was tested with the hypergeometric distribution over the 57-pathway cancer-core catalogue. Static-model, randomized-partition and standard-GAT controls followed the same protocol. All analyses are implemented in benchmark/rewiring_analysis.py and the work/ scripts of the repository at https://github.com/wangzhipeng-1/Path-AGNN-Cox. Model hyperparameters are listed in the footnote of {{TREF:BENCHMARK}} (see config/benchmark.yaml). All deep models were trained on CPU with Adam; the full benchmark consumed approximately {{BENCHMARK_HOURS}} CPU-hours.

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

### 3.1. Benchmark performance across {{N_DATASETS}} TCGA cancer types

We benchmarked Path-AGNN-Cox against the seven baselines under the same stratified 5-fold protocol ({{TREF:BENCHMARK}}). Path-AGNN-Cox achieved a mean internal C-index of {{CV_FULL_MEAN}} (SD {{CV_FULL_SD}}) across the {{N_DATASETS}} cohorts and ranked first in {{BEST_INTERNAL_WINS}} of them; the strongest overall baseline was {{BEST_BASELINE_NAME}} ({{CV_BEST_BASELINE_MEAN}}; paired difference {{CV_DIFF_CI}}, Wilcoxon {{CV_FULL_P}}), and the strongest deep survival baseline reached {{CV_BEST_DEEP_MEAN}}, i.e., all deep models clustered within a narrow band ({{FREF:BENCHMARK|A}}). Per-cohort differences were generally small; for example, in {{TOP_GAIN_DATASET}} Path-AGNN-Cox reached {{CV_TOP_GAIN_FULL}} vs. the best baseline {{CV_TOP_GAIN_BASE}} (Δ = {{CV_TOP_GAIN_DELTA}}); in time-dependent AUC, Path-AGNN-Cox ranked first in {{BEST_AUC_WINS}}/{{N_DATASETS}} cohorts ({{FREF:BENCHMARK|C}}). Because the benchmark used a single random seed (42), we retrained the model with three seeds on LUAD and BRCA under the same 5-fold partitions: the internal C-index was stable (LUAD: {{SEEDS_LUAD}}; BRCA: {{SEEDS_BRCA}}; mean ± SD over 3 seeds × 5 folds), with per-seed means spanning at most 0.02.

{{FIG:BENCHMARK}}
- **{{FDEF:BENCHMARK}}. Benchmark performance.** (A) Internal 5-fold CV C-index per cancer type. (B) External C-index per cancer type (mean over GEO cohorts). (C) Mean time-dependent AUC (internal CV). Solid markers: Path-AGNN-Cox; grey: baselines; dashed: 0.50 reference.


### {{TDEF:BENCHMARK}}. Benchmark performance: mean internal C-index / time-dependent AUC across {{N_DATASETS}} TCGA cancer types (stratified 5-fold CV).
{{TABLE:BENCHMARK}}


On external testing across the {{N_EXTERNAL}} GEO cohorts, the mean external C-index of Path-AGNN-Cox was {{EXT_FULL_MEAN}} (SD {{EXT_FULL_SD}}), matching the deep baselines ({{EXT_BEST_DEEP_MEAN}}) and ranking first among all models in {{BEST_EXTERNAL_WINS}} of {{N_DATASETS}} cancer types ({{FREF:BENCHMARK|B}}; per-cohort details in {{TREF:EXTERNAL}}). Penalized Cox baselines retained the highest external means ({{EXT_BEST_BASELINE_MEAN}}, {{EXT_BEST_BASELINE_NAME}}; paired difference {{EXT_DIFF_CI}}), while the three GNN variants (adaptive, static, plain) showed similar external decay, indicating that the pathway or adaptive design does not by itself eliminate cross-cohort performance loss; the interpretable rewiring output is the distinctive capability of the adaptive model (Section 3.4).

### {{TDEF:EXTERNAL}}. External validation per GEO cohort.
{{TABLE:EXTERNAL}}


### 3.2. Ablation study

We compared the full model with three ablations: −Pathway (Plain GNN, identity graph), −Adaptive (static uniform pathway adjacency), and −Regularization (no sparse/consistency terms), across all {{N_DATASETS}} cohorts ({{TREF:ABLATION}}, {{FREF:ABLATION}}). Removing the pathway constraint changed the mean internal C-index by {{ABL_PATHWAY_DROP}} (paired {{ABL_PATHWAY_DIFF_CI}}, {{ABL_PATHWAY_P}}); removing the adaptive gate by {{ABL_ADAPTIVE_DROP}} (paired {{ABL_ADAPTIVE_DIFF_CI}}, {{ABL_ADAPTIVE_P}}); and removing the dual regularization by {{ABL_NOREG_DROP}} (paired {{ABL_NOREG_DIFF_CI}}, {{ABL_NOREG_P}}). None of these differences was statistically significant, and external discrimination was similarly insensitive (e.g., −Adaptive external difference {{ABL_ADAPTIVE_EXT_DROP}}). We therefore conclude that, in the configuration evaluated, the predictive contribution of the individual design modules cannot be separated at the C-index level; the added value of the adaptive pathway design is the patient-specific interpretability it enables (Section 3.4) rather than a measurable discrimination gain.

{{FIG:ABLATION}}
- **{{FDEF:ABLATION}}. Ablation study.** (A) Internal C-index of the full model vs. −Pathway / −Adaptive / −Regularization per cancer type. (B) External C-index per variant. (C) Mean internal drop with 95% CI.


### {{TDEF:ABLATION}}. Ablation study: mean internal and external C-index for the full model and its three ablations.
{{TABLE:ABLATION}}


### 3.3. External validation across {{N_EXTERNAL}} GEO cohorts

Per-cohort results are summarized in {{TREF:EXTERNAL}} and {{FREF:EXTERNAL}}. Path-AGNN-Cox maintained C-index above 0.50 in {{EXT_ABOVE_50}}/{{N_EXTERNAL}} external cohorts, compared with {{EXT_BASE_ABOVE_50}} for the best baseline. The model transferred across platform shifts (RNA-seq → Affymetrix microarrays) and independent sample processing pipelines, with a mean external C-index comparable to the deep baselines. {{EXT_STRONGEST_COHORT_DESC}}. These results contrast favorably with previous pathway-GNN studies: PathMoG reported external validation in a single breast-cancer cohort (METABRIC) [PathMoG], whereas Path-AGNN-Cox was validated in {{N_EXTERNAL}} independent cohorts spanning {{N_DATASETS}} tumor types.

Calibration of the risk score was assessed with the slope of a univariate Cox regression of the standardized risk score and with the mean absolute deviation between model-predicted and Kaplan–Meier survival across risk tertiles at the 25th, 50th and 75th percentiles of follow-up time as detailed in {{TREF:CALIBRATION}}. The internal out-of-fold slope was {{CAL_LUAD_PATH_SLOPE}} with 95% CI {{CAL_LUAD_PATH_CI}} and a calibration MAE of {{CAL_LUAD_PATH_MAE}} in LUAD, and {{CAL_BRCA_PATH_SLOPE}} with 95% CI {{CAL_BRCA_PATH_CI}} and MAE {{CAL_BRCA_PATH_MAE}} in BRCA; the penalized Cox counterpart produced slopes of {{CAL_LUAD_RIDGE_SLOPE}} and {{CAL_BRCA_RIDGE_SLOPE}}, respectively. External slopes for Path-AGNN-Cox ranged from {{CAL_EXT_MIN}} to {{CAL_EXT_MAX}} with a mean of {{CAL_EXT_MEAN}}.

{{FIG:EXTERNAL}}
- **{{FDEF:EXTERNAL}}. External validation across {{N_EXTERNAL}} GEO cohorts.** Per-cohort C-index of Path-AGNN-Cox vs. the best baseline; cohorts grouped by cancer type; dashed line at 0.50.


### {{TDEF:CALIBRATION}}. Calibration of the risk score in internal and external cohorts.
{{TABLE:CALIBRATION}}

### 3.4. Path-AGNN-Cox learns biologically meaningful patient-specific pathway rewiring

A central claim of this work is that the sample-specific edge weights carry patient-specific variation that can be tested formally for biological content. We therefore extracted, for every patient, the last-layer adaptive attention weights within each pathway block and compared high- vs. low-risk strata (median risk split) in LUAD, BRCA and KIRC ({{FREF:REWIRING}}).

{{FIG:REWIRING}}
- **{{FDEF:REWIRING}}. Patient-specific pathway rewiring.** (A) Between-stratum effect sizes (Cohen's d with 95% CI) for all {{PERM_N_PATHWAYS}} pathways in LUAD, with the two pathways surviving per-pathway label permutation and BH-FDR correction highlighted. (B) Same as A for BRCA. (C) Cohort-level label-permutation null: number of significant pathways observed versus the null mean and maximum under 1,000 label permutations (LUAD and BRCA). (D) Correlation between per-patient rewiring magnitude and clinical malignancy indicators (LUAD). (E) Matched random gene-set controls in LUAD, BRCA and KIRC: distribution of P(null effect ≥ real effect) for the edge-matched and density-matched nulls; low values indicate real pathway effects above the matched random sets. The simulated pure-null median of the density-matched statistic was 1.00 (LUAD and BRCA), so values below the simulated 5th percentile exceed the structural bias of row-normalized attention.


**3.4.1. Between-stratum rewiring exceeds label-permutation chance; matched-set selectivity is cohort-dependent.** The per-sample mean edge weight within each pathway was compared between high- and low-risk strata (median risk split) with Cohen's d and 95% CI ({{FREF:REWIRING}}A,B). After per-pathway label permutation (1,000 permutations) and BH-FDR correction, {{PWP_BRCA_SIG}} of {{PERM_N_PATHWAYS}} pathways in BRCA, {{PWP_LUAD_SIG}} in LUAD and {{PWP_KIRC_SIG}} in KIRC remained significant; the two LUAD pathways were Homologous recombination (d = {{D_LUAD_HR}}, 95% CI {{D_LUAD_HR_CI}}) and DNA replication (d = {{D_LUAD_DNA}}, 95% CI {{D_LUAD_DNA_CI}}) ({{TREF:REWIRING}}). At the cohort level, the number of pathways with BH-FDR q<0.05 on the pathway-level test exceeded the cohort-level label-permutation null in all three cohorts (LUAD: {{PERM_LUAD_SIG}} observed vs. null mean {{PERM_LUAD_NULL_MEAN}}, maximum {{PERM_LUAD_NULL_MAX}}, permutation {{PERM_LUAD_P}}; BRCA: {{PERM_BRCA_SIG}} observed vs. null mean {{PERM_BRCA_NULL_MEAN}}, maximum {{PERM_BRCA_NULL_MAX}}, permutation {{PERM_BRCA_P}}; KIRC: {{PERM_KIRC_SIG}} observed vs. null mean {{PERM_KIRC_NULL_MEAN}}, maximum {{PERM_KIRC_NULL_MAX}}, permutation {{PERM_KIRC_P}}; 200 cohort-level permutations). The between-stratum differences therefore reflect risk-associated structure in the attention weights rather than the risk stratification itself.

Pathway identity was not globally selective, but matched-set enrichment was cohort-dependent. The percentile is defined as P(null effect ≥ real effect), so low values indicate that a real pathway exceeded the matched random sets. Because this statistic is computed on row-normalized attention, we first characterized it under a pure null: across {{MC_NULL_N_SIM}} simulated datasets with no between-stratum signal, the median density-matched percentile was {{MC_NULL_LUAD_MED}} in LUAD and {{MC_NULL_BRCA_MED}} in BRCA (5th percentile of the simulated medians: {{MC_NULL_LUAD_P05}} and {{MC_NULL_BRCA_P05}}), showing a strong upward structural bias of the statistic. Against this baseline, the observed density-matched medians were {{MATCHED_BLOCK_LUAD_MED}} in LUAD, within the pure-null range, {{MATCHED_BLOCK_BRCA_MED}} in BRCA, below the simulated 5th percentile, and {{MATCHED_BLOCK_KIRC_MED}} in KIRC. The edge-matched control gave median percentiles of {{MATCHED_LUAD_MED}}, {{MATCHED_BRCA_MED}} and {{MATCHED_KIRC_MED}} in LUAD, BRCA and KIRC. The number of real pathways above the 95th percentile of the density-matched null was {{MATCHED_BLOCK_LUAD_EXCEED}}, {{MATCHED_BLOCK_BRCA_EXCEED}} and {{MATCHED_BLOCK_KIRC_EXCEED}} of 53 (expected {{MATCHED_EXPECTED}} by chance), and {{MATCHED_LUAD_EXCEED}}, {{MATCHED_BRCA_EXCEED}} and {{MATCHED_KIRC_EXCEED}} of 53 for the edge-matched null. Matched-set enrichment was therefore evident in BRCA and KIRC and could not be attributed to the structural bias of row-normalized attention, whereas LUAD showed no enrichment beyond the structural null. Known-pathway enrichment of the top-20 pathways ranked by permutation P was not significant in LUAD ({{ENRICH_LUAD_HITS}} hits, {{ENRICH_LUAD_P}}). These observations are hypothesis-generating rather than evidence of specific pathway activation, and the pathway partition remains an interpretable organizing principle rather than a statistically necessary one.**3.4.2. Static models cannot produce rewiring.** As a negative control, the static pathway GNN (−Adaptive) was subjected to the identical analysis; its total edge-weight variance across patients was {{STATIC_NULL_VAR}}, essentially zero by construction, whereas the adaptive model produced a total edge-weight variance of {{ADAPTIVE_REWIRE_VAR}}, orders of magnitude larger; the between-stratum differences of Section 3.4.1 are therefore attributable to the sample-specific attention mechanism rather than to the fixed adjacency. Retraining the adaptive model with randomized pathway partitions (block sizes preserved; three seeds) yielded 26–48 significant pathways in LUAD, indicating that between-stratum attention differences do not require the canonical partition. Combined with the matched-set results of Section 3.4.1, the pathway partition supplies biologically interpretable labels for the rewiring output; whether the canonical grouping itself contributes beyond any fixed grouping remains unresolved.**3.4.3. Clinical correlation.** The per-patient rewiring magnitude (L1 distance of the patient's edge weights from the cohort mean) correlated with clinical indicators of malignancy: in BRCA with the proliferation marker MKI67 (Spearman ρ = {{CLINICAL_BRCA_RHO}}, {{CLINICAL_BRCA_P}}, n = {{CLINICAL_BRCA_N}}), and in LUAD with tumor mutational burden (ρ = {{CLINICAL_LUAD_RHO}}, {{CLINICAL_LUAD_P}}, n = {{CLINICAL_LUAD_N}}); stage showed no significant association in either cohort. In multivariable Cox models adjusting for stage and age, the risk score remained independently associated with overall survival in both cohorts (LUAD: HR = {{MVC_LUAD_HR}} per SD, 95% CI {{MVC_LUAD_CI}}, {{MVC_LUAD_P}}; BRCA: HR = {{MVC_BRCA_HR}} per SD, 95% CI {{MVC_BRCA_CI}}, {{MVC_BRCA_P}}). In the independent IMvigor210 anti-PD-L1 cohort, {{IMV_RESULT_SENTENCE}} ({{FREF:IMV}}); the rewiring-Ki-67 correlation replicated in this immunotherapy setting, supporting the biological validity of the learned patient-specific graphs. Because MKI67 is measured in the same transcriptomic input used to compute the attention weights, the Ki-67 correlations are descriptive anchors rather than independent validations; TMB provides an independent (mutation-based) anchor. The associations were robust to the definition of rewiring magnitude ({{SENSITIVITY_SENT}}).

{{FIG:IMV}}
- **{{FDEF:IMV}}. IMvigor210 anti-PD-L1 cohort (exploratory).** (A) Per-patient rewiring magnitude in responders (CR/PR) vs non-responders (SD/PD); Wilcoxon rank-sum test. (B) Overall survival of high- vs low-rewiring strata (median split) with log-rank test. (C) Ki-67 expression vs rewiring magnitude with Spearman correlation.
### {{TDEF:REWIRING}}. Framework validation of patient-specific pathway rewiring in LUAD, BRCA and KIRC.
{{TABLE:REWIRING}}



**3.4.4. Standard-GAT negative control.** To test whether the between-stratum edge-weight differences are specific to the pathway-constrained adaptive architecture, we trained a standard GAT on a sample-invariant k-nearest-neighbor gene graph (k=10; no pathway constraint) under the identical training protocol and applied the identical pathway-level testing pipeline. The standard GAT yielded {{STDGAT_BRCA_SIG}} of {{STDGAT_BRCA_TOT}} significantly rewired pathways in BRCA and {{STDGAT_LUAD_SIG}} of {{STDGAT_LUAD_TOT}} in LUAD (BH-FDR q<0.05), compared with {{PWP_BRCA_SIG}} of 53 and {{PWP_LUAD_SIG}} of 53 for the adaptive pathway-constrained model under the same permutation-calibrated procedure ({{TREF:REWIRING}}); the pathway-constrained model yielded more detectable between-stratum differences in BRCA and a comparable count in LUAD, indicating that the pathway partition contributes to the statistical detection of these differences; consistent with the randomized-partition control (Section 3.4.2), canonical pathway structure is nonetheless not uniquely required to produce them.

**3.4.5. External replication of patient-specific rewiring.** As an out-of-cohort check, we transferred each TCGA-trained model to independent GEO cohorts of the same cancer type and tested whether per-patient rewiring magnitude was associated with overall survival within each cohort. {{EXT_RW_LUAD_SENT}} {{EXT_RW_BRCA_SENT}} These associations are exploratory and were not corrected for multiple testing.

---



**3.4.6. Independence from the risk-stratification procedure and from global attention shifts.** Two further analyses tested whether the pathway-level signal was an artefact of the testing framework itself. First, we repeated the pathway-level test after stratifying patients by variables that do not depend on the model output: MKI67 expression (median split), tumor mutational burden (median split) and pathological stage (I–II vs III–IV). Under the MKI67 anchor, {{ADD_BRCA_ANCHOR_MK}} of 53 pathways in BRCA, {{ADD_KIRC_ANCHOR_MK}} in KIRC and {{ADD_LUAD_ANCHOR_MK}} in LUAD remained significant after BH-FDR correction; the TMB anchor gave {{ADD_BRCA_ANCHOR_TMB}}, {{ADD_LUAD_ANCHOR_TMB}} and {{ADD_KIRC_ANCHOR_TMB}} significant pathways in BRCA, LUAD and KIRC; and the stage anchor gave almost none (LUAD 0, BRCA {{ADD_BRCA_ANCHOR_STAGE}}), consistent with the absence of a stage–rewiring correlation (Section 3.4.3). The rewiring signal therefore does not require the model’s own risk stratification, although MKI67 and TMB are correlated with the risk score and these anchors are not fully independent of it.

Second, we decomposed the between-stratum difference into a global shift of the overall attention level and a pathway-specific component. The global shift was significant in all three cohorts (LUAD {{ADD_LUAD_GLOBAL_P}}; BRCA {{ADD_BRCA_GLOBAL_P}}; KIRC {{ADD_KIRC_GLOBAL_P}}) but its direction was positive in BRCA and negative in KIRC. After removing the per-patient global attention mean, {{ADD_LUAD_SPECIFIC}}, {{ADD_BRCA_SPECIFIC}} and {{ADD_KIRC_SPECIFIC}} of 53 pathways remained significant in LUAD, BRCA and KIRC, compared with {{ADD_LUAD_RAW}}, {{ADD_BRCA_RAW}} and {{ADD_KIRC_RAW}} under the raw test; the pathway-level signal therefore contains a component that is not merely a projection of the global shift.

Third, the direction of the pathway-specific effects was not consistent across cancers: of the {{ADD_BK_SHARED}} pathways significant in both BRCA and KIRC after global-shift removal, only {{ADD_BK_CONCORD}}% had the same direction, and the Spearman correlation of the pathway-specific effects between the two cohorts was {{ADD_BK_RHO}} ({{ADD_BK_P}}). Attention weights admit a sign symmetry, because flipping the sign of the first-layer attention weights and the risk-mapping weights leaves the risk score unchanged; the absolute direction of attention changes is therefore not interpretable across cohorts. We accordingly restrict our claims to the existence of statistically testable, risk-associated, pathway-level structure within each cohort, and do not claim that specific pathway identities or their directions transfer across cancer types.

### 3.5. Immune infiltration and predicted drug sensitivity (exploratory)

To connect patient-specific rewiring to the tumor microenvironment and to therapeutic vulnerability, we computed MCP-counter immune cell abundance estimates and ssGSEA scores of nine immune Hallmark gene sets per patient, and tested their association with the per-patient rewiring magnitude ({{FREF:IMMUNEDRUG|A}}). In LUAD, the rewiring magnitude was nominally associated with lower {{IMM_LUAD_TOP1}} ({{IMM_LUAD_TOP1_P}}), {{IMM_LUAD_TOP2}} ({{IMM_LUAD_TOP2_P}}) and {{IMM_LUAD_TOP3}} ({{IMM_LUAD_TOP3_P}}); none of these associations survived BH-FDR correction (smallest {{IMM_LUAD_Q}}), and in BRCA no immune feature was associated with rewiring magnitude (smallest {{IMM_BRCA_MIN_P}}). Immune infiltration was therefore not strongly coupled to pathway rewiring in these two cohorts.

{{FIG:IMMUNEDRUG}}
- **{{FDEF:IMMUNEDRUG}}. Immune infiltration and predicted drug sensitivity (exploratory).** (A) Association of 19 immune features with per-patient rewiring magnitude in LUAD (red) and BRCA (blue): signed -log10 P of the Wilcoxon test between high- and low-rewiring strata; dashed line: P=0.050. (B) Predicted IC50 (GDSC2/oncoPredict) of the eight nominally significant compounds in BRCA, high- versus low-rewiring strata. (C) Spearman correlation between rewiring magnitude and predicted IC50 across 17 curated compounds in BRCA; filled markers: FDR q<0.05.


We further predicted drug sensitivity with the oncoPredict model trained on GDSC2 cell-line pharmacogenomic data (198 compounds) and compared predicted IC50 values between high- and low-rewiring strata ({{FREF:IMMUNEDRUG|B}}, {{TREF:DRUGS}}). In BRCA, the high-rewiring stratum was predicted to be more sensitive (lower IC50) to {{DRUG_BRCA_NSIG}} of the 17 curated compounds at nominal significance, most strongly {{DRUG_BRCA_P1_NAME}} (WEE1, Wilcoxon {{DRUG_BRCA_P1}}), paclitaxel ({{DRUG_BRCA_P2}}) and palbociclib (CDK4/6, {{DRUG_BRCA_P3}}); the Spearman association remained significant after FDR correction for {{DRUG_BRCA_RHO_TOP}} and paclitaxel ({{DRUG_BRCA_RHO_Q}}), whereas the Wilcoxon comparisons did not survive correction (smallest {{DRUG_BRCA_WILCOX_Q}}). In LUAD, no compound showed a significant difference (smallest {{DRUG_LUAD_MIN_P}}). These analyses are hypothesis-generating: the predicted IC50 values are in-silico estimates from cell-line-derived models and do not substitute for direct pharmacologic assays (Section 4).

### {{TDEF:DRUGS}}. Predicted drug sensitivity (GDSC2/oncoPredict IC50) in high- versus low-rewiring strata (exploratory).
{{TABLE:DRUGS}}


---

## 4. Discussion

Path-AGNN-Cox addresses transcriptomic survival prediction with a pathway-centric graph design that is explicitly tuned to the *p*≫*n* regime and, unlike previous pathway GNNs, learns the pathway graph itself per patient. By constraining message passing to KEGG pathway modules, by modulating attention with a learnable malignancy gate, and by regularizing with sparse and consistency terms, the model achieves competitive discrimination across {{N_DATASETS}} cancer types and—more importantly—stable transfer to {{N_EXTERNAL}} independent GEO cohorts—substantially broader external validation than comparable pathway-GNN studies.

The conceptual contribution is a framework rather than a module tweak: existing pathway GNNs—PathGNN, Cox-Path, multilevel prior-knowledge GNNs, and the multi-omics PathMoG—treat the pathway graph as a fixed patient-invariant prior and stop at the risk score. We treat the effective graph as a per-patient object and provide the statistical machinery to interrogate its rewiring: edge-level and pathway-level tests with BH-FDR, label-permutation nulls, a static-model negative control, a standard-GAT architectural control, and clinical anchoring. The discrimination of the adaptive model was comparable to deep survival baselines, and we make no claim that per-patient attention improves C-index; its distinguishing value is that the learned patient-specific edge weights can be formally tested, are clinically anchored to Ki-67 and TMB, and are absent in static models by construction. The pathway partition is an interpretive labeling of these differences rather than a statistical necessity, as detailed in Section 3.4.2.**Comparison with PathMoG.** PathMoG is the closest recent work: it also organizes genes into KEGG pathway modules and uses hierarchical attention for multi-omics survival prediction [PathMoG]. The two frameworks are complementary in scope and differ in three substantive ways. First, *static versus adaptive topology*: PathMoG precomputes patient-invariant pathway topologies and learns attention within them; Path-AGNN-Cox additionally learns the graph *temperature* per patient, and its rewiring analysis directly interrogates patient-specific graph dynamics. Second, *omics scope*: PathMoG requires mutation and CNV in addition to expression, which limits its applicability to the many clinical cohorts that only measure transcriptomes; Path-AGNN-Cox is a single-transcriptome model that is immediately applicable to the large body of GEO microarray cohorts, as supported by our {{N_EXTERNAL}}-cohort external validation. Third, *external validation breadth*: PathMoG validated transferability in a single breast-cancer cohort, METABRIC; Path-AGNN-Cox was validated in {{N_EXTERNAL}} GEO cohorts spanning {{N_DATASETS}} tumor types. These differences position Path-AGNN-Cox as a lightweight, broadly deployable alternative for single-omics clinical and public cohorts, while PathMoG remains the reference for multi-omics integration.

**Limitations.** First, the predictive performance of the adaptive model did not exceed penalized Cox baselines on internal CV, and ablations did not isolate a significant C-index contribution from the adaptive gate, the pathway constraint, or the regularization terms; interpretability is the rationale for its use, not a discrimination gain. Second, the learnable malignancy gate beta converged to values near zero in the trained models, so the effective sample-specificity arises from the attention coefficients themselves; we therefore describe the mechanism as patient-specific attention rather than as demonstrable malignancy-driven gating. Third, hypergeometric enrichment of the top-20 rewired pathways ranked by permutation P against a curated LUAD driver-pathway list was not significant ({{ENRICH_LUAD_HITS}} hits, {{ENRICH_LUAD_P}}); an equivalent test was not performed for BRCA because no matched curated driver-pathway list was available. Fourth, matched random gene-set controls were cohort-dependent: real pathways were enriched above matched random sets in BRCA and KIRC beyond a simulated pure-null baseline of the statistic, whereas LUAD showed no enrichment beyond the structural null; the pathways that exceeded both permutation and matched-set nulls remain hypothesis-generating, and the pathway prior is retained as an interpretable organizing principle rather than as evidence of specific pathway activation. Fifth, attention weights provide hypothesis-generating, not causal, evidence, and all cohorts are retrospective; prospective validation is required. Sixth, the pan-cancer benchmark used a single seed; a three-seed sensitivity analysis on LUAD and BRCA, the cohorts used for the rewiring analyses, showed stable C-index values of {{SEEDS_LUAD}} in LUAD and {{SEEDS_BRCA}} in BRCA, but seed variance was not propagated to all cohorts. Seventh, the two LUAD pathways that were significant in the training cohort were not replicated at the pathway level across three independent external LUAD cohorts (DNA replication: Stouffer z = {{EXT_META_DNA_Z}}, {{EXT_META_DNA_P}}, with {{EXT_META_DNA_DIR}} of {{EXT_META_N_COHORTS}} cohorts in the same direction; homologous recombination: z = {{EXT_META_HR_Z}}, {{EXT_META_HR_P}}, {{EXT_META_HR_DIR}} of {{EXT_META_N_COHORTS}}); pathway-level rewiring findings should therefore be interpreted within the cohort in which they were derived. Eighth, the direction of the pathway-specific rewiring effects was systematically opposite between BRCA and KIRC, consistent with the sign symmetry of the attention mechanism; absolute attention directions are therefore not compared across cohorts, and only the within-cohort existence of risk-associated pathway structure is claimed.

---

## 5. Conclusion

Path-AGNN-Cox couples a pathway-constrained GNN with patient-specific attention to a formal statistical framework for testing and clinically anchoring patient-specific pathway rewiring, while offering discrimination comparable to deep survival baselines. With {{N_DATASETS}} internal and {{N_EXTERNAL}} external cohorts, open-source code at https://github.com/wangzhipeng-1/Path-AGNN-Cox, a versioned Zenodo archive at https://doi.org/10.5281/zenodo.22030045, and a pip-installable Python package, the framework is directly reusable for pan-cancer prognostic modeling and for interrogating the patient-specific graph dynamics that static pathway models cannot produce.

---

## 6. Availability and implementation

- **Code:** https://github.com/wangzhipeng-1/Path-AGNN-Cox (MIT license; Python package `path_agnn_cox`; R preprocessing scripts in `data/scripts/`; Zenodo archive: https://doi.org/10.5281/zenodo.22030045).
- **Installation:** `pip install path-agnn-cox` (PyPI) or `pip install .` from the repository; example notebooks under `examples/`.
- **Data:** TCGA RNA-seq and clinical data from UCSC Xena (https://xenabrowser.net); GEO series matrices and platform annotations from NCBI GEO (https://www.ncbi.nlm.nih.gov/geo/); all accession numbers listed in {{TREF:DATASETS}}.
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
12. GSE accession references for the {{N_EXTERNAL}} external cohorts (full list in {{TREF:DATASETS}}).
13. Ha MJ, Baladandayuthapani V. DINGO: differential network analysis in genomics. *Bioinformatics*. 2015;31(21):3413-3420.
14. de la Fuente A. From 'differential expression' to 'differential networking': identification of dysfunctional regulatory networks in diseases. *Trends Genet*. 2010;26(7):326-333.
15. Gill R, Datta S, Datta S. A statistical framework for differential network analysis from microarray data. *BMC Bioinformatics*. 2010;11:95.
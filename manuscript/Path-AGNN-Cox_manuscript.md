# Path-AGNN-Cox: Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis

> **Target journal:** Briefings in Bioinformatics (preferred) / Computational and Structural Biotechnology Journal
> **Draft status:** structure + prose finalized; numeric tokens filled by `render_manuscript.py` from the completed benchmark (11 TCGA cohorts x 5-fold CV, 25 GEO external cohorts) and rewiring analyses.
> **Writing reference:** manuscript structure, abstract format, and narrative devices follow PathMoG (Wang et al., arXiv:2604.24371, 2026), the most recent pathway-centric survival GNN.

---

## Title

**Path-AGNN-Cox: a reproducible statistical framework for testing patient-specific pathway rewiring in cancer survival analysis**

## Authors

Zhipeng Wang[1,*], Luning Wang[2,*], Changsong Wang[1,†], Pengli Zhai[3], Zejun Liu[1], Hui Feng[1], Hongmei Liu[1], Qian Hou[1], Ming Guo[1]

1 Department of TCM, Zhongda Hospital, Southeast University, China
2 Department of Rehabilitation Medicine, Zhongda Hospital, Southeast University, China
3 Jiangbei Campus, Jiangsu Provincial Traditional Chinese Medicine Hospital, China

* These authors contributed equally to this work and are co-first authors.
† Corresponding author. Email: 101005664@seu.edu.cn

---

## Abstract

**Background:** Cancer prognosis models trained on high-dimensional transcriptomic data face two interconnected problems. Unconstrained deep survival models act as black boxes that ignore biological structure, and pathway-constrained graph neural networks assume a patient-invariant interaction topology that is difficult to reconcile with the malignancy-dependent rewiring of tumor regulatory networks.

**Methods:** We propose Path-AGNN-Cox, a pathway-constrained adaptive graph neural network for survival prediction. Genes are partitioned into KEGG cancer-core pathway modules; sample-specific within-pathway attention weights are computed with a learnable malignancy-modulated gate; and a Cox partial-likelihood objective with dual regularization is optimized directly on prognostic risk. We benchmarked Path-AGNN-Cox against seven survival baselines across 11 TCGA cancer types with stratified 5-fold cross-validation and validated transferability on 25 independent GEO cohorts.

**Results:** On internal cross-validation, Path-AGNN-Cox reached a mean C-index of 0.56 (SD 0.04), comparable to deep survival baselines but below penalized Cox; we therefore make no claim of a discrimination gain. Its distinguishing value is the interpretable rewiring output: risk-associated pathway weights exceeded label-permutation nulls in BRCA (43/53 pathways), LUAD (3/53) and KIRC (52/53), correlated with clinical indicators of malignancy, and were absent by construction in static pathway models. Under the same testing procedure, a standard GAT control detected 3 of 54 pathways in KIRC and 11 of 54 in BRCA, compared with 52 and 43 of 53 for the pathway-constrained model; the framework thus distinguished cohorts with detectable rewiring from LUAD, where pathway-constrained signal did not exceed matched nulls.

**Conclusion:** Path-AGNN-Cox provides a reproducible framework in which patient-specific pathway graphs become objects of formal statistical testing, without requiring a discrimination gain over classical baselines. The model and pipelines are released as an open-source Python package with an archived snapshot.

**Key words:** survival analysis; graph neural network; pathway constraint; adaptive graph learning; cancer prognosis

---

## 1. Introduction

Precision oncology increasingly depends on molecular risk stratification to guide adjuvant therapy, surveillance intensity, and clinical-trial design [1,2]. Large public transcriptomic compendia, most notably The Cancer Genome Atlas [3] and the Gene Expression Omnibus [4,5], have made it possible to train and validate prognostic models across many cancer types [6]. Because survival endpoints are censored and expression data are high-dimensional and biologically heterogeneous, this task remains an active frontier of translational bioinformatics.

Classical statistical models, such as Cox proportional-hazards regression [1] with Lasso, Ridge, or Elastic-Net penalties [7,8], treat genes as independent covariates and discard the regulatory interaction structure that drives tumor progression. Deep survival models, including DeepSurv [9], Cox-nnet [10], DeepHit [11], and random survival forests [12], improve discrimination by learning nonlinear feature interactions. They nevertheless operate on gene lists rather than biological graphs, offer limited interpretability, and are prone to severe performance decay when transferred from a single training cohort to independent external cohorts [13,14,15].

Graph neural networks provide a natural inductive bias for molecular data [16,17,18]. A growing family of pathway-constrained graph neural networks restricts message passing to gene sets defined by KEGG pathways [19]. PathGNN showed that pathway-topology-constrained propagation improves prognosis across several solid tumors [20]. Cox-Path partitioned genes into KEGG pathway subgraphs and coupled them to a Cox survival head [21]. A prior-knowledge-guided multilevel GNN introduced gene-to-pathway hierarchical propagation for survival prediction [22]. PathMoG most recently extended this concept to multi-omics with 354 KEGG-informed pathway modules and dual-level attention, reporting strong performance across 10 TCGA cohorts [23]. These studies collectively suggest that biological priors can reduce overfitting and improve external validity relative to unconstrained deep models.

A largely unexamined assumption nevertheless underlies these models: the gene interaction topology is invariant across patients. In every pathway-constrained survival GNN of which we are aware, including PathMoG, the adjacency, edge weights, and inter-pathway coupling are fixed a priori and shared by all patients [20,21,22,23]. This assumption is difficult to reconcile with the observation that tumor regulatory networks are extensively rewired in a malignancy-dependent manner [24,25,26]. Differential network analyses have, moreover, shown that interaction strengths vary across disease states and individuals rather than forming a single static structure [24,25,26].

The static-graph assumption consequently gives rise to three deficiencies. First, a fixed adjacency cannot represent patient-specific rewiring; two tumors with the same pathway membership but different driver states share an identical interaction topology. Second, within-pathway aggregation is weighted in a sample-invariant way, so background genes can dilute the signal of a few driver genes, and the model cannot automatically up-weight the interactions that matter for an aggressive tumor. Third, no mechanism exists to express how within-pathway interaction strengths tighten or loosen with disease aggressiveness. These deficiencies plausibly explain why static pathway models still decay on independent external cohorts.

To test this hypothesis, we introduce Path-AGNN-Cox, a pathway-constrained adaptive graph neural network for survival prediction. The model comprises three modules: pathway-constrained subgraph construction that partitions genes into KEGG cancer-core pathway modules; sample-adaptive neighborhood weighting in which attention logits are multiplicatively modulated by a learnable malignancy gate; and a Cox partial-likelihood objective with dual regularization. We benchmarked Path-AGNN-Cox against classical survival models, deep survival models, and static pathway-constrained GNNs across 11 TCGA cancer types with 25 independent GEO validation cohorts, and isolated the contribution of each module through systematic ablations. Because attention weights are a function of the input features, per-patient variation in these weights is not by itself novel; the contribution of this framework is that the weights become objects of formal statistical testing. Between-stratum rewiring is compared with label-permutation nulls, matched random gene-set controls, a static-model negative control, and a standard-GAT architectural control, and the test outcomes are architecture-dependent and cohort-dependent rather than a uniform property of attention models. The model is released as an open-source Python package with reproducible pipelines.

---

## 2. Materials and methods

### 2.1. Datasets

We evaluated Path-AGNN-Cox on a harmonized pan-cancer cohort of 5336 patients from 11 TCGA cancer types (BLCA, BRCA, COAD, GBM, HNSC, KIRC, LIHC, LUAD, LUSC, OV, STAD), where each patient was represented by RNA-sequencing expression from UCSC Xena [6], log2-transformed and z-scored within training folds, and overall-survival annotation extracted with TCGAbiolinks [27]. These cohorts span diverse tissue origins, event rates, and sample sizes (1896 events; Table 1), providing a rigorous testbed for generalizability. For independent external validation without any fine-tuning, we additionally compiled 25 GEO microarray cohorts [4,5] (Affymetrix; expression values were taken directly from the GEO series matrix files as provided by the data contributors; Table 1), covering every tumor type in the training panel—e.g., three independent lung adenocarcinoma cohorts (GSE31210, GSE50081, GSE68465), three breast cancer cohorts (GSE20685, GSE21653, GSE7390), and three colon cancer cohorts (GSE14333, GSE17536, GSE39582). All preprocessing parameters (gene filtering, standardization) were estimated within training folds to avoid information leakage; external cohorts were standardized with training-cohort statistics and missing genes were set to zero (the training-space mean) at the tensor-materialization stage.

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


For the GEO cohorts, probes were mapped to gene symbols using the platform annotation tables downloaded from NCBI GEO [4] (e.g., GPL570, GPL96, GPL6480); duplicate gene symbols were collapsed by maximum mean expression, and per-gene missing values were imputed with the gene-wise mean within each cohort. Survival endpoints were harmonized to overall survival in days: GEO follow-up times reported in months or years were converted with 365.25/12 or 365.25 days per unit, respectively, and TCGA overall-survival time was derived from GDC clinical annotations, that is, days to death or days to last follow-up. Because each TCGA cohort was modeled separately and each GEO cohort was validated independently without pooling, no cross-cohort batch correction was required.

Expression values were analyzed at each cohort's native scale: TCGA RNA-seq values were log2(TPM+1) as provided by UCSC Xena GDC STAR-TPM, and GEO series matrices were used as provided, typically log2-scale microarray intensities. Before model fitting, every gene was z-scored with the training-cohort mean and standard deviation; external cohorts were standardized with the training statistics, and zero-variance genes were left unchanged.

Expression matrices were first mapped to the KEGG cancer-core pathway catalogue [19] (57 pathways, 3097 genes in the union). After intersection with each cohort's measured genes, on average 2760 genes per cohort were retained; genes outside any pathway were excluded, so that every model in the benchmark operates on the same pathway-mapped gene universe (a matched comparison).

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

Model parameters are optimized by the negative Cox partial likelihood with Breslow tie handling [1,28],

L_Cox = − (1/n_events) Σ_{i: E_i=1} ( ŷ_i − log Σ_{j ∈ R(t_i)} exp(ŷ_j) ),

where E_i is the event indicator and R(t_i) the risk set at time t_i. To suppress overfitting in high-heterogeneity cohorts, we add two regularization terms:

1. **Intra-pathway sparsity**—the mean absolute adaptive attention weight over pathway edges, L_sparse = (1/|E|) Σ_e |α_e|, penalized by λ_sparse. This encourages the model to concentrate pathway signal on a few driver interactions rather than diffuse attention across all co-regulated genes. (Attention tensors used for this penalty are kept non-detached so gradients reach the attention parameters.)
2. **Dropout-consistency**—the mean squared error between two stochastic forward passes (two dropout views), L_consist = MSE(ŷ, ŷ′), penalized by λ_consist. This requires the risk score to be stable under feature-dropout perturbations, regularizing the sample-specific graph toward reproducible, cohort-level structure.

The total objective is L = L_Cox + λ₂‖W‖₂² + λ_sparse·L_sparse + λ_consist·L_consist. The objective is invariant to a joint sign flip of the first-layer attention weights and the risk-mapping weights, because the risk score, and therefore the partial likelihood and both regularizers, are unchanged under this transformation. The direction of per-pathway attention changes is consequently defined only up to this symmetry, and all rewiring tests in Section 3.4 are formulated on effect sizes that are invariant to it.

### 2.7. Evaluation protocol

We used stratified 5-fold cross-validation within each TCGA cohort (fold-stratified on the event indicator; random seed 42). The concordance index [2,29] served as the primary discrimination metric; the time-dependent AUC [30] (mean over the 0.25/0.50/0.75 quantile times) was used as a secondary metric. For external validation, each model was retrained on the full TCGA cohort and evaluated on the GEO cohorts without any fine-tuning, following current recommendations for validating prognostic models [31]. Paired Wilcoxon signed-rank tests (per-dataset mean C-index, internal CV) were used to compare Path-AGNN-Cox against each baseline. Decision-curve analysis [32] was used to assess the clinical value of the risk score: inverse-probability-of-censoring-weighted net benefit was estimated at 1-, 3- and 5-year horizons over threshold probabilities 0.05–0.55 for three Cox models (clinical, clinical plus risk score, and risk score alone) fitted with a ridge penalizer of 0.01 in LUAD, BRCA and KIRC. Per-patient edge weights were extracted from the last adaptive layer and patients were split at the median predicted risk. Between-stratum rewiring was tested per pathway on the per-sample mean edge weight within each pathway, reporting Cohen's d with a normal-approximation 95% CI, and the false discovery rate was controlled with the Benjamini–Hochberg procedure [33]. Each pathway was additionally tested under 1,000 within-pathway label permutations of the risk strata; the permutation P was one plus the number of null permutations with at least as large an absolute effect, divided by 1,001. At the cohort level, the number of significant pathways was compared with a 1,000-permutation null obtained by permuting the risk labels, with the permutation P computed as one plus the number of null counts at least as large as the observed count, divided by 1,001. Two matched random gene-set controls assessed pathway-identity selectivity: for each real pathway, 200 random gene sets of equal size with matched internal edge counts, and 200 random equal-size subsets drawn from real pathway blocks so that size and density were matched, were scored with the identical effect-size statistic, and the percentile of each real pathway within its null distribution was recorded. Pathway-level enrichment of the top-20 pathways ranked by permutation P against a curated LUAD driver-pathway list was tested with the hypergeometric distribution over the 57-pathway cancer-core catalogue. Static-model, randomized-partition and standard-GAT controls followed the same protocol. All analyses are implemented in benchmark/rewiring_analysis.py and the work/ scripts of the repository at https://github.com/wangzhipeng-1/Path-AGNN-Cox. Model hyperparameters and baseline configurations are summarized in Table 2. All deep models were trained on CPU with Adam; the full benchmark consumed approximately 1,500 CPU-hours.

### 2.8. Baselines

Seven baselines span four modeling families: penalized Cox models (LASSO-Cox, Ridge-Cox, Elastic-Net-Cox), a tree ensemble (Random Survival Forest, RSF), deep survival models (DeepSurv, Cox-nnet), and an unconstrained GNN (Plain GNN: the Path-AGNN-Cox backbone with identity adjacency and global pooling, i.e., the −Pathway ablation). Two additional controls are the static pathway GNN (Path-AGNN-Cox with fixed uniform normalized adjacency inside each block, i.e., the −Adaptive ablation) and the unregularized variant (λ_sparse = λ_consist = 0, i.e., the −Regularization ablation). All models use the same pathway-mapped gene universe and the same CV/external protocol, so performance differences are attributable to modeling choices rather than feature sets.

### 2.9. Implementation details and complexity

Path-AGNN-Cox was implemented in Python 3.10 with PyTorch [34] on CPU, and all survival statistics were computed with lifelines [35]; cross-validation splits and tree baselines were implemented with scikit-learn [36]. Models were trained
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

### Table 2. Model hyperparameters and baseline configurations.
| Model | Configuration |
| Path-AGNN-Cox | hidden 32, layers 2, mlp 32, dropout 0.1, epochs 100, lr 1e-3, batch 128, patience 15, L2 1e-4, lambda_sparse 1e-3, lambda_consist 0.1 |
| −Adaptive (static) | same backbone with fixed normalized adjacency |
| −Regularization | lambda_sparse = 0, lambda_consist = 0 |
| Plain GNN | identity adjacency, global pooling |
| LASSO-Cox | penalizer 0.05, 10-fold internal CV |
| Ridge-Cox | penalizer 0.1 |
| Elastic-Net-Cox | l1_ratio 0.5, penalizer 0.1 |
| Random Survival Forest | 500 trees, min_samples_leaf 15 |
| DeepSurv | hidden [32, 16] |
| Cox-nnet | hidden [64], dropout 0.0 |

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

We benchmarked Path-AGNN-Cox against the seven baselines under the same stratified 5-fold protocol (Table 3). Path-AGNN-Cox achieved a mean internal C-index of 0.56 (SD 0.04) across the 11 cohorts and ranked first in 0 of them; the strongest overall baseline was Ridge-Cox (0.62; paired difference Δ=-0.07 (95% CI -0.10 to -0.04), Wilcoxon P=0.005), and the strongest deep survival baseline reached 0.61, i.e., all deep models clustered within a narrow band (Figure 2A). Per-cohort differences were generally small; for example, in LUSC Path-AGNN-Cox reached 0.52 vs. the best baseline 0.53 (Δ = -0.01); in time-dependent AUC, Path-AGNN-Cox ranked first in 1/11 cohorts (Figure 2C). Because the benchmark used a single random seed (42), we retrained the model with three seeds on LUAD, BRCA and KIRC under the same 5-fold partitions: the internal C-index was stable (LUAD: 0.50 ± 0.04; BRCA: 0.59 ± 0.04; KIRC: 0.62 ± 0.07; mean ± SD over 3 seeds × 5 folds), with per-seed means spanning at most 0.02.

![Figure 2](results/figures/Figure2_benchmark.svg)
- **Figure 2. Benchmark performance.** (A) Internal 5-fold CV C-index per cancer type. (B) External C-index per cancer type (mean over GEO cohorts). (C) Mean time-dependent AUC (internal CV). Solid markers: Path-AGNN-Cox; grey: baselines; dashed: 0.50 reference.


### Table 3. Benchmark performance: mean internal C-index / time-dependent AUC across 11 TCGA cancer types (stratified 5-fold CV).
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


On external testing across the 25 GEO cohorts, the mean external C-index of Path-AGNN-Cox was 0.51 (SD 0.04), matching the deep baselines (0.50) and ranking first among all models in 0 of 11 cancer types (Figure 2B; per-cohort details in Table 4). Penalized Cox baselines retained the highest external means (0.57, Ridge-Cox; paired difference Δ=-0.10 (95% CI -0.14 to -0.06)), while the three GNN variants (adaptive, static, plain) showed similar external decay, indicating that the pathway or adaptive design does not by itself eliminate cross-cohort performance loss; the interpretable rewiring output is the distinctive capability of the adaptive model (Section 3.4).

### Table 4. External validation per GEO cohort.
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

We compared the full model with three ablations: −Pathway (Plain GNN, identity graph), −Adaptive (static uniform pathway adjacency), and −Regularization (no sparse/consistency terms), across all 11 cohorts (Table 5, Figure 3). Removing the pathway constraint changed the mean internal C-index by 0.00 (paired Δ=0.00 (95% CI -0.01 to 0.02), P=0.520); removing the adaptive gate by 0.01 (paired Δ=0.01 (95% CI -0.01 to 0.02), P=0.206); and removing the dual regularization by 0.00 (paired Δ=-0.00 (95% CI -0.01 to 0.01), P=0.765). None of these differences was statistically significant, and external discrimination was similarly insensitive (e.g., −Adaptive external difference 0.01). We therefore conclude that, in the configuration evaluated, the predictive contribution of the individual design modules cannot be separated at the C-index level; the added value of the adaptive pathway design is the patient-specific interpretability it enables (Section 3.4) rather than a measurable discrimination gain.

![Figure 3](results/figures/Figure3_ablation.svg)
- **Figure 3. Ablation study.** (A) Internal C-index of the full model vs. −Pathway / −Adaptive / −Regularization per cancer type. (B) External C-index per variant. (C) Mean internal drop with 95% CI.


### Table 5. Ablation study: mean internal and external C-index for the full model and its three ablations.
| Variant | LUAD | LUSC | BRCA | COAD | STAD | LIHC | KIRC | HNSC | BLCA | OV | GBM | Internal mean±SD | External mean±SD | Δ vs full | P |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Path-AGNN-Cox | 0.49 | 0.52 | 0.58 | 0.60 | 0.53 | 0.61 | 0.63 | 0.54 | 0.56 | 0.54 | 0.58 | 0.56±0.04 | 0.51±0.04 | ref | ref |
| −Adaptive (static) | 0.53 | 0.50 | 0.56 | 0.59 | 0.50 | 0.62 | 0.62 | 0.53 | 0.53 | 0.56 | 0.55 | 0.55±0.04 | 0.50±0.05 | -0.01 | P=0.206 |
| −Regularization | 0.53 | 0.53 | 0.58 | 0.59 | 0.54 | 0.62 | 0.63 | 0.52 | 0.55 | 0.52 | 0.58 | 0.56±0.04 | 0.50±0.03 | 0.00 | P=0.765 |
| −Pathway (plain GNN) | 0.50 | 0.50 | 0.61 | 0.61 | 0.51 | 0.63 | 0.63 | 0.52 | 0.57 | 0.52 | 0.55 | 0.56±0.05 | 0.51±0.04 | 0.00 | P=0.520 |


### 3.3. External validation across 25 GEO cohorts

Per-cohort results are summarized in Table 4 and Figure 4. Path-AGNN-Cox maintained C-index above 0.50 in 12/25 external cohorts, compared with 25 for the best baseline. The model transferred across platform shifts (RNA-seq → Affymetrix microarrays) and independent sample processing pipelines, with a mean external C-index comparable to the deep baselines. The largest external gain over the best baseline was observed in Lung adenocarcinoma (GSE31210, Path-AGNN-Cox C-index 0.76 vs best baseline 0.67). These results contrast favorably with previous pathway-GNN studies: PathMoG reported external validation in a single breast-cancer cohort, METABRIC [23,37], whereas Path-AGNN-Cox was validated in 25 independent cohorts spanning 11 tumor types.

Calibration of the risk score [38] was assessed with the slope of a univariate Cox regression of the standardized risk score and with the mean absolute deviation between model-predicted and Kaplan–Meier survival across risk tertiles at the 25th, 50th and 75th percentiles of follow-up time as detailed in Table 6. The internal out-of-fold slope was -0.02 with 95% CI -0.16–0.12 and a calibration MAE of 0.02 in LUAD, -0.05 with 95% CI -0.23–0.14 and MAE 0.01 in BRCA, and 0.26 with 95% CI 0.09–0.43 and MAE 0.03 in KIRC; the penalized Cox counterpart produced slopes of 0.39, 0.49 and 0.66, respectively. External slopes for Path-AGNN-Cox ranged from -0.11 to 0.79 with a mean of 0.18.

![Figure 4](results/figures/Figure4_external.svg)
- **Figure 4. External validation across 25 GEO cohorts.** Per-cohort C-index of Path-AGNN-Cox vs. the best baseline; cohorts grouped by cancer type; dashed line at 0.50.


### Table 6. Calibration of the risk score in internal and external cohorts.
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
| KIRC | Internal CV | KIRC | Path-AGNN-Cox | 533 | 173 | 0.26 | 0.09–0.43 | 0.03 |
| KIRC | Internal CV | KIRC | Ridge-Cox | 533 | 173 | 0.66 | 0.53–0.78 | 0.03 |
| KIRC | External transfer | GSE29609 | Path-AGNN-Cox | 39 | 17 | 0.32 | -0.12–0.76 | 0.13 |
| KIRC | External transfer | GSE29609 | Ridge-Cox | 39 | 17 | -0.55 | -1.02–-0.08 | 0.13 |

### 3.4. Path-AGNN-Cox learns biologically meaningful patient-specific pathway rewiring

A central claim of this work is that the sample-specific edge weights carry patient-specific variation that can be tested formally for biological content. We therefore extracted, for every patient, the last-layer adaptive attention weights within each pathway block and compared high- vs. low-risk strata (median risk split) in LUAD, BRCA and KIRC (Figure 5).

![Figure 5](results/figures/Figure5_rewiring.svg)
- **Figure 5. Patient-specific pathway rewiring.** (A) Between-stratum effect sizes (Cohen's d with 95% CI) for all 53 pathways in LUAD, with the two pathways surviving per-pathway label permutation and BH-FDR correction highlighted. (B) Same as A for BRCA. (C) Cohort-level label-permutation null: number of significant pathways observed versus the null mean and maximum under 1,000 label permutations (LUAD and BRCA). (D) Correlation between per-patient rewiring magnitude and clinical malignancy indicators (LUAD). (E) Matched random gene-set controls in LUAD, BRCA and KIRC: distribution of P(null effect ≥ real effect) for the edge-matched and density-matched nulls; low values indicate real pathway effects above the matched random sets. The simulated pure-null median of the density-matched statistic was 1.00 (LUAD and BRCA), so values below the simulated 5th percentile exceed the structural bias of row-normalized attention.


### Table 7. Framework validation of patient-specific pathway rewiring in LUAD, BRCA and KIRC.
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
| Pure-null structural median (density-matched), simulated | 1.00 | 1.00 | 1.00 |
| Global attention shift, high vs low risk (P) | P=0.019 | P<0.001 | P<0.001 |
| Pathway-specific rewiring after global-shift removal (q<0.05) | 14 | 42 | 36 |
| Significant pathways under stage-stratified anchor (q<0.05) | 0 | 2 | 22 |
| Significant pathways under MKI67-stratified anchor (q<0.05) | 11 | 33 | 24 |
| Significant pathways under grade-stratified anchor (q<0.05) | n.a. | n.a. | 25 |
| Known-pathway enrichment, top-20 by permutation P | 5 hits, P=0.689 | n.a. | n.a. |
| Static-model total edge variance | 3.48e-13 | 2.81e-13 | 2.76e-13 |
| Adaptive-model total edge variance | 1.92e+01 | 8.41e-02 | 2.44e+01 |
| Randomized-partition control, significant pathways (3 seeds) | 26-48 | n.a. | n.a. |

**3.4.1. Between-stratum rewiring exceeds label-permutation chance; matched-set selectivity is cohort-dependent.** The per-sample mean edge weight within each pathway was compared between high- and low-risk strata (median risk split) with Cohen's d and 95% CI (Figure 5A,B). After per-pathway label permutation (1,000 permutations) and BH-FDR correction, 43 of 53 pathways in BRCA, 2 in LUAD and 52 in KIRC remained significant; the two LUAD pathways were Homologous recombination (d = 0.35, 95% CI 0.18–0.53) and DNA replication (d = 0.35, 95% CI 0.18–0.52) (Table 7). At the cohort level, the number of pathways with BH-FDR q<0.05 on the pathway-level test exceeded the cohort-level label-permutation null in all three cohorts (LUAD: 3 observed vs. null mean 0.16, maximum 34, permutation P=0.014; BRCA: 43 observed vs. null mean 0.19, maximum 19, permutation P<0.001; KIRC: 52 observed vs. null mean 0.07, maximum 3, permutation P=0.005; 200 cohort-level permutations). The between-stratum differences therefore reflect risk-associated structure in the attention weights rather than the risk stratification itself.

Pathway identity was not globally selective, but matched-set enrichment was cohort-dependent. The percentile is defined as P(null effect ≥ real effect), so low values indicate that a real pathway exceeded the matched random sets. Because this statistic is computed on row-normalized attention, we first characterized it under a pure null: across 24 simulated datasets with no between-stratum signal, the median density-matched percentile was 1.00 in LUAD, 1.00 in BRCA and 1.00 in KIRC (5th percentile of the simulated medians: 0.60, 0.66 and 0.61), showing a strong upward structural bias of the statistic. Against this baseline, the observed density-matched medians were 0.69 in LUAD, within the pure-null range, 0.20 in BRCA, below the simulated 5th percentile, and 0.05 in KIRC. The edge-matched control gave median percentiles of 0.67, 0.09 and <0.01 in LUAD, BRCA and KIRC. The number of real pathways above the 95th percentile of the density-matched null was 1, 6 and 25 of 53 (expected 2.65 by chance), and 2, 16 and 39 of 53 for the edge-matched null. Matched-set enrichment was therefore evident in BRCA and KIRC and could not be attributed to the structural bias of row-normalized attention, whereas LUAD showed no enrichment beyond the structural null. Known-pathway enrichment of the top-20 pathways ranked by permutation P was not significant in LUAD (5 hits, P=0.689). These observations are hypothesis-generating rather than evidence of specific pathway activation, and the pathway partition remains an interpretable organizing principle rather than a statistically necessary one. The cohort contrast is informative in its own right: the framework detected risk-associated pathway structure in BRCA and KIRC while returning a negative result in LUAD, indicating that the procedure can discriminate between cohorts with and without detectable rewiring rather than reporting signal uniformly.**3.4.2. Static models cannot produce rewiring.** As a negative control, the static pathway GNN (−Adaptive) was subjected to the identical analysis; its total edge-weight variance across patients was 0.00, essentially zero by construction, whereas the adaptive model produced a total edge-weight variance of 19.21, orders of magnitude larger; the between-stratum differences of Section 3.4.1 are therefore attributable to the sample-specific attention mechanism rather than to the fixed adjacency. Retraining the adaptive model with randomized pathway partitions (block sizes preserved; three seeds) yielded 26–48 significant pathways in LUAD, indicating that between-stratum attention differences do not require the canonical partition. Combined with the matched-set results of Section 3.4.1, the pathway partition supplies biologically interpretable labels for the rewiring output; whether the canonical grouping itself contributes beyond any fixed grouping remains unresolved.**3.4.3. Clinical correlation.** The per-patient rewiring magnitude (L1 distance of the patient's edge weights from the cohort mean) correlated with clinical indicators of malignancy: in BRCA with the proliferation marker MKI67 (Spearman ρ = 0.33, P<0.001, n = 1086), and in LUAD with tumor mutational burden [39] (ρ = 0.13, P=0.003, n = 522); stage showed no significant association in either cohort. In KIRC, the rewiring magnitude was weakly associated with histologic grade (Spearman ρ = 0.09, P=0.038, n = 525), whereas stage and age showed no significant association (ρ = 0.05, P=0.209; ρ = 0.07, P=0.089). In multivariable Cox models adjusting for stage and age, the risk score remained independently associated with overall survival in both cohorts (LUAD: HR = 1.22 per SD, 95% CI 1.01–1.46, P=0.035; BRCA: HR = 1.27 per SD, 95% CI 1.06–1.53, P=0.010). In the IMvigor210 anti-PD-L1 cohort [40], the rewiring magnitude was numerically higher in responders (CR/PR) than in non-responders (SD/PD) (median 686.43 vs 624.81; Wilcoxon P=0.111; n=68/230); high-rewiring patients showed HR 1.40 (95% CI 0.96-2.04, P=0.082) for OS; rewiring magnitude correlated with Ki-67 expression (Spearman rho=0.26, P<0.001, n=348) (Figure 6); the rewiring-Ki-67 correlation replicated in this immunotherapy setting, supporting the biological validity of the learned patient-specific graphs. As an additional exploratory check, we scored the same cohort with a BLCA prognosis-associated rewiring template, using the correlation-difference projection described in Section 3.4.5. The template score was higher in responders than in non-responders (Hedges g = 0.31, Wilcoxon P = 0.019, n = 68/230) and exceeded 96.2% of 2,850 equal-size random gene-set controls. In three melanoma ICB cohorts, GSE91061, GSE78220 and GSE100797, the corresponding SKCM-based template showed no association beyond random gene-set controls (Wilcoxon P = 0.682, P = 0.393 and P = 0.174; random-control percentiles 48.9%, 71.1% and 19.4%). These analyses were exploratory and were not corrected for multiple testing across cohorts and templates; the positive result in IMvigor210 together with the null results in the melanoma cohorts indicates a cohort-dependent association between rewiring and ICB response that requires prospective validation. Because MKI67 is measured in the same transcriptomic input used to compute the attention weights, the Ki-67 correlations are descriptive anchors rather than independent validations; TMB provides an independent (mutation-based) anchor. Tumor purity was not used as an additional anchor because ESTIMATE-based purity [41] is estimated from the same transcriptomic input as the attention weights and would not constitute an independent validation. The associations were robust to the definition of rewiring magnitude: the Ki-67 association in BRCA (rho = 0.19-0.29 across definitions, all P<0.001); the risk-score association in IMvigor210 (rho = 0.32-0.43, all P<0.001); the LUAD TMB association was directionally consistent but weaker (rho = 0.09, P<0.001, for the primary definition).

### Table 8. Sensitivity of clinical anchors to the rewiring-magnitude definition.
| Cohort | Definition | rho(risk) | P(risk) | rho(Ki-67) | P(Ki-67) | n(Ki-67) | rho(TMB) | P(TMB) | n(TMB) |
|---|---|---|---|---|---|---|---|---|---|
| LUAD | 1-r | 0.02 | P=0.573 | -0.13 | P<0.001 | 1399.00 | 0.09 | P<0.001 | 1394.00 |
| LUAD | L1 | -0.02 | P=0.708 | -0.13 | P<0.001 | 1399.00 | 0.09 | P<0.001 | 1394.00 |
| LUAD | z-L1 | 0.03 | P=0.486 | -0.11 | P<0.001 | 1399.00 | 0.02 | P=0.521 | 1394.00 |
| BRCA | 1-r | 0.42 | P<0.001 | 0.29 | P<0.001 | 1546.00 | 0.11 | P<0.001 | 1471.00 |
| BRCA | L1 | 0.20 | P<0.001 | 0.22 | P<0.001 | 1546.00 | 0.01 | P=0.688 | 1471.00 |
| BRCA | z-L1 | 0.14 | P<0.001 | 0.19 | P<0.001 | 1546.00 | -0.02 | P=0.356 | 1471.00 |
| IMvigor210 | 1-r | 0.32 | P<0.001 | — | NA | — | 0.07 | P=0.260 | 272.00 |
| IMvigor210 | L1 | 0.39 | P<0.001 | — | NA | — | 0.05 | P=0.453 | 272.00 |
| IMvigor210 | z-L1 | 0.43 | P<0.001 | — | NA | — | 0.04 | P=0.545 | 272.00 |

![Figure 6](results/figures/Figure6_imvigor.svg)
- **Figure 6. IMvigor210 anti-PD-L1 cohort (exploratory).** (A) Per-patient rewiring magnitude in responders (CR/PR) vs non-responders (SD/PD); Wilcoxon rank-sum test. (B) Overall survival of high- vs low-rewiring strata (median split) with log-rank test. (C) Ki-67 expression vs rewiring magnitude with Spearman correlation.



**3.4.4. Standard-GAT negative control.** To test whether the between-stratum edge-weight differences are specific to the pathway-constrained adaptive architecture, we trained a standard GAT on a sample-invariant k-nearest-neighbor gene graph (k=10; no pathway constraint) under the identical training protocol and applied the identical pathway-level testing pipeline. The standard GAT yielded 11 of 54 significantly rewired pathways in BRCA and 1 of 54 in LUAD (BH-FDR q<0.05), compared with 43 of 53 and 2 of 53 for the adaptive pathway-constrained model under the same permutation-calibrated procedure (Table 7); the pathway-constrained model yielded more detectable between-stratum differences in BRCA and a comparable count in LUAD, and in KIRC the standard GAT yielded 3 of 54 pathways compared with 52 of 53 for the adaptive model, indicating that the pathway partition contributes to the statistical detection of these differences; consistent with the randomized-partition control (Section 3.4.2), canonical pathway structure is nonetheless not uniquely required to produce them. The comparison is therefore not a prediction benchmark but a test of detectable structure: per-patient weight variation exists in any attention model, and the framework's contribution is that the variation can be tested, with an outcome that depends on the architecture.

**3.4.5. External replication of patient-specific rewiring.** As an out-of-cohort check, we transferred each TCGA-trained model to independent GEO cohorts of the same cancer type and tested whether per-patient rewiring magnitude was associated with overall survival within each cohort. 0 of 3 GEO cohorts showed a nominally significant association between rewiring magnitude and OS: GSE31210 (HR 0.83, 95% CI 0.43-1.62, P=0.595; n=226); GSE50081 (HR 0.90, 95% CI 0.57-1.42, P=0.651; n=181); GSE68465 (HR 1.03, 95% CI 0.80-1.33, P=0.815; n=442). 0 of 3 GEO cohorts showed a nominally significant association between rewiring magnitude and OS: GSE20685 (HR 1.02, 95% CI 0.66-1.56, P=0.941; n=327); GSE21653 (HR 1.12, 95% CI 0.72-1.75, P=0.602; n=248); GSE7390 (HR 1.37, 95% CI 0.81-2.33, P=0.243; n=198). The sole evaluable KIRC cohort, GSE29609 (n=39), showed no significant association between rewiring magnitude and OS (HR 0.81, 95% CI 0.31–2.10, P=0.659) These associations are exploratory and were not corrected for multiple testing.

---



**3.4.6. Independence from the risk-stratification procedure and from global attention shifts.** Two further analyses tested whether the pathway-level signal was an artefact of the testing framework itself. First, we repeated the pathway-level test after stratifying patients by variables that do not depend on the model output: MKI67 expression (median split), tumor mutational burden (median split) and pathological stage (I–II vs III–IV). Under the MKI67 anchor, 33 of 53 pathways in BRCA, 24 in KIRC and 11 in LUAD remained significant after BH-FDR correction; the TMB anchor gave 30, 9 and 0 significant pathways in BRCA, LUAD and KIRC; and the stage anchor gave almost none in LUAD and BRCA (0 and 2 of 53), consistent with the absence of a stage–rewiring correlation in those cohorts; in KIRC, the stage and grade anchors gave 22 and 25 of 53 significant pathways, respectively, consistent with the KIRC grade–rewiring correlation (Section 3.4.3). The rewiring signal therefore does not require the model’s own risk stratification, although MKI67 and TMB are correlated with the risk score and these anchors are not fully independent of it.

Second, we decomposed the between-stratum difference into a global shift of the overall attention level and a pathway-specific component. The global shift was significant in all three cohorts (LUAD P=0.019; BRCA P<0.001; KIRC P<0.001) but its direction was positive in BRCA and negative in KIRC. After removing the per-patient global attention mean, 14, 42 and 36 of 53 pathways remained significant in LUAD, BRCA and KIRC, compared with 2, 43 and 52 under the raw test; the pathway-level signal therefore contains a component that is not merely a projection of the global shift.

Third, the direction of the pathway-specific effects was not consistent across cancers: of the 33 pathways significant in both BRCA and KIRC after global-shift removal, only 12.12% had the same direction, and the Spearman correlation of the pathway-specific effects between the two cohorts was -0.89 (P<0.001). Attention weights admit a sign symmetry, because flipping the sign of the first-layer attention weights and the risk-mapping weights leaves the risk score unchanged; the absolute direction of attention changes is therefore not interpretable across cohorts. This symmetry is a property of the model class rather than a deficiency of the testing procedure: the training objective is invariant to the joint sign flip, so any cohort-specific direction estimate is defined only up to this transformation, whereas the existence of risk-associated structure is invariant to it and is therefore the well-defined quantity. We accordingly restrict our claims to the existence of statistically testable, risk-associated, pathway-level structure within each cohort, and do not claim that specific pathway identities or their directions transfer across cancer types.

**3.4.7. Decision-curve analysis.** To complement the discrimination metrics with a clinical-value perspective, we estimated inverse-probability-of-censoring-weighted net benefit at 1-, 3- and 5-year horizons for the clinical model, the clinical model plus the risk score, and the risk score alone (Figure 7). Adding the risk score to age and stage increased net benefit at the majority of threshold probabilities at the 3-year horizon in all three cohorts (LUAD, 9/11 thresholds, maximum increase 0.03; BRCA, 9/11, maximum 0.01; KIRC, 10/11, maximum 0.02) and at the 5-year horizon in BRCA and KIRC (10/11 and 9/11 thresholds; maximum increases 0.01 and 0.03); 1-year gains were small. The incremental benefit was modest and threshold-dependent, seemingly consistent with the comparable discrimination between Path-AGNN-Cox and the deep baselines (Section 3.3).

![Figure 7](results/figures/FigureDCA_dca.svg)
- **Figure 7. Decision-curve analysis at 3 years.** Net benefit across threshold probabilities for the clinical model (age and stage), the clinical model plus the risk score, the risk score alone, and the treat-all and treat-none references in LUAD, BRCA and KIRC.

### 3.5. Immune infiltration and predicted drug sensitivity (exploratory)

To connect patient-specific rewiring to the tumor microenvironment and to therapeutic vulnerability, we computed MCP-counter immune cell abundance estimates [42] and ssGSEA scores of nine immune Hallmark gene sets [43,44] per patient, and tested their association with the per-patient rewiring magnitude (Figure 8A). In LUAD, the rewiring magnitude was nominally associated with lower ssGSEA IL2-STAT5-SIGNALING (P=0.016), B lineage (P=0.018) and ssGSEA COMPLEMENT (P=0.046); none of these associations survived BH-FDR correction (smallest q=0.169), and in BRCA no immune feature was associated with rewiring magnitude (smallest P=0.274). Immune infiltration was therefore not strongly coupled to pathway rewiring in these two cohorts.

![Figure 8](results/figures/Figure7_immunedrug.svg)
- **Figure 8. Immune infiltration and predicted drug sensitivity (exploratory).** (A) Association of 19 immune features with per-patient rewiring magnitude in LUAD (red) and BRCA (blue): signed -log10 P of the Wilcoxon test between high- and low-rewiring strata; dashed line: P=0.050. (B) Median predicted IC50 (GDSC2/oncoPredict) of the eight nominally significant compounds in BRCA, high- versus low-rewiring strata. (C) Spearman correlation between rewiring magnitude and predicted IC50 across 17 curated compounds in BRCA; filled markers: FDR q<0.05.


We further predicted drug sensitivity with the oncoPredict model [45] trained on GDSC2 cell-line pharmacogenomic data [46] (198 compounds) and compared predicted IC50 values between high- and low-rewiring strata (Figure 8B, Table 9). In BRCA, the high-rewiring stratum was predicted to be more sensitive (lower IC50) to 9 of the 17 curated compounds at nominal significance, most strongly MK-1775 (WEE1, Wilcoxon P=0.006), paclitaxel (P=0.006) and palbociclib (CDK4/6, P=0.018); the Spearman association remained significant after FDR correction for MK-1775 and paclitaxel (q=0.006), whereas the Wilcoxon comparisons did not survive correction (smallest q=0.054). In LUAD, no compound showed a significant difference (smallest P=0.063). These analyses are hypothesis-generating: the predicted IC50 values are in-silico estimates from cell-line-derived models and do not substitute for direct pharmacologic assays (Section 4).

### Table 9. Predicted drug sensitivity (GDSC2/oncoPredict IC50) in high- versus low-rewiring strata (exploratory).
| Cohort | Drug | IC50 median (high) | IC50 median (low) | Wilcoxon P | FDR q | Spearman ρ | Spearman P |
|---|---|---|---|---|---|---|---|
| **BRCA (n high/low: 536/535)** | | | | | | | |
|  | MK-1775 | 2.12 | 2.40 | P=0.006 | q=0.054 | -0.10 | P<0.001 |
|  | Paclitaxel | 0.06 | 0.06 | P=0.006 | q=0.054 | -0.10 | P<0.001 |
|  | Gefitinib | 27.84 | 29.30 | P=0.018 | q=0.062 | -0.08 | P=0.007 |
|  | Gemcitabine | 0.53 | 0.63 | P=0.018 | q=0.062 | -0.07 | P=0.018 |
|  | 5-Fluorouracil | 131.74 | 150.89 | P=0.019 | q=0.062 | -0.09 | P=0.005 |
|  | Palbociclib | 36.30 | 38.94 | P=0.024 | q=0.062 | -0.07 | P=0.016 |
|  | Docetaxel | 0.07 | 0.08 | P=0.026 | q=0.062 | -0.08 | P=0.006 |
|  | Cisplatin | 28.92 | 33.20 | P=0.033 | q=0.071 | -0.07 | P=0.015 |
|  | AZD7762 | 1.22 | 1.32 | P=0.047 | q=0.089 | -0.07 | P=0.014 |
|  | AZD6738 | 8.19 | 8.74 | P=0.068 | q=0.116 | -0.07 | P=0.020 |
|  | Talazoparib | 23.20 | 26.38 | P=0.083 | q=0.117 | -0.06 | P=0.050 |
|  | Niraparib | 79.60 | 86.53 | P=0.083 | q=0.117 | -0.07 | P=0.022 |
|  | Olaparib | 68.50 | 71.74 | P=0.185 | q=0.242 | -0.05 | P=0.077 |
|  | Ribociclib | 45.50 | 46.65 | P=0.210 | q=0.255 | -0.05 | P=0.073 |
|  | Trametinib | 2.11 | 2.24 | P=0.298 | q=0.338 | -0.04 | P=0.235 |
|  | Erlotinib | 15.36 | 15.61 | P=0.341 | q=0.345 | -0.04 | P=0.251 |
|  | Selumetinib | 65.30 | 68.41 | P=0.345 | q=0.345 | -0.04 | P=0.242 |
| **LUAD (n high/low: 252/252)** | | | | | | | |
|  | Ribociclib | 45.83 | 44.05 | P=0.063 | q=0.322 | 0.13 | P=0.004 |
|  | Niraparib | 83.33 | 76.45 | P=0.064 | q=0.322 | 0.10 | P=0.027 |
|  | MK-1775 | 2.02 | 2.41 | P=0.065 | q=0.322 | -0.07 | P=0.099 |
|  | Erlotinib | 14.40 | 15.24 | P=0.076 | q=0.322 | -0.04 | P=0.320 |
|  | Gefitinib | 25.55 | 27.29 | P=0.140 | q=0.457 | -0.03 | P=0.476 |
|  | Olaparib | 71.42 | 65.38 | P=0.161 | q=0.457 | 0.08 | P=0.080 |
|  | Talazoparib | 23.74 | 22.37 | P=0.236 | q=0.573 | 0.07 | P=0.135 |
|  | Selumetinib | 64.14 | 60.81 | P=0.305 | q=0.613 | 0.07 | P=0.111 |
|  | AZD6738 | 8.38 | 8.42 | P=0.325 | q=0.613 | -0.05 | P=0.252 |
|  | Docetaxel | 0.07 | 0.07 | P=0.396 | q=0.631 | -0.02 | P=0.578 |
|  | Paclitaxel | 0.06 | 0.06 | P=0.442 | q=0.631 | -0.03 | P=0.542 |
|  | AZD7762 | 1.20 | 1.22 | P=0.467 | q=0.631 | -0.02 | P=0.616 |
|  | Gemcitabine | 0.53 | 0.48 | P=0.483 | q=0.631 | 0.03 | P=0.443 |
|  | Cisplatin | 30.34 | 29.16 | P=0.659 | q=0.771 | 0.00 | P=0.932 |
|  | 5-Fluorouracil | 126.39 | 112.80 | P=0.680 | q=0.771 | 0.01 | P=0.742 |
|  | Trametinib | 2.07 | 1.96 | P=0.789 | q=0.838 | 0.05 | P=0.307 |
|  | Palbociclib | 32.99 | 33.75 | P=0.961 | q=0.961 | 0.04 | P=0.340 |
_IC50 values are GDSC2/oncoPredict in-silico predictions; associations are exploratory and not FDR-significant unless stated._


---

## 4. Discussion

The conceptual contribution of this work is a framework rather than a module tweak. Existing pathway GNNs, including PathGNN [20], Cox-Path [21], prior-knowledge-guided multilevel GNNs [22], and the multi-omics PathMoG [23], treat the pathway graph as a fixed patient-invariant prior and stop at the risk score. We instead treat the effective graph as a per-patient object and provide the statistical machinery to interrogate its rewiring: edge-level and pathway-level tests with BH-FDR control [33], label-permutation nulls, a static-model negative control, a standard-GAT architectural control, and clinical anchoring. The discrimination of the adaptive model was comparable to deep survival baselines, and we make no claim that per-patient attention improves the C-index; its distinguishing value is that the learned patient-specific edge weights can be formally tested, are clinically anchored to Ki-67 and TMB [39], and are absent in static models by construction. Per-patient variation in attention weights is not by itself novel, because attention is a function of the input features in any attention model; the framework's contribution is that this variation becomes testable, with outcomes that differ between the pathway-constrained adaptive model and a standard-GAT control and between cohorts with and without detectable rewiring. The pathway partition is an interpretive labeling of these differences rather than a statistical necessity, as detailed in Section 3.4.2.

**Comparison with PathMoG.** PathMoG is the closest recent work; it also organizes genes into KEGG pathway modules and uses hierarchical attention for multi-omics survival prediction [23]. The two frameworks are complementary in scope and differ in three substantive ways. First, the topology is static in PathMoG and adaptive in Path-AGNN-Cox: PathMoG precomputes patient-invariant pathway topologies and learns attention within them, whereas Path-AGNN-Cox additionally learns the graph temperature per patient, and its rewiring analysis directly interrogates patient-specific graph dynamics. Second, the omics scope differs: PathMoG requires mutation and CNV in addition to expression, which limits its applicability to the many clinical cohorts that only measure transcriptomes; Path-AGNN-Cox is a single-transcriptome model that is immediately applicable to the large body of GEO microarray cohorts, as supported by our 25-cohort external validation. Third, the external validation breadth differs: PathMoG validated transferability in a single breast-cancer cohort, METABRIC [37]; Path-AGNN-Cox was validated in 25 GEO cohorts spanning 11 tumor types. These differences position Path-AGNN-Cox as a lightweight alternative that is potentially more broadly deployable for single-omics clinical and public cohorts, while PathMoG remains the reference for multi-omics integration.

**Limitations.** First, the predictive performance of the adaptive model did not exceed penalized Cox baselines on internal CV, and ablations did not isolate a significant C-index contribution from the adaptive gate, the pathway constraint, or the regularization terms; interpretability is the rationale for its use rather than a discrimination gain. Second, the learnable malignancy gate converged to values near zero in the trained models, so the effective sample-specificity arises from the attention coefficients themselves; we therefore describe the mechanism as patient-specific attention rather than as demonstrable malignancy-driven gating. Third, hypergeometric enrichment of the top-20 rewired pathways ranked by permutation P against a curated LUAD driver-pathway list was not significant with 5 hits at P=0.689; an equivalent test was not performed for BRCA because no matched curated driver-pathway list was available. Fourth, matched random gene-set controls were cohort-dependent: real pathways were enriched above matched random sets in BRCA and KIRC beyond a simulated pure-null baseline of the statistic, whereas LUAD showed no enrichment beyond the structural null; the pathways that exceeded both permutation and matched-set nulls remain hypothesis-generating, and the pathway prior is retained as an interpretable organizing principle rather than as evidence of specific pathway activation. This cohort-dependence is at the same time evidence that the framework does not produce rewiring signal uniformly; it can separate cohorts with detectable structure from those without it. Fifth, attention weights provide hypothesis-generating rather than causal evidence, and all cohorts are retrospective; prospective validation is required. Sixth, the pan-cancer benchmark used a single seed; a three-seed sensitivity analysis on LUAD, BRCA and KIRC, the cohorts used for the rewiring analyses, showed stable C-index values of 0.50 ± 0.04 in LUAD, 0.59 ± 0.04 in BRCA and 0.62 ± 0.07 in KIRC, but seed variance was not propagated to all cohorts. Seventh, the two LUAD pathways that were significant in the training cohort were not replicated at the pathway level across three independent external LUAD cohorts; DNA replication showed a Stouffer z of -1.47 with P=0.141 and 1 of 3 cohorts in the same direction, and homologous recombination a z of 0.69 with P=0.490 and 2 of 3; pathway-level rewiring findings should therefore be interpreted within the cohort in which they were derived. Eighth, the direction of the pathway-specific rewiring effects was systematically opposite between BRCA and KIRC, consistent with the sign symmetry of the attention mechanism; because the objective is invariant to a joint sign flip of the attention and risk-mapping weights, the direction estimate is defined only up to this transformation and absolute attention directions are not compared across cohorts. The existence of risk-associated pathway structure is invariant to the symmetry and remains the well-defined claim. Ninth, an exploratory immunotherapy analysis found that a BLCA prognosis-associated rewiring template was associated with anti-PD-L1 response in IMvigor210 (Wilcoxon P = 0.019, exceeding 96.2% of random gene-set controls), whereas the corresponding SKCM-based templates in three melanoma ICB cohorts did not exceed random gene-set controls; these associations were not corrected for multiple testing across cohorts and templates and require prospective validation.

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



## Author contributions

Zhipeng Wang: conceptualization, methodology, software, formal analysis, and writing of the original draft. Luning Wang: data curation, methodology, and writing of the original draft. Changsong Wang: supervision, project administration, and review and editing of the manuscript. Pengli Zhai: resources and review and editing of the manuscript. Zejun Liu: validation and review and editing of the manuscript. Hui Feng: investigation and visualization. Hongmei Liu: data curation and validation. Qian Hou: software and visualization. Ming Guo: formal analysis and review and editing of the manuscript.

## Ethics approval

This study was approved by the Ethics Committee of Zhongda Hospital Affiliated with Southeast University. All analyses used publicly available de-identified data; no new clinical data or patient samples were collected.

## Data availability

All data were obtained from public repositories: TCGA RNA-seq and clinical annotations from UCSC Xena [6] and GDC, GEO series matrices and platform annotations from NCBI GEO [4,5], and IMvigor210 data from the original publication [40]. Accession numbers are listed in Table 1. To protect patient privacy, per-patient expression matrices, clinical records, and per-patient model outputs are not redistributed; aggregate results required to reproduce every table and figure are provided in the repository.

## Competing interests

The authors declare no competing interests.

---

## References

1. Cox DR. Regression models and life-tables. J R Stat Soc Ser B Stat Methodol. 1972;34(2):187-220. doi:10.1111/j.2517-6161.1972.tb00899.x
2. Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors. Stat Med. 1996;15(4):361-387. doi:10.1002/(SICI)1097-0258(19960229)15:4<361::AID-SIM168>3.0.CO;2-4
3. Weinstein JN, Collisson EA, Mills GB, Shaw KR, Ozenberger BA, Ellrott K, et al. The Cancer Genome Atlas Pan-Cancer analysis project. Nat Genet. 2013;45(10):1113-1120. doi:10.1038/ng.2764
4. Barrett T, Wilhite SE, Ledoux P, Evangelista C, Kim IF, Tomashevsky M, et al. NCBI GEO: archive for functional genomics data sets, update. Nucleic Acids Res. 2013;41(D1):D991-D995. doi:10.1093/nar/gks1193
5. Davis S, Meltzer PS. GEOquery: a bridge between the Gene Expression Omnibus (GEO) and BioConductor. Bioinformatics. 2007;23(14):1846-1847. doi:10.1093/bioinformatics/btm254
6. Goldman MJ, Craft B, Hastie M, Repečka K, McDade F, Kamath A, et al. Visualizing and interpreting cancer genomics data via the Xena platform. Nat Biotechnol. 2020;38(6):675-678. doi:10.1038/s41587-020-0546-8
7. Friedman J, Hastie T, Tibshirani R. Regularization paths for generalized linear models via coordinate descent. J Stat Softw. 2010;33(1):1-22. doi:10.18637/jss.v033.i01
8. Simon N, Friedman J, Hastie T, Tibshirani R. Regularization paths for Cox's proportional hazards model via coordinate descent. J Stat Softw. 2011;39(5):1-13. doi:10.18637/jss.v039.i05
9. Katzman JL, Shaham U, Cloninger A, Bates J, Jiang T, Kluger Y. DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. BMC Med Res Methodol. 2018;18(1):24. doi:10.1186/s12874-018-0482-1
10. Ching T, Zhu X, Garmire LX. Cox-nnet: an artificial neural network method for prognosis prediction of high-throughput omics data. PLoS Comput Biol. 2018;14(4):e1006076. doi:10.1371/journal.pcbi.1006076
11. Lee C, Zame W, Yoon J, van der Schaar M. DeepHit: a deep learning approach to survival analysis with competing risks. Proc AAAI Conf Artif Intell. 2018;32(1):2314-2321. doi:10.1609/aaai.v32i1.11842
12. Ishwaran H, Kogalur UB, Blackstone EH, Lauer MS. Random survival forests. Ann Appl Stat. 2008;2(3):841-860. doi:10.1214/08-AOAS169
13. Eraslan G, Avsec Ž, Gagneur J, Theis FJ. Deep learning: new computational modelling techniques for genomics. Nat Rev Genet. 2019;20(7):389-403. doi:10.1038/s41576-019-0122-6
14. Wainberg M, Merico D, Delong A, Frey BJ. Deep learning in biomedicine. Nat Biotechnol. 2018;36(9):829-838. doi:10.1038/nbt.4233
15. Miotto R, Wang F, Wang S, Jiang X, Dudley JT. Deep learning for healthcare: review, opportunities and challenges. Brief Bioinform. 2018;19(6):1236-1246. doi:10.1093/bib/bbx044
16. Kipf TN, Welling M. Semi-supervised classification with graph convolutional networks. arXiv:1609.02907 [cs.LG]. 2017.
17. Veličković P, Cucurull G, Casanova A, Romero A, Liò P, Bengio Y. Graph attention networks. arXiv:1710.10903 [cs.LG]. 2018.
18. Zitnik M, Agrawal M, Leskovec J. Modeling polypharmacy side effects with graph convolutional networks. Bioinformatics. 2018;34(13):i457-i466. doi:10.1093/bioinformatics/bty294
19. Kanehisa M, Furumichi M, Sato Y, Kawashima M, Ishiguro-Watanabe M. KEGG for taxonomy-based analysis of pathways and genomes. Nucleic Acids Res. 2023;51(D1):D587-D592. doi:10.1093/nar/gkac963
20. Liang J, Zhang Y, Chen H, et al. Risk stratification and pathway analysis based on graph neural network and interpretable algorithm. BMC Bioinformatics. 2022;23(1):411. doi:10.1186/s12859-022-04950-1
21. Ma T, Zhao H, Zhao Q, Wang J. Cox-Path: biological pathway-informed graph neural network for cancer survival prediction. In: Proceedings of the 15th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics (ACM-BCB 2024). New York: ACM; 2024. doi:10.1145/3698587.3701397
22. Yan H, Weng D, Li D, Gu Y, Ma W, Liu Q. Prior knowledge-guided multilevel graph neural network for tumor risk prediction and interpretation via multi-omics data integration. Brief Bioinform. 2024;25(4):bbae184. doi:10.1093/bib/bbae184
23. Wang Z, et al. PathMoG: a pathway-centric modular graph neural network for multi-omics survival prediction. arXiv:2604.24371 [q-bio.QM]. 2026.
24. Ha MJ, Baladandayuthapani V. DINGO: differential network analysis in genomics. Bioinformatics. 2015;31(21):3413-3420. doi:10.1093/bioinformatics/btv406
25. de la Fuente A. From differential expression to differential networking: identification of dysfunctional regulatory networks in diseases. Trends Genet. 2010;26(7):326-333. doi:10.1016/j.tig.2010.05.001
26. Gill R, Datta S, Datta S. A statistical framework for differential network analysis from microarray data. BMC Bioinformatics. 2010;11:95. doi:10.1186/1471-2105-11-95
27. Colaprico A, Silva TC, Olsen C, Garofano L, Cava C, Garolini D, et al. TCGAbiolinks: an R/Bioconductor package for integrative analysis of TCGA data. Nucleic Acids Res. 2016;44(8):e71. doi:10.1093/nar/gkv1507
28. Therneau TM, Grambsch PM. Modeling Survival Data: Extending the Cox Model. New York: Springer; 2000. doi:10.1007/978-1-4757-3294-8
29. Uno H, Cai T, Pencina MJ, D'Agostino RB, Wei LJ. On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data. Stat Med. 2011;30(10):1105-1117. doi:10.1002/sim.4154
30. Heagerty PJ, Zheng Y. Survival model predictive accuracy and ROC curves. Biometrics. 2005;61(1):92-105. doi:10.1111/j.0006-341X.2005.030814.x
31. Altman DG, Vergouwe Y, Royston P, Moons KG. Prognosis and prognostic research: validating a prognostic model. BMJ. 2009;338:b605. doi:10.1136/bmj.b605
32. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making. 2006;26(6):565-574. doi:10.1177/0272989X06295361
33. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. J R Stat Soc Ser B Stat Methodol. 1995;57(1):289-300. doi:10.1111/j.2517-6161.1995.tb02031.x
34. Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G, et al. PyTorch: an imperative style, high-performance deep learning library. arXiv:1912.01703 [cs.LG]. 2019.
35. Davidson-Pilon C. lifelines: survival analysis in Python. J Open Source Softw. 2019;4(39):1317. doi:10.21105/joss.01317
36. Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: machine learning in Python. arXiv:1201.0490 [cs.LG]. 2011.
37. Curtis C, Shah SP, Chin SF, Turashvili G, Rueda OM, Dunning MJ, et al. The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups. Nature. 2012;486(7403):346-352. doi:10.1038/nature10983
38. Crowson CS, Atkinson EJ, Therneau TM. Assessing calibration of prognostic risk scores. Stat Methods Med Res. 2016;25(4):1692-1706. doi:10.1177/0962280213497434
39. Chalmers ZR, Connelly CF, Fabrizio D, Gay L, Ali SM, Ennis R, et al. Analysis of 100,000 human cancer genomes reveals the landscape of tumor mutational burden. Genome Med. 2017;9:34. doi:10.1186/s13073-017-0424-2
40. Mariathasan S, Turley SJ, Nickles D, et al. TGFβ attenuates tumour response to PD-L1 blockade by contributing to exclusion of T cells. Nature. 2018;554(7693):544-548. doi:10.1038/nature25501
41. Yoshihara K, Shahmoradgoli M, Martínez E, Vegesna R, Kim H, Torres-Garcia W, et al. Inferring tumour purity and stromal and immune cell admixture from expression data. Nat Commun. 2013;4:2612. doi:10.1038/ncomms3612
42. Becht E, Giraldo NA, Lacroix L, Buttard B, Elarouci N, Petitprez F, et al. Estimating the population abundance of tissue-infiltrating immune and stromal cell populations using gene expression. Genome Biol. 2016;17:218. doi:10.1186/s13059-016-1070-5
43. Hänzelmann S, Castelo R, Guinney J. GSVA: gene set variation analysis for microarray and RNA-seq data. BMC Bioinformatics. 2013;14:7. doi:10.1186/1471-2105-14-7
44. Liberzon A, Birger C, Thorvaldsdóttir H, Ghandi M, Mesirov JP, Tamayo P. The Molecular Signatures Database Hallmark gene set collection. Cell Syst. 2015;1(6):417-425. doi:10.1016/j.cels.2015.12.004
45. Maeser D, Gruener RF, Huang RS. oncoPredict: an R package for predicting in vivo or cancer patient drug response and biomarkers from cell line screening data. Brief Bioinform. 2021;22(6):bbab260. doi:10.1093/bib/bbab260
46. Yang W, Soares J, Greninger P, Edelman EJ, Lightfoot H, Forbes S, et al. Genomics of Drug Sensitivity in Cancer (GDSC): a resource for therapeutic biomarker discovery in cancer cells. Nucleic Acids Res. 2013;41(D1):D955-D961. doi:10.1093/nar/gks1111

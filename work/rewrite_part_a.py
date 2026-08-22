# -*- coding: utf-8 -*-
"""Rewrite manuscript template: new front matter, abstract, intro, discussion, end sections, references.
Part A: section strings."""
import re, json, io

TEMPLATE = r"manuscript/Path-AGNN-Cox_manuscript_template.md"
REFJSON = r"work/refs_final.json"

FRONT = """## Title

**Path-AGNN-Cox: a reproducible statistical framework for testing patient-specific pathway rewiring in cancer survival analysis**

## Authors

Zhipeng Wang[1,*], Luning Wang[2,*], Changsong Wang[1,\u2020], Pengli Zhai[3], Zejun Liu[1], Hui Feng[1], Hongmei Liu[1], Qian Hou[1], Ming Guo[1]

1 Department of TCM, Zhongda Hospital, Southeast University, China
2 Department of Rehabilitation Medicine, Zhongda Hospital, Southeast University, China
3 Jiangbei Campus, Jiangsu Provincial Traditional Chinese Medicine Hospital, China

* These authors contributed equally to this work and are co-first authors.
\u2020 Corresponding author. Email: 101005664@seu.edu.cn

---

## Abstract

**Background:** Cancer prognosis models trained on high-dimensional transcriptomic data face two interconnected problems. Unconstrained deep survival models act as black boxes that ignore biological structure, and pathway-constrained graph neural networks assume a patient-invariant interaction topology that is difficult to reconcile with the malignancy-dependent rewiring of tumor regulatory networks.

**Methods:** We propose Path-AGNN-Cox, a pathway-constrained adaptive graph neural network for survival prediction. Genes are partitioned into KEGG cancer-core pathway modules; sample-specific within-pathway attention weights are computed with a learnable malignancy-modulated gate; and a Cox partial-likelihood objective with dual regularization is optimized directly on prognostic risk. We benchmarked Path-AGNN-Cox against seven survival baselines across {{N_DATASETS}} TCGA cancer types with stratified 5-fold cross-validation and validated transferability on {{N_EXTERNAL}} independent GEO cohorts.

**Results:** On internal cross-validation, Path-AGNN-Cox reached a mean C-index of {{CV_FULL_MEAN}} (SD {{CV_FULL_SD}}), comparable to deep survival baselines but below penalized Cox; we therefore make no claim of a discrimination gain. Its distinguishing value is the interpretable rewiring output: risk-associated pathway weights exceeded label-permutation nulls in BRCA ({{PERM_BRCA_SIG}}/{{PERM_N_PATHWAYS}} pathways), LUAD ({{PERM_LUAD_SIG}}/{{PERM_N_PATHWAYS}}) and KIRC ({{PERM_KIRC_SIG}}/{{PERM_N_PATHWAYS}}), correlated with clinical indicators of malignancy, and were absent by construction in static pathway models.

**Conclusion:** Path-AGNN-Cox provides a reproducible framework in which patient-specific pathway graphs become objects of formal statistical testing, without requiring a discrimination gain over classical baselines. The model and pipelines are released as an open-source Python package with an archived snapshot.

**Key words:** survival analysis; graph neural network; pathway constraint; adaptive graph learning; cancer prognosis

---
"""

INTRO = """## 1. Introduction

Precision oncology increasingly depends on molecular risk stratification to guide adjuvant therapy, surveillance intensity, and clinical-trial design [cox1972,harrell1996]. Large public transcriptomic compendia, most notably The Cancer Genome Atlas [tcga2013] and the Gene Expression Omnibus [geo2013,geoquery2007], have made it possible to train and validate prognostic models across many cancer types [xena2020]. Because survival endpoints are censored and expression data are high-dimensional and biologically heterogeneous, this task remains an active frontier of translational bioinformatics.

Classical statistical models, such as Cox proportional-hazards regression [cox1972] with Lasso, Ridge, or Elastic-Net penalties [glmnet2010,glmnetcox2011], treat genes as independent covariates and discard the regulatory interaction structure that drives tumor progression. Deep survival models, including DeepSurv [deepsurv2018], Cox-nnet [coxnnet2018], DeepHit [deephit2018], and random survival forests [rsf2008], improve discrimination by learning nonlinear feature interactions. They nevertheless operate on gene lists rather than biological graphs, offer limited interpretability, and are prone to severe performance decay when transferred from a single training cohort to independent external cohorts [eraslan2019,wainberg2018,miotto2018].

Graph neural networks provide a natural inductive bias for molecular data [gcn2017,gat2018,zitnik2018]. A growing family of pathway-constrained graph neural networks restricts message passing to gene sets defined by KEGG pathways [kegg2023]. PathGNN showed that pathway-topology-constrained propagation improves prognosis across several solid tumors [pathgnn2022]. Cox-Path partitioned genes into KEGG pathway subgraphs and coupled them to a Cox survival head [coxpath2024]. A prior-knowledge-guided multilevel GNN introduced gene-to-pathway hierarchical propagation for survival prediction [priorgnn2024]. PathMoG most recently extended this concept to multi-omics with 354 KEGG-informed pathway modules and dual-level attention, reporting strong performance across 10 TCGA cohorts [pathmog2026]. These studies collectively suggest that biological priors can reduce overfitting and improve external validity relative to unconstrained deep models.

A largely unexamined assumption nevertheless underlies these models: the gene interaction topology is invariant across patients. In every pathway-constrained survival GNN of which we are aware, including PathMoG, the adjacency, edge weights, and inter-pathway coupling are fixed a priori and shared by all patients [pathgnn2022,coxpath2024,priorgnn2024,pathmog2026]. This assumption is difficult to reconcile with the observation that tumor regulatory networks are extensively rewired in a malignancy-dependent manner [dingo2015,delfuente2010,gill2010]. Differential network analyses have, moreover, shown that interaction strengths vary across disease states and individuals rather than forming a single static structure [dingo2015,delfuente2010,gill2010].

The static-graph assumption consequently gives rise to three deficiencies. First, a fixed adjacency cannot represent patient-specific rewiring; two tumors with the same pathway membership but different driver states share an identical interaction topology. Second, within-pathway aggregation is weighted in a sample-invariant way, so background genes can dilute the signal of a few driver genes, and the model cannot automatically up-weight the interactions that matter for an aggressive tumor. Third, no mechanism exists to express how within-pathway interaction strengths tighten or loosen with disease aggressiveness. These deficiencies plausibly explain why static pathway models still decay on independent external cohorts.

To test this hypothesis, we introduce Path-AGNN-Cox, a pathway-constrained adaptive graph neural network for survival prediction. The model comprises three modules: pathway-constrained subgraph construction that partitions genes into KEGG cancer-core pathway modules; sample-adaptive neighborhood weighting in which attention logits are multiplicatively modulated by a learnable malignancy gate; and a Cox partial-likelihood objective with dual regularization. We benchmarked Path-AGNN-Cox against classical survival models, deep survival models, and static pathway-constrained GNNs across {{N_DATASETS}} TCGA cancer types with {{N_EXTERNAL}} independent GEO validation cohorts, and isolated the contribution of each module through systematic ablations. Beyond predictive performance, we assess the biological content of the learned sample-specific edge weights: between-stratum rewiring is compared with label-permutation nulls, matched random gene-set controls, and clinical indicators of malignancy, whereas static models by construction cannot produce such signal. The model is released as an open-source Python package with reproducible pipelines.

---
"""

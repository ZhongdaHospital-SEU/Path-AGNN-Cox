# Cover letter (draft)

**To the Editors, Briefings in Bioinformatics**

Dear Editors,

We are pleased to submit our manuscript, **"Path-AGNN-Cox: a reproducible statistical framework for testing patient-specific pathway rewiring in cancer survival analysis,"** for consideration as a research article in *Computational and Structural Biotechnology Journal*.

**What the manuscript contributes.** Transcriptomic survival models based on graph neural networks (GNNs) typically treat the underlying gene interaction topology as fixed across patients. A growing line of work organizes genes into KEGG pathway modules before message passing, but the effective graph remains patient-invariant. Our manuscript provides three things:

1. **A model that produces patient-specific pathway graphs.** Path-AGNN-Cox constrains message passing to KEGG pathway subgraphs and learns per-patient attention edge weights, with a Cox partial-likelihood head and dual regularization. We do not claim that this improves discrimination: benchmarked on 11 TCGA cancer types (5-fold cross-validation) with 25 independent GEO external cohorts, its C-index is comparable to, but not better than, penalized Cox baselines.

2. **A statistical testing framework for pathway rewiring.** The paper formalizes edge- and pathway-level tests of between-stratum rewiring with BH-FDR control, label-permutation nulls, a static-model negative control, and a standard-GAT architectural control. To our knowledge, this is the first reproducible framework in which sample-specific pathway rewiring is defined operationally and tested formally, rather than reported qualitatively from attention maps.

3. **Clinical and external anchoring.** Rewiring magnitude correlates with Ki-67 expression in two independent cohorts (TCGA-BRCA and the IMvigor210 anti-PD-L1 cohort) and with tumor mutational burden in TCGA-LUAD; multivariable Cox models confirm independent association of the risk score with overall survival. We report the negative results transparently: in six external GEO cohorts, rewiring magnitude was not significantly associated with overall survival, and in three melanoma ICB cohorts a prognosis-associated rewiring template did not exceed random gene-set controls. As an exploratory immunotherapy finding, the same template was associated with anti-PD-L1 response in IMvigor210 (Wilcoxon P = 0.019, exceeding 96.2% of random gene-set controls), and this association was concentrated in the low-TMB stratum (P = 0.004; response rate 18.4% vs 6.9%, Fisher exact P = 0.032). The direction was consistent in a second urothelial ICB cohort, GSE176307, with a combined BLCA-template effect of g = 0.28 (95% CI 0.04-0.53, P = 0.022, I2 = 0%).

**Reproducibility.** All code is released as an open-source Python package with a pip-installable distribution, example notebooks, and reproducible pipelines for data download, benchmarking, rewiring testing, and figure generation, available at https://github.com/ZhongdaHospital-SEU/Path-AGNN-Cox and on PyPI (https://pypi.org/project/path-agnn-cox/), with an archived snapshot at Zenodo (https://doi.org/10.5281/zenodo.22030045). We believe this satisfies the journal's emphasis on transparent, reproducible methods.

All authors have read and approved the manuscript, and none of the authors has a competing financial interest. The manuscript is not under consideration elsewhere.

Thank you for your consideration.

Sincerely,

Zhipeng Wang, Luning Wang, Changsong Wang, Pengli Zhai, Zejun Liu, Hui Feng, Hongmei Liu, Qian Hou, Ming Guo

Corresponding author: Changsong Wang, Department of TCM, Zhongda Hospital, Southeast University, China. Email: 101005664@seu.edu.cn

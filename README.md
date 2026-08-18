# Path-AGNN-Cox

**A reproducible statistical framework for testing patient-specific pathway rewiring in cancer survival analysis**

Repository for the manuscript *"Path-AGNN-Cox: a reproducible statistical
framework for testing patient-specific pathway rewiring in cancer survival
analysis"* (manuscript in `manuscript/`). The benchmark covers 11 TCGA cancer
types (5-fold CV) with 25 independent GEO external validation cohorts,
comparing Path-AGNN-Cox against classical, deep, and graph survival baselines.

## What this framework provides

- **Patient-specific pathway graphs**: per-patient attention edge weights over
  KEGG pathway-constrained subgraphs, with a Cox risk head.
- **Statistical testing machinery for rewiring**: edge- and pathway-level
  between-stratum tests with BH-FDR, label-permutation nulls, a static-model
  negative control (zero rewiring by construction), and a standard-GAT
  architectural control.
- **Clinical anchoring**: rewiring magnitude vs. proliferation (Ki-67) and TMB;
  multivariable Cox models; independent immunotherapy cohort (IMvigor210) and
  external GEO replication analyses.
- **Honest benchmarking**: performance is reported as comparable to penalized
  Cox baselines; the framework makes no claim of a discrimination gain. Its
  distinguishing value is interpretable, testable, patient-specific rewiring.

## Method in one paragraph

Path-AGNN-Cox builds a pathway-constrained graph: genes are partitioned into
KEGG pathway subgraphs, and message passing happens **only among genes that
co-occur in the same pathway**. A sample-adaptive attention layer learns
per-patient edge weights (the effective pathway graph is recomputed for every
patient), and a Cox partial likelihood head optimizes the risk score directly.
Dual regularization (intra-pathway sparsity + consistency between stochastic
views) suppresses overfitting in high-heterogeneity cohorts.

## Repository layout

```
path_agnn_cox/        core package (pathway, models, loss, data, train, evaluate)
baselines/            LASSO-Cox, Ridge-Cox, Elastic-Net, RSF, DeepSurv, Cox-nnet, plain GNN
benchmark/            dataset manifest + run_benchmark.py + summarize.py + rewiring_analysis.py
config/               datasets.yaml (11 cancers), benchmark.yaml (hyperparameters)
data/scripts/         R download/preprocessing scripts for TCGA + GEO
data/processed/       per-cancer train.csv + external/<GSE>.csv (see data/README.md)
examples/quickstart.py  synthetic end-to-end demo
tests/smoke_test.py     all-baseline smoke test
work/                  analysis scripts (permutation test, clinical build, multivariable Cox,
                         standard-GAT control, external rewiring replication, IMvigor210)
manuscript/            template + render_manuscript.py + make_figures.py + check_formatting.py
```

## Install

```bash
pip install -r requirements.txt
# or: pip install -e .
```

## Quickstart (no real data needed)

```bash
python examples/quickstart.py
python tests/smoke_test.py
```

## Full benchmark (requires processed data)

```bash
# 1. download + preprocess (R)  -> data/processed/<CANCER>/train.csv, external/<GSE>.csv
# 2. place pathway GMT at data/pathways/KEGG_2021_Human.gmt (or edit config)
python -m benchmark.run_benchmark                       # all datasets, all models
python -m benchmark.run_benchmark --datasets LUAD,BRCA --models path_agnn_cox,lasso_cox,rsf
python -m benchmark.summarize                           # tables + figures in results/
```

## Evaluation

- Harrell C-index (lifelines)
- Time-dependent AUC (scikit-survival, at 25/50/75% quantiles of event times)
- Internal: stratified 5-fold CV on each TCGA cohort (seed 42)
- External: retrain on full TCGA, evaluate on each GEO cohort without fine-tuning
- Rewiring analyses (LUAD/BRCA): pathway-level between-stratum tests with
  BH-FDR, 200 label-permutation nulls, static-model negative control,
  randomized-partition control, clinical correlations, multivariable Cox

## Reproducing the manuscript

```bash
python manuscript/render_manuscript.py      # manuscript/Path-AGNN-Cox_manuscript.md (tables 1-5)
python manuscript/make_figures.py           # results/figures/Figure1-5 (SVG + PNG) + figure_manifest.json
python manuscript/check_formatting.py       # numbering / decimal / P-value audit (must pass)
# rewiring analyses (per dataset):
python benchmark/rewiring_analysis.py --dataset LUAD --train-csv data/processed/LUAD/train.csv \
    --gmt data/pathways/kegg_cancer_core.gmt --out results/rewiring/LUAD --known-pathways data/pathways/luad_known_pathways.txt
python work/permutation_test.py LUAD
python work/multivariable_cox.py LUAD
```

## Benchmark design (11 TCGA + 25 GEO cohorts)

| Cancer | TCGA train | GEO external |
|---|---|---|
| LUAD | TCGA-LUAD | GSE31210, GSE50081, GSE68465 |
| LUSC | TCGA-LUSC | GSE37745, GSE8894 |
| BRCA | TCGA-BRCA | GSE20685, GSE21653, GSE7390 |
| COAD | TCGA-COAD | GSE14333, GSE17536, GSE39582 |
| STAD | TCGA-STAD | GSE15459, GSE84437 |
| LIHC | TCGA-LIHC | GSE116174, GSE14520 |
| KIRC | TCGA-KIRC | GSE29609 |
| HNSC | TCGA-HNSC | GSE41613, GSE65858 |
| BLCA | TCGA-BLCA | GSE13507, GSE32894 |
| OV | TCGA-OV | GSE17260, GSE26712, GSE32062 |
| GBM | TCGA-GBM | GSE108474, GSE7696 |

## Results summary (see manuscript for details)

- Internal 5-fold CV: Path-AGNN-Cox mean C-index 0.56 (SD 0.04) across 11
  cancer types; Ridge-Cox 0.62 was the strongest baseline; DeepSurv/Cox-nnet 0.61.
- External (25 GEO cohorts): Path-AGNN-Cox 0.51 (SD 0.04), matching deep baselines
  (0.50); Ridge-Cox 0.57. Performance is honestly reported as comparable, not
  superior; the model's distinguishing value is interpretability.
- Patient-specific rewiring (LUAD/BRCA): between-stratum edge-weight differences
  far exceed label-permutation nulls (BRCA 43/53 pathways, P=0.005; LUAD 3/53,
  P=0.020), correlate with MKI67 (BRCA rho=0.33) and TMB (LUAD rho=0.13), and are
  absent by construction in static pathway models. The risk score remains
  significant after adjustment for stage and age (multivariable Cox: LUAD
  HR 1.22 per SD, P=0.035; BRCA HR 1.27 per SD, P=0.010). Full results in `results/`.

## Citation

To be updated after publication.

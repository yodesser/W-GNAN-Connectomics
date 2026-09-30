# W-GNAN: Weighted Graph Neural Additive Networks for Connectomics

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

Code and paper for **"Weighted Graph Neural Additive Networks for Structural Connectomes:
Interpretable by Design, Audited Out-of-Sample"** (Graph Machine Learning course project, Tel Aviv
University). The compiled paper is [`paper/main.pdf`](paper/main.pdf).

W-GNAN extends the Graph Neural Additive Network (GNAN) from hop distances on binary graphs to dense,
continuously weighted graphs. Region-specific shape functions produce node terms, and a learned
monotone edge-weight function `g(w)` modulates a one-hop interaction, so the model's logit for a fixed
graph decomposes exactly into node terms and pairwise interaction terms.

> **Data availability.** The structural connectomes and PCL-5 scores were provided by Yaniv Assaf's
> lab at Tel Aviv University and are not public. Subject-level data (matrices, manifests, demographic
> tables, cached features, trained fold weights) are **not** included in this repository. The code is
> released for methodological transparency; `results/` holds the aggregate outputs reported in the paper. The frozen split file
> (`splits/cv_splits.csv`) and per-subject predictions/explanations are withheld for the same reason;
> running the pipeline with authorized data regenerates them.

## Repository structure

```text
|-- models/           # W-GNAN architecture, real-data training, synthetic validation
|-- baselines/        # Logistic-regression, GCN and GAT baselines; benchmark table/figure
|-- audit/            # Interpretability audit: permutation tests, stability, GNNExplainer
|-- common/           # Paths, data loading/preprocessing, frozen CV splits, metrics, tables
|-- splits/           # Frozen CV split file location (3 repeats x 5 folds, seed 2026; not distributed)
|-- results/          # Aggregate predictions, tables, figures and audit outputs
|-- paper/            # ACL-format LaTeX source and compiled PDF
`-- requirements.txt
```

## Installation

Tested on Windows/macOS and Google Colab (T4 GPU).

```bash
git clone https://github.com/yodesser/W-GNAN-Connectomics.git
cd W-GNAN-Connectomics
pip install -r requirements.txt

# For exact reproducibility, limit threads:
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

With authorized data access, point `GML_ROOT` at the data folder (see `common/paths.py`); heavy
derived caches are written to `GML_CACHE` (default `~/.gml_cache`).

## Reproducing the pipeline

Run every module from the repository root.

| # | Command | Output | Runtime |
|---|---|---|---|
| 0a | `python -m common.splits --make` | Regenerates the frozen split file deterministically (seed 2026) | seconds |
| 0b | `python -m common.smoke_test` | Checks inputs, builds the local cache | ~2 min |
| 1 | `python -m baselines.train_linear` | Logistic-regression baselines and nuisance checks | ~70 min (CPU) |
| 2 | `python -m baselines.train_gat --repeats 0` | GAT predictions and fold weights | ~16 min (T4) |
| 3 | `python -m baselines.train_gcn --repeats 0` | GCN predictions | ~20 min (T4) |
| 4 | `python -m models.train_wgnan --variant {learned,linear,topology,nodeonly}` | W-GNAN variants | ~20 min each (T4) |
| 5 | `python -m models.train_synthetic` | Synthetic validation, Table 1, Figure 2 | ~3 min (T4) |
| 6 | `python -m audit.plot_stability` | In-sample vs. held-out ranking agreement | < 1 min |
| 7 | `python -m audit.run_permutation_tests --backend lr_strength --n-perm 500` | LR network permutation null | ~30 min |
| 8 | `python -m audit.run_permutation_tests --backend wgnan_learned --n-perm 100` | W-GNAN network permutation null | long; GPU recommended |
| 9 | `python -m audit.run_gnnexplainer` | GNNExplainer node masks on held-out GAT folds | ~2 hours |
| 10 | `python -m common.make_tables` | `benchmark.csv/.tex` | seconds |
| 11 | `python -m baselines.plot_benchmark` | Table 2, Table 3 | seconds |
| 12 | `python -m audit.tabulate_interpretability` | `interp.tex` | seconds |
| 13 | `python -m audit.plot_interpretability` | Figure 3 (`fig4_interp.pdf`) | seconds |

## Building the paper

```bash
cd paper
python verify_math.py      # sanity check of the preprocessing and logit decomposition
latexmk -pdf main.tex
```

The tables in `paper/tables/` are layout-edited versions of the generated files in `results/tables/`
(same numbers); the figures in `paper/figures/` come from `results/figures/`.

## Main findings

- On synthetic graphs with uninformative topology, weighted W-GNAN variants reach AUROC 0.87-0.95 and
  recover the generating weight-function shape; topology-only variants are at chance.
- On 869 structural connectomes (60 probable-PTSD positive screens), W-GNAN reaches AUROC 0.584
  [0.52, 0.65] versus 0.588 [0.52, 0.65] for edge-wise logistic regression; the paired interval does
  not resolve a difference.
- No hypothesis-motivated network exceeds its label-permutation null, and GNNExplainer node masks are
  stable but nearly label-invariant.

# GpLSI: Topic Modeling with Document Graphs

Official implementation of **Graph Topic Modeling for Documents with Spatial or Covariate Dependencies**, by Yeo Jin Jung and Claire Donnat. [Paper: arXiv:2412.14477](https://arxiv.org/abs/2412.14477).

GpLSI estimates document-topic proportions `W` and topic-feature profiles `A` from a row-normalized count matrix and a document graph. Graph-aligned SVD shares information across neighboring documents; vertex hunting recovers the topic geometry. The package also supports graph-free pLSI, synthetic experiments, and one experiment pipeline for four real datasets: CRC and mouse-spleen CODEX cellular neighborhoods, What's Cooking recipes, and Visium DLPFC spatial transcriptomics.

## Changes in this revision

- **Six vertex hunters:** the original SPA, SVS, SVS*, pp-SPA, PALM archetypal analysis, and accelerated PALM with monotone restart. Hunters share an interface and record selected vertices, optimization diagnostics, and explicit failure reasons.
- **Feature preprocessing:** whole-column Tran thresholding, empirical inverse-square-root frequency weighting, their combination, optional weight floors/caps, and retained-feature/rank diagnostics. The paper and source-script threshold rules are separate reproducible options.
- **Document and anchor-word recovery:** recover proportions from document embeddings or normalized feature profiles. Comparisons reuse the same spectral representation and random seeds.
- **Topic-profile recovery:** preserve the historical recovery, undo spectral feature weights, or refit profiles on the full vocabulary with simplex-constrained least squares or sparse fixed-`W` Poisson likelihood. Poisson recovery records objective history and an optimality-gap convergence certificate.
- **Graph solver controls:** reproducible SVD initialization, selectable cross-validation folds and worker counts, explicit weighted debiasing, graph/lambda diagnostics, and `lambda_selection_mode="cv_once"` (choose the penalty once, then hold it fixed). Historical defaults remain available.
- **Validated real-data preprocessing:** align integer counts, document lengths, feature and observation IDs, coordinates, and graphs; distinguish CRC sample-local cell IDs; support all three spleens jointly using a block-diagonal graph; validate against a packaged count/graph hash contract.
- **One experiment pipeline for all datasets:** a JSON config declares the dataset, task grid, spectral settings, GpLSI variants, A recoveries, and baselines; every requested method gets one result row (including failures), fits are restartable and cached, and post-hoc A refits reuse saved W. See [Running experiments](#running-experiments).
- **Reproducibility:** hash-checked data contracts, portable relative result paths, compact numerical fixtures, and regression tests covering geometry, likelihood, preprocessing, splits, and the pipeline.

Implementation and data details are in [docs/revision.md](docs/revision.md). Experiment outputs, raw data exports, cluster job ledgers, and manuscript files are not distributed with this revision.

## Installation

Clone first, then create the environment using the repository's **`environment.yaml`** file:

```sh
git clone https://github.com/yeojin-jung/GpLSI.git
cd GpLSI
conda env create -f environment.yaml
conda activate gplsi-env
python -m pip install -e '.[dev,benchmark]'
```

The graph solver requires [pycvxcluster](https://github.com/signal-lab-uchicago/pycvxcluster-0.1.0), installed by the conda environment's pip section. For an existing compatible Python environment, install it separately (including its SuiteSparse/scikit-sparse prerequisites):

```sh
python -m pip install 'pycvxcluster @ git+https://github.com/signal-lab-uchicago/pycvxcluster-0.1.0.git'
python -m pip install -e '.[dev,benchmark]'
python -c "from gplsi import GpLSI; import pycvxcluster.pycvxcluster; print('OK')"
```

Python 3.11 is the environment target. `benchmark` adds AnnData, Parquet, and resource-monitoring support. `legacy` adds R/rpy2 and MPI support for the original experiment scripts; a working R installation is needed before installing that extra. The core estimator and Python Topic-SCORE adapter do not require R. Spatial-LDA comparisons use `utils/spatial_lda`, so run those workflows from an editable checkout. Slurm orchestration requires a POSIX cluster; ordinary estimators and unit tests run locally.

## Basic usage

`X` is an `n × p` nonnegative frequency matrix, `N` is the mean original count total, `edge_df` has integer `src`, `tgt` and numeric `weight` columns, and `weights` is an `n × n` sparse adjacency matrix.

```python
from gplsi import GpLSI

model = GpLSI(lamb_start=1e-4, step_size=1.25, grid_len=29, eps=1e-5)
model.fit(X, N, K, edge_df, weights)
W = model.W_hat  # n documents × K topics
A = model.A_hat  # K topics × p features
```

The defaults retain the original unweighted SPA/current-recovery path. Use `method="pLSI"` for graph-free factorization. To select all graph-CV folds, explicitly set `graph_cv_fold_mode="all"`; `graph_n_jobs=1` uses serial folds, and `random_state` controls seeded spectral computations.

### Enable preprocessing and a new vertex hunter

```python
import numpy as np
from gplsi import GpLSI

# D is the original dense nonnegative integer count matrix.
N_i = D.sum(axis=1)
assert np.all(N_i > 0)  # remove empty documents and remap the graph first
X = D / N_i[:, None]

model = GpLSI(
    threshold_method="tran_script_exact",
    alpha=0.005,
    weight_method="ke_empirical",
    vertex_hunter="palm_accelerated",
    vertex_hunter_parameters={"lambda_": 1.0},
    A_recovery="A_full_Pois",
    random_state=42,
    graph_cv_fold_mode="all",
    graph_n_jobs=1,
)
model.fit(X, N_i, K, edge_df, weights, counts=D)
W, A = model.W_hat, model.A_hat
print(model.preprocessing_result.threshold.retained_indices)
print(model.vertex_result.status)
print(model.vertex_result.parameters["optimizer_converged"])
print(model.A_recovery_result.status, model.A_recovery_result.converged)
```

Pass `counts=D` to `A_full_Pois` to preserve the original counts. If omitted, the API reconstructs counts from `X` and the supplied depths; a mean depth cannot reconstruct unequal document counts. Supplying new options can change estimates, and a completed call can have a nonconverged recovery status. For PALM, inspect `optimizer_converged` as well as the vertex-result status: a finite candidate can be returned after its iteration budget is exhausted. The dense `GpLSI.fit` API is intended for matrices that fit in memory.

## Vertex-hunting procedures

Set `vertex_hunter` in `GpLSI`, or call `vertex_hunt(embedding, K, method=..., random_state=..., **parameters)` directly.

| Value | Procedure |
|---|---|
| `spa_current` | Original successive projection, including optional preconditioning. |
| `svs` | K-means centers followed by exhaustive simplex search; fixed or adaptive center count. |
| `svs_star` | SPA on the same K-means centers; adaptive mode reuses the SVS-selected center count, and `L_mode="svs_star_stability"` selects it from SVS*'s own vertex stability (scalable to large K). |
| `pp_spa` | Projected, locally denoised pseudo-points followed by SPA; economy SVD avoids a full document-by-document factor. |
| `palm` | Simplex-constrained archetypal analysis with alternating projected updates. |
| `palm_accelerated` | Accelerated PALM with backtracking and monotone restart. |

Pass method options in `vertex_hunter_parameters`. For example, `{"L_mode": "fixed", "L": 2*K}` sets a fixed center count for SVS/SVS*; pp-SPA accepts `radius_divisor`, `m_neighbors`, and `min_neighbors`; PALM requires an explicit `lambda_` and accepts `max_iterations` and `tolerance`. Adaptive SVS can be combinatorially expensive and has an explicit search budget. pp-SPA can fail when too few pseudo-points survive. Direct `vertex_hunt` calls record failures; `GpLSI.fit` raises `VertexHuntingError` when vertex hunting fails. No fallback hunter is substituted.

## Preprocessing and recovery

Let `eta_hat[j] = mean(X[:, j])`. Thresholding uses `alpha * sqrt(log(max(n,p)) / (n * mean(N_i)))` and selects whole feature columns **without renormalizing rows afterward**.

| Option | Behavior |
|---|---|
| `threshold_method="none"` | Keep all columns. |
| `"tran"` or `"tran_paper_exact"` | Keep `eta_hat >= threshold`; no fallback. |
| `"tran_script_exact"` | Keep `eta_hat > threshold`; if fewer than 10% survive, retain the highest-frequency `ceil(0.1*p)` columns with stable ties. |
| `weight_method="ke_empirical"` | Multiply retained columns by `(eta_hat + tau)**(-1/2)`. Default `tau=0`. |
| `"ke_empirical_additive_floor"` | Same additive-`tau` rule, explicitly labeled as stabilized weighting. |
| `"ke_empirical_capped"` | Apply the requested positive `weight_cap`. |

`weight_common_scale="rms_one"` optionally rescales weights to unit RMS. Unregularized inverse-frequency weighting is undefined for zero-frequency columns; remove them before weighting or explicitly choose a positive `tau`. Synthetic oracle modes (`oracle_eta`, `oracle_h`) require supplied population quantities and are not real-data estimators. `preprocess_features` exposes these operations independently of model fitting.

The experiment labels are P0 (unweighted), P1 (thresholded), P2 (weighted), and P3 (thresholded then weighted). Consult each config for its exact threshold rule: older synthetic configs use the paper alias, the real-data P1/P3 use `tran_script_exact` at alpha = 0.005, and the Cooking sensitivity variants (`P1_tran_alpha_0p01_drop_zero_rows`, ...) use the paper rule at alpha = 0.01 or 0.1. The pipeline removes zero-frequency columns before spectral weighting and keeps the full vocabulary for profile recovery.

| `A_recovery` | Profiles returned |
|---|---|
| `current` | Historical estimator. |
| `A_spectral_unweighted` | Undo feature weighting on the retained vocabulary and embed back into full feature space. |
| `A_full_L2` | Fixed-`W` simplex least-squares refit using original full-vocabulary frequencies. |
| `A_full_Pois` | Fixed-`W` simplex Poisson refit using original full-vocabulary counts and document-depth offsets. |

The Poisson implementation uses sparse count entries, chunked updates, monotone EM steps, and explicit convergence/support diagnostics. Held-out impossible events remain infinite-loss outcomes rather than being silently floored. Weighted spectral debiasing is separately configurable with `initialization`; unequal-depth use requires its explicitly labeled approximation. See [docs/revision.md](docs/revision.md).

## Real-data loaders

```python
from gplsi import GpLSI, load_real_data

bundle = load_real_data("spleen", group="BALBc-1")
model = GpLSI(
    vertex_hunter="svs_star",
    vertex_hunter_parameters={"L_mode": "fixed", "L": K + 2},
    A_recovery="A_full_Pois",
    random_state=42,
)
model.fit(bundle.frequencies, bundle.document_lengths, K,
          bundle.edge_df, bundle.weights, counts=bundle.counts)
```

`load_real_data` accepts `"crc"`, `"spleen"` (`group` = `BALBc-1/2/3` or `"joint"` for one fit across all three spleens with a block-diagonal graph), `"cook"` (historical 13,597 x 1,019 corpus), and `"cook_v2"` (raw recipes with at least 8 ingredients and a rebuilt Jaccard graph). Counts, lengths, graph indices, and metadata are validated together against frozen hashes; missing inputs or mismatches are errors. Outcomes and labels are kept for evaluation, outside estimator inputs. Visium DLPFC sections are prepared per task by the pipeline (`gplsi.pipeline.datasets`). Data layout and preparation are described in [data/README.md](data/README.md).

## Running experiments

The paper's real-data experiments follow one protocol, written out in [docs/protocol.md](docs/protocol.md): GpLSI (SVS\*, three A estimators) against pLSI, Topic-SCORE, graph-denoised Topic-SCORE, LDA and Spatial LDA; 20% of each document's counts held out; five seeds; one config per dataset, `configs/<dataset>/production.json` (DLPFC also `production_tran.json`, a Tran α = 0.1 vocabulary instead of the 2,000-gene panel; MERFISH and Xenium are prepared by `scripts/data/prepare_{merfish,xenium}.py --download`) ([configs/README.md](configs/README.md) documents the fields). The config's `grid` (K, seeds, and for DLPFC section/panel size) and `parts` define the tasks:

```sh
python scripts/analysis/word_frequency_diagnostics.py configs/cook/production.json  # before fitting
python scripts/run_experiment.py configs/crc/smoke.json            # the protocol on a tiny subset, one job
python scripts/run_experiment.py configs/crc/production.json --list
python scripts/run_experiment.py configs/crc/production.json --task 3
python scripts/summarize.py configs/crc/production.json            # -> results/<name>/fit_rows.csv
python scripts/refit_A.py configs/dlpfc/ablation/core.json         # post-hoc A from saved W (ablation)
```

On Slurm: `n=$(python scripts/run_experiment.py CONFIG --list | wc -l)` then `sbatch --array=0-$((n-1)) --export=ALL,CONFIG=CONFIG scripts/slurm/run_experiment.sh` (set `TASK_STRIDE` to pack tasks under a job limit; keep the slow `spatial_lda` part unpacked, see docs/protocol.md 1.8). Results go to `results/<name>/<task>/`: one JSON row and one `W_hat`/`A_hat` archive per fit, the shared spectral factors, and the task's data hashes and provenance. Load them with `gplsi.pipeline.load_rows` / `load_arrays`. Reruns skip fits whose data, settings, and code are unchanged.

Analyses read `results/<name>/` and write to its `figures/`. Figures that show one fit per method use the smallest successful seed and match topics to the same-K LDA fit by full-A cosine; shared helpers are in `scripts/analysis/shared.py`:

```sh
python scripts/analysis/plot_metrics_vs_K.py CONFIG      # deviance, runtime, PAS, CHAOS, Moran, roughness, topic overlap vs K
python scripts/analysis/seed_stability.py CONFIG         # cross-seed stability of A
python scripts/analysis/near_pure_counts.py CONFIG       # documents with W >= 0.95 per topic
python scripts/analysis/plot_topic_composition.py CONFIG # A rows (small vocabularies)
python scripts/analysis/plot_top_features.py CONFIG      # top-10 features per topic (large vocabularies)
python scripts/analysis/crc/evaluate_crc_patient_outcomes.py --fit-table results/crc_production/fit_rows.csv --output-dir ...
python scripts/analysis/crc/plot_outcome_composition.py configs/crc/production.json    # patient-pooled W by outcome; tissue maps
python scripts/analysis/crc/tumor_phenotype_alignment.py configs/crc/production.json   # topics vs focal tumor-cell type
python scripts/analysis/spleen/compartment_comparison.py configs/spleen/production.json  # ARI/AMI/LOSO, zone shares, maps
python scripts/analysis/cook/plot_cuisine_composition.py configs/cook/production.json  # mean W per cuisine
python scripts/analysis/dlpfc/plot_production.py configs/dlpfc/production.json        # layer ARI/NMI per section; maps
python scripts/analysis/plot_label_recovery.py CONFIG   # held-out label ARI/NMI/classifier per unit (DLPFC, MERFISH, Xenium)
python scripts/analysis/merfish/plaque_proximity.py configs/merfish/production.json  # plaque distance from W (5xFAD animals)
python scripts/analysis/xenium/disease_classification.py configs/xenium/production.json  # healthy vs UC from unit topic composition
python scripts/analysis/plot_unit_maps.py CONFIG       # dominant-topic maps vs a reference label per unit (DLPFC, MERFISH, Xenium)
```

The spleen scripts need `data/spleen/dataset/compartments/`, built by `scripts/data/prepare_spleen_compartment_annotations.py --archive CODEX_SpleenDataset.zip` (CytoCommunity archive). The DLPFC ablation (`configs/dlpfc/ablation/`) has its own scripts in `scripts/analysis/dlpfc/`; `configs/handoff/` reproduces the handoff runs.

## Simulations and tests

```sh
# Small synthetic comparison, no external Tran generator needed.
python scripts/simulations/run_vertex_hunting_weighting.py \
  --config configs/simulations/vertex_hunting_weighting/smoke.json

# Document and anchor-word comparison; requires the external Tran R generator.
python scripts/simulations/run_anchor_word_gplsi.py \
  --config configs/simulations/anchor_word_gplsi/smoke.json

python -m pytest -q
```

Set `GPLSI_TRAN_ROOT` to a checkout containing `r/experiments/synthetic/synthetic_dataset.R` for exact Tran simulations. The original simulation entry point is `scripts/simulations/run_sim.py` (grid in `scripts/simulations/config.txt`), with the notebook [scripts/simulations/example_synthetic_experiment.ipynb](scripts/simulations/example_synthetic_experiment.ipynb). Tests that need a dataset which is not present report explicit skips.

## Repository layout

```text
src/gplsi/                 Estimator: graph SVD, preprocessing, vertex hunting, recovery,
                           Topic-SCORE and LDA/NMF baselines, real-data loaders
src/gplsi/pipeline/        Experiment pipeline: configs, task data, method grid, metrics, results
scripts/run_experiment.py  Run tasks of a config (refit_A.py, summarize.py alongside)
scripts/data/              Build processed data (DLPFC h5ad, Cooking v2, spleen labels)
scripts/analysis/          Dataset-specific evaluation and figures
scripts/simulations/       Synthetic experiments, R generators, plots
scripts/slurm/             Slurm array launchers
configs/                   One JSON per experiment: base.json + <dataset>/production.json (paper protocol), dlpfc/ablation/, handoff/, simulations/
data/                      crc/, spleen/, cook/, dlpfc/ (see data/README.md)
utils/                     Original TopicSCORE R script and Spatial-LDA sources
tests/                     Regression tests and compact reference fixtures
docs/                      Implementation notes and the DLPFC ablation design/results
```

## External methods and citation

The original [TopicSCORE](https://github.com/ZhengTracyKe/TopicSCORE) R implementation remains in `utils/topicscore.r`, and [Calico Spatial-LDA](https://github.com/calico/spatial_lda) remains in `utils/spatial_lda`. Python adapters and parity fixtures document the additional vertex-hunting and Topic-SCORE procedures; supplied PALM reference hashes are retained in the implementation. Attribution and source checksums identify provenance and do not grant licenses for separately obtained reference code.

```bibtex
@article{jung2024gplsi,
  title={Graph Topic Modeling for Documents with Spatial or Covariate Dependencies},
  author={Jung, Yeo Jin and Donnat, Claire},
  journal={arXiv preprint arXiv:2412.14477},
  year={2025}
}
```

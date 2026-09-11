# GpLSI: Topic Modeling with Document Graphs

Official implementation of **Graph Topic Modeling for Documents with Spatial or Covariate Dependencies**, by Yeo Jin Jung and Claire Donnat. [Paper: arXiv:2412.14477](https://arxiv.org/abs/2412.14477).

GpLSI estimates document-topic proportions `W` and topic-feature profiles `A` from a row-normalized count matrix and a document graph. Graph-aligned SVD shares information across neighboring documents; vertex hunting recovers the topic geometry. The package also supports graph-free pLSI, synthetic experiments, CODEX cellular neighborhoods, recipe data, and joint-cohort spatial transcriptomics benchmarks.

## Changes in this revision

- **Six vertex hunters:** the original SPA, SVS, SVS*, pp-SPA, PALM archetypal analysis, and accelerated PALM with monotone restart. Hunters share an interface and record selected vertices, optimization diagnostics, and explicit failure reasons.
- **Feature preprocessing:** whole-column Tran thresholding, empirical inverse-square-root frequency weighting, their combination, optional weight floors/caps, and retained-feature/rank diagnostics. The paper and source-script threshold rules are separate reproducible options.
- **Document and anchor-word recovery:** recover proportions from document embeddings or normalized feature profiles. Comparisons reuse the same spectral representation and random seeds.
- **Topic-profile recovery:** preserve the historical recovery, undo spectral feature weights, or refit profiles on the full vocabulary with simplex-constrained least squares or sparse fixed-`W` Poisson likelihood. Poisson recovery records objective history and an optimality-gap convergence certificate.
- **Graph solver controls:** reproducible SVD initialization, selectable cross-validation folds and worker counts, explicit weighted debiasing, and graph/lambda diagnostics. Historical defaults remain available.
- **Validated real-data preprocessing:** align integer counts, document lengths, feature and observation IDs, coordinates, and graphs; distinguish CRC sample-local cell IDs; support all three spleens jointly using a block-diagonal graph; validate against a packaged count/graph hash contract.
- **Joint-cohort benchmarks:** donor/animal/patient holdouts, section/core-restricted graphs, conserved molecule splits, training-only gene panels, fixed-profile fold-in, common-vocabulary evaluation, restartable artifacts, and Slurm scheduling. Reporting checks comparison support, missing fits, and both training and fold-in convergence before restricted comparisons.
- **Reproducibility:** portable configuration paths, optional benchmark/legacy dependencies, compact numerical fixtures, and regression tests covering geometry, likelihood, preprocessing, split leakage, caching, scheduling, and reporting.

Implementation and data details are in [docs/revision.md](docs/revision.md) and [joint_v2_docs/README.md](joint_v2_docs/README.md). Experiment outputs, local dashboards, raw data exports, cluster job ledgers, and manuscript files are not distributed with this revision.

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

Pass `counts=D` to `A_full_Pois` to preserve the original counts. If omitted, the API reconstructs counts from `X` and the supplied depths; a mean depth cannot reconstruct unequal document counts. Supplying new options can change estimates, and a completed call can have a nonconverged recovery status. For PALM, inspect `optimizer_converged` as well as the vertex-result status: a finite candidate can be returned after its iteration budget is exhausted. The dense `GpLSI.fit` API is intended for matrices that fit in memory; the separate joint-cohort implementation provides sparse/chunked count workflows.

## Vertex-hunting procedures

Set `vertex_hunter` in `GpLSI`, or call `vertex_hunt(embedding, K, method=..., random_state=..., **parameters)` directly.

| Value | Procedure |
|---|---|
| `spa_current` | Original successive projection, including optional preconditioning. |
| `svs` | K-means centers followed by exhaustive simplex search; fixed or adaptive center count. |
| `svs_star` | SPA on the same K-means centers; adaptive mode reuses the SVS-selected center count. |
| `pp_spa` | Projected, locally denoised pseudo-points followed by SPA; economy SVD avoids a full document-by-document factor. |
| `palm` | Simplex-constrained archetypal analysis with alternating projected updates. |
| `palm_accelerated` | Accelerated PALM with backtracking and monotone restart. |

Pass method options in `vertex_hunter_parameters`. For example, `{"L_mode": "fixed", "L": 2*K}` sets a fixed center count for SVS/SVS*; pp-SPA accepts `radius_divisor`, `m_neighbors`, and `min_neighbors`; PALM requires an explicit `lambda_` and accepts `max_iterations` and `tolerance`. Adaptive SVS can be combinatorially expensive and has an explicit search budget. pp-SPA can fail when too few pseudo-points survive. Direct `vertex_hunt` calls record failures; `GpLSI.fit` raises `VertexHuntingError` when vertex hunting fails. No fallback hunter is substituted. The current joint-cohort grid uses SPA, SVS*, and accelerated PALM; the other hunters remain available for methodological comparisons.

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

The experiment labels are P0 (unweighted), P1 (thresholded), P2 (weighted), and P3 (thresholded then weighted). Consult each config for its exact threshold rule: older synthetic configs use the paper alias, while the real-data and joint-cohort pipelines use `tran_script_exact`. Joint workflows remove zero-frequency columns before spectral weighting and preserve the selected full panel for profile recovery.

| `A_recovery` | Profiles returned |
|---|---|
| `current` | Historical estimator. |
| `A_spectral_unweighted` | Undo feature weighting on the retained vocabulary and embed back into full feature space. |
| `A_full_L2` | Fixed-`W` simplex least-squares refit using original full-vocabulary frequencies. |
| `A_full_Pois` | Fixed-`W` simplex Poisson refit using original full-vocabulary counts and document-depth offsets. |

The Poisson implementation uses sparse count entries, chunked updates, monotone EM steps, and explicit convergence/support diagnostics. Held-out impossible events remain infinite-loss outcomes rather than being silently floored. Weighted spectral debiasing is separately configurable with `initialization`; unequal-depth use requires its explicitly labeled approximation. See [docs/revision.md](docs/revision.md).

## Real-data preprocessing

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

The canonical loader accepts `"crc"`, `"spleen"`, and `"cook"`, or their full dataset names (see [docs/revision.md](docs/revision.md)). Use `load_real_data("spleen", group="joint")` for one combined fit across all three spleens. Counts, lengths, graph indices, and metadata are validated together. CRC keys include both sample and cell IDs; joint spleen keeps sample graphs separate. Outcomes and labels are retained for evaluation, outside estimator inputs.

Place source inputs under `data/`, or set `GPLSI_DATA_ROOT` to the data directory. The packaged contract contains hashes and aggregate metadata, not raw observations. `GPLSI_AUDIT_CONTRACT` can point to an explicitly generated alternative contract. Missing inputs or hash mismatches are errors. The audit script can record the preprocessing of a supplied dataset collection:

```sh
python scripts/real_data_anchor_word_gplsi/audit_sources.py --help
python scripts/real_data_anchor_word_gplsi/run_experiment.py \
  configs/real_data_anchor_word_gplsi/spleen/smoke.json
```

## Experiments and tests

Run from the repository root after installation:

```sh
# Small synthetic comparison, no external Tran generator needed.
python codes/run_vertex_hunting_weighting.py \
  --config configs/vertex_hunting_weighting/smoke.json

# Document and anchor-word comparison; requires the external Tran R generator.
python codes/run_anchor_word_gplsi.py --config configs/anchor_word_gplsi/smoke.json

python -m pytest -q
```

Set `GPLSI_TRAN_ROOT` to a checkout containing `r/experiments/synthetic/synthetic_dataset.R` for exact Tran simulations. The original reference implementations are not vendored by this revision. Published-output audit helpers accept `GPLSI_PUBLISHED_REPO` for a checkout containing the historical artifact commit. Tests requiring separately supplied datasets or historical outputs report explicit skips when those inputs are absent.

The original entrypoints remain in `codes/`: `run_sim.py`, `run_spleen.py`, `run_crc.py`, `run_cook.py`, `run_crc_choose_ntopics.py`, and `postprocess_crc.py`. The notebook is [codes/example_synthetic_experiment.ipynb](codes/example_synthetic_experiment.ipynb). These legacy scripts have additional R/MPI/data requirements; use their `--help` output for options.

## Repository layout

```text
src/gplsi/                    Core estimator, preprocessing, hunters, recovery, loaders
src/gplsi_joint_v2/            Joint-cohort data, fitting, evaluation, reporting, scheduling
src/gplsi_spatial_benchmark/   Earlier independent-fit benchmark helpers
codes/                        Synthetic and original real-data entrypoints
scripts/real_data_anchor_word_gplsi/  Reusable fitting, audit, and analysis scripts
configs/                      Scientific experiment configurations
joint_v2_scripts/             Configurable Slurm launch wrappers
joint_v2_docs/                 Joint-cohort setup and data contract
utils/                        Original TopicSCORE R script and Spatial-LDA sources
tests/                        Regression tests and compact reference fixtures
docs/revision.md              Implementation and compatibility notes
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

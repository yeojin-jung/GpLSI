# Joint-cohort benchmark

`gplsi_joint_v2` fits one shared topic-profile matrix per platform and outer split, with observation-specific proportions. Visium DLPFC, MERFISH Trem2/5xFAD, and Xenium UC are fitted separately. The earlier `gplsi_spatial_benchmark` package remains for independent-fit comparisons.

## Data and preprocessing

The data root is supplied through `--root`. Raw data and metadata must already exist in the expected layout; these commands do not download datasets. `data.py` documents the input paths and required metadata fields. Inventory and split freezing are metadata operations:

```sh
python -m gplsi_joint_v2.cli inventory --root /path/to/benchmark --dataset visium_dlpfc
python -m gplsi_joint_v2.cli freeze --root /path/to/benchmark
```

Visium uses the full raw count export (Matrix Market counts plus feature and observation TSVs). MERFISH uses raw RNA counts; Xenium uses the annotated/QC raw-count cohort. Normalized expression and outcome labels are not estimator inputs. Cohort contracts preserve source hashes, QC choices, row/feature order, and artifact hashes.

Outer splits respect the biological grouping: Visium donor/section rotations and optional spatial halves, MERFISH animal holdouts with alternating training sections, and Xenium patient holdouts keeping all patient time points/cores together. Frozen design metadata is separate from evaluation annotations.

Graphs use six other observations within a platform/biological-unit/section/core stratum, distance-based weights, and maximum symmetrization. Self exclusion uses observation identity even at duplicate coordinates. The spectral penalty consumes unique undirected edges once.

Nested binomial thinning couples retention fractions 1, 0.75, 0.5, and 0.25. Independent stream namespaces separate training and held-out roles. Retained counts split 80/20 into fit/score or adaptation/score counts. Each molecule is conserved, and masks and seeds are recorded. A dataset/split/count cache is shared across K, hunter, and estimator seeds.

Visium gene panels are nested prefixes of a raw variance/mean ranking learned from fitting molecules only. Genes must be detected in at least `max(1, floor(0.01*n_eligible_training))` eligible rows. Inner graph folds learn their own feature selection. MERFISH and Xenium retain their native panels. Spectral preprocessing removes zero-frequency features; P1/P3 apply the strict Tran source-script rule and its logged top-10% fallback. Full selected panels remain available for profile recovery. Visium common-reference scores share vocabulary and count masks across native panel sizes.

## Fitting and reporting

The default scientific grid crosses P0/P1/P2/P3 with SPA, SVS*, and accelerated PALM, with graph-selected and exact-zero controls. Anchor-feature GpLSI uses P0/SPA. Competitors include raw and graph-denoised Topic-SCORE, LDA, KL-NMF, Spatial-LDA, and graph-KL-NMF. All reported proportions receive the same corrected fixed-W Poisson profile recovery. Historical/current recovery, unstarred SVS, plain PALM, and pp-SPA are not in this joint grid.

Scores use fixed-A fold-in and separate scoring counts. Reporting verifies vocabulary/mask hashes before pairing methods, retains infinite-support failures, represents missing fits explicitly, and uses expected manifest seed identities. Convergence-restricted transfer summaries require training and all count-eligible fold-in rows to converge. Repeated splits/seeds are averaged within biological subject before bootstrap resampling; uncertainty is conditional on fitted models. Visium reports donor values rather than treating repeated folds as independent subjects.

## Cluster setup

Install `.[benchmark]` and pycvxcluster in the active environment. Scientific configuration is available from `gplsi_joint_v2.config.default_config()` and may be saved as JSON for `--config`. Set resource account/partitions for your cluster; environment overrides are `GPLSI_SLURM_ACCOUNT` and `GPLSI_SLURM_PARTITION`. Heavy materialization, fitting, and reporting commands enforce a Slurm allocation. Correctness/smoke/pilot gates and resource reviews precede production dispatch.

The repository-local `joint_v2_scripts/run_action.sh` uses the current Python environment and accepts an explicit `--root`; `GPLSI_BENCHMARK_ROOT` is the fallback root. Cache and temporary files live below that root. No scheduler is launched by installation or testing.

```sh
# Inside a Slurm allocation, after preparing source inputs:
joint_v2_scripts/run_action.sh materialize --root /path/to/benchmark
python -m gplsi_joint_v2.cli --help
python -m gplsi_joint_v2.scheduler --help
```

Artifacts include content-addressed count caches, frozen manifests, one stored W per estimator, native/reference A profiles, compact biological/section score summaries, and checksummed completion markers. Transfer W and observation score vectors are transient. Missing completion markers allow restart; corrupted completed artifacts fail validation. Job state and completed scientific results must be assessed from the generated artifacts, not inferred from these source documents.

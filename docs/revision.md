# Revision implementation notes

This revision consolidates the reusable code from the vertex-hunting/weighting experiments, anchor-word experiments, real-data reruns, the Visium DLPFC ablation, and subsequent corrections. How the September 2026 real-data handoff was merged is recorded in [handoff_merge.md](handoff_merge.md). It preserves the newer seeded-SVD compatibility helper, cached adaptive SVS selection, multi-sample Spatial-LDA adapter, and joint-spleen loader while incorporating sparse Poisson recovery and the pp-SPA economy-SVD fix.

## Geometry and feature transforms

`preprocessing.py` returns the original, retained, and weighted matrices together with retained/discarded column indices, row mass, rank information when available, weight quantiles, singular values, and effective method names. Feature selection is columnwise and does not restore removed row mass. Consequently, full-vocabulary recovery must use the original data rather than interpreting retained frequencies as renormalized documents.

The four experimental preprocessings are independently configurable. Weight floors, caps, common rescaling, and weighted initialization are explicit options, not hidden changes to the unweighted baseline. `initialization="weighted_debiased"` uses the weighted diagonal correction; unequal document lengths require `"weighted_debiased_mean_N_approx"`, which records that the correction substitutes a mean depth. `initialization="current"` keeps the historical initialization. A retained vocabulary with fewer than K columns produces an explicit rank-related failure by default.

`vertex_hunting.py` returns structured results, including the embedding coordinate convention needed by recovery. SPA signs are carried into unweighted spectral recovery. SVS searches candidate simplexes; SVS* uses SPA on the shared centers. Adaptive SVS results can be reused exactly when computing SVS*. PALM constrains its archetypal factors to simplexes and exposes stopping/line-search diagnostics; the accelerated version accepts extrapolation only under its monotone restart rule. No hunter silently substitutes another method after failure.

`anchor_word.py` constructs normalized feature profiles from the fitted spectral matrix and converts feature vertices to document proportions. Document-side and anchor-word estimates are distinct estimators; the experiment runners pair them with shared preprocessing, counts, graph folds, and seeds.

## Full-vocabulary profiles and prediction

`recovery.py` provides historical, spectral-unweighted, L2, and Poisson recovery. The fixed-W Poisson objective uses nonnegative counts, observation depths, and simplex topic profiles; supply original integer counts for count-data analyses. Sparse inputs are canonicalized, duplicate count entries are combined, and updates operate on chunks of observed entries. The optimizer reports status, objective history, iterations, and a simplex optimality-gap certificate. Check both `status` and `converged`: `max_iter_reached` marks an exhausted budget, and `unidentified_inactive_topics` marks topic profiles that cannot be identified from W. Inactive profiles are preserved after the initial interiorization. The separately labeled MAP option is not the unpenalized MLE.

The experiment pipeline scores every fit on held-out molecules from binomial count thinning (exact multinomial likelihood, with impossible support reported as infinite loss and a separately named floored diagnostic), on training-count fit and graph smoothness of W, and, where evaluation-only labels exist (DLPFC layers), on label agreement. Library size acts as an observed offset: these scores assess conditional composition.

## Canonical real-data inputs

`RealDataBundle` holds integer counts, row frequencies, row-specific count totals, feature/observation/group IDs, graph edges and sparse weights, plus optional coordinates/outcomes. Validation checks alignment and nonnegative counts before fitting. The packaged contract additionally checks preprocessing hashes.

- CRC uses sample-qualified cell identifiers when merging counts, coordinates, edges, and outcomes, preventing collisions between cells with the same local ID in different samples.
- Spleen supports BALBc-1, BALBc-2, and BALBc-3. `load_real_data("spleen", group="joint")` concatenates rows on the common feature order and creates a block-diagonal graph, retaining group IDs and independently validated component hashes.
- What's Cooking retains the canonical recipe/ingredient selection and count/graph construction, with stable feature and observation IDs.

Use `load_real_data` for validated named datasets: `"crc"`/`"stanford_crc_codex"`, `"spleen"`/`"mouse_spleen_codex"`, and `"cook"`/`"whats_cooking"`. `GPLSI_DATA_ROOT` selects a data directory, and `GPLSI_AUDIT_CONTRACT` selects an alternate frozen contract. Raw source inputs are supplied separately. Contract files contain feature names, aggregate statistics, and hashes, not individual observations or fitted outcomes.

## Compatibility and validation

The historical default GpLSI path remains unweighted SPA with current A recovery. Graph controls are forwarded in both the default and extended paths; `legacy_first_three` scores lambda choices using folds 0, 1, and 2, while `all` scores every nonempty configured fold. Both modes fit every nonempty fold. Explicit seeds are forwarded to the seeded SVD helper. Legacy top-level helpers remain available without eagerly importing optional R/MPI experiment modules.

The dense `gplsi.graphSVD` implementation preserves the historical numerical algorithm.

The regression suite checks no-op baseline behavior, threshold boundary/fallback parity, fixed-center SVS comparisons, pp-SPA reference output, PALM objective behavior, geometry recovery, independent Poisson optimization oracles, count conservation, biological holdouts, feature/annotation leakage, graph boundaries, artifact corruption and restart handling, and reporting support/convergence rules. Compact synthetic fixtures travel with the code. Tests that require unbundled real datasets or historical outputs are skipped explicitly when those inputs are absent.

Configuration files preserve scientific settings. Submission ledgers, job IDs, results, raw exports, manuscript revisions, and deployed dashboards are not source dependencies. Slurm resource/account choices must be configured for the target cluster; a local unit-test pass does not imply a completed full-cohort experiment.

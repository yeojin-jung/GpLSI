# Visium DLPFC ablations of GpLSI

Status (2026-09-27): the data pipeline, ablation code, and task manifests are
implemented and unit-tested (305 tests pass). Component timing and Poisson
convergence diagnostics were run on one section. **No ablation results (layer
ARI, held-out deviance, stability) have been produced yet**; the core, panel,
and penalty-grid runs have not been launched. Nothing is committed.

Contents: [1 Data](#1-data) · [2 Processing](#2-per-task-processing) ·
[3 Shared GpLSI pipeline](#3-the-shared-gplsi-pipeline) ·
[4 Ablation table](#4-ablation-table) · [5 Baselines](#5-baselines) ·
[6 Why 2,000 genes](#6-why-2000-genes) · [7 Designs](#7-experimental-designs-which-cells-are-run) ·
[8 Metrics](#8-evaluation-metrics) · [9 Diagnostics so far](#9-diagnostics-run-so-far-and-results) ·
[10 Code map](#10-code-map) · [11 Running](#11-running) · [12 Plans](#12-open-decisions-and-plans)

---

## 1. Data

10x Genomics Visium, human dorsolateral prefrontal cortex (Maynard et al., *Nature
Neuroscience* 2021), the same cohort as §3.1 of `GPLSI_experiments.pdf`.

| Input | Source | Local copy |
|---|---|---|
| Raw UMI counts, filtered in-tissue spots | `https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/<id>_filtered_feature_bc_matrix.h5` | `data/interim/visium_dlpfc/raw/` |
| Spot positions (`array_row`, `array_col`, pixel row/col) | `LieberInstitute/HumanPilot/10X/<id>/tissue_positions_list.txt` | same |
| Manual layer labels | `ground_truth` column of `HumanPilot/outputs/SpatialDE_clustering/cluster_labels_<id>.csv` (the `layer_guess_reordered` annotation distributed by spatialLIBD) | same |

These public files were used instead of `spatialLIBD::fetch_data("spe")` because
spatialLIBD is not installed locally; the resulting cohort matches the PDF exactly.

| Donor | Sections | Spots | Layers present |
|---|---|---|---|
| Br5292 | 151507, 151508, 151509, 151510 | 4,226 / 4,384 / 4,789 / 4,634 | L1–L6, WM |
| Br5595 | 151669, 151670, 151671, 151672 | 3,661 / 3,498 / 4,110 / 4,015 | **L3–L6, WM only** |
| Br8100 | 151673, 151674, 151675, 151676 | 3,639 / 3,673 / 3,592 / 3,460 | L1–L6, WM |

Totals: 47,681 spots × 33,538 genes, 82,659,007 nonzero entries, 165,059,561 UMIs;
47,329 labeled spots (352 blank).

**One-time preparation** — `scripts/visium_dlpfc/prepare_data.py` →
`data/processed/visium_dlpfc/visium_dlpfc.h5ad` (+ `summary.json`):

* `X`: raw integer UMIs, all 33,538 genes, CSR. No normalization, QC filter, or gene
  selection at this stage (genes are chosen per task, §2).
* `obs`: `sample_id`, `subject` (donor), `position` (0 / 300 µm), `replicate`, and the
  evaluation-only `layer_guess_reordered` (renamed `Layer_1`→`Layer1`). The 534
  `discard=True` spots are kept, as in the PDF.
* `obsm["spatial"]` $= (\text{array\_col},\ \sqrt{3}\,\text{array\_row})$. Visium spots lie
  on a hexagonal lattice; the $\sqrt3$ scaling makes all six lattice neighbours
  equidistant, so a 6-NN graph is exactly the hex grid. `obsm["spatial_pixel"]` keeps
  pixel coordinates for plotting.
* `uns`: `benchmark_unit_column = graph_unit_column = "sample_id"`,
  `fit_forbidden_obs_columns = ["layer_guess_reordered"]`, source-file SHA-256s.

## 2. Per-task processing

A task is (design, section $u$, $K$, panel size $p$, retained fraction $r$, seed $s$).
`prepare_task_data` in `src/gplsi_spatial_benchmark/ablation_runner.py` performs every
step deterministically from the seed and training counts, so a later recovery-only
refit (Poisson, §4) rebuilds identical data.

1. **Molecule split** (`thin_and_split_sparse_counts`, `splits.py`). For each stored
   nonzero $D_{ij}$: $D^{(r)}_{ij} \sim \mathrm{Bin}(D_{ij}, r)$,
   $D^{\text{test}}_{ij} \sim \mathrm{Bin}(D^{(r)}_{ij}, 0.2)$,
   $D^{\text{train}} = D^{(r)} - D^{\text{test}}$, via shared inverse-CDF uniforms so
   retained counts are nested across $r$ for a fixed seed. Works on nonzeros only, so
   the 33.5k-gene matrix is never densified.
2. **Training-only gene panel** (`rank_features_by_dispersion`, `panel_indices`,
   `panels.py`). Among genes detected in $\ge \max(1,\lfloor 0.01\,n\rfloor)$ training
   spots, rank by raw variance-to-mean ratio
   $\widehat{\mathrm{Var}}(D^{\text{train}}_{\cdot j}) / \widehat{\mathbb E}(D^{\text{train}}_{\cdot j})$
   (stable ties). The fitted panel is the top $p$; the *reference panel* is the top 500.
   Panels of different sizes are nested prefixes of one ranking.
3. **Spot mask.** Keep spots with positive training counts on the reference panel — the
   same spots for every panel size.
4. **Graph** (`build_within_unit_knn_graph`, `graph.py`). Symmetric 6-NN graph within
   the section, $w_{ii'} = \exp\{-(d_{ii'}/\tilde d)^2\}$, $\tilde d$ = median positive
   neighbour distance.
5. **Estimator inputs.** Training counts $D$ ($n\times p$), depths
   $N_i=\sum_j D_{ij}$, frequencies $X_{ij}=D_{ij}/N_i$, edges and weights. Layer labels
   are withheld.

## 3. The shared GpLSI pipeline

Model: $\mathbb E[X] = W A$, $W\in\mathbb R^{n\times K}$ and $A\in\mathbb R^{K\times p}$
with rows on the probability simplex $\Delta$. Every GpLSI variant runs:

| Step | What happens | Code |
|---|---|---|
| (a) Preprocess columns | $\tilde X = X_{\cdot,J}\,\mathrm{diag}(r_J)$: threshold set $J$ and weights $r$ from the ablation (§4, dimension C). Zero-frequency training genes are removed first. | `preprocess_features` (`gplsi/preprocessing.py`), called from `fit_spectral_block` (`gplsi/real_experiment.py`) |
| (b) Graph-aligned SVD | Alternate $V,\Lambda \leftarrow \mathrm{SVD}_K(\tilde X^\top U)$ and $\hat U \leftarrow \arg\min_U \tfrac12\lVert \tilde X V - U\rVert_F^2 + \lambda\sum_{(i,i')\in E} w_{ii'}\lVert U_{i\cdot}-U_{i'\cdot}\rVert_2$ (convex clustering, SSNAL solver), then orthonormalize. $\lambda$ is chosen by 5-fold graph cross-validation (held-out nodes interpolated from neighbours) over $\{10^{-6}\}\cup\{10^{-4}\cdot1.2^j\}_{j=0}^{28}$ (top ≈ 0.0165), all folds scored; ≤ 50 outer iterations, tolerance $10^{-5}$. | `graphSVD`, `update_U_tilde`, `lambda_search` (`gplsi/graphSVD.py`); `pycvxcluster.SSNAL` |
| (c) Vertex hunting | Find $K$ vertices $H\in\mathbb R^{K\times K}$ of the point cloud (rows of $\hat U$) — ablation dimension B. | `vertex_hunt` (`gplsi/vertex_hunting.py`), via `fit_geometry` |
| (d) Recover $W$ | Barycentric coordinates $W^{\text{raw}} = \hat U H^{-1}$, each row Euclidean-projected onto $\Delta$. | `recover_W` (`gplsi/recovery.py`) |
| (e) Estimate $A$ | Given the fixed $\hat W$, fit $A$ on **all $p$ panel genes** (including genes dropped by thresholding in (a)) — ablation dimension A. | `recover_A_for_geometry` (`gplsi/real_experiment.py`) |

Steps (a)–(b) are computed once per preprocessing and shared by every hunter and every
$A$ estimator in a task, so the $A$ estimators always see an identical $\hat W$.
W-only metrics (layer ARI/NMI, spatial) are therefore identical across dimension A
by construction.

**Anchor-feature GpLSI** (`gplsi_anchor__*`) is a different geometry, not an ablation
axis: vertices are hunted on normalized gene profiles
$Z = \mathrm{diag}(\hat\eta_J)^{-1}\mathrm{diag}(r_J)^{-1}\hat V\hat\Lambda$
(`build_word_profile`, `gplsi/anchor_word.py`) and mapped back to spot proportions
(`recover_W_from_word_vertices`). It is included once (P0, SPA) in the core design.

## 4. Ablation table

Three dimensions. Names in the first column are the strings used in the config
(`configs/visium_dlpfc/ablation.json`) and in result method names
`gplsi_document__<preprocessing>__<hunter>__<A>`.

### Dimension A — estimating the topic-gene matrix $A$ (given $\hat W$)

| Setting | Estimator | Formula | Source |
|---|---|---|---|
| `A_current` (**original GpLSI**) | Least squares, then row-wise projection onto the simplex. Unweighted in spots; uses frequencies. Projecting the unconstrained solution is not the same as solving the constrained problem unless $\hat W^\top\hat W\propto I$. | $\hat A = \Pi_\Delta\big[(\hat W^\top\hat W)^{-1}\hat W^\top X\big]$ (rows projected separately) | `refit_A_current` (`gplsi/recovery.py`) |
| `A_full_L2` | Simplex-**constrained** least squares, the actual Euclidean minimizer | $\hat A = \arg\min_{A:\,A_{k\cdot}\in\Delta}\lVert \hat W A - X\rVert_F^2$; projected gradient, step $1/(2\lVert\hat W\rVert_2^2)$ with backtracking, warm start `A_current`, stop when relative change $\le10^{-8}$ (≤ 2,000 iterations) | `refit_A_full_l2` (`gplsi/recovery.py`); branch added in `recover_A_for_geometry` |
| `A_full_Pois` | Poisson (equivalently multinomial) maximum likelihood on training **counts**. Spots are weighted by depth $N_i$, and the loss is log-likelihood instead of squared error. | $\hat A = \arg\min_{A:\,A_{k\cdot}\in\Delta}\sum_{ij}\big[N_i(\hat WA)_{ij} - D_{ij}\log(\hat WA)_{ij}\big]$. Since rows of $\hat W$ and $A$ sum to 1, $\sum_j N_i(\hat WA)_{ij}=N_i$ is constant, so this is the multinomial MLE. EM update: $s_{kj}=\sum_i \hat W_{ik}D_{ij}/(\hat WA)_{ij}$, $A_{kj}\leftarrow A_{kj}s_{kj}/\sum_{j'}A_{kj'}s_{kj'}$. Convergence certificate (Frank–Wolfe/KKT gap): $G=\sum_k[\max_j s_{kj}-\sum_j A_{kj}s_{kj}]$, converged when $G/\sum_{ij}D_{ij}\le10^{-8}$. | `refit_A_full_poisson` (`gplsi/recovery.py`, unchanged); run post hoc by `scripts/visium_dlpfc/refit_poisson.py` |

### Dimension B — vertex hunting on the rows of $\hat U$

| Setting | Procedure | Source |
|---|---|---|
| `spa_current` (**original GpLSI**) | Successive projection: flip column signs so $\hat U_{1k}\ge0$, then $K$ times pick the row with largest residual norm $\lVert s\rVert$ and project all rows onto $s^\perp$. No preconditioning. | `_spa_current` (`gplsi/vertex_hunting.py`) |
| `svs` | Sketched vertex search (MixedSCORE): k-means with $L=K+2$ centers, then exhaustively evaluate all $\binom{L}{K}$ subsets and pick the one minimizing the maximum distance from the remaining centers to the subset's convex hull. | `_run_svs`, `exhaustive_vertex_search` |
| `svs_star` | Same k-means centers as SVS, but choose vertices by SPA on the centers. | `_run_svs_star` |
| `pp_spa` | Pseudo-point SPA: project points to the $(K{-}1)$-dim affine hull; replace each point by the mean of its $m=4$ nearest neighbours within radius $\max\text{dist}/20$ (drop points with fewer than 3 neighbours); then affine SPA on the pseudo-points. Denoises before SPA. | `_project_affine`, `_pp_pseudo_points`, `_affine_spa` |
| `palm` | Archetypal analysis: $\min_{W,H}\tfrac12\lVert \hat U - WH\rVert_F^2 + \tfrac{\lambda}{2}\sum_k \mathrm{dist}(H_{k\cdot},\mathrm{conv}(\hat U))^2$ with $W$ rows on $\Delta$; PALM alternating proximal updates, $\lambda=1$, ≤ 300 iterations, tolerance $10^{-7}$. | `palm_aa_vertex_hunt` (`gplsi/accelerated_palm_aa.py`) |
| `palm_accelerated` | Same objective; accelerated PALM with extrapolation and a monotone-restart safeguard. | `accelerated_palm_aa` (same file) |

Hunter settings are defaults set in `_vertex_parameters` (`gplsi/real_experiment.py`);
`vertex_parameters` in the config can override them. A failed hunter is recorded as a
failure; no other hunter is substituted.

### Dimension C — gene thresholding (and weighting)

C1 chooses **which genes enter the model**; C2 chooses **how the spectral step (a) treats
them**. Dimension A always estimates $A$ on all $p$ panel genes.

| Setting | Rule | Source |
|---|---|---|
| **C1 panel size** $p\in\{500,1000,2000,5000\}$ | Top-$p$ genes by training variance/mean among genes detected in ≥ 1% of spots (§2). **2,000 is the main setting** (§6). | `panel_indices`, `rank_features_by_dispersion` (`gplsi_spatial_benchmark/panels.py`); `panel_size` task column |
| **C2** `P0_raw` (**original GpLSI**) | $J$ = all positive-frequency genes, $r_j=1$. | `PREPROCESSING_SPECS` (`gplsi/real_experiment.py`) → `preprocess_features` |
| `P1_tran_alpha_0p005` | Tran threshold. With $\hat\eta_j=n^{-1}\sum_iX_{ij}$ and $\bar N=n^{-1}\sum_iN_i$, keep $j$ if $\hat\eta_j > \alpha\sqrt{\log\max(n,p)/(n\bar N)}$ with $\alpha=0.005$. If fewer than 10% survive, keep the top $\lceil0.1p\rceil$ by $\hat\eta_j$. Rows are not renormalized. | `select_feature_columns(method="tran_script_exact")` |
| `P2_ke_weighted` | No threshold; Ke–Wang inverse-square-root frequency weighting $r_j=\hat\eta_j^{-1/2}$, which upweights low-frequency genes. | `frequency_weights(method="ke_empirical")` |
| `P3_tran_then_ke` | P1 threshold, then P2 weights on the retained genes. | both |

On a 2,000-gene panel chosen for high dispersion, P1 may drop few genes; the runs
record `retained_feature_count` to check this.

## 5. Baselines

| Name | Method | Details | Source |
|---|---|---|---|
| `topicscore_raw` | Topic-SCORE (Ke & Wang) | SVD of $X\,\mathrm{diag}(\hat\eta)^{-1/2}$ → right singular vectors $\xi_1..\xi_K$; ratios $R_{jk}=\xi_{k+1}(j)/\xi_1(j)$; SPA on the ratio cloud; barycentric weights $\Pi$; $A_{\cdot j}\propto\sqrt{\hat\eta_j}\,\xi_1(j)\,\Pi_j$; then $W$ by simplex-constrained least squares (≤ 5,000 projected-gradient iterations). | `fit_topicscore_raw` (`gplsi/topicscore.py`) |
| `topicscore_graph_denoised` | Topic-SCORE on GpLSI's graph-aligned factors | Same ratio/vertex/$A$/$W$ steps, using $\hat V,\hat\Lambda$ from step (b) (P0) instead of the raw SVD. | `fit_topicscore_graph_denoised` |
| `lda` | Latent Dirichlet allocation | scikit-learn `LatentDirichletAllocation` (batch variational Bayes, default priors $1/K$, 10 passes) on integer training counts; $W$ = normalized document-topic posterior, $A$ = normalized topic-word pseudo-counts. | `fit_lda` (`gplsi/baselines.py`) |
| `kl_nmf` | Nonnegative matrix factorization with KL (Poisson) loss | $\min_{\tilde W,\tilde H\ge0}\sum_{ij}\big[D_{ij}\log\frac{D_{ij}}{(\tilde W\tilde H)_{ij}} - D_{ij} + (\tilde W\tilde H)_{ij}\big]$; scikit-learn `NMF(beta_loss="kullback-leibler", solver="mu", init="nndsvda", max_iter=500)`. Then $A=\tilde H$ with rows normalized; $W=\tilde W$ rescaled by topic mass and rows normalized. The non-graph counterpart of `A_full_Pois`. | `fit_kl_nmf` (`gplsi_spatial_benchmark/methods.py`) |
| `graph_kl_nmf` | KL-NMF with a graph-Laplacian penalty on $\tilde W$ | KL objective + $\gamma\,\mathrm{tr}(\tilde W^\top L\tilde W)$, $\gamma=0.25$, multiplicative updates (≤ 500 iterations). A matched spatial baseline written for this benchmark, not an external published method. | `fit_graph_kl_nmf` (same file) |
| `spatial_lda` | Spatial-LDA (Chen et al. 2020, Calico) | LDA whose document-topic proportions are penalized toward spatial neighbours (difference penalty 0.25), fit by ADMM; vendored implementation. | `fit_spatial_lda` (`gplsi/baselines.py`), `utils/spatial_lda/` |

## 6. Why 2,000 genes?

2,000 is the **main** panel size, not a claim that it is optimal; dimension C1 tests
500–5,000 directly.

1. **Theory.** GpLSI's guarantees are stated for a vocabulary that is small relative to
   the number of documents and their lengths. A Visium section has $n\approx3.5$–$4.8$k
   spots, but 33,538 genes, most of them detected in almost no spots. Using all genes
   puts us in $p\gg n$, where the estimation theory does not apply and most columns are
   near-zero noise. On section 151673 only 13,368 genes pass the 1% detection filter.
2. **Comparability.** The Midway benchmark in the PDF uses 2,000 genes ranked by
   variance-to-mean, so results line up with Claire's reruns. Here the panel is chosen
   from each task's *training* counts rather than from all 12 sections, which removes
   the leak the PDF flags.
3. **Convention.** 2,000 highly variable genes is the default in Seurat and Scanpy and
   common in DLPFC spatial-domain benchmarks, so practitioners will recognize it.
4. **Signal.** High-dispersion genes carry the layer structure (they vary across spots),
   whereas low-dispersion genes add mostly Poisson noise.
5. **Compute.** Graph SVD and the $A$ refits work on dense $n\times p$ matrices: about
   4k × 2k (≈ 64 MB) at $p=2{,}000$ versus about 4k × 33.5k (≈ 1 GB) for all genes, with
   run time scaling roughly linearly in $p$.

Held-out deviance on the fitted panel is not comparable across panel sizes (different
vocabularies). The panel sweep therefore also scores on the common top-500 reference
panel (§8).

## 7. Experimental designs (which cells are run)

Defined under `designs` in `configs/visium_dlpfc/ablation.json`; manifests are written by
`scripts/visium_dlpfc/make_tasks.py`. Seeds are 26090401–26090405, as in the PDF.

| Design | Dim. A | Dim. B | Dim. C | Other methods | Tasks (manifest) |
|---|---|---|---|---|---|
| `core` | `A_current`, `A_full_L2` (+ `A_full_Pois` post hoc) | all 6 | $p=2{,}000$ × P0–P3 | anchor GpLSI (P0/SPA) + all 6 baselines* | 151507, 151669, 151673 (one per donor) × 3 seeds, $K=7$, $r=1$ → **9** (`tasks_core.csv`) |
| `panel` | same | SPA, SVS\*, accel. PALM | $p\in\{500,1000,2000,5000\}$ × {P0, P1} | raw Topic-SCORE, LDA, KL-NMF | 3 sections × 3 seeds × 4 sizes → **36** (`tasks_panel.csv`) |
| `lambda_wide` | `A_current`, `A_full_L2` | SPA | $p=2{,}000$ × {P0, P2} | — | 3 sections × 1 seed → **3**; penalty grid extended to $j\le49$ (top ≈ 0.758) |
| extended `core` (after meeting) | as core | all 6 | as core | as core | 12 sections × 5 seeds, $K\in\{5,7,9\}$ at $r=1$, and $r\in\{0.75,0.5,0.25\}$ at $K=7$ → **360** (`tasks_extended.csv`) |
| `smoke` | as core | all 6 | as core | as core | 1 task, tiny penalty grid (3 values, 3 iterations, 3 folds) |

\* Spatial-LDA's inclusion is pending (§12). Per core task this is
4 × 6 × 3 = 72 document-side GpLSI fits, plus 3 anchor fits and 6 baselines.
Analysis compares each variant with the original GpLSI (P0, SPA, `A_current`), one
dimension at a time, and also reports the full factorial.

## 8. Evaluation metrics

For a fit $(\hat W,\hat A)$ with $P=\hat W\hat A$ (rows renormalized) and test counts $Y$,
$m_i=\sum_jY_{ij}$ (`score_fit` in `ablation_runner.py`; functions in
`gplsi_spatial_benchmark/metrics.py`):

| Metric | Definition | Depends on |
|---|---|---|
| Layer ARI / NMI | between $\hat z_i=\arg\max_k\hat W_{ik}$ and the manual layer (labelled spots only) | $W$ |
| Layer balanced accuracy | 5-fold cross-validated class-weighted multinomial logistic regression of layer on $\hat W_{i\cdot}$ | $W$ |
| Held-out deviance / molecule | $\mathrm{Dev}=2\sum_{ij}\big[Y_{ij}\log\frac{Y_{ij}}{m_iP_{ij}}-(Y_{ij}-m_iP_{ij})\big]$ divided by $\sum Y$; lower is better. Conditional on $m_i$: it tests allocation of held-out molecules across genes. | $W,A$ |
| Held-out log score / molecule | $\sum_{ij}Y_{ij}\log P_{ij}/\sum Y$; higher is better | $W,A$ |
| Zero-probability molecules | count of held-out molecules with $P_{ij}=0$ (infinite deviance; also reported floored at $10^{-12}$) | $W,A$ |
| Reference-panel deviance | same, with $P$ and $Y$ restricted to the top-500 genes and $P$ renormalized; comparable across panel sizes | $W,A$ |
| Moran's I, neighbour agreement | spatial smoothness of $\hat W$ (diagnostic only) | $W$ |
| Topic entropy, top-gene exclusivity | $-\sum_jA_{kj}\log A_{kj}/\log p$; $e_j=\max_kA_{kj}/\sum_\ell A_{\ell j}$ over each topic's top-20 genes | $A$ |
| Seed stability | Hungarian-matched Jensen–Shannon divergence between topics of different seeds (shared genes), in `summarize.py` | $A$ |

Metrics are stored for every finite fit, with the optimizer status recorded separately
(`status`, `A_recovery.converged`, `vertex_hunting.optimizer_converged`). Seeds are
averaged within a section before sections are compared; with three donors, per-section
values are reported rather than p-values.

JSON keys (per record, under `metrics`) and the column names used by `summarize.py`:

| JSON key | Summary column |
|---|---|
| `external__layer_guess_reordered__ari` / `__nmi` / `__cv_balanced_accuracy` | `layer_ARI` / `layer_NMI` / `layer_balanced_acc` |
| `heldout_poisson_deviance_per_molecule`, `heldout_log_likelihood_per_molecule` | `deviance_per_molecule`, `loglik_per_molecule` |
| `heldout_zero_probability_molecules` (also `_entries`, `_rows`) | `zero_prob_molecules` |
| `heldout_poisson_deviance_per_molecule_floored_1e-12` | `deviance_floored` |
| `reference_panel__heldout_poisson_deviance_per_molecule`, `reference_panel__heldout_zero_probability_molecules` | `ref500_deviance_per_molecule`, `ref500_zero_prob_molecules` |
| `spatial_topic_moran_mean`, `spatial_hard_topic_neighbor_agreement`, `spatial_W_edge_squared_difference` | `moran_I`, `neighbor_agreement` |
| `topic_entropy_mean`, `top_gene_exclusivity_mean`; `top_features` (top-20 gene symbols per topic) | `topic_entropy`, `top_gene_exclusivity` |

### Run diagnostics tracked per record

| Diagnostic | Where recorded | Why |
|---|---|---|
| Status / failure reason and traceback | `status`, `metadata.exception*` | failure rates are compared explicitly; failed fits are never dropped silently |
| Runtime (s) | `runtime_seconds` (includes the shared spectral time) | practicality for large tissues |
| Selected graph penalty $\lambda$, spectral iterations, CV errors | `metadata.selected_rho`, `spectral_iterations` | detect grid-endpoint selection (over-smoothing) |
| Genes kept by the spectral step | `metadata.retained_feature_count` | how much P1/P3 actually thresholds |
| Hunter convergence (PALM) | `metadata.vertex_hunting.optimizer_converged` | a PALM result can be finite but not converged |
| $A$ convergence, iterations, normalized gap | `metadata.A_recovery.*` | `A_full_L2` / `A_full_Pois` certification |
| Spots kept / dropped, train and test molecules, eligible genes | task JSON `data` | row mask and panel audit |
| Peak memory, elapsed time, config and data hashes, git commit | task JSON `provenance` | reproducibility |

### Which metric answers which ablation question

| Question | Primary metric | Secondary |
|---|---|---|
| A: Poisson vs Euclidean $A$ (same $\hat W$) | paired held-out deviance per molecule; zero-probability molecules | seed stability of $A$ (JSD), entropy/exclusivity, marker genes in top-20, convergence rate, runtime |
| B: vertex hunting | layer ARI / NMI | balanced accuracy, failure rate, runtime, seed stability |
| C1: panel size | layer ARI vs $p$; reference-panel (top-500) deviance vs $p$ | runtime vs $p$, baselines vs $p$ |
| C2: threshold / weighting | layer ARI; genes retained | deviance, $\lambda$ selected |
| GpLSI vs baselines | layer ARI, deviance | Moran's I (diagnostic only), runtime |
| Smoothing sensitivity | selected $\lambda$ and layer ARI, default grid vs wide grid | Moran's I |

## 9. Diagnostics run so far, and results

All diagnostics used section 151673, seed 26090401, $p=2{,}000$ (3,639 spots, 11,164
edges; 13,368 eligible genes), $K=7$, one CPU per job. Several jobs ran at once, so
timings are approximate.

**Smoke run.** Killed after about 35 minutes without output. Cause:
`project_rows_simplex` projected rows in a Python loop, and Topic-SCORE's $W$ refit calls
it for 3,600 rows × up to 5,000 iterations. Fixed (§10).

**Component timings:**

| Component | Time | Notes |
|---|---|---|
| Graph SVD, full penalty grid, 5 fold workers | 57 s per preprocessing | 2 outer iterations; chosen $\lambda=0.0114$, **inside** the grid (top ≈ 0.0165) |
| All six hunters, anchor SPA | ≤ 0.7 s each | all succeeded |
| `A_current` | ≈ 0 s | |
| `A_full_L2` (before speed-up) | 51–71 s | converged, 1,133 iterations; not re-timed after the Gram-form change |
| `A_full_Pois` (defaults) | 365 s / 2,000 iterations | **not converged** |
| Topic-SCORE graph / raw | ~157 s / ~257 s | after the projection fix |
| LDA / KL-NMF / graph KL-NMF | 74 / 13 / 53 s | |
| Spatial-LDA | not measured | stopped; exceeded 15 min in the smoke run |

**Poisson convergence** (tolerance: normalized gap $10^{-8}$):

| Start | Iterations | Time | Normalized gap | Objective |
|---|---|---|---|---|
| `A_current`, interior mass $10^{-6}$ (default) | 500 | 105 s | $1.17\times10^{-2}$ | |
| same | 2,000 | 393 s | $2.34\times10^{-3}$ | relative change $2\times10^{-10}$ per 100 iterations |
| pooled gene frequencies | 300 | 83 s | $4.98\times10^{-4}$ | 6.75442082e7 |
| pooled gene frequencies | 1,500 | 388 s | $8.50\times10^{-6}$ | 6.75442035e7 |
| `A_current`, interior mass 0.01 | 300 | 68 s | $1.83\times10^{-3}$ | 6.75442044e7 |
| `A_current`, interior mass 0.01 | 1,500 | 336 s | $8.29\times10^{-6}$ | 6.75442034e7 |
| `A_current`, interior mass 0.1 | 300 | 69 s | $3.33\times10^{-4}$ | 6.75442055e7 |

Interpretation:
* **Why the default stalls.** `A_current` has 14.6% exact zeros. The solver's
  interiorization, $A\leftarrow(1-\varepsilon)A+\varepsilon/p$ with $\varepsilon=10^{-6}$,
  lifts them only to about $5\times10^{-10}$, and multiplicative EM regrows them
  geometrically slowly. So the gap stays large while the objective is flat.
* **Better starts** are 300–1,000× closer after 1,500 iterations but still above
  $10^{-8}$; the objectives differ by about $10^{-7}$ relative.
* **Dense EM statistics** (BLAS) match the sparse ones to $10^{-16}$ but are only 1.5×
  faster (0.13 vs 0.20 s per iteration), so implementation speed is not the fix.

**No ablation results yet.**

## 10. Code map

New:

| File | Purpose |
|---|---|
| `scripts/visium_dlpfc/prepare_data.py` | download and build the processed H5AD (§1) |
| `src/gplsi_spatial_benchmark/panels.py` | training-only nested gene panels (sparse-aware) |
| `src/gplsi_spatial_benchmark/ablation_runner.py` | `prepare_task_data`, `score_fit`, `run_task`; writes `results/visium_dlpfc/<design>/<task>.json` and `.npz` (W, A, spot IDs, panel genes, coordinates). Separate module, so the Midway `runner.py` is untouched. |
| `configs/visium_dlpfc/ablation.json` | settings and designs (§7) |
| `configs/visium_dlpfc/tasks_*.csv` | smoke 1, core 9, panel 36, lambda_wide 3, extended 360 |
| `scripts/visium_dlpfc/make_tasks.py` | writes the manifests |
| `scripts/visium_dlpfc/run_tasks.py` | resumable local parallel launcher; single-threaded BLAS; logs in `logs/visium_dlpfc/` |
| `scripts/visium_dlpfc/slurm_array.sh` | Slurm array version |
| `scripts/visium_dlpfc/refit_poisson.py` | Poisson $A$ from saved $W$ on rebuilt identical data (checks spot/gene IDs). **Untested; its `poisson_refit` settings (initial, interior_mass, max_iter, tolerance) still need to be added to the config.** |
| `scripts/visium_dlpfc/summarize.py` | tables and figures: ARI heatmap, per-axis marginals, paired A contrasts, panel curves, baselines, seed stability, spatial maps. **Not yet run on real results.** |
| `tests/test_dlpfc_ablation.py` | sparse split conservation and nesting; nested training-only panels; `A_full_L2` wiring; vectorized projection = row loop; Gram-form L2 = residual-form L2 |

Modified:

| File | Change |
|---|---|
| `src/gplsi/real_experiment.py` | `A_full_L2` branch in `recover_A_for_geometry` (warm start = paired `A_current`) |
| `src/gplsi_spatial_benchmark/methods.py` | `fit_method_suite` takes `A_recoveries`, `vertex_parameters`, `competitors`, `recovery_parameters` (previously hard-coded); records hunter convergence and retained-gene count |
| `src/gplsi_spatial_benchmark/splits.py` | `thin_and_split_sparse_counts` |
| `src/gplsi/recovery.py` | speed only: vectorized `project_rows_simplex` (bitwise identical to the row loop, about 70× faster); `refit_A_full_l2` computes its gradient and objective from $\hat W^\top\hat W$, $\hat W^\top X$, $\lVert X\rVert_F^2$ (matches the original to $10^{-9}$). **The Poisson solver is unchanged.** |

Environment (`gplsi-env`): reinstalled `pycvxcluster` (its editable install pointed to a
deleted folder); added `anndata`, `h5py`, `pytest`, `pyarrow`, `psutil`,
`threadpoolctl`, `py-spy`.

## 11. Running

```sh
conda activate gplsi-env
python scripts/visium_dlpfc/prepare_data.py --download
python scripts/visium_dlpfc/make_tasks.py
python scripts/visium_dlpfc/run_tasks.py --tasks configs/visium_dlpfc/tasks_core.csv --workers 6
python scripts/visium_dlpfc/refit_poisson.py --designs core panel lambda_wide --workers 6
python scripts/visium_dlpfc/summarize.py
# Slurm: sbatch --array=0-8 --export=ALL,TASKS=configs/visium_dlpfc/tasks_core.csv scripts/visium_dlpfc/slurm_array.sh
```

Outputs go to `results/visium_dlpfc/` (gitignored); the summary goes to
`results/visium_dlpfc/summary/` (`report.md`, `all_records.csv`, `seed_stability.csv`,
figures).

## 12. Open decisions and plans

Decisions needed:
1. **Spatial-LDA:** drop it from `core` for now (one line in the config)? Graph KL-NMF
   and graph-denoised Topic-SCORE remain as graph-aware baselines.
2. **Poisson settings** for the refit. Proposal: start from pooled frequencies, or
   `A_current` with interior mass 0.01; about 1,500 iterations; report the achieved gap
   and label fits "near-converged" instead of requiring $10^{-8}$. Cost: about 6 min per
   geometry, about 2.5 CPU-hours per core task. Needs Claire's agreement, since it
   changes how Poisson results are labelled.

Then:
1. Smoke run end to end.
2. Core, then lambda_wide, then panel runs locally with about 6 workers (estimated
   15–20 min per core task without Spatial-LDA and Poisson; $p=5{,}000$ tasks slowest).
3. Poisson refit from the saved $W$, overnight.
4. `summarize.py` → figures and tables for the meeting.
5. After the meeting: the extended 360-task grid on Midway; commit on a branch and flag
   the `recovery.py` speed changes to Claire.

Cautions:
* Three donors only — sections and seeds are not independent replicates.
* Br5595 sections have five layers, so $K=7$ over-splits them relative to the labels.
* Deviance on the fitted panel is not comparable across panel sizes; use the
  reference-panel columns.
* The chosen graph penalty was interior on one section; check `selected_rho` and the
  `lambda_wide` design before interpreting smoothing effects.

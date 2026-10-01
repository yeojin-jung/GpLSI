# Real-data experiment protocol (paper)

First full pass on CRC, spleen, What's Cooking and Visium DLPFC. Fixed on
2026-09-30. Thresholding and weighting (P1–P3) come after this pass. Code:
`configs/base.json` (shared), `configs/<dataset>/production.json`, and
`configs/<dataset>/smoke.json` (one-job wiring check of the same protocol).
Claire's 2026-09-15 designs are kept in `configs/handoff/`; the DLPFC
ablation designs are in `configs/dlpfc/ablation/` (`docs/visium_dlpfc_ablation.md`).

## 1. General protocol (all datasets)

### 1.1 Steps

| Step | What | Command |
|---|---|---|
| 0 | Data diagnostics: word-frequency heterogeneity and Tran-threshold curves on the training counts | `scripts/analysis/word_frequency_diagnostics.py CONFIG` |
| 1 | Fit every method on every task (dataset × K × seed [× section]) | `scripts/run_experiment.py CONFIG [--task i]`; Slurm: `scripts/slurm/run_experiment.sh` |
| 2 | Collect rows | `scripts/summarize.py CONFIG` → `results/<name>/fit_rows.csv` |
| 3 | General, A and W metrics against K | `scripts/analysis/plot_metrics_vs_K.py CONFIG` |
| 4 | Cross-seed stability of A | `scripts/analysis/seed_stability.py CONFIG` |
| 5 | Topic compositions (A) | `plot_topic_composition.py` (CRC, spleen: small vocabularies), `plot_top_features.py` (Cooking, DLPFC) |
| 6 | Near-pure documents (W ≥ 0.95) | `scripts/analysis/near_pure_counts.py CONFIG` |
| 7 | Dataset tasks and spatial maps | §2–§5 |

### 1.2 Methods

All methods see the same training counts, graph and K. Labels and outcomes are never passed to a fit.

| Method | Estimator | A | Notes |
|---|---|---|---|
| **GpLSI** | Graph-aligned SVD of the frequency matrix (document geometry `document_U`, no preprocessing `P0_raw`) → SVS* vertex hunting → W | `A_current`, `A_full_L2`, `A_full_Pois` from the same W | One fit row per A estimator; W, spatial metrics and λ are shared |
| pLSI | Truncated SVD of X (no graph) → SPA (Klopp et al.) | same three estimators | |
| Topic-SCORE | Ke & Wang, on raw frequencies | native | |
| Topic-SCORE (graph) | Topic-SCORE on GpLSI's graph-aligned singular vectors (same λ) | native | Runs with GpLSI (shares the spectral fit) |
| LDA | scikit-learn variational LDA, seed = task seed | native | |
| Spatial LDA | Vendored Calico spatial LDA (difference penalty 0.25) | native | Builds its own graph from coordinates (Voronoi, reduced to a spanning tree, per tissue). Cooking has no coordinates: the Jaccard graph's edges are penalized directly |

GpLSI settings (`configs/base.json`):

* **λ (graph penalty):** grid `1e-4 · 1.2^j`, j = 0..28 (top ≈ 0.0165); chosen **once** by 5-fold graph cross-validation (held-out nodes interpolated from their neighbours, all folds scored) on the initialized right singular subspace, then held fixed (`cv_once`); ≤ 50 alternating iterations, tolerance 1e-5; initialization `weighted_debiased_mean_N_approx`. Recorded per fit: `spectral.rho_selected` (λ), `spectral.rho_grid`, `spectral.rho_cv` (CV error per λ and fold), `spectral.iterations`, `spectral.singular_values`. Flag fits whose λ is the top of the grid.
* **SVS\*:** L (number of k-means centers) chosen by SVS\*'s own vertex-stability rule (`svs_star_stability`) at every K. The adaptive MixedSCORE rule borrowed from SVS cannot run at K ≥ 8 (spleen K = 10). The two rules gave the same Cooking topics at K = 4 for two of four preprocessings and similar ones otherwise. Recorded: `geometry.vertex_parameters.L`, the candidate-L scores, vertex condition number.
* **A estimators:** `A_current` = least squares then row-wise projection onto the simplex (original GpLSI); `A_full_L2` = simplex-constrained least squares; `A_full_Pois` = multinomial (Poisson) MLE on training counts by EM from the pooled-frequency start (a warm start from A_current stalls: its exact zeros regrow only geometrically), ≤ 2,000 iterations, converged when the normalized KKT gap ≤ 1e-8; `A_recovery_info.*` records convergence.

### 1.3 Splits, seeds, held-out counts

* **Held-out counts:** each document's counts are split by binomial thinning, 80% train / **20% held out** (`heldout_fraction`). Every method fits the training counts; scores use the held-out counts of the same documents (a document-completion evaluation, not held-out documents).
* **Seeds:** **5 per dataset**: CRC 26090301–05, spleen 26090401–05, Cooking 26090501–05, DLPFC 26090601–05. The seed sets the thinning split, the graph-CV folds and the random starts of LDA, Spatial LDA and k-means. Every method runs every seed.
* **Reporting:** curves show the mean over seeds ± SE (`sd / √n`, n ≥ 2). DLPFC: average over seeds within a section, then show the distribution over sections. Compare methods by their per-seed differences where possible (paired).

### 1.4 Graphs

| Dataset | Nodes | Edges | Weights |
|---|---|---|---|
| CRC | cell-centered 3-hop neighbourhoods | SPACE-GM cell adjacency within each region (`<region>.edge.csv`); no edge crosses regions | `exp(-0.1 · ‖Δx‖²)` on coordinates scaled by the dataset's bounding-box diagonal |
| Spleen | B-cell-centered neighbourhoods | Source cell adjacency within each spleen (block-diagonal across the three mice) | same, φ = 0.1 |
| Cooking | recipes | each recipe links its 5 nearest recipes by binary-set Jaccard similarity; union, undirected | unit |
| DLPFC | Visium spots | symmetric 6-NN within the section (= the hexagonal grid) | `exp(-(d / median 1-NN distance)²)` |

### 1.5 Metrics

Per fit (stored in the row by the pipeline, keys in parentheses):

| Axis | Metric | Notes |
|---|---|---|
| General | Held-out Poisson deviance per held-out count, smoothed (`heldout_metrics.heldout_poisson_deviance_per_molecule_smoothed_1e-03`) | Prediction `N_i^test · p̃_i` with `p̃ = (1 − ε)(W A)_i + ε/p`, ε = 10⁻³; also ε = 10⁻⁴ and 10⁻² (`…_smoothed_1e-04`, `…_1e-02`) to check that rankings do not depend on ε. Smoothing is needed because A_current and Topic-SCORE put exact zeros in A, and one held-out count on a zero-probability word makes the exact deviance (`…_per_molecule`, also kept) infinite. Neither zeroing nor dropping those terms is used: both reward the fits that make impossible predictions, and dropping them per method scores methods on different counts. A zero still costs ≈ log(p/ε) nats per count. Smoothing also caps the cost of tiny positive probabilities, which dominate the exact deviance even without zeros (CRC smoke: GpLSI with Poisson A 1.83 exact vs 1.68 at ε = 10⁻⁴; pLSI with Poisson A 3.56 vs 1.85), so zeros and near-zeros are treated alike. Report next to the share of held-out counts given probability 0 (`heldout_zero_probability_molecules` / `heldout_molecules`) |
| | Fitting time (`runtime.total`, plus `runtime.spectral`, `runtime.vertex_hunting`, `runtime.A_recovery`) | Wall clock, one CPU, BLAS single-threaded |
| A | Topic compositions (`arrays/<fit>.npz: A_hat`) | Plots in step 5 |
| | Max and mean pairwise cosine between topics (`metrics.topic_cosine_max`, `…_mean`) | Max near 1 = duplicated topic (Cao et al., 2009) |
| | Topic diversity (`metrics.topic_diversity_top{3,10,25}`) | Share of unique words among each topic's top t (Dieng, Ruiz & Blei, 2020: t = 25), t capped at p; use top 3 for CRC (p = 8) and spleen (p = 24) |
| | Entropy, exclusivity (`metrics.topic_entropy_mean`, `metrics.top_gene_exclusivity_mean`) | |
| | Cross-seed stability (step 4) | Hungarian-matched cosine between seeds (mean, worst topic) and top-10-word Average Jaccard (Greene et al., 2014), per A estimator |
| W | Topic prevalence (`metrics.topic_prevalence_argmax`, `metrics.topic_prevalence_mean_W`, `metrics.topic_effective_number`, `metrics.empty_topic_count`) | Share of documents whose dominant topic is k, and mean weight |
| | PAS (`metrics.spatial_PAS`) | SpatialPCA definition (Shang & Zhou, 2022): share of points whose dominant topic differs from ≥ 6 of their 10 nearest points (within tissue). Lower = smoother. Coordinates required |
| | CHAOS (`metrics.spatial_CHAOS`) | SpatialPCA definition: mean distance from each point to the nearest point with the same dominant topic, in units of the tissue's median nearest-neighbour spacing. Lower = more compact |
| | Moran's I (`metrics.spatial_topic_moran_mean`) | Graph-weighted, per topic weight, averaged over topics |
| | W roughness (`diagnostics.graph_W_smoothness`), neighbour agreement (`metrics.spatial_hard_topic_neighbor_agreement`) | Graph-weighted; also kept: `diagnostics.historical_1_minus_PAS` (the handoff's PAS on the model graph), comparable to the handoff reports only |
| | Spatial maps of the dominant topic | Step 7 (CRC, spleen, DLPFC) |
| Task | Dataset-specific | §2–§5 |

Smoothness is not quality by itself (a constant W is perfectly smooth): read the W metrics next to held-out deviance.

### 1.6 Figures that show one fit per method

Compositions, maps and near-pure counts use one representative fit per method and K: the smallest successful seed, chosen without looking at the fit. Topics are matched one-to-one to the same-K LDA fit of that seed by the Hungarian assignment maximizing the cosine similarity of full A rows. This is a display correspondence only. Dominant topics are taken in the fit's own topic order (ties go to the lowest topic) before relabelling.

### 1.7 What is saved

Per task, `results/<name>/<task id>/`:

* `rows/<fit id>.json`: one row per method × A estimator, including failures (with traceback) — settings, metrics above, λ and CV curves, convergence, runtimes, warnings;
* `arrays/<fit id>.npz`: **`W_hat` (n × K) and `A_hat` (K × p) for every successful fit of every method** (float64; W is repeated in each GpLSI/pLSI A-estimator row);
* `spectral/P0_raw.npz`: GpLSI's graph-aligned factors (`U_hat`, `V_hat`, singular values);
* `data.npz`: observation ids, feature ids/names, group ids (region / spleen / cuisine / section), coordinates;
* `task.json`: resolved config, data hashes, software versions and git commit.

Rows are cached by data, settings and source-code hash, so reruns skip finished fits.

### 1.8 Compute

Each task runs as three jobs (`parts`): `gplsi` (GpLSI with its three A estimators + Topic-SCORE graph), `baselines` (pLSI, Topic-SCORE, LDA), `spatial_lda` (slowest). Jobs: CRC 60, spleen 75, Cooking 15, DLPFC 180 per vocabulary (360). Tasks are listed part by part (`run_experiment.py CONFIG --list`), so an index range selects one part.

Longest single fits in the handoff runs (full data, one CPU): Spatial LDA 6.3 h on CRC, 4.2 h on spleen, 0.7 h on Cooking; GpLSI 0.3–0.6 h; LDA, pLSI and Topic-SCORE minutes or less. The Slurm script caps a job at 12 h and `TASK_STRIDE` runs several tasks in sequence inside one job, so **submit `spatial_lda` parts separately** with at most one CRC/spleen task per job (or a longer `--time`), and pack the other parts. Smoke timing on one DLPFC section (30% of counts, 500 genes): Spatial LDA 71 s, LDA 28 s, GpLSI 12–15 s.

### 1.9 Run order (first pass)

Task index ranges per part (`run_experiment.py CONFIG --list`):

| Config | `gplsi` | `baselines` | `spatial_lda` |
|---|---|---|---|
| `dlpfc/production.json` and `dlpfc/production_tran.json` (same ranges) | K7 0–39, K5 120–139 | K7 40–79, K5 140–159 | K7 80–119, K5 160–179 |
| `cook/production.json` | 0–4 | 5–9 | 10–14 |
| `crc/production.json` | 0–19 | 20–39 | 40–59 |
| `spleen/production.json` | 0–24 | 25–49 | 50–74 |

Within a part, tasks run seed by seed (DLPFC: all sections of seed 1 first).
Submit a range packed into at most 10 jobs with
`sbatch --array=0-9 --export=ALL,CONFIG=<config>,TASK_FIRST=a,TASK_LAST=b,TASK_STRIDE=10 scripts/slurm/run_experiment.sh`.
Finished fits are cached, so resubmitting a range only runs what is missing or failed.

0. **Setup on the cluster:** pull, build the data (`prepare_dlpfc.py --download`, `prepare_cook_v2.py`, spleen compartment labels), `pytest`, then each `configs/<dataset>/smoke.json` and `configs/dlpfc/smoke_tran.json` (one job each).
1. **DLPFC pilot, seed 1 of every section, GpLSI part, both vocabularies:** tasks 0–7 and 120–123 of `production.json` and of `production_tran.json`. Check `spectral.rho_selected` is below the grid top (≈ 0.0165), `A_recovery_info.converged` for A_full_Pois, runtimes; `dlpfc/plot_production.py` on each pilot.
2. **DLPFC, everything else:** 8–39 and 124–139, then the baselines and Spatial LDA ranges, for both configs (sections are small, so all parts can be packed: `TASK_FIRST=0,TASK_LAST=179`).
3. **Cooking:** all 15 tasks (small; its results drive the thresholding decision).
4. **CRC and spleen GpLSI + baselines:** CRC 0–39, spleen 0–49 (packed). Spleen K = 10 is the first run of SVS\* stability-L at that K: look at it first.
5. **CRC and spleen Spatial LDA:** CRC 40–59, spleen 50–74, about one task per job (up to 6.3 h / 4.2 h each, ≈ 220 job-hours in total, ≈ a day at 10 concurrent jobs). Start them as soon as slots free up: they are the long pole.

## 2. CRC (Stanford colorectal cancer CODEX)

* **Documents:** 113,561 cell-centered 3-hop neighbourhoods (focal tumor cells with ≥ 10 neighbours) from 196 tissue regions of 109 patients. **Words:** p = 8 non-tumor cell types (CD4 T, CD8 T, B, macrophage, granulocyte, blood vessel, stroma, other). Mean training length ≈ 14 counts. Graph: 185,922 within-region edges.
* **K** = 2, 3, 4, 5. Tasks: 4 K × 5 seeds (× 3 parts).
* **Diagnostics:** heterogeneity is mild (max/min mean frequency 5.5); Tran keeps all 8 words at every α ≤ 0.1, so thresholding does not apply.
* **Downstream task — recurrence and primary outcome from W** (`scripts/analysis/crc/evaluate_crc_patient_outcomes.py --fit-table results/crc_production/fit_rows.csv`): patients pool all their regions (patient = prefix of `sample_label_visualizer`; 109 patients for primary outcome, 68/41; 103 for recurrence, 72/31). Predictors: patient mean of per-cell ILR(W) (also mean W and log argmax shares). Ridge logistic (C = 1) and random forest (100 trees), 5 × 5 repeated stratified patient-level CV (seed 260908). Scores: ROC AUC, F1, balanced accuracy, PR AUC, Brier; mean ± split-repeat SE. Reference: the same classifiers on directly observed patient cell-type proportions. W was fit on all patients without outcomes (transductive). Outcome codes 0/1 are kept as supplied.
* **Interpretation:** `crc/plot_outcome_composition.py` (patient-pooled W by outcome group; tissue maps, 2 regions per class nearest the median size, with the focal tumor-cell type row); `crc/tumor_phenotype_alignment.py` (topics against the 7 focal tumor-cell phenotypes: mean W, dominant-topic shares, patient Spearman, ARI/NMI/AMI).

## 3. Spleen (mouse spleen CODEX, joint model)

* **Documents:** 100,840 B-cell-centered neighbourhoods from three spleens (BALBc-1/2/3: 35,271 / 33,492 / 32,077), one joint model with a block-diagonal graph (302,393 edges). **Words:** p = 24 neighbourhood cell types. Mean training length ≈ 9.
* **K** = 3, 4, 5, 7, 10. Tasks: 5 K × 5 seeds (× 3 parts).
* **Diagnostics:** strongly heterogeneous (max/min mean frequency ≈ 7,200; Gini 0.68), but Tran at α = 0.005 keeps all 24 words. Weighting (Ke), not thresholding, is the relevant follow-up.
* **Downstream task — manual compartments** (`spleen/compartment_comparison.py`; labels: CytoCommunity B-zone / marginal zone / PALS / red pulp, 100,583 of 100,840 labelled, built by `scripts/data/prepare_spleen_compartment_annotations.py`): ARI and AMI of dominant topics; at K = 4 the Hungarian exact-match accuracy; leave-one-spleen-out ridge probe on √W (balanced accuracy, macro-F1). Maps of BALBc-1 with the reference panel; compartment shares against dominant-topic shares.

## 4. What's Cooking (v2)

* **Documents:** 19,017 recipes with ≥ 8 ingredients from 20 cuisines. **Words:** p = 4,911 ingredients (344 have no training count after thinning). Mean training length ≈ 10. Graph: 79,570 Jaccard edges. No coordinates: no PAS, CHAOS or maps.
* **K** = 7. Tasks: 5 seeds (× 3 parts).
* **Diagnostics:** extremely heterogeneous (Gini 0.92; the top 10% of ingredients hold 89% of the counts). Tran keeps 1,333 ingredients at α = 0.005, 889 at 0.01, 181 at 0.1. This is the dataset where thresholding matters.
* **Downstream task — topic proportions by cuisine** (`cook/plot_cuisine_composition.py`): mean W per cuisine (recipes weighted equally), cuisines ordered by Jensen–Shannon similarity of their profiles; top-10 ingredients per topic (`plot_top_features.py`).

## 5. Visium DLPFC (spatialLIBD)

* **Documents:** spots of 12 sections (47,681 in total; 3,460–4,789 per section), one model per section. Graph: 6-NN hexagonal grid within the section.
* **Words — two vocabularies, both chosen per section and seed on the training counts only, run as two configs with the same seeds (identical count splits, so they compare seed by seed):**
  * `dlpfc/production.json` — **dispersion panel:** the top 2,000 genes by variance/mean among genes detected in ≥ 1% of spots; spots with no training count on the top-500 genes are dropped, and those 500 also give a reference-panel score. Mean training length 1,150–3,190 UMIs; holds 66–71% of the counts.
  * `dlpfc/production_tran.json` — **Tran threshold, α = 0.1, on all 33,538 genes:** keep gene j when η̂_j ≥ 0.1·√(log max(n, p) / (n N̄)) (cut ≈ 0.8–1.2 × 10⁻⁴), the paper rule without the top-10% fallback (the fallback would trigger here and replace the cut by the top 3,354 genes). Keeps 1,273–2,299 genes holding 67–73% of the counts; 1,021–1,625 of them are also in the dispersion panel. Spots need a training count on the vocabulary. Implementation: `gplsi.pipeline.panels.tran_vocabulary`.
  * Held-out deviance is computed on different vocabularies and is **not comparable between the two**; compare them by layer ARI/NMI, the spatial metrics and A stability.
* **K** = 7 on the eight sections with layers L1–L6 + WM (151507–151510, 151673–151676); **K = 5** on the four Br5595 sections (151669–151672), which only have L3–L6 + WM. Tasks: 12 sections × 5 seeds (× 3 parts) per vocabulary.
* **Diagnostics** (`word_frequency_diagnostics.py configs/dlpfc/production.json [--all-genes] [--alpha a]`): all genes are extremely heterogeneous (max/min ≈ 10⁶, Gini ≈ 0.9; about 12,000 genes have no training count). On all genes Tran keeps 11,300–12,900 genes at α = 0.005, 9,000–11,000 at 0.01 and 1,300–2,300 at 0.1. Within the dispersion panel Tran at α = 0.005 removes almost nothing (≥ 1,993 of 2,000 kept).
* **Downstream task — layer recovery:** ARI and NMI of dominant topics against `layer_guess_reordered` on labelled spots (`metrics.external__layer_guess_reordered__ari`, `…__nmi`; plus a cross-validated logistic probe, `…__cv_balanced_accuracy`). `dlpfc/plot_production.py`: per-section ARI/NMI per method; maps of every section next to the manual layers.

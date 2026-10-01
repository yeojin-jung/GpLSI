# MERFISH and Xenium: data processing and experiment notes

Added 2026-10-01. The shared protocol is in [protocol.md](protocol.md) (§1 general,
§6 MERFISH, §7 Xenium); this note records how the two datasets were obtained
and processed, what is specific to their experiments, and how they differ from
the original benchmark (`docs/GPLSI_experiments.pdf`, Claire Donnat's
spatial-transcriptomics benchmark note of 2026-09-10, whose preparation code is
not available to us; everything below was rebuilt from the public sources).

## 0. What was built

| Piece | File | What it does |
|---|---|---|
| MERFISH data | `scripts/data/prepare_merfish.py` | Downloads `MERFISH_Data.h5ad` in 8 parallel byte ranges, checks its SHA-256, keeps the raw counts, renames the columns, checks the published dimensions, writes `data/merfish/merfish_trem2_5xfad.h5ad` (§2.2). |
| Xenium data | `scripts/data/prepare_xenium.py` | Downloads the annotated Dataset 1 object, checks its MD5, keeps the raw counts of the 581,967 assigned cells, renames the columns, **builds the cell-type neighbourhoods** (the only spatial reference label; §3.2), checks the published dimensions, writes `data/xenium/xenium_uc.h5ad`. |
| Loader | `src/gplsi/pipeline/datasets.py` | One loader for every processed spatial H5AD (`dlpfc`, `merfish`, `xenium`): reads one fitting unit, builds its 6-NN graph within each stratum, scores only `evaluation_label_columns`; vocabulary `all` for targeted panels. Evaluation labels are now part of each fit's cache key, so changing a label recomputes the scores instead of reusing stale rows. |
| Grid axis | `src/gplsi/pipeline/config.py` | `unit` axis (animal / patient-timepoint) next to DLPFC's `section`; task ids such as `xenium__HS44_PRE_VDZ_R__K12__seed26090801`. |
| Configs | `configs/{merfish,xenium}/{production,smoke}.json` | K = 12, five seeds, all genes, 50-value λ grid, the protocol's methods and parts (§4). |
| Label recovery | `scripts/analysis/plot_label_recovery.py` | ARI, NMI and the CV classifier per label, per unit; any dataset with evaluation labels (§4.1). |
| Maps | `scripts/analysis/plot_unit_maps.py` | Dominant-topic maps next to a reference label, per unit and stratum, for DLPFC, MERFISH and Xenium (§4.4). |
| Plaque task | `scripts/analysis/merfish/plaque_proximity.py` | §4.2. |
| Disease task | `scripts/analysis/xenium/disease_classification.py` | §4.3; cross-unit topic matching in `shared.consensus_alignment`. |
| Tests | `tests/test_protocol.py` | Configs, consensus alignment, neighbourhood construction, label cache key. |

## 1. What is shared with DLPFC

Both datasets use the same machinery as Visium DLPFC: a processed H5AD per
dataset, one topic model per **fitting unit**, a spatial graph built inside
**graph strata**, and annotations kept out of every fit and used only for
evaluation. The processed file declares, in `uns`:

| Key | MERFISH | Xenium | DLPFC |
|---|---|---|---|
| `benchmark_unit_column` (one model per value) | `unit` = animal (15) | `unit` = patient × condition/timepoint (25) | `sample_id` = section (12) |
| `graph_unit_column` (no edge crosses it) | `stratum` = half-section (19) | `stratum` = slide/core (114) | `sample_id` |
| `evaluation_label_columns` (scored) | `cell_type_coarse`, `cell_type`, `region_coarse`, `region` | `compartment`, `cell_type_coarse`, `cell_type_fine`, `neighborhood` | `layer_guess_reordered` |
| `fit_forbidden_obs_columns` (never reach a fit) | the four labels + `genotype`, `plaque_distance`, `leiden` | the four labels + `leiden_annotation`, `condition`, `patient` | `layer_guess_reordered` |

`X` holds integer counts (CSR, int32), `obsm["spatial"]` the coordinates, and
`uns["source"]` the source URL and checksum. The pipeline loads one unit at a
time (`gplsi.pipeline.datasets._prepare_dlpfc`, dispatched for `dlpfc`,
`merfish` and `xenium`); the configs select units with the `unit` grid axis.

## 2. MERFISH: TREM2-R47H/5xFAD mouse brain

### 2.1 Source

* Johnston, Berackey, Tran, et al., *Molecular Psychiatry* 30:461–477 (2025),
  doi:10.1038/s41380-024-02651-0.
* Brain Image Library dataset **ace-ear-nap** (submission b72faf9d87d7fc00),
  "MERSCOPE Imaging of WT, Trem2-R47H, 5xFAD and Trem2-R47H; 5xFAD samples",
  licence CC BY-SA 4.0. File:
  `https://download.brainimagelibrary.org/b7/2f/b72faf9d87d7fc00/repository/Data_Repository/MERFISH_Data.h5ad`,
  5,483,600,151 bytes. The library publishes no checksum; the script checks the
  SHA-256 of the copy downloaded on 2026-10-01,
  `d6876020cfdb5cb2fdeb848550e8334cfc2684b6ce6f7755b862cac567f06a38`.
* Not used: the per-run folders (raw transcript locations, DAPI images,
  segmentation) and `Plaques_Object.h5ad` (plaque distance is already a column
  of `MERFISH_Data.h5ad`).
* The server serves about 0.2 MB/s per connection; `prepare_merfish.py
  --download` fetches 8 byte ranges in parallel (about 20 minutes instead of
  about 8 hours) and checks size and SHA-256 before use.

### 2.2 Processing (`scripts/data/prepare_merfish.py`)

1. Read `layers/RNA`, the integer molecule counts (stored dense, float64), in
   blocks of 50,000 cells; check every entry is a nonnegative integer; store as
   int32 CSR. The source `X` and `layers/Normalized_Expression` are dropped.
2. Keep all cells and genes: **432,794 cells × 300 genes** (targeted panel),
   30,608,835 nonzero entries, 139,256,236 molecules (mean 322 per cell). No
   cell or gene filter is applied (the pipeline drops cells with no training
   count, per task).
3. Coordinates: `center_x`, `center_y` of the source `obs` (the source
   `obsm["spatial"]` differs from them and is not used). Each section has its
   own coordinate frame; graphs never cross sections.
4. Columns kept and renamed:

| Source | Processed | Meaning | Values |
|---|---|---|---|
| `gen` | `unit` | animal (fitting unit) | 15 |
| `gen_fine` | `stratum` | coronal half-section (graph boundary) | 19 |
| `gen_coarse` | `genotype` | WT, 5xFAD, Trem2 (R47H), Trem2_5xFAD | 4 |
| `cluster_coarse` | `cell_type_coarse` | AST, Amygdala ExN, Cortical ExN, Cortical Inh, Hippocampal ExN, Mic, OGC, Other Glia, Subcortical Neurons | 9 |
| `cluster` | `cell_type` | detailed cell types | 37 |
| `region_labels_coarse` | `region_coarse` | Amygdala, CA1, CA3, Choroid Plexus, DG, Hypothalamus, Lower Cortex, Midbrain, Striatum/Fiber, Thalamus, Upper Cortex | 11 |
| `region_labels` | `region` | detailed anatomical regions | 17 |
| `plaque_distance` | `plaque_distance` | distance to the nearest amyloid plaque, unit undocumented | continuous |
| `leiden` | `leiden` | released clusters (not scored) | 22 |

5. Checks that fail the build: the six published dimensions (cells, genes,
   animals, sections, nonzeros, molecules, from the benchmark note §3.2);
   genotype constant within an animal; every section inside one animal.
6. Output `data/merfish/merfish_trem2_5xfad.h5ad` (85 MB) and
   `data/merfish/summary.json`.

### 2.3 Units

| Genotype | Animals (cells) | Sections |
|---|---|---|
| WT | WT1 (45,321), WT3 (17,863), WT5 (25,307) | 4 (WT1 has two) |
| 5xFAD | 5xFAD1 (49,405), 5xFAD3 (22,965), 5xFAD4 (24,087), 5xFAD5 (19,034) | 5 (5xFAD1 has two) |
| Trem2 | Trem2_1 (42,814), Trem2_3 (22,539), Trem2_4 (26,716), Trem2_5 (23,223) | 5 (Trem2_1 has two) |
| Trem2_5xFAD | Trem2_5xFAD1 (51,339), Trem2_5xFAD3 (22,868), Trem2_5xFAD4 (20,719), Trem2_5xFAD5 (18,594) | 5 (Trem2_5xFAD1 has two) |

Sections hold 14,282–28,532 cells. An animal with two sections is one model
(one A); its graph has no edge between the two sections. Animal numbering has
gaps (no WT2, 5xFAD2, ...): the second sections carry those names in
`gen_fine`.

### 2.4 Specificities

* **Documents are single cells** (the cell's own counts), unlike CRC and spleen
  where documents are neighbourhoods. Topics are expected to resemble cell
  types or states; spatial structure enters through the graph.
* **Plaque distance.** In the 8 animals carrying 5xFAD it ranges from 0 to
  770–2,121 (medians 85–295 by animal). In WT and Trem2-only animals, which
  have no plaques, it is mostly in the thousands (medians about 3,150–3,260; range 8–6,600),
  which is not a distance to a plaque. It is therefore used only in the 8 5xFAD animals, only
  as a within-animal rank (the unit is not documented), and it is not passed
  to the pipeline's generic label metrics.
* **Genotype is constant within a fit**; it cannot be scored per unit and is
  for cross-animal summaries only.
* **Biological replicates are the 15 animals** (3–4 per genotype), not
  sections or cells.

## 3. Xenium: ulcerative colitis

### 3.1 Source

* Mennillo, Lotstein, Lee, et al., *Journal of Clinical Investigation*
  136(14):e202488 (2026), doi:10.1172/JCI202488.
* Figshare article **27327813**, version 1 ("Xenium datasets"), CC BY 4.0. File
  `25_11_22_Xenium_Dataset1_290_IntReps1and2_Annotated.h5ad`
  (`https://ndownloader.figshare.com/files/59799746`, 1.52 GB, MD5
  `419cab98a3be238d310db6e3fc92bc42`, checked).
* Dataset 1 is the 290-gene custom panel on FFPE colon mucosal biopsies,
  integrated replicates 1 and 2. Not used: Datasets 2 (5K panel) and 3 (480
  panel), the raw objects (the raw Dataset 1 object holds 858,896 cells; we use
  the released annotated/QC subset of 582,188 with its raw-count layer, as the
  benchmark did).

### 3.2 Processing (`scripts/data/prepare_xenium.py`)

1. Read `layers/raw_counts` (CSR, float32 with integer values); check
   nonnegative integers; store as int32 CSR. `layers/log_normalized_counts`,
   the PCA/UMAP embeddings and neighbour graphs are dropped.
2. Drop the 221 cells whose unit is `unassigned`: **581,967 cells × 290
   genes**, 28,331,111 nonzero entries, 82,576,573 molecules (mean 142 per cell).
3. Coordinates: `x_centroid`, `y_centroid` in µm (identical to the source
   `obsm["spatial"]`). Cores on one slide share a coordinate frame; graphs are
   built within each slide/core, so overlap between cores does not matter.
4. Columns kept and renamed:

| Source | Processed | Meaning | Values |
|---|---|---|---|
| `24_01_17_HS` | `unit` | patient × condition/timepoint (fitting unit) | 25 |
| `Patient_ID_cores_combined` | `stratum` | slide/core (graph boundary) | 114 |
| `24_01_17_Condition` | `condition` | HC, PRE_VDZ_R, PRE_VDZ_NR, POST_VDZ_R, POST_VDZ_NR | 5 |
| (prefix of `unit`) | `patient` | HS31–HS50 | 20 |
| `25_06_11_Compartments` | `compartment` | broad cell class, not a tissue region: Endothelial, Enteric_glia, IEC (epithelial), Immune, Stromal; each coarse cell type belongs to exactly one | 5 |
| `25_06_11_Common_Coarse_Xenium_Combinedv3` | `cell_type_coarse` | B, Cytotoxic, Endothelial, Enteric_glia, Fibroblast, IEC, MAST, MNP, Myofibroblast, Neutrophil, Pericyte, Plasma, T | 13 |
| `24_05_29_Fine_annotations_Xenium_combined` | `cell_type_fine` | fine cell types | 34 |
| `24_01_08_EM_combined_from_leiden` | `leiden_annotation` | Leiden-derived clusters (not scored) | 26 |
| (built here, step 5) | `neighborhood` | cellular neighbourhood of the cell (spatial reference label) | 8 |

   The patient is the text before the first underscore of the unit name
   (`HS46__PRE_VDZ_NR` has a double underscore in the source and is kept as is).
5. **Cellular neighbourhoods** (`cellular_neighborhoods`), the spatial
   reference label. The release has no tissue regions (§3.4), so we build the
   standard CODEX "cellular neighbourhoods" (Schürch et al., *Cell* 2020) from
   the authors' cell-type annotations:
   1. for every cell, take its 10 nearest cells (itself included) in the same
      slide/core, by Euclidean distance on the centroids;
   2. its feature vector is the share of each of the 13 coarse cell types
      among those 10 cells;
   3. k-means with 8 clusters (the authors' number of CellCharter
      neighbourhoods), on all 581,967 cells jointly so a neighbourhood means
      the same in every unit; seed 0, 10 restarts;
   4. clusters are numbered by size (N1 largest); their mean compositions are
      stored in `uns["neighborhood_composition"]` and `summary.json`.

   Window size, cluster count and seed were fixed before any topic model was
   fitted (constants `NEIGHBORHOOD_WINDOW`, `NEIGHBORHOOD_COUNT`,
   `NEIGHBORHOOD_SEED`); a rebuild reproduces the labels exactly. Only the
   annotations are used, never expression, so the label shares no input with
   the topic models. The resulting neighbourhoods:

| | Cells | Defining mix (≥ 8% of the window) | Reading |
|---|---|---|---|
| N1 | 166,756 (28.7%) | epithelial (IEC) 85% | epithelium |
| N2 | 97,477 (16.7%) | IEC 48%, fibroblast 14%, T 12% | epithelium–lamina propria border |
| N3 | 97,302 (16.7%) | fibroblast 30%, endothelial 20% | stroma and vessels |
| N4 | 58,844 (10.1%) | plasma 50%, fibroblast 11%, T 8% | plasma-cell-rich lamina propria |
| N5 | 52,601 (9.0%) | T 47%, B 14%, fibroblast 10% | T-cell zone |
| N6 | 45,424 (7.8%) | MNP 41%, fibroblast 10% | myeloid-rich |
| N7 | 38,735 (6.7%) | B 58%, T 16% | B-cell aggregates (lymphoid follicles) |
| N8 | 24,828 (4.3%) | myofibroblast 55%, fibroblast 13%, endothelial 11% | myofibroblast layer |

   The "reading" column is our interpretation of the compositions.
6. Checks that fail the build: the Figshare MD5; cells, genes, units, strata
   and patients equal the benchmark note's values (§3.3); one condition per unit.
7. Output `data/xenium/xenium_uc.h5ad` (72 MB) and `data/xenium/summary.json`.

### 3.3 Units

| Condition | Units (cells) |
|---|---|
| Healthy control (HC), 9 | HS31 (7,539), HS33 (19,334), HS35 (2,315), HS37 (29,926), HS39 (19,835), HS40 (9,704), HS41 (30,117), HS42 (16,067), HS43 (29,058) |
| Pre-vedolizumab responder, 5 | HS32 (20,474), HS34 (35,693), HS36 (29,959), HS38 (42,887), HS44 (63,487) |
| Pre-vedolizumab non-responder, 3 | HS46 (32,251), HS48 (837), HS50 (11,835) |
| Post-vedolizumab responder, 3 | HS32 (41,777), HS34 (28,689), HS36 (35,140) |
| Post-vedolizumab non-responder, 5 | HS45 (36,226), HS47 (8,973), HS48 (1,977), HS49 (18,745), HS50 (9,122) |

Five patients (HS32, HS34, HS36, HS48, HS50) have a pre and a post unit; the
other 15 contribute one unit. Units span 1–9 slides/cores; cores hold 87–15,583
cells. The smallest units (HS48 pre: 837 cells, one core; HS48 post: 1,977) are
fitted like the others but give noisy per-unit scores.

### 3.4 Specificities

* **Documents are single cells**, as in MERFISH.
* **The released labels are cell classes, not regions.** All three Xenium
  annotations are per-cell classes at nested resolutions (5 compartments ⊃ 13
  coarse ⊃ 34 fine cell types; e.g. Immune = T, B, Plasma, Cytotoxic, MNP,
  Neutrophil, MAST; Stromal = Fibroblast, Myofibroblast, Pericyte).
  "Compartment" here is a cell class, unlike the spleen compartments (B-zone,
  PALS, ...), which are tissue regions, and unlike MERFISH's anatomical
  regions. Recovering them tests whether topics separate cell classes.
* **The spatial reference is built here** (§3.2 step 5). The authors drew no
  tissue regions, and their 8 CellCharter neighbourhoods (Datasets 1 and 3
  combined, Harmony expression PCA, 3-hop Delaunay aggregation, k chosen from
  a stability scan; `UCSF-DSCOLAB/spatial_transcriptomics_colitis_analysis_2024`)
  were saved only to a private path. Rerunning that recipe would need Dataset 3
  and a stochastic fit, and would itself be an unsupervised clustering of
  expression and space, i.e. a competing method rather than a reference. Our
  cell-type neighbourhoods use only the annotations. Because documents are
  single cells, a cell's neighbourhood depends on its neighbours rather than
  on itself, which is what the spatial graph should help with; they still
  partly reflect the cell's own type (it is one of the 10 cells in its window).
* **Condition and patient are constant within a fit**; they are used only in
  the cross-unit task.
* **Biological replicates are the 20 patients**; units of the same patient
  (pre/post) are not independent.
* The Leiden-derived annotation is kept but not scored: it was derived from
  the same cells' expression, so agreement with it is close to circular.

## 4. Experiment settings specific to MERFISH and Xenium

Everything not listed follows protocol §1 (methods, three A estimators, 20%
count thinning, cv_once, SVS\* with stability L, metrics, parts).

| Setting | Value | Why |
|---|---|---|
| K | 12 | The benchmark's primary resolution for both platforms. |
| Vocabulary | all genes (`vocabulary: "all"`), no thresholding (decided 2026-10-01) | Targeted panels (300 / 290 genes); even at α = 0.1 Tran would drop only 0–49 MERFISH and 14–87 Xenium genes per unit (< 1% and 0.4–5.4% of the counts). |
| λ grid | `1e-4 · 1.2^j`, j = 0..49 (`spectral.grid_len = 50`, top ≈ 0.76) | The benchmark selected the top of the 29-value grid (≈ 0.0165) in every MERFISH and Xenium fit. Other datasets keep 29 values. |
| Graph | symmetric 6-NN within each stratum, weight `exp(-(d / median 1-NN distance in the stratum)²)` | Same as DLPFC and the benchmark. |
| Seeds | MERFISH 26090701–05, Xenium 26090801–05 | The seed sets the count split, graph-CV folds, and random starts. |
| Tasks | MERFISH 15 × 5 × 3 parts = 225; Xenium 25 × 5 × 3 = 375 | Part ranges: MERFISH gplsi 0–74, baselines 75–149, spatial_lda 150–224; Xenium 0–124, 125–249, 250–374. |
| Smoke | `configs/merfish/smoke.json` (animal 5xFAD5, 19,034 cells; exercises the plaque task), `configs/xenium/smoke.json` (unit HS35_HC, 2,315 cells) | One seed, one λ, one graph-SVD iteration, all methods in one job. |

### 4.1 Label recovery (task 2 of both datasets)

Two kinds of label are scored. **Spatial domains:** MERFISH's anatomical
regions and Xenium's cell-type neighbourhoods. **Cell classes:** MERFISH's
cell types and Xenium's compartments and cell types (§3.4).

Computed by the pipeline for each fit (`metrics.external__<label>__*`), on the
unit's cells:

* **ARI and NMI** between the dominant topic (argmax W) and the label.
* **CV balanced accuracy**: class-weighted multinomial logistic regression from
  W, stratified K-fold CV with K = min(5, smallest class size); not computed
  when a class has fewer than 2 cells in the unit (common for the 34–37 fine
  types in small units).
* A label is scored only when it has between 2 and min(100, n/3) classes in
  the unit.

`scripts/analysis/plot_label_recovery.py` averages each metric over seeds within
a unit and shows the distribution over units per method (units, not cells,
are the replicates).

### 4.2 MERFISH task 1: plaque proximity (`scripts/analysis/merfish/plaque_proximity.py`)

* Animals: the 8 carrying 5xFAD (5xFAD1, 3, 4, 5; Trem2_5xFAD1, 3, 4, 5).
* Target: each cell's plaque distance as a within-animal rank in [0, 1].
* Predictor: the cell's W (K = 12), standardized in the training folds; ridge
  regression (α = 1).
* Cross-validation: 5 spatial blocks per section (k-means on the section's
  coordinates); fold j holds out block j of every section. Random folds would
  leak, because neighbouring cells share their plaque distance.
* Scores on all held-out predictions: Spearman between predicted and true
  rank; AUC for the 10% of the animal's cells nearest a plaque, with the
  prediction as the score.
* References with the same blocks: one-hot annotated coarse cell type (uses
  the labels); all 300 genes' log-normalized expression (`log1p(1e4 · x)`; no
  model).
* Summary: mean over seeds within an animal, then mean ± SE over the 8
  animals.

### 4.3 Xenium task 1: healthy vs ulcerative colitis (`scripts/analysis/xenium/disease_classification.py`)

* Topic matching across units: every unit is fitted separately, so for each
  method and seed the 25 units' topics are matched to a consensus
  (`shared.consensus_alignment`): start from the first unit's A, match every
  unit's A rows to the reference by cosine (Hungarian), replace the reference
  by the mean of the matched profiles, and repeat until the matching is
  stable. The 290-gene panel is common to all units.
* Predictor per unit: mean W over its cells in the consensus topic order, in
  Helmert ILR coordinates (11 dimensions).
* Classes: the 9 HC units vs the 8 pre-treatment UC units (PRE_VDZ_R and
  PRE_VDZ_NR), one unit per patient, so leave-one-out is leave-one-patient-out.
  Post-treatment units are left out (they repeat patients and mix treatment
  with disease).
* Classifier: training-fold standardization + L2 logistic regression (C = 1,
  liblinear). Scores per seed: ROC AUC and balanced accuracy (threshold 0.5)
  of the 17 held-out probabilities; summary mean ± SE over seeds.
* References with the same classifier: annotated coarse cell-type proportions
  per unit (ILR; uses the labels) and mean gene frequencies per unit (log; no
  model).
* Also written: each unit's consensus-aligned mean W, as a table and a figure
  ordered by condition (all 25 units, descriptive).
* Caveats: 17 units, so the AUC is exploratory; W is fitted on all cells
  without labels (transductive); the consensus matching is for summary only.

### 4.4 Maps (`scripts/analysis/plot_unit_maps.py`)

For each unit, one fit per method (smallest successful seed, or `--seed`).
Each cell is coloured by its dominant topic; a topic takes the colour of the
reference class it overlaps most under a one-to-one Hungarian matching over
the unit's cells (display only; unmatched topics grey), so a method that
recovers the reference looks like the reference panel. Default references:
MERFISH `region_coarse`, Xenium `neighborhood`, DLPFC
`layer_guess_reordered` (`--label` picks another, e.g. `cell_type_coarse`).
Coordinates are comparable only within a stratum, so each figure row is one
section or core: the unit's largest by default, all with `--all-strata`.
Output: `<run>/figures/maps/<unit>.png`.

## 5. Diagnostics and checks run on 2026-10-01

* **Word frequencies** (`word_frequency_diagnostics.py`, training counts of the
  first seed, per unit): MERFISH mean length 132–357 molecules, max/min word
  frequency 680–2,700, Gini 0.67–0.78, Tran keeps 300/300 genes at α = 0.005
  and 251–300 at α = 0.1. Xenium mean length 64–163, max/min 600–17,500,
  Gini 0.54–0.77, Tran keeps 281–290/290 at α = 0.005 and 203–276 at α = 0.1.
* **Smoke runs:** every method finished on both datasets. MERFISH (19,034
  cells): 637 s in total; GpLSI 142 s, of which SVS\*'s L selection 109 s; Spatial
  LDA 371 s; LDA 37 s; graph Topic-SCORE 41 s. Xenium (2,315 cells): 84 s.
* **Classifier check** (17 Xenium units, smoke settings, one seed, GpLSI and LDA
  only): leave-one-patient-out AUC 0.83 (GpLSI) and 0.71 (LDA); references 0.94
  (annotated cell types) and 0.86 (mean gene frequencies). Wiring check only.
* **Plaque check** (one animal, smoke settings): Spearman 0.06–0.17 for the topic
  models, 0.28 for all genes. Wiring check only.
* **Neighbourhood label** (Xenium smoke unit HS35_HC): scored for every method
  (ARI 0.04–0.13 at smoke settings). In this small healthy unit, N7 (B-cell
  aggregates) has 1 cell, so the CV classifier is skipped for neighbourhoods
  there, as §4.1 describes.
* **Maps** rendered for the MERFISH, Xenium and DLPFC smoke runs.
* **Compute guidance:** at K = 12, SVS\*'s L selection takes about 1 minute at
  7,500 cells and 4 minutes at 36,000 cells even with smoke settings; Spatial
  LDA at production settings will take hours on the largest units (51,339 and
  63,487 cells). Run the `spatial_lda` part about one task per job.

## 6. Differences from the original benchmark (`GPLSI_experiments.pdf`)

| | Original benchmark | Here |
|---|---|---|
| Fitting unit, graph strata, gene panels, counts | per animal / patient-timepoint; `gen_fine` / `Patient_ID_cores_combined`; all panel genes; raw counts | same |
| K | 8, 12, 16 (primary 12) | 12 |
| Count retention | r = 1, 0.75, 0.5, 0.25 at the primary K | r = 1 only |
| λ selection | graph CV scored on the legacy first three folds, "current" initialization, 29-value grid (top selected in every fit) | CV once on all 5 folds, debiased initialization, 50-value grid |
| GpLSI variants | 4 preprocessings × 6 hunters + anchor-feature GpLSI, A_current and A_full_Pois | P0 × SVS\*, A_current, A_full_L2, A_full_Pois |
| Poisson A | preliminary artifacts invalid (optimizer defect) | corrected EM with a convergence certificate, pooled start |
| Baselines | Topic-SCORE, graph Topic-SCORE, LDA, KL-NMF, Spatial LDA, graph KL-NMF | pLSI, Topic-SCORE, graph Topic-SCORE, LDA, Spatial LDA |
| Held-out deviance | floored at 1e-12 | exact (kept), ε = 1e-3 uniform-smoothed (reported), zero-probability share |
| Downstream | per-unit label NMI/ARI/CV accuracy, plaque Spearman | same label metrics plus Xenium cell-type neighbourhoods; plaque prediction with spatial-block CV; HC vs UC classification; dominant-topic maps |
| Storage | float32 W, A | float64 W, A for every method |

## 7. Reproduce

```sh
python scripts/data/prepare_merfish.py --download        # 5.5 GB source -> data/merfish/
python scripts/data/prepare_xenium.py --download         # 1.5 GB source -> data/xenium/
python scripts/analysis/word_frequency_diagnostics.py configs/merfish/production.json
python scripts/run_experiment.py configs/merfish/smoke.json
python scripts/run_experiment.py configs/xenium/smoke.json
# cluster: see protocol.md 1.9 for ranges; then
python scripts/analysis/plot_label_recovery.py configs/merfish/production.json
python scripts/analysis/merfish/plaque_proximity.py configs/merfish/production.json
python scripts/analysis/plot_label_recovery.py configs/xenium/production.json
python scripts/analysis/xenium/disease_classification.py configs/xenium/production.json
python scripts/analysis/plot_unit_maps.py configs/merfish/production.json
python scripts/analysis/plot_unit_maps.py configs/xenium/production.json
```

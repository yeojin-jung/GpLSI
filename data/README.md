# Data

One folder per dataset. The loaders check every dataset against frozen hashes
(`src/gplsi/contracts/`, or the manifest written by the preparation script), so
a changed or partial input fails loudly instead of silently changing results.
`GPLSI_DATA_ROOT` points the loaders at a different copy of this folder.

| Folder | Dataset (`dataset.name` in configs) | In git | How to obtain |
|---|---|---|---|
| `crc/` | Stanford CRC CODEX, 3-hop cell-type neighborhoods of 113,561 cells x 8 cell types, with patient outcomes (`crc`) | yes | tracked |
| `spleen/` | Mouse spleen CODEX, BALBc-1/2/3 (`spleen`, `group` = sample or `joint`) | yes | tracked |
| `cook/` | What's Cooking recipes: historical 13,597 x 1,019 corpus (`cook`) and v2, 19,017 recipes with >= 8 ingredients x 4,911 ingredients (`cook_v2`) | sources only | `python scripts/data/prepare_cook_v2.py` builds `cook/dataset/raw_jaccard_v1/` and `raw_min8_jaccard_v2/` from `train.json` |
| `dlpfc/` | Visium DLPFC, 12 sections, 47,681 spots x 33,538 genes, manual layers kept as evaluation-only labels (`dlpfc`) | no (320 MB) | `python scripts/data/prepare_dlpfc.py --download` writes `dlpfc/raw/` and `dlpfc/visium_dlpfc.h5ad` |
| `merfish/` | MERFISH TREM2-R47H/5xFAD mouse brain (Johnston et al.; Brain Image Library ace-ear-nap), 432,794 cells x 300 genes, 15 animals / 19 sections; cell types, regions, plaque distance and genotype kept as evaluation-only columns (`merfish`) | no (85 MB; source 5.5 GB) | `python scripts/data/prepare_merfish.py --download` writes `merfish/raw/` and `merfish/merfish_trem2_5xfad.h5ad` (checks the recorded SHA-256 and the published dimensions) |
| `xenium/` | Xenium ulcerative colitis (Mennillo et al.; Figshare 27327813 Dataset 1), 581,967 cells x 290 genes, 25 patient-condition/timepoint units from 20 patients; compartments, cell types and condition kept as evaluation-only columns (`xenium`) | no (72 MB; source 1.5 GB) | `python scripts/data/prepare_xenium.py --download` writes `xenium/raw/` and `xenium/xenium_uc.h5ad` (checks the Figshare MD5 and the published dimensions) |

The spleen compartment labels used by `scripts/analysis/spleen/` are built by
`scripts/data/prepare_spleen_compartment_annotations.py` from the public
CytoCommunity archive.

DLPFC, MERFISH and Xenium share one processed format: integer counts in `X`,
coordinates in `obsm["spatial"]`, and in `uns` the fitting-unit column
(`benchmark_unit_column`), the graph-boundary column (`graph_unit_column`),
the scored labels (`evaluation_label_columns`) and every column that must not
reach a fit (`fit_forbidden_obs_columns`).

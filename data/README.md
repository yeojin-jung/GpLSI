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

The spleen compartment labels used by `scripts/analysis/spleen/` are built by
`scripts/data/prepare_spleen_compartment_annotations.py` from the public
CytoCommunity archive.

# Visium DLPFC

Not tracked in git (320 MB). Build with

    python scripts/data/prepare_dlpfc.py --download

which downloads the 12 sections to `raw/` and writes `visium_dlpfc.h5ad`
(raw UMI counts for all genes; layer labels and donor as evaluation-only
columns) and `summary.json`. Configs in `configs/dlpfc/` read the h5ad.

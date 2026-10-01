"""CRC reference data for the analysis scripts: patients, outcomes, tumor-cell labels.

Two label systems exist and must not be confused:

* the eight modeled features are non-tumor cell-type counts in the 3-hop
  neighbourhood of a focal tumor cell (they define A);
* ``CELL_TYPE`` in ``<region>.type.csv`` is the phenotype of the focal tumor
  cell itself (Tumor 1 ... Tumor 7, with "Tumor 2 (Ki67 Proliferating)" and
  "Tumor 6 / DC"); names are kept as supplied.

The patient is the prefix before the underscore of ``sample_label_visualizer``
(source-validated against the SPACE-GM supplement: 109 patients, 24130 and
24131 distinct). Outcomes keep their 0/1 codes; their clinical direction is not
asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gplsi.real_data import DATA_ROOT

CRC_ROOT = DATA_ROOT / "crc"
TARGETS = ("recurrence", "primary_outcome")


def region_table() -> pd.DataFrame:
    """One row per region: ``region_id``, ``patient``, and the two outcome codes."""

    labels = pd.read_csv(CRC_ROOT / "charville_labels.csv")
    labels["region_id"] = labels["region_id"].astype(str)
    labels["patient"] = labels["sample_label_visualizer"].astype(str).str.split("_").str[0]
    return labels.set_index("region_id")[["patient", *TARGETS]]


def split_observation_ids(observation_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``'<region>::<CELL_ID>'`` -> (region ids, integer cell ids)."""

    parts = pd.Series(np.asarray(observation_ids).astype(str)).str.split("::", n=1, expand=True)
    return parts[0].to_numpy(), parts[1].astype(int).to_numpy()


def tumor_phenotypes(observation_ids: np.ndarray) -> np.ndarray:
    """Focal tumor-cell ``CELL_TYPE`` of each modeled row, joined by region and ``CELL_ID``."""

    regions, cells = split_observation_ids(observation_ids)
    output = np.empty(len(cells), dtype=object)
    for region in pd.unique(regions):
        mask = regions == region
        types = pd.read_csv(CRC_ROOT / "output" / "output_3hop" / f"{region}.type.csv", index_col=0)
        types = types.set_index("CELL_ID")["CELL_TYPE"]
        output[mask] = types.loc[cells[mask]].to_numpy()
    return output.astype(str)


def example_regions(group_ids: np.ndarray, target: str, per_class: int = 2) -> dict[int, list[str]]:
    """Regions to map per outcome class, chosen before looking at any W.

    Within each class, regions are ranked by distance from the class's median
    modeled-cell count (natural region id breaks ties), and the first
    ``per_class`` from distinct patients are kept.
    """

    table = region_table()
    counts = pd.Series(np.asarray(group_ids).astype(str)).value_counts()
    frame = table.loc[counts.index].assign(n=counts.to_numpy()).dropna(subset=[target]).rename_axis("region_id")
    chosen: dict[int, list[str]] = {}
    for code, block in frame.groupby(target):
        median = block["n"].median()
        ranked = block.assign(distance=(block["n"] - median).abs()).reset_index()
        ranked = ranked.sort_values(["distance", "region_id"])
        picked, patients = [], set()
        for _, row in ranked.iterrows():
            if row["patient"] not in patients:
                picked.append(row["region_id"])
                patients.add(row["patient"])
            if len(picked) == per_class:
                break
        chosen[int(code)] = picked
    return chosen

#!/usr/bin/env python3
"""Freeze the public manual spleen-compartment labels in model-row order.

The source archive is distributed by the CytoCommunity authors at
https://github.com/huBioinfo/CytoCommunity/blob/main/CODEX_SpleenDataset.zip.
This script verifies that archive, aligns its cell annotations to the audited
Goltsev tables, and writes compact deterministic files keyed by original cell
identifier.  Model fitting never reads these downstream labels.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "data/spleen/dataset"
SOURCE_URL = (
    "https://github.com/huBioinfo/CytoCommunity/blob/main/"
    "CODEX_SpleenDataset.zip"
)
SOURCE_ARCHIVE_SHA256 = (
    "24ed314538efd61e4b0bd744b48ddf2039be82bb88d24b65bd68501036ff7d7c"
)
SOURCE_CSV_SHA256 = {
    "BALBc-1": "476cb06f1a1a96883c40c4a92215548395fbf3d4113a91954e9f113342b867d8",
    "BALBc-2": "30a934a033fdd79a3e3dd445227f1bc8cf899f0aa2240e7a955ec09dc2f46b0a",
    "BALBc-3": "6fbdb489d46fb3a3fb149ffbb9495b4f1c03bb08bdba08fea57bd0e3ff7bc8b8",
}
COMPARTMENTS = ("B-zone", "marginal zone", "PALS", "red pulp", "NoAnnotation")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _source_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the CytoCommunity coordinate convention for local raw cells."""

    output = frame.copy()
    output["x_coordinate"] = (
        output["sample.X"] - (1344 + 42 * output["Xtile"])
    ).astype(int)
    output["y_coordinate"] = (-(output["sample.Y"] - 1008)).astype(int)
    return output


def prepare_annotations(archive: Path, output_dir: Path) -> dict[str, object]:
    content = archive.read_bytes()
    observed_archive_hash = _sha256(content)
    if observed_archive_hash != SOURCE_ARCHIVE_SHA256:
        raise ValueError(
            "unexpected CytoCommunity archive hash: "
            f"{observed_archive_hash} != {SOURCE_ARCHIVE_SHA256}"
        )

    raw_frames = pd.read_pickle(DATASET_ROOT / "spleen_dfs.pkl")
    model_counts = pd.read_pickle(DATASET_ROOT / "merged_D.pkl")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_groups: dict[str, object] = {}

    with zipfile.ZipFile(io.BytesIO(content)) as source:
        for group, expected_csv_hash in SOURCE_CSV_SHA256.items():
            member = f"CODEX_SpleenDataset/{group}.csv"
            source_content = source.read(member)
            observed_csv_hash = _sha256(source_content)
            if observed_csv_hash != expected_csv_hash:
                raise ValueError(
                    f"unexpected {group} source hash: "
                    f"{observed_csv_hash} != {expected_csv_hash}"
                )
            labels = pd.read_csv(io.BytesIO(source_content)).rename(
                columns={"CellType": "cluster"}
            )
            if set(labels["Compartment"].astype(str)) != set(COMPARTMENTS):
                raise ValueError(f"unexpected {group} compartment vocabulary")

            raw = _source_coordinates(raw_frames[group].reset_index(names="raw_cell_id"))
            keys = ["x_coordinate", "y_coordinate", "cluster"]
            if raw.duplicated(keys).any() or labels.duplicated(keys).any():
                raise ValueError(f"{group} annotation keys are not unique")
            aligned = raw.merge(
                labels[keys + ["Compartment"]],
                on=keys,
                how="left",
                validate="one_to_one",
            ).set_index("raw_cell_id")
            if aligned["Compartment"].isna().any():
                raise ValueError(f"{group} has unmatched local cells")

            model_ids = pd.Index(
                [value[1] for value in model_counts.loc[group].index],
                name="raw_cell_id",
            )
            modeled = aligned.reindex(model_ids)[["Compartment"]].rename(
                columns={"Compartment": "compartment"}
            )
            if modeled["compartment"].isna().any():
                raise ValueError(f"{group} has unmatched modeled cells")
            modeled.insert(0, "raw_cell_id", model_ids.to_numpy(dtype=int))

            output = output_dir / f"{group}.csv.gz"
            csv_content = modeled.to_csv(index=False, lineterminator="\n").encode()
            with output.open("wb") as destination:
                with gzip.GzipFile(
                    filename="", fileobj=destination, mode="wb", mtime=0
                ) as compressed:
                    compressed.write(csv_content)
            output_hash = _sha256(output.read_bytes())
            counts = modeled["compartment"].value_counts().sort_index()
            manifest_groups[group] = {
                "source_member": member,
                "source_csv_sha256": observed_csv_hash,
                "output": output.name,
                "output_sha256": output_hash,
                "model_rows": int(len(modeled)),
                "labeled_rows": int((modeled["compartment"] != "NoAnnotation").sum()),
                "counts": {str(key): int(value) for key, value in counts.items()},
            }

    manifest = {
        "schema_version": 1,
        "source_url": SOURCE_URL,
        "source_archive_sha256": observed_archive_hash,
        "coordinate_alignment": {
            "x_coordinate": "sample.X - (1344 + 42 * Xtile)",
            "y_coordinate": "-(sample.Y - 1008)",
            "join_keys": ["x_coordinate", "y_coordinate", "cluster"],
        },
        "allowed_compartments": list(COMPARTMENTS),
        "groups": manifest_groups,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATASET_ROOT / "compartments",
    )
    args = parser.parse_args()
    manifest = prepare_annotations(args.archive, args.output_dir)
    total = sum(
        int(group["model_rows"])
        for group in manifest["groups"].values()
    )
    labeled = sum(
        int(group["labeled_rows"])
        for group in manifest["groups"].values()
    )
    print(f"wrote {labeled:,}/{total:,} labeled model rows to {args.output_dir}")


if __name__ == "__main__":
    main()

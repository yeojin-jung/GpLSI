#!/usr/bin/env python3
"""Report complete and retryable tasks from the unsubmitted full-run manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.real_data_anchor_word_gplsi.run_experiment import (  # noqa: E402
    _code_hashes,
    _runtime_provenance,
)


def expected_task(config: dict, local_index: int) -> tuple[int, int, str]:
    tasks = [(int(K), int(seed)) for K in config["K_values"] for seed in config["seeds"]]
    K, seed = tasks[local_index]
    group = config.get("group") if config["dataset"] == "spleen" else None
    pieces = [str(config["dataset"])]
    if group:
        pieces.append(str(group))
    pieces.extend((f"K{K}", f"seed{seed}"))
    return K, seed, "__".join(pieces).replace("/", "-")


def runtime_provenance_matches(recorded: dict, current: dict) -> bool:
    """Compare runtime content while ignoring the equivalent import location."""
    recorded = dict(recorded)
    current = dict(current)
    recorded.pop("pycvxcluster_package_root", None)
    current.pop("pycvxcluster_package_root", None)
    return recorded == current


def valid_completion(
    task_dir: Path,
    config: dict,
    K: int,
    seed: int,
    code_hashes: dict[str, str],
    runtime_provenance: dict,
) -> bool:
    manifest_path = task_dir / "task_manifest.json"
    complete_path = task_dir / "complete.json"
    if not manifest_path.exists() or not complete_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        complete = json.loads(complete_path.read_text())
    except (OSError, ValueError):
        return False
    return bool(
        manifest.get("config") == config
        and int(manifest.get("K", -1)) == K
        and int(manifest.get("seed", -1)) == seed
        and manifest.get("code_hashes") == code_hashes
        and runtime_provenance_matches(
            manifest.get("runtime_provenance", {}), runtime_provenance
        )
        and complete.get("task_config_hash") == manifest.get("task_config_hash")
        and int(complete.get("row_count", 0)) > 0
    )


def slurm_array_spec(indices: list[int]) -> str:
    if not indices:
        return ""
    values = sorted(set(indices))
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=REPO_ROOT / "configs/real_data_anchor_word_gplsi/full_manifest.tsv",
    )
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    code_hashes = _code_hashes()
    runtime_provenance = _runtime_provenance()
    retry: list[int] = []
    complete_count = 0
    with manifest_path.open(newline="") as handle:
        batches = list(csv.DictReader(handle, delimiter="\t"))
    for batch in batches:
        config_path = REPO_ROOT / batch["config"]
        config = json.loads(config_path.read_text())
        output_root = Path(
            config.get(
                "output_root", REPO_ROOT / "results/real_data_anchor_word_gplsi"
            )
        )
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root
        for local_index in range(int(batch["task_count"])):
            array_index = int(batch["array_first"]) + local_index
            K, seed, prefix = expected_task(config, local_index)
            candidates = sorted((output_root / config["run_name"]).glob(f"{prefix}__*"))
            if any(
                valid_completion(
                    path, config, K, seed, code_hashes, runtime_provenance
                )
                for path in candidates
            ):
                complete_count += 1
            else:
                retry.append(array_index)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path.resolve()),
                "expected_tasks": complete_count + len(retry),
                "complete_tasks": complete_count,
                "retry_array_indices": retry,
                "retry_array_spec": slurm_array_spec(retry),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

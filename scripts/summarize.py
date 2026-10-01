#!/usr/bin/env python3
"""Collect every row of one experiment into a single table.

    python scripts/summarize.py configs/crc/production.json
    python scripts/summarize.py results/crc_production

Writes ``<run dir>/fit_rows.csv`` (one row per fit, nested fields flattened
with dots, artifact paths absolute) and prints fit counts by status. The table
is the ``--fit-table`` input of the CRC and spleen evaluation scripts; load it
in Python with ``gplsi.pipeline.load_rows``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gplsi.pipeline import load_config, load_rows
from gplsi.real_data import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="experiment config or its results directory")
    args = parser.parse_args()
    run_dir = args.target
    if run_dir.suffix == ".json":
        config = load_config(run_dir)
        run_dir = REPO_ROOT / config.get("output_root", "results") / config["name"]
    frame = load_rows(run_dir)
    if frame.empty:
        raise SystemExit(f"no rows under {run_dir}")
    frame.to_csv(run_dir / "fit_rows.csv", index=False)
    counts = frame.groupby(["estimator_family", "status"]).size().unstack(fill_value=0)
    print(counts.to_string())
    print(f"\n{len(frame)} rows from {frame['task_dir'].nunique()} tasks -> {run_dir / 'fit_rows.csv'}")


if __name__ == "__main__":
    main()

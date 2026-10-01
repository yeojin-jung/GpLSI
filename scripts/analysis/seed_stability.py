#!/usr/bin/env python3
"""Cross-seed stability of the topic matrix A, per method and K.

    python scripts/analysis/seed_stability.py configs/crc/production.json
    python scripts/analysis/seed_stability.py results/dlpfc_production --top 10

Every A estimator (A_current, A_full_L2, A_full_Pois, native) is compared
separately. Seeds change the 20% count split (and, for LDA-type methods, the random start),
so stability measures how much A moves under a perturbation of the data. For
every pair of seeds of the same method, K and unit (DLPFC section), topics are
matched one-to-one by the Hungarian assignment maximising the cosine similarity
of A rows. Reported per pair:

* ``matched_cosine_mean`` / ``matched_cosine_min``: cosine of matched topics,
  averaged and worst over topics (a low minimum = one topic is not reproduced);
* ``average_jaccard``: Greene et al. (2014) top-weighted Jaccard of the matched
  topics' top-``--top`` words, averaged over depths 1..top and over topics.

When vocabularies differ across seeds (the DLPFC panel is chosen on each
seed's training counts), A is placed on the union of feature ids with zeros.
Writes ``<run>/figures/seed_stability.csv`` (pairs) and ``seed_stability_summary.csv``.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import align_topics, method_label, normalize_rows, run_directory, task_data  # noqa: E402

from gplsi.pipeline import load_arrays, load_rows  # noqa: E402


def average_jaccard(first: np.ndarray, second: np.ndarray, top: int) -> float:
    """Top-weighted Jaccard of two rankings (indices sorted by decreasing weight)."""

    scores = []
    for depth in range(1, top + 1):
        a, b = set(first[:depth]), set(second[:depth])
        scores.append(len(a & b) / len(a | b))
    return float(np.mean(scores))


def on_union(A: np.ndarray, features: np.ndarray, union: dict[str, int]) -> np.ndarray:
    full = np.zeros((A.shape[0], len(union)))
    full[:, [union[f] for f in features]] = A
    return full


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="experiment config or its results directory")
    parser.add_argument("--top", type=int, default=10, help="depth of the top-word Jaccard (capped at p)")
    args = parser.parse_args()

    # Every A estimator is its own method here: A stability is the point.
    rows = load_rows(run_directory(args.target))
    rows["label"] = [
        method_label(row) + (f" · {row['A_recovery']}" if row["A_recovery"] != "native" else "")
        for _, row in rows.iterrows()
    ]
    rows = rows[rows["status"] == "ok"].copy()
    rows["unit"] = rows["task.section"].astype(str) if "task.section" in rows else "all"

    records = []
    for (label, K, unit), group in rows.groupby(["label", "K", "unit"]):
        fits = []
        for _, row in group.sort_values("seed").iterrows():
            features = task_data(row["task_dir"])["feature_ids"].astype(str)
            fits.append((int(row["seed"]), normalize_rows(load_arrays(row)["A_hat"]), features))
        union = {f: i for i, f in enumerate(sorted(set().union(*(set(f) for _, _, f in fits))))}
        for (seed_a, A_a, f_a), (seed_b, A_b, f_b) in combinations(fits, 2):
            A_a, A_b = on_union(A_a, f_a, union), on_union(A_b, f_b, union)
            order, cosine = align_topics(A_b, A_a)
            depth = min(args.top, A_a.shape[1])
            ranks_a = np.argsort(-A_a, axis=1, kind="stable")
            ranks_b = np.argsort(-A_b[order], axis=1, kind="stable")
            jaccard = np.mean([average_jaccard(ranks_a[k], ranks_b[k], depth) for k in range(int(K))])
            records.append(dict(method=label, K=int(K), unit=unit, seed_a=seed_a, seed_b=seed_b,
                                matched_cosine_mean=float(cosine.mean()), matched_cosine_min=float(cosine.min()),
                                average_jaccard=float(jaccard)))
    pairs = pd.DataFrame(records)
    out = run_directory(args.target) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "seed_stability.csv", index=False)
    if pairs.empty:
        print("fewer than two successful seeds per method; nothing to compare")
        return
    summary = pairs.groupby(["method", "K"])[["matched_cosine_mean", "matched_cosine_min", "average_jaccard"]].agg(["mean", "min"])
    summary.columns = ["__".join(column) for column in summary.columns]
    summary["pairs"] = pairs.groupby(["method", "K"]).size()
    summary.reset_index().to_csv(out / "seed_stability_summary.csv", index=False)
    print(summary.round(3).to_string())
    print(f"wrote {out / 'seed_stability.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare realized Tran rank-frequency profiles at two Zipf settings."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {0.1: "#0072B2", 10.0: "#D55E00"}


def _parse_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key] = value
    return metadata


def _load_profiles(data_dir: Path, design: str, decay: float) -> tuple[np.ndarray, list[int], dict[str, str]]:
    decay_label = f"{decay:g}".replace(".", "p")
    pattern = f"{design}_decay_{decay_label}_seed_*"
    dataset_dirs = sorted(
        (path for path in data_dir.glob(pattern) if path.is_dir()),
        key=lambda path: int(re.search(r"_seed_(\d+)$", path.name).group(1)),
    )
    if not dataset_dirs:
        raise FileNotFoundError(f"No datasets matched {data_dir / pattern}")

    profiles: list[np.ndarray] = []
    observed_counts: list[int] = []
    reference_metadata: dict[str, str] | None = None
    for dataset_dir in dataset_dirs:
        metadata = _parse_metadata(dataset_dir / "metadata.txt")
        requested_p = int(metadata["requested_p"])
        counts = pd.read_csv(dataset_dir / "counts.csv").to_numpy(dtype=float)
        word_counts = counts.sum(axis=0)
        total = word_counts.sum()
        if total <= 0:
            raise ValueError(f"Empty corpus in {dataset_dir}")
        profile = np.sort(word_counts / total)[::-1]
        profile = np.pad(profile, (0, requested_p - profile.size))
        profiles.append(profile)
        observed_counts.append(int(np.count_nonzero(word_counts)))
        reference_metadata = metadata

    return np.vstack(profiles), observed_counts, reference_metadata or {}


def _quantiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.quantile(values, q, axis=0) for q in (0.25, 0.5, 0.75))


def _summary_row(decay: float, profiles: np.ndarray, observed_counts: list[int]) -> dict[str, float]:
    cumulative = np.cumsum(profiles, axis=1)
    effective_vocab = 1.0 / np.sum(profiles**2, axis=1)
    return {
        "a_zipf": decay,
        "n_seeds": profiles.shape[0],
        "median_observed_words": float(np.median(observed_counts)),
        "q25_observed_words": float(np.quantile(observed_counts, 0.25)),
        "q75_observed_words": float(np.quantile(observed_counts, 0.75)),
        "median_top_10_mass": float(np.median(cumulative[:, 9])),
        "median_top_25_mass": float(np.median(cumulative[:, 24])),
        "median_effective_vocabulary": float(np.median(effective_vocab)),
    }


def plot_comparison(run_dir: Path, output_stem: Path) -> pd.DataFrame:
    data_dir = run_dir / "data"
    design = "tran_mixed_word_decay_exact"
    decays = (0.1, 10.0)
    loaded = {decay: _load_profiles(data_dir, design, decay) for decay in decays}

    first_metadata = loaded[decays[0]][2]
    p = int(first_metadata["requested_p"])
    n = int(first_metadata["n"])
    N = int(first_metadata["N"])
    ranks = np.arange(1, p + 1)

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), constrained_layout=True)

    summary_rows: list[dict[str, float]] = []
    for decay in decays:
        profiles, observed_counts, _ = loaded[decay]
        q25, median, q75 = _quantiles(profiles)
        color = COLORS[decay]
        label = rf"$a_{{Zipf}}={decay:g}$"

        positive = median > 0
        axes[0].plot(ranks[positive], 100 * median[positive], color=color, lw=2.4, label=label)
        band = q25 > 0
        axes[0].fill_between(
            ranks[band], 100 * q25[band], 100 * q75[band], color=color, alpha=0.18, linewidth=0
        )

        cumulative = np.cumsum(profiles, axis=1)
        c25, c50, c75 = _quantiles(cumulative)
        axes[1].plot(ranks, 100 * c50, color=color, lw=2.4, label=label)
        axes[1].fill_between(ranks, 100 * c25, 100 * c75, color=color, alpha=0.18, linewidth=0)

        row = _summary_row(decay, profiles, observed_counts)
        summary_rows.append(row)
        observed_label = f"median observed: {row['median_observed_words']:.0f}/{p} words"
        axes[0].text(
            0.98,
            0.94 if decay == 0.1 else 0.84,
            rf"$a_{{Zipf}}={decay:g}$ — {observed_label}",
            transform=axes[0].transAxes,
            ha="right",
            va="top",
            color=color,
            fontsize=9.3,
        )

    axes[0].set_title("Rank–frequency profile")
    axes[0].set_xlabel("Word rank (most to least frequent)")
    axes[0].set_ylabel("Share of all generated tokens (%)")
    axes[0].set_yscale("log")
    axes[0].set_xlim(1, p)
    axes[0].grid(True, which="major", alpha=0.25)
    axes[0].grid(True, which="minor", axis="y", alpha=0.10)
    axes[0].text(
        0.98,
        0.02,
        "Lines: median across seeds · bands: interquartile range\n"
        "Zero-count tail is omitted on the logarithmic axis.",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.5},
    )

    axes[1].set_title("Cumulative vocabulary mass")
    axes[1].set_xlabel("Number of top-ranked words retained")
    axes[1].set_ylabel("Cumulative share of generated tokens (%)")
    axes[1].set_xlim(1, p)
    axes[1].set_ylim(0, 101)
    axes[1].grid(True, alpha=0.25)
    axes[1].axhline(80, color="#777777", lw=1, ls="--", alpha=0.6)
    axes[1].legend(loc="lower right", frameon=False)

    seed_count = loaded[decays[0]][0].shape[0]
    fig.suptitle(
        "Realized Tran word-frequency decay: $a_{Zipf}=0.1$ versus $a_{Zipf}=10$\n"
        f"Exact Tran benchmark · n={n} documents · N={N} tokens/document · p={p} words · {seed_count} seeds",
        fontsize=14,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_stem.with_name(output_stem.name + "_summary.csv"), index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Anchor-word pilot result directory")
    parser.add_argument("--output-stem", type=Path, default=None)
    args = parser.parse_args()
    output_stem = args.output_stem or args.run_dir / "figures" / "word_frequency_decay_a0p1_vs_a10"
    summary = plot_comparison(args.run_dir, output_stem)
    print(summary.to_string(index=False))
    print(f"Wrote {output_stem.with_suffix('.png')}")
    print(f"Wrote {output_stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

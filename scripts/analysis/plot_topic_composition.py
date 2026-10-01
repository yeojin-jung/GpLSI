#!/usr/bin/env python3
"""Topic composition (rows of A) of every fitted method in one task, side by side.

    python scripts/analysis/plot_topic_composition.py results/crc_smoke_K4_svs_star
    python scripts/analysis/plot_topic_composition.py CONFIG --task crc__K4__seed26090301

Topics of every method are matched one-to-one to the LDA fit (else the first
fit) by maximum cosine similarity of full A rows (Hungarian), the convention of
the handoff reports, so topic k is the same row in every panel. The match is
for display; a shared row need not be the same topic. Each row is labelled with its
prevalence, the mean of that topic's column of W. Writes a PNG and a tidy CSV
to ``<run dir>/figures/``. Meant for small vocabularies (e.g. CRC's 8 cell
types); larger ones should show top features instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gplsi.pipeline import load_arrays, load_config, load_rows  # noqa: E402
from gplsi.real_data import REPO_ROOT  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import align_topics  # noqa: E402

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
# Sequential blue ramp, light (near zero) to dark.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
FAMILY_LABEL = {
    "document_gplsi": "GpLSI",
    "anchor_feature_gplsi": "GpLSI anchor",
    "plsi": "pLSI",
    "lda": "LDA",
    "spatial_lda": "Spatial LDA",
    "topicscore_raw": "TopicSCORE",
    "topicscore_graph_denoised": "TopicSCORE (graph)",
    "kl_nmf": "KL-NMF",
    "graph_kl_nmf": "graph KL-NMF",
}
FAMILY_ORDER = list(FAMILY_LABEL)
RECOVERY_ORDER = ["A_current", "A_full_L2", "A_full_Pois", "A_full_Pois_SQUAREM", "native"]


def label(row: dict) -> str:
    name = FAMILY_LABEL.get(row["estimator_family"], row["estimator_family"])
    if row["estimator_family"] in ("document_gplsi", "anchor_feature_gplsi"):
        name += f" · {row['preprocessing']} · {row['vertex_hunter']}"
    if row["A_recovery"] != "native":
        name += f"\n{row['A_recovery']}"
    return name


def order_key(row: dict) -> tuple:
    family = row["estimator_family"]
    recovery = row["A_recovery"]
    return (
        FAMILY_ORDER.index(family) if family in FAMILY_ORDER else len(FAMILY_ORDER),
        row["preprocessing"], row["vertex_hunter"],
        RECOVERY_ORDER.index(recovery) if recovery in RECOVERY_ORDER else len(RECOVERY_ORDER),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="experiment config or its results directory")
    parser.add_argument("--task", help="task directory name (default: the first)")
    parser.add_argument("--plsi-recoveries", action="store_true", help="show every pLSI A recovery, not only A_current")
    args = parser.parse_args()

    run_dir = args.target
    if run_dir.suffix == ".json":
        config = load_config(run_dir)
        run_dir = REPO_ROOT / config.get("output_root", "results") / config["name"]
    rows = [row for row in load_rows(run_dir, flatten=False) if row["status"] == "ok"]
    tasks = sorted({Path(row["task_dir"]).name for row in rows})
    task = args.task or tasks[0]
    rows = [row for row in rows if Path(row["task_dir"]).name == task]
    if not args.plsi_recoveries:
        rows = [r for r in rows if r["estimator_family"] != "plsi" or r["A_recovery"] == "A_current"]
    rows.sort(key=order_key)
    if not rows:
        raise SystemExit(f"no finished fits in {run_dir / task}")

    with np.load(Path(rows[0]["task_dir"]) / "data.npz") as data:
        features = data["feature_names"].astype(str)
    fits = []
    for row in rows:
        arrays = load_arrays(row)
        fits.append((row, arrays["A_hat"], arrays["W_hat"]))
    reference = next((A for row, A, _ in fits if row["estimator_family"] == "lda"), fits[0][1])

    tidy = []
    K = reference.shape[0]
    columns = min(3, len(fits))
    nrows = int(np.ceil(len(fits) / columns))
    fig, axes = plt.subplots(nrows, columns, figsize=(4.6 * columns, 0.42 * K * nrows + 1.6 * nrows + 0.8),
                             squeeze=False, facecolor=SURFACE)
    vmax = max(float(A.max()) for _, A, _ in fits)
    for ax, (row, A, W) in zip(axes.flat, fits):
        order, _ = align_topics(A, reference)
        A, prevalence = A[order], W.mean(axis=0)[order]
        ax.imshow(A, cmap=SEQUENTIAL, vmin=0, vmax=vmax, aspect="auto")
        for k in range(K):
            for j in range(A.shape[1]):
                value = A[k, j]
                if value >= 0.05:  # label the cells that carry the composition
                    ax.text(j, k, f"{value:.2f}", ha="center", va="center", fontsize=7,
                            color="#ffffff" if value > 0.55 * vmax else INK)
                tidy.append(dict(task=task, method=row["method"], topic=k, prevalence=prevalence[k],
                                 feature=features[j], proportion=value))
        ax.set_yticks(range(K), [f"T{k + 1}  {100 * prevalence[k]:.0f}%" for k in range(K)], fontsize=8, color=INK2)
        ax.set_xticks(range(len(features)), features, rotation=45, ha="right", fontsize=7.5, color=INK2)
        ax.set_title(label(row), fontsize=9, color=INK, loc="left")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(features)), minor=True)
        ax.set_yticks(np.arange(-0.5, K), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
    for ax in list(axes.flat)[len(fits):]:
        ax.axis("off")
    fig.suptitle(f"Topic composition (rows of A), {task}. Row label: topic and prevalence (mean W). "
                 "Topics matched to LDA (cosine).", fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"topic_composition__{task}.png", dpi=160, facecolor=SURFACE)
    pd.DataFrame(tidy).to_csv(out / f"topic_composition__{task}.csv", index=False)
    print(f"wrote {out / f'topic_composition__{task}.png'}")


if __name__ == "__main__":
    main()

"""Side-by-side spatial maps for the follow-up designs, one figure per section and seed.

    python scripts/analysis/dlpfc/plot_design_maps.py --design p2_wide
    python scripts/analysis/dlpfc/plot_design_maps.py --design panel
    python scripts/analysis/dlpfc/plot_design_maps.py --design k5_br5595

* p2_wide: rows are the KE-weighted (P2) GpLSI fits on the core penalty grid (top
  ~0.0165) and on the wide grid (top ~0.75); columns are the six vertex hunters.
* panel: rows are the panel sizes 500 / 1,000 / 2,000 / 5,000; columns are the GpLSI
  hunters in the panel design (P0) and the baselines.
* k5_br5595: rows are K = 7 (core) and K = 5 on Br5595 (151669) for the same seed and
  split; columns are the P0 GpLSI hunters and the baselines. P2 is left out because its
  K = 7 core fit used the short penalty grid (compare it in the summary tables).

Colouring follows plot_spatial_maps.py (topics drawn in their Hungarian-matched layer's
colour). Writes to results/dlpfc/diagnostics/<design>/spatial/. Needs the
processed h5ad, so run it on a compute node.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import plot_spatial_maps as maps  # noqa: E402  (same directory)

from records import DIAGNOSTICS, RESULTS, load_task  # noqa: E402


def task_dirs(design: str) -> list[Path]:
    return sorted(path for path in (RESULTS / design).iterdir() if (path / "rows").is_dir())

PANEL_METHODS = [
    ("gplsi_document__P0_raw__spa_current__A_current", "GpLSI P0 · spa_current"),
    ("gplsi_document__P0_raw__svs_star__A_current", "GpLSI P0 · svs_star"),
    ("gplsi_document__P0_raw__palm_accelerated__A_current", "GpLSI P0 · palm_acc."),
    ("lda", "LDA"),
    ("kl_nmf", "KL-NMF"),
    ("topicscore_raw", "TopicSCORE raw"),
]

K5_METHODS = [
    ("gplsi_document__P0_raw__spa_current__A_current", "GpLSI P0 · spa_current"),
    ("gplsi_document__P0_raw__svs__A_current", "GpLSI P0 · svs"),
    ("gplsi_document__P0_raw__svs_star__A_current", "GpLSI P0 · svs_star"),
    ("gplsi_document__P0_raw__pp_spa__A_current", "GpLSI P0 · pp_spa"),
    ("spatial_lda", "Spatial-LDA"),
    ("lda", "LDA"),
    ("graph_kl_nmf", "graph KL-NMF"),
]


def load(path: Path):
    """(payload, arrays) of one task directory."""

    return load_task(path)


def rho(payload, method: str):
    index = payload["array_index"].get(method)
    return None if index is None else payload["results"][index].get("metadata", {}).get("selected_rho")


def grid_figure(rows, columns, labels, xy, title, path: Path) -> None:
    """rows: [(row label, payload, saved)]; columns: [(method, column title)]."""

    fig, axes = plt.subplots(len(rows), len(columns) + 1, figsize=(2.3 * (len(columns) + 1), 2.45 * len(rows)),
                             squeeze=False)
    for r, (row_label, payload, saved) in enumerate(rows):
        maps.layer_panel(axes[r, 0], xy, labels, f"{row_label}\nmanual layers", 2.4)
        for c, (method, col_title) in enumerate(columns, start=1):
            lam = rho(payload, method)
            label = col_title + (f" · λ {lam:.3g}" if lam is not None else "")
            maps.panel(axes[r, c], payload, saved, method, label, xy, labels, 2.4)
    maps.legend(fig, labels)
    fig.suptitle(title, fontsize=10, y=0.995)
    fig.patch.set_facecolor(maps.SURFACE)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=140, facecolor=maps.SURFACE)
    plt.close(fig)
    print("wrote", path)


def p2_wide(out: Path) -> None:
    columns = [(f"gplsi_document__P2_ke_weighted__{h}__A_current", f"P2 · {h}") for h in maps.HUNTERS]
    # Task directories are named by section/K/panel/seed only, so the matching
    # core task has the same name.
    for path in task_dirs("p2_wide"):
        wide = load(path)
        core = load(RESULTS / "core" / path.name)
        ids = wide[1]["observation_ids"].astype(str)
        if not np.array_equal(ids, core[1]["observation_ids"].astype(str)):
            raise ValueError(f"{path.name}: spots differ from the matching core task")
        labels, xy = maps.layer_labels(ids), wide[1]["coordinates"].astype(float)
        task = wide[0]["task"]
        grid_figure([("core grid (top 0.0165)", *core), ("wide grid (top 0.75)", *wide)], columns, labels, xy,
                    f"GpLSI with KE weighting (P2), core vs wide penalty grid · section {task['unit_id']} · seed {task['seed']}",
                    out / f"p2_wide_{task['unit_id']}_s{task['seed']}.png")


def panel(out: Path) -> None:
    by_task = defaultdict(list)
    for path in task_dirs("panel"):
        payload, saved = load(path)
        task = payload["task"]
        by_task[(task["unit_id"], task["seed"])].append((int(task["panel_size"]), payload, saved))
    for (section, seed), items in sorted(by_task.items()):
        items.sort(key=lambda item: item[0])
        # Panels share spots (only genes differ), so one label vector serves every row.
        ids = items[0][2]["observation_ids"].astype(str)
        labels, xy = maps.layer_labels(ids), items[0][2]["coordinates"].astype(float)
        rows = []
        for p, payload, saved in items:
            if not np.array_equal(saved["observation_ids"].astype(str), ids):
                raise ValueError(f"{section} s{seed} p{p}: spots differ across panel sizes")
            rows.append((f"{p:,} genes", payload, saved))
        grid_figure(rows, PANEL_METHODS, labels, xy,
                    f"Panel size · section {section} · seed {seed} (P1 = P0 here, so only P0 is shown)",
                    out / f"panel_{section}_s{seed}.png")


def k5_br5595(out: Path) -> None:
    for path in task_dirs("k5_br5595"):
        k5 = load(path)
        core = load(RESULTS / "core" / path.name.replace("__K5__", "__K7__", 1))
        ids = k5[1]["observation_ids"].astype(str)
        if not np.array_equal(ids, core[1]["observation_ids"].astype(str)):
            raise ValueError(f"{path.name}: spots differ from the matching core task")
        labels, xy = maps.layer_labels(ids), k5[1]["coordinates"].astype(float)
        task = k5[0]["task"]
        grid_figure([("K = 7 (core)", *core), ("K = 5", *k5)], K5_METHODS, labels, xy,
                    f"K = 7 vs K = 5 · section {task['unit_id']} (Br5595: L3–L6 + WM) · seed {task['seed']}",
                    out / f"k5_{task['unit_id']}_s{task['seed']}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design", required=True, choices=["p2_wide", "panel", "k5_br5595"])
    args = parser.parse_args()
    out = DIAGNOSTICS / args.design / "spatial"
    out.mkdir(parents=True, exist_ok=True)
    {"p2_wide": p2_wide, "panel": panel, "k5_br5595": k5_br5595}[args.design](out)


if __name__ == "__main__":
    main()

"""Write the DLPFC ablation task manifests into configs/visium_dlpfc/."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

# One section per donor for the pre-meeting tier; 151669 (Br5595) has only L3-L6 + WM.
CORE_SECTIONS = ["151507", "151669", "151673"]
ALL_SECTIONS = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]
SEEDS = [26090401, 26090402, 26090403, 26090404, 26090405]
COLUMNS = ["design", "unit_id", "K", "panel_size", "retained_fraction", "seed"]


def grid(design, sections, Ks, panels, fractions, seeds):
    return [
        dict(design=design, unit_id=s, K=K, panel_size=p, retained_fraction=r, seed=seed)
        for s, K, p, r, seed in product(sections, Ks, panels, fractions, seeds)
    ]


def main() -> None:
    out = Path("configs/visium_dlpfc")
    manifests = {
        "tasks_smoke.csv": grid("smoke", ["151673"], [7], [2000], [1.0], SEEDS[:1]),
        "tasks_core.csv": grid("core", CORE_SECTIONS, [7], [2000], [1.0], SEEDS[:3]),
        "tasks_panel.csv": grid("panel", CORE_SECTIONS, [7], [500, 1000, 2000, 5000], [1.0], SEEDS[:3]),
        "tasks_lambda_wide.csv": grid("lambda_wide", CORE_SECTIONS, [7], [2000], [1.0], SEEDS[:1]),
        # Post-meeting tier matching the PDF grid (K sweep + count thinning at K=7).
        "tasks_extended.csv": grid("core", ALL_SECTIONS, [5, 7, 9], [2000], [1.0], SEEDS)
        + grid("core", ALL_SECTIONS, [7], [2000], [0.75, 0.5, 0.25], SEEDS),
    }
    for name, rows in manifests.items():
        frame = pd.DataFrame(rows, columns=COLUMNS)
        frame.to_csv(out / name, index=False)
        print(f"{name}: {len(frame)} tasks")


if __name__ == "__main__":
    main()

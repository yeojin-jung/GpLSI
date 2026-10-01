#!/usr/bin/env python3
"""Build the What's Cooking v2 corpus used by configs/cook/ (``dataset: cook_v2``).

Two frozen steps, each writing a hash-checked directory under data/cook/dataset/:

1. ``prepare_cook_raw_jaccard.py`` - literal ingredient-token counts of the
   cuisine-balanced recipes (raw_jaccard_v1);
2. ``prepare_cook_raw_min8_jaccard_v2.py`` - keep recipes with >= 8
   ingredients, drop ingredient columns that become all-zero, and rebuild the
   binary-set Jaccard graph (raw_min8_jaccard_v2).

Step 2 imports the graph builder of ``prepare_cook_raw_min8_jaccard.py``.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def main() -> None:
    for script in ("prepare_cook_raw_jaccard.py", "prepare_cook_raw_min8_jaccard_v2.py"):
        print(f"== {script}", flush=True)
        subprocess.run([sys.executable, str(HERE / script)], check=True)


if __name__ == "__main__":
    main()

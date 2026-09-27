#!/usr/bin/env python3
"""Freeze small, text-only published artifacts from the last artifact commit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPO = Path(os.environ.get("GPLSI_PUBLISHED_REPO", str(REPO_ROOT)))
COMMIT = "7367525f9e5d4f47c272e71f13c582f9a7510615"
TARGET = REPO_ROOT / "tests" / "fixtures" / "real_data_anchor_word_gplsi"
FILES = {
    "crc_published_Ahat_gplsi_K6.csv": "data/stanford-crc/model/model_3hop/Ahats_aligned/Ahats_gplsi/Ahat_gplsi_6_aligned.csv",
    "crc_published_Ahat_plsi_K6.csv": "data/stanford-crc/model/model_3hop/Ahats_aligned/Ahats_plsi/Ahat_plsi_6_aligned.csv",
    "crc_published_Ahat_lda_K6.csv": "data/stanford-crc/model/model_3hop/Ahats_aligned/Ahats_lda/Ahat_lda_6_aligned.csv",
    "spleen_published_metrics.csv": "data/spleen/model/results_spleen.csv",
    "cook_published_anchors_K7.csv": "data/whats-cooking/model/cooking_model_results_7_anchors.csv",
    "cook_published_weights_K7.csv": "data/whats-cooking/model/cooking_model_results_7_weights.csv",
}


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_repository": str(SOURCE_REPO),
        "source_commit": COMMIT,
        "note": "Frozen published outputs only; these are not claimed reproducible.",
        "files": {},
    }
    for target_name, source_path in FILES.items():
        content = subprocess.check_output(
            ["git", "-C", str(SOURCE_REPO), "show", f"{COMMIT}:{source_path}"]
        )
        (TARGET / target_name).write_bytes(content)
        manifest["files"][target_name] = {
            "source_path": source_path,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    (TARGET / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"fixtures": len(FILES), "target": str(TARGET)}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import importlib
import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/signal-lab-uchicago/pycvxcluster-0.1.0.git"
REPO_DIR_NAME = "pycvxcluster-0.1.0"
PKG_SUBDIR = "pycvxcluster_pkg"    


def main() -> None:
    try:
        importlib.import_module("pycvxcluster.pycvxcluster")
        print("✅ pycvxcluster is already installed and importable.")
        return
    except ImportError:
        print("ℹ️  pycvxcluster not found. Cloning and installing...")

    root = Path(__file__).resolve().parents[1]   # repo root
    repo_dir = root / REPO_DIR_NAME
    pkg_dir = repo_dir / PKG_SUBDIR

    if not repo_dir.exists():
        print(f"📥 Cloning {REPO_URL} into {repo_dir} ...")
        subprocess.check_call(
            ["git", "clone", REPO_URL, str(repo_dir)],
            cwd=root,
        )
    else:
        print(f"📂 Using existing clone at {repo_dir}")


    importlib.import_module("pycvxcluster.pycvxcluster")
    print("✅ Successfully installed and imported pycvxcluster.")


if __name__ == "__main__":
    main()
"""Freeze launch artifacts, environment evidence and independently gated pilots."""
from __future__ import annotations
from pathlib import Path
from contextlib import redirect_stdout
import datetime
import io
import json
import os
import platform
import shutil
import subprocess

from .artifacts import atomic_json, sha256_file
from .cli import source_identity
from .config import default_config, DATASETS, fingerprint
from .data import _verify_files
from .manifests import build_base_manifests, write_launch_manifests
from .pilots import write_pilot_manifest, _correctness_evidence, _smoke_evidence


def _command(command):
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=30)
        return {"command": command, "exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "unavailable": str(exc)}


def environment_evidence(root):
    import numpy as np
    import scipy
    import pandas as pd
    import sklearn
    import importlib.metadata as metadata
    from threadpoolctl import threadpool_info
    stream = io.StringIO()
    with redirect_stdout(stream):
        np.show_config()
    packages = {str(distribution.metadata.get("Name", "unknown")): distribution.version
                for distribution in metadata.distributions()}
    return {"accessed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "python": platform.python_version(), "python_compiler": platform.python_compiler(),
            "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "packages": packages,
            "operating_system": platform.platform(), "uname": list(platform.uname()),
            "numpy_configuration": stream.getvalue(), "loaded_threadpools": threadpool_info(),
            "gcc": _command(["gcc", "--version"]), "R_available_in_job": _command(["R", "--version"]),
            "R_original_environment": str(Path(root) / "environments/R-sessionInfo.txt"),
            "CUDA": "not used; all specified pilot/production adapters use CPU environments",
            "thread_environment": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]},
            "runtime_library_path": os.environ.get("LD_LIBRARY_PATH"),
            "slurm": _command(["scontrol", "--version"])}


def external_evidence(root):
    output = []
    references = Path(root) / "external_references"
    if not references.is_dir():
        return output
    for path in sorted(references.iterdir()):
        if not path.is_dir():
            continue
        commit = _command(["git", "-C", str(path), "rev-parse", "HEAD"])
        remote = _command(["git", "-C", str(path), "remote", "get-url", "origin"])
        licenses = [p for p in path.iterdir() if p.is_file() and p.name.upper().startswith(("LICENSE", "LICENCE", "COPYING"))]
        output.append({"path": str(path), "commit": commit, "remote": remote,
                       "license_files": {str(p): sha256_file(p) for p in licenses},
                       "license_issue": "reuse terms require attribution audit" if not licenses else None,
                       "read_only_reference_use": True,
                       "accessed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    return output


def prepare_launch(root, config=None):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("launch preflight including full data hashes must run within Slurm")
    root = Path(root)
    os.environ["GPLSI_BENCHMARK_ROOT"] = str(root.resolve())
    config = config or default_config()
    source = source_identity()
    source_directory = Path(__file__).resolve().parents[2]
    dirty = subprocess.check_output(["git", "-C", str(source_directory), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty:
        raise RuntimeError("launch requires a committed clean tracked source snapshot")
    contracts, splits = {}, []
    for dataset in DATASETS:
        cohort = root / "data/processed/joint_v2" / dataset
        contract = json.loads((cohort / "contract.json").read_text())
        _verify_files(cohort, contract["artifacts"])
        contracts[dataset] = contract
        splits.extend(json.loads((root / "data/manifests/joint_v2/splits" / dataset / "splits.json").read_text()))
    version = fingerprint({"code": source["source_tree_hash"], "config": config, "contracts": contracts})[:20]
    report_directory = root / "reports/joint_v2" / version
    report_directory.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "environments/joint_v2" / f"system_{version}.json", environment_evidence(root))
    atomic_json(report_directory / "external_software.json", external_evidence(root))
    for path in (source_directory / "joint_v2_docs").iterdir():
        destination = report_directory / path.name
        if destination.exists() and sha256_file(destination) != sha256_file(path):
            raise ValueError("versioned report already exists with different content")
        if not destination.exists():
            shutil.copyfile(path, destination)
    manifests = build_base_manifests(splits, config)
    launch_selection = write_launch_manifests(
        root / "data/manifests/joint_v2" / f"production_{version}",
        manifests, config, source["source_tree_hash"], contracts)
    production = Path(launch_selection["initial_manifest"])
    for manifest_path in [production, Path(launch_selection["deferred_manifest"])]:
        atomic_json(manifest_path.parent / "source_identity.json", source)
    pilots = write_pilot_manifest(root, config, source, contracts)
    resources = {"default": {"cpus": 2, "memory_gb": 64, "time": "12:00:00", "partition": config["resources"]["default_partition"],
                             "basis": "full-size pilot resource cap; production must use measured resources"}}
    for dataset in DATASETS:
        resources[dataset + "/prepare"] = {"cpus": 2, "memory_gb": 32, "time": "00:30:00", "partition": config["resources"]["default_partition"],
                                            "basis": "sparse count preparation pilot cap"}
    resource_path = root / "configs/joint_v2" / f"pilot_resources_{version}.json"
    atomic_json(resource_path, resources)
    from .reporting_data import plot_frozen_split_masks
    masks = plot_frozen_split_masks(root)
    correctness = _correctness_evidence(root, source["source_tree_hash"], fingerprint(config))
    smoke = _smoke_evidence(root, source["source_tree_hash"], fingerprint(config))
    result = {"version": version, "source": source, "production_manifest": str(production),
              "pilot_manifest": pilots["manifest"], "pilot_resources": str(resource_path),
              "production_workload": launch_selection["initial_workload"], "pilot_workload": pilots,
              "launch_selection": launch_selection,
              "deferred_spatial_half_manifest": launch_selection["deferred_manifest"],
              "correctness": correctness, "smoke": smoke,
              "ready_for_full_size_pilots": correctness["passed"] and smoke["passed"],
              "production_approved": False, "split_mask_figures": masks,
              "report_directory": str(report_directory)}
    atomic_json(root / "reports/joint_v2/LAUNCH_STATUS.json", result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(prepare_launch(arguments.root), indent=2))

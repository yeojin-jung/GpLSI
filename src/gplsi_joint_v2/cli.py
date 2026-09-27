"""Auditable joint_v2 command line. Heavy actions require a Slurm allocation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess

from .artifacts import atomic_json, sha256_file
from .config import default_config, DATASETS
from .data import freeze_splits, inventory_cohort, materialize_cohort
from .manifests import build_base_manifests, build_dag, write_manifests, summarize_dag


def require_slurm():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("heavy computation must run within a Slurm allocation")


def source_identity():
    source = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    # Working-copy hashes make test snapshots explicit, never pretend dirty code
    # equals HEAD. Production gate additionally requires clean tracked source.
    files = {}
    for directory in ["src", "utils", "codes", "joint_v2_scripts"]:
        for p in sorted((source / directory).rglob("*")):
            if p.is_file() and p.suffix.lower() in {".py", ".r", ".sh", ".sbatch", ".pyx"}:
                files[str(p.relative_to(source))] = sha256_file(p)
    root = Path(os.environ.get("GPLSI_BENCHMARK_ROOT", str(source))).resolve()
    environment = {}
    for directory in [root / "environments", root / "environments/joint_v2"]:
        for name in ["python-requirements.lock.txt", "conda-linux-64.lock.txt", "environment.lock.yml", "renv.lock"]:
            path = directory / name
            if path.is_file():
                environment[str(path.relative_to(root))] = sha256_file(path)
    for name in ["pyproject.toml", "environment.yaml"]:
        path = source / name
        if path.is_file():
            environment[name] = sha256_file(path)
    from .config import fingerprint
    return {"commit": commit, "source_tree_hash": fingerprint({"files": files, "environment_locks": environment}),
            "files": files, "environment_locks": environment}


def freeze(root, config):
    rows = []
    for dataset in DATASETS:
        rows.extend(freeze_splits(root, dataset, config["seeds"]["outer_split_seed"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inventory", "freeze", "materialize", "manifest", "stage", "audit", "smoke", "pilots", "assess-pilots", "aggregate"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--task-id")
    parser.add_argument("--run-name", default="production_candidate_01")
    args = parser.parse_args()
    os.environ["GPLSI_BENCHMARK_ROOT"] = str(args.root.resolve())
    config = json.loads(args.config.read_text()) if args.config else default_config()
    if args.action == "inventory":
        output = [inventory_cohort(args.root, d) for d in ([args.dataset] if args.dataset else DATASETS)]
    elif args.action == "freeze":
        output = freeze(args.root, config)
    elif args.action == "materialize":
        require_slurm()
        output = [str(materialize_cohort(args.root, d)) for d in ([args.dataset] if args.dataset else DATASETS)]
    elif args.action == "manifest":
        splits = freeze(args.root, config)
        identity = source_identity()
        sources = {d: json.loads((args.root / "data/processed/joint_v2" / d / "contract.json").read_text())
                   for d in DATASETS}
        manifests = build_base_manifests(splits, config)
        dag = build_dag(manifests, config, identity["source_tree_hash"], sources)
        directory = args.root / "data/manifests/joint_v2" / args.run_name
        path = write_manifests(directory, manifests, dag)
        atomic_json(directory / "source_identity.json", identity)
        output = {"manifest": str(path), **summarize_dag(dag, manifests)}
    elif args.action == "stage":
        require_slurm()
        from .runner import run_stage
        output = run_stage(args.root, args.manifest, args.task_id)
    elif args.action == "audit":
        from .runner import audit_manifest
        output = audit_manifest(args.root, args.manifest)
    elif args.action == "pilots":
        from .pilots import write_pilot_manifest
        contracts = {d: json.loads((args.root / "data/processed/joint_v2" / d / "contract.json").read_text()) for d in DATASETS}
        output = write_pilot_manifest(args.root, config, source_identity(), contracts)
    elif args.action == "assess-pilots":
        from .pilots import assess_pilots
        output = assess_pilots(args.root, args.manifest)
    elif args.action == "aggregate":
        require_slurm()
        from .results_reporting import report_results
        output = report_results(args.root, args.manifest)
    else:
        require_slurm()
        from .smoke import run_smoke
        output = run_smoke(args.root, config)
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()

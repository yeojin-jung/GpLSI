"""Immutable scientific task graph, separate primary/stability/thinning inventories."""
from __future__ import annotations

from collections import Counter
from contextlib import closing
from pathlib import Path
import json
import os
import sqlite3

from .artifacts import atomic_json
from .config import fingerprint


def scientific_task_key(stage, spec, parents, *, code_hash, config, source_hashes):
    return fingerprint({"stage": stage, "spec": spec, "parents": sorted(parents),
                        "code_hash": code_hash, "config": config, "source_hashes": source_hashes})


def validate_task_identity(task, dag):
    expected = scientific_task_key(task["stage"], task["spec"], task["parents"],
                                   code_hash=dag["code_hash"], config=dag["config"],
                                   source_hashes=dag["source_hashes"])
    if task["task_id"] != expected:
        raise ValueError("scientific task identity does not match immutable specification")
    return True


def build_base_manifests(splits, config):
    manifests = {"core": [], "stability": [], "thinning": []}
    for split in splits:
        dataset = split["dataset"]
        protocol = split["protocol"]
        base = {"dataset": dataset, "protocol": protocol, "outer_split_id": split["split_id"],
                "split_hash": fingerprint(split), **config["seeds"]}
        panels = config["visium_panels"] if dataset == "visium_dlpfc" else [config["primary_panel"][dataset]]
        for panel in panels:
            for K in config["K_values"]:
                manifests["core"].append({**base, "K": K, "panel_requested": panel, "retention": 1.0})
        primary = {**base, "K": config["primary_K"][dataset],
                   "panel_requested": config["primary_panel"][dataset]}
        for estimator_seed in config["initialization_seeds"]:
            manifests["stability"].append({**primary, "estimator_seed": estimator_seed, "retention": 1.0})
        for retention in config["retention_secondary"]:
            manifests["thinning"].append({**primary, "retention": retention})
    for tier, records in manifests.items():
        for record in records:
            record["base_id"] = fingerprint(record)[:24]
    return manifests


INITIAL_LAUNCH_POLICY = {
    "selection": "defer_visium_spatial_half_only",
    "excluded_dataset": "visium_dlpfc",
    "excluded_protocol": "spatial_half",
    "deferred_reason": "User deferred Visium spatial-half experiments until a later launch.",
    "frozen_split_definitions": "retained without changes",
    "scientific_task_identity": "unchanged by initial/deferred launch membership",
    "deferred_launch_authorized": False,
}


def split_launch_manifests(manifests):
    """Partition scheduling inventories, without changing scientific records.

    The dataset AND protocol must match: no other protocol or dataset is
    excluded. All core, stability, and thinning memberships remain intact.
    """
    selected = {"initial": {tier: [] for tier in manifests},
                "deferred_visium_spatial_half": {tier: [] for tier in manifests}}
    for tier, rows in manifests.items():
        for base in rows:
            deferred = (base["dataset"] == "visium_dlpfc"
                        and base["protocol"] == "spatial_half")
            key = "deferred_visium_spatial_half" if deferred else "initial"
            selected[key][tier].append(base)
    return selected


def _combine_disjoint_workloads(summaries):
    """Exact union summary without persisting a duplicate full DAG/database."""
    summaries = list(summaries)
    tiers = Counter(); stages = Counter(); variants = Counter()
    for summary in summaries:
        tiers.update(summary["base_manifest_sizes"])
        for row in summary["by_dataset_stage"]:
            stages[row["dataset"], row["stage"]] += row["tasks"]
        for row in summary["by_variant"]:
            variants[row["dataset"], row["stage"], row["variant"]] += row["tasks"]
    return {
        "base_manifest_sizes": dict(tiers),
        "unique_base_tasks": sum(s["unique_base_tasks"] for s in summaries),
        "stage_tasks_total": sum(s["stage_tasks_total"] for s in summaries),
        "by_dataset_stage": [{"dataset": d, "stage": s, "tasks": n}
                             for (d, s), n in sorted(stages.items())],
        "by_variant": [{"dataset": d, "stage": s, "variant": v, "tasks": n}
                       for (d, s, v), n in sorted(variants.items())],
        "storage": "union of disjoint initial and deferred manifests; no duplicate full DAG",
    }


def write_launch_manifests(directory, manifests, config, code_hash, source_hashes):
    """Write initial/deferred DAGs plus the small, complete frozen base inventory.

    Scientific config, base records, and task keys are identical to build_dag on
    the full inventory. Launch policy lives outside the scientific identity.
    No submission or launch authorization occurs here.
    """
    directory = Path(directory)
    selected = split_launch_manifests(manifests)
    paths = {}; summaries = {}; task_ids = set(); base_ids = set()
    for selection, rows in selected.items():
        dag = build_dag(rows, config, code_hash, source_hashes)
        selected_task_ids = {task["task_id"] for task in dag["tasks"]}
        selected_base_ids = {base["base_id"] for base in dag["bases"]}
        if selected_task_ids & task_ids or selected_base_ids & base_ids:
            raise ValueError("Initial/deferred inventories unexpectedly share scientific tasks")
        task_ids.update(selected_task_ids); base_ids.update(selected_base_ids)
        dag["launch_selection"] = {**INITIAL_LAUNCH_POLICY, "membership": selection}
        # The scheduler uses the manifest's immediate parent name as its ledger
        # namespace. Include the versioned run directory name, not just initial.
        selection_directory = directory / f"{selection}_{directory.name}"
        paths[selection] = str(write_manifests(selection_directory, rows, dag))
        summaries[selection] = summarize_dag(dag, rows)
        del dag
    result = {
        "launch_policy": dict(INITIAL_LAUNCH_POLICY),
        "initial_manifest": paths["initial"],
        "deferred_manifest": paths["deferred_visium_spatial_half"],
        "full_inventory": str(directory / "full_base_inventory.json"),
        "selection_manifest": str(directory / "launch_selection.json"),
        "initial_workload": summaries["initial"],
        "deferred_workload": summaries["deferred_visium_spatial_half"],
        "full_design_workload": _combine_disjoint_workloads(summaries.values()),
        "code_hash": code_hash, "config_hash": fingerprint(config),
        "source_hashes_digest": fingerprint(source_hashes),
    }
    for name, value in (("full_base_inventory.json", manifests), ("launch_selection.json", result)):
        path = directory / name
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise FileExistsError(f"immutable manifest differs: {path}")
        else:
            atomic_json(path, value)
    return result


def build_dag(manifests, config, code_hash, source_hashes):
    if config["recoveries"] != ["A_full_Pois"]:
        raise ValueError("The active benchmark requires A_full_Pois for every reported topic profile")
    nodes = {}
    all_bases = {}
    memberships = {}
    for tier, rows in manifests.items():
        for base in rows:
            all_bases[base["base_id"]] = base
            memberships.setdefault(base["base_id"], []).append(tier)

    def add(stage, spec, parents, base_id, variant):
        # Scientific identity, including all effective defaults, participates
        # in keys. Array location and requested resources intentionally do not.
        key = scientific_task_key(stage, spec, parents, code_hash=code_hash,
                                  config=config, source_hashes=source_hashes)
        if key not in nodes:
            nodes[key] = {"task_id": key, "stage": stage, "spec": spec,
                          "parents": sorted(parents), "base_ids": [], "variant": variant,
                          "dataset": spec["dataset"], "code_hash": code_hash,
                          "source_hashes_digest": fingerprint(source_hashes), "config_hash": fingerprint(config)}
        if base_id not in nodes[key]["base_ids"]:
            nodes[key]["base_ids"].append(base_id)
        return key

    for base_id, base in sorted(all_bases.items()):
        # Scope follows the scientific seed, not tier membership. Subsetting a
        # manifest must never alter which tasks or keys this same base produces.
        extra_initialization = base["estimator_seed"] != config["seeds"]["estimator_seed"]
        initialization_scope = config["initialization_scope"]
        hunters = [hunter for hunter in config["hunters"] if not extra_initialization
                   or hunter in initialization_scope["document_hunters"]]
        anchor = not extra_initialization or initialization_scope["anchor"]
        competitors = [method for method in config["competitors"] if not extra_initialization
                       or method in initialization_scope["competitors"]]
        data_spec = {k: base[k] for k in ("dataset", "protocol", "outer_split_id", "split_hash",
                     "panel_requested", "retention", "molecule_seed", "outer_split_seed")}
        prepared = add("prepare", data_spec, [], base_id, "training_counts_and_test_adapt_score")
        scientific = {k: v for k, v in base.items() if k != "base_id"}
        spectral = {}
        W_sources = []
        for prep in config["preprocessings"]:
            if not hunters and not (prep == "P0_raw" and (anchor or "topicscore_graph_denoised" in competitors)):
                continue
            spectral[prep] = add("spectral", {**scientific, "preprocessing": prep}, [prepared], base_id, prep)
            if (base["K"] == config["primary_K"][base["dataset"]]
                and base["panel_requested"] == config["primary_panel"][base["dataset"]]
                and base["estimator_seed"] == config["seeds"]["estimator_seed"]
                and base["retention"] == 1.0):
                add("diagnostics", {**scientific, "preprocessing": prep},
                    [prepared, spectral[prep]], base_id, prep + "__lambda_path")
            families = [("document", hunter) for hunter in hunters]
            if prep == "P0_raw" and anchor:
                families.append(("anchor", "spa_current"))
            for family, hunter in families:
                for control in ("selected", "lambda_zero"):
                    variant = f"gplsi_{family}__{prep}__{hunter}__{control}"
                    geometry_spec = {**scientific, "preprocessing": prep, "family": family,
                                     "hunter": hunter, "control": control, "method": variant}
                    geometry = add("geometry", geometry_spec, [prepared, spectral[prep]], base_id, variant)
                    W_sources.append((geometry, geometry_spec))
                    for recovery in config["recoveries"]:
                        r_spec = {**geometry_spec, "recovery": recovery, "vocabulary": "native"}
                        rec = add("recovery", r_spec, [prepared, geometry], base_id, variant + "__" + recovery)
                        add("evaluation", r_spec, [prepared, rec], base_id, variant + "__" + recovery)
        for method in competitors:
            c_spec = {**scientific, "method": method}
            parents = [prepared]
            if method == "topicscore_graph_denoised":
                parents.append(spectral["P0_raw"])
            competitor = add("competitor", c_spec, parents, base_id, method)
            W_sources.append((competitor, c_spec))
            # Preserve the competitor's fitted W, but standardize every reported
            # A with the same fixed-W raw-count Poisson recovery. Internally
            # fitted native A is implementation evidence/warm start, not output.
            r_spec = {**c_spec, "vocabulary": "native", "recovery": "A_full_Pois"}
            rec = add("recovery", r_spec, [prepared, competitor], base_id,
                      method + "__A_full_Pois")
            add("evaluation", r_spec, [prepared, rec], base_id, method + "__A_full_Pois")
        if base["dataset"] == "visium_dlpfc":
            for w_source, spec in W_sources:
                ref_spec = {**spec, "recovery": "A_full_Pois", "vocabulary": "common_reference"}
                rec = add("reference_recovery", ref_spec, [prepared, w_source], base_id,
                          spec["method"] + "__reference")
                add("evaluation", ref_spec, [prepared, rec], base_id, spec["method"] + "__reference")
    return {"schema_version": 2, "code_hash": code_hash, "config": config,
            "config_hash": fingerprint(config), "source_hashes": source_hashes,
            "base_memberships": memberships, "bases": list(all_bases.values()),
            "tasks": list(nodes.values())}


def summarize_dag(dag, manifests):
    counts = Counter((t["dataset"], t["stage"]) for t in dag["tasks"])
    variants = Counter((t["dataset"], t["stage"], t["variant"]) for t in dag["tasks"])
    return {"base_manifest_sizes": {k: len(v) for k, v in manifests.items()},
            "unique_base_tasks": len(dag["bases"]), "stage_tasks_total": len(dag["tasks"]),
            "by_dataset_stage": [{"dataset": d, "stage": s, "tasks": n} for (d, s), n in sorted(counts.items())],
            "by_variant": [{"dataset": d, "stage": s, "variant": v, "tasks": n}
                           for (d, s, v), n in sorted(variants.items())],
            "actual_panels": "Set from training-only ranking in each prepare effective configuration; never padded.",
            "production_gate": "correctness -> smoke -> full-size platform/panel pilots -> measured resource gate"}


def write_manifests(directory, manifests, dag):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # A versioned run directory is write-once. Existing compatible manifests
    # are a resume; incompatible ones must get a new configuration directory.
    outputs = {f"{tier}.json": rows for tier, rows in manifests.items()}
    outputs.update({"dag.json": dag, "workload.json": summarize_dag(dag, manifests)})
    for name, value in outputs.items():
        path = directory / name
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise FileExistsError(f"immutable manifest differs: {path}")
        else:
            atomic_json(path, value)
    database = directory / "tasks.sqlite"
    if not database.exists():
        temporary = directory / f"tasks.partial.{os.getpid()}.sqlite"
        if temporary.exists():
            raise FileExistsError("unexpected prior manifest build; choose new run directory")
        with closing(sqlite3.connect(temporary)) as db, db:
            db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            db.execute("CREATE TABLE tasks (task_id TEXT PRIMARY KEY,stage TEXT,dataset TEXT,value TEXT NOT NULL)")
            db.executemany("INSERT INTO metadata VALUES (?,?)", [(key, json.dumps(value)) for key, value in dag.items() if key != "tasks"])
            db.executemany("INSERT INTO tasks VALUES (?,?,?,?)", [(row["task_id"],row["stage"],row["dataset"],json.dumps(row)) for row in dag["tasks"]])
        os.replace(temporary, database)
    return directory / "dag.json"

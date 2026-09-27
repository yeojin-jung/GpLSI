"""Print exact prespecified workload and conservative uncompressed disk bounds."""
from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
import json
import shutil

from .artifacts import atomic_json
from .config import default_config, DATASETS
from .manifests import (INITIAL_LAUNCH_POLICY, _combine_disjoint_workloads,
                        build_base_manifests, build_dag, split_launch_manifests, summarize_dag)


def preflight(root, config=None, *, write=True):
    root = Path(root)
    config = config or default_config()
    splits = []
    for dataset in DATASETS:
        splits.extend(json.loads((root / "data/manifests/joint_v2/splits" / dataset / "splits.json").read_text()))
    manifests = build_base_manifests(splits, config)
    lookup = {(s["dataset"], s["split_id"]): s for s in splits}
    workloads = {}; sizes = {}
    for selection, selected in split_launch_manifests(manifests).items():
        dag = build_dag(selected, config, "preflight_count_only_not_a_production_cache_key", {})
        workloads[selection] = summarize_dag(dag, selected)
        size = defaultdict(int)
        for task in dag["tasks"]:
            spec, stage = task["spec"], task["stage"]
            split = lookup[(spec["dataset"], spec["outer_split_id"])]
            n = split["role_counts"]["train"]
            K, p = spec.get("K", 0), spec["panel_requested"]
            if stage == "spectral":
                size[spec["dataset"]] += 8 * (2*n*K + 2*p*K)
            elif stage in ("geometry", "competitor"):
                size[spec["dataset"]] += 8 * (n*K + (0 if stage == "competitor" else K*K))
            elif stage in ("recovery", "reference_recovery"):
                size[spec["dataset"]] += 8*K*(max(config["visium_panels"]) if stage == "reference_recovery" else p)
            elif stage == "evaluation":
                # Per-row scores and transferred W are transient. This is a
                # provisional compact-summary allowance, replaced by pilot bytes.
                size[spec["dataset"]] += 128 * 1024
        sizes[selection] = dict(size)
        del dag
    size = sizes["initial"]
    capacity = shutil.disk_usage(root)
    disk = {"uncompressed_factor_and_numeric_scores_bytes_by_dataset": dict(size),
            "uncompressed_factor_and_numeric_scores_bytes": sum(size.values()),
            "scope": "initial launch only; deferred Visium spatial-half storage excluded",
            "deferred_factor_and_numeric_scores_bytes_by_dataset": sizes["deferred_visium_spatial_half"],
            "deferred_factor_and_numeric_scores_bytes": sum(sizes["deferred_visium_spatial_half"].values()),
            "excluded_from_estimate": ["compressed sparse count split caches", "observation-ID strings",
                                       "CV histories and metadata", "figures and bootstrap summaries"],
            "compression": "NPZ/Parquet reduce disk use; ratio must be measured, not assumed",
            "removed_from_persistence": ["per-observation score tables", "all transferred W arrays",
                                         "redundant spectral centers", "repeated observation ID tables"],
            "identical_selected_zero_alias": "additional reduction where exact equality holds, not assumed in estimate",
            "filesystem_free_bytes_at_preflight": capacity.free,
            "production_disk_gate": "measured extrapolation + 25% overhead < 75% of current free project space"}
    table = []
    for s in splits:
        if "assignments" in s:
            detail = " ; ".join(f"{a['bio_id']}: train={','.join(a['train_sections'])}, primary={','.join(a['primary_test'])}, additional={','.join(a['additional_test'])}" for a in s["assignments"])
        elif "test_biological_ids" in s:
            detail = "held out=" + ",".join(s["test_biological_ids"]) + "; train=complement"
        else:
            detail = "coordinate-only half; exact per-section thresholds/ties in split JSON"
        table.append({"dataset": s["dataset"], "split_id": s["split_id"], "protocol": s["protocol"],
                      "assignment": detail, "role_counts": s["role_counts"],
                      "launch_membership": "deferred_visium_spatial_half" if
                          s["dataset"] == "visium_dlpfc" and s["protocol"] == "spatial_half" else "initial"})
    document_w = len(config["preprocessings"]) * len(config["hunters"]) * 2
    anchor_w = 2 if "P0_raw" in config["preprocessings"] else 0
    competitors = len(config["competitors"])
    all_w = document_w + anchor_w + competitors
    initial_splits = sum(s["launch_membership"] == "initial" for s in table)
    output = {"frozen_splits": table, "lambda_grid": config["graph_cv"], "workload": workloads["initial"],
              "deferred_workload": workloads["deferred_visium_spatial_half"],
              "full_design_workload": _combine_disjoint_workloads(workloads.values()),
              "launch_policy": dict(INITIAL_LAUNCH_POLICY),
              "initialization": {"core_and_thinning_seed": config["seeds"]["estimator_seed"],
                                 "primary_K_panel_total_seeds": config["initialization_seeds"],
                                 "scope": config["initialization_scope"],
                                 "primary_seed_shared_with_core": True,
                                 "invariance_claim": "None: seed repetitions probe stability but do not prove invariance"},
              "storage": disk, "resources": config["resources"],
              "pilots": {"CPU_per_stage": 2, "memory_GB": 64, "wall_hours": 12,
                         "note": "resource requests are provisional caps, not a measured ETA; production requires pilot results"},
              "scientific_variants_per_base": {
                 "document_W": f"{len(config['preprocessings'])} preprocessings x {len(config['hunters'])} hunters x 2 lambda controls ={document_w}",
                 "document_hunters": config["hunters"],
                 "anchor_W": f"P0 SPA x 2 lambda controls ={anchor_w}",
                 "native_GpLSI_A": f"{document_w + anchor_w} W x fixed-W Poisson ={document_w + anchor_w}",
                 "competitors": f"{competitors} shared-profile W fits; every reported A uses fixed-W Poisson recovery",
                 "native_outputs": all_w,
                 "native_A_recovery": "A_full_Pois only; internally fitted competitor A is not reported",
                 "Visium_reference_A": f"{all_w} W sources x fixed-W Poisson ={all_w}",
                 "diagnostics": f"{len(config['preprocessings'])} preprocessing paths on each of {initial_splits} initial primary configs; all {len(config['hunters'])} configured hunters per lambda"}}
    if write:
        atomic_json(root / "reports/joint_v2/PREFLIGHT.json", output)
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(preflight(args.root), indent=2))

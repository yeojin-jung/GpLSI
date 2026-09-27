"""Prespecified scientific and resource configuration, serialized before fitting."""
from __future__ import annotations

import hashlib
import json
import os

DATASETS = ("visium_dlpfc", "merfish_trem2_5xfad", "xenium_uc")
PREPROCESSINGS = ("P0_raw", "P1_tran_alpha_0p005", "P2_ke_weighted", "P3_tran_then_ke")
HUNTERS = ("spa_current", "svs_star", "palm_accelerated")
RECOVERIES = ("A_full_Pois",)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def default_config():
    partitions = [value.strip() for value in os.environ.get("GPLSI_SLURM_PARTITIONS", "").split(",")
                  if value.strip()]
    partition = os.environ.get("GPLSI_SLURM_PARTITION") or (partitions[0] if partitions else None)
    if partition and partition not in partitions:
        partitions.append(partition)
    return {
        "schema_version": 2, "experiment": "spatial-joint-cohort-v2",
        "K_values": [7, 10, 12, 15, 20],
        "visium_panels": [500, 2000, 5000, 10000, 15000],
        "primary_K": {"visium_dlpfc": 7, "merfish_trem2_5xfad": 12, "xenium_uc": 12},
        "primary_panel": {"visium_dlpfc": 2000, "merfish_trem2_5xfad": 300, "xenium_uc": 290},
        "seeds": {"outer_split_seed": 26091001, "molecule_seed": 26091002,
                  "graph_cv_seed": 26091003, "estimator_seed": 26091004},
        "initialization_seeds": [26091004, 26091005, 26091006],
        "initialization_scope": {
            "document_hunters": list(HUNTERS), "anchor": True,
            "competitors": ["topicscore_raw", "topicscore_graph_denoised", "lda", "kl_nmf",
                            "graph_kl_nmf_legacy_0p25", "graph_kl_nmf_tuned"],
            "excluded_competitors": {
                "spatial_lda_legacy_0p25": "Calico fixes its internal seed at zero",
                "spatial_lda_tuned": "Calico fixes its internal seed at zero"}},
        "retention_secondary": [0.75, 0.50, 0.25], "score_fraction": 0.2,
        "preprocessings": list(PREPROCESSINGS), "hunters": list(HUNTERS),
        "recoveries": list(RECOVERIES), "matched_zero_controls": True,
        "graph": {"neighbors": 6, "stratum_blocked": True,
                  "weights": "exp(-(d/median_positive_neighbor_distance)^2)",
                  "penalty_convention": "legacy_unnormalized_sum_unique_undirected_edges"},
        "graph_cv": {"folds": 5, "initial_grid": sorted(set([0.0, 1e-6] + [1e-4 * 1.7**j for j in range(29)])),
                     "growth": 1.7, "extension_batch": 5, "max_candidates": 51,
                     "lambda_ceiling": 1e8, "plateau_relative_tolerance": 1e-4,
                     "tie_rule": "smallest_lambda", "max_spectral_iterations": 200,
                     "spectral_tolerance": 1e-5, "max_cv_seconds": 86400,
                     "fold_workers": 1, "inner_preprocessing": "fold_fitted_v2"},
        "solvers": {"A_max_iter": 10000, "A_tolerance": 1e-8,
                    "foldin_max_iter": 5000, "foldin_tolerance": 1e-7,
                    "entry_chunk_size": 100000, "row_chunk_size": 256,
                    "factor_storage": "float64", "scoring_floor_diagnostic": 1e-12,
                    "svs_face_solve_budget": 1000000},
        "competitors": ["topicscore_raw", "topicscore_graph_denoised", "lda", "kl_nmf",
                        "spatial_lda_legacy_0p25", "graph_kl_nmf_legacy_0p25",
                        "spatial_lda_tuned", "graph_kl_nmf_tuned"],
        "competitor_tuning": {"folds": 5, "penalties": [0.0, 0.025, 0.25, 2.5],
                              "spatial_lda_inverse_penalties": [0.025, 0.25, 2.5, 25.0],
                              "selection": "training_only_inner_conditional_deviance",
                              "spatial_lda": "adapter_capability_gate"},
        "resources": {"account": os.environ.get("GPLSI_SLURM_ACCOUNT") or None, "partitions": partitions,
                      "default_partition": partition, "global_concurrency": 24,
                      "max_submitted_stage_tasks": 512, "array_cap": 8,
                      "pilot_cpus": 2, "pilot_memory_gb": 64,
                      "pilot_time": "12:00:00", "production_requires_measured_pilots": True},
        "bootstrap": {"draws": 2000, "unit": "biological_subject",
                      "claim": "conditional_on_fitted_models"},
        "storage": {"training_W": "one_float64_compressed_artifact_per_geometry_or_competitor",
                    "A_profiles": "keep_native_and_common_reference",
                    "transfer_W": "transient_during_evaluation",
                    "per_observation_scores": "transient_aggregate_to_biological_and_section_sufficient_statistics",
                    "observation_ids": "shared_prepared_row_contract",
                    "spectral_centers": "not_persisted_exact_diagnostic_scalars_retained_in_metadata",
                    "identical_selected_zero_factors": "store_once_with_alias",
                    "legacy_result_deletion": False},
    }

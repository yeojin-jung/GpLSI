"""Training-only lambda-path diagnostics; never consulted for lambda selection.

One embedding is fitted at a time and shared across hunters. Only scalar or
topic/stratum records are retained; n-by-K memberships are released each time.
The graph-CV scores are joined after fitting, and no labels enter this API.
"""
from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

import numpy as np
from scipy import sparse

from .geometry import GeometryResourceBlocked, fit_geometry
from .spectral import (SpectralConfig, fit_fixed_lambda, learn_feature_plan,
                       transformed_frequencies, graph_penalty, graph_roughness)


HUNTERS = ("spa_current", "svs", "svs_star", "pp_spa", "palm", "palm_accelerated")


def embedding_center_diagnostics(block, graph):
    """Use exact recorded center summaries when n-by-K centers are not stored.

    The source U_bar was formed before the final V update. A smoother rerun on
    the saved final V would be a different computation, so it is never used to
    reconstruct these historical diagnostic values.
    """
    last_smoother = block.metadata.get("final_smoother")
    if last_smoother is None:
        history = block.metadata.get("history", [])
        if not history:
            raise ValueError("embedding lacks its final smoother diagnostics")
        last_smoother = history[-1]["smoother"]
    if block.U_bar is None:
        try:
            penalty = float(last_smoother["graph_penalty"])
            roughness = float(block.metadata["unorthogonalized_embedding_roughness"])
        except KeyError as exc:
            raise ValueError("omitted centers require exact saved penalty and roughness summaries") from exc
    else:
        penalty = graph_penalty(block.U_bar, graph)
        roughness = graph_roughness(block.U_bar, graph)
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("invalid stored graph-center penalty")
    return {"graph_penalty": penalty, "roughness": roughness,
            "objective": float(last_smoother["objective"])}


def run_lambda_diagnostics(counts, adjacency, coordinates, strata, K, preprocessing, *,
                           lambdas, estimator_seed, requested_panel=None,
                           final_panel_indices=None, config=None, hunters=HUNTERS,
                           cv_metadata=None, reused_embeddings=None,
                           max_svs_face_solves=1_000_000,
                           max_diameter_pairs=200_000_000, record_callback=None):
    """Final-data refits along a frozen CV path; returns no per-lambda factors.

    ``reused_embeddings`` can supply saved selected/zero factors by exact lambda.
    Caller must validate their complete input/configuration hash before reuse.
    ``record_callback`` receives completed records for restartable stage logging.
    """
    from .metrics import spatial_metrics, pas10_neighbors
    config = config or SpectralConfig()
    D = sparse.csr_matrix(counts)
    graph = sparse.csr_matrix(adjacency)
    coordinates, strata = np.asarray(coordinates, float), np.asarray(strata, str)
    if coordinates.shape != (D.shape[0], 2) or strata.shape != (D.shape[0],):
        raise ValueError("diagnostic coordinates/strata must match frozen training rows")
    if graph.shape != (D.shape[0], D.shape[0]):
        raise ValueError("diagnostic graph shape mismatch")
    edges = graph.tocoo()
    if np.any(strata[edges.row] != strata[edges.col]):
        raise ValueError("diagnostic graph crosses forbidden strata")
    grid = sorted(set(float(value) for value in lambdas))
    if not grid or grid[0] != 0 or any(value < 0 or not np.isfinite(value) for value in grid):
        raise ValueError("diagnostic path must contain exact zero and finite nonnegative lambdas")
    plan = learn_feature_plan(D, np.ones(D.shape[0], dtype=bool), preprocessing,
                              requested_panel=requested_panel, panel_indices=final_panel_indices, config=config)
    if np.any(plan.row_totals == 0):
        raise ValueError("use the shared count-defined native training row mask")
    X = transformed_frequencies(D, plan)
    correction = np.asarray(X.sum(axis=0)).ravel() / plan.metadata["N_mean"]
    squared_input_norm = float(X.multiply(X).sum())
    neighbors10 = pas10_neighbors(coordinates, strata)
    curves = {float(row["lambda"]): row for row in (cv_metadata or {}).get("aggregate", [])}
    spectral_records, geometry_records = [], []
    reused_embeddings = reused_embeddings or {}
    started = perf_counter()
    for lam in grid:
        block = None
        record = {"stage": "spectral_path", "lambda": lam, "preprocessing": preprocessing,
                  "K": K, "used_for_lambda_selection": False,
                  "CV_aggregate_score": curves.get(lam, {}).get("score"),
                  "CV_five_scores": curves.get(lam, {}).get("fold_scores"),
                  "CV_candidate_selectable": curves.get(lam, {}).get("selectable")}
        try:
            if lam in reused_embeddings:
                block = reused_embeddings[lam]
                if block.lambda_value != lam:
                    raise ValueError("reused embedding lambda does not match diagnostic key")
                fit_runtime = 0.
            else:
                tick = perf_counter()
                block = fit_fixed_lambda(X, graph, K, lam, seed=estimator_seed,
                                          config=config, correction=correction)
                fit_runtime = perf_counter() - tick
            centers = embedding_center_diagnostics(block, graph)
            penalty = centers["graph_penalty"]
            record.update(status="ok" if block.converged else "nonconverged", converged=block.converged,
                          iterations=block.iterations, runtime_seconds=fit_runtime,
                          reused_shared_embedding=lam in reused_embeddings,
                          initial_embedding_roughness=block.metadata["unregularized_initial_embedding_roughness"],
                          graph_embedding_roughness=graph_roughness(block.U, graph),
                          unorthogonalized_embedding_roughness=centers["roughness"],
                          optimized_unscaled_edge_group_norm=penalty,
                          optimized_lambda_times_edge_group_norm=lam * penalty,
                          last_smoother_objective=centers["objective"],
                          projection_reconstruction_error=float(np.sqrt(max(0., squared_input_norm - float(block.singular_values @ block.singular_values)))),
                          singular_values=block.singular_values.tolist(),
                          spectral_stopping=block.metadata["convergence_status"])
        except Exception as exc:
            record.update(status="spectral_failed", converged=False, error=str(exc))
        spectral_records.append(record)
        if record_callback is not None:
            record_callback(record)
        for hunter in hunters:
            row = {"stage": "geometry_path", "lambda": lam, "preprocessing": preprocessing,
                   "K": K, "hunter": hunter, "used_for_lambda_selection": False}
            if block is None:
                row.update(status="blocked_parent_spectral", error=record.get("error"))
            else:
                try:
                    geometry = fit_geometry(block, plan, graph, hunter=hunter, estimator_seed=estimator_seed,
                                            max_svs_face_solves=max_svs_face_solves,
                                            max_diameter_pairs=max_diameter_pairs)
                    spatial = spatial_metrics(geometry.W, graph, coordinates, strata,
                                              pas_neighbors=neighbors10)
                    row.update(status="ok" if block.converged else "parent_nonconverged",
                               spectral_converged=block.converged, geometry=geometry.metadata,
                               moran_I_equal_stratum_mean=spatial["moran_I_equal_stratum_mean"],
                               moran_I_observation_weighted=spatial["moran_I_observation_weighted"],
                               moran_valid_topic_strata=spatial["moran_valid_topic_strata"],
                               moran_expected_topic_strata=spatial["moran_expected_topic_strata"],
                               moran_per_stratum_topic=spatial["moran_per_stratum_topic"],
                               W_edge_squared_difference=spatial["W_edge_squared_difference"],
                               one_minus_PAS_10=spatial["one_minus_PAS_10"],
                               PAS_valid_observations=spatial["PAS_10"]["valid_observations"],
                               PAS_total_observations=spatial["PAS_10"]["total_observations"],
                               PAS_per_stratum=spatial["PAS_10"]["per_stratum"],
                               occupied_topics=spatial["occupied_topics"], topic_sizes=spatial["topic_sizes"])
                    del spatial, geometry
                except GeometryResourceBlocked as exc:
                    row.update(status="resource_blocked", error=str(exc), resource=exc.metadata)
                except Exception as exc:
                    row.update(status="geometry_failed", error=str(exc))
            geometry_records.append(row)
            if record_callback is not None:
                record_callback(row)
        del block
    return {"version": "joint_v2_training_lambda_diagnostic_path", "configuration": asdict(config),
            "grid": grid, "hunters": list(hunters), "feature_metadata": plan.metadata,
            "spectral_path": spectral_records, "geometry_path": geometry_records,
            "runtime_seconds": perf_counter() - started, "used_for_lambda_selection": False,
            "Moran_centering": "within_original_spatial_stratum",
            "PAS_neighbor_cache": "common_coordinate_only_10NN_self_excluded_within_stratum",
            "interpretation": "convex smoother penalty checks do not imply nonconvex final-W smoothness monotonicity"}

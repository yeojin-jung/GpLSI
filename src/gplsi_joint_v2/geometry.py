"""One independently restartable geometry stage from a shared embedding.

Five hunters call the pinned GpLSI implementation directly. pp-SPA uses its
same affine projection, exact diameter/radius, neighbors and affine SPA, with
bounded-memory exact diameter search and radius counts instead of n² arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from math import comb
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from .spectral import Embedding, FeaturePlan, graph_roughness


class GeometryResourceBlocked(RuntimeError):
    def __init__(self, message, metadata=None):
        super().__init__(message)
        self.metadata = metadata or {}
        self.diagnostics = self.metadata


@dataclass
class GeometryResult:
    W: np.ndarray
    vertices: np.ndarray
    raw_memberships: np.ndarray
    metadata: dict
    auxiliary_arrays: dict = field(default_factory=dict)


def vertex_resource_preflight(K, point_count, hunter, *, max_svs_face_solves=1_000_000,
                              max_diameter_pairs=200_000_000, parameters=None):
    settings = dict(parameters or {})
    record = {"hunter": hunter, "K": int(K), "points": int(point_count), "status": "feasible_preflight"}
    if hunter == "svs":
        L = int(settings.get("L", min(K + 2, point_count)))
        if settings.get("L_mode", "fixed") != "fixed":
            raise ValueError("the frozen joint grid uses documented fixed L=K+2 SVS")
        candidates = comb(L, K)
        face_solves = candidates * (L - K) * (2**K - 1)
        record.update(L=L, candidate_simplexes=candidates, exact_active_face_solves=face_solves,
                      max_svs_face_solves=max_svs_face_solves)
        if L <= K:
            record.update(status="mathematically_ineligible", reason="SVS requires L>K")
        elif face_solves > max_svs_face_solves:
            record.update(status="resource_blocked", reason="exact SVS active-face enumeration exceeds frozen budget")
    if hunter == "pp_spa":
        record.update(exact_diameter_worst_case_pairs=point_count * (point_count - 1) // 2,
                      max_diameter_pairs=max_diameter_pairs,
                      diameter_algorithm="exact_KD_tree_branch_and_bound_with_budget")
    return record


def exact_diameter(points, *, max_pair_comparisons=200_000_000, leafsize=32):
    """Exact Euclidean diameter without a dense/condensed n-by-n array.

    Axis-aligned bounding boxes provide rigorous upper bounds. A frozen budget
    limits point pair distance work; budget exhaustion raises, never returns a
    heuristic radius. Heap/node metadata remains O(n plus active node pairs).
    """
    X = np.asarray(points, dtype=float)
    if len(X) <= 1:
        return 0., {"point_pair_comparisons": 0, "bound_pruned_pairs": 0}
    tree = cKDTree(X, leafsize=leafsize, balanced_tree=True, compact_nodes=True)
    nodes = {}
    def bounds(node):
        key = id(node)
        if key not in nodes:
            if node.split_dim == -1:
                block = X[node.indices]
                lo, hi = block.min(axis=0), block.max(axis=0)
            else:
                left, right = bounds(node.lesser), bounds(node.greater)
                lo, hi = np.minimum(left[0], right[0]), np.maximum(left[1], right[1])
            nodes[key] = (lo, hi)
        return nodes[key]
    def upper(a, b):
        al, ah = bounds(a); bl, bh = bounds(b)
        delta = np.maximum(np.abs(al - bh), np.abs(ah - bl))
        return float(delta @ delta)
    # Deterministic feasible lower bound; it cannot affect exactness.
    extreme = np.unique(np.r_[np.argmin(X, axis=0), np.argmax(X, axis=0)])
    lower = float(cdist(X[extreme], X[extreme], metric="sqeuclidean").max())
    comparisons = len(extreme)**2
    heap = []
    serial = 0
    def push(a, b):
        nonlocal serial
        bound = upper(a, b)
        # A conservative roundoff allowance avoids pruning nearly equal extrema.
        if bound > lower * (1 - 32 * np.finfo(float).eps):
            serial += 1
            heapq.heappush(heap, (-bound, serial, a, b))
    push(tree.tree, tree.tree)
    pruned = 0
    maximum_heap = len(heap)
    while heap:
        neg_bound, _, a, b = heapq.heappop(heap)
        if -neg_bound < lower * (1 - 32 * np.finfo(float).eps):
            pruned += 1
            continue
        a_leaf, b_leaf = a.split_dim == -1, b.split_dim == -1
        if a_leaf and b_leaf:
            work = len(a.indices) * len(b.indices)
            if comparisons + work > max_pair_comparisons:
                raise GeometryResourceBlocked("exact pp-SPA diameter exceeded its frozen pair-comparison budget",
                                              {"point_pair_comparisons": comparisons, "max_diameter_pairs": max_pair_comparisons,
                                               "diameter_lower_bound": float(np.sqrt(lower)), "unresolved_upper_bound": float(np.sqrt(-neg_bound))})
            lower = max(lower, float(cdist(X[a.indices], X[b.indices], metric="sqeuclidean").max()))
            comparisons += work
        elif a is b:
            push(a.lesser, a.lesser); push(a.lesser, a.greater); push(a.greater, a.greater)
        elif not a_leaf and (b_leaf or a.children >= b.children):
            push(a.lesser, b); push(a.greater, b)
        else:
            push(a, b.lesser); push(a, b.greater)
        maximum_heap = max(maximum_heap, len(heap))
        if maximum_heap > 2_000_000:
            raise GeometryResourceBlocked("exact diameter node-pair queue exceeded two-million entry memory budget",
                                          {"maximum_heap": maximum_heap, "point_pair_comparisons": comparisons})
    return float(np.sqrt(lower)), {"point_pair_comparisons": comparisons, "bound_pruned_pairs": pruned,
                                  "maximum_heap_entries": maximum_heap, "exact": True}


def _pp_spa_bounded(points, K, parameters, max_diameter_pairs):
    from gplsi.vertex_hunting import (_affine_spa, _project_affine, PPSPA_COMMIT,
                                      VertexHuntResult, _vertex_diagnostics)
    radius_divisor = float(parameters.get("radius_divisor", 20.))
    m = int(parameters.get("m_neighbors", 4))
    minimum = int(parameters.get("min_neighbors", 3))
    if radius_divisor <= 0 or m < 1 or minimum < 1:
        raise ValueError("invalid pp-SPA neighborhood parameters")
    started = perf_counter()
    projected = _project_affine(points, K)
    diameter, diameter_metadata = exact_diameter(projected, max_pair_comparisons=max_diameter_pairs)
    radius = diameter / radius_divisor
    tree = cKDTree(projected)
    # return_length performs the exact radius search without constructing lists
    # of all neighbors, which can otherwise be quadratic in a nearly flat cloud.
    sizes = tree.query_ball_point(projected, radius, return_length=True)
    retained = np.flatnonzero(sizes >= minimum)
    if not len(retained):
        raise ValueError("pp-SPA discarded every projected observation")
    pseudo = np.empty((len(retained), points.shape[1]))
    chosen_neighbors = np.full((len(retained), m), -1, dtype=np.int64)
    for position, index in enumerate(retained):
        if sizes[index] <= m:
            neighbors = np.asarray(sorted(tree.query_ball_point(projected[index], radius)), dtype=int)
        else:
            _, neighbors = tree.query(projected[index], k=m)
            neighbors = np.atleast_1d(neighbors).astype(int)
        pseudo[position] = projected[neighbors].mean(axis=0)
        chosen_neighbors[position, :len(neighbors)] = neighbors
    vertices, selected = _affine_spa(pseudo, K)
    condition, smallest = _vertex_diagnostics(vertices)
    result = VertexHuntResult(method="pp_spa", vertices=vertices, embedding_used=points,
                             selected_observation_indices=retained[selected], selected_center_indices=selected,
                             projected_points=projected, pseudo_points=pseudo, neighborhood_sizes=sizes,
                             retained_point_indices=retained, discarded_point_indices=np.flatnonzero(sizes < minimum),
                             condition_number=condition, smallest_singular_value=smallest,
                             runtime_seconds=perf_counter() - started,
                             parameters={"radius_divisor": radius_divisor, "epsilon": radius,
                                         "m_neighbors": m, "official_parameter_N": m, "min_neighbors": minimum,
                                         "official_parameter_t": minimum, "affine_dimension": K - 1,
                                         "source_commit": PPSPA_COMMIT, "diameter": diameter_metadata,
                                         "implementation": "source_parity_bounded_memory_exact"})
    return result, {"pp_spa_selected_neighbor_indices": chosen_neighbors}


def _hunter_defaults(hunter, K, point_count):
    if hunter in ("svs", "svs_star"):
        return {"L_mode": "fixed", "L": min(K + 2, point_count), "max_simplexes": 200_000}
    if hunter == "spa_current":
        return {"precondition": False}
    if hunter == "pp_spa":
        return {"radius_divisor": 20., "m_neighbors": 4, "min_neighbors": 3}
    if hunter in ("palm", "palm_accelerated"):
        return {"lambda_": 1., "max_iterations": 300, "tolerance": 1e-7,
                "c_h": 1.1, "c_w": 1.1, "projection_tolerance": 1e-10,
                "projection_max_iterations": 10_000, "hull_reduction": "none",
                "initialize_weights_iterations": 200, "final_weight_refit_iterations": 200,
                "initialization_precondition": False}
    raise ValueError(f"unrecognized hunter {hunter}")


def fit_geometry(block: Embedding, features: FeaturePlan, adjacency, *, hunter,
                 estimator_seed, family="document", vertex_parameters=None,
                 max_svs_face_solves=1_000_000, max_diameter_pairs=200_000_000,
                 condition_threshold=1e12):
    """Return W once; A_current and Poisson descendants share this saved W."""
    from gplsi.vertex_hunting import vertex_hunt
    from gplsi.recovery import recover_W
    from gplsi.anchor_word import recover_W_from_word_vertices
    started = perf_counter()
    K = block.U.shape[1]
    if family in ("document", "document_U"):
        points = block.U
    elif family in ("anchor", "word_Z"):
        if hunter != "spa_current" or features.metadata["preprocessing"] != "P0_raw":
            raise ValueError("the frozen anchor-feature grid is P0/current-SPA only")
        # Same source formula, with no n-by-p M_hat diagnostic allocation.
        original_cross = block.V * block.singular_values[None, :] / features.weights[:, None]
        points = original_cross / features.eta[:, None]
        if np.linalg.norm(block.U.T @ block.U - np.eye(K)) > 1e-8:
            raise ValueError("anchor profiles require orthonormal U")
    else:
        raise ValueError("family must be document or anchor")
    params = _hunter_defaults(hunter, K, len(points))
    params.update(vertex_parameters or {})
    resources = vertex_resource_preflight(K, len(points), hunter, max_svs_face_solves=max_svs_face_solves,
                                         max_diameter_pairs=max_diameter_pairs, parameters=params)
    if resources["status"] != "feasible_preflight":
        raise GeometryResourceBlocked(resources["reason"], resources)
    auxiliary = {}
    if hunter == "pp_spa":
        vertices, auxiliary = _pp_spa_bounded(points, K, params, max_diameter_pairs)
    else:
        vertices = vertex_hunt(points, K, hunter, random_state=estimator_seed,
                               condition_threshold=condition_threshold, raise_on_failure=True, **params)
    if family in ("document", "document_U"):
        recovery_embedding = vertices.embedding_used if vertices.embedding_used is not None else block.U
        recovered = recover_W(recovery_embedding, vertices.vertices, condition_threshold=condition_threshold)
        residual = np.linalg.norm(recovered.raw @ vertices.vertices - recovery_embedding)
    else:
        signs = np.asarray(vertices.parameters["coordinate_signs"])
        recovered = recover_W_from_word_vertices(block.U * signs, vertices.vertices,
                                                  condition_threshold=condition_threshold)
        residual = recovered.prevalence_residual
        auxiliary["anchor_point_cloud"] = points
        if vertices.selected_observation_indices is not None:
            auxiliary["anchor_gene_indices"] = features.retained_indices[vertices.selected_observation_indices]
    W = recovered.simplex_projected
    if np.any(W < -1e-12) or not np.allclose(W.sum(axis=1), 1., atol=1e-10):
        raise ValueError("recovered memberships violate row-simplex constraints")
    hard = np.argmax(W, axis=1)
    topic_sizes = np.bincount(hard, minlength=K)
    variances = W.var(axis=0)
    vertex_optimizer_converged = vertices.parameters.get("optimizer_converged")
    vertex_converged = vertices.status == "ok" and (vertex_optimizer_converged is not False)
    membership_converged = bool(getattr(recovered, "converged", recovered.stable))
    metadata = {"family": family, "hunter": hunter, "estimator_seed": estimator_seed,
                "lambda": block.lambda_value, "spectral_converged": block.converged,
                "vertex_parameters": vertices.parameters, "resource_preflight": resources,
                "condition_number": recovered.condition_number,
                "smallest_singular_value": recovered.smallest_singular_value,
                "recovery_stable": recovered.stable,
                "unconstrained_reconstruction_residual": float(residual),
                "projection_displacement_frobenius": float(np.linalg.norm(W - recovered.raw)),
                "projection_displacement_per_row_mean": float(np.linalg.norm(W - recovered.raw, axis=1).mean()),
                "embedding_roughness": graph_roughness(block.U, adjacency),
                "unconstrained_membership_roughness": graph_roughness(recovered.raw, adjacency),
                "final_membership_roughness": graph_roughness(W, adjacency),
                "occupied_topics": int((topic_sizes > 0).sum()), "topic_sizes": topic_sizes.tolist(),
                "topic_variances": variances.tolist(), "near_constant_topics": (variances <= 1e-12).tolist(),
                "runtime_seconds": perf_counter() - started, "vertex_runtime_seconds": vertices.runtime_seconds,
                "warnings": list(vertices.warnings) + list(recovered.warnings),
                "vertex_status": vertices.status,
                "vertex_optimizer_status": vertices.parameters.get("optimizer_status"),
                "vertex_optimizer_converged": vertex_optimizer_converged,
                "vertex_converged": bool(vertex_converged),
                "membership_recovery_converged": membership_converged,
                "converged": bool(block.converged and vertex_converged and membership_converged and recovered.stable),
                "selected_observation_indices": None if vertices.selected_observation_indices is None
                    else vertices.selected_observation_indices.tolist(),
                "selected_center_indices": None if vertices.selected_center_indices is None
                    else vertices.selected_center_indices.tolist(),
                "initialization_observation_indices": None if vertices.initialization_observation_indices is None
                    else vertices.initialization_observation_indices.tolist(),
                "selected_anchor_full_gene_indices": auxiliary.get("anchor_gene_indices", np.array([], dtype=int)).tolist(),
                "kmeans_centers": None if vertices.centers is None else vertices.centers.tolist()}
    if hunter == "pp_spa":
        metadata["selected_vertex_neighbor_indices"] = auxiliary["pp_spa_selected_neighbor_indices"][
            vertices.selected_center_indices].tolist()
    return GeometryResult(W, vertices.vertices, recovered.raw, metadata, auxiliary)

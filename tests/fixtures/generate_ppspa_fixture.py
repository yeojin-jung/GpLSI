"""Run one official VertexHunting pp-SPA example as a parity fixture."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("ppspa_official_fixture.npz"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite fixture: {args.output}")

    source = args.repo_root / "external_references" / "VertexHunting"
    # The pinned repository imports two unused helpers from a missing utils.py.
    # Stubbing only those unused names lets the unmodified official files load.
    stub = types.ModuleType("utils")
    stub.find_max_distance_to_average = lambda *unused: None
    stub.find_min_distance_to_average = lambda *unused: None
    sys.modules["utils"] = stub
    # projection.py imports Plotly only for commented plotting examples.  Keep
    # the scientific functions unmodified while avoiding an irrelevant runtime
    # dependency in the fixture environment.
    plotly_stub = types.ModuleType("plotly")
    plotly_go_stub = types.ModuleType("plotly.graph_objects")
    plotly_express_stub = types.ModuleType("plotly.express")
    plotly_stub.graph_objects = plotly_go_stub
    plotly_stub.express = plotly_express_stub
    sys.modules["plotly"] = plotly_stub
    sys.modules["plotly.graph_objects"] = plotly_go_stub
    sys.modules["plotly.express"] = plotly_express_stub
    sys.path.insert(0, str(source))
    official_vertex = importlib.import_module("vertexH")
    official_projection = importlib.import_module("projection")
    official_spa = importlib.import_module("spa")

    seed = 4242
    np.random.seed(seed)
    ideal_vertices = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    samples = official_vertex.get_sample(
        N_sample=60, vertex=ideal_vertices, pure_nodes=3, d=2
    )
    projected = official_projection.get_projected_points(samples.T, K=3).T
    epsilon = official_vertex.knn(projected, M=20)
    pseudo_points = official_vertex.compute_points_within_epsilon(
        projected, epsilon=epsilon, N=4, t=3, d=2
    )
    vertices = official_spa.SuccessiveProj(pseudo_points, 3)

    metadata = {
        "seed": seed,
        "N_sample": 60,
        "pure_nodes": 3,
        "K": 3,
        "d": 2,
        "radius_divisor_M": 20,
        "official_neighbor_cap_N": 4,
        "official_min_neighbors_t": 3,
        "epsilon": float(epsilon),
        "source_commit": "e3826fca5ee08916e36ae936d6fbe791a2126c18",
        "source_loading_note": "stubbed two unused imports from absent utils.py",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        samples=samples,
        projected_points=projected,
        pseudo_points=pseudo_points,
        vertices=vertices,
        ideal_vertices=ideal_vertices.T,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(args.output)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

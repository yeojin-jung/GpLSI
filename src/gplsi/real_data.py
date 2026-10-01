"""Canonical, validated loaders for the CRC, spleen, and What's Cooking data.

Each loader checks its output against the frozen hash contract in
``contracts/canonical_data_contract_summary.json``.  Experimental feature
selection and weighting are deliberately excluded: every method receives the
same canonical count matrix, frequencies, document lengths, and graph.

Data live under ``data/{crc,spleen,cook}`` (``GPLSI_DATA_ROOT`` overrides the
root).  Visium DLPFC sections are loaded by :mod:`gplsi.pipeline.datasets`.
"""

from __future__ import annotations

import ast
from collections import Counter, deque
from dataclasses import dataclass, field, replace
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("GPLSI_DATA_ROOT", REPO_ROOT / "data")).expanduser()
AUDIT_CONTRACT = Path(
    os.environ.get(
        "GPLSI_AUDIT_CONTRACT",
        Path(__file__).resolve().parent
        / "contracts"
        / "canonical_data_contract_summary.json",
    )
).expanduser()


class RealDataContractError(RuntimeError):
    """Raised when canonical data no longer match their frozen audit contract."""


def _sha256_numeric(value: Any, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_strings(value: Any) -> str:
    encoded = json.dumps(
        [str(item) for item in value], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_coordinates(coordinates: np.ndarray) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=float).copy()
    ranges = np.ptp(coords, axis=0)
    diagonal = float(np.linalg.norm(ranges))
    if diagonal <= np.finfo(float).eps:
        raise RealDataContractError("coordinate range has zero diagonal length")
    return (coords - coords.min(axis=0)) / diagonal


def _edge_weights_from_coordinates(
    edges: np.ndarray, coordinates: np.ndarray, phi: float
) -> np.ndarray:
    differences = coordinates[edges[:, 0]] - coordinates[edges[:, 1]]
    return np.exp(-float(phi) * np.sum(differences**2, axis=1))


@dataclass
class RealDataBundle:
    dataset: str
    counts: np.ndarray
    frequencies: np.ndarray
    document_lengths: np.ndarray
    feature_ids: np.ndarray
    observation_ids: np.ndarray
    group_ids: np.ndarray
    edge_df: pd.DataFrame
    weights: csr_matrix
    coordinates: np.ndarray | None = None
    outcomes: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.counts.shape[0])

    @property
    def p(self) -> int:
        return int(self.counts.shape[1])

    @property
    def N_mean(self) -> float:
        return float(np.mean(self.document_lengths))

    def validate(self) -> "RealDataBundle":
        D = np.asarray(self.counts)
        X = np.asarray(self.frequencies, dtype=float)
        N_i = np.asarray(self.document_lengths, dtype=float).reshape(-1)
        if D.ndim != 2 or X.shape != D.shape:
            raise RealDataContractError("counts and frequencies must have equal 2-D shape")
        if not np.issubdtype(D.dtype, np.integer) or np.any(D < 0):
            raise RealDataContractError("canonical counts must be nonnegative integers")
        if N_i.size != D.shape[0] or np.any(N_i <= 0):
            raise RealDataContractError("document lengths must be positive and row-aligned")
        if not np.array_equal(D.sum(axis=1).astype(float), N_i):
            raise RealDataContractError("document lengths do not equal count row totals")
        if not np.allclose(X, D / N_i[:, None], rtol=0.0, atol=2e-15):
            raise RealDataContractError("frequencies are not canonical row-normalized counts")
        if len(self.feature_ids) != D.shape[1]:
            raise RealDataContractError("feature ids are not column-aligned")
        if len(self.observation_ids) != D.shape[0] or len(self.group_ids) != D.shape[0]:
            raise RealDataContractError("observation/group ids are not row-aligned")
        required = {"src", "tgt", "weight"}
        if not required.issubset(self.edge_df.columns):
            raise RealDataContractError(f"edge table must contain {sorted(required)}")
        endpoints = self.edge_df[["src", "tgt"]].to_numpy(dtype=np.int64)
        if endpoints.size and (endpoints.min() < 0 or endpoints.max() >= D.shape[0]):
            raise RealDataContractError("graph contains a row index outside canonical D")
        if self.weights.shape != (D.shape[0], D.shape[0]):
            raise RealDataContractError("sparse graph shape is not n by n")
        if self.coordinates is not None and np.asarray(self.coordinates).shape != (D.shape[0], 2):
            raise RealDataContractError("coordinates must have shape n by 2")
        return self

    def hashes(self) -> dict[str, str]:
        endpoints = self.edge_df[["src", "tgt"]].to_numpy(dtype=np.int64)
        output = {
            "D_sha256": _sha256_numeric(self.counts, "<i8"),
            "X_sha256": _sha256_numeric(self.frequencies, "<f8"),
            "N_i_sha256": _sha256_numeric(self.document_lengths, "<f8"),
            "feature_ids_sha256": _sha256_strings(self.feature_ids),
            "observation_ids_sha256": _sha256_strings(self.observation_ids),
            "group_ids_sha256": _sha256_strings(self.group_ids),
            "edges_sha256": _sha256_numeric(endpoints, "<i8"),
            "edge_weights_sha256": _sha256_numeric(
                self.edge_df["weight"].to_numpy(), "<f8"
            ),
        }
        if self.coordinates is not None:
            output["coordinates_sha256"] = _sha256_numeric(self.coordinates, "<f8")
        return output

    def induced_row_subset(
        self,
        indices: np.ndarray,
        *,
        metadata_updates: dict[str, Any] | None = None,
    ) -> "RealDataBundle":
        """Return the graph-induced subset on sorted row ``indices``.

        The explicit sorted-index contract keeps the canonical edge ordering
        stable and makes row-removal hashes reproducible.  Feature columns are
        never changed here.
        """

        raw_indices = np.asarray(indices)
        if raw_indices.ndim != 1 or not np.issubdtype(
            raw_indices.dtype, np.integer
        ):
            raise RealDataContractError(
                "induced row-subset indices must be a one-dimensional integer array"
            )
        selected = np.asarray(raw_indices, dtype=np.int64)
        if selected.size == 0:
            raise RealDataContractError("an induced row subset cannot be empty")
        if (
            np.any(selected < 0)
            or np.any(selected >= self.n)
            or np.any(np.diff(selected) <= 0)
        ):
            raise RealDataContractError(
                "induced row-subset indices must be unique, sorted, and in range"
            )
        if selected.size == self.n and np.array_equal(
            selected, np.arange(self.n, dtype=np.int64)
        ):
            output = replace(self, metadata=dict(self.metadata))
            if metadata_updates:
                output.metadata.update(metadata_updates)
            return output.validate()

        remap = np.full(self.n, -1, dtype=np.int64)
        remap[selected] = np.arange(selected.size, dtype=np.int64)
        endpoints = self.edge_df[["src", "tgt"]].to_numpy(dtype=np.int64)
        keep = (remap[endpoints[:, 0]] >= 0) & (remap[endpoints[:, 1]] >= 0)
        edge = self.edge_df.loc[keep].copy().reset_index(drop=True)
        edge["src"] = remap[edge["src"].to_numpy(dtype=np.int64)]
        edge["tgt"] = remap[edge["tgt"].to_numpy(dtype=np.int64)]
        sparse = csr_matrix(
            (
                edge["weight"].to_numpy(dtype=float),
                (
                    edge["src"].to_numpy(dtype=np.int64),
                    edge["tgt"].to_numpy(dtype=np.int64),
                ),
            ),
            shape=(selected.size, selected.size),
        )
        metadata = dict(self.metadata)
        metadata.update(
            {
                "subset_of_canonical": True,
                "parent_n": self.n,
                "subset_strategy": "induced_row_subset",
                "subset_indices_sha256": _sha256_numeric(selected, "<i8"),
            }
        )
        if metadata_updates:
            metadata.update(metadata_updates)
        return replace(
            self,
            counts=np.ascontiguousarray(self.counts[selected]),
            frequencies=np.ascontiguousarray(self.frequencies[selected]),
            document_lengths=np.ascontiguousarray(self.document_lengths[selected]),
            observation_ids=self.observation_ids[selected].copy(),
            group_ids=self.group_ids[selected].copy(),
            edge_df=edge,
            weights=sparse,
            coordinates=(
                None
                if self.coordinates is None
                else np.ascontiguousarray(self.coordinates[selected])
            ),
            outcomes=(
                None
                if self.outcomes is None
                else self.outcomes.iloc[selected].reset_index(drop=True)
            ),
            metadata=metadata,
        ).validate()

    def connected_subset(self, max_observations: int, seed: int = 0) -> "RealDataBundle":
        """Return a deterministic connected induced subgraph for smoke/pilot checks."""

        if max_observations <= 0:
            raise ValueError("max_observations must be positive")
        if max_observations >= self.n:
            return self
        adjacency: list[list[int]] = [[] for _ in range(self.n)]
        for src, tgt in self.edge_df[["src", "tgt"]].to_numpy(dtype=np.int64):
            if src == tgt:
                continue
            adjacency[int(src)].append(int(tgt))
            adjacency[int(tgt)].append(int(src))
        seen = np.zeros(self.n, dtype=bool)
        components: list[list[int]] = []
        for root in range(self.n):
            if seen[root] or not adjacency[root]:
                continue
            queue = deque([root])
            seen[root] = True
            component: list[int] = []
            while queue:
                node = queue.popleft()
                component.append(node)
                for neighbor in sorted(adjacency[node]):
                    if not seen[neighbor]:
                        seen[neighbor] = True
                        queue.append(neighbor)
            components.append(component)
        if not components:
            raise RealDataContractError("cannot form a connected subset from an empty graph")
        largest = max(components, key=lambda values: (len(values), -min(values)))
        if len(largest) < max_observations:
            raise RealDataContractError(
                f"largest graph component has {len(largest)} rows, below requested "
                f"subset size {max_observations}"
            )
        rng = np.random.default_rng(seed)
        start = int(rng.choice(np.asarray(largest, dtype=int)))
        selected: list[int] = []
        selected_set: set[int] = {start}
        queue = deque([start])
        while queue and len(selected) < max_observations:
            node = queue.popleft()
            selected.append(node)
            neighbors = [value for value in sorted(adjacency[node]) if value not in selected_set]
            for neighbor in neighbors:
                selected_set.add(neighbor)
                queue.append(neighbor)
        indices = np.asarray(selected, dtype=int)
        if indices.size != max_observations:
            raise RealDataContractError("connected-subset traversal ended unexpectedly")
        remap = np.full(self.n, -1, dtype=int)
        remap[indices] = np.arange(indices.size)
        endpoints = self.edge_df[["src", "tgt"]].to_numpy(dtype=int)
        keep = (remap[endpoints[:, 0]] >= 0) & (remap[endpoints[:, 1]] >= 0)
        edge = self.edge_df.loc[keep].copy().reset_index(drop=True)
        edge["src"] = remap[edge["src"].to_numpy(dtype=int)]
        edge["tgt"] = remap[edge["tgt"].to_numpy(dtype=int)]
        sparse = csr_matrix(
            (edge["weight"].to_numpy(dtype=float), (edge["src"], edge["tgt"])),
            shape=(indices.size, indices.size),
        )
        metadata = dict(self.metadata)
        metadata.update(
            {
                "subset_of_canonical": True,
                "parent_n": self.n,
                "subset_seed": int(seed),
                "subset_indices_sha256": _sha256_numeric(indices, "<i8"),
            }
        )
        return replace(
            self,
            counts=np.ascontiguousarray(self.counts[indices]),
            frequencies=np.ascontiguousarray(self.frequencies[indices]),
            document_lengths=np.ascontiguousarray(self.document_lengths[indices]),
            observation_ids=self.observation_ids[indices].copy(),
            group_ids=self.group_ids[indices].copy(),
            edge_df=edge,
            weights=sparse,
            coordinates=None if self.coordinates is None else self.coordinates[indices].copy(),
            outcomes=None if self.outcomes is None else self.outcomes.iloc[indices].reset_index(drop=True),
            metadata=metadata,
        ).validate()

    def graph_stratified_subset(
        self,
        max_observations: int,
        *,
        seed: int = 0,
        max_groups: int | None = None,
    ) -> "RealDataBundle":
        """Select graph neighborhoods seeded across several declared groups.

        This is intended for structurally valid smoke/pilot subsets: CRC keeps
        several regions, and Cooking seeds every cuisine before graph expansion.
        """

        if max_observations <= 1 or max_observations >= self.n:
            if max_observations >= self.n:
                return self
            raise ValueError("graph-stratified subsets require at least two rows")
        adjacency: list[list[int]] = [[] for _ in range(self.n)]
        for src, tgt in self.edge_df[["src", "tgt"]].to_numpy(dtype=np.int64):
            if src == tgt:
                continue
            adjacency[int(src)].append(int(tgt))
            adjacency[int(tgt)].append(int(src))
        eligible: dict[str, list[int]] = {}
        label_values: dict[str, Any] = {}
        for index, label in enumerate(self.group_ids):
            if adjacency[index]:
                key = str(label)
                eligible.setdefault(key, []).append(index)
                label_values[key] = label
        labels = sorted(eligible)
        group_limit = max_observations // 2
        if max_groups is not None:
            group_limit = min(group_limit, int(max_groups))
        if group_limit < 1 or not labels:
            raise RealDataContractError("no graph-supported groups are available")
        rng = np.random.default_rng(seed)
        if len(labels) > group_limit:
            chosen_positions = np.sort(rng.choice(len(labels), group_limit, replace=False))
            labels = [labels[int(position)] for position in chosen_positions]

        starts = [int(rng.choice(np.asarray(eligible[label], dtype=int))) for label in labels]
        selected: list[int] = []
        selected_set: set[int] = set()
        queue: deque[int] = deque()
        for start in starts:
            if start not in selected_set:
                selected.append(start)
                selected_set.add(start)
                queue.append(start)
        # Give each group seed a graph neighbor before general expansion.
        for start in starts:
            candidates = [value for value in sorted(adjacency[start]) if value not in selected_set]
            if candidates and len(selected) < max_observations:
                neighbor = int(rng.choice(np.asarray(candidates, dtype=int)))
                selected.append(neighbor)
                selected_set.add(neighbor)
                queue.append(neighbor)
        while queue and len(selected) < max_observations:
            node = queue.popleft()
            for neighbor in sorted(adjacency[node]):
                if neighbor not in selected_set:
                    selected.append(neighbor)
                    selected_set.add(neighbor)
                    queue.append(neighbor)
                    if len(selected) == max_observations:
                        break
        if len(selected) < max_observations:
            remaining = np.asarray(
                [index for index in range(self.n) if adjacency[index] and index not in selected_set],
                dtype=int,
            )
            if remaining.size < max_observations - len(selected):
                raise RealDataContractError("not enough graph-supported observations for subset")
            selected.extend(
                map(
                    int,
                    rng.choice(
                        remaining, max_observations - len(selected), replace=False
                    ),
                )
            )
        indices = np.asarray(selected, dtype=int)
        remap = np.full(self.n, -1, dtype=int)
        remap[indices] = np.arange(indices.size)
        endpoints = self.edge_df[["src", "tgt"]].to_numpy(dtype=int)
        keep = (remap[endpoints[:, 0]] >= 0) & (remap[endpoints[:, 1]] >= 0)
        edge = self.edge_df.loc[keep].copy().reset_index(drop=True)
        edge["src"] = remap[edge["src"].to_numpy(dtype=int)]
        edge["tgt"] = remap[edge["tgt"].to_numpy(dtype=int)]
        sparse = csr_matrix(
            (edge["weight"].to_numpy(dtype=float), (edge["src"], edge["tgt"])),
            shape=(indices.size, indices.size),
        )
        group_counts = pd.Series(self.group_ids[indices]).astype(str).value_counts().to_dict()
        metadata = dict(self.metadata)
        metadata.update(
            {
                "subset_of_canonical": True,
                "subset_strategy": "graph_stratified",
                "parent_n": self.n,
                "subset_seed": int(seed),
                "subset_indices_sha256": _sha256_numeric(indices, "<i8"),
                "subset_group_counts": {str(key): int(value) for key, value in group_counts.items()},
                "subset_seed_groups": [str(label_values[label]) for label in labels],
            }
        )
        return replace(
            self,
            counts=np.ascontiguousarray(self.counts[indices]),
            frequencies=np.ascontiguousarray(self.frequencies[indices]),
            document_lengths=np.ascontiguousarray(self.document_lengths[indices]),
            observation_ids=self.observation_ids[indices].copy(),
            group_ids=self.group_ids[indices].copy(),
            edge_df=edge,
            weights=sparse,
            coordinates=None if self.coordinates is None else self.coordinates[indices].copy(),
            outcomes=None if self.outcomes is None else self.outcomes.iloc[indices].reset_index(drop=True),
            metadata=metadata,
        ).validate()

    def binomial_thinning(
        self, test_fraction: float, *, seed: int
    ) -> tuple["RealDataBundle", np.ndarray]:
        """Split each canonical count into train/test counts without label use."""

        if not (0.0 < test_fraction < 1.0):
            raise ValueError("test_fraction must lie strictly between zero and one")
        rng = np.random.default_rng(seed)
        train = rng.binomial(self.counts, 1.0 - test_fraction).astype(np.int64)
        test = self.counts - train
        for index in range(self.n):
            if train[index].sum() == 0:
                feature = int(np.flatnonzero(test[index] > 0)[0])
                train[index, feature] += 1
                test[index, feature] -= 1
            if test[index].sum() == 0 and self.document_lengths[index] > 1:
                feature = int(np.flatnonzero(train[index] > 0)[0])
                test[index, feature] += 1
                train[index, feature] -= 1
        train_lengths = train.sum(axis=1).astype(float)
        if np.any(train_lengths <= 0):
            raise RealDataContractError("count thinning could not preserve positive training rows")
        metadata = dict(self.metadata)
        metadata.update(
            {
                "count_thinning": True,
                "thinning_seed": int(seed),
                "thinning_test_fraction": float(test_fraction),
                "pre_thinning_D_sha256": _sha256_numeric(self.counts, "<i8"),
                "heldout_D_sha256": _sha256_numeric(test, "<i8"),
                "heldout_positive_row_count": int(np.count_nonzero(test.sum(axis=1) > 0)),
                "heldout_zero_row_count": int(np.count_nonzero(test.sum(axis=1) == 0)),
            }
        )
        training = replace(
            self,
            counts=np.ascontiguousarray(train),
            frequencies=np.ascontiguousarray(train / train_lengths[:, None]),
            document_lengths=np.ascontiguousarray(train_lengths),
            metadata=metadata,
        ).validate()
        return training, np.ascontiguousarray(test)


def _make_bundle(
    *,
    dataset: str,
    counts: np.ndarray,
    feature_ids: Any,
    observation_ids: Any,
    group_ids: Any,
    edge_df: pd.DataFrame,
    coordinates: np.ndarray | None,
    outcomes: pd.DataFrame | None,
    metadata: dict[str, Any],
) -> RealDataBundle:
    D = np.ascontiguousarray(np.asarray(counts, dtype=np.int64))
    N_i = np.ascontiguousarray(D.sum(axis=1).astype(float))
    X = np.ascontiguousarray(D / N_i[:, None], dtype=float)
    edges = edge_df[["src", "tgt", "weight"]].copy().reset_index(drop=True)
    edges[["src", "tgt"]] = edges[["src", "tgt"]].astype(np.int64)
    edges["weight"] = edges["weight"].astype(float)
    sparse = csr_matrix(
        (edges["weight"].to_numpy(), (edges["src"], edges["tgt"])),
        shape=(D.shape[0], D.shape[0]),
    )
    return RealDataBundle(
        dataset=dataset,
        counts=D,
        frequencies=X,
        document_lengths=N_i,
        feature_ids=np.asarray(feature_ids, dtype=object),
        observation_ids=np.asarray(observation_ids, dtype=object),
        group_ids=np.asarray(group_ids, dtype=object),
        edge_df=edges,
        weights=sparse,
        coordinates=None if coordinates is None else np.asarray(coordinates, dtype=float),
        outcomes=outcomes,
        metadata=metadata,
    ).validate()


def load_crc(*, phi: float = 0.1) -> RealDataBundle:
    root = DATA_ROOT / "crc"
    source = root / "output" / "output_3hop"
    metadata = pd.read_csv(root / "charville_labels.csv")
    selected_metadata = metadata.loc[metadata["primary_outcome"].notna()].copy()
    counts: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    edge_blocks: list[np.ndarray] = []
    observation_ids: list[str] = []
    groups: list[str] = []
    outcomes: list[pd.DataFrame] = []
    offset = 0
    feature_ids: list[str] | None = None
    for region in selected_metadata["region_id"].astype(str):
        D_frame = pd.read_csv(f"{source / region}.D.csv", index_col=0)
        if feature_ids is None:
            feature_ids = list(map(str, D_frame.columns))
        elif feature_ids != list(map(str, D_frame.columns)):
            raise RealDataContractError(f"CRC feature order changed in region {region}")
        raw_cell_ids = np.asarray(
            [int(ast.literal_eval(value)[1]) for value in D_frame.index], dtype=int
        )
        keep = D_frame.sum(axis=1).to_numpy() >= 10
        retained_cells = raw_cell_ids[keep]
        retained_D = D_frame.loc[keep].to_numpy(dtype=np.int64)
        coord = pd.read_csv(f"{source / region}.coord.csv", index_col=0).set_index("CELL_ID")
        coordinates.append(coord.loc[retained_cells, ["X", "Y"]].to_numpy(dtype=float))
        edge = pd.read_csv(f"{source / region}.edge.csv", index_col=0)
        retained_set = set(map(int, retained_cells))
        edge = edge.loc[edge["src"].isin(retained_set) & edge["tgt"].isin(retained_set)]
        mapping = {int(cell): offset + index for index, cell in enumerate(retained_cells)}
        edge_blocks.append(
            np.column_stack(
                (
                    edge["src"].map(mapping).to_numpy(dtype=np.int64),
                    edge["tgt"].map(mapping).to_numpy(dtype=np.int64),
                )
            )
        )
        observation_ids.extend(f"{region}::{cell}" for cell in retained_cells)
        groups.extend([region] * retained_D.shape[0])
        row_metadata = selected_metadata.loc[selected_metadata["region_id"].astype(str) == region]
        outcomes.append(pd.concat([row_metadata] * retained_D.shape[0], ignore_index=True))
        counts.append(retained_D)
        offset += retained_D.shape[0]
    D = np.vstack(counts)
    raw_coords = np.vstack(coordinates)
    coords = _normalize_coordinates(raw_coords)
    endpoints = np.vstack(edge_blocks)
    weights = _edge_weights_from_coordinates(endpoints, coords, phi)
    edge_df = pd.DataFrame(
        {"src": endpoints[:, 0], "tgt": endpoints[:, 1], "weight": weights}
    )
    return _make_bundle(
        dataset="stanford_crc_codex",
        counts=D,
        feature_ids=feature_ids,
        observation_ids=observation_ids,
        group_ids=groups,
        edge_df=edge_df,
        coordinates=coords,
        outcomes=pd.concat(outcomes, ignore_index=True),
        metadata={
            "canonical_scope": "196 regions with nonmissing primary_outcome; rows with count total >=10",
            "graph": "corrected contiguous global ids; within-region spatial edges",
            "phi": float(phi),
            "independent_unit": "region; no patient mapping supplied",
            "raw_coordinates_sha256": _sha256_numeric(raw_coords, "<f8"),
        },
    )


def load_spleen(group: str = "BALBc-1", *, phi: float = 0.1) -> RealDataBundle:
    root = DATA_ROOT / "spleen" / "dataset"
    D_all = pd.read_pickle(root / "merged_D.pkl")
    coords_all = pd.read_pickle(root / "merged_coord.pkl")
    edges_all = pd.read_pickle(root / "merged_data.pkl")
    available = list(map(str, D_all.index.get_level_values(0).unique()))
    if group not in available:
        raise ValueError(f"unknown spleen group {group!r}; expected one of {available}")
    D_frame = D_all.loc[group]
    D = D_frame.to_numpy(dtype=np.int64)
    raw_coords = coords_all.loc[group].to_numpy(dtype=float)
    coords = _normalize_coordinates(raw_coords)
    raw_edge = edges_all.loc[group]
    endpoints = raw_edge[["src", "dst"]].to_numpy(dtype=np.int64)
    edge_weight = _edge_weights_from_coordinates(endpoints, coords, phi)
    edge_df = pd.DataFrame(
        {"src": endpoints[:, 0], "tgt": endpoints[:, 1], "weight": edge_weight}
    )
    return _make_bundle(
        dataset="mouse_spleen_codex",
        counts=D,
        feature_ids=list(map(str, D_frame.columns)),
        observation_ids=list(map(str, D_frame.index)),
        group_ids=[group] * D.shape[0],
        edge_df=edge_df,
        coordinates=coords,
        outcomes=None,
        metadata={
            "biological_group": group,
            "canonical_scope": "one supplied BALB/c spleen",
            "phi": float(phi),
            "independent_unit": "biological spleen",
            "raw_coordinates_sha256": _sha256_numeric(raw_coords, "<f8"),
        },
    )


JOINT_SPLEEN_GROUP = "joint"
JOINT_SPLEEN_COMPONENTS = ("BALBc-1", "BALBc-2", "BALBc-3")
JOINT_SPLEEN_DERIVED_CONTRACT = {
    "n": 100840,
    "p": 24,
    "D_sha256": "4856c96de463fde08f30d0a3373209562e296a993378f4f9daef01786b880fba",
    "X_sha256": "37ce8e4e163879d5f3e7672726bf7d3258cc4662c0fc02d937440fc850540e7c",
    "N_i_sha256": "6a322dce523ff1a64c7fbc32615632d4eda0ecb730046da35c6b96f2fc2444d0",
    "feature_ids_sha256": "09da04a6d56b0a5919ae0df92f0af339656d90d5c88319525f21c20ffa7ace3d",
    "observation_ids_sha256": "c55f9b6e8d53bc4f162c02f8660cc95aa4de2f69e09b503b1a8ecef09a8a25d6",
    "group_ids_sha256": "d2e95f71678473736669bf9c88a36d5ba4d8a9b7bf0c0fa26c13f754478bcc65",
    "edges_sha256": "34f81bbea3562b265df6a4172176cda7706ebe6f95f51e1587172ab8ad273d6e",
    # NumPy/libm versions can differ by one ULP when evaluating exp.  Freeze
    # the scientifically meaningful weights after 10-decimal quantization so
    # the same audited graph loads on both the workstation and Midway.
    "edge_weights_rounded_10_sha256": "161087db751263fa33a73c467b20d37a1373738087af09c6e7be2d4136fc91f2",
    "coordinates_sha256": "29e7399af57e19ca0590893244a4d3e7d649c2a067e1c703a5787cbdeef30cb3",
}


def load_joint_spleen(*, phi: float = 0.1) -> RealDataBundle:
    """Load one joint model matrix over all three audited spleens.

    Counts share their canonical 24-feature ordering.  Supplied graph edges
    remain block diagonal by biological spleen, while ``group_ids`` preserve
    the component identity for stratified summaries and visualization.
    """

    components = [
        validate_frozen_contract(load_spleen(group, phi=phi))
        for group in JOINT_SPLEEN_COMPONENTS
    ]
    feature_ids = components[0].feature_ids.copy()
    for component in components[1:]:
        if not np.array_equal(component.feature_ids, feature_ids):
            raise RealDataContractError(
                "joint spleen components do not share the canonical feature order"
            )

    edge_frames: list[pd.DataFrame] = []
    coordinates: list[np.ndarray] = []
    observation_ids: list[str] = []
    group_slices: dict[str, list[int]] = {}
    offset = 0
    for panel, (group, component) in enumerate(
        zip(JOINT_SPLEEN_COMPONENTS, components, strict=True)
    ):
        edge = component.edge_df.copy()
        edge[["src", "tgt"]] += offset
        edge_frames.append(edge)
        # Translation leaves within-spleen distances unchanged and prevents
        # the three panels from being drawn on top of one another.
        translated = component.coordinates.copy()
        translated[:, 0] += 2.0 * panel
        coordinates.append(translated)
        observation_ids.extend(
            f"{group}::{value}" for value in component.observation_ids
        )
        group_slices[group] = [offset, offset + component.n]
        offset += component.n

    metadata = {
        "biological_group": JOINT_SPLEEN_GROUP,
        "canonical_scope": "one joint fit across BALBc-1, BALBc-2, and BALBc-3",
        "graph": "block-diagonal union of the three supplied within-spleen spatial graphs",
        "joint_coordinate_convention": (
            "each frozen spleen is normalized independently; display-only x offsets "
            "of 0, 2, and 4 are then applied"
        ),
        "joint_edge_weight_convention": (
            "preserve the independently audited within-spleen edge weights exactly"
        ),
        "phi": float(phi),
        "independent_unit": "biological spleen within a joint topic model",
        "joint_model": True,
        "joint_components": list(JOINT_SPLEEN_COMPONENTS),
        "joint_group_slices": group_slices,
        "component_contract_hashes": {
            group: component.metadata["canonical_contract_hashes"]
            for group, component in zip(
                JOINT_SPLEEN_COMPONENTS, components, strict=True
            )
        },
        "component_contracts_validated": True,
    }
    bundle = _make_bundle(
        dataset="mouse_spleen_codex",
        counts=np.vstack([component.counts for component in components]),
        feature_ids=feature_ids,
        observation_ids=observation_ids,
        group_ids=np.concatenate([component.group_ids for component in components]),
        edge_df=pd.concat(edge_frames, ignore_index=True),
        coordinates=np.vstack(coordinates),
        outcomes=None,
        metadata=metadata,
    )
    return validate_frozen_contract(bundle)


def _sample_cuisine(group: pd.DataFrame) -> pd.DataFrame:
    return group.sample(2000, random_state=1) if len(group) > 2000 else group


def load_cook() -> RealDataBundle:
    root = DATA_ROOT / "cook" / "dataset"
    raw = pd.read_json(root / "train.json")
    with (root / "ingredient_mapping.pkl").open("rb") as handle:
        ingredient_mapping = pickle.load(handle)
    reverse_mapping = {
        value: key for key, values in ingredient_mapping.items() for value in values
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sampled = (
            raw.groupby("cuisine", group_keys=False)
            .apply(_sample_cuisine)
            .reset_index(drop=True)
        )
    sampled["ingredients"] = sampled["ingredients"].apply(
        lambda values: [reverse_mapping.get(value, value) for value in values]
    )
    strings = sampled["ingredients"].apply(lambda values: ",".join(values))
    counts = Counter(value for values in sampled["ingredients"] for value in values)
    vocabulary = [value for value, count in counts.items() if count >= 10]
    vectorizer = CountVectorizer(
        tokenizer=lambda value: value.split(","), vocabulary=vocabulary
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = pd.DataFrame(
            vectorizer.fit_transform(strings).toarray(),
            columns=vectorizer.get_feature_names_out(),
        )
    row_keep = frame.sum(axis=1).to_numpy() >= 10
    frame = frame.loc[row_keep]
    sampled = sampled.loc[row_keep]
    column_keep = frame.sum(axis=0).to_numpy() >= 10
    frame = frame.loc[:, column_keep].reset_index(drop=True)
    sampled = sampled.reset_index(drop=True)
    edge_df = pd.read_pickle(root / "processed_edge_df.pkl")
    return _make_bundle(
        dataset="whats_cooking",
        counts=frame.to_numpy(dtype=np.int64),
        feature_ids=list(map(str, frame.columns)),
        observation_ids=sampled["id"].to_numpy(),
        group_ids=sampled["cuisine"].to_numpy(),
        edge_df=edge_df,
        coordinates=None,
        outcomes=pd.DataFrame({"cuisine": sampled["cuisine"].to_numpy()}),
        metadata={
            "canonical_scope": "historical 13,597 x 1,019 preprocessing",
            "graph": "frozen historical unweighted top-five Jaccard graph",
            "independent_unit": "recipe; split stratified by cuisine",
            "historical_sampling_seed": 1,
        },
    )


def _expected_contract(dataset: str, group: str | None) -> dict[str, Any]:
    if not AUDIT_CONTRACT.exists():
        raise RealDataContractError(f"frozen audit contract is missing: {AUDIT_CONTRACT}")
    contract = json.loads(AUDIT_CONTRACT.read_text())["datasets"]
    if dataset == "stanford_crc_codex":
        return contract[dataset]
    if dataset == "mouse_spleen_codex":
        if group is None:
            raise ValueError("spleen contract validation requires group")
        output = dict(contract[dataset]["groups"][group])
        output["feature_ids"] = contract[dataset]["feature_ids"]
        return output
    return contract[dataset]


def validate_frozen_contract(bundle: RealDataBundle) -> RealDataBundle:
    group = bundle.metadata.get("biological_group")
    if bundle.dataset == "mouse_spleen_codex" and group == JOINT_SPLEEN_GROUP:
        observed = bundle.hashes()
        expected = JOINT_SPLEEN_DERIVED_CONTRACT
        if (bundle.n, bundle.p) != (expected["n"], expected["p"]):
            raise RealDataContractError("joint spleen dimensions changed")
        rounded_weight_hash = _sha256_numeric(
            np.round(bundle.edge_df["weight"].to_numpy(), 10), "<f8"
        )
        for key, expected_value in expected.items():
            if key in {"n", "p"}:
                continue
            observed_value = (
                rounded_weight_hash
                if key == "edge_weights_rounded_10_sha256"
                else observed.get(key)
            )
            if observed_value != expected_value:
                raise RealDataContractError(
                    f"joint spleen {key} changed: {observed_value} != {expected_value}"
                )
        bundle.metadata["canonical_contract_validated"] = True
        bundle.metadata["canonical_contract_validation"] = (
            "frozen derived contract constructed from three independently validated components"
        )
        bundle.metadata["canonical_contract_hashes"] = {
            **observed,
            "edge_weights_rounded_10_sha256": rounded_weight_hash,
        }
        return bundle
    expected = _expected_contract(bundle.dataset, group)
    observed = bundle.hashes()
    for key in ("D_sha256", "X_sha256", "N_i_sha256", "observation_ids_sha256"):
        if observed[key] != expected[key]:
            raise RealDataContractError(
                f"{bundle.dataset} {key} changed: {observed[key]} != {expected[key]}"
            )
    if bundle.dataset != "mouse_spleen_codex":
        if observed["group_ids_sha256"] != expected["group_ids_sha256"]:
            raise RealDataContractError(f"{bundle.dataset} group ids changed")
    if (bundle.n, bundle.p) != (int(expected["n"]), int(expected["p"])):
        raise RealDataContractError("canonical dimensions no longer match the audit")
    if list(map(str, bundle.feature_ids)) != list(map(str, expected.get("feature_ids", bundle.feature_ids))):
        raise RealDataContractError("canonical feature names or ordering changed")
    expected_edge_key = "corrected_edges_sha256" if bundle.dataset == "stanford_crc_codex" else "edges_sha256"
    if observed["edges_sha256"] != expected[expected_edge_key]:
        raise RealDataContractError("canonical graph endpoints or ordering changed")
    if bundle.dataset == "whats_cooking":
        if observed["edge_weights_sha256"] != expected["edge_weights_sha256"]:
            raise RealDataContractError("canonical Cooking graph weights changed")
    elif bundle.metadata.get("raw_coordinates_sha256") != expected["coordinates_sha256"]:
        raise RealDataContractError("canonical spatial coordinates changed")
    bundle.metadata["canonical_contract_validated"] = True
    bundle.metadata["canonical_contract_hashes"] = observed
    return bundle


def load_real_data(dataset: str, *, group: str = "BALBc-1") -> RealDataBundle:
    """Load a validated named dataset.

    ``dataset`` is one of ``crc``, ``spleen`` (``group`` = BALBc-1/2/3 or
    ``joint``), ``cook`` (historical 13,597 x 1,019 corpus), or ``cook_v2``
    (raw recipes with >= 8 ingredients and a rebuilt Jaccard graph; build it
    with ``scripts/data/prepare_cook_v2.py``).  Long canonical names are
    accepted as aliases.
    """

    aliases = {
        "crc": "stanford_crc_codex",
        "stanford_crc_codex": "stanford_crc_codex",
        "spleen": "mouse_spleen_codex",
        "mouse_spleen_codex": "mouse_spleen_codex",
        "cook": "whats_cooking",
        "whats_cooking": "whats_cooking",
        "cook_v2": "whats_cooking_raw_min8_jaccard_v2",
        "cook_raw_min8_jaccard_v2": "whats_cooking_raw_min8_jaccard_v2",
        "whats_cooking_raw_min8_jaccard_v2": "whats_cooking_raw_min8_jaccard_v2",
    }
    try:
        canonical = aliases[dataset]
    except KeyError as error:
        raise ValueError(f"unknown real dataset {dataset!r}") from error
    if canonical == "stanford_crc_codex":
        bundle = load_crc()
    elif canonical == "mouse_spleen_codex":
        if group == JOINT_SPLEEN_GROUP:
            return load_joint_spleen()
        bundle = load_spleen(group)
    elif canonical == "whats_cooking":
        bundle = load_cook()
    else:
        # The v2 loader validates its own manifest and hashes.
        from .cook_min8_jaccard_v2 import load_contract

        return load_contract()
    return validate_frozen_contract(bundle)

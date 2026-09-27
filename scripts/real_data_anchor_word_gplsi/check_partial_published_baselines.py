#!/usr/bin/env python3
"""Check reproducible pieces of the published baselines at the primary K values.

The graph-GpLSI producer is statically blocked by missing historical source and
dependencies, so this script deliberately checks only pLSI and sklearn LDA.
Results are diagnostic and cannot open the experiment stage gate by themselves.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import pickle
from pathlib import Path
import platform
import subprocess
import time
from collections import Counter
from typing import Any
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import svds
import sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from gplsi.recovery import project_rows_simplex


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
SOURCE_REPO = Path(os.environ.get("GPLSI_PUBLISHED_REPO", str(REPO_ROOT)))
ARTIFACT_COMMIT = "7367525f9e5d4f47c272e71f13c582f9a7510615"


def git_bytes(path: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), "show", f"{ARTIFACT_COMMIT}:{path}"]
    )


def load_crc() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    root = DATA_ROOT / "stanford-crc"
    source = root / "output" / "output_3hop"
    meta = pd.read_csv(root / "charville_labels.csv")
    frames = []
    for region in meta.loc[meta["primary_outcome"].notna(), "region_id"].astype(str):
        D = pd.read_csv(f"{source / region}.D.csv", index_col=0)
        frames.append(D.loc[D.sum(axis=1) >= 10])
    counts = pd.concat(frames).to_numpy(dtype=float)
    pW = "data/stanford-crc/model/model_3hop/Whats_aligned"
    pA = "data/stanford-crc/model/model_3hop/Ahats_aligned"
    W_plsi = pd.read_csv(
        io.BytesIO(git_bytes(f"{pW}/Whats_plsi/What_plsi_6_aligned.csv")),
        header=None,
    ).to_numpy(float)
    A_plsi = pd.read_csv(
        io.BytesIO(git_bytes(f"{pA}/Ahats_plsi/Ahat_plsi_6_aligned.csv")),
        header=None,
    ).to_numpy(float)
    W_lda = pd.read_csv(
        io.BytesIO(git_bytes(f"{pW}/Whats_lda/What_lda_6_aligned.csv")),
        header=None,
    ).to_numpy(float)
    A_lda = pd.read_csv(
        io.BytesIO(git_bytes(f"{pA}/Ahats_lda/Ahat_lda_6_aligned.csv")),
        header=None,
    ).to_numpy(float)
    return counts, W_plsi, A_plsi, W_lda, A_lda, 6


def load_spleen() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    counts = pd.read_pickle(DATA_ROOT / "spleen/dataset/merged_D.pkl").loc[
        "BALBc-1"
    ].to_numpy(float)
    artifact = pickle.loads(
        git_bytes("data/spleen/model/BALBc-1_spleen_model_results_5_.pkl")
    )[1]
    return (
        counts,
        np.asarray(artifact["Whats"][1], dtype=float),
        np.asarray(artifact["Ahats"][1], dtype=float).T,
        np.asarray(artifact["Whats"][3], dtype=float),
        np.asarray(artifact["Ahats"][3], dtype=float),
        5,
    )


def _sample(group: pd.DataFrame) -> pd.DataFrame:
    return group.sample(2000, random_state=1) if len(group) > 2000 else group


def load_cook() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    root = DATA_ROOT / "whats-cooking/dataset"
    raw = pd.read_json(root / "train.json")
    with (root / "ingredient_mapping.pkl").open("rb") as handle:
        mapping = pickle.load(handle)
    reverse = {value: key for key, values in mapping.items() for value in values}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sampled = (
            raw.groupby("cuisine", group_keys=False).apply(_sample).reset_index(drop=True)
        )
        sampled["ingredients"] = sampled["ingredients"].apply(
            lambda values: [reverse.get(value, value) for value in values]
        )
        strings = sampled["ingredients"].apply(lambda values: ",".join(values))
        corpus = Counter(value for values in sampled["ingredients"] for value in values)
        vocabulary = [value for value, count in corpus.items() if count >= 10]
        vectorizer = CountVectorizer(
            tokenizer=lambda value: value.split(","), vocabulary=vocabulary
        )
        matrix = vectorizer.fit_transform(strings)
    counts = np.asarray(matrix.toarray())
    counts = counts[counts.sum(axis=1) >= 10]
    counts = counts[:, counts.sum(axis=0) >= 10].astype(float)
    artifact = pickle.loads(
        git_bytes("data/whats-cooking/model/cooking_model_results_7.pkl")
    )[0]
    return (
        counts,
        np.asarray(artifact["Whats"][1], dtype=float),
        np.asarray(artifact["Ahats"][1], dtype=float).T,
        np.asarray(artifact["Whats"][2], dtype=float),
        np.asarray(artifact["Ahats"][2], dtype=float),
        7,
    )


def historical_plsi(X: np.ndarray, K: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    U, _, _ = svds(X, k=K, rng=np.random.default_rng(seed))
    U = U.copy()
    U[:, U[0] < 0] *= -1
    residual = U.T.copy()
    selected: list[int] = []
    for _ in range(K):
        index = int(np.argmax(np.linalg.norm(residual, axis=0)))
        selected.append(index)
        vector = residual[:, index][:, None]
        denominator = float((vector.T @ vector).item())
        residual = (np.eye(K) - vector @ vector.T / denominator) @ residual
    H = U[np.asarray(selected)]
    W_raw = U @ (H.T @ np.linalg.inv(H @ H.T))
    W = project_rows_simplex(W_raw)
    A_raw = np.linalg.inv(W.T @ W) @ W.T @ X
    A = project_rows_simplex(A_raw)
    return W, A


def compare(
    W: np.ndarray, A: np.ndarray, reference_W: np.ndarray, reference_A: np.ndarray
) -> dict[str, Any]:
    norms = np.linalg.norm(W, axis=0)[:, None] * np.linalg.norm(reference_W, axis=0)[None]
    similarity = (W.T @ reference_W) / np.maximum(norms, np.finfo(float).eps)
    candidate, reference = linear_sum_assignment(-similarity)
    order = candidate[np.argsort(reference)]
    W_aligned = W[:, order]
    A_aligned = A[order]

    def stats(value: np.ndarray, target: np.ndarray) -> dict[str, float]:
        difference = value - target
        return {
            "max_abs": float(np.max(np.abs(difference))),
            "mean_abs": float(np.mean(np.abs(difference))),
            "relative_frobenius": float(
                np.linalg.norm(difference) / max(np.linalg.norm(target), 1e-15)
            ),
        }

    W_stats = stats(W_aligned, reference_W)
    A_stats = stats(A_aligned, reference_A)
    return {
        "topic_permutation_candidate_to_published": order.tolist(),
        "matched_W_cosines": similarity[order, np.arange(reference_W.shape[1])].tolist(),
        "W": W_stats,
        "A": A_stats,
        "agreement": (
            "exact"
            if max(W_stats["max_abs"], A_stats["max_abs"]) <= 1e-10
            else "near_identity"
            if max(W_stats["relative_frobenius"], A_stats["relative_frobenius"])
            <= 1e-5
            else "mismatch"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("crc", "spleen", "cook"),
        default=("crc", "spleen", "cook"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "results/real_data_anchor_word_gplsi/audit/partial_baseline_reproduction.json",
    )
    args = parser.parse_args()
    loaders = {"crc": load_crc, "spleen": load_spleen, "cook": load_cook}
    output: dict[str, Any] = {
        "artifact_commit": ARTIFACT_COMMIT,
        "diagnostic_only": True,
        "stage_gate_open": False,
        "stage_gate_reason": "published graph-GpLSI producer cannot be reconstructed exactly",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        "datasets": {},
    }
    for name in args.datasets:
        counts, ref_Wp, ref_Ap, ref_Wl, ref_Al, K = loaders[name]()
        X = counts / counts.sum(axis=1, keepdims=True)
        dataset: dict[str, Any] = {"n": int(X.shape[0]), "p": int(X.shape[1]), "K": K}
        started = time.perf_counter()
        Wp, Ap = historical_plsi(X, K)
        dataset["plsi"] = compare(Wp, Ap, ref_Wp, ref_Ap)
        dataset["plsi"]["runtime_seconds"] = time.perf_counter() - started
        published_A_refit = project_rows_simplex(
            np.linalg.inv(ref_Wp.T @ ref_Wp) @ ref_Wp.T @ X
        )
        dataset["plsi"]["published_A_internal_consistency"] = compare(
            ref_Wp, published_A_refit, ref_Wp, ref_Ap
        )["A"]
        dataset["plsi"]["published_reconstruction_frobenius"] = float(
            np.linalg.norm(ref_Wp @ ref_Ap - X)
        )
        dataset["plsi"]["refit_reconstruction_frobenius"] = float(
            np.linalg.norm(ref_Wp @ published_A_refit - X)
        )
        started = time.perf_counter()
        lda = LatentDirichletAllocation(n_components=K, random_state=0).fit(counts)
        Wl = lda.transform(counts)
        Al = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
        dataset["lda"] = compare(Wl, Al, ref_Wl, ref_Al)
        dataset["lda"]["runtime_seconds"] = time.perf_counter() - started
        dataset["published_gplsi"] = {
            "agreement": "not_run_static_provenance_failure",
            "opens_stage_gate": False,
        }
        output["datasets"][name] = dataset
        print(name, dataset["plsi"]["agreement"], dataset["lda"]["agreement"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import Counter

import json
import gc
import time
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from . import gplsi


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def jaccard_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Jaccard-like similarity for binary/count ingredient vectors.

    Here defined as:
        intersection / (n - intersection)
    where 'intersection' is number of positions with x_i > 0 and y_i > 0,
    and n is vector length.
    """
    intersection = np.count_nonzero(x * y)
    n = len(x)
    return intersection / (n - intersection) if n > intersection else 0.0


def sample_cuisine(group: pd.DataFrame, max_per_cuisine: int = 2000) -> pd.DataFrame:
    """Downsample cuisines to at most max_per_cuisine rows each."""
    if len(group) > max_per_cuisine:
        return group.sample(max_per_cuisine, random_state=1)
    return group


def map_ingredient(ingredient: str, reverse_mapping: Dict[str, str]) -> str:
    """Map ingredient to canonical form using reverse_mapping."""
    return reverse_mapping.get(ingredient, ingredient)


def get_threshold(alpha: float, n: int, p: int, N: float) -> float:
    """
    Threshold used for ingredient frequency filtering.

    t = alpha * sqrt( log(max(n, p)) / (n * N) )
    """
    return alpha * np.sqrt(np.log(max(n, p)) / (n * N))


def preprocess_cook(
    df: pd.DataFrame,
    ingredient_mapping: Dict[str, List[str]],
    neighbor_countries_mapping: Dict[str, List[str]],
    threshold: bool = True,
    min_ingredient_count: int = 10,
    alpha: float = 0.005,
    top_k_neighbors: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, pd.DataFrame, csr_matrix, int]:
    """
    Preprocess the WhatsCooking dataset:
      - balance cuisine counts
      - map ingredients to canonical categories
      - drop rare ingredients and cuisines
      - build ingredient count matrix D
      - construct a country-neighbor graph based on Jaccard similarity
      - produce row-normalized X and graph weights.

    Parameters
    ----------
    df : pd.DataFrame
        Raw WhatsCooking DataFrame with 'cuisine' and 'ingredients' fields
        (ingredients as list of strings).
    ingredient_mapping : dict
        Mapping from canonical ingredient to list of raw ingredient variants.
    neighbor_countries_mapping : dict
        Mapping from cuisine to list of "neighbor" cuisines (graph prior).
    threshold : bool
        Whether to apply frequency thresholding on average ingredient use.
    min_ingredient_count : int
        Minimal ingredient count across dataset to keep the ingredient.
    alpha : float
        Threshold parameter for dropping low-frequency ingredients.
    top_k_neighbors : int
        For each node, connect to top_k_neighbors neighbors by similarity.

    Returns
    -------
    ingredient_df : pd.DataFrame
        Processed ingredient × cuisine data with extra columns 'ingredients',
        'cuisine', and 'node' (integer node id).
    D : pd.DataFrame
        Count matrix (rows = nodes, columns = ingredients).
    X : pd.DataFrame
        Row-normalized version of D (rows sum to 1).
    N : float
        Average row sum of D.
    edge_df : pd.DataFrame
        Edge list DataFrame with columns ['src', 'tgt', 'weight'].
    weights : csr_matrix
        Sparse adjacency/weight matrix (n x n).
    n : int
        Number of nodes (rows in X).
    """
    # --------------------------------------------------------------
    # 1. Balance cuisines
    # --------------------------------------------------------------
    sampled_df = (
        df.groupby("cuisine", group_keys=False)
        .apply(sample_cuisine)
        .reset_index(drop=True)
    )

    # --------------------------------------------------------------
    # 2. Build reverse mapping for ingredients
    # --------------------------------------------------------------
    reverse_mapping: Dict[str, str] = {}
    for key, values in ingredient_mapping.items():
        for value in values:
            reverse_mapping[value] = key

    sampled_df["ingredients"] = sampled_df["ingredients"].apply(
        lambda x: [map_ingredient(ing, reverse_mapping) for ing in x]
    )
    sampled_df["ingredients_str"] = sampled_df["ingredients"].apply(
        lambda x: ",".join(x)
    )

    # --------------------------------------------------------------
    # 3. Drop rare ingredients
    # --------------------------------------------------------------
    raw_ingred: List[str] = []
    for ingredients_list in sampled_df["ingredients"]:
        raw_ingred.extend(ingredients_list)

    ingredient_counts = Counter(raw_ingred)
    ingredient_vocab = [
        key
        for key in ingredient_counts.keys()
        if ingredient_counts[key] >= min_ingredient_count
    ]

    vectorizer = CountVectorizer(
        tokenizer=lambda x: x.split(","),
        vocabulary=ingredient_vocab,
    )
    ingredient_matrix = vectorizer.fit_transform(sampled_df["ingredients_str"])
    ingredient_df = pd.DataFrame(
        ingredient_matrix.toarray(),
        columns=vectorizer.get_feature_names_out(),
    )

    # --------------------------------------------------------------
    # 4. Attach cuisine and raw ingredients, drop low-count rows
    # --------------------------------------------------------------
    ingredient_df["ingredients"] = sampled_df["ingredients_str"]
    ingredient_df["cuisine"] = sampled_df["cuisine"].values

    # rows with total ingredient count < 10
    ingredient_df = ingredient_df[
        ingredient_df.iloc[:, :-2].sum(axis=1) >= min_ingredient_count
    ]

    # --------------------------------------------------------------
    # 5. Drop columns that never reach count >= 10
    # --------------------------------------------------------------
    column_sums = ingredient_df.iloc[:, :-2].sum(axis=0)
    columns_to_keep = column_sums[column_sums >= min_ingredient_count].index
    ingredient_df = ingredient_df[list(columns_to_keep) + ["ingredients", "cuisine"]]

    # --------------------------------------------------------------
    # 6. Subset words via thresholding of mean usage
    # --------------------------------------------------------------
    n_rows, n_cols = ingredient_df.shape
    # columns except ['ingredients', 'cuisine']
    n, p = ingredient_df.iloc[:, :-2].shape
    N = float(ingredient_df.iloc[:, :-2].sum(axis=1).max())
    M = ingredient_df.iloc[:, :-2].mean(axis=0)
    t = get_threshold(alpha, n, p, N)
    mask = M > t

    if threshold:
        ingredient_df = ingredient_df.loc[:, mask.tolist() + [True, True]]  # keep last 2

    ingredient_df = ingredient_df.reset_index(drop=True)
    ingredient_df["node"] = ingredient_df.index

    print(f"[Cook] There are {ingredient_df.shape[0]} cuisines (rows).")
    print(f"[Cook] There are {ingredient_df.shape[1] - 3} ingredients (columns).")

    # --------------------------------------------------------------
    # 7. Build edge list based on Jaccard similarity between neighbors
    # --------------------------------------------------------------
    edge_list: List[Tuple[int, int, float]] = []

    ingredient_array = ingredient_df.values
    for source in range(len(ingredient_array)):
        x = ingredient_array[source]
        x_array = x[:-3]  # all ingredient columns
        cuisine = x[-2]
        neighbor_list = neighbor_countries_mapping.get(cuisine, [])
        neighbor_indices = ingredient_df[
            ingredient_df["cuisine"].isin(neighbor_list)
        ].index

        if len(neighbor_indices) == 0:
            continue

        neighbor_array = ingredient_array[neighbor_indices, :-3]
        neighbor_nodes = ingredient_array[neighbor_indices, -1]
        similarity = np.array(
            [jaccard_similarity(row, x_array) for row in neighbor_array]
        )

        if len(similarity) == 0 or np.all(similarity == 0):
            continue

        k = min(top_k_neighbors, len(similarity))
        top_idx = np.argpartition(similarity, -k)[-k:]
        neighbor_nodes_top = [neighbor_nodes[i] for i in top_idx]
        for idx in neighbor_nodes_top:
            src, tgt = sorted([source, int(idx)])
            edge_list.append((src, tgt, 1.0))

    edge_df = pd.DataFrame(edge_list, columns=["src", "tgt", "weight"]).drop_duplicates()
    print(f"[Cook] There are {edge_df.shape[0]} edges.")

    # --------------------------------------------------------------
    # 8. Count matrix D, normalized X, and weights matrix
    # --------------------------------------------------------------
    D = ingredient_df.iloc[:, :-3]
    row_sums = D.sum(axis=1)
    N = float(row_sums.mean())
    X = D.div(row_sums, axis=0)
    n = X.shape[0]
    weights = csr_matrix(
        (edge_df["weight"].values, (edge_df["src"].values, edge_df["tgt"].values)),
        shape=(n, n),
    )

    return ingredient_df, D, X, N, edge_df, weights, n


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------


def run_cook_analysis(
    K: int,
    lamb_start: float,
    step_size: float,
    grid_len: int,
    eps: float,
    data_root: Path,
    threshold: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run GpLSI, pLSI, and LDA on the WhatsCooking dataset.

    Parameters
    ----------
    K : int
        Number of topics.
    lamb_start, step_size, grid_len, eps :
        Hyperparameters for GpLSI.
    data_root : Path
        Path to 'whats-cooking' data root (contains 'dataset/').
    threshold : bool
        Whether to apply ingredient frequency thresholding in preprocess.

    Returns
    -------
    results : list of dict
        Single-element list with keys:
        - 'Whats', 'Ahats', 'times', 'edge_df', 'ingredient_df', 'D'
    models : dict
        Dictionary with keys 'gplsi', 'plsi', 'lda', and basic metadata.
    """
    root_path = data_root
    dataset_root = root_path / "dataset"

    path_to_data = dataset_root / "train.json"
    ingredient_mapping_path = dataset_root / "ingredient_mapping.pkl"
    neighbor_mapping_path = dataset_root / "neighbor_countries_mapping.pkl"

    if not path_to_data.exists():
        raise FileNotFoundError(f"train.json not found at {path_to_data}")

    with open(path_to_data) as data_file:
        data = json.load(data_file)

    with open(ingredient_mapping_path, "rb") as f:
        ingredient_mapping = pickle.load(f)

    with open(neighbor_mapping_path, "rb") as f:
        neighbor_countries_mapping = pickle.load(f)

    ingredient_df_path = dataset_root / "processed_ingredient_df.pkl"
    edge_df_path = dataset_root / "processed_edge_df.pkl"
    D_path = dataset_root / "D_df.pkl"

    # --------------------------------------------------------------
    # Preprocessing (with caching)
    # --------------------------------------------------------------
    if ingredient_df_path.exists() and edge_df_path.exists() and D_path.exists():
        with open(ingredient_df_path, "rb") as f:
            ingredient_df = pickle.load(f)
        with open(edge_df_path, "rb") as f:
            edge_df = pickle.load(f)
        with open(D_path, "rb") as f:
            D = pickle.load(f)

        # Recompute X, N, weights from cached D / edge_df
        row_sums = D.sum(axis=1)
        N = float(row_sums.mean())
        X = D.div(row_sums, axis=0)
        n = X.shape[0]
        weights = csr_matrix(
            (edge_df["weight"].values, (edge_df["src"].values, edge_df["tgt"].values)),
            shape=(n, n),
        )
        print("[Cook] Loaded cached processed data.")
    else:
        df = pd.DataFrame(data)
        ingredient_df, D, X, N, edge_df, weights, n = preprocess_cook(
            df,
            ingredient_mapping=ingredient_mapping,
            neighbor_countries_mapping=neighbor_countries_mapping,
            threshold=threshold,
        )
        del df
        gc.collect()

        with open(ingredient_df_path, "wb") as f:
            pickle.dump(ingredient_df, f)
        with open(edge_df_path, "wb") as f:
            pickle.dump(edge_df, f)
        with open(D_path, "wb") as f:
            pickle.dump(D, f)

        print("[Cook] Saved processed data to cache.")

    # --------------------------------------------------------------
    # Models: GpLSI, pLSI, LDA
    # --------------------------------------------------------------
    # GpLSI
    start_time = time.time()
    model_gplsi = gplsi.GpLSI_(
        lamb_start=lamb_start,
        step_size=step_size,
        grid_len=grid_len,
        eps=eps,
    )
    model_gplsi.fit(X.values, N, K, edge_df, weights)
    time_gplsi = time.time() - start_time

    # pLSI
    start_time = time.time()
    model_plsi = gplsi.GpLSI_(method="pLSI")
    model_plsi.fit(X.values, N, K, edge_df, weights)
    time_plsi = time.time() - start_time

    # LDA
    start_time = time.time()
    model_lda = LatentDirichletAllocation(
        n_components=K,
        random_state=0,
    )
    model_lda.fit(D.values)
    time_lda = time.time() - start_time

    # --------------------------------------------------------------
    # Collect W, A, and timings
    # --------------------------------------------------------------
    W_gplsi = model_gplsi.W_hat
    W_plsi = model_plsi.W_hat
    W_lda = model_lda.transform(D.values)

    A_hat_gplsi = model_gplsi.A_hat.T
    A_hat_plsi = model_plsi.A_hat.T
    A_hat_lda = (
        model_lda.components_ / model_lda.components_.sum(axis=1)[:, np.newaxis]
    )

    names = ["GpLSI", "pLSI", "LDA"]
    times = [time_gplsi, time_plsi, time_lda]
    Whats = [W_gplsi, W_plsi, W_lda]
    Ahats = [A_hat_gplsi, A_hat_plsi, A_hat_lda]

    print(f"[Cook] Runtimes (GpLSI, pLSI, LDA): {times}")

    results: List[Dict[str, Any]] = []
    results.append(
        {
            "names": names,
            "Whats": Whats,
            "Ahats": Ahats,
            "times": times,
            "edge_df": edge_df,
            "ingredient_df": ingredient_df,
            "D": D,
        }
    )

    models: Dict[str, Any] = {
        "gplsi": model_gplsi,
        "plsi": model_plsi,
        "lda": model_lda,
        "K": K,
    }

    return results, models
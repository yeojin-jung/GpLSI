"""Helpers shared by the analysis scripts (docs/protocol.md).

They read the pipeline's result layout (``gplsi.pipeline.results``),
``results/<run>/``, and follow the reporting conventions first used in the
2026-09 handoff reports:

* one representative fit per method and K: the smallest successful seed,
  chosen without looking at W (:func:`representative_fits`);
* topics of every fit matched one-to-one to the same-K LDA fit of that seed by
  a Hungarian assignment maximising the cosine similarity of full A rows
  (:func:`align_topics`); the matching is for display only;
* argmax W taken in the original topic order, before any normalisation or
  permutation, so ties resolve to the lowest original topic (:func:`hard_topics`);
* near-pure documents, ``W[i, k] >= 0.95`` after row normalisation
  (:func:`near_pure_counts`);
* mean-W and argmax-share compositions of groups, optionally pooling rows
  into parent units (patients) that then count equally (:func:`group_composition`);
* label agreement of argmax W: ARI, NMI, AMI and in-sample purity (:func:`label_agreement`);
* metric curves: seed mean with SE ``sd / sqrt(n)`` only when n >= 2 (:func:`seed_summary`).

Scripts import it with ``sys.path`` pointing at ``scripts/analysis``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from gplsi.pipeline import load_arrays, load_config, load_rows
from gplsi.real_data import REPO_ROOT

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
PURITY_THRESHOLD = 0.95

FAMILY_LABEL = {
    "lda": "LDA",
    "plsi": "pLSI",
    "spatial_lda": "Spatial LDA",
    "topicscore_raw": "Topic-SCORE",
    "topicscore_graph_denoised": "Topic-SCORE (graph)",
    "kl_nmf": "KL-NMF",
    "graph_kl_nmf": "graph KL-NMF",
}
HUNTER_LABEL = {
    "spa_current": "SPA",
    "svs": "SVS",
    "svs_star": "SVS*",
    "pp_spa": "pp-SPA",
    "palm": "PALM",
    "palm_accelerated": "PALM-AA",
}
PREPROCESSING_LABEL = {
    "P0_raw": "Original",
    "P1_tran_alpha_0p005": "Tran",
    "P2_ke_weighted": "Ke",
    "P3_tran_then_ke": "Tran + Ke",
}
# Handoff A conventions: pLSI keeps its current A, GpLSI uses Poisson A,
# the other baselines their native A.
DEFAULT_RECOVERY = {"plsi": "A_current", "document_gplsi": "A_full_Pois", "anchor_feature_gplsi": "A_full_Pois"}
METHOD_ORDER = ["lda", "plsi", "spatial_lda", "topicscore_raw", "topicscore_graph_denoised",
                "kl_nmf", "graph_kl_nmf", "document_gplsi", "anchor_feature_gplsi"]
# Colour-blind-safe categorical palette, fixed slot order.
TOPIC_COLORS = ["#E69F00", "#0072B2", "#009E73", "#CC79A7", "#D55E00", "#56B4E9",
                "#332288", "#999933", "#882255", "#444444", "#117733", "#AA4499"]


def run_directory(target: str | Path) -> Path:
    """Results directory of an experiment config, or the directory itself."""

    target = Path(target)
    if target.suffix == ".json":
        config = load_config(target)
        return REPO_ROOT / config.get("output_root", "results") / config["name"]
    return target


def method_label(row: dict[str, Any] | pd.Series) -> str:
    family = row["estimator_family"]
    if family in ("document_gplsi", "anchor_feature_gplsi"):
        hunter = HUNTER_LABEL.get(row["vertex_hunter"], row["vertex_hunter"])
        preprocessing = PREPROCESSING_LABEL.get(row["preprocessing"], row["preprocessing"])
        prefix = "GpLSI" if family == "document_gplsi" else "GpLSI anchor"
        return f"{prefix} {hunter} ({preprocessing})"
    return FAMILY_LABEL.get(family, family)


def method_sort_key(row: dict[str, Any] | pd.Series) -> tuple:
    family = row["estimator_family"]
    return (
        METHOD_ORDER.index(family) if family in METHOD_ORDER else len(METHOD_ORDER),
        list(PREPROCESSING_LABEL).index(row["preprocessing"]) if row["preprocessing"] in PREPROCESSING_LABEL else 9,
        list(HUNTER_LABEL).index(row["vertex_hunter"]) if row["vertex_hunter"] in HUNTER_LABEL else 9,
    )


def select_rows(
    frame: pd.DataFrame,
    *,
    recoveries: dict[str, str] | None = None,
    families: Iterable[str] | None = None,
    hunters: Iterable[str] | None = None,
    preprocessings: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Keep one A recovery per family (``DEFAULT_RECOVERY``) and the requested methods."""

    recoveries = {**DEFAULT_RECOVERY, **(recoveries or {})}
    keep = np.ones(len(frame), dtype=bool)
    for family, recovery in recoveries.items():
        keep &= ~((frame["estimator_family"] == family) & (frame["A_recovery"] != recovery))
    if families is not None:
        keep &= frame["estimator_family"].isin(list(families))
    gplsi = frame["estimator_family"].isin(["document_gplsi", "anchor_feature_gplsi"])
    if hunters is not None:
        keep &= ~gplsi | frame["vertex_hunter"].isin(list(hunters))
    if preprocessings is not None:
        keep &= ~gplsi | frame["preprocessing"].isin(list(preprocessings))
    output = frame.loc[keep].copy()
    output["label"] = [method_label(row) for _, row in output.iterrows()]
    return output


def load_selected_rows(target: str | Path, **selection: Any) -> pd.DataFrame:
    frame = load_rows(run_directory(target))
    if frame.empty:
        raise SystemExit(f"no rows under {run_directory(target)}")
    return select_rows(frame, **selection)


@dataclass
class Fit:
    row: dict[str, Any]
    W: np.ndarray
    A: np.ndarray
    label: str
    order: np.ndarray = field(default=None)  # aligned topic k is original topic order[k]
    cosine: np.ndarray = field(default=None)

    @property
    def K(self) -> int:
        return int(self.row["K"])

    @property
    def task_dir(self) -> Path:
        return Path(self.row["task_dir"])


def representative_fits(rows: pd.DataFrame, K: int, *, seed: int | None = None) -> list[Fit]:
    """One successful fit per method at K: the given seed, else the smallest successful seed."""

    rows = rows[(rows["K"] == K) & (rows["status"] == "ok")]
    fits = []
    for _, group in rows.groupby("label", sort=False):
        group = group.sort_values("seed")
        if seed is not None and (group["seed"] == seed).any():
            group = group[group["seed"] == seed]
        row = group.iloc[0].to_dict()
        arrays = load_arrays(row)
        W = normalize_rows(arrays["W_hat"])
        A = normalize_rows(arrays["A_hat"])
        fits.append(Fit(row=row, W=W, A=A, label=row["label"]))
    fits.sort(key=lambda fit: method_sort_key(fit.row))
    return fits


def task_data(task_dir: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(task_dir) / "data.npz", allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def normalize_rows(matrix: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    """Clip roundoff negatives (>= -tolerance) and scale rows to sum to one."""

    matrix = np.asarray(matrix, dtype=float)
    if not np.isfinite(matrix).all() or matrix.min() < -tolerance:
        raise ValueError("matrix has non-finite or materially negative entries")
    matrix = np.maximum(matrix, 0.0)
    totals = matrix.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("matrix has a zero row")
    return matrix / totals


def align_topics(A: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order with ``A[order[k]]`` matched to reference topic k, and the matched cosines."""

    a = A / np.linalg.norm(A, axis=1, keepdims=True)
    r = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    cosine = a @ r.T
    own, ref = linear_sum_assignment(-cosine)
    order = own[np.argsort(ref)]
    return order, cosine[order, np.arange(len(order))]


def align_to_reference(fits: list[Fit], reference: str = "lda") -> Fit:
    """Align every fit in place to the reference family's fit (default LDA)."""

    anchor = next((fit for fit in fits if fit.row["estimator_family"] == reference), fits[0])
    for fit in fits:
        fit.order, fit.cosine = align_topics(fit.A, anchor.A)
    return anchor


def hard_topics(W: np.ndarray, order: np.ndarray | None = None) -> np.ndarray:
    """Argmax topic per row in the original order, relabelled to aligned topic numbers."""

    hard = np.argmax(W, axis=1)
    if order is None:
        return hard
    relabel = np.empty_like(order)
    relabel[order] = np.arange(len(order))
    return relabel[hard]


def near_pure_counts(W: np.ndarray, threshold: float = PURITY_THRESHOLD) -> np.ndarray:
    """Per topic, the number of rows with normalised ``W[i, k] >= threshold``."""

    pure = normalize_rows(W) >= threshold
    if threshold > 0.5 and np.any(pure.sum(axis=1) > 1):
        raise AssertionError("a row is near-pure for two topics")
    return pure.sum(axis=0)


def group_composition(
    W: np.ndarray,
    groups: np.ndarray,
    *,
    parents: np.ndarray | None = None,
) -> pd.DataFrame:
    """Mean W and argmax share per group; columns ``group, statistic, topic, value, n``.

    With ``parents`` (e.g. patient ids), rows are first pooled within each
    parent, and each parent then counts once in its group's average; a
    parent must belong to a single group.
    """

    W = normalize_rows(W)
    K = W.shape[1]
    hard = np.eye(K)[np.argmax(W, axis=1)]
    groups = np.asarray(groups).astype(str)
    keys = groups if parents is None else np.asarray(parents).astype(str)
    frame = pd.DataFrame({"key": keys, "group": groups})
    unit_group = frame.groupby("key", sort=False)["group"].agg(lambda values: values.unique())
    if any(len(values) != 1 for values in unit_group):
        raise ValueError("a parent unit belongs to several groups")
    records = []
    for statistic, values in (("mean_W", W), ("argmax_share", hard)):
        unit = pd.DataFrame(values).groupby(keys, sort=False).mean()
        unit["group"] = [unit_group[key][0] for key in unit.index]
        for group, block in unit.groupby("group"):
            mean = block.drop(columns="group").to_numpy().mean(axis=0)
            records.extend(
                dict(group=group, statistic=statistic, topic=k, value=float(mean[k]), n=len(block))
                for k in range(K)
            )
    return pd.DataFrame(records)


def label_agreement(hard: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """ARI, NMI, AMI and in-sample purity of hard topics against reference labels.

    Purity names each topic after its most frequent label (topics may share a
    label); ``majority_share`` is the purity of a single all-majority cluster.
    """

    labels = np.asarray(labels).astype(str)
    contingency = pd.crosstab(hard, labels).to_numpy()
    return {
        "ari": float(adjusted_rand_score(labels, hard)),
        "nmi": float(normalized_mutual_info_score(labels, hard)),
        "ami": float(adjusted_mutual_info_score(labels, hard)),
        "purity": float(contingency.max(axis=1).sum() / contingency.sum()),
        "majority_share": float(contingency.sum(axis=0).max() / contingency.sum()),
        "n": int(len(labels)),
    }


def seed_summary(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """Mean, SE (``sd / sqrt(n)``, only for n >= 2 seeds) and n per group, long format."""

    records = []
    for key, group in frame.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            records.append({
                **dict(zip(keys, key)),
                "metric": metric,
                "mean": float(values.mean()),
                "se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan,
                "n_seeds": int(len(values)),
            })
    return pd.DataFrame(records)


def similarity_order(names: list[str], profiles: np.ndarray) -> np.ndarray:
    """Row order placing similar profiles together.

    Jensen-Shannon distances, average linkage with optimal leaf ordering, and
    the orientation whose name sequence sorts first (the handoff dashboard's
    cuisine ordering).
    """

    profiles = normalize_rows(profiles)
    if len(names) <= 2:
        return np.arange(len(names))
    distances = pdist(profiles, metric="jensenshannon")
    if not np.isfinite(distances).all() or distances.max() <= 1e-12:
        return np.arange(len(names))
    order = leaves_list(linkage(distances, method="average", optimal_ordering=True))
    forward = tuple(names[i] for i in order)
    return order[::-1] if tuple(reversed(forward)) < forward else order


def raw_cell_ids(observation_ids: np.ndarray) -> np.ndarray:
    """Integer cell id at the end of each observation id (``'BALBc-1::('BALBc-1', 7)'`` -> 7)."""

    return np.array([int(re.findall(r"\d+", str(value))[-1]) for value in observation_ids])


def matched_topic_colors(hard: np.ndarray, labels: np.ndarray, label_colors: dict[str, str], K: int) -> list[str]:
    """Colour each topic by its Hungarian-matched reference label; unmatched topics get greys.

    Display only: a shared colour marks the best one-to-one overlap, not an identity.
    """

    present = [label for label in label_colors if np.any(labels == label)]
    overlap = np.array([[np.sum((hard == k) & (labels == label)) for label in present] for k in range(K)])
    rows, cols = linear_sum_assignment(-overlap)
    colors = {int(r): label_colors[present[c]] for r, c in zip(rows, cols) if overlap[r, c] > 0}
    greys = iter(["#444444", "#8a8985", "#b8b7b0", "#6b6a66", "#d0cfca"] * 3)
    return [colors.get(k) or next(greys) for k in range(K)]


def consensus_alignment(profiles: list[np.ndarray], iterations: int = 5) -> tuple[list[np.ndarray], np.ndarray]:
    """Match the topics of separately fitted units (same vocabulary) to a common reference.

    Start from the first unit's A; repeatedly match every unit to the
    reference by full-row cosine (Hungarian) and replace the reference by the
    mean of the matched, row-normalized profiles, until the matching stops
    changing. Returns each unit's order (aligned topic k = original
    ``order[k]``) and the consensus profiles. Display and summary use only:
    a matched topic need not be the same biology in every unit.
    """

    profiles = [normalize_rows(A) for A in profiles]
    reference = profiles[0]
    orders: list[np.ndarray] = []
    for _ in range(iterations):
        new_orders = [align_topics(A, reference)[0] for A in profiles]
        reference = normalize_rows(np.mean([A[order] for A, order in zip(profiles, new_orders)], axis=0))
        if orders and all(np.array_equal(a, b) for a, b in zip(orders, new_orders)):
            break
        orders = new_orders
    return orders, reference


def ilr(composition: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Helmert isometric log-ratio coordinates of compositions (rows), after flooring and re-closing."""

    from scipy.linalg import helmert

    composition = np.maximum(np.asarray(composition, dtype=float), floor)
    composition /= composition.sum(axis=1, keepdims=True)
    log = np.log(composition)
    return (log - log.mean(axis=1, keepdims=True)) @ helmert(composition.shape[1]).T


def dataset_file(target: str | Path) -> Path:
    """The processed H5AD of a spatial-unit experiment (DLPFC, MERFISH, Xenium), from its config or results."""

    import json

    target = Path(target)
    if target.suffix == ".json":
        return REPO_ROOT / load_config(target)["dataset"]["file"]
    task = json.loads(next(run_directory(target).glob("*/task.json")).read_text())
    return REPO_ROOT / task["config"]["dataset"]["file"]

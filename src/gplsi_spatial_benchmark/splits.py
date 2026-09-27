"""Count-native thinning and held-out splits shared by all estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.stats import binom


@dataclass(frozen=True)
class CountSplit:
    train: np.ndarray
    test: np.ndarray
    original_total: int
    retained_total: int
    metadata: dict


def thin_and_split_counts(
    counts,
    *,
    retained_fraction: float,
    test_fraction: float,
    seed: int,
) -> CountSplit:
    """Independently assign molecules to discarded, train, or test counts."""

    if not 0 < retained_fraction <= 1 or not 0 < test_fraction < 1:
        raise ValueError("retained_fraction and test_fraction must be in (0,1]")
    array = counts.toarray() if issparse(counts) else np.asarray(counts)
    if array.ndim != 2 or np.any(array < 0) or not np.all(array == np.floor(array)):
        raise ValueError("counts must be a nonnegative integer matrix")
    array = array.astype(np.int64, copy=False)
    # Shared inverse-CDF uniforms couple every retention level monotonically:
    # for a fixed seed, 25% counts are a subset (in count order) of 50%, 75%,
    # and 100%. A separate shared stream couples the train/test assignment.
    retained_rng, test_rng = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2)]
    lower = np.nextafter(0.0, 1.0)
    retained_u = np.clip(retained_rng.random(array.shape), lower, 1.0)
    test_u = np.clip(test_rng.random(array.shape), lower, 1.0)
    retained = binom.ppf(retained_u, array, retained_fraction).astype(np.int64)
    test = binom.ppf(test_u, retained, test_fraction).astype(np.int64)
    train = retained - test
    keep = train.sum(axis=1) > 0
    if not np.all(keep):
        # Estimators require positive training lengths. The same deterministic
        # row mask is applied once, before every method sees the split.
        train = train[keep]
        test = test[keep]
    return CountSplit(
        train=train.astype(np.int32),
        test=test.astype(np.int32),
        original_total=int(array.sum()),
        retained_total=int(retained.sum()),
        metadata={
            "seed": int(seed),
            "retained_fraction": float(retained_fraction),
            "test_fraction": float(test_fraction),
            "row_keep_mask": keep,
            "dropped_zero_training_rows": int(np.count_nonzero(~keep)),
        },
    )

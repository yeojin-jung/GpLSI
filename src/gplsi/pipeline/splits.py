"""Count-native thinning and held-out splits shared by all estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import binom


@dataclass(frozen=True)
class SparseCountSplit:
    train: csr_matrix
    test: csr_matrix
    original_total: int
    retained_total: int
    metadata: dict


def thin_and_split_sparse_counts(
    counts,
    *,
    retained_fraction: float,
    test_fraction: float,
    seed: int,
) -> SparseCountSplit:
    """Split counts into nested retained subsets and train/test molecules.

    Uniforms are drawn only for stored nonzero entries (zeros stay zero), with
    the same inverse-CDF coupling: for a fixed seed, retained counts are nested
    across retention fractions. Rows are *not* dropped here; the caller decides
    the row mask after choosing its feature panel.
    """

    if not 0 < retained_fraction <= 1 or not 0 < test_fraction < 1:
        raise ValueError("retained_fraction and test_fraction must be in (0,1]")
    matrix = csr_matrix(counts, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    values = matrix.data
    if np.any(values < 0) or not np.all(values == np.floor(values)):
        raise ValueError("counts must be a nonnegative integer matrix")
    values = values.astype(np.int64)
    retained_rng, test_rng = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2)]
    lower = np.nextafter(0.0, 1.0)
    retained_u = np.clip(retained_rng.random(values.shape), lower, 1.0)
    test_u = np.clip(test_rng.random(values.shape), lower, 1.0)
    retained = binom.ppf(retained_u, values, retained_fraction).astype(np.int64)
    test = binom.ppf(test_u, retained, test_fraction).astype(np.int64)
    train = retained - test

    def rebuild(data: np.ndarray) -> csr_matrix:
        out = csr_matrix((data.astype(np.int32), matrix.indices.copy(), matrix.indptr.copy()), shape=matrix.shape)
        out.eliminate_zeros()
        return out

    return SparseCountSplit(
        train=rebuild(train),
        test=rebuild(test),
        original_total=int(values.sum()),
        retained_total=int(retained.sum()),
        metadata={
            "seed": int(seed),
            "retained_fraction": float(retained_fraction),
            "test_fraction": float(test_fraction),
            "implementation": "sparse_nonzero_inverse_cdf",
        },
    )

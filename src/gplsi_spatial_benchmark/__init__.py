"""Leakage-audited spatial transcriptomics benchmark for GpLSI."""

from .data import SpatialSlice, load_spatial_slice
from .graph import build_within_unit_knn_graph
from .splits import CountSplit, thin_and_split_counts

__all__ = [
    "SpatialSlice",
    "CountSplit",
    "build_within_unit_knn_graph",
    "load_spatial_slice",
    "thin_and_split_counts",
]

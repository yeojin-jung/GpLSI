"""
gplsi: Graph-regularized pLSI (GpLSI) for topic modeling on data with spatial / covariate dependencies.

This package provides:
- GpLSI       : main model class
- generate_data, generate_weights_edge : simulation helpers
- graphSVD     : graph-regularized SVD 
- utility functions for aligning and evaluating topics
"""

from importlib import import_module
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("gplsi")
except PackageNotFoundError:  
    __version__ = "0.0.0"


# Keep package import lightweight. Experiment helpers should be imported from
# their own modules so optional R/MPI dependencies are not required just to use
# the core estimator.
from .gplsi import GpLSI
from .anchor_word import build_word_profile, recover_W_from_word_vertices
from .estimators import ExperimentalEstimate
from .preprocessing import preprocess_features
from .vertex_hunting import vertex_hunt
from .real_data import RealDataBundle, load_real_data
from .generate_topic_model import generate_data, generate_weights_edge
from .graphSVD import graphSVD
from .utils import (
    _euclidean_proj_simplex,
    get_component_mapping,
    get_F_err,
    get_l1_err,
    get_accuracy,
    moran,
    get_PAS,
)


# Preserve the original experiment entry points without loading their optional
# R/MPI dependencies when importing the estimator package.
_LAZY_EXPORTS = {
    "run_simulation_grid": "simulation",
    "SimulationConfig": "simulation",
    "run_spleen_analysis": "realdata_spleen",
    "run_crc_analysis": "realdata_crc",
    "run_cook_analysis": "realdata_cook",
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_LAZY_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = [
    "__version__",
    "GpLSI",
    "generate_data",
    "generate_weights_edge",
    "graphSVD",
    "_euclidean_proj_simplex",
    "get_component_mapping",
    "get_F_err",
    "get_l1_err",
    "get_accuracy",
    "moran",
    "get_PAS",
    "build_word_profile",
    "recover_W_from_word_vertices",
    "ExperimentalEstimate",
    "preprocess_features",
    "vertex_hunt",
    "RealDataBundle",
    "load_real_data",
]

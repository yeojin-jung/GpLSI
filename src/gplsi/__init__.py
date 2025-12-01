"""
gplsi: Graph-regularized pLSI (GpLSI) for topic modeling on data with spatial / covariate dependencies.

This package provides:
- GpLSI       : main model class
- generate_data, generate_weights_edge : simulation helpers
- graphSVD     : graph-regularized SVD 
- utility functions for aligning and evaluating topics
"""

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("gplsi")
except PackageNotFoundError:  
    __version__ = "0.0.0"


# Core model ---------------------------------------------------------------------
from .gplsi import GpLSI

# Experiments ---------------------------------------------------------------------
from .simulation import run_simulation, run_simulation_grid, SimulationConfig
from .realdata_spleen import run_spleen_analysis
from .realdata_crc import run_crc_analysis, choose_crc_ntopics
from .realdata_cook import run_cook_analysis, choose_cook_ntopics

# Simulation helpers -------------------------------------------------------------
from .generate_topic_model import generate_data, generate_weights_edge


# Graph-related SVD helpers ------------------------------------------------------
try:
    from .graphSVD import graphSVD
except ImportError:
    graphSVD = None


# Utility functions --------------------------------------------------------------
from .utils import (
    _euclidean_proj_simplex,
    get_component_mapping,
    get_F_err,
    get_l1_err,
    get_accuracy,
    moran,
    get_PAS
)


# Public API ---------------------------------------------------------------------
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
    "moran"
    "get_PAS"
]
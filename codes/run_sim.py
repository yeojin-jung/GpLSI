import sys
import os.path
import pickle

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

# !git clone https://github.com/dx-li/pycvxcluster.git
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.extend([
    os.path.join(parent_dir, "pycvxcluster"),
    parent_dir 
])

import pycvxcluster.pycvxcluster
import simulation_helpers
import numpy as np
from sklearn.model_selection import ParameterGrid
import pandas as pd

import logging

logging.captureWarnings(True)


if __name__ == "__main__":
    task_id = int(sys.argv[1])
    configs = pd.read_csv("config.txt", sep=" ")
    config = configs[configs["task_id"] == task_id]
    K = int(config["K"])
    N = int(config["N"])
    n = int(config["n"])
    nsim = int(config["nsim"])
    p = int(config["p"])
    del config
    results_dir = os.path.join(parent_dir, "output/results")
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
        except:
            pass
    msg = "Running experiment with K={}, N={}, n={}, p={}".format(K, N, n, p)
    os.system(f"echo {msg}")
    results = simulation_helpers.run_simul(
        nsim=50, N=N, n=n, K=K, p=p, start_seed=50
    )
    results_csv_loc = os.path.join(results_dir, f"results_N={N}_n={n}_K={K}_p={p}.csv")
    results.to_csv(
        results_csv_loc,
        mode="a",
        header=not os.path.exists(results_csv_loc),
        index=False,
    )
    os.system(f"echo Done with experiment {task_id}!")

param_sets = [
    {
        "nsim": [50],
        "N": [10, 30, 50, 100, 200, 1000],
        "n": [1000],
        "K": [3],
        "p": [20, 30, 50, 100, 500, 1000],
    },
    {"nsim": [50], "N": [30], "n": [100, 500, 1000], "K": [3], "p": [30]},
    {"nsim": [50], "N": [30], "n": [1000], "K": [2, 5, 7, 10], "p": [30]},
]

from sklearn.model_selection import ParameterGrid
import pandas as pd

res = pd.DataFrame()
for exper in range(3):
    grid = ParameterGrid(param_sets[exper])
    for params in grid:
        res = pd.concat([res, pd.DataFrame(params, index=[0])], axis=0)
# remove duplicates
res = res.drop_duplicates()
res["task_id"] = range(1, len(res) + 1)
res = res[["task_id", "nsim", "N", "n", "K", "p"]]
res.to_csv("config.txt", index=False, sep=" ")

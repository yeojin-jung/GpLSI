import random
import ast
import numpy as np
import pandas as pd
import networkx as nx
from numpy.linalg import norm, svd
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment


def get_shortest_paths(mst, srn):
    shortest_paths = dict(nx.shortest_path_length(mst, source=srn))
    return shortest_paths

def get_folds(mst):
    srn = np.random.choice(mst.nodes)
    path = get_shortest_paths(mst, srn)
    fold1 = [key for key, value in path.items() if value % 2 == 0]
    fold2 = [key for key, value in path.items() if value % 2 == 1]
    return srn, fold1, fold2

def get_folds_disconnected_G(edge_df, nfolds=5):
    G = nx.from_pandas_edgelist(edge_df, "src", "tgt")
    connected_subgraphs = list(nx.connected_components(G))
    folds = {i: [] for i in range(nfolds)}
    for graph in connected_subgraphs:
        G_sub = G.subgraph(graph)
        mst = nx.minimum_spanning_tree(G_sub)
        srn = np.random.choice(mst.nodes)
        path = get_shortest_paths(mst, srn)
        for node, length in path.items():
            folds[length % nfolds].append(node)
    return srn, folds, G, mst

def interpolate_X(X, G, folds, foldnum):
    fold = folds[foldnum]

    X_tilde = X.copy()
    for node in fold:
        neighs = list(G.neighbors(node))
        neighs = list(set(neighs) - set(fold))
        X_tilde[node, :] = np.mean(X[neighs, :], axis=0)
    return X_tilde

def trunc_svd(X, K):
    U, L, VT = svd(X, full_matrices=False)
    U_k = U[:, :K]
    L_k = np.diag(L[:K])
    VT_k = VT[:K, :]
    return U_k, L_k, VT_k.T

def proj_simplex(v):
    n = len(v)
    if np.sum(v) == 1 and np.all(v >= 0):
        return v
    u = np.sort(v)[::-1]
    rho = np.max(np.where(u * np.arange(1, n + 1) > (np.cumsum(u) - 1)))
    theta = (np.cumsum(u) - 1) / rho
    w = np.maximum(v - theta, 0)
    return w

def get_component_mapping(stats_1, stats_2):
    similarity = stats_1 @ stats_2.T
    cost_matrix = 1-np.abs(similarity)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    P = np.zeros_like(cost_matrix)
    P[row_ind, col_ind] = 1
    return P

def get_cosine_sim(A_1, A_2):
    K = A_1.shape[0]
    A_1_norm = A_1 / norm(A_1, axis=1, keepdims=True)
    A_2_norm = A_2 / norm(A_2, axis=1, keepdims=True)
    s = np.sum(np.diag(A_1_norm @ A_2_norm.T))
    return s/K

def get_accuracy(coords_df, n, W_hat):
    assgn = np.argmax(W_hat, axis=1)
    accuracy = np.sum(assgn == coords_df["grp"].values) / n
    return accuracy

def get_F_err(W, W_hat):
    err = norm(W.T - W_hat, ord="fro")
    return err

def get_l1_err(W, W_hat):
    err = (abs(W.T - W_hat)).sum()
    return err

def inverse_L(L):
    d = np.diagonal(L)
    non_zero = d != 0
    inv_d = np.zeros_like(d)
    inv_d[non_zero] = 1.0 / d[non_zero]
    inv = np.diag(inv_d)
    return L

def create_1NN_edge(coord_df):
    nn_model = NearestNeighbors(n_neighbors=2, algorithm="auto")
    nn_model.fit(coord_df[["x", "y"]])
    distances, indices = nn_model.kneighbors(coord_df[["x", "y"]])

    edges = []
    for i in range(len(coord_df)):
        for j in indices[i]:
            if i != j:
                edges.append({"src": i, "tgt": j, "distance": distances[i][1]})
    edge_df = pd.DataFrame(edges)
    return edge_df


def get_CHAOS(W, nodes, coord_df, n, K):
    # based on https://www.nature.com/articles/s41467-022-34879-1#citeas
    d_ij = 0
    d_all = []
    edge_df_1NN = create_1NN_edge(coord_df)
    edge_df_1NN = edge_df_1NN.assign(
        normalized_distance=np.apply_along_axis(
            norm,
            1,
            coord_df.loc[edge_df_1NN["src"], ["x", "y"]].values
            - coord_df.loc[edge_df_1NN["tgt"], ["x", "y"]].values,
        )
    )
    src_nodes = edge_df_1NN["src"]
    tgt_nodes = edge_df_1NN["tgt"]
    distances = edge_df_1NN["normalized_distance"]
    nodes = np.asarray(nodes)
    for k in range(K):
        K_nodes = nodes[np.argmax(W, axis=1) == k]
        src = np.isin(src_nodes, K_nodes)
        tgt = np.isin(tgt_nodes, K_nodes)
        d_ijk = np.sum(distances[src & tgt])
        d_all.append(distances[src & tgt])
        d_ij += d_ijk
    chaos = d_ij / n
    return 1 - chaos, d_all

def moran(W, edge_df):
    # based on https://www.paulamoraga.com/book-spatial/spatial-autocorrelation.html
    weights = edge_df["weight"]
    tpc = np.argmax(W, axis=1)
    tpc_avg = tpc - np.mean(tpc)
    n = tpc_avg.shape[0]
    edge_df["cov"] = weights * tpc_avg[edge_df["src"]] * tpc_avg[edge_df["tgt"]]
    src_grouped = edge_df.groupby("src")["cov"].sum().reset_index()
    tgt_grouped = edge_df.groupby("tgt")["cov"].sum().reset_index()
    result_df = pd.merge(
        src_grouped, tgt_grouped, left_on="src", right_on="tgt", how="outer"
    )
    val = result_df["cov_x"].fillna(0) + result_df["cov_y"].fillna(0)
    val_by_node = val.values
    m2 = np.sum(tpc_avg**2)
    I_local = n * (val_by_node / (m2 * 2))
    I = np.sum(I_local) / np.sum(weights)
    return I, I_local

def get_PAS(W, edge_df):
    topics = np.argmax(np.array(W), axis=1)
    edge_df["tpc_src"] = topics[edge_df["src"]]
    edge_df["tpc_tgt"] = topics[edge_df["tgt"]]
    src_grouped = (
        edge_df.groupby("src")
        .apply(lambda x: (x["tpc_src"] != x["tpc_tgt"]).mean())
        .rename("prop")
    )
    tgt_grouped = (
        edge_df.groupby("tgt")
        .apply(lambda x: (x["tpc_tgt"] != x["tpc_src"]).mean())
        .rename("prop")
    )
    result_df = pd.merge(
        src_grouped, tgt_grouped, left_on="src", right_on="tgt", how="outer"
    )
    val = result_df["prop_x"].fillna(0) + result_df["prop_y"].fillna(0)
    pas = (val >= 0.6).mean()
    return 1 - pas

def get_Kfolds(n, nfolds):
    indices = list(range(n))
    random.shuffle(indices)
    folds = []
    fold_size = [n // nfolds for _ in range(nfolds)]
    r = n % nfolds 

    for i in range(nfolds):
        if i < r:
            fold_size[i] += 1

    start = 0  
    for i in range(nfolds):
        end = start + fold_size[i]  
        folds.append(indices[start:end])  
        start = end  
    return folds

def tuple_converter(s):
    return ast.literal_eval(s)

def dist_to_exp_weight(df, coords, phi):
    diff = (
        coords.loc[df["src"], ["x", "y"]].values
        - coords.loc[df["tgt"], ["x", "y"]].values
    )
    w = np.exp(-phi * np.apply_along_axis(norm, 1, diff) ** 2)
    return w

def normaliza_coords(coords):
    minX = min(coords["x"])
    maxX = max(coords["x"])
    minY = min(coords["y"])
    maxY = max(coords["y"])
    diaglen = np.sqrt((minX - maxX) ** 2 + (minY - maxY) ** 2)
    coords["x"] = (coords["x"] - minX) / diaglen
    coords["y"] = (coords["y"] - minY) / diaglen

    return coords[["x", "y"]].values

def _euclidean_proj_simplex(v, s=1):
    n = v.shape[0]
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        return v
    
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    
    theta = (cssv[rho] - s) / (rho + 1.0)
    w = (v - theta).clip(min=0)
    return w
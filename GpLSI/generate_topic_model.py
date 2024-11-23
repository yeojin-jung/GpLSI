import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import csr_matrix
import networkx as nx


def get_initial_centers(val, centers):
    quantiles = []
    for i in range(centers):
        quantiles.append(i * int(val.shape[0] / centers))
    return quantiles


def align_order(k, K):
    order = np.zeros(K, dtype=int)
    order[np.where(np.arange(K) != k)[0]] = np.random.choice(
        np.arange(1, K), K - 1, replace=False
    )
    order[k] = 0
    return order


def reorder_with_noise(v, order, K, r):
    u = np.random.rand()
    if u < r:
        return v[order[np.random.choice(range(K), K, replace=False)]]
    else:
        sorted_row = np.sort(v)[::-1]
        return sorted_row[order]


def sample_MN(p, N):
    return np.random.multinomial(N, p, size=1)


def generate_graph(N, n, p, K, r, n_clusters=30):
    coords = np.zeros((n, 2))
    coords[:, 0] = np.random.uniform(0, 1, n)
    coords[:, 1] = np.random.uniform(0, 1, n)

    cluster_obj = KMeans(
        n_clusters=n_clusters, init=coords[get_initial_centers(coords, n_clusters), :], n_init=1
    )
    grps = cluster_obj.fit_predict(coords)

    coords_df = pd.DataFrame(coords, columns=["x", "y"])
    coords_df["grp"] = grps % K
    coords_df["grp_blob"] = grps
    return coords_df


def generate_W(coords_df, N, n, p, K, r):
    W = np.zeros((K, n))
    for k in range(K):
        alpha = np.random.uniform(0.1, 0.5, K)
        cluster_size = coords_df[coords_df["grp"] == k].shape[0]
        order = align_order(k, K)
        inds = coords_df["grp"] == k
        W[:, inds] = np.transpose(
            np.apply_along_axis(
                reorder_with_noise,
                1,
                np.random.dirichlet(alpha, size=cluster_size),
                order,
                K,
                r,
            )
        )

        # generate pure doc
        cano_ind = np.random.choice(np.where(inds)[0], 1)
        W[:, cano_ind] = np.eye(K)[0, :].reshape(K, 1)
    return W


def generate_W_strong(coords_df, N, n, p, K, r):
    W = np.zeros((K, n))
    for k in coords_df["grp"].unique():
        for b in coords_df[coords_df["grp"] == k]["grp_blob"].unique():
            alpha = np.random.uniform(0.1, 0.5, K)
            alpha = np.random.dirichlet(alpha)
            subset_df = coords_df[
                (coords_df["grp"] == k) & (coords_df["grp_blob"] == b)
            ]

            c = subset_df.shape[0]
            order = align_order(k, K)
            weight = reorder_with_noise(alpha, order, K, r)
            inds = (coords_df["grp"] == k) & (coords_df["grp_blob"] == b)
            W[:, inds] = np.column_stack([weight] * c) + np.abs(
                np.random.normal(scale=0.03, size=c * K).reshape((K, c))
            )

        # generate pure doc
        cano_ind = np.random.choice(np.where(inds)[0], 1)
        W[:, cano_ind] = np.eye(K)[0, :].reshape(K, 1)

    col_sums = np.sum(W, axis=0)
    W = W / col_sums
    return W


def generate_A(coords_df, N, n, p, K, r):
    A = np.random.uniform(0, 1, size=(p, K))

    # generate pure word
    cano_ind = np.random.choice(np.arange(p), K, replace=False)
    A[cano_ind, :] = np.eye(K)
    A = np.apply_along_axis(lambda x: x / np.sum(x), 0, A)
    return A


def generate_data(N, n, p, K, r, n_clusters, method="strong"):
    coords_df = generate_graph(N, n, p, K, r, n_clusters)
    if method == "strong":
        W = generate_W_strong(coords_df, N, n, p, K, r)
    else:
        W = generate_W(coords_df, N, n, p, K, r)
    A = generate_A(coords_df, N, n, p, K, r)
    D0 = np.dot(A, W)
    D = np.apply_along_axis(sample_MN, 0, D0, N).reshape(p, n)
    assert np.sum(np.apply_along_axis(np.sum, 0, D) != N) == 0
    D = D / N

    return coords_df, W, A, D.T


def generate_weights_edge(coords_df, nearest_n, phi):
    K = rbf_kernel(coords_df[["x", "y"]], gamma=phi)
    np.fill_diagonal(K, 0)
    weights = np.zeros_like(K)

    for i in range(K.shape[0]):
        top_indices = np.argpartition(K[i], -nearest_n)[-nearest_n:]
        weights[i, top_indices] = K[i, top_indices]

    weights = (weights + weights.T) / 2
    weights_csr = csr_matrix(weights)

    rows, cols = weights_csr.nonzero()
    w = weights[weights.nonzero()]
    edge_df = pd.DataFrame({"src": rows, "tgt": cols, "weight": w})
    return weights_csr, edge_df


def plot_scatter(coords_df):
    unique_groups = coords_df["grp"].unique()
    cmap = plt.get_cmap("Set3", len(unique_groups))
    colors = [cmap(i) for i in range(len(unique_groups))]

    for group, color in zip(unique_groups, colors):
        grp_data = coords_df[coords_df["grp"] == group]
        plt.scatter(grp_data["x"], grp_data["y"], label=group, color=color)


def get_colors(coords_df):
    grps = list(set(coords_df["grp"]))
    colors = []
    color_palette = ["cyan", "yellow", "greenyellow", "coral", "plum"]
    colormap = {value: color for value, color in zip(grps, color_palette[: len(grps)])}

    for value in coords_df["grp"]:
        colors.append(colormap[value])
    return colors


def plot_2d_tree(colors, G, mst):
    pos = nx.get_node_attributes(G, "pos")
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=10,
        node_color=colors,
        edge_color="gray",
        alpha=0.6,
    )
    nx.draw(
        mst,
        pos,
        with_labels=False,
        node_size=10,
        node_color=colors,
        edge_color="r",
        alpha=1,
    )
    plt.show()
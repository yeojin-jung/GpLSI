import sys
import os
import time
import pickle
import numpy as np
from numpy.linalg import norm, svd, solve, qr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import ast

from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# !git clone https://github.com/dx-li/pycvxcluster.git
sys.path.append("./pycvxcluster/src/")
import pycvxcluster.pycvxcluster

from GpLSI.utils import *
from utils.spatial_lda.featurization import make_merged_difference_matrices


def tuple_converter(s):
    return ast.literal_eval(s)

def normaliza_coords(coords):
    minX = min(coords["x"])
    maxX = max(coords["x"])
    minY = min(coords["y"])
    maxY = max(coords["y"])
    diaglen = np.sqrt((minX - maxX) ** 2 + (minY - maxY) ** 2)
    coords["x"] = (coords["x"] - minX) / diaglen
    coords["y"] = (coords["y"] - minY) / diaglen

    return coords[["x", "y"]].values


def dist_to_exp_weight(df, coords, phi):
    diff = (
        coords.loc[df["src"], ["x", "y"]].values
        - coords.loc[df["tgt"], ["x", "y"]].values
    )
    w = np.exp(-phi * np.apply_along_axis(norm, 1, diff) ** 2)
    return w


def dist_to_normalized_weight(distance):
    dist_inv = distance
    norm_dist_inv = (dist_inv - np.min(dist_inv)) / (
        np.max(dist_inv) - np.min(dist_inv)
    )
    return norm_dist_inv


def plot_topic(spatial_models, ntopics_list, fig_root, tumor, s):
    aligned_models = apply_order(spatial_models, ntopics_list)
    color_palette = sns.color_palette("husl", 10)
    colors = np.array(color_palette[:10])

    names = ["SPLSI", "PLSI", "SLDA"]
    for ntopic in ntopics_list:
        img_output = os.path.join(fig_root, tumor + "_" + str(ntopic))
        chaoss = spatial_models[ntopic][0]["chaoss"]
        morans = spatial_models[ntopic][0]["morans"]
        pas = spatial_models[ntopic][0]["pas"]
        times = spatial_models[ntopic][0]["times"]
        plt.clf()
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for j, ax in enumerate(axes):
            w = np.argmax(aligned_models[ntopic][0]["Whats"][j], axis=1)
            samp_coord_ = aligned_models[ntopic][0]["coord_df"].copy()
            samp_coord_["tpc"] = w
            ax.scatter(samp_coord_["x"], samp_coord_["y"], s=s, c=colors[w])
            name = names[j]
            ax.set_title(
                f"{name} (chaos:{np.round(chaoss[j],7)}, moran:{np.round(morans[j],2)}, pas:{np.round(pas[j],2)}, time:{np.round(times[j],2)})"
            )
        plt.tight_layout()
        plt.savefig(img_output, dpi=300, bbox_inches="tight")
        plt.close()
    return aligned_models


def plot_What(What, coord_df, ntopic):
    samp_coord_ = coord_df.copy()
    fig, axes = plt.subplots(1, ntopic, figsize=(18, 6))
    for j, ax in enumerate(axes):
        w = What[:, j]
        samp_coord_[f"w{j+1}"] = w
        sns.scatterplot(
            x="x",
            y="y",
            hue=f"w{j+1}",
            data=samp_coord_,
            palette="viridis",
            ax=ax,
            s=17,
        )
        ax.set_title(f"Original Plot {j+1}")
    plt.tight_layout()
    plt.show()
    plt.close()


def get_component_mapping(stats_1, stats_2):
    similarity = stats_1 @ stats_2.T
    cost_matrix = -similarity
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    P = np.zeros_like(cost_matrix)
    P[row_ind, col_ind] = 1
    return P


def get_component_mapping_(stats_1, stats_2):
    similarity = stats_1 @ stats_2.T
    assignment = linear_sum_assignment(-similarity)
    mapping = {k: v for k, v in zip(*assignment)}
    return mapping


def get_consistent_order(stats_1, stats_2, ntopic):
    ntopics_1 = stats_1.shape[1]
    ntopics_2 = stats_2.shape[1]
    mapping = get_component_mapping_(stats_1[:, :ntopic].T, stats_2.T)
    mapped = mapping.values()
    unmapped = set(range(ntopics_1)).difference(mapped)
    order = [mapping[k] for k in range(ntopics_2)] + list(unmapped)
    return order


def apply_order(spatial_models, ntopics_list):
    nmodels = len(spatial_models[3][0]['Whats'])
    init_topic = ntopics_list[0]

    for i in range(nmodels):
        P = get_component_mapping(
        spatial_models[init_topic][0]["Whats"][i].T,
        spatial_models[init_topic][0]["Whats"][0].T)
        W_hat = spatial_models[init_topic][0]["Whats"][i] @ P
        spatial_models[init_topic][0]["Whats"][i] = W_hat

    for ntopic1, ntopic2 in zip(ntopics_list[:-1], ntopics_list[1:]):
        # alignment within (K-1, K) initial topic
        src = spatial_models[ntopic1][0]["Whats"][0]
        tgt = spatial_models[ntopic2][0]["Whats"][0]
        P1 = get_component_mapping(tgt.T, src.T)
        P = np.zeros((ntopic2, ntopic2))
        P[:, :ntopic1] = P1
        col_ind = np.where(np.all(P == 0, axis=0))[0].tolist()
        row_ind = np.where(np.all(P == 0, axis=1))[0].tolist()
        for i, col in enumerate(col_ind):
            P[row_ind[i], col] = 1
        spatial_models[ntopic2][0]["Whats"][0] = (
            spatial_models[ntopic2][0]["Whats"][0] @ P
        )

        for i in range(nmodels):
            # alignment within each ntopic
            P = get_component_mapping(
                spatial_models[ntopic2][0]["Whats"][i].T,
                spatial_models[ntopic2][0]["Whats"][0].T)
            W_hat = spatial_models[ntopic2][0]["Whats"][i] @ P
            spatial_models[ntopic2][0]["Whats"][i] = W_hat
    return spatial_models


def align_everything(ingredient_df, file_path, ntopics, cuisine_to_group):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)

    # What
    W_plsi = model[0]['Whats'][1]
    W_gplsi = model[0]['Whats'][0]
    W_lda = model[0]['Whats'][2]

    P_lda = get_component_mapping(W_lda.T, W_gplsi.T)
    P_plsi = get_component_mapping(W_plsi.T, W_gplsi.T)
    W_lda = W_lda @ P_lda
    W_plsi = W_plsi @ P_plsi

    W_plsi = pd.DataFrame(W_plsi)
    W_plsi.columns = [f'Topic{i+1}' for i in range(ntopics)]
    W_plsi['topics'] = np.argmax(W_plsi.values, axis=1)
    W_plsi['cuisine'] = ingredient_df['cuisine']
    W_plsi['ingredients'] = ingredient_df['ingredients']

    W_lda = pd.DataFrame(W_lda)
    W_lda.columns = [f'Topic{i+1}' for i in range(ntopics)]
    W_lda['topics'] = np.argmax(W_lda.values, axis=1)
    W_lda['cuisine'] = ingredient_df['cuisine']
    W_lda['ingredients'] = ingredient_df['ingredients']

    W_gplsi = pd.DataFrame(W_gplsi)
    W_gplsi.columns = [f'Topic{i+1}' for i in range(ntopics)]
    W_gplsi['topics'] = np.argmax(W_gplsi.values, axis=1)
    W_gplsi['cuisine'] = ingredient_df['cuisine']
    W_gplsi['ingredients'] = ingredient_df['ingredients']

    # Ahat
    A_plsi = model[0]['Ahats'][1]
    A_plsi = P_plsi.T @ A_plsi.T
    A_plsi = pd.DataFrame(A_plsi)
    A_plsi.columns= ingredient_df.columns[:-3]
    A_plsi.rows = [f'Topic {k+1}' for k in range(ntopics)]

    A_lda = model[0]['Ahats'][2]
    A_lda.shape
    A_lda = P_lda.T @ A_lda
    A_lda = pd.DataFrame(A_lda)
    A_lda.columns= ingredient_df.columns[:-3]
    A_lda.rows = [f'Topic {k+1}' for k in range(ntopics)]

    A_gplsi = model[0]['Ahats'][0].T
    A_gplsi = pd.DataFrame(A_gplsi)
    A_gplsi.columns= ingredient_df.columns[:-3]
    A_gplsi.rows = [f'Topic {k+1}' for k in range(ntopics)]
    
    # create a 'group' column from 'cuisine'
    W_gplsi['group'] = W_gplsi['cuisine'].map(cuisine_to_group)
    W_plsi['group'] = W_plsi['cuisine'].map(cuisine_to_group)
    W_lda['group'] = W_lda['cuisine'].map(cuisine_to_group)

    return W_gplsi, W_plsi, W_lda, A_gplsi, A_plsi, A_lda


def plot_topic_by_cuisine(W, cuisine_order):
    pivot_table = W.drop(columns=['topics', 'ingredients', 'group']).pivot_table(index='cuisine', aggfunc='mean')
    pivot_table = pivot_table.div(pivot_table.sum(axis=1), axis=0)
    pivot_table = pivot_table.loc[cuisine_order]

    plt.figure(figsize=(2, 6))
    sns.heatmap(pivot_table, cmap="Blues",
        linewidths=0.2,
        annot=False,
        cbar=True,
        vmin=0,
        vmax=0.45)
    plt.show()


def get_top_anchor_topics(A, model_root, ntopics, method, nlargest = 10):
    # top weight words
    top_df = []
    for topic in range(ntopics):
        A_row = A.iloc[topic]
        top = A_row.nlargest(nlargest).index
        top_values = A_row.nlargest(nlargest).values
        for i in range(nlargest):
            top_df.append({
                'topic': topic+1,
                'top5': top[i],
                'top5_weights': top_values[i]
            })

    # anchor words
    anchor_df = []
    A_df_normalized = A.div(A.sum(axis=0), axis=1)
    max_word = A_df_normalized.max()
    second_max_word = A_df_normalized.apply(lambda col: col.nlargest(2).iloc[-1], axis=0) 
    for topic in range(ntopics):
        anchor_words_ = [
            word for word in A_df_normalized.columns 
            if (A_df_normalized.loc[topic, word] == max_word[word]) and 
            (max_word[word] / second_max_word[word] >= 8)
        ]
        anchor_weights = A.loc[topic, anchor_words_]
        sorted_anchor_weights = anchor_weights.sort_values(ascending=False)
        
        if len(sorted_anchor_weights) > 10:
            sorted_anchor_weights = sorted_anchor_weights.head(10)
        for i in range(len(sorted_anchor_weights)):
            anchor_df.append({
                'topic': topic+1,
                'anchor_words': sorted_anchor_weights.index[i],
                'anchor_weights': sorted_anchor_weights.values[i]
            })
    top_df = pd.DataFrame(top_df)
    anchor_df = pd.DataFrame(anchor_df)
    top_df.to_csv(os.path.join(model_root,f'top_words_{method}_{ntopics}.csv'), index=False)
    anchor_df.to_csv(os.path.join(model_root,f'anchor_words_{method}_{ntopics}.csv'), index=False)
    
    return top_df, anchor_df


def rf_classifier(W):
    X = W.drop(columns=['cuisine', 'ingredients', 'topics', 'group'])
    y = W['group']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=214)
    rf = RandomForestClassifier(random_state=214)
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, 50, None]
        }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Accuracy:", grid_search.best_score_)

    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy)
    
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    return y_pred_decoded, accuracy
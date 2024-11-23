import sys
import os
import gc
import time
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation

# !git clone https://github.com/dx-li/pycvxcluster.git
sys.path.append("./pycvxcluster/")
import pycvxcluster.pycvxcluster

import logging
logging.captureWarnings(True)

from GpLSI.utils import *
from utils.data_helpers import *
from GpLSI import gplsi

def jaccard_similarity(x, y):
    intersection = np.count_nonzero(x * y)
    n = len(x)
    return intersection / (n-intersection)

def cosine_similarity(x, y):
    dot_product = np.dot(x, y)
    return dot_product / (norm(x) * norm(y)) if norm(x) != 0 and norm(y) != 0 else 0

def sample_cuisine(group):
    if len(group) > 2000:
        return group.sample(2000, random_state=1)
    return group

def map_ingredient(ingredient, reverse_mapping):
    return reverse_mapping.get(ingredient, ingredient)

def map_cuisine(cuisine, reverse_mapping):
    return reverse_mapping.get(cuisine, cuisine)

def get_threshold(alpha, n, p, N):
    thres = alpha*np.sqrt(np.log(np.max([n,p]))/(n*N))
    return thres

def preprocess_cook(df, threshold=True):
    # Balance the number of cuisines across countries
    sampled_df = df.groupby('cuisine',group_keys=False).apply(sample_cuisine).reset_index(drop=True)

    # Reverse mapping
    reverse_mapping = {}
    for key, values in ingredient_mapping.items():
        for value in values:
            reverse_mapping[value] = key

    sampled_df['ingredients'] = sampled_df['ingredients'].apply(lambda x: [map_ingredient(ing, reverse_mapping) for ing in x])
    sampled_df['ingredients_str'] = sampled_df['ingredients'].apply(lambda x: ','.join(x))

    # Extract raw ingredients
    raw_ingred = []
    for ingredients_list in sampled_df['ingredients']:
        raw_ingred.extend(ingredients_list)

    # Drop rare ingredients
    ingredient_counts = Counter(raw_ingred)
    ingredient_vocab = [key for key in ingredient_counts.keys() if ingredient_counts[key] >= 10]

    # Create cuisine-ingredient count dataframe
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','),
                                vocabulary=ingredient_vocab)
    ingredient_matrix = vectorizer.fit_transform(sampled_df['ingredients_str'])
    ingredient_df = pd.DataFrame(ingredient_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Drop cuisines with ingredient count < 10
    ingredient_df['ingredients'] = sampled_df['ingredients_str']
    ingredient_df['cuisine'] = sampled_df['cuisine'].values
    ingredient_df = ingredient_df[ingredient_df.iloc[:,:-2].sum(axis=1)>=10]

    # Subset ingredients again to check count >= 10
    column_sums = ingredient_df.iloc[:, :-2].sum(axis=0)
    columns_to_keep = column_sums[column_sums >= 10].index
    ingredient_df = ingredient_df[columns_to_keep.tolist() + ['ingredients', 'cuisine']]
    
    # Subset words
    n, p = ingredient_df.shape
    alpha = 0.005
    N = np.max(ingredient_df.iloc[:,:-2].sum(axis=1))
    M = ingredient_df.iloc[:,:-2].mean(axis=0)
    t = get_threshold(alpha, n, p, N)
    mask = M > t

    if threshold:
         ingredient_df = ingredient_df.loc[:,mask]
    
    ingredient_df = ingredient_df.reset_index(drop=True)
    ingredient_df['node'] = ingredient_df.index

    print(f'There are {ingredient_df.shape[0]} cuisines.')
    print(f'There are {len(ingredient_df.columns)-3} ingredients in total.')

    # Create Edge dataframe
    edge_list = []
    
    ingredient_array = ingredient_df.values
    for source in range(len(ingredient_array)):
        x = ingredient_array[source]
        x_array = x[:-3]
        cuisine = x[-2]
        neigbor_list = neighbor_countries_mapping[cuisine]
        neighbor_indices = ingredient_df[ingredient_df['cuisine'].isin(neigbor_list)].index
        neighbor_array = ingredient_array[neighbor_indices, :-3]
        neighbor_nodes = ingredient_array[neighbor_indices, -1]
        similarity = np.array([jaccard_similarity(row, x_array) for row in neighbor_array])
        top5_idx = np.argpartition(similarity, -5)[-5:]
        neighbor_nodes = [neighbor_nodes[i] for i in top5_idx]
        for j, idx in enumerate(neighbor_nodes):
            src, tgt = sorted([source, idx])
            edge_list.append((src, tgt, 1))
    edge_df = pd.DataFrame(edge_list)
    edge_df = edge_df.drop_duplicates()
    edge_df.columns = ['src', 'tgt', 'weight']
    print(f'There are {edge_df.shape[0]} edges.')
        

    # Get count / weight matrix
    D = ingredient_df.iloc[:,:-3]
    row_sums = D.sum(axis=1)
    X = D.div(row_sums, axis=0)
    n, p = X.shape
    weights = csr_matrix(
        (edge_df["weight"].values, (edge_df["src"].values, edge_df["tgt"].values)),
        shape=(n, n),
    )

    return ingredient_df, D, X, edge_df, weights, n


if __name__ == "__main__":
    K = int(sys.argv[1])
    lamb_start = float(sys.argv[2])
    step_size = float(sys.argv[3])
    grid_len = int(sys.argv[4])
    eps = float(sys.argv[5])

    root_path = os.path.join(os.getcwd(), "data/whats-cooking")
    dataset_root = os.path.join(root_path, "dataset")
    model_root = os.path.join(root_path, "model_full3_jaccard")
    path_to_data = os.path.join(dataset_root, 'train.json')
    os.makedirs(model_root, exist_ok=True)
        
    with open(path_to_data) as data_file:    
        data = json.load(data_file)

    with open(os.path.join(dataset_root, 'ingredient_mapping.pkl'), 'rb') as f:
        ingredient_mapping = pickle.load(f)

    with open(os.path.join(dataset_root, 'neighbor_countries_mapping.pkl'), 'rb') as f:
        neighbor_countries_mapping = pickle.load(f)

    ingredient_df_path = os.path.join(dataset_root, 'processed_ingredient_df.pkl')
    edge_df_path = os.path.join(dataset_root, 'processed_edge_df.pkl')
    D_path = os.path.join(dataset_root, 'D_df.pkl')

    if os.path.exists(ingredient_df_path) & os.path.exists(edge_df_path):
        with open(ingredient_df_path, 'rb') as f:
            ingredient_df = pickle.load(f)

        with open(edge_df_path, 'rb') as f:
            edge_df = pickle.load(f)

        with open(D_path, 'rb') as f:
            D = pickle.load(f)

    else:
        df = pd.DataFrame(data)
        ingredient_df, D, X, edge_df, weights, n = preprocess_cook(df, threshold=False)
        del df
        gc.collect()

        with open(ingredient_df_path, 'wb') as f:
            pickle.dump(ingredient_df, f)
        
        with open(edge_df_path, 'wb') as f:
            pickle.dump(edge_df, f)
        
        with open(D_path, 'wb') as f:
            pickle.dump(D, f)

    # GpLSI
    start_time = time.time()
    model_gplsi = gplsi.GpLSI_(
        lamb_start=lamb_start, step_size=step_size, grid_len=grid_len, eps=eps
    )
    model_gplsi.fit(X.values, K, edge_df, weights)
    time_gplsi = time.time() - start_time

    # PLSI
    start_time = time.time()
    model_plsi = gplsi.GpLSI_(
        method="pLSI"
    )
    model_plsi.fit(X.values, K, edge_df, weights)
    time_plsi = time.time() - start_time

    # LDA
    start_time = time.time()
    model_lda = LatentDirichletAllocation(n_components=K, random_state=0)
    model_lda.fit(D)
    time_lda = time.time() - start_time

    save_path = os.path.join(model_root, f'cooking_model_gplsi_all_{K}.pkl')
    with open(save_path, "wb") as f:
            pickle.dump(model_gplsi, f)

    save_path = os.path.join(model_root, f'cooking_model_plsi_all_{K}.pkl')
    with open(save_path, "wb") as f:
            pickle.dump(model_plsi, f)

    save_path = os.path.join(model_root, f'cooking_model_lda_all_{K}.pkl')
    with open(save_path, "wb") as f:
            pickle.dump(model_lda, f)

    # align models
    results = []

    W_gplsi = model_gplsi.W_hat
    W_plsi = model_plsi.W_hat
    W_lda = model_lda.transform(D.values)

    # Align A_hat
    A_hat_gplsi = model_gplsi.A_hat.T
    A_hat_plsi = model_plsi.A_hat.T
    A_hat_lda = model_lda.components_ / model_lda.components_.sum(axis=1)[:, np.newaxis]

    # Plot
    names = ["GpLSI", "pLSI", "LDA"]
    times = [time_gplsi, time_plsi, time_lda]
    Whats = [W_gplsi, W_plsi, W_lda]
    Ahats = [A_hat_gplsi, A_hat_plsi, A_hat_lda]

    print(times)

    # fig, axes = plt.subplots(1,3, figsize=(18,6))
    # for j, ax in enumerate(axes):
    #    w = np.argmax(Whats[j], axis=1)
    #    samp_coord_ = coord_df.copy()
    #    samp_coord_['tpc'] = w
    #    sns.scatterplot(x='x',y='y',hue='tpc', data=samp_coord_, palette='viridis', ax=ax, s=20)
    #    name = names[j]
    #    ax.set_title(f'{name} (chaos:{np.round(chaoss[j],8)}, moran:{np.round(morans[j],3)}, time:{np.round(times[j],2)})')
    # plt.tight_layout()
    # plt.show()

    results.append(
        {
            "Whats": Whats,
            "Ahats": Ahats,
            "times": times,
            "edge_df": edge_df,
        }
    )

    save_path = os.path.join(model_root, f'cooking_model_results_{K}.pkl')
    with open(save_path, "wb") as f:
            pickle.dump(results, f)


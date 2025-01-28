import sys
import os
import gc
import time
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer

from scipy.sparse import csr_matrix
from collections import defaultdict
from sklearn.decomposition import LatentDirichletAllocation

# !git clone https://github.com/dx-li/pycvxcluster.git
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.extend([
    os.path.join(parent_dir, "pycvxcluster"),
    parent_dir 
])

import logging
logging.captureWarnings(True)

from GpLSI.utils import *
from utils.data_helpers import *
from GpLSI import gplsi

from mpi4py import MPI



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

def preprocess_cook(ingredient_mapping, neighbor_countries_mapping, df, threshold=True):
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
            #if similarity[top5_idx[j]] > 0.5:
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


def divide_folds(ingredient_df, nfolds):
    nsample = ingredient_df.shape[0]
    avg_length = nsample / float(nfolds)
    sample_idx = np.random.permutation(nsample)

    divided_folds = []
    last = 0.0

    while last < nsample:
        divided_folds.append(sample_idx[int(last):int(last + avg_length)])
        last += avg_length

    return divided_folds



def shuffle_folds(ingredient_df, edge_df, nfolds):
    data_inputs = []

    divided_folds = divide_folds(ingredient_df, nfolds)

    for i in range(nfolds):
        set = divided_folds[i]
        D_fold = ingredient_df.iloc[set, :-3]
        edge_fold = edge_df[(edge_df['src'].isin(set)) & (edge_df['tgt'].isin(set))]

        idx_dict = dict(zip(D.index, range(D.shape[0])))

        edge_fold['src'] = edge_fold['src'].map(idx_dict)
        edge_fold['tgt'] = edge_fold['tgt'].map(idx_dict)
        
        D_fold.reset_index(drop=True, inplace=True)
        row_sums = D_fold.sum(axis=1)
        X_fold = D_fold.div(row_sums, axis=0)
        n, p = X_fold.shape
        weights = csr_matrix(
            (edge_fold["weight"].values, (edge_fold["src"].values, edge_fold["tgt"].values)),
            shape=(n, n),
        )
        
        data_inputs.append((D_fold, X_fold, edge_fold, weights))
        del D_fold, X_fold, edge_fold, weights
        gc.collect()

    return data_inputs


def get_resolutions(all_Ahats, method_name):
    results = defaultdict(list)
    for i, j in combinations(range(len(all_Ahats[0])), 2):
        P = get_component_mapping(all_Ahats[i][0], all_Ahats[j][0])
        A_1 = all_Ahats[i][0]
        A_2 = all_Ahats[j][0]
        A_1 = P.T @ A_1
        K = A_1.shape[0]

        # l1 distance
        l1_dist = np.sum(np.abs(A_1 - A_2))

        # cosine similarity
        A_1_norm = A_1 / norm(A_1, axis=1, keepdims=True)
        A_2_norm = A_2 / norm(A_2, axis=1, keepdims=True)
        diag_cos = np.mean(np.diag(A_1_norm @ A_2_norm.T))

        # cosine similarity ratio
        off_cos = (np.sum(A_1_norm @ A_2_norm.T)-np.sum(np.diag(A_1_norm @ A_2_norm.T)))/(K*K - K)
        r = diag_cos/off_cos

        results['K'].append(K)
        results['l1_dist'].append(l1_dist/K)
        results['cos_sim'].append(diag_cos)
        results['cos_sim_ratio'].append(r)
        results['method'].append(method_name)
    
    return results


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    size = comm.Get_size() 
    rank = comm.Get_rank() 

    K = int(sys.argv[1])
    lamb_start = float(sys.argv[2])
    step_size = float(sys.argv[3])
    grid_len = int(sys.argv[4])
    eps = float(sys.argv[5])

    if rank == 0:
        print("Processing data...")
        nfolds = 5
        root_path = os.path.join(parent_dir, "data/whats-cooking")
        dataset_root = os.path.join(root_path, "dataset")
        model_root = os.path.join(parent_dir, "output/whats-cooking")

        filename = os.path.join(dataset_root, 'processed_edge_df.pkl')
        with open(filename, 'rb') as f:
            edge_df = pickle.load(f)

        filename = os.path.join(dataset_root, 'processed_ingredient_df.pkl')
        with open(filename, 'rb') as f:
            ingredient_df = pickle.load(f)

        data_inputs = shuffle_folds(ingredient_df, edge_df, nfolds)

        X_full = np.vstack([input[1] for input in data_inputs])
        chunks = [[] for _ in range(size)]
        for i, data in enumerate(data_inputs):
            chunks[i % size].append(data)
        del data_inputs
        gc.collect()

    else:
        chunks = None
        X_full = None

    print("Scattering inputs...")
    tasks = comm.scatter(chunks, root=0)

    # Run in each node
    start_time = time.time()
    print(f"Process {rank} calculating alignment score...")

    local_As_gplsi = []
    local_As_plsi = []
    local_As_lda = []
    for task in tasks:
        D_fold, X, edge_df, weights = task

        # run GpLSI
        model_gplsi = gplsi.GpLSI_(
                lamb_start=lamb_start, step_size=step_size, grid_len=grid_len, eps=eps
            )
        model_gplsi.fit(X.values, K, edge_df, weights)
        local_As_gplsi.append(model_gplsi.A_hat)

        # run pLSI
        model_plsi = gplsi.GpLSI_(
            method='pLSI'
        )
        model_plsi.fit(X.values, K, edge_df, weights)
        local_As_plsi.append(model_plsi.A_hat)
        
        # run LDA
        lda = LatentDirichletAllocation(n_components=K, random_state=0)
        lda.fit(D_fold.values)
        W_hat_lda = lda.transform(D_fold.values)
        A_hat_lda = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
        local_As_lda.append(A_hat_lda)

    all_Ahats_gplsi = comm.gather(local_As_gplsi, root=0)
    all_Ahats_plsi = comm.gather(local_As_plsi, root=0)
    all_Ahats_lda = comm.gather(local_As_lda, root=0)

    if rank == 0:
        results = defaultdict(list)
        results_gplsi = get_resolutions(all_Ahats_gplsi, 'gplsi')
        results_plsi = get_resolutions(all_Ahats_plsi, 'plsi')
        results_lda = get_resolutions(all_Ahats_lda, 'lda')

        for key, value in results_gplsi.items():
            results[key].extend(value)

        for key, value in results_plsi.items():
            results[key].extend(value)

        for key, value in results_lda.items():
            results[key].extend(value)

        save_path = os.path.join(model_root, 'cook_chooseK_results.csv')
        results_df = pd.DataFrame(results)
        results_df.to_csv('cook_chooseK_results.csv')
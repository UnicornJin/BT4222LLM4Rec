import os
import numpy as np
from scipy.sparse import load_npz, save_npz, csr_matrix, lil_matrix
from scipy.sparse.csgraph import shortest_path
import heapq
import torch
from scipy.spatial.distance import cdist
from scipy.sparse import lil_matrix, csr_matrix


# ------------------------------------------------
# This script is to show the data processing of 
# BT4222 LLM for Recommendation Example Code
#
# This file will combine the training, val, and test dataset.
# And form an "interaction matrix".
# 
# The script is based on project 
# 'LLM4REC' https://github.com/anord-wang/LLM4REC
#
# Edition: 2025.05.01 by Jin Yuze
# ------------------------------------------------

dataset = 'luxury'

data_root = os.path.join("dataset", dataset)

def read_and_combine_matrices(data_root):
    train_matrix = load_npz(os.path.join(data_root, 'train_matrix.npz'))
    val_matrix = load_npz(os.path.join(data_root, 'val_matrix.npz'))
    test_matrix = load_npz(os.path.join(data_root, 'test_matrix.npz'))
    combined_matrix = train_matrix + val_matrix + test_matrix
    combined_matrix[combined_matrix > 1] = 1

    return combined_matrix

def create_interaction_matrix(combined_matrix):
    num_users, num_items = combined_matrix.shape
    # total_size = num_users + num_items
    num_nodes = num_users + num_items
    # graph = np.zeros((num_nodes, num_nodes))
    graph = lil_matrix((num_nodes, num_nodes))
    if not isinstance(combined_matrix, lil_matrix):
        combined_matrix = combined_matrix.tolil()
    # fill in the user-item interaction, build graph
    graph[:num_users, num_users:] = combined_matrix
    graph[num_users:, :num_users] = combined_matrix.T

    # calculate the min distance between nodes in graph matrix, build dist_matrix
    dist_matrix = shortest_path(csgraph=graph, directed=False, method='FW')
    transformed_dist_matrix = 6 - dist_matrix
    transformed_dist_matrix = np.maximum(transformed_dist_matrix, 0)
    min_val = np.min(transformed_dist_matrix)
    max_val = np.max(transformed_dist_matrix)
    dist_matrix = (transformed_dist_matrix - min_val) / (max_val - min_val)
    print('dist_matrix', dist_matrix)
    print('dist_matrix.shape', dist_matrix.shape)

    # graph + dist_matrix + cluster_matrix + embedding_matrix
    # map to 0-1
    graph_dense = graph.toarray() if isinstance(graph, lil_matrix) else graph
    dist_matrix_dense = dist_matrix.toarray() if isinstance(dist_matrix, lil_matrix) else dist_matrix
    total_matrix = graph_dense + dist_matrix_dense
    # max min standardization
    min_val = np.min(total_matrix)
    max_val = np.max(total_matrix)
    scaled_matrix = (total_matrix - min_val) / (max_val - min_val)
    # Change the processed matrix to sparse format
    interaction_matrix = csr_matrix(scaled_matrix)

    return interaction_matrix

combined_matrix = read_and_combine_matrices(data_root)
print(combined_matrix.shape)

interaction_matrix = create_interaction_matrix(combined_matrix)
print(interaction_matrix.shape)

file_path = os.path.join(data_root, 'interaction_matrix.npz')
save_npz(file_path, interaction_matrix)

print('done')

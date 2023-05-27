import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def cosine_search(index, vector):
    v_norm = np.linalg.norm(vector)
    scores = np.zeros((index.shape[0],))
    for i in prange(index.shape[0]):
        scores[i] = np.dot(index[i], vector) / (np.linalg.norm(index[i]) * v_norm)
    return scores

data = np.load('vectorized_features.npz')
index = data['vectors']
product_ids = data['ids']

vector_index = 14
vector = index[vector_index]

scores = cosine_search(index, vector)

n_closest = np.argsort(scores)[::-1][:5]

closest_product_ids = product_ids[n_closest]

print(closest_product_ids)

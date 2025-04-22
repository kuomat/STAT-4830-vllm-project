# backend/recommendation.py

import os
import csv
import ast
import numpy as np

# 1) Load your CSV
BASE = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE, 'data', 'embeddings_final.csv')

ITEM_KEYS = []
EMB_LIST = []

with open(CSV_PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = int(row['image_key'])
        try:
            txt = ast.literal_eval(row['text_embedding'])
            img = ast.literal_eval(row['image_embedding'])
        except Exception:
            # skip rows with bad data
            continue
        ITEM_KEYS.append(key)
        EMB_LIST.append(txt + img)

# 2) Build NumPy embedding matrix of shape (n_items, dim)
EMB_MATRIX = np.array(EMB_LIST, dtype=float)

# 3) Manually compute cosine‐similarity matrix
#    sim[i,j] = (v_i · v_j) / (||v_i|| * ||v_j||)
norms = np.linalg.norm(EMB_MATRIX, axis=1)
SIM_MATRIX = EMB_MATRIX.dot(EMB_MATRIX.T) / (norms[:, None] * norms[None, :] + 1e-10)
np.fill_diagonal(SIM_MATRIX, 0)   # ignore self-similarity

def content_filtering_recommend(user_ratings: dict[int, int], k: int = 10) -> list[int]:
    """
    user_ratings: { image_key: +1 or -1 }
    k: number of recommendations
    returns: list of top-k image_keys
    """
    n = len(ITEM_KEYS)
    key_to_idx = {key: i for i, key in enumerate(ITEM_KEYS)}

    # 4) Build a flat rating vector R of length n
    R = np.zeros(n, dtype=float)
    for key, val in user_ratings.items():
        idx = key_to_idx.get(key)
        if idx is not None:
            R[idx] = val

    # 5) Predict score for every item j
    P = np.zeros(n, dtype=float)
    K = 10  # neighborhood size
    for j in range(n):
        if R[j] != 0:
            P[j] = R[j]
            continue
        sims = SIM_MATRIX[j]
        nbrs = np.argsort(sims)[::-1][:K]
        rated_mask = R[nbrs] != 0
        if not rated_mask.any():
            P[j] = 0
        else:
            num = np.dot(R[nbrs][rated_mask], sims[nbrs][rated_mask])
            den = np.sum(np.abs(sims[nbrs][rated_mask])) + 1e-10
            P[j] = num / den

    # 6) Exclude items the user already rated
    for key in user_ratings:
        idx = key_to_idx.get(key)
        if idx is not None:
            P[idx] = -np.inf

    # 7) Take top‐k
    top_idxs = np.argsort(P)[::-1][:k]
    return [ITEM_KEYS[i] for i in top_idxs]

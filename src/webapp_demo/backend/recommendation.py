# backend/recommendation.py

import os
import csv
import ast
import numpy as np
import torch
from functools import lru_cache
import torch.nn.functional as F
import torch.nn as nn

from sklearn.metrics.pairwise import cosine_similarity

# 1) Load your CSV
BASE = os.path.dirname(__file__)
CSV_PATH = os.path.abspath(os.path.join(BASE, '..', '..', '..', 'dataset', 'embeddings_final.csv'))

ITEM_KEYS = []
EMB_LIST = []
TEXT_EMB_LIST = []
IMG_EMB_LIST  = []

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
        TEXT_EMB_LIST.append(txt)
        IMG_EMB_LIST.append(img)


# 2) Build NumPy embedding matrix of shape (n_items, dim)
EMB_MATRIX = np.array(EMB_LIST, dtype=float)
FEATURES = np.array(EMB_MATRIX, dtype=np.float32) # shape: (n_items=2519, dim=1024)
TEXT_EMB_MATRIX = np.array(TEXT_EMB_LIST, dtype=np.float32)  # shape (n_items, D)
IMG_EMB_MATRIX  = np.array(IMG_EMB_LIST,  dtype=np.float32)
KEY_TO_IDX      = {k:i for i,k in enumerate(ITEM_KEYS)}

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


# --- Load and Binarize Ratings Matrix ---
# RATINGS_PATH = os.path.join(BASE, 'data', 'sparse_ratings_matrix.csv')
RATINGS_PATH = os.path.abspath(os.path.join(BASE, '..', '..', '..', 'dataset', 'sparse_ratings_matrix.csv'))
RELEVANT_ROWS = list(range(0, 1373)) + list(range(5073, 6219))

# Step 1: Load full matrix into list of lists
ratings_all = []

with open(RATINGS_PATH, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)  # skip header

    for row in reader:
        row_vals = row[1:]  # skip first column (row index)
        ratings_all.append(row_vals)

# Step 2: Filter relevant rows
ratings_filtered = [ratings_all[i] for i in RELEVANT_ROWS]

# Step 3: Convert to binary matrix
ratings_binary = []

for row in ratings_filtered:
    binary_row = []
    for val in row:
        if val.strip() == '':
            binary = 0.0  # blank = unrated
        else:
            score = float(val)
            binary = 1.0 if score >= 7 else -1.0
        binary_row.append(binary)
    ratings_binary.append(binary_row)

# Step 4: Convert to NumPy array and transpose (n_items x n_users)
RATINGS_MATRIX = np.array(ratings_binary, dtype=float)

# --- Add New User Column to Ratings Matrix ---
def augment_with_new_user(user_ratings: dict[int, int]) -> np.ndarray:
    """
    Add a new user column with +1 for selected items and np.nan for the rest.
    Returns updated_matrix: shape [n_items, n_users + 1] with nan for unrated
    """
    n_items = len(ITEM_KEYS)
    new_user_col = np.full((n_items,), np.nan)

    key_to_index = {key: i for i, key in enumerate(ITEM_KEYS)}
    for image_key, rating in user_ratings.items():
        idx = key_to_index.get(image_key)
        if idx is not None:
            new_user_col[idx] = rating  # should be 1.0

    updated_matrix = np.column_stack([RATINGS_MATRIX, new_user_col])
    return updated_matrix

# --- Collaborative Filtering Recs ---
from sklearn.metrics.pairwise import cosine_similarity

def collaborative_filtering_recommend(user_ratings: dict[int, int], k: int = 10) -> list[int]:
    """
    Recommend top-k items using user-user collaborative filtering.
    """
    # Step 1: Add the new user column
    augmented = augment_with_new_user(user_ratings)

    # Step 2: Transpose to shape [n_users + 1, n_items]
    user_item_matrix = augmented.T
    user_item_matrix = np.nan_to_num(user_item_matrix, nan=0.0)

    # Step 3: Compute cosine similarity between all users
    sim_matrix = cosine_similarity(user_item_matrix)
    np.fill_diagonal(sim_matrix, 0)  # ignore self-similarity

    # Step 4: Predict ratings for the new user
    new_user_idx = sim_matrix.shape[0] - 1
    sims = sim_matrix[new_user_idx, :-1]  # similarity to other users
    known_ratings = user_item_matrix[:-1]  # exclude new user

    # Weighted sum of other users' ratings
    weighted_scores = np.dot(sims, known_ratings)
    norm_factor = np.nansum(np.abs(sims)) + 1e-10
    predicted_scores = weighted_scores / norm_factor

    # Step 5: Exclude items the user has already rated
    key_to_idx = {k: i for i, k in enumerate(ITEM_KEYS)}
    for img_key in user_ratings:
        idx = key_to_idx.get(img_key)
        if idx is not None:
            predicted_scores[idx] = -np.inf

    # Step 6: Return top-k image_keys
    top_indices = np.argsort(predicted_scores)[-k:][::-1]
    return [ITEM_KEYS[i] for i in top_indices]

# --- Low Rank Recs ---

def train_best_projection(
    rank_list=[4, 8, 12, 16, 20, 24, 28],
    lr=0.01,
    epochs=300,
    lambda_reg=0.05
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_items, n_users = RATINGS_MATRIX.shape
    M_tensor = torch.tensor(RATINGS_MATRIX, dtype=torch.float32, device=device)

    best_rmse = float("inf")
    best_proj = None

    for rank in rank_list:
        # Step 1: Create and project item embeddings (U)
        projection = torch.nn.Linear(FEATURES.shape[1], rank).to(device)
        torch.nn.init.xavier_uniform_(projection.weight)
        projection.bias.data.fill_(0)

        with torch.no_grad():
            U = projection(torch.tensor(FEATURES, dtype=torch.float32, device=device))
        U.requires_grad = False

        # Step 2: Train user matrix (V)
        V = torch.nn.Parameter(torch.randn(n_users, rank, device=device))
        optimizer = torch.optim.Adam([V], lr=lr)

        for _ in range(epochs):
            pred = U @ V.T
            loss = torch.mean((pred - M_tensor)**2) + lambda_reg * torch.norm(V)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Step 3: Evaluate RMSE
        with torch.no_grad():
            preds = (U @ V.T).cpu().numpy()
            truth = RATINGS_MATRIX
            mask = truth != 0
            rmse = np.sqrt(np.mean((preds[mask] - truth[mask])**2))

        if rmse < best_rmse:
            best_rmse = rmse
            best_proj = projection

    return best_proj

PROJECTION = None
@lru_cache(maxsize=1)
def get_best_projection() -> torch.nn.Module:
    return train_best_projection()

def low_rank_recommend(user_ratings: dict[int, int], k=10) -> list[int]:
    global PROJECTION

    if PROJECTION is None:
        PROJECTION = get_best_projection()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = PROJECTION.out_features

    augmented_matrix = augment_with_new_user(user_ratings)
    new_user_col = augmented_matrix[:, -1]  # shape: [n_items]
    M = torch.tensor(new_user_col.reshape(-1, 1), dtype=torch.float32, device=device)

    with torch.no_grad():
        U = PROJECTION(torch.tensor(FEATURES, dtype=torch.float32, device=device))
    U.requires_grad = False

    V = torch.nn.Parameter(torch.randn(1, rank, device=device))
    optimizer = torch.optim.Adam([V], lr=0.01)
    mask = ~torch.isnan(M)

    for _ in range(300):
        pred = U @ V.T
        loss = torch.mean((pred[mask] - M[mask]) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scores = (U @ V.T).detach().cpu().numpy().flatten()

    key_to_idx = {k: i for i, k in enumerate(ITEM_KEYS)}
    for img_key in user_ratings:
        idx = key_to_idx.get(img_key)
        if idx is not None:
            scores[idx] = -np.inf

    top_indices = np.argsort(scores)[-k:][::-1]
    return [ITEM_KEYS[i] for i in top_indices]


class TwoTowerModel(nn.Module):
    def __init__(self, embedding_dim=512, hidden_dim=64):
        super(TwoTowerModel, self).__init__()

        # Projection layers to transform original embeddings
        self.user_tower = nn.Sequential(
            nn.Linear(2519, 128),
            nn.ReLU()
        )
        self.item_tower = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU()
        )

        # Map similarity score to rating
        self.rating_predictor = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, user_emb, item_emb):
        # Get embeddings from towers
        user_repr = self.user_tower(user_emb)
        item_repr = self.item_tower(item_emb)

        # Compute cosine similarity
        user_repr = F.normalize(user_repr, p=2, dim=1)
        item_repr = F.normalize(item_repr, p=2, dim=1)
        similarity = torch.sum(user_repr * item_repr, dim=1, keepdim=True)

        rating = self.rating_predictor(similarity)
        return rating.squeeze(-1)

MODEL = TwoTowerModel(embedding_dim=TEXT_EMB_MATRIX.shape[1])
MODEL.load_state_dict(torch.load(
    os.path.join(BASE, "data", "two_tower_model_params.pth"),
    map_location="cpu"
))
MODEL.eval()

def two_tower_recommend(user_ratings: dict[int,int], k: int = 10) -> list[int]:
    # 1) build a single “user vector” by averaging text‐embs of items they selected
    idxs = [KEY_TO_IDX[k] for k in user_ratings if k in KEY_TO_IDX]
    if not idxs:
        return []

    user_emb = torch.tensor([+1 if i in user_ratings else -1 for i in range(2519)]).reshape(1, 2519).type(torch.float32)
    item_imgs = torch.tensor(EMB_LIST)  # (2519, 1024)

    with torch.no_grad():
        scores = MODEL(user_emb, item_imgs)
    scores = scores.numpy()

    # 3) exclude anything the user already picked
    for key in user_ratings:
        if key in KEY_TO_IDX:
            scores[KEY_TO_IDX[key]] = -np.inf

    # 4) take top-k
    top_idxs = np.argsort(scores)[::-1][:k]
    return [ITEM_KEYS[i] for i in top_idxs]

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix


## User-user similarity matrix
def compute_user_similarity(ratings_matrix):
    similarity = cosine_similarity(ratings_matrix)
    np.fill_diagonal(similarity, 0)
    return similarity

## Fills the sparse matrix with the predicted values
def predict_user_cf(ratings_matrix, similarity_matrix, k=10):
    n_users, n_items = ratings_matrix.shape
    ratings_array = ratings_matrix.toarray()
    predicted_ratings = np.zeros((n_users, n_items))

    for user_idx in range(n_users):
        # Get top-k similar users
        user_similarities = similarity_matrix[user_idx]
        top_k_users = np.argsort(user_similarities)[::-1][:k]

        for item_idx in range(n_items):
            if ratings_array[user_idx, item_idx] > 0: ## User has already rated this item
                predicted_ratings[user_idx, item_idx] = ratings_array[user_idx, item_idx]
                continue

            sim_users_ratings = ratings_array[top_k_users, item_idx]
            sim_users_sims = user_similarities[top_k_users]

            # Filter out users who haven't rated this item
            mask = sim_users_ratings > 0
            sim_users_ratings = sim_users_ratings[mask]
            sim_users_sims = sim_users_sims[mask]

            if len(sim_users_ratings) > 0:
                predicted_ratings[user_idx, item_idx] = np.sum(sim_users_ratings * sim_users_sims) / (np.sum(sim_users_sims) + 1e-10)

    return predicted_ratings

## Precision and recall at k
def precision_recall_at_k(train_data, val_data, predictions, k=10, threshold=7):
    n_users = predictions.shape[0]
    precision_scores = []
    recall_scores = []

    for user_idx in range(n_users):
        relevant_items = set()
        for image_id, user_id, rating in val_data:
            if user_id == user_idx and rating >= threshold:
                relevant_items.add(image_id)

        # Skip users with no relevant items
        if not relevant_items:
            continue

        # Get top-k recommendations
        user_ratings = predictions[user_idx]
        for image_id, user_id, rating in train_data:
            if user_id == user_idx:
                user_ratings[image_id] = -1
        top_k_items = np.argsort(user_ratings)[::-1][:k]
        top_k_items = set(top_k_items)

        # Calculate metrics
        n_rel_and_rec = len(relevant_items.intersection(top_k_items))

        precision = n_rel_and_rec / k if k != 0 else 0
        recall = n_rel_and_rec / len(relevant_items) if len(relevant_items) != 0 else 0

        precision_scores.append(precision)
        recall_scores.append(recall)

    avg_precision = np.mean(precision_scores) if precision_scores else 0
    avg_recall = np.mean(recall_scores) if recall_scores else 0

    return avg_precision, avg_recall


# Evaluate the model using train/val split
def evaluate_cf_model(train_data, val_data, n_users, n_items, k=10):
    ## Create training matrix
    train_matrix = np.zeros((n_items, n_users))
    for image_id, user_id, rating in train_data:
        train_matrix[image_id, user_id] = rating

    train_matrix_sparse = csr_matrix(train_matrix)

    ## Compute user similarity matrix and perform prediction
    similarity_matrix = compute_user_similarity(train_matrix_sparse.T)
    predictions = predict_user_cf(train_matrix_sparse.T, similarity_matrix, k=k) ## Fill the sparse matrix with the predicted values

    ## Recall and precision @ k
    precision, recall = precision_recall_at_k(train_data, val_data, predictions, k=k, threshold=7)

    ## Calculate the error
    val_predictions = []
    val_true_ratings = []

    for image_id, user_id, rating in val_data:
        predicted = predictions[user_id, image_id]
        if predicted > 0:
            val_predictions.append(predicted)
            val_true_ratings.append(rating)

    rmse = math.sqrt(mean_squared_error(val_true_ratings, val_predictions))
    mae = np.mean(np.abs(np.array(val_predictions) - np.array(val_true_ratings)))
    return rmse, mae, precision, recall

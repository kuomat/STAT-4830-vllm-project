import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from sklearn.metrics import mean_squared_error
import numpy as np

## Helper method to prepare data for the two-tower model
def _prepare_data(data, item_embeddings_df, user_embeddings_df):
    item_embeddings = torch.tensor(item_embeddings_df['combined_embedding'].tolist()).float()
    user_embeddings = torch.tensor(user_embeddings_df['combined_embeddings'].tolist()).float()

    user_embs = []
    item_embs = []
    ratings = []

    for image_id, user_id, rating in data:
        user_embs.append(user_embeddings[user_id])
        item_embs.append(item_embeddings[image_id])
        ratings.append(rating)

    return torch.stack(user_embs), torch.stack(item_embs), torch.tensor(ratings).float()

## Prepare training and validation data for the two-tower model
def prepare_two_tower_data(train_data, val_data, item_embeddings_df, user_embeddings_df):
    """
    Prepare training and validation data for the two-tower model
    """
    train_user_embs, train_item_embs, train_ratings = _prepare_data(train_data, item_embeddings_df, user_embeddings_df)
    val_user_embs, val_item_embs, val_ratings = _prepare_data(val_data, item_embeddings_df, user_embeddings_df)

    return (train_user_embs, train_item_embs, train_ratings), (val_user_embs, val_item_embs, val_ratings)


## Training loop for the two-tower model
def train_two_tower_model(model, train_data, val_data, num_epochs=10, batch_size=32):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_user_embs, train_item_embs, train_ratings = train_data
    val_user_embs, val_item_embs, val_ratings = val_data

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Training
        for i in range(0, len(train_ratings), batch_size):
            batch_user_embs = train_user_embs[i:i+batch_size]
            batch_item_embs = train_item_embs[i:i+batch_size]
            batch_ratings = train_ratings[i:i+batch_size]

            optimizer.zero_grad()
            predictions = model(batch_user_embs, batch_item_embs)
            loss = criterion(predictions, batch_ratings)  # No scaling needed
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / (len(train_ratings) / batch_size)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(val_user_embs, val_item_embs)
            val_loss = criterion(val_predictions, val_ratings)
            val_losses.append(val_loss.item())

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {val_loss.item():.4f}')

    return train_losses, val_losses


## Evaluate the two-tower model
def evaluate_two_tower_model(model, val_data_tensors, val_data_original, embeddings_df, users_embeddings_df):
    model.eval()
    val_user_embs, val_item_embs, val_ratings = val_data_tensors

    with torch.no_grad():
        predictions = model(val_user_embs, val_item_embs)

        # Calculate RMSE and MAE directly (no need to scale)
        rmse = math.sqrt(mean_squared_error(val_ratings.numpy(), predictions.numpy()))
        mae = np.mean(np.abs(predictions.numpy() - val_ratings.numpy()))

        # Calculate Precision and Recall@k
        precision, recall = _precision_recall_at_k_two_tower(
            model, embeddings_df, users_embeddings_df, val_data_original, k=10, threshold=7)

    return rmse, mae, precision, recall

## Precision and recall at k for the two-tower model
def _precision_recall_at_k_two_tower(model, embeddings_df, users_embeddings_df, val_data_original, k=10, threshold=7):
    model.eval()
    item_embeddings = torch.tensor(embeddings_df['combined_embedding'].tolist()).float()
    user_embeddings = torch.tensor(users_embeddings_df['combined_embeddings'].tolist()).float()

    precision_scores = []
    recall_scores = []

    # Group validation data by user
    user_relevant_items = {}
    for image_id, user_id, rating in val_data_original:
        if rating >= threshold:
            if user_id not in user_relevant_items:
                user_relevant_items[user_id] = set()
            user_relevant_items[user_id].add(image_id)

    with torch.no_grad():
        for user_id, relevant_items in user_relevant_items.items():
            if not relevant_items:
                continue

            # Get predictions for all items for this user
            user_emb = user_embeddings[user_id].unsqueeze(0).repeat(len(item_embeddings), 1)
            predictions = model(user_emb, item_embeddings)

            # Get top k predictions
            top_k_indices = torch.argsort(predictions, descending=True)[:k]
            recommended_items = set(top_k_indices.cpu().numpy())

            # Calculate metrics
            n_rel_and_rec = len(relevant_items.intersection(recommended_items))
            precision = n_rel_and_rec / k
            recall = n_rel_and_rec / len(relevant_items)

            precision_scores.append(precision)
            recall_scores.append(recall)

    avg_precision = np.mean(precision_scores) if precision_scores else 0
    avg_recall = np.mean(recall_scores) if recall_scores else 0

    return avg_precision, avg_recall
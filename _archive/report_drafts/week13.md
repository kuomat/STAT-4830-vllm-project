# Week 13 Report
---

## Problem Statement

In this project, we aim to optimize a recommendation system that provides personalized item suggestions based on a user’s limited shopping history from other platforms. Our approach combines four methods—content-based filtering, collaborative filtering, low-rank matrix completion, and a two-tower neural network model—to address cold-start challenges and improve recommendation quality.

Cold-start issues in recommendation systems lead to poor user experience, making it difficult for users to receive relevant suggestions when they switch to a new platform. This can cause frustration, reduce engagement, and limit conversions for businesses.

### Success Metrics
- **Recommendation Relevance**: Measured through Precision@k, Recall@k, and NDCG to evaluate if suggested items align with user preferences.
- **User Engagement**: Click-through rate (CTR) is tracked to understand user interest in the recommendations.
- **Cold-Start Performance**: Effectiveness is specifically measured for new users with minimal history using hold-out validation sets.

### Constraints
- **Data Availability**: Metadata from different platforms often varies in format and quality. For example, some sites may provide only basic descriptions without images, limiting feature extraction.
- **Real-Time Performance**: The system must generate recommendations quickly, as users expect results instantly. This requires fast inference from embeddings and low-latency similarity computations.
- **User Privacy**: Cross-platform data usage requires compliance with privacy laws (e.g., GDPR), ensuring that no personally identifiable information is exposed.

---

## Data Preparation

We utilized three key datasets:
1. **Images and Captions**: ~3000 clothing items (1500 from Myntra, 1500 from ASOS) with structured metadata including descriptions, categories, and prices.
2. **Synthetic Personas**: 30 synthetic user profiles with unique style preferences, generated using ChatGPT with detailed bios to simulate real-world diversity.
3. **Ratings Matrix**: A sparse (80% missing) user-item matrix that represents what each persona thinks of each item, generated via large language models using prompt engineering techniques.

To augment our data, we downloaded image datasets using URLs directly instead of scraping HTML. We normalized image metadata fields and built a Python pipeline to handle malformed inputs, such as JSON-like strings or comma-separated lists. All image data was cleaned and linked with corresponding metadata for use in embedding models.

---

## Technical Approach

### Model Overview

We designed and compared the following models:

| Model                   | Metadata Used | Ratings Used | Scalability | Recommendation Type           |
|------------------------|---------------|--------------|-------------|--------------------------------|
| Content-Based Filtering| ✅             | ❌            | Easy        | Personalized by item content   |
| Collaborative Filtering| ❌             | ✅            | Moderate    | Preference-based               |
| Low-Rank Matrix Factorization | Optional | ✅         | Moderate    | Interpolative and hybrid       |
| Two-Tower Neural Model | ✅             | ✅            | High        | Hybrid embeddings              |

---

## Collaborative Filtering

### Approach

Collaborative Filtering predicts a user’s preference for an item based on the preferences of similar users (User-based CF), similar items (Item-based CF), or through a neural network (Neural CF).

### Mathematical Formulations

1. **User-Based CF**:
\[
\hat{r}_{ui} = \frac{\sum_{v \in N_k(u)} \text{sim}(u,v) \cdot r_{vi}}{\sum_{v \in N_k(u)} \text{sim}(u,v)}
\]

2. **Item-Based CF**:
\[
\hat{r}_{ui} = \frac{\sum_{j \in N_k(i)} \text{sim}(i,j) \cdot r_{uj}}{\sum_{j \in N_k(i)} \text{sim}(i,j)}
\]

3. **Neural CF**:
\[
\hat{r}_{ui} = f(W_2 \cdot ReLU(W_1 \cdot [e_u; e_i] + b_1) + b_2)
\]

### Implementation

- Cosine similarity computed using `sklearn`.
- Neural CF implemented in PyTorch with MSE loss.
- Used batched sparse matrix operations for scalability.

### Validation

- Offline metrics: RMSE, MAE, Precision@10.
- Cold-start tested using hold-out users.
- 5-fold cross-validation used for robustness.

---

## Content-Based Filtering

### Approach

Based on CLIP embeddings of item text and image features. Users are represented by the average of embeddings of items they like.

### Math

\[
\vec{v_u} = \frac{1}{|L_u|} \sum_{j \in L_u} \vec{s_j} \quad \text{and} \quad score(j) = \cos(\vec{v_u}, \vec{s_j})
\]

### Implementation

- Preprocess and tokenize text.
- Feed images and text through CLIP.
- Compute cosine similarity to user profile vector.
- Return top-k highest scoring items.

### Validation

- Metrics: Precision@5, NDCG@5, MAP.
- Tested on cold-start users.
- Visual validation of top results.

---

## Low-Rank Matrix Completion

### Formulation

\[
\hat{R} = U V^T \quad \text{where} \quad \min_{U,V} \sum_{(i,j) \in \Omega} (R_{ij} - (UV^T)_{ij})^2 + \lambda (\|U\|_F^2 + \|V\|_F^2)
\]

### Implementation

- PyTorch `nn.Embedding` layers for U, V.
- Adam optimizer and masked MSE loss.
- Hyperparameter grid search over rank and regularization.

### Validation

- Elbow method for best rank selection.
- Offline metrics: RMSE, Precision@10.

---

## Two-Tower Model

### Overview

Uses two neural networks to independently map user and item embeddings into a shared space.

\[
Z_u = \frac{\sum_{i \in R_u} r_{ui} Z_i}{\sum_{i \in R_u} r_{ui}}, \quad S(\tilde{Z}_u, \tilde{Z}_i) = \frac{\tilde{Z}_u \cdot \tilde{Z}_i}{\|\tilde{Z}_u\| \cdot \|\tilde{Z}_i\|}
\]

### Implementation

- CLIP embeddings for both items and user profiles.
- Feedforward networks as towers.
- Cosine similarity used in final scoring.
- Trained using MSE loss on known (user, item) relevance.

---

## Persona Generation

Using ChatGPT and Mistral 7B, we defined personas based on traits like age, profession, and style. Prompts were crafted to ensure diverse preferences. For each persona, we simulated ratings using LLM-generated scores on product descriptions and CLIP embeddings.

This allowed us to:
- Model diverse tastes without real data
- Inject stylistic variance into cold-start tests
- Evaluate how models behave across unique profiles

---

### Visual Results

- CLIP-based t-SNE plots showed strong clustering by style.
- Personalized recommendations appeared aligned with persona profiles (e.g., minimalists vs. trendsetters).

---

## Limitations

- **Synthetic Bias**: LLM-generated ratings may not fully reflect real preferences.
- **Image Quality**: CLIP sensitivity to image style/format caused semantic drift.
- **Cold-Start**: CF models underperform without shared preferences.

---

## Future Work

- Replace synthetic data with real feedback (clicks, purchases).
- Incorporate more efficient image pipelines (multi-threaded).
- Explore ANN-based retrieval for real-time scalability.
- Expand to include reinforcement learning or transformer-based recommenders.

---

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

### Required Data
- **User interactions** from shopping sites (order history, wish lists, browsing activity).
- **Product metadata** (titles, descriptions, images, categories, prices, brands) across multiple websites.
- **User-generated content** (ratings, reviews, preferences) from different platforms.

### Potential Pitfalls
- **Sparse shopping history from other sites**: Users may have limited or highly specific purchase patterns from other sites, making it difficult to generate diverse recommendations.
- **Metadata Mismatch**: Variability in product data formats (e.g., different naming conventions or image resolutions) can hinder model training.
- **Scalability**: Handling millions of items and users requires efficient retrieval methods, such as ANN (Approximate Nearest Neighbors) for large-scale searches.
- **Privacy Concerns**: Tracking user activity across websites must comply with data protection regulations.

---

## Technical Approach

### Collaborative Filtering

#### Mathematical Formulation
##### Objective Function:
For collaborative filtering, we aim to predict missing user-item interactions through three main approaches:

1. **User-Based CF:**

$\hat{r}_{ui}$ = $\frac{\sum_{v \in N_k(u)} sim(u,v) \cdot r_{vi}}{\sum_{v \in N_k(u)} sim(u,v)}$

where $\hat{r}_{ui}$ is the predicted rating for user u on item i, $N_k(u)$ is the set of k most similar users to u, and sim(u,v) is the cosine similarity between users.

2. **Item-Based CF:**
$$\hat{r}_{ui} = \frac{\sum_{j \in N_k(i)} sim(i,j) \cdot r_{uj}}{\sum_{j \in N_k(i)} sim(i,j)}$$

where $N_k(i)$ is the set of k most similar items to i.

3. **Neural CF:**

$$\hat{r}_{ui} = f(W_2 \cdot ReLU(W_1 \cdot [e_u; e_i] + b_1) + b_2)$$

where $e_u$ and $e_i$ are user and item embeddings, and $W_1$, $W_2$, $b_1$, $b_2$ are learned parameters.


##### Constraints:
1. **Cold-Start:** Limited effectiveness for new users/items
2. **Scalability:** Computation grows with user/item count

. . . . . . . . . . . . . . . . . . . .

#### Algorithm/Approach Choice and Justification
We implemented three complementary collaborative filtering approaches:

1. **User-Based CF:**
- Leverages user similarity patterns
- Effective for users with overlapping preferences
- Quick to adapt to new user preferences

2. **Item-Based CF:**
- More stable than user-based approach
- Better handles the user cold-start problem
- More computationally efficient for many systems

3. **Neural CF:**
- Captures non-linear user-item interactions
- Learns latent features automatically
- Better handles sparsity through embedding learning

##### Justification:
- Multiple approaches provide robustness
- Each method compensates for others' weaknesses
- Neural CF adds non-linear modeling capability

. . . . . . . . . . . . . . . . . . . .

#### PyTorch Implementation Strategy

1. **Extract Vectors:** Build user-item interaction matrix from dataset and convert to tensor format.
2. **Compute Similarities:** Use cosine similarity (`sklearn.metrics.pairwise.cosine_similarity`) to measure closeness between users (for user-based CF) or items (for item-based CF).
3. **Select Top-N Neighbors:** For each target user or item, select the top-k most similar users/items using torch topk operations.
4. **Score Prediction:** Predict ratings or scores for unseen items by taking a weighted average of ratings from neighbors.
5. **Recommendation:** Sort predicted scores and recommend the top-ranked items to the user.

For efficiency, batch operations were implemented using PyTorch tensors. Cosine similarities and predictions were computed on the fly using sparse matrix multiplication.

. . . . . . . . . . . . . . . . . . . .

#### Validation Methods
1. **Offline Evaluation:**
- **Metrics:** RMSE, MAE, Precision@K, Recall@K
- **Cross-validation:** 5-fold cross-validation
- **Cold-start Testing:** Hold-out new users/items

2. **Online Testing:**
- A/B testing different approaches
- User engagement metrics
- Click-through rates

3. **Comparative Analysis:**
- Compare performance across all three approaches
- Analyze strengths/weaknesses for different user segments

. . . . . . . . . . . . . . . . . . . .

#### Resource Requirements and Constraints
1. **Computational Resources:**
- Memory: $O(|U| \times |I|)$ for similarity matrices
- CPU: Significant for large-scale similarity computations
- GPU: Required for efficient Neural CF training

2. **Storage Requirements:**
- User-item interaction matrix
- Similarity matrices
- Model parameters

3. **Scalability Considerations:**
- User-based CF: $O(|U|^2)$ similarity computations
- Item-based CF: $O(|I|^2)$ similarity computations
- Neural CF: $O$(batch_size × embedding_dim)

4. **Performance Constraints:**
- Real-time recommendation latency
- Batch update frequency
- Memory limitations for large datasets

The collaborative filtering implementation provides complementary recommendations to our content-based system. While content-based filtering leverages item features, collaborative filtering captures user behavior patterns and preferences. The combination of both approaches in a hybrid system offers more robust and accurate recommendations.

---

### Content-Based Filtering

#### Mathematical Formulation
##### Objective Function:
The objective is to recommend items that maximize the similarity between a user’s preference vector and item feature vectors (text and image embeddings).

We define the user preference vector $\vec{v_u}$ as the average of embeddings for all liked items $L_u$. Each item $j$ is then scored using cosine similarity:

$$
\vec{v_u} = \frac{1}{|L_u|} \sum_{j \in L_u} \vec{s_j} \quad \text{and} \quad score(j) = \cos(\vec{v_u}, \vec{s_j})
$$

##### Constraints:
1. **Cold-Start Handling:** Users have no historical ratings, so only item content embeddings (text, image) are used.
2. **Diversity Constraint:** Optional constraint to limit similar items in recommendations.
3. **Resource Constraint:** Limited memory and computational resources.

. . . . . . . . . . . . . . . . . . . .

#### Algorithm/Approach Choice and Justification
##### Approach: Content-Based Filtering with CLIP Embeddings
We use CLIP (Contrastive Language-Image Pretraining) to encode both item text descriptions and images into a shared feature space. By relying on item content rather than user interaction history, we address the cold-start problem effectively.

##### Justification:
- **Rich Representations:** CLIP embeddings capture semantic meaning from both text and images.
- **No User History Required:** Suitable for cold-start scenarios.
- **Efficient Similarity Computation:** Cosine similarity is fast and efficient.

. . . . . . . . . . . . . . . . . . . .

#### PyTorch Implementation Strategy
1. **Model and Preprocessing**:
- Use a pre-trained CLIP model from `openai/clip`.
- Tokenize item descriptions and preprocess images for model input.

2. **Embedding Extraction:**
- **Text Embeddings:** `model.encode_text()` with tokenized text.
- **Image Embeddings:** `model.encode_image()` with preprocessed images.

3. **Vector Combination:**
- Concatenate image and text embeddings into a single vector for each item.

4. **Similarity Computation:**
- Compute cosine similarity between user preference vectors and all item vectors using `cosine_similarity()` from `sklearn`.

5. **Recommendation:**
- Rank items by similarity scores.
- Exclude items the user has already interacted with.

. . . . . . . . . . . . . . . . . . . .

#### Validation Methods
1. **Offline Evaluation:**
- **Metrics:** Precision, Recall, F1-score, Mean Average Precision (MAP), and NDCG.
- **Cross-validation:** Use k-fold cross-validation on synthetic preference sets.

2. **Cold-Start Scenario Testing:**
- Test with users having no prior interactions.
- Measure performance for different user personas (e.g., Vivian vs. Megan).

3. **Qualitative Evaluation:**
- Visual inspection of recommendations.
- User feedback sessions.

. . . . . . . . . . . . . . . . . . . .

#### Resource Requirements and Constraints
1. **Hardware:**
- **GPU:** Required for efficient PyTorch and CLIP model inference.
- **Memory:** Minimum 16GB RAM for large embeddings.

2. **Software:**
- Python 3.10+
- PyTorch with CUDA support
- `openai/clip`, `pandas`, `numpy`, `sklearn`

3. **Time Constraints:**
- Embedding extraction may be slow for large datasets (batch processing recommended).
- Recommendation retrieval is fast (O(N) with N items).

4. **Scalability Constraints:**
- Limited by GPU memory for large batches.
- Possible optimization: Approximate Nearest Neighbors (ANN) for large-scale search.

---

### Low-Rank Matrix Completion

#### Mathematical Formulation

Given a user-item rating matrix $R \in \mathbb{R}^{m \times n}$, we aim to find two low-rank matrices $U \in \mathbb{R}^{m \times k}$ and $V \in \mathbb{R}^{n \times k}$ such that:

$$
\hat{R} = U V^T
$$

Our optimization objective (with regularization) is:

$$
\min_{U,V} \sum_{(i,j)\in \Omega}(R_{ij} - (UV^T)_{ij})^2 + \lambda(\|U\|_F^2 + \|V\|_F^2)
$$

Where:
- $\Omega$ is the set of observed ratings
- $\lambda$ is a regularization hyperparameter to prevent overfitting
- $\|\cdot\|_F$ is the Frobenius norm

##### Constraints
- **Rank Constraint**: Matrices $U, V$ have dimension $k \ll m, n$
- **Cold-Start**: Performance drops for unseen users/items
- **Data Sparsity**: Requires regularization and robust loss functions

. . . . . . . . . . . . . . . . . . . .

#### Algorithm/Approach Choice and Justification

We apply **low-rank matrix factorization using the Burer–Monteiro parameterization**. Our approach includes:

- A grid search to select the best rank $k$ that minimizes validation loss.
- Use of a projection layer to initialize $U$ from content embeddings.
- Training via gradient descent with regularization.

##### Advantages:
- Memory efficient (only $(m + n)\times k$ parameters)
- Flexible: easily incorporates side information (e.g., text/image features)
- Compatible with binary or real-valued ratings

. . . . . . . . . . . . . . . . . . . .

#### PyTorch Implementation Strategy

1. **Rating Matrix Construction:**
   - Use a user-item matrix with NaNs for missing entries and mask known entries.

2. **Grid Search & Rank Selection:**
   - Try multiple values of $k$ using a fixed train/test split to find optimal rank.

3. **Projection Initialization:**
   - Use item-level features (text + image embeddings + price) and project to $k$-dim space.

4. **Training Loop:**
   - Define $U, V$ as PyTorch parameters
   - Compute predictions as $\hat{R} = UV^T$
   - Compute masked MSE loss + L2 regularization
   - Optimize with Adam

5. **Prediction and Recommendation:**
   - Compute $\hat{R}$ after training
   - For each user, recommend highest predicted unrated items

. . . . . . . . . . . . . . . . . . . .

#### Validation Methods

- **Elbow Method:** Grid search ranks and plot validation loss to find optimal $k$
- **Offline Metrics:** RMSE and MSE on held-out test ratings
- **Qualitative Evaluation:** Top-N recommendations visualized for users
- **Binary Mode Testing:** Converted real-valued matrix into $\{-1, 1\}$ labels to simulate preference learning

. . . . . . . . . . . . . . . . . . . .

#### Resource Requirements and Constraints

- **Hardware:**
  - GPU strongly recommended for efficient training
  - Memory scales with $(m + n)\times k$

- **Software:**
  - Python, PyTorch, NumPy, pandas
  - matplotlib for visualization

- **Performance Considerations:**
  - Batch gradient updates for fast convergence
  - Masked loss computation avoids affecting optimization with missing values


The low-rank model successfully filled in missing user-item ratings and generated personalized recommendations. The elbow method confirmed that rank = 7 yielded optimal validation performance, and final loss values dropped from ~40 to <1.5 in many trials.

---

### Two-Tower Model

#### Mathematical Formulation

The Two-Tower model learns separate representations for users and items and compares them using a similarity function. Each user is represented as a weighted combination of the embeddings of items they have interacted with:

$$
Z_u = \frac{\sum_{i \in R_u} r_{ui} \cdot Z_i}{\sum_{i \in R_u} r_{ui}}
$$

Then, the user and item embeddings are passed through separate neural networks (towers):

$$
\tilde{Z}_u = f_U(Z_u; \theta_U), \quad \tilde{Z}_i = f_I(Z_i; \theta_I)
$$

The similarity between the transformed user and item embeddings is computed as cosine similarity:

$$
S(\tilde{Z}_u, \tilde{Z}_i) = \frac{\tilde{Z}_u \cdot \tilde{Z}_i}{\|\tilde{Z}_u\| \cdot \|\tilde{Z}_i\|}
$$

The model is trained to minimize the Mean Squared Error (MSE) between predicted similarity and ground-truth labels:

$$
\mathcal{L} = \frac{1}{N} \sum_{(u,i)} (S(\tilde{Z}_u, \tilde{Z}_i) - y_{ui})^2
$$

. . . . . . . . . . . . . . . . . . . .

#### Algorithm/Approach Choice and Justification

Instead of learning one large joint embedding space for user-item pairs, the Two-Tower model learns:
- A **user tower**: maps user embedding into latent space
- An **item tower**: maps item embedding into latent space

##### Advantages:
- Enables **precomputing item embeddings** for efficient retrieval
- Effective for **multimodal inputs** (text, images)
- Scalable for large candidate sets

It also supports **online retrieval** by indexing the item tower outputs and comparing with real-time user tower outputs.

. . . . . . . . . . . . . . . . . . . .

#### PyTorch Implementation Strategy

1. **Initial Embeddings:**
   - Extract CLIP embeddings for each item
   - Represent users as weighted sums of embeddings of liked items

2. **Model Architecture:**
   - Build two small feedforward networks: one for user tower, one for item tower

3. **Forward Pass:**
   - Pass user and item embeddings through respective towers
   - Normalize outputs and compute cosine similarity

4. **Loss & Optimization:**
   - MSE loss between predicted similarity and binary label (1 if interacted, 0 if not)
   - Optimize with Adam

5. **Inference:**
   - Precompute $\tilde{Z}_i$ for all items
   - For each user, compute $\tilde{Z}_u$ and recommend most similar items

. . . . . . . . . . . . . . . . . . . .

#### Validation Methods

To evaluate the model's performance, precision-based metrics are used. Precision at K (Precision@K) measures the proportion of relevant items within the top K recommendations provided to a user. It is computed by checking how many of the recommended items were actually interacted with by the user.

Mean Average Precision at K (MAP@K) extends this concept by incorporating ranking order. It calculates the average precision for each user and then computes the mean across all users. This metric is useful for ensuring that highly relevant items appear earlier in the recommendation list.

The qualitative validation of recommendations is also performed by visually inspecting the results. The first image shows personalized recommendations for multiple users, displaying the top five suggested items along with their images and descriptions. The recommendations appear diverse and relevant to user preferences, suggesting that the model effectively learns item relationships.

. . . . . . . . . . . . . . . . . . . .

#### Resource Requirements and Constraints

Given the relatively small dataset used in this implementation, computational constraints are minimal. The primary resource requirement is GPU acceleration for training efficiency, as encoding both text and images using CLIP can be computationally expensive. Since item embeddings can be precomputed, inference time is reduced significantly, making the system scalable for larger datasets.

Memory requirements increase with dataset size, especially when storing high-dimensional embeddings. If the dataset were to grow, approximate nearest neighbor search techniques could be incorporated to improve retrieval speed without significantly increasing computational cost.

While the current implementation is effective for smaller-scale testing, a real-world deployment would require additional optimizations such as efficient indexing, batch processing, and caching of frequently accessed embeddings.

. . . . . . . . . . . . . . . . . . . .

#### Conclusion

The Two-Tower model enables fast, scalable recommendations by decoupling user and item encoding. The model effectively learns from multimodal features using CLIP and delivers diverse, personalized suggestions through efficient similarity-based ranking.

The final loss converges quickly, and visual analysis confirms the formation of meaningful clusters in embedding space. Future work may include expanding to multi-head attention towers or integrating reinforcement signals to dynamically update user preferences.

---


## Results

### Evidence Your Implementation Works

We successfully implemented and tested four different approaches:

- **Collaborative Filtering**: Leveraged user-item interaction matrices to recommend items using cosine similarity. Produced personalized recommendations for users like Laura and Matt.

- **Content-Based Filtering**: Used CLIP embeddings to generate recommendations based on text and image features of items. Users such as Vivian received relevant suggestions aligned with past preferences.

- **Low-Rank Matrix Completion**: Completed sparse rating matrices by learning user and item factors. Verified predictions aligned with test ratings and improved over training epochs.

- **Two-Tower Model**: Generated user-item recommendations using dual neural networks and cosine similarity. Verified with MSE loss minimization and meaningful item cluster visualizations.

All models were tested with synthetic and real-world scenarios and produced distinct, reasonable recommendations per user.

. . . . . . . . . . . . . . . . . . . .

### Basic Performance Metrics

- **Collaborative Filtering**  
  - Precision@5: **0.64**  
  - Recall@5: **0.59**  
  - RMSE (rating prediction): **6.12**

- **Content-Based Filtering**  
  - Mean Average Precision (MAP): **0.86**  
  - NDCG@5: **0.84**  
  - Precision@5 (cold-start users): **0.72**

- **Low-Rank Matrix Completion**  
  - Validation RMSE (rank = 7): **5.46**  
  - Final training loss (masked MSE): **< 1.5**  
  - Precision@5: **0.60**

- **Two-Tower Model**  
  - Final MSE loss: **0.021**  
  - Precision@5: **0.78**  
  - NDCG@5: **0.76**  
  - t-SNE plot confirms embedding separation 

. . . . . . . . . . . . . . . . . . . .

### Test Case Results

- **Collaborative Filtering**: Accurately surfaced items across brands for users with similar interaction histories. For example, Matthew and Vivian received overlapping suggestions like black dresses and relaxed sweaters, while Laura’s results skewed toward Uniqlo basics and fitted jeans, demonstrating the model's ability to detect subtle co-preference signals.

- **Content-Based Filtering**: Returned highly style-consistent outfits. For instance, Megan was recommended a mix of fitted denim, flowy dresses, and neutral tones aligned with her past items. CLIP embeddings successfully captured both textual and visual themes, even for users with minimal prior data.

- **Low-Rank Matrix Completion**: For all users, the model inferred preferences that balanced novelty and familiarity. Laura and Megan were shown sophisticated, minimal clothing pieces in line with their sparse but focused history. Matt received both new Uniqlo picks and previously unseen brands, verifying that matrix completion bridged sparse gaps.

- **Two-Tower Model**: Produced the most coherent style clustering per user. Vivian received clean, muted tones and skirts from Uniqlo and Lewkin, while Laura was served fitted sweaters and statement dresses. The Two-Tower model captured high-level semantic patterns and generalized well across users and item categories.

. . . . . . . . . . . . . . . . . . . .

### Current Limitations

- **Collaborative Filtering**: Still underperforms for users with zero or near-zero historical ratings.

- **Content-Based Filtering**: Limited by the domain adaptation of CLIP; certain nuanced style preferences are lost.

- **Low-Rank Matrix Completion**: Model sensitivity to rank and regularization requires manual tuning.

- **Two-Tower Model**: Suffers from embedding drift if CLIP updates aren't locked; batch normalization was key to stabilizing convergence.

. . . . . . . . . . . . . . . . . . . .

### Resource Usage Measurements

- **Collaborative Filtering**: Memory-efficient for small datasets (~2GB RAM); similarity matrix computation and ranking took ~4–5 minutes for 40+ users and 60+ items. No GPU required.

- **Content-Based Filtering**: Required ~4GB GPU memory for CLIP inference; generating embeddings for 62 items (images + text) took approximately 10–12 minutes. Retrieval is fast post-embedding (O(N) cosine similarity).

- **Low-Rank Matrix Completion**: Executed efficiently on both CPU and GPU. Training with a rank of 7 and masked loss converged in under 2 minutes. Memory usage scaled linearly with $(m+n)\times k$ embedding storage.

- **Two-Tower Model**: Required ~7GB GPU memory for training 5 epochs on a batch size of 32. Item embeddings were precomputed, which sped up inference. CLIP usage and dual-tower architecture increased compute demand during training.

. . . . . . . . . . . . . . . . . . . .

### Unexpected Challenges

- **Collaborative Filtering**: Encountered sparse matrix issues (e.g., division-by-zero) due to missing ratings. Also saw instability in cosine similarity ranking for edge users with few co-rated items.

- **Content-Based Filtering**: CLIP required strict image formatting and batching to avoid GPU memory overflow. Semantic mismatches occasionally occurred when product descriptions were vague or noisy.

- **Low-Rank Matrix Completion**: Sensitive to initialization—improper starting weights led to degenerate matrices. Hyperparameter tuning (rank, regularization) was critical to avoid underfitting.

- **Two-Tower Model**: Convergence depended heavily on careful application of normalization and dropout. Also needed to freeze CLIP weights to avoid unintended embedding drift during training.

--- 


## Next Steps

### Immediate Improvements

- **Continuously expand our dataset**: Add more users and clothing items to increase model robustness and diversity in recommendations.  
  - Develop an automated scraping pipeline to collect images and metadata from retail sites.  
  - Generate synthetic user profiles and rating histories to simulate realistic behavior.

- **Improve image preprocessing and embeddings**:  
  - Standardize lighting, angles, and resolution across product photos.  
  - Fine-tune embedding quality using domain-specific augmentations (e.g., garment textures, folds).

- **Optimize clustering for unsupervised analysis**:  
  - Refine HDBSCAN parameters and similarity metrics.  
  - Explore embedding dimensionality reduction to improve interpretability and grouping accuracy.

- **Benchmark and consolidate models**:  
  - Evaluate trade-offs in accuracy, scalability, and interpretability.  
  - Narrow down to 1–2 models (e.g., Two-Tower + Content-Based) that best match our application goals.

. . . . . . . . . . . . . . . . . . . .

### Technical Challenges to Address

- **Scalability**: Efficiently index and search across large-scale item databases.  
- **Data Heterogeneity**: Harmonize metadata and formats across platforms (e.g., Zara vs. Uniqlo).  
- **Real-Time Inference**: Reduce latency for browser-based personalization and dynamic ranking.

. . . . . . . . . . . . . . . . . . . .

### Research Questions

- **Best Embedding Architectures**: How do CLIP, ViT, or hybrid networks compare in capturing subtle fashion semantics?  
- **User Feedback Loops**: What’s the best way to incorporate post-recommendation feedback (implicit or explicit)?  
- **Privacy and Compliance**: What are the guardrails for cross-site scraping and usage under evolving data policies?

. . . . . . . . . . . . . . . . . . . .

### Alternative Approaches to Explore

- **Hybrid Systems**: Blend collaborative and content-based signals using learned weighting schemes.  
- **Active Learning Pipelines**: Use human-labeled preferences or curated clusters to guide model refinement.  
- **Reinforcement Learning**: Adapt recommendations in real-time using clickstream or purchase behavior.

. . . . . . . . . . . . . . . . . . . .

### Lessons Learned

- **Visual Clustering**: CLIP is powerful but must be paired with preprocessing and tuning to ensure quality groupings.  
- **Data Scale and Cleanliness**: Small or noisy datasets lead to overfitting or generic outputs—expanding clean data pipelines is critical.  
- **Model Design Tradeoffs**: Balancing interpretability, cold-start coverage, and runtime complexity is key for deployment readiness.
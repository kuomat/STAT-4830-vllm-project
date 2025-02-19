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

## Technical Approach
### Collaborative Filtering

#### Mathematical Formulation
##### Objective Function:
For collaborative filtering, we aim to predict missing user-item interactions through three main approaches:

1. **User-Based CF:**
\[
\hat{r}_{ui} = \frac{\sum_{v \in N_k(u)} sim(u,v) \cdot r_{vi}}{\sum_{v \in N_k(u)} sim(u,v)}
\]
where $\hat{r}_{ui}$ is the predicted rating for user u on item i, $N_k(u)$ is the set of k most similar users to u, and sim(u,v) is the cosine similarity between users.

2. **Item-Based CF:**
\[
\hat{r}_{ui} = \frac{\sum_{j \in N_k(i)} sim(i,j) \cdot r_{uj}}{\sum_{j \in N_k(i)} sim(i,j)}
\]
where $N_k(i)$ is the set of k most similar items to i.

3. **Neural CF:**
\[
\hat{r}_{ui} = f(W_2 \cdot ReLU(W_1 \cdot [e_u; e_i] + b_1) + b_2)
\]
where $e_u$ and $e_i$ are user and item embeddings, and $W_1$, $W_2$, $b_1$, $b_2$ are learned parameters.


##### Constraints:
1. **Cold-Start:** Limited effectiveness for new users/items
2. **Scalability:** Computation grows with user/item count

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

**Justification:**
- Multiple approaches provide robustness
- Each method compensates for others' weaknesses
- Neural CF adds non-linear modeling capability

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

#### Resource Requirements and Constraints
1. **Computational Resources:**
- Memory: O(|U| × |I|) for similarity matrices
- CPU: Significant for large-scale similarity computations
- GPU: Required for efficient Neural CF training

2. **Storage Requirements:**
- User-item interaction matrix
- Similarity matrices
- Model parameters

3. **Scalability Considerations:**
- User-based CF: O(|U|²) similarity computations
- Item-based CF: O(|I|²) similarity computations
- Neural CF: O(batch_size × embedding_dim)

4. **Performance Constraints:**
- Real-time recommendation latency
- Batch update frequency
- Memory limitations for large datasets

The collaborative filtering implementation provides complementary recommendations to our content-based system. While content-based filtering leverages item features, collaborative filtering captures user behavior patterns and preferences. The combination of both approaches in a hybrid system offers more robust and accurate recommendations.


### Content-Based Filtering

#### Mathematical Formulation
##### Objective Function:
Our objective is to recommend items that maximize the similarity between user preferences and item features, represented by text and image embeddings:

![Equations](content_filtering.png)

##### Constraints:
1. **Cold-Start Handling:** Users have no historical ratings, so only item content embeddings (text, image) are used.
2. **Diversity Constraint:** Optional constraint to limit similar items in recommendations.
3. **Resource Constraint:** Limited memory and computational resources.

#### Algorithm/Approach Choice and Justification
##### Approach: Content-Based Filtering with CLIP Embeddings
We use CLIP (Contrastive Language-Image Pretraining) to encode both item text descriptions and images into a shared feature space. By relying on item content rather than user interaction history, we address the cold-start problem effectively.

**Justification:**
- **Rich Representations:** CLIP embeddings capture semantic meaning from both text and images.
- **No User History Required:** Suitable for cold-start scenarios.
- **Efficient Similarity Computation:** Cosine similarity is fast and efficient.

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

### Low Rank
#### Mathematical Formulation

**Objective Function:**  
For low-rank matrix completion, we treat the user–item rating matrix $M$ as approximately factored into 
$U \in \mathbb{R}^{m \times r}$ and $V \in \mathbb{R}^{n \times r}$, giving $M \approx U \times V^\top$.  
We only compute reconstruction error over known entries in $M$. Let $\Omega$ be the set of observed $(i,j)$ pairs; then our objective is:

$$
\min_{U, V} \sum_{(i,j)\in \Omega} \bigl(M_{ij} - (UV^\top)_{ij}\bigr)^2
$$

**Constraints:**  
- **Rank Constraint:** $U$ and $V$ must each have rank $\le r$.  
- **Cold-Start:** New users/items with no known entries still pose challenges.  
- **Sparsity:** Additional regularization or side information may be needed for very sparse data.

#### Algorithm/Approach Choice and Justification

- **Burer–Monteiro Factorization:** We directly optimize $U$ and $V$ instead of a full $(m \times n)$ matrix.
- **Memory Efficiency:** Requires storing only $(m + n)\times r$ floats, much smaller than a full matrix for large $m,n$.
- **Flexibility:** Easy to add user or item biases, or partial side embeddings.

#### Implementation Steps

1. **Initialize** $U$ ($(m \times r)$) and $V$ ($(n \times r)$) randomly.
2. **Create a mask** for known entries in $M$; fill unknown cells with 0 (or mean).
3. **Compute Predictions** as $\hat{M} = U \times V^\top$.
4. **Loss Computation** on known entries only:
   $\sum_{(i,j)\in \Omega} (M_{ij} - \hat{M}_{ij})^2$.
5. **Backprop/Optimize** using Adam or SGD to update $U$ and $V$.
6. **Repeat** until convergence or until the epoch limit is reached.

#### Validation Methods

- **Offline Evaluation:** Compute MSE, RMSE, or Precision@K on the reconstructed matrix.
- **Cross-Validation:** Hide some known entries to measure generalization.
- **Cold-Start Users:** Specifically hide items for new users to see how predictions fare.

#### Resource Requirements and Constraints

- **Hardware:**
   - **GPU:** Recommended for large-scale factorization with PyTorch.
   - **Memory:** Dependent on $(m + n)\times r$; large datasets may require 16GB+ RAM.

- **Software:**
   - Python 3.10+
   - PyTorch (with CUDA support for GPU)
   - `pandas`, `numpy`, `sklearn`


### Two-Tower Model for Personalized Recommendations

#### Introduction

The Two-Tower Model is a deep learning-based recommendation system designed to learn representations for both users and items in a shared embedding space. The goal is to efficiently compute similarity between users and items, enabling personalized recommendations. The model consists of two separate neural networks—one for users and one for items—which map their respective inputs to a common latent space. The cosine similarity between user and item embeddings is then used to determine relevance.

This document provides an in-depth explanation of the Two-Tower Model's structure, its implementation in PyTorch, the training process, validation methods, and evaluation metrics. Additionally, it analyzes the visual results obtained from the training process.

---

#### Mathematical Formulation

The model learns to encode users and items as fixed-length vectors, allowing for efficient retrieval through similarity computation. The similarity between a user and an item is determined using cosine similarity, which is calculated as follows:

\[
S(U, I) = \frac{U \cdot I}{\| U \| \| I \|}
\]

where \( U \) is the user embedding and \( I \) is the item embedding. The closer the cosine similarity is to 1, the more relevant the item is predicted to be for the user.

The model is trained using contrastive learning, with a Binary Cross-Entropy (BCE) loss function. The loss function ensures that positive user-item pairs (items the user has interacted with) have high similarity scores, while negative pairs (randomly chosen items the user has not interacted with) have low similarity scores. The loss function is defined as:

\[
L = - \sum_{(U, I^+)} \log(S(U, I^+)) - \sum_{(U, I^-)} \log(1 - S(U, I^-))
\]

where \( I^+ \) denotes a positively interacted item and \( I^- \) denotes a negative item.

---

#### Algorithm Choice and Justification

The Two-Tower Model was chosen for its efficiency in large-scale recommendation tasks. Unlike traditional collaborative filtering, which requires direct user-item interaction matrices, the Two-Tower architecture allows for independent processing of users and items. This separation enables precomputing item embeddings, making inference more efficient. The model is particularly effective in cases where multiple input modalities, such as text and images, are involved, as it can process them separately and learn their representations independently.

One of the key advantages of this architecture is its ability to handle new items efficiently. Since the item tower functions independently of specific user data, new items can be embedded without retraining the entire model. However, a key limitation is the cold-start problem for new users who have limited or no interaction history.

---

#### Training Process

The model is trained using positive and negative pairs of user-item interactions. The user and item embeddings are computed separately and then compared using cosine similarity. The objective is to push the similarity of positive pairs closer to 1 while ensuring that negative pairs have lower similarity scores.

During training, a dataset is created where for each user, a positive item is chosen from their interaction history, and a negative item is selected randomly. The model is trained using the BCE loss function, and optimization is performed using the Adam optimizer.

The training process is run for multiple epochs, with the loss being monitored to ensure convergence. The training loss curve, as seen in the third image, shows a consistent decrease over epochs, indicating that the model is learning effectively. The initial loss starts high but decreases over time, stabilizing after several epochs.

---

#### Validation

To evaluate the model's performance, precision-based metrics are used. Precision at K (Precision@K) measures the proportion of relevant items within the top K recommendations provided to a user. It is computed by checking how many of the recommended items were actually interacted with by the user.

Mean Average Precision at K (MAP@K) extends this concept by incorporating ranking order. It calculates the average precision for each user and then computes the mean across all users. This metric is useful for ensuring that highly relevant items appear earlier in the recommendation list.

The qualitative validation of recommendations is also performed by visually inspecting the results. The first image shows personalized recommendations for multiple users, displaying the top five suggested items along with their images and descriptions. The recommendations appear diverse and relevant to user preferences, suggesting that the model effectively learns item relationships.

---

#### Visualization Analysis

##### t-SNE Visualization of Item Embeddings

The second image presents a t-SNE visualization of item embeddings, showing how items are clustered based on their learned representations. Each point represents an item, and the colors correspond to different clusters. The distinct grouping of points suggests that the model has successfully captured meaningful patterns in the data.

##### Cluster Distribution Analysis

The second image also contains a bar chart displaying the cluster distribution of items. Each bar represents the number of items assigned to a particular cluster. The presence of multiple clusters with fairly even distribution suggests that the embeddings capture diverse item characteristics rather than collapsing into a single dominant category.

##### Training Loss Curve

The third image displays the training loss curve, showing how the model's loss decreases over epochs. The sharp decline in the early epochs followed by gradual stabilization indicates that the model is learning efficiently. This suggests that the optimization process is working as expected, leading to improved recommendations over time.

---

#### Resource Requirements and Constraints

Given the relatively small dataset used in this implementation, computational constraints are minimal. The primary resource requirement is GPU acceleration for training efficiency, as encoding both text and images using CLIP can be computationally expensive. Since item embeddings can be precomputed, inference time is reduced significantly, making the system scalable for larger datasets.

Memory requirements increase with dataset size, especially when storing high-dimensional embeddings. If the dataset were to grow, approximate nearest neighbor search techniques could be incorporated to improve retrieval speed without significantly increasing computational cost.

While the current implementation is effective for smaller-scale testing, a real-world deployment would require additional optimizations such as efficient indexing, batch processing, and caching of frequently accessed embeddings.

---

#### Conclusion

The Two-Tower Model successfully learns representations for both users and items, allowing for efficient personalized recommendations. The use of contrastive learning ensures that positive interactions are ranked higher than randomly chosen negative samples. The evaluation results indicate that the model is able to generate diverse and relevant recommendations for different users.

The qualitative and quantitative validation processes, including t-SNE visualization, cluster distribution analysis, and training loss monitoring, confirm that the model is learning meaningful item embeddings. The architecture is efficient and scalable, making it well-suited for real-world recommendation systems. Future improvements could focus on handling the cold-start problem for new users, refining hyperparameters for better performance, and integrating additional user behavior signals to enhance recommendation quality.

## Initial Results

### CLIP Embeddings for Unsupervised Clothing Image Clustering

To test the feasibility of using CLIP embeddings for unsupervised clothing image clustering, I conducted two separate experiments using different datasets.

- **Dataset 1:** A small collection of **76 images** obtained from shopping websites, featuring various types of clothing, including pants, t-shirts, blouses, and hoodies. Some images contained models wearing the clothes, while others displayed the garments alone.
- **Dataset 2:** A subset of **1,000 images** from the **Fashion-MNIST dataset**, used to assess the model's performance on a larger but still manageable dataset.

For clustering, I initially attempted **DBSCAN** using both Euclidean distance and cosine similarity. However, DBSCAN failed to form any clusters in both datasets, likely due to the high dimensionality and sparse nature of the embedding space.

I then switched to **HDBSCAN**, which is better suited for variable density data:
- **Custom shopping dataset**: Resulted in **three clusters**.
- **Fashion-MNIST dataset**: Produced **33 clusters**.

Given the relatively small dataset sizes, the code executed in under a minute, consuming less than 10% of the CPU.

---

## Evidence Your Implementation Works

We successfully implemented collaborative filtering, content-based filtering, and the two-tower model using PyTorch, demonstrating their ability to produce recommendations based on user history across multiple platforms.

Note: the .ipynb files can be found under the notebooks folder in the root direction.

- **Collaborative Filtering**: Generated personalized recommendations for users such as Laura and Matt based on user and item similarity scores using cosine similarity. Results displayed distinct item recommendations for different users.

- **Content-Based Filtering**: Produced recommendations based on text and image embeddings from CLIP, confirming the model’s capability to match items with similar features. We visualized top recommendations for users like Vivian, highlighting the model’s effectiveness.

- **Low-Rank Matrix Completion**: Factorized the sparse user-item rating matrix into two lower-dimensional matrices (U and V). This allowed us to infer missing ratings, particularly useful for users or items with limited history. Using a Burer–Monteiro approach in PyTorch, we verified that unknown entries could be reconstructed to produce reasonable recommendations.

- **Two-Tower Model**: Delivered user-item recommendations by jointly learning user and item embeddings, validated through loss convergence and t-SNE visualization of item embeddings.
- 
## Basic Performance Metrics

- **Collaborative Filtering**: Precision@5: 0.72, Recall@5: 0.68 on test users.

- **Content-Based Filtering**: Mean Average Precision (MAP): 0.81; NDCG@5: 0.79.

- **Low-Rank Matrix Completion**: Achieved a test MSE of 33.88 (corresponding to RMSE = 5.82) after 500 epochs of factorization. During training, the loss on masked entries decreased from ~47.0 down to ~0.16.

- **Two-Tower Model**: Final training loss converged to 0.021 after 5 epochs.

## Test Case Results

- **Collaborative Filtering**: Laura received high-rated Uniqlo and Abercrombie items consistent with her preferences.

- **Content-Based Filtering**: Vivian’s recommendations matched her history from Forever 21 and similar styles.

- **Low-Rank Matrix Completion**: Inferred missing ratings for Matt across a wide range of items, enabling suggested outfits that matched his purchase patterns on other items.

- **Two-Tower Model**: Provided distinct recommendations for each user based on shared embeddings, differentiating between users.

## Current Limitations

- **Collaborative Filtering**: Poor performance for cold-start users with minimal ratings.

- **Content-Based Filtering**: Struggles with users whose preferences are not well-captured in text/image embeddings.

- **Low-Rank Matrix Completion**: Requires some known ratings per user and item to achieve meaningful factorization; purely cold-start users still pose a challenge.

- **Two-Tower Model**: Requires significant training time and computational resources for large datasets.

## Resource Usage Measurements

- **Collaborative Filtering**: RAM usage: ~2GB; Training time: 5 minutes for 44 users and items.

- **Content-Based Filtering**: GPU usage: 4GB; Time to generate embeddings: 12 minutes for 62 items.

- **Low-Rank Matrix Completion**: Memory usage scales with \((m+n)\times\)$rank. For our small dataset (62 items × 4 users, rank=3), training took under 2 minutes on CPU/GPU.

- **Two-Tower Model**: GPU usage: 7GB; Training time: 15 minutes for 5 epochs with batch size 32.

## Unexpected Challenges

- **Collaborative Filtering**: Encountered division-by-zero errors due to sparse user-item matrices.

- **Content-Based Filtering**: Slow inference times from large CLIP embeddings.

- **Low-Rank Matrix Completion**: Needed careful initialization and regularization for high sparsity; otherwise, factorization could converge to trivial solutions.

- **Two-Tower Model**: Convergence issues requiring batch normalization adjustments.

## Future Work
- **Sscrape websites for images and metadata**: start automating process of data retrieval.
- **Test GPU acceleration** for larger datasets to reduce execution time.

## Next Steps
### Immediate Improvements Needed
- **Enhance Image Preprocessing**: Standardize image formats, normalize lighting/angles, and use domain-specific augmentations (e.g., garment features).
- **Keep adding new data to current simulated dataset**: adding new users.
- **Optimize clustering parameters** to improve grouping accuracy and to potentially create features to be used in content-filtering.
- **Refine and narrow down current 4 recommendation algorithms**: pick which one(s) perform best and can be implemented reasonably.

### Technical Challenges to Address
- **Data Heterogeneity**: Standardizing metadata formats across multiple e-commerce platforms.
- **Scalability**: Managing millions of items efficiently across various sites.
- **Real-Time Performance**: Ensuring low-latency recommendations for browser-based inference.

### Questions You Need Help With
- **Model Selection**: Which embeddings or pretrained networks (e.g., CLIP, ViT) best handle diverse clothing styles?
- **Privacy & Compliance**: How can we align cross-site data collection with evolving privacy regulations and user consent?
- **Metadata Integration**: Best practices to scrape and incorporate textual product data without site-related restrictions.

### Alternative Approaches to Try
- **Hybrid Models**: Combine text and image embeddings to capture richer, context-aware representations.
- **Active Learning**: Involve human-in-the-loop labeling to refine clusters and continually improve accuracy.
- **Reinforcement Learning**: Create adaptive recommendation systems that learn dynamically from user interactions.

### What You've Learned So Far
- **Clustering & Image Embeddings**: Grouping by visual similarity shows promise but requires careful preprocessing.
- **Training Data**: Inconsistent or insufficient data hinders cluster quality, highlighting the need for more robust pipelines.

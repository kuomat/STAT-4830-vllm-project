## Problem Statement  

We aim to optimize personalized recommendations for users when they visit a new e-commerce website where they have minimal prior purchase history. Our approach leverages metadata (e.g., product images, reviews, descriptions) from other shopping platforms the user has interacted with to generate relevant recommendations. We integrate this metadata into a graph-based model (GraphSAGE) to improve cold-start recommendation quality.

Cold-start issues in recommendation systems lead to poor user experience, making it difficult for users to receive relevant suggestions when they switch to a new platform. This can cause frustration, reduce engagement, and limit conversions for businesses.  

### Success Metrics  
- **Recommendation Relevance**: Precision@k, Recall@k, and NDCG to evaluate how well our recommendations match user interests.  
- **User Engagement**: Click-through rate (CTR) on recommended products.  

### Constraints  
- **Data Availability**: Not all websites expose metadata in the same format (e.g., missing product descriptions, different image qualities).  
- **Real-Time Performance**: The web extension must generate recommendations quickly without excessive computational overhead.  
- **User Privacy**: Ensuring ethical data usage without tracking sensitive information.  

### Required Data  
- **User interactions** from shopping sites (order history, wish lists, browsing activity).  
- **Product metadata** (titles, descriptions, images, categories, prices, brands) across multiple websites.  
- **User-generated content** (ratings, reviews, preferences) from different platforms.  

### Potential Pitfalls  
- **Sparse shopping history from other sites**: Recommendations may be weak.  
- **Metadata Mismatch**: Makes standardization potentially difficult.  
- **Scale**: Large product graphs may introduce latency in recommendations.  
- **Privacy Concerns**: Tracking user activity across websites must comply with data protection regulations.


## Technical Approach
### Collaborative Filtering

### Content-Based Filtering

### Low Rank

### Two Tower

### Mathematical formulation (objective function, constraints)
### Objective Function  

The objective function optimizes a ranking loss for cold-start users/items:

$\max_{\theta} \sum_{u \in U_{cs}} \sum_{i \in I} y_{ui} \log (\sigma(f(u, i))) + (1 - y_{ui}) \log (1 - \sigma(f(u, i)))$

where:  


- $\( U_{cs} \)$ = cold-start users  
- $\( I \)$ = items  
- $\( y_{ui} \)$ = binary indicator if user $\( u \)$ engaged with item $\( i \)$  
- $\( f(u, i) \)$ = GraphSAGE-based scoring function  
- $\( \sigma \)$ = sigmoid function  


### Algorithm/approach choice and justification
- One of the libraries we will be using is PyTorch Geometric (PyG) which is for GraphSAGE-based embedding learning
- A specific use case of the PyG library is that it provides a neighbor sampling method which handles large graphs like the graph we will be using for this project extremely well.


### PyTorch implementation strategy
- A PyTorch library that is going to be helpful for this project is the Transformers library maintained by Hugging Face. This library not only lets us extract embeddings from images or texts, but we can also obtain some pre-trained models here.
- Our initial plan is to scrape data from different e-commerce sites, clean the metadata, and store the structured graph format in PyTorch as tensors.


### Validation methods
- We will split our data into training, validation, and testing sets and will be validating our model and algorithm on the validation set to simulate cold-start conditions
- To tune hyperparameters better, and to get a more reliable performance estimation, we will be using the k-Fold Cross-Validation technique where we are setting k to 5 for now.


### Resource requirements and constraints
- Since we are fine-tuning the embeddings as well as running machine learning models to generate our predictions, we will probably need GPUs beyond the free-tier provided by Google Colab

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

- Successfully **extracted CLIP embeddings** for all images.  
- Applied **clustering algorithms** to organize similar clothing items.  
- HDBSCAN produced meaningful groupings:  
  - In the **custom dataset**, three clusters predominantly grouped **sweaters, sweatpants, and dresses/skirts**.  
  - In **Fashion-MNIST**, 33 clusters were identified, many correctly grouping similar clothing items.  
- However, there were **many outliers**, particularly among **patterned clothing**, suggesting the model struggles with learning fabric textures.  

---

## Basic Performance Metrics  

Performance varied between the two datasets:  

- **Custom Shopping Dataset (76 images)**  
  - Ran in under **one minute**, using less than **5% CPU**.  
  - HDBSCAN identified **three clusters**, successfully distinguishing between broad categories of clothing.  
  - **Struggled with patterned items.**  

- **Fashion-MNIST Dataset (1,000 images)**  
  - Completed in a **few minutes**, maintaining minimal CPU usage.  
  - HDBSCAN produced **33 clusters**, showing a more refined grouping of clothing items.  
  - Some clusters mixed different clothing types, but overall, the larger dataset improved clustering performance.  

---

## Test Case Results  

- **DBSCAN (Euclidean & Cosine Similarity)**: Failed to generate meaningful clusters in both datasets.  
- **HDBSCAN (Custom Dataset)**:  
  - Formed **three distinct clusters**: **sweaters, sweatpants, and dresses/skirts**.  
- **HDBSCAN (Fashion-MNIST Dataset)**:  
  - Formed **33 clusters**, many accurately grouping similar clothing items.  
  - Some clusters contained **mixed clothing types**.  
  - **Outliers remained an issue**, especially for patterned garments.  

These results suggest that **CLIP embeddings are useful for capturing high-level visual similarities**, but may struggle with **fine-grained details such as fabric textures and small design variations**.  

---

## Current Limitations  

1. **Presence of outliers**, particularly among patterned clothing.  
2. **Lack of texture understanding**: CLIP may not fully capture fabric material similarities.  
3. **Small dataset size**:  
   - DBSCAN may require more samples to form meaningful clusters.  
   - Small sample size could contribute to suboptimal clustering results.  
4. **High number of clusters in Fashion-MNIST**:  
   - Some clusters contained **mixed clothing categories**.  
   - Hyperparameters such as `min_samples` and `cluster_selection_epsilon` may need fine-tuning.  

---

## Resource Usage Measurements  

- **Custom Shopping Dataset (76 images)**:  
  - **Executed in under a minute**.  
  - **Used less than 5% CPU**.  
  - Minimal resource demand due to small dataset size.  

- **Fashion-MNIST Dataset (1,000 images)**:  
  - **Completed in a few minutes**.  
  - **Minimal CPU usage**.  
  - Could benefit from **GPU acceleration** for larger datasets.  

At this stage, **memory consumption and processing time have not posed significant challenges**.  

---

## Unexpected Challenges  

- **DBSCAN Failure:**  
  - Did not cluster the data, even with cosine similarity.  
  - Likely due to the **sparse distribution of high-dimensional feature vectors** from CLIP embeddings.  

- **Significant number of outliers**:  
  - Especially among **patterned clothing**, suggesting that CLIP embeddings may not capture texture well.  

- **Higher-than-expected cluster count in Fashion-MNIST**:  
  - Some clusters mixed different types of clothing.  
  - Clustering parameters need further **refinement and hyperparameter tuning**.  

---

## Future Work  

- **Optimize clustering parameters** to improve grouping accuracy.  
- **Experiment with additional datasets**, particularly fashion-specific datasets.  
- **Explore fine-tuning CLIP embeddings** or incorporating additional **texture-sensitive feature extraction techniques**.  
- **Test GPU acceleration** for larger datasets to reduce execution time.  
- **Investigate hybrid approaches**: Combine CLIP embeddings with other visual features (e.g., handcrafted texture features) for better clustering performance.  

## Next Steps
### Immediate Improvements Needed
- **Refine Clustering Techniques**: Adjust hyperparameters (e.g., DBSCAN vs. k-means) and apply advanced metrics to validate cluster quality.
- **Enhance Image Preprocessing**: Standardize image formats, normalize lighting/angles, and use domain-specific augmentations (e.g., garment features).
- **Incorporate Textual Metadata**: Integrate product titles, descriptions, and brand data for more robust feature representation.

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

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


## Technical Approach (1/2 page)
### Mathematical formulation (objective function, constraints)
### Objective Function  

The objective function optimizes a ranking loss for cold-start users/items:

```\[
\max_{\theta} \sum_{u \in U_{cs}} \sum_{i \in I} y_{ui} \log (\sigma(f(u, i))) + (1 - y_{ui}) \log (1 - \sigma(f(u, i)))
\]```

where:  

```
- \( U_{cs} \) = cold-start users  
- \( I \) = items  
- \( y_{ui} \) = binary indicator if user \( u \) engaged with item \( i \)  
- \( f(u, i) \) = GraphSAGE-based scoring function  
- \( \sigma \) = sigmoid function  
```

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

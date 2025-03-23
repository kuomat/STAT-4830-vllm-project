# LLM Exploration Summary  

## Session Focus  
Our project focuses on optimizing personalized recommendations for users facing the **cold-start problem** on e-commerce sites. We explored **GraphSAGE embeddings**, **CLIP-based clustering**, and **metadata integration** to improve recommendations with minimal user history. More recently, we investigated **content filtering techniques** using CLIP to **categorize and cluster product images** effectively. Discussions focused on refining our **mathematical formulation**, **optimizing PyTorch implementations**, **tuning clustering algorithms**, and **validating model performance**.  

## Surprising Insights  

### Conversation: Cold-Start Recommendation Objective Function & Constraints  
**Prompt That Worked:**  
- *"What is the best mathematical formulation for optimizing a recommendation model under cold-start conditions using GraphSAGE embeddings?"*  

**Key Insights:**  
- **GraphSAGE embeddings** optimize neighbor-aware representations, making them ideal for cold-start scenarios.  
- A **ranking-based loss function (e.g., Bayesian Personalized Ranking (BPR) or Binary Cross-Entropy (BCE))** works better than standard regression-based losses.  
- **Graph sparsification** can help mitigate issues caused by missing metadata while preserving relevant user-item relationships.  

### Conversation: CLIP-Based Content Filtering & Clustering  
**Prompt That Worked:**  
- *"Why is DBSCAN failing to cluster CLIP embeddings of product images, and what alternatives should we use?"*  

**Key Insights:**  
- **DBSCAN struggled due to the high-dimensional nature of CLIP embeddings**.  
- Switching to **HDBSCAN improved clustering performance**, particularly in adapting to variable densities.  
- **CLIP embeddings don’t inherently capture texture and material differences**, leading to poor distinction between patterned vs. plain clothing.  
- **Hybrid embeddings** (combining CLIP with text metadata) were suggested to improve clustering performance.  

### Conversation: Optimizing CLIP for E-Commerce Product Categorization  
**Prompt That Worked:**  
- *"How can we improve product categorization using CLIP embeddings while handling patterned and textured items better?"*  

**Key Insights:**  
- **Fine-tuning CLIP on domain-specific datasets** (e.g., fashion-specific product images) can improve differentiation.  
- **Contrastive learning** could be useful to refine CLIP embeddings and prevent irrelevant clustering.  
- **Image augmentations (e.g., grayscale conversion, edge detection)** may help the model better differentiate patterns and textures.  

### Conversation: PyTorch Implementation for Scalable Graph & Content-Based Filtering  
**Prompt That Worked:**  
- *"How can we efficiently implement GraphSAGE and CLIP-based filtering in PyTorch for real-time recommendations?"*  

**Key Insights:**  
- **PyTorch Geometric (PyG) was the best choice for GraphSAGE**, particularly with its efficient neighbor sampling.  
- **Batch processing** and **on-the-fly neighbor aggregation** are necessary to prevent memory overload when handling large graphs.  
- **Transformers from Hugging Face** can be leveraged to **extract both image and text embeddings** efficiently.  

## Techniques That Worked  
- **Asking “Why did X fail?” rather than just requesting alternatives** led to deeper insights on clustering failures.  
- **Using structured prompts** like *"Given our cold-start problem, which embeddings and losses optimize ranking performance?"* helped extract precise recommendations.  
- **Iterative refinement of prompts** (e.g., asking for PyTorch optimizations based on prior responses) led to better implementation strategies.  

## Dead Ends Worth Noting  

### Approach: Using DBSCAN for CLIP Image Clustering  
- **DBSCAN failed** because high-dimensional embeddings don’t have uniform density distributions.  
- **HDBSCAN worked better**, but still struggled with differentiating textures and patterns.  

### Approach: Treating CLIP as a Universal Solution  
- **CLIP embeddings alone weren’t enough**—they lacked the ability to distinguish fine-grained clothing details.  
- **Hybrid embeddings (image + text) performed better** but required additional processing.  

### Approach: Not Considering Scalability from the Start  
- Early GraphSAGE implementations **didn’t account for real-time performance**, requiring neighbor sampling optimizations.  
- The **initial lack of batch processing** led to **high memory consumption** during training.  

## Next Steps  
- [ ] **Fine-tune CLIP embeddings** with **domain-specific datasets** to improve product categorization.  
- [ ] **Integrate text metadata** (titles, descriptions, brands) alongside image embeddings for improved recommendation quality.  
- [ ] **Test hybrid embedding models** (e.g., CLIP + Contrastive Learning) to refine product similarity scoring.  
- [ ] **Optimize GraphSAGE’s real-time inference pipeline** to ensure scalability.  

### Questions to Explore  
- [ ] **How do we best fine-tune CLIP for fashion recommendations?**  
- [ ] **Can reinforcement learning improve cold-start recommendations?**  
- [ ] **What’s the best way to ensure privacy compliance while leveraging multi-platform user data?**  

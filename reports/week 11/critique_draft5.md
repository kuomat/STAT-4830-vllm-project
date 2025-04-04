## Critique of Fifth Draft

### **Strengths**

- **Persona Generation Script**: The script for generating personas was well-designed and efficient, allowing us to generate ratings for each persona with high accuracy and consistency. Because of this script, we can infinitely expand the number of personas and have a very diverse set of ratings for each persona.
- **Image Data**: We added a lot more high-quality product images to our dataset, which helped improve the quality of content-based and two-tower recommendations. This also allowed us to run more robust models, such as low-rank matrix completion and collaborative filtering. With the increase in the number of images and personas, it will be easier for our models to match the new users to an existing persona and generate very accurate ratings on things the users will potentially like.
- **Robust Technical Approaches**: As we experiment more with our different models, we are starting to see which ones are the msot effective and thus we should put more emphasis on them. This will let us also focus more on other things such as putting the images on the cloud or optimizing the two tower algorithm even more for scalability.


### **Weaknesses**

- **No Hybrid Benchmark Yet**: We still have yet to find a good benchmark that people have been working on to compare our results with them,.
- **Cold-Start Metric Reporting**: While cold-start is the core motivation, explicit performance breakdowns for zero-history users are still limited beyond one or two metrics.
- **Metadata Limitations Not Quantified**: Variability across brand metadata is acknowledged, but no quantitative examples or preprocessing impact are explored.


### **Critical Risks/Assumptions**

- **Embedding Generalization**: CLIP may not generalize well to niche or region-specific fashion domains without fine-tuning.
- **Scalability Constraints**: Two-tower model assumes infrastructure capable of storing and retrieving embeddings at scale—requires confirmation of system design feasibility.
- **Generated Ratings**: We are assuming that the LLMs that we used are not hallucinating and generating random ratings. Although we have done some quality checks on the generated ratings, there is still a possibility that the ratings are biased or not representative of the personas.


### **Concrete Next Actions**

1. **Generating Images**: Either ask large models to generate the images or find more datasets to make sure we have a comprehensive set of images.
2. **Cold-Start Focus**: Design evaluation sets with true cold-start users/items and report performance delta across models.
3. **Metadata Audit**: Analyze which metadata types (text, price, image) contribute most to content-based model accuracy.
4. **Clustering-to-Filtering Pipeline**: Explore using unsupervised item clusters as features in collaborative models.


### **Resource Needs**

- **Larger-Scale User Behavior Data**: Additional real user-item interactions across platforms to better simulate cold-start settings.
- **Fine-Grained GPU Profiling**: More detailed GPU utilization tracking to optimize batching, embedding caching, and similarity search.
- **Human Feedback Integration**: UX testing or user feedback on actual recommendations to validate semantic relevance beyond metrics.

# Week 13 Self-Critique

## OBSERVE  
- The narrative is coherent, but some sections feel uneven in depth.
- Re-ran front-to-back benchmarks and unit tests: performance numbers hold up, but no coverage for edge cases like empty selections or missing embeddings.  
- Initial questions: Where are the offline ranking metrics? How are cold-start users explicitly validated here?

---

## ORIENT

### Strengths  
- **End-to-end integration demonstrated**: Clear linkage from data (CLIP embeddings, synthetic personas) through backend (FastAPI + PyTorch) to frontend (React pages) with concrete latency and render timings.  
- **Concrete performance profiling**: Provides p99 API latency (45 ms), frontend render time (180 ms), bundle size—excellent for deployment planning.  
- **Comprehensive model lineup**: Summarizes four distinct recommendation approaches with their trade-offs in a compact table, giving readers a quick mental map of the system.

### Areas for Improvement  
- **Missing offline ranking metrics in Week 13 snapshot**: The report omits Precision@5/NDCG@5 numbers, which were central success metrics earlier in the project.  
- **Validation coverage gaps**: No mention of tests or results for true cold-start cases (users with zero prior picks) or handling of missing data in the pipeline.  
- **Under-specified training regimen**: “Technical Approach” lacks details on training hyperparameters (learning rate, batch size), data splits, and convergence behaviour for the Two-Tower model.

### Critical Risks / Assumptions  
We rely entirely on synthetic personas and LLM-generated ratings—if those diverge from real user tastes, our offline metrics and even UI validation may be misleading. Additionally, we assume CLIP embeddings encapsulate style nuances; failure to capture fabric or brand cues could degrade recommendation relevance in production.

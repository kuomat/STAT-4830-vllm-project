
# Two-Tower LLM Exploration Log

This document chronicles our use of an LLM to explore, explain, implement, and finalize a two-tower (dual-encoder) architecture for fashion recommendation tasks within the PennOS project, leveraging CLIP embeddings and existing code modules.

---

## Session Focus

**What sparked this exploration?**
A requirement to build an efficient retrieval system mapping users and fashion items into a shared embedding space—facilitating scalable, real-time recommendations by precomputing item vectors and querying via nearest-neighbor search.

---

## Implementation Journey

### 1. Prototyping the Two-Tower Model

**Initial LLM-Assisted Prototype:**
We first used an LLM to scaffold a PyTorch Lightning two-tower model, generating `UserTower` and `ItemTower` modules with a contrastive loss.

**Key Notes:**

* We used a 2-layer MLP on the cosine similarity to regress ratings.
* This lean module replaced the heavier Lightning prototype for faster inference in our microservice.

### 2. Integrating CLIP Embeddings Integrating CLIP Embeddings

**Why CLIP?**
CLIP provides joint image-and-text embeddings; we used its text encoder to ensure our item descriptions live in the same semantic space as user preferences.

**Integration Steps:**

1. **Loading CLIP:** In `recommendation_service.py`, instantiated `clip.load("ViT-B/32")`, freezing the image encoder and fine‑tuning the text tower on fashion captions.
2. **Embedding Pipeline:** Wrapped CLIP text preprocessing (`clip.tokenize`) to convert item descriptions into tensors, then extracted 512‑dim vectors.

---

## Evaluation & Results

* **Embedding Quality:** Validated semantic coherence by manually inspecting nearest neighbors for sample users—recommendations aligned with known user interests.
* **Performance Benchmarks:** Measured average recommendation latency at 35ms under load (100 QPS) on a single GPU instance.
* **Recommendation Quality:** Achieved Recall\@10 of 0.62 on a held-out validation set of user–item interactions.

---

## Techniques That Worked

* **Modular Prompts:** Asked the LLM for individual code modules (model classes, service code, FAISS integration) rather than full monolith.
* **Few-Shot Examples:** Provided a minimal CLIP usage snippet to guide the model’s code generation toward our API.
* **Iterative Refinement:** After each generated code block, tested and asked follow-up prompts to fix errors (e.g., dimension mismatches, missing `self.save_hyperparameters()`).

---

## Final Product & Lessons Learned

* A robust two-tower retrieval system combining transformer-based user encoding with CLIP text embeddings for items.
* LLM assistance accelerated scaffolding, but required careful testing and iterative correction.
* Freezing most of CLIP’s backbone saved training time; fine‑tuning only text layers balanced performance and speed.

---

## Next Steps
* **Building the Retrieval Service for the web app**: Figuring out the endpoints needed to retrieve the data, images, and embeddings; figure out how to leverage caching and batches to speed up and make sure models aren't retraining every time we add data or run the model.
---



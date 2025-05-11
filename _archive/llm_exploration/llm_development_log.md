# Dataset LLM Exploration Log

This document captures key experiments, prompts, insights, and next steps from our ongoing exploration of large language models (LLMs) to generate synthetic fashion recommendation data via persona-based prompting.

---

## Session Focus

**What sparked this conversation?**
A need to generate realistic, diverse user reactions (ratings) for fashion items by leveraging LLMs (Mistral 7B) and persona-driven prompts, to augment sparse or unbalanced real-world recommendation datasets.

---

## Surprising Insights

### Conversation: Prompt calibration for negative feedback

**Prompt That Worked:**

```text
"this is my original prompt but it seems like not only is the model not returning any -1's but also it has a lot of 6's and 8's"
```

**Key Insights:**

* The LLM tended to avoid extreme negative ratings, clustering around mid-to-high scores.
* Mistral 7B defaults to safe outputs; explicit instruction to use `-1` for dislikes was not sufficient.
* We discovered that framing the task as a strict classification (only numeric output) and reiterating the scale could push the model toward a fuller dynamic range.

### Conversation: Expanding persona diversity

**Prompt That Worked:**

```text
"generate 10 more personas. 5 more males and 5 for females"
```

**Key Insights:**

* Mistral 7B quickly produced diverse names, backgrounds, and style preferences.
* When used in conjunction with the rating prompt, generated personas improved variance in synthetic ratings.
* We saw that richer persona bios (two-sentence minimum) yielded more varied and realistic rating distributions.

---

## Techniques That Worked

* **Strict Output Enforcement:** Using `[INST]...[/INST]` tags and stating "Return *only* the single numeric rating" prevented extra text or justification.
* **Explicit Scale Description:** Listing each point on the scale (−1, 1–10) clarified model expectations.
* **Persona-Bio Enrichment:** Adding details like age, occupation, style aesthetic, and favorite brands produced more nuanced ratings.
* **Iterative Prompt Refinement:** Soliciting feedback on the model’s errors (too many 6s/8s) guided targeted adjustments.

---

## Dead Ends Worth Noting

* **Open-Ended Rating Prompts:** Early prompts without tags or strict instructions led to verbose justifications rather than numeric outputs.
* **Single-Persona Batch Ratings:** Asking for ratings across all personas in one call caused collapsed outputs (e.g., one overall rating).
* **Omitting Negative Examples:** Providing only positive examples made the model shy away from `-1` outputs.

---

## Next Steps

* [ ] Experiment with temperature settings and sampling strategies on Mistral 7B to encourage extremes.
* [ ] Prepend a few few-shot examples explicitly showing `-1` ratings for disliked items.
* [ ] Automate persona generation and rating collection via API calls.
* [ ] Compare outputs across alternative LLMs (e.g., GPT‑4, Llama 2) for consistency and variance.

---

*Note: This log was initiated by me and refined with assistance from LLMs to structure, verify, and enrich the content.*

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



# Web App LLM Exploration Log

## Session Focus
> How we leveraged LLMs to accelerate development of our React-based recommendation web app by generating UI components, wiring up data flow, and handling edge cases.

## Surprising Insights

### Conversation: Generating Page Skeletons
**Prompt That Worked:**
- “Generate a React component for a selection page that lets users pick categories and items, with state management using hooks.”
  
**Key Insights:**
- The LLM produced a complete functional component with `useState` hooks and event handlers, saving us an hour of boilerplate coding.
- It suggested accessibility attributes (e.g., `aria-label`) we hadn’t initially considered.

### Conversation: Wiring Recommendation Service
**Prompt That Worked:**
- “Write a service module to call our recommendation endpoint and handle loading/error states in React.”
  
**Key Insights:**
- The LLM proposed a clean async/await pattern with centralized error handling.
- It recommended caching results in context to avoid redundant API calls.

## Techniques That Worked
- **Few-shot prompting** with two examples of similar pages to guide component structure.
- **Chain-of-thought prompting** to get the LLM to outline the data flow (from user action → API call → rendering).
- **Constraint specification** (“use only functional components and hooks”) to keep code consistent.

## Dead Ends Worth Noting
- **Prompting for styling details** (CSS-in-JS vs. CSS modules) initially led to inconsistent suggestions; we reverted to manual styling.
- **Asking for full routing setup** returned overly generic boilerplate that conflicted with our existing `react-router` configuration.

## Next Steps
**Code Review & Cleanup**: Conduct pair-review sessions, using the LLM to suggest improvements in naming and modularization.

---
*This log was drafted retrospectively with the help of an LLM to capture key moments in our web app development.*
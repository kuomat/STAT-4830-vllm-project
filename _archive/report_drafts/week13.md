# Week 13 Report Draft

## Problem Statement

**What are we optimizing?**  
We are building and evaluating a multimodal, hybrid recommendation system for a brand-agnostic fashion web app.  Our goal is to maximize relevance of “cold-start” recommendations—serving users with little or no purchase history—by combining image- and text-based embeddings (via CLIP), collaborative signals, and a Two-Tower neural ranking model.  We then deliver those recommendations in a React-based front end with low latency.

**Why does this matter?**  
Cold-start remains one of the biggest pain points in real-world recommenders: new users abandon sites when suggestions feel irrelevant.  By leveraging pretrained vision+language embeddings and a lightweight neural architecture, we can bootstrap personalization from product metadata alone, improving early engagement and ultimately conversion.

**What are your constraints?**  
- **Compute**: Precompute item embeddings offline; user tower inference must run in ≤ 50 ms on a single CPU core.  
- **Data**: We only have public image URLs and product descriptions from two open datasets (ASOS, Myntra).  
- **Privacy**: No user PII; we simulate interactions with synthetic personas.

**What data do you need?**  
- CLIP embeddings for all items (images + descriptions).  
- Synthetic user-item interactions derived from Mistral-generated persona ratings.  
- Cached item vectors in `recommendation_service.py`; live user vectors computed in `recommendation.py`.

**What could go wrong?**  
- CLIP embeddings might not capture fine-grained style nuances (fabric texture, cut).  
- Synthetic personas could introduce bias or fail to reflect real user diversity.  
- Latency spikes if React state management or API calls are mis-optimized.

---

## Technical Approach

### Mathematical formulation  
We learn two towers, \(f_u\) and \(f_i\), that map user and item feature vectors into a shared 128-dim embedding space.  For user \(u\) and item \(i\):
\[
\hat{y}_{ui} = g\bigl(\cos\bigl(f_u(x_u),\,f_i(x_i)\bigr)\bigr),\quad
\min_\theta \frac1N\sum_{(u,i)} \bigl(\hat{y}_{ui} - y_{ui}\bigr)^2
\]
where \(g\) is a small MLP (“rating_predictor”) mapping cosine similarity → predicted affinity, and \(y_{ui}\) are ground-truth ratings.

### Algorithm choice & justification  
- **CLIP embeddings** (via Hugging Face) provide strong multimodal features without retraining.  
- **Two-Tower architecture** (in `recommendation.py`) decouples user/item processing, enabling precompute of all item tower outputs.  
- **MSE loss** on synthetic ratings focuses the model on ranking fidelity for cold-start scenarios.

### PyTorch implementation strategy  
- Item embeddings: batch load image + text vectors, pass through item tower (`TwoTowerModel.item_tower`), cache to disk.  
- User embeddings: aggregate CLIP item vectors of a user’s rated items, pass through `user_tower`.  
- In `recommendation_service.py`, precompute all item tower outputs and serve cosine-similarity ranking in a single API call.  

### Validation methods  
- **Integration tests**: unit tests for `recommendation_service.py` endpoints and React components in `SelectionPage.jsx` / `ResultsPage.jsx`.  
- **Benchmark**: measure Python API latency and React render time with automated scripts.

### Resource requirements & constraints  
- **Browser**: modern Chrome/Firefox on desktop for React performance testing.

---

## Initial Results

**Evidence your implementation works**  
- React UI renders top-5 recommendations in 180 ms on average (Chrome network throttle disabled).

**Basic performance metrics**  
| Metric            | Value         |
|-------------------|--------------:|
| API latency (p99) | 45 ms         |
| Frontend render   | 180 ms        |

**Test case results**  
- Four test personas (minimal, casual, formal, athletic) each received sensible, style-aligned top-5 items in `ResultsPage.jsx`.  
- Unit tests for Flask endpoints passed (100% coverage on recommendation logic).

**Current limitations**  
- Synthetic personas may not fully mirror real user tastes.  
- CLIP embeddings ignore price/branded cues.  
- Cold-start for new items (no description or image) remains unhandled.

**Resource usage measurements**  
- Precompute item tower: ~10 minutes on GPU for 6 k items.  
- Serving API: 5 ms per request on average (Flask + cosine lookup).  
- React build bundle: 250 KB gzipped.

**Unexpected challenges**  
- Initial naive pre-aggregation of user embeddings led to cold cache thrashing; solved via in-memory context caching.  
- Mistral-generated ratings occasionally produced out-of-range scores (> 10) until we clipped them in prompt logic.

---
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


# Project Title & Team Members

**Project Title**: Decentralized Recommendation System  
**Team Members**: Matthew Kuo, Laura Li, Vivian Xiao, Megan Yang

---

# High-Level Summary

**Problem Statement**  
Current recommendation systems are siloed and specific to individual platforms, making it difficult for users to find similar products across websites. Our goal is to build a cross-platform fashion recommendation engine that uses product metadata (images, descriptions, price, brand) and limited user shopping history to generate personalized suggestions—especially for cold-start users.

**Approach**  
We implemented and evaluated four distinct recommendation strategies, supported by a custom dataset creation pipeline:

- **Synthetic User Dataset Creation**:  
  To simulate cold-start conditions and varied shopping behavior, we generated detailed user personas (e.g., "Alex, a 28-year-old designer into Scandinavian minimalism") using GPT via HuggingFace's `transformers` library. For each persona, we auto-generated user ratings (integers from -1 to 10, inclusive) on a subset of items using a style-aware prompting strategy, producing a sparse user-item ratings matrix that mimics real-world user preferences across styles and brands.

- **Product Dataset Assembly & Embedding**:  
  We compiled a dataset of ~3,000 clothing items from two sources:  
  - **ASOS Women’s Clothing** dataset (~30k items with image URLs, titles, prices)  
  - **Myntra Men’s Clothing** dataset (~60k items with structured metadata)  

  We preprocessed images and text to align formats across sources, and used OpenAI’s CLIP model to extract multimodal embeddings. Each item was represented as a fused vector of its image and description embeddings. These embeddings were later used in all downstream models.

- **Content-Based Filtering** using CLIP embeddings of image and text data
- **Collaborative Filtering** through comparisons with a synthetic personas dataset
- **Low-Rank Matrix Completion** with Burer-Monteiro factorization
- **Two-Tower Neural Network** with dual encoders for user and item embeddings

**Key Findings**
- Content filtering achieved high cold-start performance; was able to recommend similar items that the user has already liked/bought.
- The Two-Tower model provided the most personalized and semantically rich recommendations.
- Matrix completion effectively recovered sparse ratings using side info.
- Combining methods (e.g., hybrid or ensemble) improves robustness and user experience.

---

# Repository Structure Overview

├── src/                     # Final code scripts and modules for each model
├── evaluation.ipynb         # Interactive demo and experiment notebooks
├── webapp_demo/             # Web application demo
│   ├── backend/...
│   └── frontend/...
|   └── .gitignore
|   └── package-lock.json
├── report.md                # Final project report
├── requirements.txt         # Python dependencies
├── sparse_ratings_matrix.csv
├── dataset_m.xlsx           # Metadata for items (ASOS, Myntra)
├── content_filtering.ipynb  # Image/text embedding preprocessing
├── _archive/                # Old scripts, early experiments, exploratory notebooks
└── README.md                # Project summary and setup instructions

---

# Setup Instructions

1. **Environment Setup**
   - Recommended Python version: `>=3.10`
   - Make sure you are in the root directory (STAT-4830-VLLM-PROJECT)
   - Install dependencies:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt
     ```

2. **Dependencies**
   - `torch`, `pandas`, `numpy`, `scikit-learn`
   - `openai-clip` or `CLIP` via `torch.hub`
   - `matplotlib`, `seaborn`

3. **Additional Notes**
   - For image processing, ensure `Pillow` is installed.
   - If using Google Colab, GPU acceleration is highly recommended (Runtime > Change runtime type > GPU).

---

# Running the Code

### 1. Run 4 Models + Evaluation (`evaluation.ipynb`)
- To evaluate all four recommendation models (Content-Based, Collaborative Filtering, Low-Rank Matrix Completion, Two-Tower), open the notebook:
    ```bash
    jupyter notebook notebooks/evaluation.ipynb
    ```

### 2. Run Web App Demo
#### Prerequisites

- **Node.js & npm** (v14+)
- **Python 3.10+** (with `venv` or `virtualenv`)
- Internet connection to fetch NPM & PyPI packages

#### 1. Backend: Python (FastAPI)

1. Open a new terminal and create/activate a virtualenv:

    ```bash
    cd src/webapp_demo/backend
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Launch the FastAPI service (serves `/recommend/{method}` on port 8000):

    ```bash
    uvicorn recommendation_service:app --reload --port 8000
    ```

#### 2. Backend: Node (Express)

1. In a **second** terminal (no need to activate the Python venv here):

    ```bash
    cd webapp_demo/backend
    npm install
    ```

2. Start the Express server (serves `/api/items` and proxies to FastAPI on port 4000):

    ```bash
    npm start
    # logs: "Backend listening on http://localhost:4000"
    ```

#### 3. Frontend: React

1. In a **third** terminal:

    ```bash
    cd webapp_demo/frontend
    npm install
    ```

2. Run the React development server on port 3000:

    ```bash
    npm start
    ```

#### 4. Try it out

- Browse to **http://localhost:3000/**  
- Select **at least 15** images and click **Submit**  
- You’ll be redirected to `/results`, where you can view your picks and switch among the four algorithms’ recommendations.

---

#### Troubleshooting

- **`invalid json response`** in Express: make sure your Express proxy in `backend/index.js` is pointing to `http://localhost:8000`, not 3000.
- **Python ABI errors** (e.g. “numpy.dtype size changed”): recreate your venv and reinstall **both** NumPy and scikit-learn (or remove scikit-learn and use a NumPy‐only cosine‐sim).
- **React blank page**: ensure your `src/index.js` uses the React 18 `createRoot` API and is wrapped in `BrowserRouter`.

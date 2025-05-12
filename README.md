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

```
├── _archive/                # Old scripts, early experiments, exploratory notebooks
│   ├── llm_exploration/     # LLM Logs
│   ├── notebooks/           # Old scripts for 4 algorithms
│   ├── report_drafts/
│   ├── self_critique/
│   ├── slides/
├── dataset/                 # Folder to manually upload files from Google Drive
│   ├── images/              # Where image pngs are stored
│   ├── embeddings_final.csv        # Image/Text embeddings
│   ├── sparse_ratings_matrix.csv   # User ratings matrix (synthetic)
├── docs/
│   ├── STAT 4830 Final Presentation.pdf
│   ├── Final Critique.md
├── src/                     # Final code scripts and modules for each model
│   ├── evaluation.ipynb     # Interactive demo and experiment notebooks
│   ├── utils/...            # Contains collab filtering & two tower scripts
├── webapp_demo/             # Web application demo
│   ├── backend/...
│   └── frontend/...
|   └── .gitignore
|   └── package-lock.json
├── report.md                # Final project report
├── requirements.txt         # Python dependencies (for evaluation.ipynb)
└── README.md                # Project summary and setup instructions
```

---

# Setup Instructions
## 1. Python Environment(s)

We recommend using **two separate** virtual environments:

1. **Root venv** for the evaluation notebook  
2. **Backend venv** for the webapp_demo (FastAPI service)

### 1.1 Root venv (for `evaluation.ipynb`)
```
# from project root, do not cd yet
python3 -m venv .venv-root
source .venv-root/bin/activate
```
### 1.2 Backend venv
Open a new terminal:
```
cd src/webapp_demo/backend
python3 -m venv .venv-backend
source .venv-backend/bin/activate
```

## 2. Install Python Dependencies
### 2.1. In .venv-root (evaluation)
Go back to the first terminal:
```
# with .venv-root activated
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install jupyter
```
### 2.2. In .venv-backend (FastAPI backend)
```
# from src/webapp_demo/backend, .venv-backend activated
pip install --upgrade pip setuptools wheel
pip install numpy pandas scipy scikit-learn torch fastapi uvicorn
```
---

# Running the Code

## 1. Run 4 Models + Evaluation (`evaluation.ipynb`)
### 1.1 Activate .venv-root
```
source .venv-root/bin/activate
```
### 1.2. Launch Jupyter Lab
```
# make sure you are in src folder; venv-root activated
cd src
jupyter notebook evaluation.ipynb
```
### 1.3. Run evaluation.ipynb
In the notebook, run all cells. You’ll generate:
- Precision/Recall plots
- RMSE/MSE metrics

Press CTRL-C to quit the Jupyter Kernel. We used Jupyter instead of Google Colab as it allows us to more closely deal with our local environment (needed for our webapp). If desired, the evaluation notebook can also be run in Colab, provided that all the files are uploaded.

## 2. Run Web App Demo
### Prerequisites
The webapp has three moving parts:
- FastAPI (Python) on port 8000
- Express (Node) on port 4000
- React (frontend) on port 3000
You can start them in any order, but make sure each one is up before you use it.

### 2.1. Backend: Python (FastAPI)
```
# from src/webapp_demo/backend; venv-backend activated
source .venv-backend/bin/activate
uvicorn recommendation_service:app --reload --port 8000
```
You should see:
```
INFO: Uvicorn running on http://127.0.0.1:8000
```

### 2.2. Backend: Node (Express)
```
# in a new terminal (no Python venv needed here)
cd src/webapp_demo/backend
npm install
npm start
```
You should see:
```
Backend listening on http://localhost:4000
```

#### 3. Frontend: React
```
# in another terminal
cd src/webapp_demo/frontend
npm install
npm start
```
The app will open at http://localhost:3000/. It uses the proxy in package.json to talk to the Express backend on port 4000.

#### 4. Try it out

- Browse to **http://localhost:3000/**  
- Select **at least 15** images and click **Submit**  
- You’ll be redirected to `/results`, where you can view your picks and switch among the four algorithms’ recommendations.

---

#### Troubleshooting

- **`invalid json response`** in Express: make sure your Express proxy in `backend/index.js` is pointing to `http://localhost:8000`, not 3000.
- **Python ABI errors** (e.g. “numpy.dtype size changed”): recreate your venv and reinstall **both** NumPy and scikit-learn (or remove scikit-learn and use a NumPy‐only cosine‐sim).
- **React blank page**: ensure your `src/index.js` uses the React 18 `createRoot` API and is wrapped in `BrowserRouter`.
- Use `deactivate` to deactivate any .venv you currently have open.

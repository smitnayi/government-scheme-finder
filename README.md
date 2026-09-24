# Semantic Search Engine — Government Scheme Finder

An end-to-end NLP retrieval system for matching citizen natural-language life situations to Indian government welfare schemes using dense vector representations and cosine similarity.

---

## 📌 Key Highlights & Results

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
- **Corpus:** 3,400 official Indian government schemes from `updated_data.csv` with full details, benefits, eligibility, application processes, and documents across central ministries and 36 states/UTs.
- **Retrieval Performance:**
  - **Top-1 Accuracy:** **40.0%** (Semantic) vs **4.0%** (Keyword Baseline) — **10x improvement**.
  - **Recall@5:** **72.0%** (Semantic) vs **8.0%** (Keyword Baseline) — **9x improvement**.
- **Search Latency:** **~0.62 ms** per query on CPU using vectorized NumPy matrix operations.
- **Interactive UI:** Streamlit interface with state/category filters, rich scheme drawers, and side-by-side comparison mode.

---

## 📂 Project Structure

```text
├── app.py                      # Interactive Streamlit Web Application
├── build_index.py              # Offline text embedding & indexing script
├── search_engine.py            # Core vector retrieval & TF-IDF baseline search
├── evaluate.py                 # Quantitative evaluation harness (Top-1 & Recall@5)
├── faiss_benchmark.py          # Latency benchmark (NumPy BLAS vs FAISS)
├── eval_queries.json           # 25 curated test queries across 10 domains
├── requirements.txt            # Project dependencies
├── updated_data.csv            # Raw corpus (3,400 schemes with benefits & eligibility)
├── govt_schemes_dataset.csv    # Legacy corpus reference
├── embeddings.npy              # (Generated) Normalized 384-dim vector matrix
├── schemes_indexed.csv         # (Generated) Validated corpus with search_text
└── results/
    ├── eval_results.csv        # Tabular evaluation results
    └── comparison_plot.png     # Metric comparison visualization chart
```

---

## 🚀 Quickstart Guide

### Option A: One-Click Launcher (Easiest)
Simply double-click or run:
```powershell
.\run_app.bat
```

### Option B: Run via Virtual Environment directly
To avoid running with your system/global Python (which lacks project dependencies):
```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```
*(Or activate the virtual environment first with `.\venv\Scripts\activate`, then run `streamlit run app.py`)*

Open `http://localhost:8501` in your browser. Toggle **"Show keyword vs semantic comparison"** in the sidebar to observe the differences side-by-side.

### CLI Commands (via venv):
- **Run Search Test:**
  ```powershell
  .\venv\Scripts\python.exe search_engine.py
  ```
- **Run Evaluation Suite:**
  ```powershell
  .\venv\Scripts\python.exe evaluate.py
  ```
  *(Generates updated metrics in `results/eval_results.csv` and chart in `results/comparison_plot.png`)*
- **Run FAISS Scaling Benchmark:**
  ```powershell
  .\venv\Scripts\python.exe faiss_benchmark.py
  ```

---

## 🔬 Benchmark & Architecture Summary

Detailed documentation is available in:
- **[DOWNLOADS_REGISTRY.md](DOWNLOADS_REGISTRY.md):** Details every package, why it was chosen, and its specific role.
- **[DECISION_LOG.md](DECISION_LOG.md):** Complete record of design choices, including model selection, Windows encoding fixes, and vector search strategies.

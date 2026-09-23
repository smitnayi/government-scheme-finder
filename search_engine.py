"""
search_engine.py — Core search module (Phase 3 + Phase 5)

Provides:
  - semantic_search(): embed query → dot product → top-K
  - keyword_search(): word-overlap baseline for comparison
  - load_index(): load pre-built embeddings + metadata at startup

This file is the ONLINE component — it never re-embeds the dataset.
It only loads the pre-built embeddings.npy and schemes_indexed.csv.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re
from collections import Counter

# ── Index Loading ─────────────────────────────────────────────────────────────

_BASE_DIR = os.path.dirname(__file__)
_model = None
_embeddings = None
_df = None


def load_index():
    """Load embeddings and metadata from disk. Call once at startup."""
    global _model, _embeddings, _df

    embeddings_path = os.path.join(_BASE_DIR, "embeddings.npy")
    csv_path = os.path.join(_BASE_DIR, "schemes_indexed.csv")

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"embeddings.npy not found at {embeddings_path}. "
            "Run build_index.py first."
        )

    print("Loading search index...")
    _embeddings = np.load(embeddings_path)
    _df = pd.read_csv(csv_path)
    _model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"  Loaded {len(_df)} schemes, embeddings shape: {_embeddings.shape}")
    return _df


def get_metadata() -> pd.DataFrame:
    """Return the loaded metadata DataFrame."""
    if _df is None:
        load_index()
    return _df


def get_unique_states() -> list:
    """Return sorted list of unique states for filter dropdowns."""
    df = get_metadata()
    # States can be semicolon-separated; expand them
    all_states = set()
    for val in df["state"].dropna().unique():
        for s in val.split(";"):
            s = s.strip()
            if s:
                all_states.add(s)
    return sorted(all_states)


def get_unique_categories() -> list:
    """Return sorted list of unique categories for filter dropdowns."""
    df = get_metadata()
    all_cats = set()
    for val in df["category"].dropna().unique():
        for c in val.split(";"):
            c = c.strip()
            if c:
                all_cats.add(c)
    return sorted(all_cats)


# ── Semantic Search (Phase 3) ────────────────────────────────────────────────

def semantic_search(query: str, k: int = 5, state: str = None, category: str = None) -> pd.DataFrame:
    """
    Embed the query, compute cosine similarity against all schemes,
    apply optional hard filters, return top-K results with scores.

    Args:
        query: plain-language description of the user's situation
        k: number of results to return
        state: if set, only return schemes available in this state
        category: if set, only return schemes in this category

    Returns:
        DataFrame with top-K rows + a 'score' column, sorted descending.
    """
    if _model is None or _embeddings is None:
        load_index()

    # Encode query with same model → 384-dim normalized vector
    query_vec = _model.encode([query], normalize_embeddings=True)[0]

    # Dot product = cosine similarity (vectors are unit-length)
    scores = _embeddings @ query_vec  # shape: (num_schemes,)

    # Apply hard filters by zeroing out excluded rows
    if state:
        # A scheme matches if the user's state appears in its semicolon-separated state list,
        # OR if the scheme is marked "All" (nationwide)
        state_mask = _df["state"].fillna("").apply(
            lambda s: state in [x.strip() for x in s.split(";")] or "All" in [x.strip() for x in s.split(";")]
        ).values
        scores = np.where(state_mask, scores, -1.0)

    if category:
        cat_mask = _df["category"].fillna("").apply(
            lambda c: category in [x.strip() for x in c.split(";")]
        ).values
        scores = np.where(cat_mask, scores, -1.0)

    # Filter out excluded rows (-1.0)
    valid_mask = scores > -0.99
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        empty = _df.head(0).copy()
        empty["score"] = pd.Series(dtype=float)
        return empty

    # Top-K by descending score among valid candidates
    sorted_order = np.argsort(-scores[valid_indices])[:k]
    top_indices = valid_indices[sorted_order]
    results = _df.iloc[top_indices].copy()
    results["score"] = scores[top_indices]

    return results


# ── Keyword Search (Phase 5 — baseline) ──────────────────────────────────────

def _tokenize(text: str) -> list:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r'[a-z0-9]+', text.lower())


def keyword_search(query: str, k: int = 5, state: str = None, category: str = None) -> pd.DataFrame:
    """
    Naive keyword-overlap search: count how many query words appear in
    each scheme's search_text. Normalized by query length.

    This is the baseline that semantic search should beat.
    """
    if _df is None:
        load_index()

    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return _df.head(0)

    def overlap_score(text):
        doc_tokens = set(_tokenize(str(text)))
        return len(query_tokens & doc_tokens) / len(query_tokens)

    scores = _df["search_text"].apply(overlap_score).values

    # Apply same hard filters
    if state:
        state_mask = _df["state"].fillna("").apply(
            lambda s: state in [x.strip() for x in s.split(";")] or "All" in [x.strip() for x in s.split(";")]
        ).values
        scores = np.where(state_mask, scores, -1.0)

    if category:
        cat_mask = _df["category"].fillna("").apply(
            lambda c: category in [x.strip() for x in c.split(";")]
        ).values
        scores = np.where(cat_mask, scores, -1.0)

    # Filter out excluded rows (-1.0) and non-matching documents (0.0)
    valid_mask = scores > 0.0
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        empty = _df.head(0).copy()
        empty["score"] = pd.Series(dtype=float)
        return empty

    sorted_order = np.argsort(-scores[valid_indices])[:k]
    top_indices = valid_indices[sorted_order]
    results = _df.iloc[top_indices].copy()
    results["score"] = scores[top_indices]

    return results


# ── Quick CLI test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_index()

    test_queries = [
        "my father is a farmer and needs a loan for equipment",
        "scholarship for disabled students in college",
        "financial help for widow with school-going children",
    ]

    for q in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {q}")

        print(f"\n--- Semantic Search ---")
        results = semantic_search(q, k=3)
        for _, row in results.iterrows():
            print(f"  [{row['score']:.3f}] {row['scheme_name']}")

        print(f"\n--- Keyword Search ---")
        results = keyword_search(q, k=3)
        for _, row in results.iterrows():
            print(f"  [{row['score']:.3f}] {row['scheme_name']}")

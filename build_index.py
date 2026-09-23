"""
build_index.py — Offline indexing script (Phase 1 + Phase 2)

Run once to:
  1. Load govt_schemes_dataset.csv, validate schema
  2. Embed all search_text with all-MiniLM-L6-v2
  3. Save embeddings.npy + schemes_indexed.csv

Re-run only when the dataset changes.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import time

# ── Phase 1: Data Pipeline ────────────────────────────────────────────────────

DATA_FILE = os.path.join(os.path.dirname(__file__), "govt_schemes_dataset.csv")
EMBEDDINGS_FILE = os.path.join(os.path.dirname(__file__), "embeddings.npy")
INDEXED_CSV = os.path.join(os.path.dirname(__file__), "schemes_indexed.csv")
MODEL_NAME = "all-MiniLM-L6-v2"


def load_and_validate(path: str) -> pd.DataFrame:
    """Load CSV and validate that required columns exist."""
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"  Rows loaded: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Required columns for search + display
    required = ["scheme_name", "search_text", "state", "category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check search_text is not empty
    empty_count = df["search_text"].isna().sum()
    if empty_count > 0:
        print(f"  WARNING: {empty_count} rows have empty search_text — filling with scheme_name")
        df["search_text"] = df["search_text"].fillna(df["scheme_name"])

    # Show one sample
    print(f"\n  Sample search_text (row 0):")
    print(f"  {df['search_text'].iloc[0][:200]}...")

    return df


# ── Phase 2: Indexing ─────────────────────────────────────────────────────────

def build_embeddings(df: pd.DataFrame, model_name: str) -> np.ndarray:
    """Encode all search_text values and return normalized embeddings."""
    print(f"\nLoading model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = df["search_text"].tolist()
    print(f"Encoding {len(texts)} texts...")

    start = time.time()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,  # length-1 vectors → dot product = cosine similarity
        show_progress_bar=True,
        batch_size=64,
    )
    elapsed = time.time() - start

    print(f"  Done in {elapsed:.1f}s")
    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings


def save_index(df: pd.DataFrame, embeddings: np.ndarray):
    """Save embeddings and aligned metadata to disk."""
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"\nSaved embeddings to: {EMBEDDINGS_FILE}")

    df.to_csv(INDEXED_CSV, index=False)
    print(f"Saved indexed metadata to: {INDEXED_CSV}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Phase 1
    df = load_and_validate(DATA_FILE)

    # Phase 2
    embeddings = build_embeddings(df, MODEL_NAME)

    # Verify shape
    assert embeddings.shape == (len(df), 384), (
        f"Unexpected shape: {embeddings.shape}, expected ({len(df)}, 384)"
    )

    # Save
    save_index(df, embeddings)

    print("\n[OK] Index built successfully.")
    print(f"  {embeddings.shape[0]} schemes × {embeddings.shape[1]} dimensions")
    print(f"  embeddings.npy: {os.path.getsize(EMBEDDINGS_FILE) / 1e6:.1f} MB")

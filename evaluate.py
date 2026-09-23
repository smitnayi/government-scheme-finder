"""
evaluate.py — Evaluation harness (Phase 6)

Computes Top-1 accuracy and Recall@5 for keyword vs semantic search
on a hand-labeled eval set. Optionally compares two embedding models.

Usage:
    python evaluate.py                     # evaluate with default model
    python evaluate.py --compare-models    # also evaluate bge-base-en-v1.5
"""

import json
import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fix Windows cp1252 console encoding for scheme names containing symbols like the Rupee sign
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import search_engine

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)
EVAL_FILE = os.path.join(BASE_DIR, "eval_queries.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
K = 5  # Recall@K


# ── Evaluation Logic ─────────────────────────────────────────────────────────

def evaluate_method(method_fn, queries: list, k: int = 5) -> dict:
    """
    Run queries through a search method and compute metrics.

    Args:
        method_fn: callable(query, k) → DataFrame with 'scheme_id' column
        queries: list of dicts with 'query' and 'relevant_schemes'
        k: top-K for recall

    Returns:
        dict with 'top1_accuracy', 'recall_at_k', 'per_query' details
    """
    top1_hits = 0
    recall_hits = 0
    per_query = []

    for item in queries:
        query = item["query"]
        relevant = set(item["relevant_schemes"])

        results = method_fn(query, k=k)
        retrieved_ids = results["scheme_id"].tolist()

        # Top-1: is the first result in the relevant set?
        top1 = 1 if retrieved_ids and retrieved_ids[0] in relevant else 0

        # Recall@K: is any relevant scheme in the top-K?
        recall = 1 if any(rid in relevant for rid in retrieved_ids) else 0

        top1_hits += top1
        recall_hits += recall

        per_query.append({
            "query": query,
            "top1_hit": top1,
            "recall_hit": recall,
            "retrieved": retrieved_ids[:k],
            "relevant": list(relevant),
        })

    n = len(queries)
    return {
        "top1_accuracy": top1_hits / n if n > 0 else 0,
        "recall_at_k": recall_hits / n if n > 0 else 0,
        "per_query": per_query,
    }


# ── Auto-label: match eval queries to actual scheme_ids ──────────────────────

def auto_label_eval_queries(queries: list, k: int = 10) -> list:
    """
    Since the eval_queries.json has approximate scheme_ids,
    use semantic search to find the actual best-matching scheme_ids
    for each query. This helps build a more accurate eval set.

    Prints suggestions so the user can verify.
    """
    print("\n[AUTO-LABEL] Labeling eval queries against actual dataset...")
    print("   Review these and update eval_queries.json if needed:\n")

    for item in queries:
        results = search_engine.semantic_search(item["query"], k=k)
        top_ids = results["scheme_id"].tolist()
        top_names = results["scheme_name"].tolist()
        top_scores = results["score"].tolist()

        print(f"  Query: \"{item['query']}\"")
        print(f"  Current labels: {item['relevant_schemes']}")
        print("  Top matches:")
        for sid, name, score in zip(top_ids[:5], top_names[:5], top_scores[:5]):
            marker = " [HIT]" if sid in item["relevant_schemes"] else ""
            print(f"    [{score:.3f}] {sid} - {name}{marker}")
        print()

    return queries


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate search methods")
    parser.add_argument("--compare-models", action="store_true",
                        help="Also evaluate bge-base-en-v1.5")
    parser.add_argument("--auto-label", action="store_true",
                        help="Show suggested scheme_ids for each eval query")
    args = parser.parse_args()

    # Load eval queries
    with open(EVAL_FILE, "r") as f:
        queries = json.load(f)
    print(f"Loaded {len(queries)} eval queries from {EVAL_FILE}")

    # Load index
    search_engine.load_index()

    # Auto-label mode
    if args.auto_label:
        auto_label_eval_queries(queries)
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Evaluate keyword search ──────────────────────────────────────────
    print("\n[KW] Evaluating Keyword Search...")
    kw_results = evaluate_method(search_engine.keyword_search, queries, k=K)
    print(f"   Top-1 Accuracy: {kw_results['top1_accuracy']:.1%}")
    print(f"   Recall@{K}:      {kw_results['recall_at_k']:.1%}")

    # ── Evaluate semantic search (MiniLM) ────────────────────────────────
    print("\n[SEM] Evaluating Semantic Search (all-MiniLM-L6-v2)...")
    sem_results = evaluate_method(search_engine.semantic_search, queries, k=K)
    print(f"   Top-1 Accuracy: {sem_results['top1_accuracy']:.1%}")
    print(f"   Recall@{K}:      {sem_results['recall_at_k']:.1%}")

    # ── Results table ────────────────────────────────────────────────────
    results_data = {
        "Method": ["Keyword Search", "Semantic (MiniLM)"],
        "Top-1 Accuracy": [
            f"{kw_results['top1_accuracy']:.1%}",
            f"{sem_results['top1_accuracy']:.1%}",
        ],
        f"Recall@{K}": [
            f"{kw_results['recall_at_k']:.1%}",
            f"{sem_results['recall_at_k']:.1%}",
        ],
    }

    # ── Optional: second model comparison ────────────────────────────────
    if args.compare_models:
        print("\n[BGE] Evaluating Semantic Search (bge-base-en-v1.5)...")
        print("   Loading second model and re-encoding dataset...")

        # Re-encode with bge model
        bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        df = search_engine.get_metadata()
        bge_embeddings = bge_model.encode(
            df["search_text"].tolist(),
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=64,
        )

        def bge_search(query, k=5):
            q = bge_model.encode([query], normalize_embeddings=True)[0]
            scores = bge_embeddings @ q
            top = np.argsort(-scores)[:k]
            result = df.iloc[top].copy()
            result["score"] = scores[top]
            return result

        bge_results = evaluate_method(bge_search, queries, k=K)
        print(f"   Top-1 Accuracy: {bge_results['top1_accuracy']:.1%}")
        print(f"   Recall@{K}:      {bge_results['recall_at_k']:.1%}")

        results_data["Method"].append("Semantic (BGE)")
        results_data["Top-1 Accuracy"].append(f"{bge_results['top1_accuracy']:.1%}")
        results_data[f"Recall@{K}"].append(f"{bge_results['recall_at_k']:.1%}")

    # ── Save results ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(results_data)
    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print(f"{'='*50}")
    print(results_df.to_string(index=False))

    csv_path = os.path.join(RESULTS_DIR, "eval_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved to: {csv_path}")

    # ── Plot ─────────────────────────────────────────────────────────────
    methods = results_data["Method"]
    top1_vals = [float(v.strip('%')) / 100 for v in results_data["Top-1 Accuracy"]]
    recall_vals = [float(v.strip('%')) / 100 for v in results_data[f"Recall@{K}"]]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, top1_vals, width, label="Top-1 Accuracy", color="#2196F3")
    bars2 = ax.bar(x + width/2, recall_vals, width, label=f"Recall@{K}", color="#4CAF50")

    ax.set_ylabel("Score")
    ax.set_title("Keyword vs Semantic Search — Retrieval Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.bar_label(bars1, fmt="%.0f%%", label_type="edge")
    ax.bar_label(bars2, fmt="%.0f%%", label_type="edge")

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "comparison_plot.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot to: {plot_path}")
    plt.close()

    # ── Per-query breakdown ──────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("PER-QUERY BREAKDOWN (Semantic)")
    print(f"{'='*50}")
    for pq in sem_results["per_query"]:
        status = "[HIT]" if pq["recall_hit"] else "[MISS]"
        print(f"  {status} \"{pq['query'][:60]}...\"")
        if not pq["recall_hit"]:
            print(f"    Expected: {pq['relevant']}")
            print(f"    Got:      {pq['retrieved'][:3]}")


if __name__ == "__main__":
    main()

"""
faiss_benchmark.py — FAISS vs NumPy latency benchmark (Phase 7, optional)

Compares brute-force NumPy dot product against FAISS IndexFlatIP
over 1,000 random queries on the same 4,614 embeddings.

Expected finding: no measurable speed difference at this scale.
"""

import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("faiss-cpu not installed. Install with: pip install faiss-cpu")

BASE_DIR = os.path.dirname(__file__)
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings.npy")
NUM_QUERIES = 1000
K = 5


def benchmark_numpy(embeddings: np.ndarray, queries: np.ndarray, k: int) -> float:
    """Benchmark brute-force NumPy dot product search."""
    start = time.perf_counter()
    for q in queries:
        scores = embeddings @ q
        _ = np.argsort(-scores)[:k]
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_faiss(embeddings: np.ndarray, queries: np.ndarray, k: int) -> float:
    """Benchmark FAISS IndexFlatIP search."""
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))

    start = time.perf_counter()
    for q in queries:
        _, _ = index.search(q.reshape(1, -1).astype(np.float32), k)
    elapsed = time.perf_counter() - start
    return elapsed


def main():
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"embeddings.npy not found. Run build_index.py first.")
        return

    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"Loaded embeddings: {embeddings.shape}")

    # Generate random query vectors (normalized, same dimension)
    rng = np.random.default_rng(42)
    queries = rng.standard_normal((NUM_QUERIES, embeddings.shape[1]))
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries.astype(np.float32)
    embeddings = embeddings.astype(np.float32)

    print(f"\nBenchmarking {NUM_QUERIES} queries, top-{K}, over {embeddings.shape[0]} vectors...\n")

    # NumPy
    numpy_time = benchmark_numpy(embeddings, queries, K)
    numpy_per_query = numpy_time / NUM_QUERIES * 1000  # ms
    print(f"NumPy brute-force:  {numpy_time:.3f}s total  |  {numpy_per_query:.3f} ms/query")

    # FAISS
    if HAS_FAISS:
        faiss_time = benchmark_faiss(embeddings, queries, K)
        faiss_per_query = faiss_time / NUM_QUERIES * 1000
        print(f"FAISS IndexFlatIP:  {faiss_time:.3f}s total  |  {faiss_per_query:.3f} ms/query")

        speedup = numpy_time / faiss_time if faiss_time > 0 else float('inf')
        print(f"\nSpeedup: {speedup:.2f}x")

        if speedup < 1.5:
            print(f"\n[FINDING] At {embeddings.shape[0]} vectors, FAISS provides no meaningful")
            print(f"   speed advantage over plain NumPy. Both methods return results in")
            print(f"   under {max(numpy_per_query, faiss_per_query):.1f}ms per query.")
            print(f"   FAISS's indexing structures (HNSW, IVF) become beneficial at")
            print(f"   ~100K-1M+ vectors, where brute-force becomes impractical.")
        else:
            print(f"\n[FINDING] FAISS is {speedup:.1f}x faster than NumPy brute-force")
            print(f"   at this scale ({embeddings.shape[0]} vectors).")
    else:
        print("FAISS not available — skipping FAISS benchmark.")

    # Save finding
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "faiss_benchmark.txt"), "w") as f:
        f.write(f"FAISS vs NumPy Benchmark\n")
        f.write(f"========================\n")
        f.write(f"Dataset size: {embeddings.shape[0]} vectors × {embeddings.shape[1]} dimensions\n")
        f.write(f"Queries: {NUM_QUERIES}\n")
        f.write(f"Top-K: {K}\n\n")
        f.write(f"NumPy: {numpy_per_query:.3f} ms/query\n")
        if HAS_FAISS:
            f.write(f"FAISS: {faiss_per_query:.3f} ms/query\n")
            f.write(f"Speedup: {speedup:.2f}x\n")


if __name__ == "__main__":
    main()

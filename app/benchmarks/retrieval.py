import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from benchmarks.scifact import RetrievalDataset
from evaluation.retrieval_metrics import evaluate_rankings


# Turns a list of texts into a (len(texts), dimension) array
EncodeFn = Callable[[list[str]], np.ndarray]


@dataclass
class BenchmarkResult:
    metrics: dict[str, float]
    rankings: dict[str, list[str]]
    corpus_encode_seconds: float
    query_encode_seconds: float
    num_documents: int
    num_queries: int


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    return vectors / np.clip(norms, 1e-12, None)


def exact_search(
    query_vectors: np.ndarray,
    doc_vectors: np.ndarray,
    doc_ids: list[str],
    top_k: int,
) -> list[list[str]]:
    """
    Exact (brute-force) cosine search.

    The benchmark measures the *embedding model*, so it compares every
    query with every document. An approximate index such as HNSW could
    miss a few true neighbours and blur the comparison between models.
    """

    scores = normalize(query_vectors) @ normalize(doc_vectors).T

    top_k = min(top_k, len(doc_ids))

    # argpartition finds the top k without fully sorting every row
    top = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]

    rankings = []

    for row, candidates in enumerate(top):
        ordered = candidates[np.argsort(-scores[row, candidates])]
        rankings.append([doc_ids[i] for i in ordered])

    return rankings


def run_retrieval_benchmark(
    dataset: RetrievalDataset,
    encode: EncodeFn,
    top_k: int = 100,
) -> BenchmarkResult:

    doc_ids = list(dataset.corpus)
    query_ids = list(dataset.queries)

    start = time.perf_counter()
    doc_vectors = encode([dataset.corpus[d] for d in doc_ids])
    corpus_encode_seconds = time.perf_counter() - start

    start = time.perf_counter()
    query_vectors = encode([dataset.queries[q] for q in query_ids])
    query_encode_seconds = time.perf_counter() - start

    ranked = exact_search(query_vectors, doc_vectors, doc_ids, top_k)

    rankings = dict(zip(query_ids, ranked))

    return BenchmarkResult(
        metrics=evaluate_rankings(rankings, dataset.qrels),
        rankings=rankings,
        corpus_encode_seconds=corpus_encode_seconds,
        query_encode_seconds=query_encode_seconds,
        num_documents=len(doc_ids),
        num_queries=len(query_ids),
    )

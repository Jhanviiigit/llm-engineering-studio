import numpy as np

from benchmarks.retrieval import exact_search, run_retrieval_benchmark
from benchmarks.scifact import RetrievalDataset


VECTORS = {
    "apples are red": [1.0, 0.0],
    "the sky is blue": [0.0, 1.0],
    "cherries are red": [0.9, 0.1],
    "red fruit": [1.0, 0.05],
    "blue things": [0.05, 1.0],
}


def fake_encode(texts):
    return np.array([VECTORS[t] for t in texts])


def test_exact_search_orders_by_cosine_similarity():

    doc_vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])

    rankings = exact_search(
        np.array([[1.0, 0.0]]),
        doc_vectors,
        ["apples", "sky", "cherries"],
        top_k=3,
    )

    assert rankings == [["apples", "cherries", "sky"]]


def test_exact_search_top_k_larger_than_corpus():

    rankings = exact_search(
        np.array([[1.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        ["a", "b"],
        top_k=100,
    )

    assert rankings == [["a", "b"]]


def test_run_retrieval_benchmark():

    dataset = RetrievalDataset(
        corpus={
            "d1": "apples are red",
            "d2": "the sky is blue",
            "d3": "cherries are red",
        },
        queries={
            "q1": "red fruit",
            "q2": "blue things",
        },
        qrels={
            "q1": {"d1", "d3"},
            "q2": {"d2"},
        },
    )

    result = run_retrieval_benchmark(dataset, fake_encode, top_k=3)

    assert result.rankings["q2"][0] == "d2"
    assert result.metrics["recall@5"] == 1.0
    assert result.metrics["ndcg@10"] == 1.0
    assert result.num_documents == 3
    assert result.num_queries == 2

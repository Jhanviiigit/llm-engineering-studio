import math

import pytest

from evaluation.retrieval_metrics import (
    evaluate_rankings,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_recall_at_k():

    ranked = ["a", "b", "c", "d"]
    relevant = {"b", "d"}

    assert recall_at_k(ranked, relevant, 1) == 0.0
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 4) == 1.0


def test_reciprocal_rank():

    assert reciprocal_rank_at_k(["a", "b", "c"], {"a"}, 10) == 1.0
    assert reciprocal_rank_at_k(["a", "b", "c"], {"c"}, 10) == pytest.approx(1 / 3)
    assert reciprocal_rank_at_k(["a", "b", "c"], {"c"}, 2) == 0.0
    assert reciprocal_rank_at_k(["a", "b", "c"], {"z"}, 10) == 0.0


def test_ndcg_perfect_ranking_is_one():

    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, 10) == pytest.approx(1.0)


def test_ndcg_hand_computed():

    # One relevant doc at position 2: DCG = 1/log2(3), ideal = 1/log2(2) = 1
    assert ndcg_at_k(["x", "a"], {"a"}, 10) == pytest.approx(1 / math.log2(3))

    # Two relevant docs at positions 1 and 3
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)

    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, 10) == pytest.approx(dcg / idcg)


def test_ndcg_nothing_found_is_zero():

    assert ndcg_at_k(["x", "y"], {"a"}, 10) == 0.0


def test_evaluate_rankings_averages_over_queries():

    rankings = {
        "q1": ["a", "b"],
        "q2": ["x", "c"],
    }

    qrels = {
        "q1": {"a"},
        "q2": {"c"},
    }

    results = evaluate_rankings(rankings, qrels, recall_ks=(1, 2))

    assert results["recall@1"] == 0.5
    assert results["recall@2"] == 1.0
    assert results["mrr@10"] == pytest.approx((1.0 + 0.5) / 2)


def test_missing_ranking_counts_as_zero():

    results = evaluate_rankings({}, {"q1": {"a"}}, recall_ks=(1,))

    assert results["recall@1"] == 0.0
    assert results["ndcg@10"] == 0.0

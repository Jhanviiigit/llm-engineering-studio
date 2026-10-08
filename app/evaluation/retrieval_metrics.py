"""
Information-retrieval metrics.

All functions take, for one query:
    ranked:   document ids in the order the system returned them
    relevant: the set of document ids that are actually correct
"""

import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """
    Fraction of the relevant documents that appear in the top k.
    """

    if not relevant:
        return 0.0

    hits = len(set(ranked[:k]) & relevant)

    return hits / len(relevant)


def reciprocal_rank_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """
    1 / position of the first relevant document, or 0 if none is in the
    top k. Rewards putting a correct document at the very top.
    """

    for position, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / position

    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain with binary relevance.

    Each relevant document found scores 1 / log2(position + 1), so hits
    near the top count more. Dividing by the best possible score (all
    relevant documents ranked first) puts the result between 0 and 1.
    """

    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, doc_id in enumerate(ranked[:k], start=1)
        if doc_id in relevant
    )

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1.0 / math.log2(position + 1)
        for position in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_rankings(
    rankings: dict[str, list[str]],
    qrels: dict[str, set[str]],
    recall_ks: tuple[int, ...] = (1, 5, 10, 100),
    cutoff: int = 10,
) -> dict[str, float]:
    """
    Average each metric over all queries that have relevance labels.

    rankings: query id -> ranked document ids
    qrels:    query id -> relevant document ids ("query relevance judgements")
    """

    query_ids = [query_id for query_id in qrels if qrels[query_id]]

    if not query_ids:
        raise ValueError("No queries with relevance judgements")

    def mean(values):
        return sum(values) / len(values)

    results = {}

    for k in recall_ks:
        results[f"recall@{k}"] = mean([
            recall_at_k(rankings.get(q, []), qrels[q], k)
            for q in query_ids
        ])

    results[f"mrr@{cutoff}"] = mean([
        reciprocal_rank_at_k(rankings.get(q, []), qrels[q], cutoff)
        for q in query_ids
    ])

    results[f"ndcg@{cutoff}"] = mean([
        ndcg_at_k(rankings.get(q, []), qrels[q], cutoff)
        for q in query_ids
    ])

    return results

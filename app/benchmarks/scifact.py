"""
SciFact retrieval dataset (from the BEIR benchmark).

5,183 scientific abstracts and 1,109 claims ("queries"). Each claim in the
train and test splits is labelled with the abstract(s) that support or
refute it. Files are downloaded from Hugging Face on first use and cached.
"""

from dataclasses import dataclass

import pandas as pd
from huggingface_hub import hf_hub_download


CORPUS_REPO = "BeIR/scifact"
QRELS_REPO = "BeIR/scifact-qrels"


@dataclass
class RetrievalDataset:
    corpus: dict[str, str]        # document id -> text (title + abstract)
    queries: dict[str, str]       # query id -> query text
    qrels: dict[str, set[str]]    # query id -> relevant document ids


def _download(repo_id: str, filename: str) -> str:
    return hf_hub_download(repo_id, filename, repo_type="dataset")


def load_scifact(split: str = "test") -> RetrievalDataset:
    """
    Load the corpus and the queries of one split ("train" or "test").
    """

    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")

    corpus_df = pd.read_parquet(
        _download(CORPUS_REPO, "corpus/corpus-00000-of-00001.parquet")
    )
    queries_df = pd.read_parquet(
        _download(CORPUS_REPO, "queries/queries-00000-of-00001.parquet")
    )
    qrels_df = pd.read_csv(
        _download(QRELS_REPO, f"{split}.tsv"),
        sep="\t",
        dtype=str,
    )

    # Title + abstract, as in the BEIR benchmark
    corpus = {
        str(row["_id"]): f"{row['title']}. {row['text']}".strip()
        for _, row in corpus_df.iterrows()
    }

    qrels: dict[str, set[str]] = {}

    for _, row in qrels_df.iterrows():
        if int(row["score"]) > 0:
            qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])

    all_queries = {
        str(row["_id"]): row["text"]
        for _, row in queries_df.iterrows()
    }

    # Only the queries that belong to this split
    queries = {
        query_id: all_queries[query_id]
        for query_id in qrels
    }

    return RetrievalDataset(corpus=corpus, queries=queries, qrels=qrels)

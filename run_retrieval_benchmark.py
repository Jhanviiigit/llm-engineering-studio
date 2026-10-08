"""
Measure how well an embedding model retrieves the right documents.

    python run_retrieval_benchmark.py
    python run_retrieval_benchmark.py --model models/my-finetuned-model

Uses the SciFact test split (300 queries, 5,183 abstracts) and saves the
results to output/benchmarks/. No LLM calls are made.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Make app/ importable
sys.path.insert(0, str(Path(__file__).parent / "app"))

from sentence_transformers import SentenceTransformer  # noqa: E402

from benchmarks.retrieval import run_retrieval_benchmark  # noqa: E402
from benchmarks.scifact import load_scifact  # noqa: E402


OUTPUT_DIR = Path("output/benchmarks")


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print(f"Loading SciFact ({args.split} split)...")
    dataset = load_scifact(args.split)

    print(f"Loading model {args.model}...")
    model = SentenceTransformer(args.model)

    def encode(texts):
        return model.encode(
            texts,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        )

    print(
        f"Encoding {len(dataset.corpus)} documents and "
        f"{len(dataset.queries)} queries..."
    )
    result = run_retrieval_benchmark(dataset, encode)

    print("\n" + "=" * 44)
    print(f"Retrieval benchmark: SciFact {args.split}")
    print(f"Model: {args.model}")
    print("=" * 44)

    for metric, value in result.metrics.items():
        print(f"{metric:<12} {value:.4f}")

    print("-" * 44)
    print(f"Corpus encoding : {result.corpus_encode_seconds:.1f}s")
    print(
        "Query encoding  : "
        f"{1000 * result.query_encode_seconds / result.num_queries:.1f} ms/query"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = Path(args.model).name
    output_path = OUTPUT_DIR / f"retrieval_{model_slug}_{args.split}_{timestamp}.json"

    output_path.write_text(
        json.dumps(
            {
                "dataset": "scifact",
                "split": args.split,
                "model": args.model,
                "num_documents": result.num_documents,
                "num_queries": result.num_queries,
                "metrics": result.metrics,
                "corpus_encode_seconds": result.corpus_encode_seconds,
                "query_encode_seconds": result.query_encode_seconds,
                "rankings": {
                    query_id: ranked[:10]
                    for query_id, ranked in result.rankings.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

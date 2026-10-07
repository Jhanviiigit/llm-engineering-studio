# System Design — LLM Engineering Studio

This document describes how the system works today, where it is going, and
*why* each design decision was made. It is updated at the end of every phase.

- **Status:** Phase 1 in progress
- **Target cloud:** Google Cloud Platform (GCP)
- **Constraint:** no GPU — all model training must run on CPU or free Colab

---

## 1. What the system does

A user asks a question about a document collection. The system:

1. **Retrieves** the most relevant chunks of text (semantic search over embeddings)
2. **Generates** an answer with an LLM, using only those chunks as context
3. **Evaluates** the answer (groundedness, relevance, completeness, conciseness,
   correctness) using a second LLM call as a judge

The same components are used offline to run **experiments** — e.g. comparing
retrieval depth (`top_k`) across a benchmark of questions.

---

## 2. Current architecture (v0 — single process)

```text
┌──────────────────────── one Python process ────────────────────────┐
│                                                                    │
│  Streamlit UI / CLI                                                │
│        │                                                           │
│        ▼                                                           │
│  RAGService ──────────────► LLMClient ──────► OpenRouter (HTTP)     │
│        │                       ▲                                   │
│        ▼                       │                                   │
│  Retriever                rag_evaluator (LLM-as-judge)             │
│        │                                                           │
│        ▼                                                           │
│  VectorStore (Python list, in memory)                              │
│        ▲                                                           │
│        │  at startup: load → chunk → embed (MiniLM, local CPU)     │
│  data/documents/sample.txt                                         │
└────────────────────────────────────────────────────────────────────┘
```

### Request path (one question)

| Step | Where | Rough cost |
|---|---|---|
| Embed query | local CPU (MiniLM, 384-dim) | ~10–50 ms |
| Similarity search | NumPy, brute force over all chunks | O(N) — fine for small N |
| Generate answer | OpenRouter LLM | ~1–5 s, tokens |
| Evaluate answer | OpenRouter LLM (2nd call) | ~1–5 s, tokens |

**Observation:** the LLM calls dominate latency, and evaluation *doubles* it.

### Limitations of v0

| # | Limitation | Consequence | Design concept |
|---|---|---|---|
| L1 | Vector store lives in process memory | Lost on restart; each replica has its own copy; can't scale horizontally | Stateless services, persistence |
| L2 | Documents are re-embedded on every startup | Startup time grows with the corpus; wasted compute | Offline vs online paths |
| L3 | Ingestion and querying share one process | A large upload blocks users | Background workers, queues |
| L4 | Evaluation runs inside the user request | 2× latency and cost per question | Async processing |
| L5 | Brute-force similarity search | O(N) per query; slow at millions of chunks | ANN indexes (HNSW / IVF) |
| L6 | Fixed-size character chunking | Chunks split mid-word/sentence, hurting retrieval | Data quality |
| L7 | No logs, metrics or traces | Can't see failures, latency or cost in production | Observability |

---

## 3. Phase 1 changes (done in this branch)

These changes don't alter behaviour for the user, but they make the system
*ready* to be split into cloud services.

| Change | Why |
|---|---|
| `app/config.py` — all settings read from environment variables | 12-factor config: the same container image runs locally and on Cloud Run; secrets are injected, never baked in |
| LLM client created lazily (`get_client()`), injected into services | Modules no longer need an API key just to be imported. Tests (and CI) run without secrets. Dependencies can be swapped (e.g. a different LLM provider) |
| `LLMClient.chat` raises `LLMError` instead of returning `"Error: ..."` as text | Failures were indistinguishable from answers — and the judge would even score an error message. Explicit errors can be shown, retried, and alerted on |
| Evaluation failures return `{"error": ...}` instead of 0.0 scores | Zero scores silently dragged down benchmark averages, making a parsing bug look like a quality regression |
| `RAGService.answer_question(..., evaluate=False)` | First step toward moving evaluation off the request path (L4) |
| `app/bootstrap.py` — one composition root | `create_rag_service()` was duplicated in 3 files. Now swapping the vector store for pgvector is a one-file change |

### Step 3: API service (FastAPI)

```text
 Before                                  After
 ──────                                  ─────
 Streamlit process                       Streamlit (thin client)
  ├─ embedding model                         │  HTTP: POST /ask, POST /documents
  ├─ vector store                            ▼
  └─ LLM client                          FastAPI service
                                          ├─ embedding model   (loaded once at startup)
                                          ├─ vector store
                                          └─ LLM client
```

| Decision | Why |
|---|---|
| UI talks to the API over HTTP | UI and API can be deployed, scaled and restarted independently. Any client (scripts, other apps, the eval runner) can use the same API |
| Models loaded once in FastAPI's `lifespan` | Loading MiniLM takes seconds; doing it per request would dominate latency |
| Endpoints are `def`, not `async def` | The LLM SDK and embedding model are blocking. FastAPI runs `def` endpoints in a thread pool; a blocking call inside `async def` would freeze every request |
| `threading.Lock` in `VectorStore` | Thread-pool requests run concurrently: a search could otherwise read the lists while an upload is half-way through extending them |
| `/ask` defaults to `evaluate=false` | Keeps the common path to one LLM call (L4); callers opt in to evaluation |
| Pydantic request/response models | Invalid input (empty question, `top_k` out of range) is rejected with 422 before any work is done, and the schema becomes the API contract shown at `/docs` |
| Upstream LLM failure → **502 Bad Gateway** | Distinguishes "our service is broken" (500) from "a dependency failed" (502), which matters for alerting |
| Upload limits (1 MB, `.txt`, UTF-8) | Never trust client input size or format |

**Still in memory:** uploaded documents are lost when the API restarts, and two
API replicas would each have different documents. Step 4 (pgvector) fixes this.

### Step 4: Persistent vector store (Postgres + pgvector)

```text
documents                              chunks
─────────                              ──────
id            BIGSERIAL PK  ◄──┐       id           BIGSERIAL PK
source        TEXT             └────── document_id  FK → documents (ON DELETE CASCADE)
content_hash  TEXT UNIQUE              chunk_index  INT
created_at    TIMESTAMPTZ              content      TEXT
                                       embedding    vector(384)   ← HNSW index (cosine)
```

| Decision | Why |
|---|---|
| `PgVectorStore` has the same methods as `VectorStore` | Nothing above the store changed. `bootstrap.py` picks one: Postgres when `DATABASE_URL` is set, in memory otherwise (tests, quick runs) |
| Documents identified by a SHA-256 **content hash** with a `UNIQUE` constraint | Re-indexing is idempotent: restarting the app no longer re-embeds `sample.txt` (fixes L2), and uploading the same file twice adds nothing. The constraint (not just a check in Python) makes this safe under concurrent uploads |
| `has_document()` checked *before* embedding | Skips the expensive part (running the embedding model) for known documents |
| A document and its chunks are inserted in **one transaction** | A crash mid-upload can't leave a document with half its chunks |
| **HNSW** index with `vector_cosine_ops` | Approximate nearest-neighbour search in ~log(N) instead of scanning every chunk (fixes L5). Same cosine metric as the in-memory store, so results are comparable |
| **Connection pool** (`psycopg_pool`, max 10) | Opening a Postgres connection costs milliseconds and server memory; reusing them keeps request latency low. Cloud SQL also limits total connections, so the pool caps usage per instance |
| Embedding dimension checked at startup | If the embedding model changes (e.g. after fine-tuning in Phase 2 to a different size), the app fails with a clear message instead of on the first insert |
| Separate `studio_test` database, and tests refuse to run on anything else | Integration tests truncate tables; they must never touch real data |

**Fixed by this step:** L1 (data lost on restart; replicas can now share one
database), L2 (no re-embedding at startup), L5 (ANN index).

---

## 4. Target architecture (v1 — on GCP)

```text
                    GitHub ──push──► GitHub Actions: test → build image → deploy
                                                                │
                                                                ▼
                                                        Artifact Registry
                                                                │
   Browser                                                      ▼
     │                                         ┌──────── Cloud Run services ────────┐
     ▼                                         │                                    │
 Streamlit UI ──── HTTPS ───► FastAPI  /ask ───┼──► Cloud SQL Postgres + pgvector   │
 (Cloud Run)                  (Cloud Run,      │      chunks, embeddings (HNSW),    │
                               autoscaled,     │      documents, eval results       │
                               stateless)      │                                    │
                                │   │          │                                    │
                     /documents │   └─► OpenRouter LLM                              │
                                ▼              │                                    │
                         Cloud Storage ──event─┼──► Ingestion worker                │
                         (raw documents)       │    chunk → embed → write pgvector  │
                                               │                                    │
                     /ask publishes ──► Pub/Sub ──► Eval worker (LLM judge)          │
                                               │    writes scores to Postgres       │
                                               └────────────────────────────────────┘
               Secret Manager (API keys)   ·   Cloud Logging / Monitoring / Trace
```

### Key decisions

**D1 — Postgres + pgvector instead of a dedicated vector database.**
One database stores documents, chunks, embeddings *and* evaluation results, so
they can be joined and kept consistent in a single transaction. pgvector
supports HNSW indexes, which are fast enough well beyond this project's scale.
A dedicated vector DB (Pinecone, Weaviate, Qdrant) becomes worth it at very
large scale or when you need features Postgres lacks.
*Trade-off:* Cloud SQL has a fixed monthly cost even when idle.

**D2 — Cloud Run for all services.**
Runs a container, scales to zero when unused (low cost for a portfolio project),
scales out automatically under load. Requires services to be **stateless** —
which is exactly why L1 must be fixed first.
*Trade-off:* cold starts — loading the embedding model adds a few seconds to
the first request after scale-to-zero.

**D3 — Separate the online path from the offline path.**
- *Online* (latency-sensitive): embed query → search → generate → respond.
- *Offline* (throughput-sensitive): ingest documents, run evaluation, train models.
They scale differently, fail differently, and should not slow each other down.

**D4 — Asynchronous evaluation via Pub/Sub.**
`/ask` returns the answer immediately and publishes an event; an eval worker
scores it later. The user sees answers ~2× faster, and evaluation can be
sampled (e.g. score 10% of traffic) to control cost.
*Trade-off:* scores are eventually consistent — not available at response time.

**D5 — Secrets in Secret Manager, config in environment variables.**
Nothing sensitive in the image or the repo.

---

## 5. Deep learning roadmap (Phase 2)

All of these are measured with the existing evaluation harness, so every model
change has a before/after number.

| Model | Technique | Problem it addresses |
|---|---|---|
| Retrieval benchmark | recall@k, MRR on synthetic Q→passage pairs | We currently measure answers, not retrieval itself |
| Fine-tuned embeddings | Contrastive learning (MultipleNegativesRankingLoss) on MiniLM | Generic embeddings miss domain vocabulary |
| Cross-encoder reranker | Retrieve 20 → rerank → keep top 3 | The `top_k` trade-off: more context improved groundedness but hurt completeness |
| Distilled judge | Train a DistilBERT classifier on LLM-judge labels | LLM-as-judge is slow and costly; a small model can score every request |

---

## 6. Scaling notes

- **Read path** scales horizontally: stateless API replicas behind Cloud Run's
  load balancer; the database becomes the bottleneck → add read replicas,
  connection pooling, and a cache for repeated questions.
- **Embedding model** is loaded once per container instance; more instances
  means more memory, not more downloads (bake the model into the image).
- **LLM provider** is the slowest and most failure-prone dependency → timeouts,
  retries with backoff, and a fallback model.
- **Cost** is dominated by LLM tokens, then Cloud SQL. Track tokens per request
  (already done) and aggregate them in monitoring.

---

## 7. Glossary

- **Stateless service** — keeps no data between requests in memory; any replica can serve any request.
- **Composition root** — the one place where objects are constructed and wired together.
- **ANN / HNSW** — approximate nearest-neighbour search; a graph index that finds similar vectors in ~log(N) time instead of scanning all N.
- **Eventual consistency** — data becomes correct *eventually*, not instantly (e.g. evaluation scores appear a few seconds after the answer).
- **Cold start** — extra latency when a scaled-to-zero service starts a new instance.
- **recall@k** — fraction of questions where the correct passage appears in the top k retrieved results.
- **MRR** — mean reciprocal rank; rewards putting the correct passage near the top.

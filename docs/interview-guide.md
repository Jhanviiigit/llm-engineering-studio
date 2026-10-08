# Interview Guide — LLM Engineering Studio

How to explain this project in a technical interview: what was built, **why**
each decision was made, what the alternatives were, and what went wrong along
the way. Written in the first person so it can be practised out loud.

> Companion to [system-design.md](system-design.md), which is the reference
> architecture. This guide is about *explaining* it. It is updated with every
> step of the project.

**Contents**

1. [The pitch](#1-the-pitch)
2. [Architecture walkthrough](#2-architecture-walkthrough)
3. [Decision log](#3-decision-log)
4. [Debugging stories](#4-debugging-stories)
5. [Concepts to know cold](#5-concepts-to-know-cold)
6. [Likely interview questions](#6-likely-interview-questions)
7. [Limitations and what I'd do next](#7-limitations-and-what-id-do-next)

---

## 1. The pitch

### 30 seconds

> I built a retrieval-augmented generation platform for experimenting with
> and evaluating LLM systems. It answers questions over a document collection,
> scores its own answers with an LLM judge, and measures retrieval quality on a
> standard benchmark. I took it from a single-process prototype to a
> containerised service: a FastAPI backend, Postgres with pgvector for
> persistent vector search, Docker, and CI on GitHub Actions. I'm now
> fine-tuning the retrieval models and measuring each change against a
> validated baseline.

### 2 minutes: add the *why*

> The prototype kept everything in one Python process: the embedding model,
> an in-memory list of vectors, and the LLM calls. That works on a laptop but
> can't be deployed. Restarting lost all data, every replica would have its own
> copy, and documents were re-embedded on every startup.
>
> So I separated the stateless parts from the stateful parts. The API is
> stateless and can scale horizontally; state lives in Postgres with pgvector,
> using an HNSW index for approximate nearest-neighbour search. Documents are
> deduplicated by content hash, so ingestion is idempotent.
>
> On the ML side, I found my answer-quality metric couldn't tell retrieval
> failures from generation failures, so I built an LLM-free retrieval
> benchmark on SciFact. My baseline reproduced the published nDCG@10 (0.648
> versus about 0.645), which validated the evaluation code. The benchmark showed
> recall@100 was 0.93 but recall@10 only 0.79: the right document is usually
> found but ranked too low. That points directly at a reranker, which is what
> I'm building.

**Why this pitch works:** it states a problem, the engineering response, and a
*measured* result, then ends on a data-driven next step.

---

## 2. Architecture walkthrough

```text
 Streamlit UI ──HTTP──► FastAPI  ──► Postgres + pgvector (documents, chunks, HNSW index)
 (thin client)          │  │
                        │  └──► LLM (OpenRouter, OpenAI-compatible API)
                        └──► Embedding model (MiniLM, loaded once per process)
```

**One request, end to end (`POST /ask`)**

1. Pydantic validates the request (question 1–2000 chars, `top_k` 1–10) → 422 if invalid.
2. The question is embedded by MiniLM → a 384-dimensional vector (~5 ms on CPU).
3. pgvector finds the closest chunks with `ORDER BY embedding <=> query LIMIT k`,
   using the HNSW index (`<=>` is cosine distance).
4. The chunks are put into a prompt that tells the LLM to answer **only** from context.
5. The LLM call goes to OpenRouter (1–5 s; this dominates latency).
6. Optionally, a second LLM call (the judge) scores the answer. It's off by default because it doubles latency and cost.
7. If the LLM fails, the API returns 502 Bad Gateway, which separates "a dependency failed" from "we are broken" (500).

**Where the time goes:** the LLM call, by far. Embedding takes ~5 ms and the
vector search a few ms; generation takes seconds. So optimisations target
the LLM path first: async evaluation, caching, choosing a faster model.

---

## 3. Decision log

Each entry: **problem → decision → why → alternatives → trade-off**.
Interviewers care most about *why* and *trade-offs*.

### Phase 1: making it deployable

#### D1. Lazy, injected LLM client

- **Problem:** `LLMClient()` was created when modules were imported, so importing the evaluator without an API key crashed. **Three test files couldn't even load**, which would have broken CI.
- **Decision:** create the client on first use (`get_client()`) and pass it into services as a parameter (dependency injection).
- **Why:** tests can pass in a fake client. No secrets are needed to run tests, and the LLM provider can be swapped in one place.
- **Alternative:** monkeypatching module globals in tests. It's fragile, depends on import order, and hides the dependency.

#### D2. Fail loudly: raise `LLMError` instead of returning error text

- **Problem:** on failure the client returned `"Error: ..."` as if it were the answer. The UI showed it as an answer, and the judge would *score the error message*.
- **Decision:** raise a typed exception; the API maps it to HTTP 502.
- **Why:** failures you can't tell apart from successes can't be alerted on, retried or counted.

#### D3. Failed evaluations return `{"error": ...}`, not zero scores

- **Problem:** when the judge's JSON couldn't be parsed, the code returned 0.0 for every metric, and those zeros were **averaged into benchmark results**.
- **Why it matters:** a parsing bug would look exactly like a quality regression. Someone could roll back a good change because of it.
- **Principle:** missing data is not the same as a zero. Exclude it, and count it separately.

#### D4. 12-factor configuration (`app/config.py`)

- **Decision:** every setting comes from an environment variable, with defaults.
- **Why:** the same container image runs locally (values from `.env`) and on Cloud Run (values injected by the platform, secrets from Secret Manager). Nothing environment-specific is baked into code or images.

#### D5. FastAPI service + thin UI client

- **Decision:** move all logic behind an HTTP API; Streamlit only makes HTTP calls.
- **Why:** the two can be deployed, scaled and restarted independently, and any client (scripts, the eval runner, other apps) can use the same API. The UI image needs no PyTorch.
- **Trade-off:** one more network hop and two processes to run locally. Docker Compose solves the second.

#### D6. Sync (`def`) endpoints, not `async def`

- **Why:** the OpenAI SDK and the embedding model are **blocking**. FastAPI runs `def` endpoints in a thread pool, so a slow request doesn't block others. A blocking call inside `async def` would freeze the event loop and stall *every* request.
- **Interview point:** `async` only helps if everything awaited is non-blocking. Mixing them is a classic performance bug.

#### D7. A lock in the in-memory vector store

- **Problem:** with a thread pool, an upload could extend the document list while a search was reading it. The search could then see `documents` and `embeddings` at different lengths and return wrong results or crash.
- **Decision:** a `threading.Lock` around writes and a snapshot read.
- **Interview point:** concurrency bugs appear as soon as you move from a script to a server.

#### D8. Postgres + pgvector instead of a dedicated vector database

- **Why:** one database for documents, chunks, embeddings and later evaluation results. That means joins, transactions and one system to operate. pgvector's HNSW index is fast well beyond this project's scale.
- **Alternatives:** Pinecone, Weaviate, Qdrant, Milvus. They're worth it at very large scale or for features Postgres lacks (built-in hybrid search, multi-tenancy at scale).
- **Trade-off:** Cloud SQL costs money even when idle; a managed vector DB may have a free tier.

#### D9. Content-hash deduplication (idempotent ingestion)

- **Decision:** each document is identified by the SHA-256 of its text, enforced by a `UNIQUE` constraint.
- **Why:** restarting no longer re-embeds the corpus. Measured: **0.273 s → 0.002 s** for the sample document, and the saving grows with corpus size. Uploading a duplicate is a no-op.
- **Why the constraint and not just a check in Python:** two identical uploads arriving at once would both pass the Python check. The database constraint is the only race-free guard, so `INSERT ... ON CONFLICT DO NOTHING RETURNING id` tells us atomically who won.
- **Optimisation:** a cheap `has_document()` check runs *before* embedding, to skip the expensive step.

#### D10. One transaction per document

- **Why:** a crash halfway through an upload can't leave a document with half its chunks stored.

#### D11. HNSW index (approximate nearest neighbour)

- **What:** a layered graph of vectors. Search starts at a coarse layer and greedily moves to closer neighbours at finer layers, giving roughly O(log N) instead of O(N).
- **Trade-off:** it's *approximate* and can miss a true neighbour. You tune recall against speed with `ef_search`. That's why the **benchmark uses exact search**: to measure the model, not the index.
- **Alternative:** IVFFlat (clusters vectors, searches the nearest clusters). It builds faster but needs training data and usually has lower recall at the same speed.

#### D12. Connection pool

- **Why:** opening a Postgres connection costs milliseconds and server memory. A pool (max 10) reuses connections and caps how many each API instance opens. Cloud SQL has a connection limit, and N instances × 10 must stay under it.

#### D13. Embedding dimension checked at startup

- **Why:** if the embedding model changes size (for example after fine-tuning), the app fails at startup with a clear message instead of on the first insert in production.

#### D14. Docker image choices

- **CPU-only PyTorch:** the default Linux wheel bundles several GB of CUDA libraries that are useless without a GPU.
- **Dependencies before code:** Docker caches layers, so changing code rebuilds in seconds.
- **Model downloaded at build time:** new containers (Cloud Run scaling up) serve immediately, with no network fetch on cold start.
- **Non-root user, `$PORT`, `.dockerignore` excludes `.env`:** security basics. Secrets must never be baked into an image.

#### D15. CI with a real database

- **Decision:** GitHub Actions starts a Postgres + pgvector service container and runs every test, including the database integration tests that are skipped locally without a database.
- **Safety:** the integration tests truncate tables, so they refuse to run against any database whose name doesn't end in `_test`.

### Phase 2: deep learning

#### D16. A separate, LLM-free retrieval benchmark

- **Problem:** the LLM judge scores the final answer. A bad answer could come from bad *retrieval* or bad *generation*, and the judge can't tell which. It's also noisy (different judge models gave different scores) and expensive (free tier: 50 requests per day).
- **Decision:** measure retrieval separately on **SciFact**, which has human relevance labels.
- **Why SciFact:** labels are included, so no LLM is needed. It has **separate train and test splits**, so we can fine-tune without leaking test data. It's small enough for CPU (5,183 docs) and has published baselines to validate against.
- **Principle:** evaluate components in isolation before evaluating the whole pipeline.

#### D17. Validating the evaluation code against a published number

- **Result:** our nDCG@10 for `all-MiniLM-L6-v2` = **0.648** versus about **0.645** published.
- **Why it matters:** a bug in metric code silently invalidates every experiment. Reproducing a known result first is how you earn trust in later numbers.

#### D18. Exact search in the benchmark, HNSW in production

- See D11. Different goals: the benchmark isolates *model quality*; production optimises *latency*.

#### D19. Reading the baseline to choose the next step

- recall@10 = 0.79 but recall@100 = 0.93: the correct document is usually **retrieved** but **ranked too low**.
- This justifies a **two-stage retrieval** design: a fast bi-encoder recalls 100 candidates, and a slower, more accurate cross-encoder re-ranks them.
- **Interview point:** let the data pick the next experiment instead of trying things at random.

---

## 4. Debugging stories

Use the **STAR** format (Situation, Task, Action, Result). Interviewers love
specific, verified root causes.

### Story 1: the "free" model router sent requests to a safety classifier

- **Situation:** after deploying the fix for evaluation errors, evaluation still failed about 1 time in 6.
- **Action:** instead of guessing, I called the judge six times in a row and logged the raw responses. One reply was literally `"User Safety: safe"`. The `openrouter/free` alias picks a random free model, and sometimes picked `nemotron-3.5-content-safety`, a classifier and not a chat model.
- **Result:** pinned a specific model (4/4 successes in testing) and started **recording the model that actually served each request**, so routing problems are visible in future.
- **Lesson:** an abstraction that hides *which* dependency you're using also hides why it fails. Log what actually happened, not what you asked for.

### Story 2: reasoning models returned empty answers

- **Situation:** some responses were empty, yet one reply used 519 tokens and returned no text.
- **Root cause:** reasoning models spend tokens "thinking" before answering. With `max_tokens=500` they ran out before writing anything; `finish_reason` was `length`.
- **Action:** an empty response now raises an error that includes `finish_reason`; limits were raised (1500 default, 2000 for the judge).

### Story 3: the judge's JSON was a different shape each time

- **Situation:** with the same prompt, different models returned code-fenced JSON, capitalised keys (`Groundedness`), or bare numbers.
- **Problem:** capitalised keys would have been counted as a *different metric* in benchmark averages, silently splitting the data.
- **Action:** put the exact JSON schema in the prompt, and wrote a normalising parser (lowercase keys, accept numbers or objects, extract the outermost `{...}`). Tests cover each format.
- **Lesson:** treat LLM output as untrusted input: specify it, validate it, normalise it.

### Story 4: the Docker build crashed with `operator torchvision::nms does not exist`

- **Root cause:** `torchvision` was installed from PyPI, built against CUDA PyTorch, while the image used CPU-only PyTorch. Version mismatch.
- **Deeper cause:** `requirements.txt` came from `pip freeze`, which captures *everything* in the environment, including unused packages. Nothing in the project uses `torchvision`.
- **Action:** removed it.
- **Lesson:** frozen dependency lists drift. Separate the packages you depend on directly from pinned lock files.

### Story 5: smaller ones worth mentioning

- `requirements.txt` was **UTF-16** (from `pip freeze >` in PowerShell); GitHub displayed it as binary.
- **Rate limits:** the free tier allows 50 LLM requests per day. Ad-hoc live testing used most of them in one session. This is why the benchmark is LLM-free and the unit tests use fake clients.
- **Line endings:** added `.gitattributes` so Dockerfiles and SQL keep LF line endings on Windows checkouts.

---

## 5. Concepts to know cold

**RAG (retrieval-augmented generation):** retrieve relevant text, put it in
the prompt, and have the LLM answer from it. It reduces hallucination and
lets the model use private or recent data without retraining.

**Embedding:** a vector representing the meaning of a text. MiniLM produces
384 numbers. Texts with similar meanings get vectors pointing in similar directions.

**Cosine similarity:** the angle between two vectors, ignoring their length.
1 means the same direction and 0 means unrelated. pgvector's `<=>` returns
cosine *distance* (1 − similarity).

**Bi-encoder vs cross-encoder:**

| | Bi-encoder (MiniLM) | Cross-encoder (reranker) |
|---|---|---|
| Input | Query and document encoded **separately** | Query and document **together** |
| Speed | Documents pre-computed; one query encoding | One model run **per (query, document) pair** |
| Accuracy | Lower: can't compare words across the two texts | Higher: attention sees both texts at once |
| Use | Search over millions → top 100 | Re-rank those 100 → top 5 |

**Contrastive learning:** train with (query, matching document) pairs. Pull
each query towards its document and push it away from the other documents in
the batch (**in-batch negatives**). Loss: softmax cross-entropy over the
similarities, also called InfoNCE (`MultipleNegativesRankingLoss` in
sentence-transformers). Bigger batches mean more negatives, usually a better
model. **Hard negatives** (wrong documents that *look* relevant) teach the most.

**Metrics:**

- **recall@k:** the fraction of relevant documents found in the top k. "Did we find it?"
- **MRR@k:** the average of 1/rank of the first relevant result. "How high was the first hit?"
- **nDCG@k:** gives credit for each relevant result, discounted by position (1/log2(rank+1)), normalised by the best possible score. The standard metric for retrieval benchmarks.

**HNSW:** see D11. **Idempotency:** doing an operation twice has the same
effect as doing it once. See D9.

**Stateless service:** keeps no data between requests in memory, so any
replica can serve any request. That's a requirement for autoscaling on Cloud Run.

**LLM-as-judge:** use an LLM to score outputs. It's cheap compared with
humans but **noisy**: scores varied between judge models in this project
(conciseness 0.7–1.0 for the same answer). It needs a fixed judge model and
calibration against human labels before you trust small differences.

**Overfitting:** a model memorises its training data and gets worse on new
data. Defences: a held-out test set the model never sees, few training
epochs, and comparing against the baseline.

---

## 6. Likely interview questions

**"Why not just use a bigger LLM instead of improving retrieval?"**
If the right document isn't in the context, no LLM can answer correctly
without hallucinating. Retrieval sets the ceiling on answer quality. It's
also far cheaper to improve: a 22M-parameter embedding model versus a
billions-parameter LLM.

**"How would you scale this to 10 million documents?"**
The API is stateless, so add replicas. The bottleneck becomes the database:
tune HNSW (`m`, `ef_construction`, `ef_search`), add read replicas and
connection pooling (PgBouncer). Ingestion moves to a queue with background
workers, so uploads don't block queries. At much larger scale, consider a
dedicated vector database, or partitioning the index.

**"How do you know your evaluation is correct?"**
I reproduced a published baseline (nDCG@10 of 0.648 versus about 0.645)
before trusting any new numbers, and the metric functions have unit tests
against hand-computed values.

**"What's the difference between your two evaluations?"**
The retrieval benchmark measures whether we found the right documents,
using human labels, deterministically and for free. The LLM judge measures
whether the final answer is good: it covers generation, but it's noisy and
costs requests. You need both, because they catch different failures.

**"Why Postgres and not Pinecone?"**
See D8: one system, transactions and joins, good enough at this scale. I'd
revisit at hundreds of millions of vectors or if I needed managed
multi-tenancy.

**"How do you handle an unreliable LLM provider?"**
Typed errors mapped to 502; recording which model actually served each
request; pinning a model rather than a router. Next steps are timeouts,
retries with exponential backoff for 429 and 5xx responses, and a fallback model.

**"What would you do differently?"**
Generate `requirements.txt` from a list of direct dependencies rather than
`pip freeze`; build the retrieval benchmark *before* the LLM judge; budget
LLM calls for testing from day one.

---

## 7. Limitations and what I'd do next

Being upfront about limitations signals seniority.

- **The API image is 2.65 GB**, because it installs UI libraries too. Splitting API-only requirements would cut Cloud Run cold-start time.
- **The chunker splits by character count**, mid-sentence. Sentence-aware or token-based chunking should improve retrieval.
- **MiniLM truncates at 256 tokens**, and the median SciFact abstract is about 300 tokens, so the tail of longer abstracts is ignored.
- **The LLM judge is uncalibrated:** it hasn't been compared with human ratings, and scores vary by judge model.
- **No retries, timeouts or caching** on LLM calls yet.
- **Evaluation runs synchronously** when requested; the plan is Pub/Sub plus a worker (D4 in the design doc).

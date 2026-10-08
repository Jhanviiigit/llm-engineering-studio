# LLM Engineering Studio

A modular Python platform for experimenting with Large Language Models (LLMs), prompt engineering, Retrieval-Augmented Generation (RAG), structured outputs, benchmarking, and evaluation.

The project is designed as an experimentation environment where LLM components can be developed, tested, evaluated, and compared systematically.

## Current Features

### LLM Operations

- Translation
- Paraphrasing
- Temperature-based experimentation
- OpenAI-compatible LLM client
- Response parsing
- Latency and token tracking

### Retrieval-Augmented Generation (RAG)

- Document loading
- Text chunking
- Sentence-transformer embeddings
- Vector storage
- Similarity-based retrieval
- Configurable `top_k` retrieval
- Context-aware answer generation
- Retrieved-context inspection

### LLM Evaluation

RAG responses are evaluated across five dimensions:

- **Groundedness** — whether the response is supported by retrieved context
- **Relevance** — whether the response directly addresses the question
- **Completeness** — whether important information is included
- **Conciseness** — whether the response avoids unnecessary information
- **Correctness** — whether the response agrees with a reference answer

The project includes:

- A controlled RAG evaluation dataset
- Batch evaluation
- Automatic metric aggregation
- Retrieval-depth experiments
- Mock-based unit tests
- Structured evaluation outputs

### Structured Outputs

The evaluation pipeline requests structured JSON responses from the LLM using the OpenAI-compatible `response_format` interface.

This allows evaluation results to be parsed programmatically into individual metrics rather than relying only on free-form model output.

### Web Interface

A Streamlit-based web interface provides an interactive way to:

- Ask questions about the document collection
- Control the RAG `top_k` retrieval parameter
- View generated answers
- Inspect retrieved context
- View evaluation metrics
- View latency and token usage

## Architecture

See [docs/system-design.md](docs/system-design.md) for the full system design: current architecture, limitations, the target GCP architecture, and the reasoning behind each decision.

```text
                         Streamlit UI
                              |
                              v
                         RAG Service
                              |
                    +---------+---------+
                    |                   |
                    v                   v
                Retriever             LLM
                    |
                    v
               Vector Store
                    |
                    v
                Embeddings
                    |
                    v
                Documents

                         |
                         v
                    RAG Response
                         |
                         v
                  LLM Evaluation
                         |
        +----------------+----------------+
        |        |        |        |      |
        v        v        v        v      v
   Grounded  Relevance Completeness Conciseness Correctness
````

## RAG Pipeline

```text
Documents
    |
    v
Document Loader
    |
    v
Text Chunking
    |
    v
Embeddings
    |
    v
Vector Store
    |
    v
Retriever
    |
    v
RAG Service
    |
    v
LLM
    |
    v
Generated Answer
    |
    v
Evaluation
```

## RAG Retrieval Experiment

An initial controlled experiment was performed using 5 evaluation questions while varying retrieval depth (`top_k`).

| Metric       | top_k = 1 | top_k = 3 |
| ------------ | --------: | --------: |
| Groundedness |      0.95 |      1.00 |
| Relevance    |      1.00 |      1.00 |
| Completeness |      0.90 |      0.70 |
| Conciseness  |      0.95 |      1.00 |
| Correctness  |      0.80 |      0.80 |

These results are from a small 5-case controlled benchmark and are intended for engineering comparison rather than as a general accuracy claim.

The experiment showed that increasing retrieval depth did not automatically improve every quality metric. In this evaluation, `top_k = 3` improved groundedness but reduced completeness, while correctness remained unchanged.

## Project Structure

```text
llm-engineering-studio/
│
├── app/
│   ├── evaluation/
│   │   ├── batch_evaluator.py
│   │   ├── case_loader.py
│   │   └── rag_evaluator.py
│   │
│   ├── llm/
│   │   └── client.py
│   │
│   ├── parsers/
│   │   └── response_parser.py
│   │
│   ├── prompts/
│   │   ├── paraphrase.py
│   │   ├── rag.py
│   │   ├── story.py
│   │   └── translation.py
│   │
│   ├── rag/
│   │   ├── chunker.py
│   │   ├── document_loader.py
│   │   ├── embeddings.py
│   │   ├── indexer.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   │
│   ├── services/
│   │   ├── paraphrase_service.py
│   │   ├── rag_service.py
│   │   ├── temperature_services.py
│   │   └── translation_service.py
│   │
│   ├── utils/
│   │   └── file_manager.py
│   │
│   └── main.py
│
├── data/
│   ├── documents/
│   │   └── sample.txt
│   │
│   └── evaluation/
│       └── rag_test_cases.json
│
├── tests/
│   ├── test_batch_evaluator.py
│   ├── test_case_loader.py
│   ├── test_chunker.py
│   ├── test_document_loader.py
│   ├── test_embeddings.py
│   ├── test_indexer.py
│   ├── test_rag_evaluator.py
│   ├── test_rag_prompt.py
│   ├── test_rag_service.py
│   ├── test_retriever.py
│   └── test_vector_store.py
│
├── output/
├── run_evaluation.py
├── streamlit_app.py
├── pytest.ini
├── requirements.txt
└── README.md
```

## Testing

The project uses `pytest` for automated testing.

Current test suite:

```text
38 passed, 4 skipped (database tests run when TEST_DATABASE_URL is set)
```

Tests cover:

* Document loading
* Text chunking
* Embeddings
* Indexing
* Retrieval
* Vector storage
* RAG prompting
* RAG service
* RAG evaluation
* Evaluation dataset loading
* Batch evaluation

LLM-dependent components are tested using mocked clients where appropriate, avoiding unnecessary API calls during unit testing.

## Configuration

Copy `.env.example` to `.env` and set your OpenRouter API key:

```text
OPENROUTER_API_KEY=your_api_key_here
```

All settings are read from environment variables in `app/config.py`. Unit tests do not need an API key.

## Running the Application

Activate the virtual environment and run:

```bash
python app/main.py
```

The terminal application currently provides:

```text
1. Translate
2. Paraphrase
3. Temperature Experiment
4. RAG Question Answering
5. Exit
```

## Database (Postgres + pgvector)

By default, documents are stored in memory and lost when the API stops. To keep them, run Postgres with the pgvector extension in Docker:

```bash
docker compose up -d
```

Then add this line to `.env`:

```text
DATABASE_URL=postgresql://studio:studio@localhost:5432/studio
```

Run the database integration tests (they use a separate `studio_test` database):

```powershell
$env:TEST_DATABASE_URL="postgresql://studio:studio@localhost:5432/studio_test"
python -m pytest tests/test_pg_vector_store.py
```

## Running the API and Web Interface

The web interface is a thin client that calls the FastAPI service, so start both.

**Terminal 1: API**

```bash
uvicorn api.main:app --app-dir app --reload
```

Interactive API docs: http://localhost:8000/docs

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `GET` | `/health` | Liveness check and number of indexed chunks |
| `POST` | `/ask` | Answer a question (`question`, `top_k`, optional `evaluate`) |
| `POST` | `/documents` | Upload a `.txt` document and index it |

**Terminal 2: Streamlit UI**

```bash
streamlit run streamlit_app.py
```

The browser interface allows users to:

* Submit RAG questions
* Adjust `top_k`
* Turn LLM-judge evaluation on or off
* Upload `.txt` documents
* Inspect retrieved context
* View evaluation metrics, latency, token usage and the model used

Set `API_URL` if the API is not running on `http://localhost:8000`.

## Running Tests

```bash
python -m pytest
```

## Running the RAG Evaluation

The benchmark runner can be executed with:

```powershell
$env:PYTHONPATH="app"
python run_evaluation.py
```

The evaluation cases are stored in:

```text
data/evaluation/rag_test_cases.json
```

## Technology Stack

* Python
* Pytest
* NumPy
* PyTorch
* Sentence Transformers
* FastAPI
* PostgreSQL + pgvector
* Docker Compose
* Streamlit
* OpenAI-compatible APIs
* OpenRouter
* Structured JSON Outputs
* Git
* GitHub

## Engineering Practices

* Modular separation of LLM, RAG, evaluation, service, and prompt components
* Unit testing with pytest
* Mocking of LLM clients for deterministic tests
* Controlled parameter experiments
* Structured evaluation outputs
* Latency and token tracking
* Environment variables for API credentials
* `.gitignore` protection for secrets and generated files

## Future Work

Planned extensions include:

* Summarization
* Sentiment Analysis
* Additional parameter benchmarking
* Larger evaluation datasets
* LangChain workflows
* More robust evaluation and monitoring
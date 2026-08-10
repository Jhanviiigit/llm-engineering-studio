
# LLM Engineering Studio

A modular Python platform for experimenting with Large Language Models (LLMs), prompt engineering, Retrieval-Augmented Generation (RAG), benchmarking, and evaluation.

The project is designed as a learning and experimentation environment where LLM components can be developed, tested, evaluated, and compared systematically.

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
- Context-aware answer generation
- Retrieved-context display

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

## Architecture

```text
Documents
    |
    v
Document Loader
    |
    v
Chunking
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
Response Parser
    |
    v
Evaluation
    |
    +--> Groundedness
    +--> Relevance
    +--> Completeness
    +--> Conciseness
    +--> Correctness
````

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
│   ├── rag/
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── indexer.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   │
│   ├── services/
│   │   ├── paraphrase_service.py
│   │   ├── rag_service.py
│   │   └── translation_service.py
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
│
├── output/
│
├── run_evaluation.py
├── pytest.ini
└── README.md
```

## Testing

The project uses `pytest` for automated testing.

Current test suite:

```text
14 passed
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

## Running the Application

Activate the virtual environment and run:

```bash
python app/main.py
```

The application currently provides:

```text
1. Translate
2. Paraphrase
3. Temperature Experiment
4. RAG Question Answering
5. Exit
```

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
* Sentence Transformers
* NumPy
* OpenAI-compatible APIs
* OpenRouter
* Git

## Future Work

Planned extensions include:

* Summarization
* Sentiment Analysis
* Structured output experiments
* Additional parameter benchmarking
* Larger evaluation datasets
* FastAPI backend
* Streamlit interface
* LangChain workflows
* Additional vector database integrations

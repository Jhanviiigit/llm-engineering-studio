"""
HTTP API for LLM Engineering Studio.

Run locally from the project root:

    uvicorn api.main:app --app-dir app --reload

Interactive docs are served at http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile

from api.schemas import (
    AskRequest,
    AskResponse,
    DocumentResponse,
    HealthResponse,
)
from bootstrap import Services, create_services
from llm.client import LLMError


logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 1_000_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model and build the index once at startup,
    # not on every request.
    app.state.services = create_services()
    yield
    # Release database connections on shutdown
    app.state.services.close()


app = FastAPI(
    title="LLM Engineering Studio API",
    version="0.1.0",
    lifespan=lifespan,
)


def get_services(request: Request) -> Services:
    return request.app.state.services


# Endpoints are plain `def` (not `async def`) because the LLM client and
# embedding model are blocking. FastAPI runs them in a thread pool so one
# slow request does not block the others.


@app.get("/health", response_model=HealthResponse)
def health(services: Services = Depends(get_services)):
    return HealthResponse(
        status="ok",
        total_chunks=services.vector_store.count(),
    )


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    services: Services = Depends(get_services),
):
    try:
        result = services.rag_service.answer_question(
            request.question,
            top_k=request.top_k,
            reference_answer=request.reference_answer,
            evaluate=request.evaluate,
        )
    except LLMError as e:
        logger.warning("LLM request failed: %s", e)

        # 502 Bad Gateway: our service is fine, an upstream one failed
        raise HTTPException(status_code=502, detail=str(e)) from e

    evaluation = result.get("evaluation") or {}

    return AskResponse(
        answer=result["response"],
        retrieved_context=result["retrieved_context"],
        model=result["model"],
        temperature=result["temperature"],
        latency_seconds=result["latency"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        total_tokens=result["total_tokens"],
        evaluation={
            metric: details
            for metric, details in evaluation.items()
            if isinstance(details, dict)
        } or None,
        evaluation_error=evaluation.get("error"),
    )


@app.post("/documents", response_model=DocumentResponse, status_code=201)
def upload_document(
    file: UploadFile,
    services: Services = Depends(get_services),
):
    if not (file.filename or "").lower().endswith(".txt"):
        raise HTTPException(
            status_code=415,
            detail="Only .txt files are supported.",
        )

    content = file.file.read(MAX_UPLOAD_BYTES + 1)

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than {MAX_UPLOAD_BYTES} bytes.",
        )

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail="File must be UTF-8 text.",
        ) from e

    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty.")

    chunks_added = services.indexer.index_text(
        text,
        source=file.filename,
    )

    return DocumentResponse(
        filename=file.filename,
        chunks_added=chunks_added,
        already_indexed=chunks_added == 0,
        total_chunks=services.vector_store.count(),
    )

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)
    evaluate: bool = Field(
        default=False,
        description=(
            "Also score the answer with the LLM judge. "
            "Roughly doubles latency and cost."
        ),
    )
    reference_answer: str | None = None


class MetricScore(BaseModel):
    score: float
    reason: str


class AskResponse(BaseModel):
    answer: str
    retrieved_context: list[str]
    model: str
    temperature: float
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    evaluation: dict[str, MetricScore] | None = None
    evaluation_error: str | None = None


class DocumentResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    total_chunks: int

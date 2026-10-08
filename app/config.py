import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """
    Application settings read from environment variables.

    Keeping configuration in one place means the same code can run
    locally (values from .env) and in the cloud (values injected by
    the platform, e.g. Cloud Run + Secret Manager) without changes.
    """

    openrouter_api_key: str | None
    llm_base_url: str
    llm_model: str
    embedding_model: str
    documents_path: str
    # Postgres + pgvector when set, otherwise an in-memory vector store
    database_url: str | None


def get_settings() -> Settings:
    return Settings(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        llm_base_url=os.getenv(
            "LLM_BASE_URL",
            "https://openrouter.ai/api/v1"
        ),
        llm_model=os.getenv(
            "LLM_MODEL",
            "nvidia/nemotron-3-super-120b-a12b:free"
        ),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "all-MiniLM-L6-v2"
        ),
        documents_path=os.getenv(
            "DOCUMENTS_PATH",
            "data/documents/sample.txt"
        ),
        database_url=os.getenv("DATABASE_URL"),
    )

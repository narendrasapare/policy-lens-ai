from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: Literal["ollama", "gemini"] = "ollama"
    ollama_model: str = "qwen3:4b-instruct"
    ollama_base_url: str = "http://127.0.0.1:11434"
    allow_cloud_llm: bool = False
    google_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.8-flash"

    reranking_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    rerank_candidate_count: int = Field(default=15, ge=1, le=20)
    rerank_log_enabled: bool = True

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
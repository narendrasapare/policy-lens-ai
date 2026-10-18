from langchain_core.language_models.chat_models import BaseChatModel

from policylens.core.config import Settings
from policylens.services.answer_generation import GeminiConfigurationError


def create_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_predict=768,
            num_ctx=8192,
        )

    if not settings.allow_cloud_llm:
        raise GeminiConfigurationError(
            "Cloud LLM calls are disabled. Set ALLOW_CLOUD_LLM=true "
            "only when you intentionally want to use Gemini."
        )

    api_key = (
        settings.google_api_key.get_secret_value()
        if settings.google_api_key is not None
        else ""
    )

    if not api_key.strip():
        raise GeminiConfigurationError(
            "Google Gemini API key is not configured."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        api_key=api_key,
        temperature=0,
        max_retries=1,
    )
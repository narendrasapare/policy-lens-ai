from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from policylens.api.dependencies import get_rag_service
from policylens.repositories.faiss_vector_repository import VectorStoreError
from policylens.schemas.chat import ChatRequest, ChatResponse
from policylens.services.answer_generation import (
    AnswerGenerationError,
    GeminiConfigurationError,
)
from policylens.services.rag_service import RagService
from policylens.services.reranking import RerankingError
from policylens.services.reranking_evidence import RerankingEvidenceError
from policylens.services.semantic_search import NoIndexedDocumentsError


router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
def ask_policy_question(
    request: ChatRequest,
    http_request: Request,
    rag_service: Annotated[RagService, Depends(get_rag_service)],
) -> ChatResponse:
    jobs = getattr(http_request.app.state, "indexing_jobs", None)
    if jobs is not None and jobs.has_pending():
        raise HTTPException(status_code=409, detail="Your documents are still being prepared. Please try again shortly.")
    try:
        return rag_service.answer_question(
            question=request.question,
            top_k=request.top_k,
        )
    except (NoIndexedDocumentsError, VectorStoreError) as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    except GeminiConfigurationError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except RerankingError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except RerankingEvidenceError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        ) from error
    except AnswerGenerationError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(error),
        ) from error
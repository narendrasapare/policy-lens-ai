from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from policylens.api.dependencies import get_semantic_search_service
from policylens.repositories.faiss_vector_repository import VectorStoreError
from policylens.schemas.search import SemanticSearchRequest, SemanticSearchResponse
from policylens.services.semantic_search import (
    NoIndexedDocumentsError,
    SemanticSearchService,
)


router = APIRouter(prefix="/api/v1/search", tags=["Search"])


@router.post("", response_model=SemanticSearchResponse)
def search_documents(
    request: SemanticSearchRequest,
    search_service: Annotated[
        SemanticSearchService,
        Depends(get_semantic_search_service),
    ],
) -> SemanticSearchResponse:
    try:
        return search_service.search(
            query=request.query,
            top_k=request.top_k,
        )
    except NoIndexedDocumentsError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    except VectorStoreError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
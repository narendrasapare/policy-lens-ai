import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from dotenv import dotenv_values
from pathlib import Path

from policylens.api.dependencies import DATA_DIRECTORY, get_document_indexing_service, get_vector_repository
from policylens.services.indexing_jobs import IndexingJobs
from policylens.api.routers.workspace import router as workspace_router

from fastapi.middleware.cors import CORSMiddleware

from policylens.api.routers.chat import router as chat_router
from policylens.api.routers.documents import router as documents_router
from policylens.api.routers.health_checks import router as health_router
from policylens.api.routers.search import router as search_router
from policylens.api.routers.document_files import router as document_files_router


@asynccontextmanager
async def lifespan(app):
    jobs = IndexingJobs(get_document_indexing_service)
    app.state.indexing_jobs = jobs
    try:
        # Recover documents whose preparation was interrupted by a restart.
        for path in (DATA_DIRECTORY / "processed").glob("*.json"):
            if len(path.stem) == 32 and not get_vector_repository().get_document_status(path.stem)[0]:
                jobs.enqueue(path.stem)
        yield
    finally:
        jobs.shutdown()
        del app.state.indexing_jobs



app = FastAPI(
    title="PolicyLens AI",
    description="AI-powered HR policy assistant",
    version="1.0.0",
    lifespan=lifespan,
)

environment = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
origin_value = os.environ.get("CORS_ALLOWED_ORIGINS") or environment.get("CORS_ALLOWED_ORIGINS") or "http://localhost:5173,http://127.0.0.1:5173"
origins = [value.strip().rstrip("/") for value in origin_value.split(",") if value.strip()]
if not origins or "*" in origins:
    raise ValueError("CORS_ALLOWED_ORIGINS must contain explicit frontend origins.")
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=False,
    allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
)

app.include_router(health_router)
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(document_files_router)
app.include_router(workspace_router)

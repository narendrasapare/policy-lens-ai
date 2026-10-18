from fastapi import APIRouter

router = APIRouter(tags=["System"])


@router.get("/health")
def get_health_status() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "PolicyLens AI",
    }
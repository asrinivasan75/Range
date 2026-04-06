"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "service": "range-api",
        "version": "0.1.0",
    }

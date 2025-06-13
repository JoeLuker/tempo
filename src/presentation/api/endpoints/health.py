"""Health check endpoints."""

import logging
from fastapi import APIRouter
from src.presentation.api.models.responses import HealthCheckResponse
from src.infrastructure.models.model_repository import ModelRepository

logger = logging.getLogger(__name__)

router = APIRouter()
model_repository = ModelRepository()


@router.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {"message": "TEMPO API is running", "status": "healthy"}


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint to verify API and model status."""
    try:
        # Quick check if model is initialized
        if not model_repository.is_initialized():
            return HealthCheckResponse(
                status="initializing",
                message="Model is not yet initialized"
            )
        
        # Get model info
        model_info = model_repository.get_model_info()
        
        if "error" in model_info:
            return HealthCheckResponse(
                status="unhealthy",
                error=model_info["error"]
            )
        
        return HealthCheckResponse(
            status="healthy",
            model_loaded=True,
            model_name=model_info.get("model_name"),
            device=model_info.get("device"),
            token_generator_initialized=model_info.get("token_generator_initialized"),
            generator_has_token_generator=model_info.get("generator_has_token_generator")
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheckResponse(
            status="unhealthy",
            error=str(e)
        )
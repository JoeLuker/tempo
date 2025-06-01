"""
Legacy v1 API router for TEMPO API.

This module contains routes for the v1 API, which redirect to v2 endpoints
for backward compatibility.
"""

from fastapi import APIRouter, Request, BackgroundTasks, status
from fastapi.responses import JSONResponse
from typing import Dict, Any

from src.utils import config
from src.utils.api_errors import RequestError
from src.api.model import get_model_components
from src.api.schemas.generation import GenerationRequest, GenerationResponse
from src.api.routers.v2 import generate_text as v2_generate_text

# Create router with v1 tag
router = APIRouter(prefix="/api/v1", tags=["v1-legacy"])


@router.get(
    "/",
    summary="API v1 Root",
    description="Legacy API root that redirects to the current version.",
    response_description="Information about API versioning.",
    response_model=Dict[str, str],
    status_code=status.HTTP_200_OK,
    tags=["Health"],
)
async def v1_root():
    """Legacy API v1 root endpoint."""
    return {
        "message": "You are using the legacy v1 API. Please update to v2.",
        "status": "deprecated",
        "current_version": config.api.api_version,
        "current_prefix": f"/api/{config.api.api_version}",
    }


@router.post(
    "/generate",
    summary="Generate Text (Legacy)",
    description="Legacy v1 API for text generation (redirects to v2).",
    response_description="Generated text and detailed token information.",
    tags=["Generation"],
    response_model=GenerationResponse,
)
async def generate_text_v1(request: dict, background_tasks: BackgroundTasks):
    """
    Legacy v1 API endpoint for text generation.

    Args:
        request: Raw dictionary with request parameters
        background_tasks: FastAPI background tasks

    Returns:
        GenerationResponse: The generation response

    Raises:
        RequestError: If the request is invalid
    """
    # Convert legacy v1 request to v2 format
    try:
        # Basic validation
        if "prompt" not in request:
            raise RequestError(message="Missing required parameter: prompt")

        # Create a v2 request object from the legacy request
        v2_request = GenerationRequest(**request)

        # Get model components
        components = await get_model_components(v2_request.model_name)

        # Forward to v2 implementation
        return await v2_generate_text(v2_request, background_tasks, components)
    except Exception as e:
        # Log the error
        import logging

        logger = logging.getLogger("tempo-api")
        logger.warning(f"Error in legacy v1 endpoint: {str(e)}")

        # Re-raise APIErrors, convert other exceptions to RequestError
        from src.utils.api_errors import APIError

        if isinstance(e, APIError):
            raise e
        raise RequestError(message=f"Invalid request: {str(e)}")

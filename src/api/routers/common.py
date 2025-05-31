"""
Common API router for TEMPO API.

This module contains routes that are common across all API versions,
such as health checks and version information.
"""

import sys
import pkg_resources
from fastapi import APIRouter, status
from typing import Dict, Any

from src.utils import config
from src.api.model import ModelSingleton
from src.api.schemas.health import HealthResponse

# Create router with common tag
router = APIRouter(prefix="/api", tags=["Common"])

@router.get(
    "/",
    summary="API Root",
    description="Returns a simple message indicating that the API is running.",
    response_description="A simple message indicating that the API is running.",
    response_model=Dict[str, str],
    status_code=status.HTTP_200_OK,
    tags=["Health"]
)
async def root():
    """API root endpoint for basic connectivity test."""
    return {
        "message": "TEMPO API is running", 
        "status": "healthy",
        "version": config.api.api_version
    }

@router.get(
    "/health",
    summary="Health Check",
    description="Checks the health of the API and model components.",
    response_description="Health status of the API and model components.",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"]
)
async def health_check():
    """
    Health check endpoint to verify API and model status.
    
    Returns:
        HealthResponse: Health status information
    """
    try:
        # Check if model is initialized
        if not ModelSingleton.initialized:
            return HealthResponse(
                status="initializing",
                model_loaded=False,
                model_name="<none>",
                device="<none>",
                token_generator_initialized=False,
                generator_has_token_generator=False,
                model_initialized_at=None,
                error="Model is not yet initialized"
            )

        # Verify components exist
        model_wrapper, tokenizer, generator, token_generator = ModelSingleton.get_instance()

        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name=ModelSingleton.last_loaded_model or config.model.model_id,
            device=generator.device if hasattr(generator, "device") else "unknown",
            token_generator_initialized=token_generator is not None,
            generator_has_token_generator=hasattr(generator, 'token_generator') and generator.token_generator is not None,
            model_initialized_at=ModelSingleton.initialization_time
        )
    except Exception as e:
        import logging
        logger = logging.getLogger("tempo-api")
        logger.error(f"Health check failed: {str(e)}")
        
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_name="<error>",
            device="<error>",
            token_generator_initialized=False,
            generator_has_token_generator=False,
            error=str(e)
        )

@router.get(
    "/versions",
    summary="API Version Information",
    description="Returns version information for the API and components.",
    response_description="Version information.",
    status_code=status.HTTP_200_OK,
    tags=["System"]
)
async def get_versions():
    """
    Get version information for the API and components.
    
    Returns:
        Dict: Version information
    """
    # Get packages relevant to TEMPO
    packages = {
        "torch": None,
        "fastapi": None,
        "transformers": None,
        "pydantic": None,
        "uvicorn": None
    }
    
    # Get package versions
    for package in packages.keys():
        try:
            version = pkg_resources.get_distribution(package).version
            packages[package] = version
        except pkg_resources.DistributionNotFound:
            packages[package] = "not installed"
    
    return {
        "api_version": config.api.api_version,
        "default_model": config.model.model_id,
        "packages": packages,
        "python_version": sys.version
    }
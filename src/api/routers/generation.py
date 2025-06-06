"""
Clean generation API router with proper REST design.
"""
from fastapi import APIRouter, HTTPException, Depends, Header, Request
from fastapi.responses import JSONResponse
from typing import Optional, Tuple
import logging
import traceback
import uuid
from datetime import datetime

from src.api.schemas.generation_v2 import (
    GenerationRequestV2,
    GenerationResponseV2,
    APIError
)
from src.api.services.generation_service import GenerationService
from src.api.utils.dependencies import get_generation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3", tags=["generation"])


@router.post(
    "/generations",
    response_model=GenerationResponseV2,
    summary="Generate text using TEMPO",
    description="""
    Generate text using TEMPO's parallel token generation approach.
    
    TEMPO explores multiple token possibilities at each step, maintaining parallel paths
    through the generation process. This allows for more diverse and creative outputs
    while maintaining coherence through attention-based pruning.
    
    ## Key Concepts
    
    - **Selection Threshold**: Controls how many parallel paths to explore (lower = more paths)
    - **Retroactive Pruning**: Refines token choices based on future context
    - **Dynamic Thresholds**: Gradually reduces diversity over the generation
    
    ## Example Request
    ```json
    {
        "prompt": "The future of AI is",
        "generation": {
            "max_tokens": 50,
            "selection_threshold": 0.1
        },
        "pruning": {
            "enabled": true,
            "attention_threshold": 0.01
        }
    }
    ```
    """,
    responses={
        200: {
            "description": "Successfully generated text",
            "model": GenerationResponseV2
        },
        400: {
            "description": "Invalid request parameters",
            "model": APIError,
            "content": {
                "application/json": {
                    "example": {
                        "error_code": "INVALID_PARAMETERS",
                        "message": "Selection threshold must be between 0 and 1",
                        "request_id": "req_123"
                    }
                }
            }
        },
        500: {
            "description": "Internal server error",
            "model": APIError
        }
    }
)
async def generate_text(
    request: GenerationRequestV2,
    generation_service: GenerationService = Depends(get_generation_service),
    x_request_id: Optional[str] = Header(None, description="Request tracking ID")
) -> GenerationResponseV2:
    """Generate text using TEMPO."""
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"
    
    try:
        logger.info(f"[{request_id}] Starting generation - prompt: {request.prompt[:50]}...")
        
        # Validate parameter combinations
        _validate_request(request)
        
        # Generate text
        response = generation_service.generate(request)
        
        logger.info(f"[{request_id}] Generation completed - {response.metadata.total_tokens_generated} tokens")
        return response
        
    except ValueError as e:
        logger.warning(f"[{request_id}] Invalid parameters: {str(e)}")
        error = APIError(
            error_code="INVALID_PARAMETERS",
            message=str(e),
            request_id=request_id
        )
        return JSONResponse(status_code=400, content=error.model_dump())
        
    except Exception as e:
        logger.error(f"[{request_id}] Generation failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        error = APIError(
            error_code="GENERATION_FAILED",
            message="An error occurred during text generation",
            details={"error_type": type(e).__name__},
            request_id=request_id
        )
        return JSONResponse(status_code=500, content=error.model_dump())


@router.get(
    "/health",
    summary="Check API health",
    description="Check if the generation service is healthy and ready to accept requests."
)
async def health_check(
    generation_service: GenerationService = Depends(get_generation_service)
) -> dict:
    """Check service health."""
    try:
        # Basic health check
        is_healthy = (
            generation_service.model_wrapper is not None and
            generation_service.tokenizer is not None and
            generation_service.generator is not None
        )
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "tempo-generation-api",
            "version": "3.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


def _validate_request(request: GenerationRequestV2) -> None:
    """Validate request parameter combinations."""
    # Validate MCTS parameters
    if request.mcts and request.mcts.enabled:
        if request.mcts.simulations_per_step < 1:
            raise ValueError("MCTS simulations must be at least 1")
        if request.mcts.exploration_constant < 0:
            raise ValueError("MCTS exploration constant must be non-negative")
    
    # Validate dynamic threshold
    if request.dynamic_threshold and request.dynamic_threshold.enabled:
        if len(request.dynamic_threshold.bezier_control_points) != 2:
            raise ValueError("Bezier curve requires exactly 2 control points")
        for point in request.dynamic_threshold.bezier_control_points:
            if not 0 <= point <= 1:
                raise ValueError("Bezier control points must be between 0 and 1")
    
    # Validate pruning configuration
    if request.pruning and request.pruning.enabled:
        if request.pruning.attention_threshold < 0:
            raise ValueError("Attention threshold must be non-negative")
        if request.pruning.use_relative_attention and not 0 <= request.pruning.relative_threshold <= 1:
            raise ValueError("Relative threshold must be between 0 and 1")
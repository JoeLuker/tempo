"""Generation endpoints for TEMPO API."""

import logging
from fastapi import APIRouter, HTTPException
from src.presentation.api.models.requests import GenerationRequest
from src.presentation.api.models.responses import GenerationResponse
from src.presentation.api.middleware.error_handler import handle_generation_errors
from src.application.services.model_service import ModelService

logger = logging.getLogger(__name__)

router = APIRouter()
model_service = ModelService()


@router.post("/generate", response_model=GenerationResponse)
@handle_generation_errors
async def generate_text(request: GenerationRequest):
    """
    Generate text using TEMPO parallel generation.
    
    This endpoint accepts various parameters to control the generation process,
    including selection thresholds, pruning strategies, and advanced features
    like MCTS and dynamic thresholding.
    """
    try:
        logger.info(
            f"Received generation request: prompt='{request.prompt[:50]}...', "
            f"max_tokens={request.max_tokens}, selection_threshold={request.selection_threshold}"
        )
        
        # Generate text using the service
        response = model_service.generate_text(request)
        
        logger.info(
            f"Generation completed successfully. Generated {len(response.clean_text)} characters of clean text."
        )
        
        return response
        
    except Exception as e:
        # The error handler decorator will catch and format the error appropriately
        raise
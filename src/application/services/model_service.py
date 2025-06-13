"""Service for coordinating model operations.

This module acts as a facade, coordinating between specialized services.
"""

import logging
from src.presentation.api.models.requests import GenerationRequest
from src.presentation.api.models.responses import GenerationResponse
from .generation_service import GenerationService

logger = logging.getLogger(__name__)


class ModelService:
    """Service for coordinating model operations."""
    
    def __init__(self):
        """Initialize the model service."""
        self.generation_service = GenerationService()
        
    def generate_text(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using TEMPO parallel generation.
        
        This method delegates to the specialized generation service.
        
        Args:
            request: Generation request parameters
            
        Returns:
            GenerationResponse with generated text and metadata
            
        Raises:
            Exception: If generation fails
        """
        logger.info(f"Delegating generation request to generation service")
        return self.generation_service.generate_text(request)
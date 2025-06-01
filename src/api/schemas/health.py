"""
Health check schemas for TEMPO API.

This module defines Pydantic models for health check responses.
"""

from pydantic import BaseModel, Field
from typing import Optional


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: str = Field(..., description="Name of loaded model")
    device: str = Field(..., description="Device being used")
    token_generator_initialized: bool = Field(..., description="TokenGenerator status")
    generator_has_token_generator: bool = Field(
        ..., description="Generator has TokenGenerator"
    )
    model_initialized_at: Optional[str] = Field(
        None, description="When model was initialized"
    )
    error: Optional[str] = Field(None, description="Error message if any")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_name": "deepcogito/cogito-v1-preview-llama-3B",
                "device": "cuda",
                "token_generator_initialized": True,
                "generator_has_token_generator": True,
                "model_initialized_at": "2023-05-01T12:34:56.789Z",
            }
        }

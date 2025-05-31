"""
Model management schemas for TEMPO API.

This module defines Pydantic models for model listing and management.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ModelParameter(BaseModel):
    """Parameter information for a model."""
    base_model: Optional[str] = Field(None, description="Base model architecture")
    version: Optional[str] = Field(None, description="Model version")
    quantization: Optional[str] = Field(None, description="Quantization applied")
    
    class Config:
        extra = "allow"  # Allow additional fields for model-specific parameters

class ModelInfo(BaseModel):
    """Information about an available model."""
    id: str = Field(..., description="Model identifier or path")
    name: str = Field(..., description="Human-readable model name")
    description: Optional[str] = Field(None, description="Model description")
    is_default: bool = Field(default=False, description="Whether this is the default model")
    size: Optional[str] = Field(None, description="Model size (e.g., '7B', '13B')")
    parameters: Optional[ModelParameter] = Field(None, description="Model parameters")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "deepcogito/cogito-v1-preview-llama-3B",
                "name": "Cogito v1 Preview (Llama 3B)",
                "description": "Optimized for performance with TEMPO generation",
                "is_default": True,
                "size": "3B",
                "parameters": {
                    "base_model": "llama",
                    "version": "v1"
                }
            }
        }

class ModelsListResponse(BaseModel):
    """Response for the models listing endpoint."""
    models: List[ModelInfo] = Field(..., description="List of available models")
    current_model: str = Field(..., description="Currently loaded model ID")
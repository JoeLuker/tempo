#!/usr/bin/env python3
"""
TEMPO API: Threshold-Enabled Multipath Parallel Output

This module implements a RESTful API for the TEMPO (Threshold-Enabled Multipath Parallel Output)
text generation system. It provides endpoints for parallel token generation, model configuration,
and system health monitoring.

The API follows RESTful principles with a structured OpenAPI specification, comprehensive
error handling, and input validation.
"""

import os
import time
import logging
import traceback
import uuid
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union, Set, Literal
from datetime import datetime

import torch
from fastapi import FastAPI, HTTPException, Depends, APIRouter, Query, Path, status, BackgroundTasks, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from pydantic import BaseModel, Field, root_validator, validator, AnyHttpUrl, HttpUrl
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from src.modeling.model_wrapper import TEMPOModelWrapper
from src.generation.parallel_generator import ParallelGenerator
from src.visualization.token_visualizer import TokenVisualizer
from src.visualization.position_visualizer import PositionVisualizer
from src.pruning import RetroactivePruner
from src.generation.token_generator import TokenGenerator
from src.utils.config import get_debug_mode
from src.utils.api_errors import (
    register_exception_handlers, 
    APIError, 
    ModelError, 
    ValidationError, 
    GenerationError, 
    ModelNotAvailableError,
    RequestError
)
from src.utils.rate_limiter import add_rate_limiting
from src.utils.api_docs import APIDocumentation

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("tempo-api")

# API version (used for API paths and models)
API_VERSION = "v2"
API_PREFIX = f"/api/{API_VERSION}"

# Default model settings
DEFAULT_MODEL = "deepcogito/cogito-v1-preview-llama-3B"
DEFAULT_MAX_TOKENS = 50
DEFAULT_SELECTION_THRESHOLD = 0.1

#######################
# Model Management
#######################

class ModelSingleton:
    """
    Singleton class to manage model components.
    
    This class ensures that the model, tokenizer, generator, and token_generator 
    are initialized only once and reused across requests.
    """
    model_wrapper = None  # Store the wrapped model
    tokenizer = None
    generator = None  # This will be ParallelGenerator
    token_generator = None  # Store the shared TokenGenerator
    initialized = False
    last_loaded_model = None  # Name of the model loaded
    initialization_time = None  # When the model was loaded
    
    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_MODEL, device: str = "auto"):
        """
        Get or initialize model instance, maintaining the singleton invariant.
        
        Args:
            model_name: The name or path of the model to load
            device: The device to use for computation ("auto", "cuda", "mps", "cpu")
            
        Returns:
            Tuple of (model_wrapper, tokenizer, generator, token_generator)
            
        Raises:
            HTTPException: If model initialization fails
        """
        # Auto-detect device if "auto" is specified
        if device == "auto":
            device = cls._get_best_device()
            
        # Initialize if needed or if a different model is requested
        if not cls.initialized or (model_name != cls.last_loaded_model and model_name != DEFAULT_MODEL):
            logger.info(f"Loading model '{model_name}' on device '{device}'...")
            cls._initialize_model(model_name, device)
            cls.initialized = True
            cls.last_loaded_model = model_name
            cls.initialization_time = datetime.now().isoformat()

        # Ensure all components exist
        if not all([cls.model_wrapper, cls.tokenizer, cls.generator, cls.token_generator]):
            error_msg = "Model components not properly initialized"
            logger.error(error_msg)
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                               detail=error_msg)
        
        # Ensure the generator has its token_generator
        if not hasattr(cls.generator, 'token_generator') or cls.generator.token_generator is None:
            error_msg = "Generator's token_generator not initialized"
            logger.error(error_msg)
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                               detail=error_msg)

        return cls.model_wrapper, cls.tokenizer, cls.generator, cls.token_generator
    
    @classmethod
    def _get_best_device(cls) -> str:
        """
        Determine the best available device for model execution.
        
        Returns:
            str: 'cuda' if an NVIDIA GPU is available, 'mps' for Apple Silicon, or 'cpu' as fallback
        """
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    @classmethod
    def _get_device_dtype(cls, device_str: str) -> torch.dtype:
        """
        Determine the appropriate dtype based on the device.
        
        Args:
            device_str: Device string ('cuda', 'mps', 'cpu')
            
        Returns:
            torch.dtype: The appropriate dtype for the device
        """
        if device_str == "cuda":
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                return torch.bfloat16  # Ampere and later GPUs
            else:
                return torch.float16  # Older GPUs
        elif device_str == "mps":
            return torch.float32  # MPS is most stable with float32
        else:  # "cpu"
            return torch.float32

    @classmethod
    def _initialize_model(cls, model_name: str, device: str):
        """
        Initialize model components with proper error handling.
        
        Args:
            model_name: The name or path of the model to load
            device: The device to use for computation
            
        Raises:
            HTTPException: If model initialization fails
        """
        try:
            # Determine device dtype
            device_str = device
            dtype = cls._get_device_dtype(device_str)
            logger.info(f"Using device: {device_str} with dtype: {dtype}")

            # Load tokenizer
            tokenizer_start = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Tokenizer loaded in {time.time() - tokenizer_start:.2f}s")

            # Load model configuration
            config_start = time.time()
            config = AutoConfig.from_pretrained(model_name)
            
            # Model-specific configuration adjustments
            if config.model_type and 'qwen' in config.model_type.lower():
                if hasattr(config, "sliding_window") and config.sliding_window is not None:
                    logger.info(f"Disabling Qwen sliding window (was {config.sliding_window})")
                    config.sliding_window = None
            
            logger.info(f"Model config loaded in {time.time() - config_start:.2f}s")

            # Load model weights
            model_load_start = time.time()
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=dtype,
                device_map="auto" if device_str == "cuda" else None,
                attn_implementation="eager",  # Use eager for stability
                low_cpu_mem_usage=True,
            )
            
            # Manually move to device if not using device_map
            if device_str != "cuda":
                model = model.to(device_str)
                
            # Set to evaluation mode
            model.eval()
            
            logger.info(f"Model loaded in {time.time() - model_load_start:.2f}s")

            # Create model wrapper 
            model_wrapper = TEMPOModelWrapper(model)
            model_wrapper.set_debug_mode(get_debug_mode("model_wrapper"))
            logger.info(f"Model wrapper created on device: {model_wrapper.device}")

            # Create shared TokenGenerator instance
            cls.token_generator = TokenGenerator(
                model=model_wrapper,
                tokenizer=tokenizer,
                device=device_str
            )
            logger.info(f"Shared TokenGenerator created on device: {cls.token_generator.device}")

            # Create ParallelGenerator with shared token_generator
            cls.generator = ParallelGenerator(
                model=model_wrapper,
                tokenizer=tokenizer,
                device=device_str,
                use_custom_rope=True,
                debug_mode=get_debug_mode("parallel_generator"),
                token_generator=cls.token_generator
            )
            logger.info(f"ParallelGenerator created with shared TokenGenerator (ID: {id(cls.token_generator)})")

            # Store components
            cls.model_wrapper = model_wrapper
            cls.tokenizer = tokenizer

        except Exception as e:
            error_msg = f"Error initializing model: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error_msg)


#######################
# Schemas
#######################

class PruningMode(str, Enum):
    """Supported pruning modes for the API."""
    KEEP_TOKEN = "keep_token"
    KEEP_UNATTENDED = "keep_unattended"
    REMOVE_POSITION = "remove_position"


class TokenInfo(BaseModel):
    """Information about a token, including text, ID, and probability."""
    token_text: str = Field(..., description="The decoded text of the token")
    token_id: int = Field(..., description="The token ID")
    probability: float = Field(..., description="The probability of the token")
    
    class Config:
        schema_extra = {
            "example": {
                "token_text": " the",
                "token_id": 262,
                "probability": 0.87654
            }
        }


class StepInfo(BaseModel):
    """Information about a generation step, including parallel and pruned tokens."""
    position: int = Field(..., description="The position in the generated sequence")
    parallel_tokens: List[TokenInfo] = Field(..., description="The tokens considered at this position")
    pruned_tokens: List[TokenInfo] = Field(..., description="The tokens pruned at this position")
    
    class Config:
        schema_extra = {
            "example": {
                "position": 3,
                "parallel_tokens": [
                    {"token_text": " the", "token_id": 262, "probability": 0.87654},
                    {"token_text": " a", "token_id": 263, "probability": 0.12345}
                ],
                "pruned_tokens": [
                    {"token_text": " an", "token_id": 264, "probability": 0.00001}
                ]
            }
        }


class ThresholdSettings(BaseModel):
    """Settings for dynamic thresholding."""
    use_dynamic_threshold: bool = Field(
        default=False, 
        description="Use a threshold that changes over generation steps"
    )
    final_threshold: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0, 
        description="Final threshold value for dynamic thresholding"
    )
    bezier_points: List[float] = Field(
        default=[0.2, 0.8], 
        description="Bezier control points for dynamic threshold curve"
    )
    use_relu: bool = Field(
        default=False, 
        description="Use ReLU transition instead of Bezier curve"
    )
    relu_activation_point: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0, 
        description="Point at which ReLU transition begins (0-1)"
    )
    
    @validator('bezier_points')
    def validate_bezier_points(cls, v):
        """Validate that bezier points are between 0 and 1."""
        if len(v) != 2:
            raise ValueError("Bezier points must be a list of 2 values")
        if not all(0 <= point <= 1 for point in v):
            raise ValueError("All Bezier points must be between 0 and 1")
        return v


class MCTSSettings(BaseModel):
    """Settings for Monte Carlo Tree Search."""
    use_mcts: bool = Field(
        default=False, 
        description="Use Monte Carlo Tree Search for text generation"
    )
    simulations: int = Field(
        default=10, 
        ge=1, 
        description="Number of MCTS simulations per step"
    )
    c_puct: float = Field(
        default=1.0, 
        ge=0.0, 
        description="Exploration constant for MCTS"
    )
    depth: int = Field(
        default=5, 
        ge=1, 
        description="Maximum depth for MCTS simulations"
    )


class RetroactivePruningSettings(BaseModel):
    """Settings for retroactive pruning."""
    enabled: bool = Field(
        default=True, 
        description="Use retroactive pruning to refine token sets based on future token attention"
    )
    attention_threshold: float = Field(
        default=0.01, 
        ge=0.0, 
        le=1.0, 
        description="Attention threshold for pruning (lower means more tokens kept)"
    )
    use_relative_attention: bool = Field(
        default=True, 
        description="Use relative attention thresholds"
    )
    relative_threshold: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0, 
        description="Threshold for relative attention-based pruning (0-1)"
    )
    use_multi_scale_attention: bool = Field(
        default=True, 
        description="Use multi-scale attention integration"
    )
    num_layers_to_use: Optional[int] = Field(
        default=None, 
        description="Number of last layers to use for attention (None means use all layers)"
    )
    use_lci_dynamic_threshold: bool = Field(
        default=True, 
        description="Use LCI-based dynamic thresholding"
    )
    use_sigmoid_threshold: bool = Field(
        default=True, 
        description="Use sigmoid-based decision boundary"
    )
    sigmoid_steepness: float = Field(
        default=10.0, 
        ge=1.0, 
        description="Controls how sharp the sigmoid transition is"
    )
    pruning_mode: PruningMode = Field(
        default=PruningMode.KEEP_TOKEN, 
        description="How to handle pruned positions"
    )


class AdvancedGenerationSettings(BaseModel):
    """Advanced settings for generation."""
    use_custom_rope: bool = Field(
        default=True, 
        description="Use custom RoPE modifications for parallel tokens"
    )
    disable_kv_cache: bool = Field(
        default=False, 
        description="Disable KV caching (slower but more consistent)"
    )
    disable_kv_cache_consistency: bool = Field(
        default=False, 
        description="Disable KV cache consistency checks"
    )
    allow_intraset_token_visibility: bool = Field(
        default=False, 
        description="Allow tokens within same parallel set to see each other"
    )
    no_preserve_isolated_tokens: bool = Field(
        default=False, 
        description="Allow pruning isolated tokens"
    )
    show_token_ids: bool = Field(
        default=False, 
        description="Include token IDs in formatted output"
    )
    system_content: Optional[str] = Field(
        default=None, 
        description="System message content for chat models"
    )
    enable_thinking: bool = Field(
        default=False, 
        description="Enable deep thinking mode"
    )
    debug_mode: bool = Field(
        default=False, 
        description="Enable debug mode for detailed logging"
    )


class GenerationRequest(BaseModel):
    """
    Request model for text generation with TEMPO.
    
    Contains all parameters for configuring and controlling the generation process.
    """
    # Core parameters
    prompt: str = Field(
        ..., 
        description="Text prompt to start generation",
        example="Explain the difference between a llama and an alpaca"
    )
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        ge=1, 
        le=512,
        description="Maximum number of tokens to generate",
        example=50
    )
    selection_threshold: float = Field(
        default=DEFAULT_SELECTION_THRESHOLD,
        ge=0.0, 
        le=1.0,
        description="Probability threshold for token selection",
        example=0.1
    )
    min_steps: int = Field(
        default=0,
        ge=0,
        description="Minimum steps to generate before considering EOS tokens",
        example=0
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Model to use for generation (defaults to system model)",
        example="deepcogito/cogito-v1-preview-llama-3B"
    )
    
    # Group advanced settings in nested models for better organization
    threshold_settings: ThresholdSettings = Field(default_factory=ThresholdSettings)
    mcts_settings: MCTSSettings = Field(default_factory=MCTSSettings)
    pruning_settings: RetroactivePruningSettings = Field(default_factory=RetroactivePruningSettings)
    advanced_settings: AdvancedGenerationSettings = Field(default_factory=AdvancedGenerationSettings)
    
    # Validators
    @validator('prompt')
    def prompt_must_not_be_empty(cls, v):
        """Validate that prompt is not empty."""
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v
    
    @validator('model_name')
    def validate_model_name(cls, v):
        """Validate model name if provided."""
        if v is not None and not v.strip():
            raise ValueError("Model name cannot be empty if provided")
        return v
    
    # For backward compatibility with older clients
    @root_validator(pre=True)
    def map_legacy_fields(cls, values):
        """Map fields from old API format to new nested structure."""
        # Map threshold fields
        if 'dynamic_threshold' in values:
            values.setdefault('threshold_settings', {})
            values['threshold_settings']['use_dynamic_threshold'] = values.pop('dynamic_threshold')
        if 'final_threshold' in values:
            values.setdefault('threshold_settings', {})
            values['threshold_settings']['final_threshold'] = values.pop('final_threshold')
        if 'bezier_p1' in values and 'bezier_p2' in values:
            values.setdefault('threshold_settings', {})
            values['threshold_settings']['bezier_points'] = [
                values.pop('bezier_p1'), 
                values.pop('bezier_p2')
            ]
        if 'use_relu' in values:
            values.setdefault('threshold_settings', {})
            values['threshold_settings']['use_relu'] = values.pop('use_relu')
        if 'relu_activation_point' in values:
            values.setdefault('threshold_settings', {})
            values['threshold_settings']['relu_activation_point'] = values.pop('relu_activation_point')
            
        # Map MCTS fields
        if 'use_mcts' in values:
            values.setdefault('mcts_settings', {})
            values['mcts_settings']['use_mcts'] = values.pop('use_mcts')
        if 'mcts_simulations' in values:
            values.setdefault('mcts_settings', {})
            values['mcts_settings']['simulations'] = values.pop('mcts_simulations')
        if 'mcts_c_puct' in values:
            values.setdefault('mcts_settings', {})
            values['mcts_settings']['c_puct'] = values.pop('mcts_c_puct')
        if 'mcts_depth' in values:
            values.setdefault('mcts_settings', {})
            values['mcts_settings']['depth'] = values.pop('mcts_depth')
            
        # Map retroactive pruning fields
        if 'use_retroactive_pruning' in values:
            values.setdefault('pruning_settings', {})
            values['pruning_settings']['enabled'] = values.pop('use_retroactive_pruning')
        if 'attention_threshold' in values:
            values.setdefault('pruning_settings', {})
            values['pruning_settings']['attention_threshold'] = values.pop('attention_threshold')
        if 'no_relative_attention' in values:
            values.setdefault('pruning_settings', {})
            values['pruning_settings']['use_relative_attention'] = not values.pop('no_relative_attention')
        if 'relative_threshold' in values:
            values.setdefault('pruning_settings', {})
            values['pruning_settings']['relative_threshold'] = values.pop('relative_threshold')
        if 'no_multi_scale_attention' in values:
            values.setdefault('pruning_settings', {})
            values['pruning_settings']['use_multi_scale_attention'] = not values.pop('no_multi_scale_attention')
        if 'num_layers_to_use' in values:
            values.setdefault('pruning_settings', {})
            values['pruning_settings']['num_layers_to_use'] = values.pop('num_layers_to_use')
        if 'no_lci_dynamic_threshold' in values:
            values.setdefault('pruning_settings', {})
            values['pruning_settings']['use_lci_dynamic_threshold'] = not values.pop('no_lci_dynamic_threshold')
        if 'no_sigmoid_threshold' in values:
            values.setdefault('pruning_settings', {})
            values['pruning_settings']['use_sigmoid_threshold'] = not values.pop('no_sigmoid_threshold')
        if 'sigmoid_steepness' in values:
            values.setdefault('pruning_settings', {})
            values['pruning_settings']['sigmoid_steepness'] = values.pop('sigmoid_steepness')
        if 'complete_pruning_mode' in values:
            values.setdefault('pruning_settings', {})
            # Map string to enum
            mode = values.pop('complete_pruning_mode')
            try:
                values['pruning_settings']['pruning_mode'] = PruningMode(mode)
            except ValueError:
                # Default to KEEP_TOKEN if invalid
                values['pruning_settings']['pruning_mode'] = PruningMode.KEEP_TOKEN
                
        # Map advanced settings
        for field in [
            'use_custom_rope', 'disable_kv_cache', 'disable_kv_cache_consistency',
            'allow_intraset_token_visibility', 'no_preserve_isolated_tokens',
            'show_token_ids', 'system_content', 'enable_thinking', 'debug_mode'
        ]:
            if field in values:
                values.setdefault('advanced_settings', {})
                values['advanced_settings'][field] = values.pop(field)
                
        return values
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Explain the difference between a llama and an alpaca",
                "max_tokens": 50,
                "selection_threshold": 0.1,
                "min_steps": 0,
                "threshold_settings": {
                    "use_dynamic_threshold": True,
                    "final_threshold": 0.9,
                    "bezier_points": [0.2, 0.8],
                    "use_relu": False,
                    "relu_activation_point": 0.5
                },
                "pruning_settings": {
                    "enabled": True,
                    "attention_threshold": 0.01,
                    "use_relative_attention": True,
                    "relative_threshold": 0.5
                },
                "advanced_settings": {
                    "use_custom_rope": True,
                    "disable_kv_cache": False,
                    "debug_mode": False
                }
            }
        }


class TimingInfo(BaseModel):
    """Performance timing information for generation."""
    generation_time: float = Field(..., description="Time spent in generation process")
    pruning_time: float = Field(..., description="Time spent in pruning process")
    elapsed_time: float = Field(..., description="Total elapsed time for the request")


class ModelInfo(BaseModel):
    """Information about the model used for generation."""
    model_name: str = Field(..., description="The name or path of the model")
    is_qwen_model: bool = Field(..., description="Whether the model is a Qwen model")
    use_custom_rope: bool = Field(..., description="Whether custom RoPE was used")
    device: str = Field(..., description="Device used for computation")
    model_type: Optional[str] = Field(None, description="The model type (e.g., llama, gpt-neo, etc.)")


class PruningInfo(BaseModel):
    """Information about pruning strategies used."""
    strategy: str = Field(..., description="The pruning strategy used")
    coherence_threshold: float = Field(..., description="Coherence threshold for pruning")
    diversity_clusters: int = Field(..., description="Number of diversity clusters")
    use_dynamic_threshold: bool = Field(..., description="Whether dynamic threshold was used")
    diversity_steps: int = Field(..., description="Number of diversity steps")
    final_threshold: float = Field(..., description="Final threshold value")
    use_relu: bool = Field(..., description="Whether ReLU transition was used")
    relu_activation_point: float = Field(..., description="ReLU activation point")
    bezier_points: List[float] = Field(..., description="Bezier control points")
    pruning_time: float = Field(..., description="Time spent in pruning")


class RetroactivePruningInfo(BaseModel):
    """Information about retroactive pruning."""
    attention_threshold: float = Field(..., description="Attention threshold for pruning")
    use_relative_attention: bool = Field(..., description="Whether relative attention was used")
    relative_threshold: float = Field(..., description="Relative attention threshold")
    use_multi_scale_attention: bool = Field(..., description="Whether multi-scale attention was used")
    num_layers_to_use: Optional[int] = Field(None, description="Number of layers used for attention")
    use_lci_dynamic_threshold: bool = Field(..., description="Whether LCI dynamic threshold was used")
    use_sigmoid_threshold: bool = Field(..., description="Whether sigmoid threshold was used")
    sigmoid_steepness: float = Field(..., description="Sigmoid steepness parameter")
    pruning_mode: str = Field(..., description="Mode for handling pruned positions")


class Token(BaseModel):
    """Information about a token in the raw token data."""
    id: int = Field(..., description="Token ID")
    text: str = Field(..., description="Token text")
    probability: float = Field(..., description="Token probability")


class TokenSetData(BaseModel):
    """Raw token set data for visualization."""
    position: int = Field(..., description="Position in the sequence")
    original_tokens: List[Token] = Field(..., description="Original tokens considered")
    pruned_tokens: List[Token] = Field(..., description="Tokens pruned from consideration")


class GenerationResponse(BaseModel):
    """
    Response model for text generation with TEMPO.
    
    Contains the generated text, token information, and performance metrics.
    """
    # Core output
    generated_text: str = Field(..., description="Complete generated text with formatting")
    raw_generated_text: str = Field(..., description="Raw generated text without formatting")
    
    # Token-level data
    steps: List[StepInfo] = Field(default_factory=list, description="Information about each generation step")
    position_to_tokens: Dict[str, List[str]] = Field(default_factory=dict, description="Mapping of positions to tokens")
    original_parallel_positions: List[int] = Field(default_factory=list, description="Positions with parallel tokens")
    
    # Performance and timing
    timing: TimingInfo = Field(..., description="Timing information for generation")
    
    # Pruning information
    pruning: Optional[PruningInfo] = Field(None, description="Information about pruning strategies")
    retroactive_pruning: Optional[RetroactivePruningInfo] = Field(None, description="Information about retroactive pruning")
    
    # Model information
    model_info: ModelInfo = Field(..., description="Information about the model used")
    
    # Generation settings
    selection_threshold: float = Field(..., description="Threshold used for token selection")
    max_tokens: int = Field(..., description="Maximum tokens generated")
    min_steps: int = Field(..., description="Minimum steps before considering EOS")
    prompt: str = Field(..., description="Input prompt")
    
    # Advanced fields
    had_repetition_loop: bool = Field(default=False, description="Whether repetition was detected")
    system_content: Optional[str] = Field(None, description="System content used")
    
    # Raw data for visualization
    token_sets: List[TokenSetData] = Field(
        default_factory=list, 
        description="Raw token sets data for visualization",
        exclude=True  # Exclude from automatically generated OpenAPI schema
    )
    tokens_by_position: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Token information by position",
        exclude=True
    )
    final_pruned_sets: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Final pruned token sets",
        exclude=True
    )
    
    # For clients that need raw data, include separate TokenSetData objects
    raw_token_data: List[TokenSetData] = Field(default_factory=list, description="Raw token data for visualization")
    
    class Config:
        schema_extra = {
            "example": {
                "generated_text": "Llamas and alpacas are both camelid species, but they have several key differences: llamas are larger, weighing 250-450 pounds compared to alpacas at 100-200 pounds.",
                "raw_generated_text": "Llamas and alpacas are both camelid species, but they have several key differences: llamas are larger, weighing 250-450 pounds compared to alpacas at 100-200 pounds.",
                "steps": [
                    {
                        "position": 0,
                        "parallel_tokens": [
                            {"token_text": "Llamas", "token_id": 123, "probability": 0.7},
                            {"token_text": "The", "token_id": 456, "probability": 0.2}
                        ],
                        "pruned_tokens": [
                            {"token_text": "A", "token_id": 789, "probability": 0.1}
                        ]
                    }
                ],
                "timing": {
                    "generation_time": 0.456,
                    "pruning_time": 0.123,
                    "elapsed_time": 0.579
                },
                "model_info": {
                    "model_name": "deepcogito/cogito-v1-preview-llama-3B",
                    "is_qwen_model": False,
                    "use_custom_rope": True,
                    "device": "cuda",
                    "model_type": "llama"
                },
                "selection_threshold": 0.1,
                "max_tokens": 50,
                "min_steps": 0,
                "prompt": "Explain the difference between a llama and an alpaca"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: str = Field(..., description="Name of loaded model")
    device: str = Field(..., description="Device being used")
    token_generator_initialized: bool = Field(..., description="TokenGenerator status")
    generator_has_token_generator: bool = Field(..., description="Generator has TokenGenerator")
    model_initialized_at: Optional[str] = Field(None, description="When model was initialized")
    error: Optional[str] = Field(None, description="Error message if any")


class ErrorResponse(BaseModel):
    """Standardized error response model."""
    status_code: int = Field(..., description="HTTP status code")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: str = Field(..., description="Error timestamp")
    path: Optional[str] = Field(None, description="Request path")
    
    @classmethod
    def create(cls, status_code: int, error: str, message: str, path: Optional[str] = None):
        """Create a standardized error response."""
        return cls(
            status_code=status_code,
            error=error,
            message=message,
            timestamp=datetime.now().isoformat(),
            path=path
        )


#######################
# API Implementation
#######################

# Create FastAPI app with metadata
app = FastAPI(
    title="TEMPO API",
    description="""
    TEMPO (Threshold-Enabled Multipath Parallel Output) is an experimental approach to text generation
    that processes multiple token possibilities simultaneously.
    
    This API provides endpoints for generating text with TEMPO, configuring model parameters,
    and monitoring system health.
    """,
    version=API_VERSION,
    docs_url=None,  # Customize docs URL below
    redoc_url=None,  # Customize redoc URL below
    openapi_url=f"{API_PREFIX}/openapi.json",
)

# Configure custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="TEMPO API",
        version=API_VERSION,
        description="""
        TEMPO (Threshold-Enabled Multipath Parallel Output) is an experimental approach to text generation
        that processes multiple token possibilities simultaneously.
        
        This API provides endpoints for generating text with TEMPO, configuring model parameters,
        and monitoring system health.
        """,
        routes=app.routes,
    )
    
    # Add custom schema components
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    if "schemas" not in openapi_schema["components"]:
        openapi_schema["components"]["schemas"] = {}
        
    # Add ErrorResponse schema
    openapi_schema["components"]["schemas"]["ErrorResponse"] = {
        "type": "object",
        "properties": {
            "status_code": {"type": "integer", "description": "HTTP status code"},
            "error": {"type": "string", "description": "Error type"},
            "message": {"type": "string", "description": "Error message"},
            "timestamp": {"type": "string", "description": "Error timestamp"},
            "path": {"type": "string", "description": "Request path"}
        },
        "required": ["status_code", "error", "message", "timestamp"]
    }
    
    # Update all endpoints to include error responses
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if method.lower() in ["get", "post", "put", "delete", "patch"]:
                responses = openapi_schema["paths"][path][method]["responses"]
                
                # Add common error responses
                if "400" not in responses:
                    responses["400"] = {
                        "description": "Bad Request",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}
                    }
                if "422" not in responses:
                    responses["422"] = {
                        "description": "Validation Error",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}
                    }
                if "500" not in responses:
                    responses["500"] = {
                        "description": "Internal Server Error",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}
                    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Create API version-specific routers
v2_router = APIRouter(prefix=API_PREFIX, tags=["v2"])

# Legacy compatibility router (redirects to current version)
v1_router = APIRouter(prefix="/api/v1", tags=["v1-legacy"])

# Version-neutral router for endpoints that don't change between versions
common_router = APIRouter(prefix="/api", tags=["common"])

# Configure CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add a unique request ID to each request for tracking."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Process the request
    response = await call_next(request)
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    return response

# Add custom API error handlers from the utils/api_errors module
register_exception_handlers(app)

# Enable rate limiting if not in debug mode
add_rate_limiting(app, enabled=not os.environ.get("TEMPO_DEBUG", "").lower() == "true")


# Dependency for getting model components
async def get_model_components(model_name: Optional[str] = None):
    """
    Dependency to get model components with proper error handling.
    
    Args:
        model_name: Optional name of the model to load
        
    Returns:
        Tuple of (model_wrapper, tokenizer, generator, token_generator)
    """
    try:
        # Use the provided model name or default
        model_to_load = model_name if model_name else DEFAULT_MODEL
        # Returns (model_wrapper, tokenizer, generator, token_generator)
        return ModelSingleton.get_instance(model_to_load)
    except Exception as e:
        logger.error(f"Failed to get model components: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail=f"Model not available: {str(e)}"
        )


# Create visualizer singletons for token visualization
token_visualizer = TokenVisualizer()
position_visualizer = PositionVisualizer()

# Cache to store recent generation results for visualization
generation_cache = {}


#######################
# API Endpoints
#######################

@common_router.get(
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
        "version": API_VERSION
    }

@v1_router.get(
    "/", 
    summary="API v1 Root",
    description="Legacy API root that redirects to the current version.",
    response_description="Information about API versioning.",
    response_model=Dict[str, str],
    status_code=status.HTTP_200_OK,
    tags=["Health"]
)
async def v1_root():
    """Legacy API v1 root endpoint."""
    return {
        "message": "You are using the legacy v1 API. Please update to v2.",
        "status": "deprecated",
        "current_version": API_VERSION,
        "current_prefix": API_PREFIX
    }

@v2_router.get(
    "/", 
    summary="API v2 Root",
    description="Root endpoint for API v2.",
    response_description="Basic API information.",
    response_model=Dict[str, str],
    status_code=status.HTTP_200_OK,
    tags=["Health"]
)
async def v2_root():
    """API v2 root endpoint."""
    return {
        "message": "TEMPO API v2 is running", 
        "status": "healthy",
        "version": API_VERSION
    }


@common_router.get(
    "/docs", 
    include_in_schema=False
)
async def custom_swagger_ui_html():
    """Custom Swagger UI endpoint."""
    return get_swagger_ui_html(
        openapi_url=f"{API_PREFIX}/openapi.json",
        title="TEMPO API",
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5.10.0/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5.10.0/swagger-ui.css",
    )


@common_router.get(
    "/redoc", 
    include_in_schema=False
)
async def redoc_html():
    """Custom ReDoc endpoint."""
    return get_redoc_html(
        openapi_url=f"{API_PREFIX}/openapi.json",
        title="TEMPO API",
        redoc_js_url="https://unpkg.com/redoc@next/bundles/redoc.standalone.js",
    )


@common_router.get(
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
            model_name=ModelSingleton.last_loaded_model or DEFAULT_MODEL,
            device=generator.device if hasattr(generator, "device") else "unknown",
            token_generator_initialized=token_generator is not None,
            generator_has_token_generator=hasattr(generator, 'token_generator') and generator.token_generator is not None,
            model_initialized_at=ModelSingleton.initialization_time
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        logger.error(traceback.format_exc())
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_name="<error>",
            device="<error>",
            token_generator_initialized=False,
            generator_has_token_generator=False,
            error=str(e)
        )


@v2_router.post(
    "/generate",
    summary="Generate Text",
    description="Generate text using TEMPO parallel generation.",
    response_description="Generated text and detailed token information.",
    response_model=GenerationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Successful text generation"},
        400: {"description": "Bad request parameter"},
        422: {"description": "Validation error"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Generation failed"},
        503: {"description": "Model not available"}
    },
    tags=["Generation"]
)
async def generate_text(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    components: Tuple = Depends(get_model_components)
):
    """
    Generate text using TEMPO parallel generation.
    
    Args:
        request: The generation request parameters
        background_tasks: FastAPI background tasks
        components: Model components from dependency injection
        
    Returns:
        GenerationResponse: Generated text and detailed token information
        
    Raises:
        HTTPException: If generation fails
    """
    try:
        # Unpack components
        model_wrapper, tokenizer, generator, shared_token_generator = components

        # --- Propagate Debug Mode from Request ---
        debug_mode = request.advanced_settings.debug_mode
        # Set on the shared TokenGenerator
        shared_token_generator.set_debug_mode(debug_mode)
        # Set on the singleton ParallelGenerator
        if hasattr(generator, 'set_debug_mode'):
            generator.set_debug_mode(debug_mode)
        # Set on the Model Wrapper
        if hasattr(model_wrapper, 'set_debug_mode'):
            model_wrapper.set_debug_mode(debug_mode)

        # Log the request
        logger.info(
            f"Received generation request: prompt={request.prompt[:50]}..., max_tokens={request.max_tokens}, debug={debug_mode}"
        )
        start_time = time.time()

        # --- Create Retroactive Pruner if enabled ---
        retroactive_pruner = None
        if request.pruning_settings.enabled:
            try:
                # Create RetroactivePruner with settings from request
                retroactive_pruner = RetroactivePruner(
                    model=model_wrapper,
                    tokenizer=tokenizer,
                    device=generator.device,
                    debug_mode=debug_mode,
                    attention_threshold=request.pruning_settings.attention_threshold,
                    use_relative_attention=request.pruning_settings.use_relative_attention,
                    relative_threshold=request.pruning_settings.relative_threshold,
                    use_multi_scale_attention=request.pruning_settings.use_multi_scale_attention,
                    num_layers_to_use=request.pruning_settings.num_layers_to_use,
                    use_lci_dynamic_threshold=request.pruning_settings.use_lci_dynamic_threshold,
                    use_sigmoid_threshold=request.pruning_settings.use_sigmoid_threshold,
                    sigmoid_steepness=request.pruning_settings.sigmoid_steepness,
                    complete_pruning_mode=request.pruning_settings.pruning_mode.value,
                )
                
                # Set the SHARED token generator on the retroactive pruner
                if hasattr(retroactive_pruner, 'set_token_generator'):
                    retroactive_pruner.set_token_generator(shared_token_generator)
                    logger.info(f"Set shared TokenGenerator on RetroactivePruner")
                else:
                    logger.warning("RetroactivePruner does not have set_token_generator method")

                logger.info(f"Created retroactive pruner with threshold: {request.pruning_settings.attention_threshold}")

            except ImportError as e:
                logger.error(f"Failed to import pruning components: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                    detail="Server configuration error: Pruning components not available"
                )
            except Exception as e:
                logger.error(f"Failed to initialize retroactive pruning: {e}")
                logger.error(traceback.format_exc())
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                    detail=f"Failed to initialize retroactive pruning: {str(e)}"
                )

        # Configure RoPE modifier KV cache consistency if RoPE is enabled
        if (request.advanced_settings.use_custom_rope and 
            hasattr(generator, "rope_modifier") and 
            generator.rope_modifier is not None):
            
            # Set debug mode on RoPE modifier instance
            generator.rope_modifier.set_debug_mode(debug_mode)

        # Prepare system content
        system_content = request.advanced_settings.system_content
        if request.advanced_settings.enable_thinking and not system_content:
            system_content = "Enable deep thinking subroutine."

        try:
            # --- Call the generator with all parameters ---
            generation_result = generator.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                selection_threshold=request.selection_threshold,
                return_parallel_sets=True,  # Needed for visualization
                use_retroactive_pruning=request.pruning_settings.enabled,
                retroactive_pruner=retroactive_pruner,
                min_steps=request.min_steps,
                show_token_ids=request.advanced_settings.show_token_ids,
                disable_kv_cache=request.advanced_settings.disable_kv_cache,
                system_content=system_content,
                isolate_parallel_tokens=not request.advanced_settings.allow_intraset_token_visibility,
                preserve_all_isolated_tokens=(
                    not request.advanced_settings.no_preserve_isolated_tokens
                    if not request.advanced_settings.allow_intraset_token_visibility
                    else None
                ),
                # MCTS parameters
                use_mcts=request.mcts_settings.use_mcts,
                mcts_simulations=request.mcts_settings.simulations,
                mcts_c_puct=request.mcts_settings.c_puct,
                mcts_depth=request.mcts_settings.depth,
                # Dynamic threshold parameters
                dynamic_threshold=request.threshold_settings.use_dynamic_threshold,
                final_threshold=request.threshold_settings.final_threshold,
                bezier_p1=request.threshold_settings.bezier_points[0],
                bezier_p2=request.threshold_settings.bezier_points[1],
                use_relu=request.threshold_settings.use_relu,
                relu_activation_point=request.threshold_settings.relu_activation_point,
            )
        except ValueError as e:
            logger.error(f"Value error during generation: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"Invalid generation parameters: {str(e)}"
            )
        except RuntimeError as e:
            logger.error(f"Runtime error during generation: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"Generation failed: {str(e)}"
            )
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="GPU memory exceeded. Try reducing max_tokens or batch size."
            )
        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"Generation failed unexpectedly: {str(e)}"
            )

        elapsed_time = time.time() - start_time
        logger.info(f"Generation completed in {elapsed_time:.2f}s")

        try:
            # Extract model type
            model_type = None
            if hasattr(model_wrapper.model, "config") and hasattr(model_wrapper.model.config, "model_type"):
                model_type = model_wrapper.model.config.model_type
            
            # Format response with proper error handling
            response = GenerationResponse(
                generated_text=generation_result["generated_text"],
                raw_generated_text=generation_result.get("raw_generated_text", ""),
                steps=[],  # Populated below
                timing=TimingInfo(
                    generation_time=generation_result.get("generation_time", elapsed_time),
                    pruning_time=generation_result.get("pruning_time", 0.0),
                    elapsed_time=elapsed_time,
                ),
                model_info=ModelInfo(
                    model_name=ModelSingleton.last_loaded_model or DEFAULT_MODEL,
                    is_qwen_model=generation_result.get("is_qwen_model", False),
                    use_custom_rope=request.advanced_settings.use_custom_rope,
                    device=generator.device,
                    model_type=model_type
                ),
                selection_threshold=request.selection_threshold,
                max_tokens=request.max_tokens,
                min_steps=request.min_steps,
                prompt=request.prompt,
                had_repetition_loop=generation_result.get("had_repetition_loop", False),
                system_content=system_content,
                position_to_tokens=generation_result.get("position_to_tokens", {}),
                original_parallel_positions=list(generation_result.get("original_parallel_positions", set())),
                tokens_by_position=generation_result.get("tokens_by_position", {}),
                final_pruned_sets=generation_result.get("final_pruned_sets", {}),
                raw_token_data=[]  # Populated below
            )

            # Process token sets safely
            token_sets_data = generation_result.get("token_sets", [])
            if token_sets_data:
                steps_list = []
                raw_token_data = []
                
                for step_data in token_sets_data:
                    try:
                        if isinstance(step_data, tuple) and len(step_data) == 3:
                            position, original_data, pruned_data = step_data
                            
                            # Ensure data is in the expected format
                            if (isinstance(original_data, tuple) and len(original_data) == 2 and
                                isinstance(pruned_data, tuple) and len(pruned_data) == 2):

                                original_ids, original_probs = original_data
                                pruned_ids_raw, pruned_probs_raw = pruned_data

                                # Convert to basic types safely
                                original_pairs = [(int(tid), float(prob)) for tid, prob in zip(original_ids, original_probs)]
                                pruned_pairs = [(int(tid), float(prob)) for tid, prob in zip(pruned_ids_raw, pruned_probs_raw)]

                                # Build step info with proper token info
                                try:
                                    # Create token info objects for parallel tokens
                                    parallel_tokens = [
                                        TokenInfo(
                                            token_text=tokenizer.decode([tid]),
                                            token_id=tid,
                                            probability=prob
                                        )
                                        for tid, prob in original_pairs
                                    ]
                                    
                                    # Create token info objects for pruned tokens
                                    pruned_tokens_info = [
                                        TokenInfo(
                                            token_text=tokenizer.decode([tid]),
                                            token_id=tid,
                                            probability=prob
                                        )
                                        for tid, prob in pruned_pairs
                                    ]
                                    
                                    # Add step info to list
                                    steps_list.append(StepInfo(
                                        position=position,
                                        parallel_tokens=parallel_tokens,
                                        pruned_tokens=pruned_tokens_info
                                    ))
                                    
                                    # Add raw token data for visualization
                                    raw_token_data.append(TokenSetData(
                                        position=position,
                                        original_tokens=[
                                            Token(id=tid, text=tokenizer.decode([tid]), probability=prob)
                                            for tid, prob in original_pairs
                                        ],
                                        pruned_tokens=[
                                            Token(id=tid, text=tokenizer.decode([tid]), probability=prob)
                                            for tid, prob in pruned_pairs
                                        ]
                                    ))
                                except Exception as e:
                                    logger.warning(f"Error processing tokens for step {position}: {e}")
                                    continue
                            else:
                                logger.warning(f"Skipping malformed token_set inner data: {step_data}")
                        else:
                            logger.warning(f"Skipping malformed token_set step data: {step_data}")
                    except Exception as e:
                        logger.warning(f"Error processing step data: {e}")
                        continue

                response.steps = steps_list
                response.raw_token_data = raw_token_data

            # Add pruning info safely
            if request.pruning_settings.enabled:
                response.retroactive_pruning = RetroactivePruningInfo(
                    attention_threshold=request.pruning_settings.attention_threshold,
                    use_relative_attention=request.pruning_settings.use_relative_attention,
                    relative_threshold=request.pruning_settings.relative_threshold,
                    use_multi_scale_attention=request.pruning_settings.use_multi_scale_attention,
                    num_layers_to_use=request.pruning_settings.num_layers_to_use,
                    use_lci_dynamic_threshold=request.pruning_settings.use_lci_dynamic_threshold,
                    use_sigmoid_threshold=request.pruning_settings.use_sigmoid_threshold,
                    sigmoid_steepness=request.pruning_settings.sigmoid_steepness,
                    pruning_mode=request.pruning_settings.pruning_mode.value
                )

            # Cache generation results for visualization (cleaned up in background)
            generation_id = str(int(time.time() * 1000))
            generation_cache[generation_id] = {
                "result": generation_result,
                "response": response,
                "timestamp": time.time()
            }
            
            # Schedule cleanup of old cache entries
            background_tasks.add_task(clean_generation_cache)
            
            return response

        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"Failed to format generation response: {str(e)}"
            )

    except ValueError as e:
        logger.error(f"Validation Error during generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Generation parameter error: {str(e)}"
        )
    except RuntimeError as e:
        logger.error(f"Runtime Error during generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Generation failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected Error during generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"An unexpected error occurred: {str(e)}"
        )


@v2_router.get(
    "/models/list",
    summary="List Available Models",
    description="Lists models that are available for loading.",
    response_description="List of available models.",
    status_code=status.HTTP_200_OK,
    tags=["Model Management"]
)
async def list_models():
    """
    List available models that can be loaded.
    
    Currently returns a fixed list but could be extended to check local cached models.
    
    Returns:
        Dict: List of available models with metadata
    """
    # For now, just return a fixed list of models that are known to work
    # This could be extended to scan a models directory or check an API
    return {
        "models": [
            {
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
        ],
        "current_model": ModelSingleton.last_loaded_model or DEFAULT_MODEL
    }


@v2_router.delete(
    "/cache/clear",
    summary="Clear Cache",
    description="Clears generation cache to free memory.",
    response_description="Confirmation of cache clearing.",
    status_code=status.HTTP_200_OK,
    tags=["System"]
)
async def clear_cache():
    """
    Clear the generation cache to free memory.
    
    Returns:
        Dict: Confirmation message
    """
    global generation_cache
    cache_size = len(generation_cache)
    generation_cache = {}
    
    return {
        "message": f"Cache cleared successfully",
        "entries_removed": cache_size
    }


def clean_generation_cache():
    """
    Clean up old generation cache entries.
    
    This is run as a background task after each generation.
    """
    global generation_cache
    
    # Keep entries for up to 30 minutes
    max_age = 30 * 60  # 30 minutes in seconds
    current_time = time.time()
    
    # Find old entries
    to_remove = []
    for key, entry in generation_cache.items():
        if current_time - entry["timestamp"] > max_age:
            to_remove.append(key)
    
    # Remove old entries
    for key in to_remove:
        del generation_cache[key]
    
    if to_remove:
        logger.info(f"Cleaned {len(to_remove)} old entries from generation cache")


@v2_router.get(
    "/history",
    summary="Generation History",
    description="Returns a list of recent generations.",
    response_description="List of recent generations with metadata.",
    status_code=status.HTTP_200_OK,
    tags=["Visualization"]
)
async def get_generation_history(
    limit: int = Query(10, ge=1, le=100, description="Maximum number of history items to return")
):
    """
    Get a list of recent generations with metadata.
    
    Args:
        limit: Maximum number of history items to return
        
    Returns:
        Dict: List of recent generations with metadata
    """
    # Convert cache to a list of simple entries
    history = []
    for generation_id, entry in generation_cache.items():
        try:
            history.append({
                "id": generation_id,
                "timestamp": entry["timestamp"],
                "prompt": entry["response"].prompt[:100] + "..." if len(entry["response"].prompt) > 100 else entry["response"].prompt,
                "length": len(entry["response"].generated_text),
                "elapsed_time": entry["response"].timing.elapsed_time
            })
        except Exception as e:
            logger.error(f"Error processing history entry {generation_id}: {e}")
    
    # Sort by timestamp (newest first) and limit
    history.sort(key=lambda x: x["timestamp"], reverse=True)
    history = history[:limit]
    
    return {
        "history": history,
        "total_entries": len(generation_cache)
    }


@common_router.get(
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
    import pkg_resources
    
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
        "api_version": API_VERSION,
        "default_model": DEFAULT_MODEL,
        "packages": packages,
        "python_version": sys.version
    }


# Create documentation router
docs_router = APIRouter(prefix="/docs", tags=["Documentation"])

@docs_router.get(
    "/sections",
    summary="List Documentation Sections",
    description="Returns a list of available documentation sections.",
    response_description="List of documentation sections.",
    status_code=status.HTTP_200_OK
)
async def list_doc_sections():
    """List available documentation sections."""
    return {
        "sections": APIDocumentation.get_section_list()
    }

@docs_router.get(
    "/section/{section_name}",
    summary="Get Documentation Section",
    description="Returns documentation content for a specific section.",
    response_description="Documentation content.",
    status_code=status.HTTP_200_OK
)
async def get_doc_section(section_name: str):
    """Get documentation for a specific section."""
    section = APIDocumentation.get_section(section_name)
    if section["title"] == "Documentation Not Found":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Documentation section '{section_name}' not found"
        )
    return section

# Include API routers in correct order (most specific first)
app.include_router(v2_router)
app.include_router(v1_router)
app.include_router(common_router)
app.include_router(docs_router)

# Add a legacy endpoint for v1 generation to maintain backward compatibility
@v1_router.post(
    "/generate",
    summary="Generate Text (Legacy)",
    description="Legacy v1 API for text generation (redirects to v2).",
    response_description="Generated text and detailed token information.",
    tags=["Generation"],
    response_model=GenerationResponse
)
async def generate_text_v1(
    request: dict,
    background_tasks: BackgroundTasks
):
    """Legacy v1 API endpoint for text generation."""
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
        return await generate_text(v2_request, background_tasks, components)
    except Exception as e:
        logger.warning(f"Error in legacy v1 endpoint: {str(e)}")
        if isinstance(e, APIError):
            raise e
        raise RequestError(message=f"Invalid request: {str(e)}")

# Run with: uvicorn api_v2:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn

    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Configure logging for uvicorn
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "use_colors": True,
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO"},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
        },
    }
    
    # Run server
    uvicorn.run(
        "api_v2:app", 
        host="0.0.0.0", 
        port=port,
        log_config=log_config
    )
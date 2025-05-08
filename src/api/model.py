"""
Model management for TEMPO API.

This module provides functionality for loading and managing models for the API,
including the ModelSingleton class for maintaining model state between requests.
"""

import time
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

import torch
from fastapi import HTTPException, status
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from src.utils import config, get_debug_mode
from src.utils.model_utils import (
    load_tempo_components, load_model, get_best_device
)
from src.utils.logger import get_logger
from src.modeling.model_wrapper import TEMPOModelWrapper
from src.generation.token_generator import TokenGenerator
from src.generation.parallel_generator import ParallelGenerator

# Configure logging
logger = get_logger("api-model")

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
    def get_instance(cls, model_name: str = None, device: str = "auto"):
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
        # Use configuration if not provided
        model_name = model_name or config.model.model_id
        
        # Auto-detect device if "auto" is specified or not provided
        if device == "auto" or device is None:
            device = get_best_device()
            
        # Initialize if needed or if a different model is requested
        if not cls.initialized or (model_name != cls.last_loaded_model):
            try:
                logger.info(f"Loading model '{model_name}' on device '{device}'...")
                cls._initialize_model(model_name, device)
                cls.initialized = True
                cls.last_loaded_model = model_name
                cls.initialization_time = datetime.now().isoformat()
            except Exception as e:
                error_msg = f"Error initializing model: {str(e)}"
                logger.error(error_msg)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                    detail=error_msg
                )

        # Ensure all components exist
        if not all([cls.model_wrapper, cls.tokenizer, cls.generator, cls.token_generator]):
            error_msg = "Model components not properly initialized"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                detail=error_msg
            )
        
        # Ensure the generator has its token_generator
        if not hasattr(cls.generator, 'token_generator') or cls.generator.token_generator is None:
            error_msg = "Generator's token_generator not initialized"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                detail=error_msg
            )

        return cls.model_wrapper, cls.tokenizer, cls.generator, cls.token_generator
    
    # This method is no longer needed as we use the centralized get_best_device function
    
    # This method is no longer needed as we use the centralized get_device_dtype function
    # from model_utils.py. It is kept here temporarily for backward compatibility.
    @classmethod
    def _get_device_dtype(cls, device_str: str) -> torch.dtype:
        """
        Determine the appropriate dtype based on the device.
        
        This method is deprecated and will be removed in a future version.
        Use model_utils.get_device_dtype() instead.
        
        Args:
            device_str: Device string ('cuda', 'mps', 'cpu')
            
        Returns:
            torch.dtype: The appropriate dtype for the device
        """
        # Delegate to the centralized function
        from src.utils.model_utils import get_device_dtype
        return get_device_dtype(device_str)

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
            # Load all TEMPO components using the centralized utility function
            start_time = time.time()
            logger.info(f"Loading TEMPO components for model '{model_name}' on device '{device}'...")
            
            components = load_tempo_components(
                model_id=model_name,
                device=device,
                load_model_wrapper=True,
                load_token_generator=True,
                load_parallel_generator=True,
                debug_mode=get_debug_mode("model_wrapper"),
                use_custom_rope=True,
                use_fast_tokenizer=config.model.use_fast_tokenizer,
                trust_remote_code=config.model.trust_remote_code,
                revision=config.model.revision,
                low_cpu_mem_usage=config.model.low_cpu_mem_usage,
                attn_implementation="eager",  # Use eager for stability
                quantization=config.model.quantization
            )
            
            # Extract and store components
            cls.model_wrapper = components.get("model_wrapper")
            cls.tokenizer = components.get("tokenizer")
            cls.token_generator = components.get("token_generator")
            cls.generator = components.get("parallel_generator")
            
            # Log successful initialization
            total_time = time.time() - start_time
            logger.info(f"All components loaded successfully in {total_time:.2f}s")
            logger.info(f"ModelWrapper device: {cls.model_wrapper.device}")
            logger.info(f"TokenGenerator device: {cls.token_generator.device}")
            logger.info(f"ParallelGenerator created with TokenGenerator (ID: {id(cls.token_generator)})")

        except Exception as e:
            import traceback
            error_msg = f"Error initializing model: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error_msg)


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
        return ModelSingleton.get_instance(model_name)
    except Exception as e:
        import traceback
        logger.error(f"Failed to get model components: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail=f"Model not available: {str(e)}"
        )
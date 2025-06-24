"""
Model utilities for TEMPO project.

This module provides centralized functionality for model loading, device detection,
and other model-related operations to ensure consistency across the project.
"""

import os
import time
import torch
import logging
from typing import Any, Optional, Union
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.utils import config
from src.utils.logger import get_logger
from src.utils.error_manager import ModelError, ErrorCode, safe_execute
from src.utils.exception_handlers import handle_model_errors, handle_exceptions

# Configure logger
logger = get_logger("model_utils")

# Constants for model defaults
DEFAULT_MODEL_ID = "deepcogito/cogito-v1-preview-llama-3B"
DEFAULT_REVISION = "main"
SUPPORTED_DEVICES = ["cuda", "mps", "cpu", "auto"]


def get_best_device() -> str:
    """
    Determine the best available device for model execution.

    First checks configuration, then auto-detects based on available hardware.

    Returns:
        str: 'cuda' if an NVIDIA GPU is available, 'mps' for Apple Silicon, or 'cpu' as fallback
    """
    # Use device from config if specified
    if config.model.device:
        device = config.model.device
        logger.info(f"Using device from configuration: {device}")
        return device

    # Auto-detect best device
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Auto-detected CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Auto-detected Apple Silicon MPS device")
        return device

    # Fallback to CPU
    logger.info("No GPU detected, using CPU device")
    return "cpu"


def get_device_dtype(device: str = None) -> torch.dtype:
    """
    Determine the appropriate dtype based on the device.

    First checks configuration, then selects based on device capabilities.

    Args:
        device: Device string ('cuda', 'mps', 'cpu', or None for auto-detection)

    Returns:
        torch.dtype: The appropriate dtype for the device
    """
    # Resolve device if not provided
    if device is None or device == "auto":
        device = get_best_device()

    # Use dtype from config if specified
    if config.model.torch_dtype:
        if config.model.torch_dtype == "float16":
            dtype = torch.float16
        elif config.model.torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif config.model.torch_dtype == "float32":
            dtype = torch.float32
        else:
            logger.warning(
                f"Unknown dtype in config: {config.model.torch_dtype}, using auto-detection"
            )
            dtype = None

        if dtype:
            logger.info(f"Using dtype from configuration: {dtype}")
            return dtype

    # Auto-select dtype based on device capabilities
    if device == "cuda":
        # Use bfloat16 for Ampere and later GPUs, float16 for older GPUs
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            logger.info("Using bfloat16 for Ampere or newer NVIDIA GPU")
            return torch.bfloat16
        else:
            logger.info("Using float16 for older NVIDIA GPU")
            return torch.float16
    elif device == "mps":
        # MPS is most stable with float32
        logger.info("Using float32 for Apple Silicon MPS")
        return torch.float32
    else:  # cpu
        logger.info("Using float32 for CPU")
        return torch.float32


@handle_model_errors(
    error_message="Failed to load model", error_code=ErrorCode.MODEL_LOAD_FAILED
)
def load_model(
    model_id: str = None,
    revision: str = None,
    device: str = None,
    torch_dtype: torch.dtype = None,
    use_fast_tokenizer: bool = None,
    trust_remote_code: bool = None,
    attn_implementation: str = "eager",
    load_tokenizer: bool = True,
    **kwargs,
) -> Union[PreTrainedModel, tuple[PreTrainedModel, PreTrainedTokenizer]]:
    """
    Load a model and optionally its tokenizer using a consistent approach.

    Args:
        model_id: Model identifier or path (defaults to config or DEFAULT_MODEL_ID)
        revision: Model revision to load (defaults to config or DEFAULT_REVISION)
        device: Device to load the model on (defaults to auto-detection)
        torch_dtype: PyTorch data type for the model (defaults to auto-detection)
        use_fast_tokenizer: Whether to use the fast tokenizer implementation
        trust_remote_code: Whether to trust remote code for the model
        attn_implementation: Attention implementation to use ("eager", "flash_attention", etc.)
        load_tokenizer: Whether to load and return the tokenizer
        **kwargs: Additional arguments to pass to from_pretrained

    Returns:
        If load_tokenizer is True, returns a tuple of (model, tokenizer)
        Otherwise, returns just the model

    Raises:
        ModelError: If model loading fails
    """
    start_time = time.time()

    # Determine model ID
    model_id = model_id or config.model.model_id or DEFAULT_MODEL_ID

    # Determine revision
    revision = revision or config.model.revision or DEFAULT_REVISION

    # Determine device
    if device is None or device == "auto":
        device = get_best_device()

    # Determine dtype
    if torch_dtype is None:
        torch_dtype = get_device_dtype(device)

    # Determine tokenizer settings
    if use_fast_tokenizer is None:
        use_fast_tokenizer = config.model.use_fast_tokenizer

    # Determine trust_remote_code setting
    if trust_remote_code is None:
        trust_remote_code = config.model.trust_remote_code

    # Log loading info
    logger.info(
        f"Loading model: {model_id} (revision: {revision}) on device: {device} with dtype: {torch_dtype}"
    )

    # Load model configuration
    config_start = time.time()
    model_config = AutoConfig.from_pretrained(
        model_id, revision=revision, trust_remote_code=trust_remote_code
    )
    logger.info(f"Loaded model config in {time.time() - config_start:.2f}s")

    # Make model-specific adjustments to configuration
    if hasattr(model_config, "model_type") and model_config.model_type:
        # Adjust Qwen models (disable sliding window)
        if "qwen" in model_config.model_type.lower():
            if (
                hasattr(model_config, "sliding_window")
                and model_config.sliding_window is not None
            ):
                logger.info(
                    f"Disabling Qwen sliding window (was {model_config.sliding_window})"
                )
                model_config.sliding_window = None

    # Prepare model loading arguments
    model_args = {
        "config": model_config,
        "torch_dtype": torch_dtype,
        "attn_implementation": attn_implementation,
        "low_cpu_mem_usage": kwargs.pop(
            "low_cpu_mem_usage", config.model.low_cpu_mem_usage
        ),
        "trust_remote_code": trust_remote_code,
        "revision": revision,
        **kwargs,
    }

    # Use device_map="auto" for CUDA
    if device == "cuda":
        model_args["device_map"] = "auto"

    # Load tokenizer if requested
    tokenizer = None
    if load_tokenizer:
        tokenizer_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=use_fast_tokenizer,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )

        # Ensure the tokenizer has a pad token
        if tokenizer.pad_token is None:
            logger.info("Tokenizer has no pad_token, using eos_token instead")
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loaded tokenizer in {time.time() - tokenizer_start:.2f}s")

    # Load model
    model_load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_args)

    # Manually move to device if not using device_map
    if device != "cuda":
        model = model.to(device)

    # Set model to evaluation mode
    model.eval()

    logger.info(f"Loaded model in {time.time() - model_load_start:.2f}s")

    # Log total loading time
    total_time = time.time() - start_time
    logger.info(f"Total model loading time: {total_time:.2f}s")

    # Return model and tokenizer if requested
    if load_tokenizer:
        return model, tokenizer

    return model


@handle_model_errors(
    error_message="Failed to load model components",
    error_code=ErrorCode.MODEL_LOAD_FAILED,
)
def load_tempo_components(
    model_id: str = None,
    device: str = None,
    load_model_wrapper: bool = True,
    load_token_generator: bool = True,
    load_parallel_generator: bool = True,
    debug_mode: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """
    Load all TEMPO components required for generation.

    This is a higher-level function that loads the model, tokenizer, and necessary
    TEMPO components based on the specified parameters.

    Args:
        model_id: Model identifier or path
        device: Device to load the model on
        load_model_wrapper: Whether to create and return a TEMPOModelWrapper
        load_token_generator: Whether to create and return a TokenGenerator
        load_parallel_generator: Whether to create and return a ParallelGenerator
        debug_mode: Whether to enable debug mode for all components
        **kwargs: Additional arguments to pass to load_model

    Returns:
        Dict with keys 'model', 'tokenizer', and depending on options:
        'model_wrapper', 'token_generator', 'parallel_generator'

    Raises:
        ModelError: If component loading fails
    """
    from src.modeling.model_wrapper import TEMPOModelWrapper

    result = {}

    # Load model and tokenizer
    model, tokenizer = load_model(
        model_id=model_id, device=device, load_tokenizer=True, **kwargs
    )
    result["model"] = model
    result["tokenizer"] = tokenizer

    # Create TEMPOModelWrapper if requested
    if load_model_wrapper:
        logger.info("Creating TEMPOModelWrapper")
        model_wrapper = TEMPOModelWrapper(model, tokenizer, device)
        model_wrapper.set_debug_mode(debug_mode)
        result["model_wrapper"] = model_wrapper
    else:
        model_wrapper = None

    # Create TokenGenerator if requested
    if load_token_generator:
        if not model_wrapper and load_model_wrapper:
            raise ModelError(
                message="ModelWrapper is required for TokenGenerator",
                code=ErrorCode.INIT_COMPONENT_MISSING,
            )

        logger.info("Creating TokenGenerator (legacy)")
        from src.generation.token_generator import TokenGenerator

        token_generator = TokenGenerator(
            model=model_wrapper or model, tokenizer=tokenizer, device=device
        )
        token_generator.set_debug_mode(debug_mode)
        result["token_generator"] = token_generator
    else:
        token_generator = None

    # Create GenerateTextUseCase if requested (new clean architecture)
    if load_parallel_generator:
        if not model_wrapper and load_model_wrapper:
            raise ModelError(
                message="ModelWrapper is required for GenerateTextUseCase",
                code=ErrorCode.INIT_COMPONENT_MISSING,
            )

        logger.info("Creating GenerateTextUseCase with clean architecture")
        from src.application.use_cases.generate_text import GenerateTextUseCase
        from src.infrastructure.generation.token_generator_impl import TokenGeneratorImpl
        from src.infrastructure.tokenization.tokenizer_adapter import TokenizerAdapter
        from src.infrastructure.generation.standard_generation_strategy import StandardGenerationStrategy
        from src.application.services.sequence_manager import SequenceManager
        from src.application.adapters.generation_adapter import GenerationAdapter

        # Create implementations
        from src.infrastructure.model.model_adapter import ModelAdapter
        
        model_adapter = ModelAdapter(model_wrapper.model if model_wrapper else model)
        token_generator_impl = TokenGeneratorImpl(model_adapter)
        tokenizer_adapter = TokenizerAdapter(tokenizer)
        generation_strategy = StandardGenerationStrategy()
        sequence_manager = SequenceManager()

        generate_text_use_case = GenerateTextUseCase(
            token_generator=token_generator_impl,
            tokenizer=tokenizer_adapter,
            generation_strategy=generation_strategy,
            sequence_manager=sequence_manager,
            debug_mode=debug_mode
        )
        
        # Wrap use case in adapter for legacy compatibility
        generation_adapter = GenerationAdapter(generate_text_use_case)
        result["generate_text_use_case"] = generation_adapter

    return result


@handle_exceptions(
    error_message="Failed to check for quantization support",
    error_code=ErrorCode.MODEL_LOAD_FAILED,
)
def check_quantization_support() -> dict[str, bool]:
    """
    Check which quantization methods are supported in the current environment.

    Returns:
        Dict with keys for each quantization method and boolean values
    """
    result = {
        "bitsandbytes_4bit": False,
        "bitsandbytes_8bit": False,
        "gptq": False,
        "awq": False,
        "eetq": False,
    }

    # Check for bitsandbytes
    try:
        import bitsandbytes as bnb

        result["bitsandbytes_4bit"] = True
        result["bitsandbytes_8bit"] = True
        logger.info("bitsandbytes quantization support detected")
    except ImportError:
        logger.info(
            "bitsandbytes not installed, 4-bit and 8-bit quantization unavailable"
        )

    # Check for GPTQ
    try:
        from transformers import GPTQConfig

        result["gptq"] = True
        logger.info("GPTQ quantization support detected")
    except ImportError:
        logger.info("GPTQ not available")

    # Check for AWQ
    try:
        import awq

        result["awq"] = True
        logger.info("AWQ quantization support detected")
    except ImportError:
        logger.info("AWQ not available")

    # Check for EETQ
    try:
        import eetq

        result["eetq"] = True
        logger.info("EETQ quantization support detected")
    except ImportError:
        logger.info("EETQ not available")

    return result


@handle_model_errors(
    error_message="Failed to load quantized model",
    error_code=ErrorCode.MODEL_LOAD_FAILED,
)
def load_quantized_model(
    model_id: str = None, quantization: str = None, device: str = None, **kwargs
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a quantized model using the specified quantization method.

    Args:
        model_id: Model identifier or path
        quantization: Quantization method to use ("4bit", "8bit", "gptq", "awq", etc.)
        device: Device to load the model on
        **kwargs: Additional arguments to pass to load_model

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ModelError: If model loading fails or quantization method is unsupported
    """
    # Default to config values if not provided
    model_id = model_id or config.model.model_id or DEFAULT_MODEL_ID
    quantization = quantization or config.model.quantization

    # Check if quantization is specified
    if not quantization:
        return load_model(model_id=model_id, device=device, **kwargs)

    # Check quantization support
    support = check_quantization_support()

    # Handle different quantization methods
    if quantization == "4bit":
        if not support["bitsandbytes_4bit"]:
            raise ModelError(
                message="4-bit quantization requires bitsandbytes",
                code=ErrorCode.MODEL_UNSUPPORTED,
            )

        logger.info("Loading model with 4-bit quantization")
        try:
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig

            # Configure 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            # Load model with quantization config
            return load_model(
                model_id=model_id,
                device=device,
                quantization_config=quantization_config,
                **kwargs,
            )
        except ImportError as e:
            raise ModelError(
                message=f"Failed to import bitsandbytes for 4-bit quantization: {str(e)}",
                code=ErrorCode.DEPENDENCY_ERROR,
            )

    elif quantization == "8bit":
        if not support["bitsandbytes_8bit"]:
            raise ModelError(
                message="8-bit quantization requires bitsandbytes",
                code=ErrorCode.MODEL_UNSUPPORTED,
            )

        logger.info("Loading model with 8-bit quantization")
        try:
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig

            # Configure 8-bit quantization
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            # Load model with quantization config
            return load_model(
                model_id=model_id,
                device=device,
                quantization_config=quantization_config,
                **kwargs,
            )
        except ImportError as e:
            raise ModelError(
                message=f"Failed to import bitsandbytes for 8-bit quantization: {str(e)}",
                code=ErrorCode.DEPENDENCY_ERROR,
            )

    # Other quantization methods could be added here
    else:
        raise ModelError(
            message=f"Unsupported quantization method: {quantization}",
            code=ErrorCode.MODEL_UNSUPPORTED,
        )


def get_model_info(model: PreTrainedModel) -> dict[str, Any]:
    """
    Get information about a loaded model.

    Args:
        model: The model to get information for

    Returns:
        Dict with model information
    """
    info = {
        "model_type": getattr(model.config, "model_type", "unknown"),
        "model_id": getattr(model.config, "_name_or_path", "unknown"),
        "device": next(model.parameters()).device.type,
    }

    # Add quantization information if available
    if hasattr(model, "is_quantized") and model.is_quantized:
        info["quantized"] = True
        info["quantization_method"] = model.quantization_method
    else:
        info["quantized"] = False

    # Add architecture information
    if hasattr(model.config, "architectures") and model.config.architectures:
        info["architecture"] = model.config.architectures[0]

    # Add vocabulary size
    if hasattr(model.config, "vocab_size"):
        info["vocab_size"] = model.config.vocab_size

    # Add model size information
    num_params = sum(p.numel() for p in model.parameters())
    info["parameters"] = num_params
    info["parameters_billions"] = num_params / 1e9

    return info

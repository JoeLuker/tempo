"""
v2 API router for TEMPO API.

This module contains the API routes for the current stable v2 API.
"""

import time
import logging
import traceback
from typing import Dict, List, Any, Tuple, Optional

import torch
from fastapi import APIRouter, BackgroundTasks, Depends, Request, Query, Path, status
from fastapi.responses import JSONResponse

from src.utils import config
from src.utils.api_errors import (
    HTTPException,
    ValidationError,
    GenerationError,
    RequestError,
)

from src.api.model import get_model_components
from src.api.schemas.generation import (
    GenerationRequest,
    GenerationResponse,
    TimingInfo,
    ModelInfo,
    RetroactivePruningInfo,
    Token,
    TokenInfo,
    StepInfo,
    TokenSetData,
)
from src.api.schemas.models import ModelsListResponse, ModelInfo as AvailableModelInfo
from src.api.utils.cache import generation_cache, clean_generation_cache, clear_cache
from src.pruning import RetroactivePruner

# Configure logging
logger = logging.getLogger("tempo-api")

# Create router with v2 tag
router = APIRouter(
    prefix=f"/api/{config.api.api_version}", tags=[f"{config.api.api_version}"]
)


@router.get(
    "/",
    summary="API v2 Root",
    description="Root endpoint for API v2.",
    response_description="Basic API information.",
    response_model=Dict[str, str],
    status_code=status.HTTP_200_OK,
    tags=["Health"],
)
async def v2_root():
    """API v2 root endpoint."""
    return {
        "message": f"TEMPO API {config.api.api_version} is running",
        "status": "healthy",
        "version": config.api.api_version,
    }


@router.post(
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
        503: {"description": "Model not available"},
    },
    tags=["Generation"],
)
async def generate_text(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    components: Tuple = Depends(get_model_components),
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
        if hasattr(generator, "set_debug_mode"):
            generator.set_debug_mode(debug_mode)

        # Set on the Model Wrapper
        if hasattr(model_wrapper, "set_debug_mode"):
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
                if hasattr(retroactive_pruner, "set_token_generator"):
                    retroactive_pruner.set_token_generator(shared_token_generator)
                    logger.info(f"Set shared TokenGenerator on RetroactivePruner")
                else:
                    logger.warning(
                        "RetroactivePruner does not have set_token_generator method"
                    )

                logger.info(
                    f"Created retroactive pruner with threshold: {request.pruning_settings.attention_threshold}"
                )

            except ImportError as e:
                logger.error(f"Failed to import pruning components: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Server configuration error: Pruning components not available",
                )
            except Exception as e:
                logger.error(f"Failed to initialize retroactive pruning: {e}")
                logger.error(traceback.format_exc())
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to initialize retroactive pruning: {str(e)}",
                )

        # Configure RoPE modifier KV cache consistency if RoPE is enabled
        if (
            request.advanced_settings.use_custom_rope
            and hasattr(generator, "rope_modifier")
            and generator.rope_modifier is not None
        ):

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
                detail=f"Invalid generation parameters: {str(e)}",
            )
        except RuntimeError as e:
            logger.error(f"Runtime error during generation: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Generation failed: {str(e)}",
            )
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GPU memory exceeded. Try reducing max_tokens or batch size.",
            )
        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Generation failed unexpectedly: {str(e)}",
            )

        elapsed_time = time.time() - start_time
        logger.info(f"Generation completed in {elapsed_time:.2f}s")

        try:
            # Extract model type
            model_type = None
            if hasattr(model_wrapper.model, "config") and hasattr(
                model_wrapper.model.config, "model_type"
            ):
                model_type = model_wrapper.model.config.model_type

            # Format response with proper error handling
            response = GenerationResponse(
                generated_text=generation_result["generated_text"],
                raw_generated_text=generation_result.get("raw_generated_text", ""),
                steps=[],  # Populated below
                timing=TimingInfo(
                    generation_time=generation_result.get(
                        "generation_time", elapsed_time
                    ),
                    pruning_time=generation_result.get("pruning_time", 0.0),
                    elapsed_time=elapsed_time,
                ),
                model_info=ModelInfo(
                    model_name=components[0].last_loaded_model or config.model.model_id,
                    is_qwen_model=generation_result.get("is_qwen_model", False),
                    use_custom_rope=request.advanced_settings.use_custom_rope,
                    device=generator.device,
                    model_type=model_type,
                ),
                selection_threshold=request.selection_threshold,
                max_tokens=request.max_tokens,
                min_steps=request.min_steps,
                prompt=request.prompt,
                had_repetition_loop=generation_result.get("had_repetition_loop", False),
                system_content=system_content,
                position_to_tokens=generation_result.get("position_to_tokens", {}),
                original_parallel_positions=list(
                    generation_result.get("original_parallel_positions", set())
                ),
                tokens_by_position=generation_result.get("tokens_by_position", {}),
                final_pruned_sets=generation_result.get("final_pruned_sets", {}),
                raw_token_data=[],  # Populated below
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
                            if (
                                isinstance(original_data, tuple)
                                and len(original_data) == 2
                                and isinstance(pruned_data, tuple)
                                and len(pruned_data) == 2
                            ):

                                original_ids, original_probs = original_data
                                pruned_ids_raw, pruned_probs_raw = pruned_data

                                # Convert to basic types safely
                                original_pairs = [
                                    (int(tid), float(prob))
                                    for tid, prob in zip(original_ids, original_probs)
                                ]
                                pruned_pairs = [
                                    (int(tid), float(prob))
                                    for tid, prob in zip(
                                        pruned_ids_raw, pruned_probs_raw
                                    )
                                ]

                                # Build step info with proper token info
                                try:
                                    # Create token info objects for parallel tokens
                                    parallel_tokens = [
                                        TokenInfo(
                                            token_text=tokenizer.decode([tid]),
                                            token_id=tid,
                                            probability=prob,
                                        )
                                        for tid, prob in original_pairs
                                    ]

                                    # Create token info objects for pruned tokens
                                    pruned_tokens_info = [
                                        TokenInfo(
                                            token_text=tokenizer.decode([tid]),
                                            token_id=tid,
                                            probability=prob,
                                        )
                                        for tid, prob in pruned_pairs
                                    ]

                                    # Add step info to list
                                    steps_list.append(
                                        StepInfo(
                                            position=position,
                                            parallel_tokens=parallel_tokens,
                                            pruned_tokens=pruned_tokens_info,
                                        )
                                    )

                                    # Add raw token data for visualization
                                    raw_token_data.append(
                                        TokenSetData(
                                            position=position,
                                            original_tokens=[
                                                Token(
                                                    id=tid,
                                                    text=tokenizer.decode([tid]),
                                                    probability=prob,
                                                )
                                                for tid, prob in original_pairs
                                            ],
                                            pruned_tokens=[
                                                Token(
                                                    id=tid,
                                                    text=tokenizer.decode([tid]),
                                                    probability=prob,
                                                )
                                                for tid, prob in pruned_pairs
                                            ],
                                        )
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Error processing tokens for step {position}: {e}"
                                    )
                                    continue
                            else:
                                logger.warning(
                                    f"Skipping malformed token_set inner data: {step_data}"
                                )
                        else:
                            logger.warning(
                                f"Skipping malformed token_set step data: {step_data}"
                            )
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
                    pruning_mode=request.pruning_settings.pruning_mode.value,
                )

            # Cache generation results for visualization (cleaned up in background)
            generation_id = str(int(time.time() * 1000))
            generation_cache[generation_id] = {
                "result": generation_result,
                "response": response,
                "timestamp": time.time(),
            }

            # Schedule cleanup of old cache entries
            background_tasks.add_task(clean_generation_cache)

            return response

        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to format generation response: {str(e)}",
            )

    except ValueError as e:
        logger.error(f"Validation Error during generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Generation parameter error: {str(e)}",
        )
    except RuntimeError as e:
        logger.error(f"Runtime Error during generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected Error during generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


@router.get(
    "/models/list",
    summary="List Available Models",
    description="Lists models that are available for loading.",
    response_description="List of available models.",
    response_model=ModelsListResponse,
    status_code=status.HTTP_200_OK,
    tags=["Model Management"],
)
async def list_models():
    """
    List available models that can be loaded.

    Currently returns a fixed list but could be extended to check local cached models.

    Returns:
        ModelsListResponse: List of available models with metadata
    """
    # For now, just return a fixed list of models that are known to work
    # This could be extended to scan a models directory or check an API
    model_info = AvailableModelInfo(
        id="deepcogito/cogito-v1-preview-llama-3B",
        name="Cogito v1 Preview (Llama 3B)",
        description="Optimized for performance with TEMPO generation",
        is_default=True,
        size="3B",
        parameters={"base_model": "llama", "version": "v1"},
    )

    # If configuration has a different model, add it as well
    models = [model_info]
    if config.model.model_id != "deepcogito/cogito-v1-preview-llama-3B":
        config_model = AvailableModelInfo(
            id=config.model.model_id,
            name=config.model.model_id.split("/")[-1],
            description="Model from configuration",
            is_default=True,
            size=None,
        )
        models = [config_model, model_info]

    from src.api.model import ModelSingleton

    return ModelsListResponse(
        models=models,
        current_model=ModelSingleton.last_loaded_model or config.model.model_id,
    )


@router.delete(
    "/cache/clear",
    summary="Clear Cache",
    description="Clears generation cache to free memory.",
    response_description="Confirmation of cache clearing.",
    status_code=status.HTTP_200_OK,
    tags=["System"],
)
async def clear_cache_endpoint():
    """
    Clear the generation cache to free memory.

    Returns:
        Dict: Confirmation message
    """
    cache_size = clear_cache()

    return {"message": f"Cache cleared successfully", "entries_removed": cache_size}


@router.get(
    "/history",
    summary="Generation History",
    description="Returns a list of recent generations.",
    response_description="List of recent generations with metadata.",
    status_code=status.HTTP_200_OK,
    tags=["Visualization"],
)
async def get_generation_history(
    limit: int = Query(
        10, ge=1, le=100, description="Maximum number of history items to return"
    )
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
            history.append(
                {
                    "id": generation_id,
                    "timestamp": entry["timestamp"],
                    "prompt": (
                        entry["response"].prompt[:100] + "..."
                        if len(entry["response"].prompt) > 100
                        else entry["response"].prompt
                    ),
                    "length": len(entry["response"].generated_text),
                    "elapsed_time": entry["response"].timing.elapsed_time,
                }
            )
        except Exception as e:
            logger.error(f"Error processing history entry {generation_id}: {e}")

    # Sort by timestamp (newest first) and limit
    history.sort(key=lambda x: x["timestamp"], reverse=True)
    history = history[:limit]

    return {"history": history, "total_entries": len(generation_cache)}

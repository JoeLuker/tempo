"""Service for handling text generation workflow.

This module orchestrates the text generation process using clean architecture principles.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional

from src.presentation.api.models.requests import GenerationRequest
from src.presentation.api.models.responses import GenerationResponse
from src.infrastructure.models.model_repository import ModelRepository
from .response_formatter import ResponseFormatter
from src.pruning import RetroactiveRemover

logger = logging.getLogger(__name__)


class GenerationService:
    """Service for handling text generation workflow."""
    
    def __init__(self):
        """Initialize the generation service."""
        self.model_repository = ModelRepository()
        self.response_formatter = ResponseFormatter()
    
    def generate_text(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using TEMPO parallel generation.
        
        Args:
            request: Generation request parameters
            
        Returns:
            GenerationResponse with generated text and metadata
            
        Raises:
            Exception: If generation fails
        """
        start_time = time.time()
        
        try:
            # Get model components
            model_wrapper, tokenizer, generator, shared_token_generator = self.model_repository.get_model_components()
            
            # Propagate debug mode
            self._set_debug_mode(request.debug_mode, shared_token_generator, generator, model_wrapper)
            
            # Log the request
            logger.info(
                f"Received generation request: prompt={request.prompt[:50]}..., "
                f"max_tokens={request.max_tokens}, debug={request.debug_mode}"
            )
            
            # Create retroactive remover if needed
            retroactive_remover = self._create_retroactive_remover(
                request, model_wrapper, tokenizer, generator.device, shared_token_generator
            )
            
            # Configure RoPE modifier if needed
            self._configure_rope_modifier(request, generator)
            
            # Prepare system content
            system_content = self._prepare_system_content(request)
            
            # Generate text
            generation_result = self._perform_generation(
                request, generator, retroactive_remover, system_content
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Generation completed in {elapsed_time:.2f}s")
            
            # Format response
            return self.response_formatter.format_response(
                request, generation_result, elapsed_time, 
                system_content, tokenizer, str(generator.device)
            )
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _set_debug_mode(self, debug_mode: bool, token_generator, generator, model_wrapper) -> None:
        """Set debug mode on all components."""
        # Set on the shared TokenGenerator
        token_generator.set_debug_mode(debug_mode)
        # Set on the singleton ParallelGenerator
        if hasattr(generator, "set_debug_mode"):
            generator.set_debug_mode(debug_mode)
        # Set on the Model Wrapper
        if hasattr(model_wrapper, "set_debug_mode"):
            model_wrapper.set_debug_mode(debug_mode)
    
    def _prepare_system_content(self, request: GenerationRequest) -> Optional[str]:
        """Prepare system content for generation."""
        system_content = request.system_content
        if request.enable_thinking and not system_content:
            system_content = "Enable deep thinking subroutine."
        return system_content
    
    def _configure_rope_modifier(self, request: GenerationRequest, generator) -> None:
        """Configure RoPE modifier if needed."""
        if (
            generator.use_custom_rope
            and hasattr(generator, "rope_modifier")
            and generator.rope_modifier is not None
        ):
            if hasattr(generator.rope_modifier, "enable_kv_cache_consistency"):
                if request.disable_kv_cache_consistency:
                    logger.info(
                        "Note: RoPE modifier KV cache consistency setting ignored (likely deprecated)."
                    )
            # Set debug mode on RoPE modifier instance
            generator.rope_modifier.set_debug_mode(request.debug_mode)
    
    def _create_retroactive_remover(
        self, request: GenerationRequest, model_wrapper, tokenizer, 
        device, shared_token_generator
    ) -> Optional[RetroactiveRemover]:
        """Create retroactive remover if enabled."""
        if not request.use_retroactive_removal:
            return None
            
        try:
            # Create RetroactiveRemover
            retroactive_remover = RetroactiveRemover(
                model=model_wrapper,
                tokenizer=tokenizer,
                device=device,
                debug_mode=request.debug_mode,
                use_relative_attention=not request.no_relative_attention,
                relative_threshold=request.relative_threshold,
                use_multi_scale_attention=not request.no_multi_scale_attention,
                num_layers_to_use=request.num_layers_to_use,
                use_sigmoid_threshold=not request.no_sigmoid_threshold,
                sigmoid_steepness=request.sigmoid_steepness,
                complete_pruning_mode=request.complete_removal_mode,
            )
            
            # Set the SHARED token generator on the retroactive remover
            if hasattr(retroactive_remover, "set_token_generator"):
                retroactive_remover.set_token_generator(shared_token_generator)
                logger.info(
                    f"Set shared TokenGenerator (ID: {id(shared_token_generator)}) on RetroactiveRemover"
                )
            else:
                logger.warning(
                    "RetroactiveRemover does not have set_token_generator method."
                )
            
            logger.info(
                f"Created retroactive remover with threshold: {request.attention_threshold}"
            )
            
            return retroactive_remover
            
        except Exception as e:
            logger.error(f"Failed to initialize retroactive removal: {e}")
            logger.error(traceback.format_exc())
            raise Exception(f"Failed to initialize retroactive removal: {str(e)}")
    
    def _perform_generation(
        self, request: GenerationRequest, generator, 
        retroactive_remover, system_content: Optional[str]
    ) -> Dict[str, Any]:
        """Perform the actual text generation."""
        try:
            return generator.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                selection_threshold=request.selection_threshold,
                return_parallel_sets=True,
                use_retroactive_removal=request.use_retroactive_removal,
                min_steps=request.min_steps,
                show_token_ids=request.show_token_ids,
                disable_kv_cache=request.disable_kv_cache,
                system_content=system_content,
                isolate_parallel_tokens=not request.allow_intraset_token_visibility,
                preserve_all_isolated_tokens=(
                    not request.no_preserve_isolated_tokens
                    if not request.allow_intraset_token_visibility
                    else None
                ),
                retroactive_remover=retroactive_remover,
                # MCTS parameters
                use_mcts=request.use_mcts,
                mcts_simulations=request.mcts_simulations,
                mcts_c_puct=request.mcts_c_puct,
                mcts_depth=request.mcts_depth,
                # Dynamic threshold parameters
                dynamic_threshold=request.dynamic_threshold,
                final_threshold=request.final_threshold,
                bezier_p1=request.bezier_p1,
                bezier_p2=request.bezier_p2,
                use_relu=request.use_relu,
                relu_activation_point=request.relu_activation_point,
            )
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            logger.error(traceback.format_exc())
            raise
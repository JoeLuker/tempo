"""Monadic generation service using functional programming patterns.

This module implements the generation service using monadic design patterns
for better error handling, composition, and dependency injection.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from src.domain.monads import Result, Ok, Err, IO, IOResult, Reader, ReaderT, Maybe, some, nothing
from src.presentation.api.models.requests import GenerationRequest
from src.presentation.api.models.responses import GenerationResponse
from src.infrastructure.models.model_repository import ModelRepository
from .response_formatter import ResponseFormatter
from src.pruning import RetroactiveRemover

logger = logging.getLogger(__name__)


@dataclass
class GenerationDependencies:
    """Dependencies for generation service."""
    model_repository: ModelRepository
    response_formatter: ResponseFormatter
    debug_mode: bool = False


@dataclass
class GenerationContext:
    """Context for a generation operation."""
    request: GenerationRequest
    model_wrapper: Any
    tokenizer: Any
    generator: Any
    token_generator: Any
    retroactive_remover: Optional[RetroactiveRemover]
    system_content: Optional[str]
    start_time: float


class MonadicGenerationService:
    """Generation service using monadic patterns for better composition and error handling."""
    
    def __init__(self):
        """Initialize the monadic generation service."""
        self.model_repository = ModelRepository()
        self.response_formatter = ResponseFormatter()
    
    def generate_text(self, request: GenerationRequest) -> Result[GenerationResponse, str]:
        """Generate text using monadic composition.
        
        Args:
            request: Generation request parameters
            
        Returns:
            Result monad containing either GenerationResponse or error message
        """
        # Create dependencies
        deps = GenerationDependencies(
            model_repository=self.model_repository,
            response_formatter=self.response_formatter,
            debug_mode=request.debug_mode
        )
        
        # Build the generation pipeline using monadic composition
        generation_pipeline = (
            self._create_context(request)
            .flat_map(self._set_debug_mode)
            .flat_map(self._log_request)
            .flat_map(self._create_retroactive_remover)
            .flat_map(self._configure_rope_modifier)
            .flat_map(self._prepare_system_content)
            .flat_map(self._perform_generation)
            .flat_map(self._format_response)
        )
        
        # Run the pipeline with dependencies
        return generation_pipeline.run(deps)
    
    def _create_context(
        self, request: GenerationRequest
    ) -> ReaderT[GenerationDependencies, GenerationContext]:
        """Create initial generation context."""
        def create(deps: GenerationDependencies) -> Result[GenerationContext, str]:
            try:
                # Get model components
                components = deps.model_repository.get_model_components()
                model_wrapper, tokenizer, generator, token_generator = components
                
                context = GenerationContext(
                    request=request,
                    model_wrapper=model_wrapper,
                    tokenizer=tokenizer,
                    generator=generator,
                    token_generator=token_generator,
                    retroactive_remover=None,
                    system_content=None,
                    start_time=time.time()
                )
                
                return Ok(context)
            except Exception as e:
                return Err(f"Failed to create context: {str(e)}")
        
        return ReaderT(create)
    
    def _set_debug_mode(
        self, context: GenerationContext
    ) -> ReaderT[GenerationDependencies, GenerationContext]:
        """Set debug mode on all components."""
        def set_debug(deps: GenerationDependencies) -> Result[GenerationContext, str]:
            try:
                # Set on the shared TokenGenerator
                context.token_generator.set_debug_mode(context.request.debug_mode)
                
                # Set on the singleton ParallelGenerator
                if hasattr(context.generator, "set_debug_mode"):
                    context.generator.set_debug_mode(context.request.debug_mode)
                
                # Set on the Model Wrapper
                if hasattr(context.model_wrapper, "set_debug_mode"):
                    context.model_wrapper.set_debug_mode(context.request.debug_mode)
                
                return Ok(context)
            except Exception as e:
                return Err(f"Failed to set debug mode: {str(e)}")
        
        return ReaderT(set_debug)
    
    def _log_request(
        self, context: GenerationContext
    ) -> ReaderT[GenerationDependencies, GenerationContext]:
        """Log the generation request."""
        def log_req(deps: GenerationDependencies) -> Result[GenerationContext, str]:
            # This is a side effect, so we wrap it in IO
            log_io = IO(lambda: logger.info(
                f"Received generation request: prompt={context.request.prompt[:50]}..., "
                f"max_tokens={context.request.max_tokens}, debug={context.request.debug_mode}"
            ))
            
            # Execute the side effect
            log_io.run()
            
            return Ok(context)
        
        return ReaderT(log_req)
    
    def _create_retroactive_remover(
        self, context: GenerationContext
    ) -> ReaderT[GenerationDependencies, GenerationContext]:
        """Create retroactive remover if enabled."""
        def create_remover(deps: GenerationDependencies) -> Result[GenerationContext, str]:
            if not context.request.use_retroactive_removal:
                return Ok(context)
            
            try:
                # Create RetroactiveRemover
                retroactive_remover = RetroactiveRemover(
                    model=context.model_wrapper,
                    tokenizer=context.tokenizer,
                    device=context.generator.device,
                    debug_mode=context.request.debug_mode,
                    use_relative_attention=not context.request.no_relative_attention,
                    relative_threshold=context.request.relative_threshold,
                    use_multi_scale_attention=not context.request.no_multi_scale_attention,
                    num_layers_to_use=context.request.num_layers_to_use,
                    use_sigmoid_threshold=not context.request.no_sigmoid_threshold,
                    sigmoid_steepness=context.request.sigmoid_steepness,
                    complete_pruning_mode=context.request.complete_removal_mode,
                )
                
                # Set the SHARED token generator on the retroactive remover
                if hasattr(retroactive_remover, "set_token_generator"):
                    retroactive_remover.set_token_generator(context.token_generator)
                    logger.info(
                        f"Set shared TokenGenerator (ID: {id(context.token_generator)}) on RetroactiveRemover"
                    )
                else:
                    logger.warning(
                        "RetroactiveRemover does not have set_token_generator method."
                    )
                
                logger.info(
                    f"Created retroactive remover with threshold: {context.request.attention_threshold}"
                )
                
                # Update context with retroactive remover
                context.retroactive_remover = retroactive_remover
                
                return Ok(context)
                
            except Exception as e:
                return Err(f"Failed to initialize retroactive removal: {str(e)}")
        
        return ReaderT(create_remover)
    
    def _configure_rope_modifier(
        self, context: GenerationContext
    ) -> ReaderT[GenerationDependencies, GenerationContext]:
        """Configure RoPE modifier if needed."""
        def configure_rope(deps: GenerationDependencies) -> Result[GenerationContext, str]:
            try:
                if (
                    context.generator.use_custom_rope
                    and hasattr(context.generator, "rope_modifier")
                    and context.generator.rope_modifier is not None
                ):
                    if hasattr(context.generator.rope_modifier, "enable_kv_cache_consistency"):
                        if context.request.disable_kv_cache_consistency:
                            logger.info(
                                "Note: RoPE modifier KV cache consistency setting ignored (likely deprecated)."
                            )
                    # Set debug mode on RoPE modifier instance
                    context.generator.rope_modifier.set_debug_mode(context.request.debug_mode)
                
                return Ok(context)
            except Exception as e:
                return Err(f"Failed to configure RoPE modifier: {str(e)}")
        
        return ReaderT(configure_rope)
    
    def _prepare_system_content(
        self, context: GenerationContext
    ) -> ReaderT[GenerationDependencies, GenerationContext]:
        """Prepare system content for generation."""
        def prepare_content(deps: GenerationDependencies) -> Result[GenerationContext, str]:
            system_content = context.request.system_content
            if context.request.enable_thinking and not system_content:
                system_content = "Enable deep thinking subroutine."
            
            context.system_content = system_content
            return Ok(context)
        
        return ReaderT(prepare_content)
    
    def _perform_generation(
        self, context: GenerationContext
    ) -> ReaderT[GenerationDependencies, GenerationContext]:
        """Perform the actual text generation."""
        def generate(deps: GenerationDependencies) -> Result[GenerationContext, str]:
            try:
                generation_result = context.generator.generate(
                    prompt=context.request.prompt,
                    max_tokens=context.request.max_tokens,
                    selection_threshold=context.request.selection_threshold,
                    return_parallel_sets=True,
                    use_retroactive_removal=context.request.use_retroactive_removal,
                    min_steps=context.request.min_steps,
                    show_token_ids=context.request.show_token_ids,
                    disable_kv_cache=context.request.disable_kv_cache,
                    system_content=context.system_content,
                    isolate_parallel_tokens=not context.request.allow_intraset_token_visibility,
                    preserve_all_isolated_tokens=(
                        not context.request.no_preserve_isolated_tokens
                        if not context.request.allow_intraset_token_visibility
                        else None
                    ),
                    retroactive_remover=context.retroactive_remover,
                    # MCTS parameters
                    use_mcts=context.request.use_mcts,
                    mcts_simulations=context.request.mcts_simulations,
                    mcts_c_puct=context.request.mcts_c_puct,
                    mcts_depth=context.request.mcts_depth,
                    # Dynamic threshold parameters
                    dynamic_threshold=context.request.dynamic_threshold,
                    final_threshold=context.request.final_threshold,
                    bezier_p1=context.request.bezier_p1,
                    bezier_p2=context.request.bezier_p2,
                    use_relu=context.request.use_relu,
                    relu_activation_point=context.request.relu_activation_point,
                )
                
                # Store result in context
                context.generation_result = generation_result
                return Ok(context)
                
            except Exception as e:
                return Err(f"Error during generation: {str(e)}")
        
        return ReaderT(generate)
    
    def _format_response(
        self, context: GenerationContext
    ) -> ReaderT[GenerationDependencies, GenerationResponse]:
        """Format the generation response."""
        def format_resp(deps: GenerationDependencies) -> Result[GenerationResponse, str]:
            try:
                elapsed_time = time.time() - context.start_time
                
                # Log completion
                logger.info(f"Generation completed in {elapsed_time:.2f}s")
                
                # Format response
                response = deps.response_formatter.format_response(
                    context.request, 
                    context.generation_result, 
                    elapsed_time,
                    context.system_content, 
                    context.tokenizer, 
                    str(context.generator.device)
                )
                
                return Ok(response)
                
            except Exception as e:
                return Err(f"Failed to format response: {str(e)}")
        
        return ReaderT(format_resp)


# Helper functions for creating monadic pipelines

def validate_request(request: GenerationRequest) -> Result[GenerationRequest, str]:
    """Validate a generation request."""
    if not request.prompt:
        return Err("Prompt cannot be empty")
    
    if request.max_tokens <= 0:
        return Err("Max tokens must be positive")
    
    if request.selection_threshold < 0 or request.selection_threshold > 1:
        return Err("Selection threshold must be between 0 and 1")
    
    return Ok(request)


def with_retry(
    operation: Callable[[], Result[T, E]], 
    max_retries: int = 3,
    delay: float = 1.0
) -> Result[T, E]:
    """Retry an operation with exponential backoff."""
    import time
    
    for attempt in range(max_retries):
        result = operation()
        if result.is_ok():
            return result
        
        if attempt < max_retries - 1:
            time.sleep(delay * (2 ** attempt))
    
    return result


def log_result(result: Result[T, E]) -> Result[T, E]:
    """Log the result of an operation."""
    result.fold(
        lambda err: logger.error(f"Operation failed: {err}"),
        lambda val: logger.debug(f"Operation succeeded")
    )
    return result
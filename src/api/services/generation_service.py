"""
Generation service with clean separation of concerns.
"""
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
from src.api.schemas.generation_v2 import (
    GenerationRequestV2,
    GenerationResponseV2,
    GenerationStep,
    TokenChoice,
    GenerationMetadata,
    PruningMode
)
from src.modeling.model_wrapper import TEMPOModelWrapper
from src.generation.parallel_generator import ParallelGenerator
from src.pruning import RetroactivePruner
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class GenerationService:
    """Service for handling text generation requests."""
    
    def __init__(self, model_wrapper: TEMPOModelWrapper, 
                 tokenizer: PreTrainedTokenizerBase,
                 generator: ParallelGenerator):
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer
        self.generator = generator
    
    def generate(self, request: GenerationRequestV2) -> GenerationResponseV2:
        """Process a generation request and return structured response."""
        start_time = time.time()
        
        # Configure components based on request
        self._configure_debug_mode(request.debug.debug_mode)
        
        # Create pruner if needed
        pruner = self._create_pruner(request) if request.pruning and request.pruning.enabled else None
        
        # Perform generation
        result = self._generate_text(request, pruner)
        
        # Process the result into clean response
        response = self._build_response(request, result, start_time)
        
        return response
    
    def _configure_debug_mode(self, debug_mode: bool):
        """Configure debug mode across components."""
        if hasattr(self.generator, 'set_debug_mode'):
            self.generator.set_debug_mode(debug_mode)
        if hasattr(self.model_wrapper, 'set_debug_mode'):
            self.model_wrapper.set_debug_mode(debug_mode)
    
    def _create_pruner(self, request: GenerationRequestV2) -> Optional[RetroactivePruner]:
        """Create retroactive pruner with proper configuration."""
        pruning_config = request.pruning
        
        # Map our clean enum to the legacy string value
        pruning_mode_map = {
            PruningMode.KEEP_BEST: "keep_token",
            PruningMode.MARK_UNATTENDED: "keep_unattended", 
            PruningMode.REMOVE_POSITION: "remove_position"
        }
        
        pruner = RetroactivePruner(
            model=self.model_wrapper,
            tokenizer=self.tokenizer,
            device=self.generator.device,
            debug_mode=request.debug.debug_mode,
            use_relative_attention=pruning_config.use_relative_attention,
            relative_threshold=pruning_config.relative_threshold,
            num_layers_to_use=pruning_config.layers_to_analyze,
            complete_pruning_mode=pruning_mode_map[pruning_config.pruning_mode]
        )
        
        # Set token generator if available
        if hasattr(self.generator, 'token_generator') and hasattr(pruner, 'set_token_generator'):
            pruner.set_token_generator(self.generator.token_generator)
            
        return pruner
    
    def _generate_text(self, request: GenerationRequestV2, pruner: Optional[Any]) -> Dict[str, Any]:
        """Call the generator with proper parameters."""
        # Build generation kwargs
        generation_kwargs = {
            "prompt": request.prompt,
            "max_tokens": request.generation.max_tokens,
            "selection_threshold": request.generation.selection_threshold,
            "min_steps": request.generation.min_steps_before_eos,
            "return_parallel_sets": True,  # Always need this for detailed response
            "use_custom_rope": request.rope.use_custom_rope,
            "disable_kv_cache": request.rope.disable_kv_cache,
            "show_token_ids": request.debug.show_token_ids,
            "debug_mode": request.debug.debug_mode,
            "system_content": request.system_prompt,
        }
        
        # Add MCTS parameters if configured
        if request.mcts and request.mcts.enabled:
            generation_kwargs.update({
                "use_mcts": True,
                "mcts_simulations": request.mcts.simulations_per_step,
                "mcts_c_puct": request.mcts.exploration_constant,
                "mcts_depth": request.mcts.max_depth,
            })
        
        # Add dynamic threshold if configured
        if request.dynamic_threshold and request.dynamic_threshold.enabled:
            generation_kwargs.update({
                "dynamic_threshold": True,
                "final_threshold": request.dynamic_threshold.final_threshold,
                "use_relu": request.dynamic_threshold.curve_type == "relu",
                "bezier_p1": request.dynamic_threshold.bezier_control_points[0],
                "bezier_p2": request.dynamic_threshold.bezier_control_points[1],
                "relu_activation_point": request.dynamic_threshold.relu_activation_point,
            })
        
        # Add pruning configuration
        if pruner:
            generation_kwargs.update({
                "use_retroactive_removal": True,
                "retroactive_pruner": pruner,
                "attention_threshold": request.pruning.attention_threshold,
            })
        
        return self.generator.generate(**generation_kwargs)
    
    def _build_response(self, request: GenerationRequestV2, 
                       result: Dict[str, Any], 
                       start_time: float) -> GenerationResponseV2:
        """Build clean response from generation result."""
        elapsed_time = time.time() - start_time
        
        # Extract clean text
        clean_text = self._extract_clean_text(result.get("generated_text", ""))
        
        # Process token sets into generation steps
        steps = self._process_token_sets(result.get("token_sets", []))
        
        # Build metadata
        metadata = GenerationMetadata(
            total_tokens_generated=len(steps),
            total_tokens_considered=sum(len(step.considered_tokens) for step in steps),
            parallel_positions=[step.position for step in steps if step.had_parallel_paths],
            generation_time_seconds=result.get("generation_time", elapsed_time),
            pruning_time_seconds=result.get("pruning_time", 0.0),
            model_name=self.model_wrapper.model.config.name_or_path,
            device=str(self.generator.device),
        )
        
        # Build settings used
        settings_used = {
            "generation": request.generation.model_dump(),
            "rope": request.rope.model_dump(),
        }
        if request.mcts and request.mcts.enabled:
            settings_used["mcts"] = request.mcts.model_dump()
        if request.dynamic_threshold and request.dynamic_threshold.enabled:
            settings_used["dynamic_threshold"] = request.dynamic_threshold.model_dump()
        if request.pruning and request.pruning.enabled:
            settings_used["pruning"] = request.pruning.model_dump()
        
        # Build debug info if requested
        debug_info = None
        if request.debug.debug_mode:
            debug_info = {
                "raw_generated_text": result.get("raw_generated_text", ""),
                "position_to_tokens": result.get("position_to_tokens", {}),
                "had_repetition_loop": result.get("had_repetition_loop", False),
            }
        
        return GenerationResponseV2(
            text=clean_text,
            steps=steps,
            metadata=metadata,
            prompt=request.prompt,
            settings_used=settings_used,
            debug_info=debug_info,
        )
    
    def _extract_clean_text(self, text_with_brackets: str) -> str:
        """Extract clean text from TEMPO's bracketed format."""
        if not text_with_brackets:
            return ""
        
        # Remove ANSI codes
        import re
        text = re.sub(r'\x1b\[[0-9;]*m', '', text_with_brackets)
        
        # Extract first token from each bracket group
        result = []
        i = 0
        while i < len(text):
            if text[i] == '[':
                # Find matching bracket
                j = i + 1
                bracket_depth = 1
                while j < len(text) and bracket_depth > 0:
                    if text[j] == '[':
                        bracket_depth += 1
                    elif text[j] == ']':
                        bracket_depth -= 1
                    j += 1
                
                if bracket_depth == 0:
                    # Extract content and take first token
                    bracket_content = text[i+1:j-1]
                    tokens = [t.strip() for t in bracket_content.split('/')]
                    if tokens and tokens[0]:
                        result.append(tokens[0])
                    i = j
                else:
                    result.append(text[i])
                    i += 1
            else:
                result.append(text[i])
                i += 1
        
        return ''.join(result)
    
    def _process_token_sets(self, token_sets: List[Tuple]) -> List[GenerationStep]:
        """Process raw token sets into structured generation steps."""
        steps = []
        
        for position, original_tokens, selected_tokens in token_sets:
            # Decode tokens
            considered = []
            selected = []
            
            # Create set of selected token IDs for quick lookup
            selected_ids = set()
            if isinstance(selected_tokens, list) and len(selected_tokens) > 0:
                if isinstance(selected_tokens[0], tuple):
                    selected_ids = {tid for tid, _ in selected_tokens}
                else:
                    selected_ids = set(selected_tokens)
            
            # Process original tokens (all considered tokens)
            if isinstance(original_tokens, list) and len(original_tokens) > 0:
                for item in original_tokens:
                    if isinstance(item, tuple) and len(item) >= 2:
                        token_id, probability = item[0], item[1]
                        token_text = self.tokenizer.decode([token_id])
                        
                        token_choice = TokenChoice(
                            token_id=token_id,
                            token_text=token_text,
                            probability=float(probability),
                            was_selected=token_id in selected_ids
                        )
                        considered.append(token_choice)
                        
                        if token_id in selected_ids:
                            selected.append(token_choice)
            
            step = GenerationStep(
                position=position,
                considered_tokens=considered,
                selected_tokens=selected,
                had_parallel_paths=len(selected) > 1
            )
            steps.append(step)
        
        return steps
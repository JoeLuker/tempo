"""Standard generation strategy implementation.

This module implements the standard threshold-based token selection strategy.
"""

import torch
from typing import List, Optional, Any

from ...domain.entities.token import TokenSet, Token
from ...domain.entities.logits import TokenLogits
from ...domain.entities.generation_state import GenerationState
from ...domain.entities.parallel_generation import GenerationConfig
from ...domain.interfaces.generation_strategy import GenerationStrategy, ThresholdStrategy
from ...domain.interfaces.token_selection import TokenSelectorInterface
from ...domain.interfaces.tokenizer import TokenizerInterface
from ...utils.logging_utils import LoggingMixin


class StandardGenerationStrategy(GenerationStrategy, LoggingMixin):
    """Standard threshold-based generation strategy."""
    
    def __init__(
        self,
        token_selector: TokenSelectorInterface,
        threshold_strategy: Optional[ThresholdStrategy] = None,
        tokenizer: Optional[TokenizerInterface] = None,
        debug_mode: bool = False
    ):
        """Initialize standard generation strategy.
        
        Args:
            token_selector: Token selector for threshold-based selection
            threshold_strategy: Optional dynamic threshold strategy
            tokenizer: Optional tokenizer for EOS detection
            debug_mode: Whether to enable debug logging
        """
        super().__init__()
        self.setup_logging("standard_generation_strategy", "strategy.log", debug_mode)
        
        self.token_selector = token_selector
        self.threshold_strategy = threshold_strategy
        self.tokenizer = tokenizer
    
    def select_tokens(
        self,
        logits: TokenLogits,
        step: int,
        config: GenerationConfig,
        state: GenerationState
    ) -> TokenSet:
        """Select tokens based on threshold.
        
        Args:
            logits: Raw logits from the model
            step: Current generation step
            config: Generation configuration
            state: Current generation state
            
        Returns:
            TokenSet containing selected tokens
        """
        # Calculate threshold for this step
        if self.threshold_strategy and config.dynamic_threshold:
            threshold = self.threshold_strategy.calculate_threshold(step, config.max_tokens)
            self.log(f"Dynamic threshold at step {step}: {threshold:.4f}")
        else:
            threshold = config.selection_threshold
        
        # Select tokens using the token selector
        token_distribution, subset_size = self.token_selector.select_tokens(
            logits.tensor,
            threshold=threshold
        )
        
        if not token_distribution:
            # Fallback to top token if nothing selected
            self.log(f"No tokens above threshold {threshold}, using top token", "warning")
            token_distribution, _ = self.token_selector.select_tokens(
                logits.tensor,
                threshold=0.0,
                max_tokens=1
            )
        
        # Convert to Token objects
        tokens = [
            Token(
                id=tid.item() if hasattr(tid, 'item') else int(tid),
                text="",  # Will be filled later if needed
                probability=float(prob),
                logit=0.0,  # Could extract if needed
                position=step
            )
            for tid, prob in token_distribution
        ]
        
        return TokenSet(
            tokens=tokens,
            position=step,
            is_parallel=len(tokens) > 1
        )
    
    def should_terminate(
        self,
        token_set: TokenSet,
        state: GenerationState
    ) -> bool:
        """Check if generation should terminate.
        
        Args:
            token_set: Most recently generated token set
            state: Current generation state
            
        Returns:
            True if generation should stop
        """
        # Check for EOS token
        if self.tokenizer and hasattr(self.tokenizer, 'eos_token_id'):
            eos_id = self.tokenizer.eos_token_id
            if any(token.id == eos_id for token in token_set.tokens):
                self.log("EOS token generated, terminating")
                return True
        
        return False


class DynamicThresholdStrategy(ThresholdStrategy):
    """Dynamic threshold strategy using Bezier curve or ReLU."""
    
    def __init__(
        self,
        initial_threshold: float,
        final_threshold: float,
        bezier_p1: float = 0.2,
        bezier_p2: float = 0.8,
        use_relu: bool = False,
        relu_activation_point: float = 0.5
    ):
        """Initialize dynamic threshold strategy.
        
        Args:
            initial_threshold: Starting threshold
            final_threshold: Ending threshold
            bezier_p1: First Bezier control point
            bezier_p2: Second Bezier control point
            use_relu: Whether to use ReLU transition
            relu_activation_point: Point at which ReLU activates
        """
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.bezier_p1 = bezier_p1
        self.bezier_p2 = bezier_p2
        self.use_relu = use_relu
        self.relu_activation_point = relu_activation_point
    
    def calculate_threshold(self, step: int, max_steps: int) -> float:
        """Calculate the threshold for a given step.
        
        Args:
            step: Current generation step
            max_steps: Maximum number of steps
            
        Returns:
            Threshold value between 0 and 1
        """
        if max_steps == 0:
            return self.initial_threshold
        
        # Calculate progress
        progress = min(step / max_steps, 1.0)
        
        if self.use_relu:
            # ReLU transition
            if progress < self.relu_activation_point:
                return self.initial_threshold
            else:
                # Linear transition after activation
                transition_progress = (progress - self.relu_activation_point) / (
                    1.0 - self.relu_activation_point
                )
                return (
                    self.initial_threshold
                    + (self.final_threshold - self.initial_threshold)
                    * transition_progress
                )
        else:
            # Bezier curve transition
            t = progress
            p0 = self.initial_threshold
            p3 = self.final_threshold
            
            # Cubic Bezier formula
            threshold = (
                (1 - t) ** 3 * p0
                + 3 * (1 - t) ** 2 * t * self.bezier_p1
                + 3 * (1 - t) * t**2 * self.bezier_p2
                + t**3 * p3
            )
            
            return threshold

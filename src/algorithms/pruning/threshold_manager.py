"""Dynamic threshold management for pruning."""

import numpy as np
from typing import Optional, Callable


class DynamicThresholdManager:
    """Manages dynamic thresholds for attention-based pruning."""
    
    def __init__(
        self,
        base_threshold: float = 0.01,
        final_threshold: Optional[float] = None,
        use_bezier: bool = True,
        bezier_p1: float = 0.3,
        bezier_p2: float = 0.7,
        use_relu: bool = False,
        relu_activation_point: float = 0.5
    ):
        self.base_threshold = base_threshold
        self.final_threshold = final_threshold or base_threshold
        self.use_bezier = use_bezier
        self.bezier_p1 = bezier_p1
        self.bezier_p2 = bezier_p2
        self.use_relu = use_relu
        self.relu_activation_point = relu_activation_point
        
    def get_threshold_at_step(
        self,
        current_step: int,
        total_steps: int
    ) -> float:
        """
        Get dynamic threshold for current generation step.
        
        Args:
            current_step: Current step in generation
            total_steps: Total expected steps
            
        Returns:
            Threshold value for this step
        """
        if total_steps <= 1:
            return self.base_threshold
            
        # Normalize step to [0, 1]
        t = current_step / (total_steps - 1)
        
        if self.use_relu:
            multiplier = self._relu_curve(t)
        elif self.use_bezier:
            multiplier = self._bezier_curve(t)
        else:
            # Linear interpolation
            multiplier = t
            
        # Interpolate between base and final threshold
        threshold = (
            self.base_threshold * (1 - multiplier) + 
            self.final_threshold * multiplier
        )
        
        return threshold
    
    def _bezier_curve(self, t: float) -> float:
        """
        Compute cubic Bezier curve value.
        
        Control points: (0,0), (p1,p1), (p2,p2), (1,1)
        """
        return (
            3 * (1 - t)**2 * t * self.bezier_p1 +
            3 * (1 - t) * t**2 * self.bezier_p2 +
            t**3
        )
    
    def _relu_curve(self, t: float) -> float:
        """
        Compute ReLU-based curve value.
        
        Stays at 0 until activation point, then linear.
        """
        if t < self.relu_activation_point:
            return 0.0
        else:
            # Scale remaining range to [0, 1]
            return (t - self.relu_activation_point) / (1 - self.relu_activation_point)
    
    def get_adaptive_threshold(
        self,
        attention_entropy: float,
        base_multiplier: float = 1.0
    ) -> float:
        """
        Get threshold adapted to attention entropy.
        
        Higher entropy = more uncertainty = higher threshold
        """
        # Scale threshold based on entropy
        # Typical entropy range: 0 (focused) to log(seq_len) (uniform)
        entropy_factor = 1.0 + (attention_entropy / 10.0)  # Normalize
        
        return self.base_threshold * base_multiplier * entropy_factor
    
    def create_custom_curve(
        self,
        curve_fn: Callable[[float], float]
    ) -> 'DynamicThresholdManager':
        """
        Create manager with custom threshold curve.
        
        Args:
            curve_fn: Function mapping [0,1] -> [0,1]
            
        Returns:
            New threshold manager with custom curve
        """
        manager = DynamicThresholdManager(
            base_threshold=self.base_threshold,
            final_threshold=self.final_threshold
        )
        
        # Override get_threshold to use custom curve
        def custom_threshold(current_step: int, total_steps: int) -> float:
            t = current_step / (total_steps - 1) if total_steps > 1 else 0
            multiplier = curve_fn(t)
            return (
                self.base_threshold * (1 - multiplier) +
                self.final_threshold * multiplier
            )
        
        manager.get_threshold_at_step = custom_threshold
        return manager
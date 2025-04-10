import numpy as np
from typing import List, Optional, Tuple

class DynamicThresholdManager:
    """
    Manages dynamic thresholds that change over the course of generation.
    """
    
    def __init__(
        self, 
        base_threshold: float = 0.3,
        max_steps: Optional[int] = None,
        bezier_points: Optional[List[float]] = None,
        final_threshold: float = 1.0
    ):
        """
        Initialize the dynamic threshold manager.
        
        Args:
            base_threshold: Starting threshold value
            max_steps: Maximum number of steps (for calculating dynamic threshold)
            bezier_points: Control points for Bezier curve [p1, p2] between 0-1
            final_threshold: Final threshold value for dynamic threshold
        """
        self.base_threshold = base_threshold
        self.max_steps = max_steps or 100  # Default if not specified
        self.current_step = 0
        self.final_threshold = final_threshold
        
        # Default Bezier control points for exponential-like curve
        self.bezier_points = bezier_points if bezier_points is not None else [0.2, 0.8]
        
        # Storage for token sets and scores
        self.all_token_sets = []
        self.all_token_scores = []
    
    def get_current_threshold(self) -> float:
        """
        Get the current threshold value based on step progress.
        
        Returns:
            float: Current threshold value
        """
        if self.max_steps <= 1:
            return self.base_threshold
            
        # Calculate progress as a value between 0 and 1
        progress = min(1.0, self.current_step / self.max_steps)
        
        # Calculate threshold using cubic Bezier curve for smooth progression
        bezier_value = self._cubic_bezier(
            progress, 
            0.0,  # Start at 0
            self.bezier_points[0], 
            self.bezier_points[1],
            1.0   # End at 1
        )
        
        # Scale between base_threshold and final_threshold
        current_threshold = self.base_threshold + (self.final_threshold - self.base_threshold) * bezier_value
        
        return current_threshold
    
    def _cubic_bezier(self, t: float, p0: float, p1: float, p2: float, p3: float) -> float:
        """
        Calculate a point on a cubic Bezier curve.
        
        Args:
            t: Parameter between 0 and 1
            p0, p1, p2, p3: Control points
            
        Returns:
            float: Value at point t on the Bezier curve
        """
        return (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
    
    def store_token_set(
        self, 
        token_set: List[Tuple[int, float]], 
        token_scores: List[Tuple[int, float]]
    ):
        """
        Store a token set and its scores for reapplication of thresholds.
        
        Args:
            token_set: List of (token_id, probability) tuples
            token_scores: List of (token_id, normalized_score) tuples
        """
        self.all_token_sets.append(token_set.copy())
        self.all_token_scores.append(token_scores)
        
        # Increment step counter after storing
        self.current_step += 1
    
    def reapply_threshold_to_all_sets(self) -> List[List[Tuple[int, float]]]:
        """
        Reapply the current threshold to all previously processed token sets.
        
        Returns:
            List[List[Tuple[int, float]]]: Updated list of pruned token sets
        """
        if not self.all_token_sets:
            return []
            
        current_threshold = self.get_current_threshold()
        updated_pruned_sets = []
        
        for i, (token_set, score_set) in enumerate(zip(self.all_token_sets, self.all_token_scores)):
            # Apply current threshold to this set's scores
            if i == len(self.all_token_sets) - 1 and self.current_step >= self.max_steps:
                # For the last set at the final step, ensure a single token remains
                if self.final_threshold >= 0.999:  # Using 0.999 for float precision
                    # Force collapse to a single token
                    if token_set:
                        max_score_idx = max(range(len(score_set)), key=lambda j: score_set[j][1])
                        pruned_set = [token_set[max_score_idx]]
                    else:
                        pruned_set = []
                else:
                    # Apply threshold normally without forcing collapse
                    pruned_set = [
                        token_set[j] for j, (_, score) in enumerate(score_set) 
                        if score >= current_threshold
                    ]
                    
                    # If all tokens were pruned, keep the highest scoring one
                    if not pruned_set and token_set:
                        max_score_idx = max(range(len(score_set)), key=lambda j: score_set[j][1])
                        pruned_set = [token_set[max_score_idx]]
            else:
                # For other sets, apply threshold without forcing single token
                pruned_set = [
                    token_set[j] for j, (_, score) in enumerate(score_set) 
                    if score >= current_threshold
                ]
                
                # If all tokens were pruned, keep the highest scoring one
                if not pruned_set and token_set:
                    max_score_idx = max(range(len(score_set)), key=lambda j: score_set[j][1])
                    pruned_set = [token_set[max_score_idx]]
            
            updated_pruned_sets.append(pruned_set)
            
        return updated_pruned_sets
    
    def reset(self):
        """Reset the dynamic threshold for a new generation."""
        self.current_step = 0
        self.all_token_sets = []
        self.all_token_scores = [] 
import numpy as np
from typing import List, Optional, Tuple, Dict, Any


class DynamicThresholdManager:
    """
    Manages dynamic thresholds that change over the course of generation.
    """

    def __init__(
        self,
        base_threshold: float = 0.3,
        max_steps: Optional[int] = None,
        bezier_points: Optional[List[float]] = None,
        final_threshold: float = 1.0,
        max_tokens_per_step: int = 20,
    ):
        """
        Initialize the dynamic threshold manager.

        Args:
            base_threshold: Starting threshold value
            max_steps: Maximum number of steps (for calculating dynamic threshold)
            bezier_points: Control points for Bezier curve [p1, p2] between 0-1
            final_threshold: Final threshold value for dynamic threshold
            max_tokens_per_step: Maximum number of tokens expected per step
        """
        # Invariant: Thresholds must be valid values between 0 and 1
        if not (0 <= base_threshold <= 1):
            raise ValueError(
                f"Invariant violation: base_threshold must be between 0 and 1, got {base_threshold}"
            )
        if not (0 <= final_threshold <= 1):
            raise ValueError(
                f"Invariant violation: final_threshold must be between 0 and 1, got {final_threshold}"
            )

        # Invariant: Max steps must be positive
        if max_steps is not None and max_steps <= 0:
            raise ValueError(
                f"Invariant violation: max_steps must be positive, got {max_steps}"
            )

        # Invariant: Bezier points must be in valid range
        if bezier_points is not None:
            if len(bezier_points) != 2:
                raise ValueError(
                    f"Invariant violation: bezier_points must contain exactly 2 values, got {len(bezier_points)}"
                )
            if not all(0 <= p <= 1 for p in bezier_points):
                raise ValueError(
                    f"Invariant violation: bezier_points must be between 0 and 1, got {bezier_points}"
                )

        self.base_threshold = base_threshold
        self.max_steps = max_steps or 100  # Default if not specified
        self.max_tokens_per_step = max_tokens_per_step
        self.current_step = 0
        self.final_threshold = final_threshold

        # Default Bezier control points for exponential-like curve
        self.bezier_points = bezier_points if bezier_points is not None else [0.2, 0.8]

        # Define dtypes for token data
        self.token_id_dtype = np.int32
        self.prob_dtype = np.float32
        self.score_dtype = np.float32

        # Pre-allocate tensors for token data with maximum expected dimensions
        self.token_ids = np.zeros(
            (self.max_steps, self.max_tokens_per_step), dtype=self.token_id_dtype
        )
        self.token_probs = np.zeros(
            (self.max_steps, self.max_tokens_per_step), dtype=self.prob_dtype
        )
        self.token_scores = np.zeros(
            (self.max_steps, self.max_tokens_per_step), dtype=self.score_dtype
        )

        # Mask to track valid positions (1 = valid token, 0 = padding)
        self.valid_mask = np.zeros(
            (self.max_steps, self.max_tokens_per_step), dtype=bool
        )

        # Track the number of tokens at each step
        self.tokens_per_step = np.zeros(self.max_steps, dtype=np.int32)

        # Pre-allocate tensor for pruned tokens
        self.pruned_mask = np.zeros(
            (self.max_steps, self.max_tokens_per_step), dtype=bool
        )

        # Cache for optimized recomputation
        self.cached_threshold = None
        self.threshold_epsilon = 0.01  # Only recompute when threshold changes by more than this

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
            1.0,  # End at 1
        )

        # Scale between base_threshold and final_threshold
        current_threshold = (
            self.base_threshold
            + (self.final_threshold - self.base_threshold) * bezier_value
        )

        return current_threshold

    def _cubic_bezier(
        self, t: float, p0: float, p1: float, p2: float, p3: float
    ) -> float:
        """
        Calculate a point on a cubic Bezier curve.

        Args:
            t: Parameter between 0 and 1
            p0, p1, p2, p3: Control points

        Returns:
            float: Value at point t on the Bezier curve
        """
        # Invariant: t must be between 0 and 1
        if not (0 <= t <= 1):
            raise ValueError(
                f"Invariant violation: Bezier parameter t must be between 0 and 1, got {t}"
            )

        return (
            (1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t**2 * p2
            + t**3 * p3
        )

    def store_token_set(
        self, token_set: List[Tuple[int, float]], token_scores: List[Tuple[int, float]]
    ):
        """
        Store a token set and its scores for reapplication of thresholds.

        Args:
            token_set: List of (token_id, probability) tuples
            token_scores: List of (token_id, normalized_score) tuples
        """
        # Invariant: Token set and scores must be the same length
        if len(token_set) != len(token_scores):
            raise ValueError(
                f"Invariant violation: token_set length ({len(token_set)}) must match token_scores length ({len(token_scores)})"
            )

        # Invariant: All scores must be between 0 and 1
        if any(not (0 <= score <= 1) for _, score in token_scores):
            raise ValueError(
                "Invariant violation: Token scores must be between 0 and 1"
            )

        # Safety check for tensor size
        if self.current_step >= self.max_steps:
            raise ValueError(
                f"Exceeded maximum steps ({self.max_steps}). Increase max_steps in initialization."
            )

        # If token set exceeds max_tokens_per_step, keep only the top tokens by score
        if len(token_set) > self.max_tokens_per_step:
            # Sort token_scores and token_set by score (descending)
            combined = list(zip(token_set, token_scores))
            combined.sort(key=lambda x: x[1][1], reverse=True)

            # Keep only the top max_tokens_per_step tokens
            combined = combined[: self.max_tokens_per_step]
            token_set = [t[0] for t in combined]
            token_scores = [t[1] for t in combined]

        # Store the number of tokens for this step
        num_tokens = len(token_set)
        self.tokens_per_step[self.current_step] = num_tokens

        # Update the valid mask for this step
        self.valid_mask[self.current_step, :num_tokens] = True

        # Extract token IDs, probabilities, and scores using vectorized operations
        if num_tokens > 0:
            # Convert to numpy arrays in one batch operation
            token_ids_array = np.array(
                [t[0] for t in token_set[: self.max_tokens_per_step]],
                dtype=self.token_id_dtype,
            )
            token_probs_array = np.array(
                [t[1] for t in token_set[: self.max_tokens_per_step]],
                dtype=self.prob_dtype,
            )
            token_scores_array = np.array(
                [s[1] for s in token_scores[: self.max_tokens_per_step]],
                dtype=self.score_dtype,
            )

            # Store in pre-allocated tensors with slice assignment (vectorized)
            self.token_ids[self.current_step, :num_tokens] = token_ids_array
            self.token_probs[self.current_step, :num_tokens] = token_probs_array
            self.token_scores[self.current_step, :num_tokens] = token_scores_array

        # Increment step counter after storing
        self.current_step += 1

    def reapply_threshold_to_all_sets(self) -> List[List[Tuple[int, float]]]:
        """
        Reapply the current threshold to all previously processed token sets.

        Returns:
            List[List[Tuple[int, float]]]: Updated list of pruned token sets
        """
        if self.current_step == 0:
            return []

        current_threshold = self.get_current_threshold()

        # Process all steps up to current_step
        pruned_sets = []
        for step in range(self.current_step):
            # Get valid tokens for this step
            step_valid_mask = self.valid_mask[step]
            num_tokens = self.tokens_per_step[step]

            # Get token IDs and scores for this step
            step_token_ids = self.token_ids[step, :num_tokens]
            step_token_scores = self.token_scores[step, :num_tokens]
            step_token_probs = self.token_probs[step, :num_tokens]

            # Create mask for tokens above threshold
            above_threshold = step_token_scores >= current_threshold

            # If no tokens would be kept, keep the highest probability token
            if not np.any(above_threshold):
                max_prob_idx = np.argmax(step_token_probs)
                above_threshold[max_prob_idx] = True

            # Create pruned set for this step
            step_pruned = []
            for i in range(num_tokens):
                if above_threshold[i]:
                    step_pruned.append(
                        (int(step_token_ids[i]), float(step_token_probs[i]))
                    )

            pruned_sets.append(step_pruned)

        return pruned_sets

    def reset(self):
        """Reset the dynamic threshold for a new generation."""
        self.current_step = 0
        # Reset all tensors
        self.valid_mask.fill(False)
        self.pruned_mask.fill(False)
        self.tokens_per_step.fill(0)
        # Reset caching mechanism
        self.cached_threshold = None

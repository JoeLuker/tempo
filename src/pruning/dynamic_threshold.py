import numpy as np
from typing import List, Optional, Tuple, Dict, Any


class DynamicThresholdManager:
    """
    Manages dynamic thresholds that change over the course of generation.

    This class provides two methods for controlling threshold progression during generation:

    1. Bezier Curve (default): Creates a smooth, gradual transition from base_threshold to
       final_threshold using a cubic Bezier curve. This provides a continuous, organic shift
       between exploration and exploitation phases.

    2. ReLU Transition: Creates a distinct phase transition with a flat period (maintaining
       base_threshold) until reaching the activation point, followed by a linear increase to
       final_threshold. This creates a clear separation between exploration and exploitation.

    The ReLU transition is particularly useful when you want to:
    - Have a well-defined exploration phase before committing to specific paths
    - Create a distinct "thinking" phase followed by a "concluding" phase
    - Control precisely when the model shifts from divergent to convergent generation
    """

    def __init__(
        self,
        base_threshold: float = 0.3,
        max_steps: Optional[int] = None,
        bezier_points: Optional[List[float]] = None,
        final_threshold: float = 1.0,
        max_tokens_per_step: int = 20,
        use_relu: bool = False,
        relu_activation_point: float = 0.5,
    ):
        """
        Initialize the dynamic threshold manager.

        Args:
            base_threshold: Starting threshold value
            max_steps: Maximum number of steps (for calculating dynamic threshold)
            bezier_points: Control points for Bezier curve [p1, p2] between 0-1
            final_threshold: Final threshold value for dynamic threshold
            max_tokens_per_step: Maximum number of tokens expected per step
            use_relu: Whether to use ReLU-based transition instead of Bezier curve
            relu_activation_point: Point at which ReLU transition begins (0-1)
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

        # Invariant: ReLU activation point must be in valid range
        if not (0 <= relu_activation_point <= 1):
            raise ValueError(
                f"Invariant violation: relu_activation_point must be between 0 and 1, got {relu_activation_point}"
            )

        self.base_threshold = base_threshold
        self.max_steps = max_steps or 100  # Default if not specified
        self.max_tokens_per_step = max_tokens_per_step

        # This counter tracks how many token sets have been stored
        # It will be synchronized with the external step counter in store_token_set
        self.current_step = 0

        self.final_threshold = final_threshold
        self.use_relu = use_relu
        self.relu_activation_point = relu_activation_point

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
        self.threshold_epsilon = (
            0.01  # Only recompute when threshold changes by more than this
        )

        # Print initialization information to console
        print(f"\n*** INITIALIZING DYNAMIC THRESHOLD MANAGER ***")
        print(f"Base threshold: {base_threshold}")
        print(f"Final threshold: {final_threshold}")
        print(f"Max steps: {self.max_steps}")
        if use_relu:
            print(
                f"Using ReLU transition with activation point: {relu_activation_point}"
            )
        else:
            print(f"Using Bezier curve with control points: {self.bezier_points}")
        print("*" * 50)

    def get_current_threshold(self, step: Optional[int] = None) -> float:
        """
        Get the current threshold value based on step progress.

        Args:
            step: The current generation step (optional, uses internal step if None)

        Returns:
            float: Current threshold value
        """
        if self.max_steps <= 1:
            return self.base_threshold

        # Use provided step if given, otherwise fall back to internal counter
        current_step = step if step is not None else self.current_step

        # Calculate progress as a value between 0 and 1
        progress = min(1.0, current_step / self.max_steps)

        if self.use_relu:
            # Calculate threshold using ReLU transition
            # Stay flat until activation point, then linear increase
            relu_value = (
                max(0, progress - self.relu_activation_point)
                / (1.0 - self.relu_activation_point)
                if self.relu_activation_point < 1.0
                else 0.0
            )
            transition_value = relu_value
        else:
            # Calculate threshold using cubic Bezier curve for smooth progression
            transition_value = self._cubic_bezier(
                progress,
                0.0,  # Start at 0
                self.bezier_points[0],
                self.bezier_points[1],
                1.0,  # End at 1
            )

        # Scale between base_threshold and final_threshold
        current_threshold = (
            self.base_threshold
            + (self.final_threshold - self.base_threshold) * transition_value
        )

        # Output threshold information to console every 10 steps
        if (
            current_step % 10 == 0 or current_step > 80
        ):  # More debug info as we approach later steps
            print(f"\n*** DYNAMIC THRESHOLD AT STEP {current_step} ***")
            print(f"Progress: {progress:.4f} ({current_step}/{self.max_steps})")
            print(f"Current threshold value: {current_threshold:.4f}")
            if current_step > 80:  # Extra debug info near the end
                print(
                    f"DETAILED STATUS: Transition value = {transition_value:.4f}, Max steps setting = {self.max_steps}"
                )
                print(
                    f"STOPPING CONDITIONS: Will stop if no tokens have probability >= {current_threshold:.4f}"
                )
                # Print whether we've reached the final threshold
                percentage_to_final = (
                    (current_threshold - self.base_threshold)
                    / (self.final_threshold - self.base_threshold)
                    * 100
                )
                print(
                    f"Progress to final threshold: {percentage_to_final:.1f}% ({current_threshold:.4f}/{self.final_threshold:.4f})"
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
        self,
        token_set: List[Tuple[int, float]],
        token_scores: List[Tuple[int, float]],
        step: Optional[int] = None,
    ):
        """
        Store a token set and its scores for reapplication of thresholds.

        Args:
            token_set: List of (token_id, probability) tuples
            token_scores: List of (token_id, normalized_score) tuples
            step: The current generation step (optional, uses internal step if None)
        """
        # Use provided step if given, otherwise use internal counter for storage
        storage_step = step if step is not None else self.current_step

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
        if storage_step >= self.max_steps:
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
        self.tokens_per_step[storage_step] = num_tokens

        # Update the valid mask for this step
        self.valid_mask[storage_step, :num_tokens] = True

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
            self.token_ids[storage_step, :num_tokens] = token_ids_array
            self.token_probs[storage_step, :num_tokens] = token_probs_array
            self.token_scores[storage_step, :num_tokens] = token_scores_array

        # Update our internal counter to match the latest step we've processed
        # This ensures we're tracking the same step as the external counter
        if step is not None and step >= self.current_step:
            self.current_step = step + 1
        else:
            # Only increment if we're using our internal counter
            self.current_step += 1

        # Get the step to use for debugging output
        debug_step = step if step is not None else self.current_step - 1

        # Extra debug output for token sets near the end of generation
        if debug_step > 80:
            print(f"\n*** TOKEN SET AT STEP {debug_step} ***")
            print(f"Number of tokens stored: {num_tokens}")
            if num_tokens > 0:
                # Print top 3 tokens with probabilities
                top_n = min(3, num_tokens)
                print(f"Top {top_n} tokens with probabilities:")
                for i in range(top_n):
                    token_id = token_ids_array[i]
                    token_prob = token_probs_array[i]
                    token_score = token_scores_array[i]
                    print(
                        f"  Token ID: {token_id}, Probability: {token_prob:.6f}, Score: {token_score:.6f}"
                    )

                # Print highest probability token
                max_prob_idx = np.argmax(token_probs_array)
                print(
                    f"Highest probability token: ID={token_ids_array[max_prob_idx]}, Prob={token_probs_array[max_prob_idx]:.6f}"
                )

                # Check if any tokens would pass the next threshold level
                current_threshold = self.get_current_threshold(debug_step)
                tokens_above_threshold = np.sum(token_scores_array >= current_threshold)
                print(
                    f"Tokens that would pass current threshold ({current_threshold:.4f}): {tokens_above_threshold}"
                )
                if tokens_above_threshold == 0:
                    print(
                        "*** WARNING: No tokens would pass the current threshold! Generation might stop at next step. ***"
                    )

    def reapply_threshold_to_all_sets(
        self, step: Optional[int] = None
    ) -> List[List[Tuple[int, float]]]:
        """
        Reapply the current threshold to all previously processed token sets.

        Args:
            step: The current generation step (optional, uses internal step if None)

        Returns:
            List[List[Tuple[int, float]]]: Updated list of pruned token sets
        """
        if self.current_step == 0:
            return []

        # Use provided step if given, otherwise fall back to internal counter
        current_step = step if step is not None else self.current_step

        current_threshold = self.get_current_threshold(current_step)

        # Add debug output about the current threshold
        print(
            f"\nReapplying dynamic threshold: {current_threshold:.4f} at step {current_step}"
        )
        print(
            f"  Base threshold: {self.base_threshold:.4f}, Final threshold: {self.final_threshold:.4f}"
        )
        print(
            f"  Progress: {min(1.0, current_step / self.max_steps):.4f} ({current_step}/{self.max_steps} steps)"
        )

        # Determine the number of valid steps to process (only process steps we've actually stored data for)
        max_valid_step = min(current_step, self.current_step)
        print(
            f"  Processing {max_valid_step} token sets (out of {self.current_step} stored)"
        )

        if self.use_relu:
            print(
                f"  Using ReLU transition with activation point: {self.relu_activation_point:.4f}"
            )
        else:
            print(f"  Using Bezier curve with control points: {self.bezier_points}")

        # For the final step before generation ends, print detailed information
        if current_step > 80:  # We know it stops around step 86
            print(f"FINAL STEP DEBUGGING - Step {current_step}:")
            print(f"  Current threshold: {current_threshold:.4f}")
            print(f"  Progress: {min(1.0, current_step / self.max_steps):.4f}")
            print(f"  Max steps setting: {self.max_steps}")

        # Process all steps up to max_valid_step
        pruned_sets = []
        total_tokens_before = 0
        total_tokens_after = 0

        for step_idx in range(max_valid_step):
            # Get valid tokens for this step
            step_valid_mask = self.valid_mask[step_idx]
            num_tokens = self.tokens_per_step[step_idx]

            # Get token IDs and scores for this step
            step_token_ids = self.token_ids[step_idx, :num_tokens]
            step_token_scores = self.token_scores[step_idx, :num_tokens]
            step_token_probs = self.token_probs[step_idx, :num_tokens]

            # Create mask for tokens above threshold
            above_threshold = step_token_scores >= current_threshold

            # Count tokens before pruning
            tokens_before = num_tokens
            total_tokens_before += tokens_before

            # If no tokens would be kept, keep the highest probability token
            if not np.any(above_threshold):
                max_prob_idx = np.argmax(step_token_probs)
                above_threshold[max_prob_idx] = True
                # Extra debug near the end
                if current_step > 80:
                    print(
                        f"  Step {step_idx}: NO TOKENS ABOVE THRESHOLD! Keeping only max prob token (ID={step_token_ids[max_prob_idx]}, prob={step_token_probs[max_prob_idx]:.6f})"
                    )

            # Create pruned set for this step
            step_pruned = []
            for i in range(num_tokens):
                if above_threshold[i]:
                    step_pruned.append(
                        (int(step_token_ids[i]), float(step_token_probs[i]))
                    )

            # Count tokens after pruning
            tokens_after = len(step_pruned)
            total_tokens_after += tokens_after

            # Add detailed debug info for each step
            print(
                f"  Step {step_idx}: {tokens_before} tokens -> {tokens_after} tokens (threshold: {current_threshold:.4f})"
            )

            pruned_sets.append(step_pruned)

        # Summary statistics
        print(
            f"Total pruning: {total_tokens_before} tokens -> {total_tokens_after} tokens"
        )
        print(
            f"Average tokens per position after pruning: {total_tokens_after / max(1, max_valid_step):.2f}"
        )

        return pruned_sets

    def reset(self):
        """
        Reset the dynamic threshold manager for a new generation.
        This resets the internal step counter and all stored token data.
        """
        # Reset the internal step counter to 0
        self.current_step = 0

        # Reset all tensors
        self.valid_mask.fill(False)
        self.pruned_mask.fill(False)
        self.tokens_per_step.fill(0)

        # Reset caching mechanism
        self.cached_threshold = None

        # Print reset notification
        print("*** Dynamic threshold manager reset ***")

import numpy as np
from typing import List, Optional, Tuple, Dict, Any


class DynamicThresholdManager:
    """
    Manages dynamic thresholds that change over the course of generation.

    This class provides three methods for controlling threshold progression during generation:

    1. Flat Threshold (default): Uses a constant threshold throughout generation

    2. Bezier Curve: Creates a smooth, gradual transition from base_threshold to
       final_threshold using a cubic Bezier curve. This provides a continuous, organic shift
       between exploration and exploitation phases.

    3. ReLU Transition: Creates a distinct phase transition with a flat period (maintaining
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
        use_relu: bool = False,
        relu_activation_point: float = 0.5,
        use_bezier: bool = False,
    ):
        """
        Initialize the dynamic threshold manager.

        Args:
            base_threshold: Starting threshold value
            max_steps: Maximum number of steps (for calculating dynamic threshold)
            bezier_points: Control points for Bezier curve [p1, p2] between 0-1
            final_threshold: Final threshold value for dynamic threshold
            use_relu: Whether to use ReLU-based transition instead of Bezier curve
            relu_activation_point: Point at which ReLU transition begins (0-1)
            use_bezier: Whether to use Bezier curve
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
        self.final_threshold = final_threshold
        self.use_relu = use_relu
        self.relu_activation_point = relu_activation_point
        self.use_bezier = use_bezier

        # Default Bezier control points for exponential-like curve
        self.bezier_points = bezier_points if bezier_points is not None else [0.2, 0.8]

        # Cache for optimized recomputation
        self.cached_threshold = None
        self.threshold_epsilon = 0.01  # Only recompute when threshold changes by more than this

        # Print initialization information to console
        print(f"\n*** INITIALIZING DYNAMIC THRESHOLD MANAGER ***")
        print(f"Base threshold: {base_threshold}")
        print(f"Final threshold: {final_threshold}")
        print(f"Max steps: {self.max_steps}")
        if use_relu:
            print(f"Using ReLU transition with activation point: {relu_activation_point}")
        elif use_bezier:
            print(f"Using Bezier curve with control points: {self.bezier_points}")
        else:
            print("Using flat threshold (default)")
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

        # Calculate progress as a value between 0 and 1
        progress = min(1.0, step / self.max_steps)

        if self.use_relu:
            # Calculate threshold using ReLU transition
            current_threshold = self._get_relu_threshold(progress)
        elif self.use_bezier:
            # Calculate threshold using cubic Bezier curve for smooth progression
            current_threshold = self._get_bezier_threshold(progress)
        else:
            # Use flat threshold (default)
            current_threshold = self.base_threshold

        # Output threshold information to console every 10 steps
        if step % 10 == 0 or step > 80:
            print(f"\n*** DYNAMIC THRESHOLD AT STEP {step} ***")
            print(f"Progress: {progress:.4f} ({step}/{self.max_steps})")
            print(f"Current threshold value: {current_threshold:.4f}")
            if step > 80:
                print(f"DETAILED STATUS: Progress = {progress:.4f}, Max steps = {self.max_steps}")
                print(f"STOPPING CONDITIONS: Will stop if no tokens have probability >= {current_threshold:.4f}")
                if self.use_relu or self.use_bezier:
                    percentage_to_final = ((current_threshold - self.base_threshold) / (self.final_threshold - self.base_threshold) * 100)
                    print(f"Progress to final threshold: {percentage_to_final:.1f}% ({current_threshold:.4f}/{self.final_threshold:.4f})")

        return current_threshold

    def _get_relu_threshold(self, progress: float) -> float:
        """
        Calculate threshold using ReLU transition.

        Args:
            progress: Current generation progress (0 to 1)

        Returns:
            float: Current threshold value
        """
        # Stay flat until activation point, then linear increase
        relu_value = (
            max(0, progress - self.relu_activation_point)
            / (1.0 - self.relu_activation_point)
            if self.relu_activation_point < 1.0
            else 0.0
        )
        return self.base_threshold + (self.final_threshold - self.base_threshold) * relu_value

    def _get_bezier_threshold(self, progress: float) -> float:
        """
        Calculate threshold using cubic Bezier curve.

        Args:
            progress: Current generation progress (0 to 1)

        Returns:
            float: Current threshold value
        """
        transition_value = self._cubic_bezier(
            progress,
            0.0,  # Start at 0
            self.bezier_points[0],
            self.bezier_points[1],
            1.0,  # End at 1
        )
        return self.base_threshold + (self.final_threshold - self.base_threshold) * transition_value

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

    def reset(self):
        """
        Reset the dynamic threshold manager for a new generation.
        This resets the internal step counter and all stored token data.
        """
        # Reset caching mechanism
        self.cached_threshold = None

        # Print reset notification
        print("*** Dynamic threshold manager reset ***")

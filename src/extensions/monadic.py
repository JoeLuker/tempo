"""Useful monadic patterns for TEMPO - Result and Writer only.

These add real value without unnecessary complexity.
"""

from dataclasses import dataclass, replace
from typing import Callable, TypeVar, Generic, Union
from .ultra_simple import GenState

A = TypeVar('A')
B = TypeVar('B')


# ============================================================================
# Result Monad - Graceful Error Handling
# ============================================================================

@dataclass
class Success(Generic[A]):
    """Successful computation."""
    value: A

    def bind(self, f: Callable[[A], 'Result[B]']) -> 'Result[B]':
        return f(self.value)

    def get_or_else(self, default: A) -> A:
        return self.value


@dataclass
class Failure(Generic[A]):
    """Failed computation."""
    error: str

    def bind(self, f: Callable[[A], 'Result[B]']) -> 'Result[B]':
        return Failure(self.error)

    def get_or_else(self, default: A) -> A:
        return default


Result = Union[Success[A], Failure[A]]

# Shorter names
Ok = Success
Err = Failure


def safe_modify_threshold(
    multiplier: float,
    min_val: float = 0.05,
    max_val: float = 0.95
) -> Callable[[GenState], Result[GenState]]:
    """Safely modify threshold with bounds checking.

    Example:
        ext = safe_modify_threshold(2.0, max_val=0.30)
        result = ext(state)
        new_state = result.get_or_else(state)  # Fallback to original on error
    """
    def extension(state: GenState) -> Result[GenState]:
        new_threshold = state.threshold * multiplier

        if new_threshold < min_val:
            return Err(f"Threshold {new_threshold:.3f} below min {min_val}")
        if new_threshold > max_val:
            return Err(f"Threshold {new_threshold:.3f} above max {max_val}")

        return Ok(replace(state, threshold=new_threshold))

    return extension


# ============================================================================
# Writer Monad - Clean Logging
# ============================================================================

@dataclass
class Writer(Generic[A]):
    """Computation with accumulated log."""
    value: A
    log: list[str]

    def bind(self, f: Callable[[A], 'Writer[B]']) -> 'Writer[B]':
        result = f(self.value)
        return Writer(result.value, self.log + result.log)

    @staticmethod
    def of(value: A) -> 'Writer[A]':
        return Writer(value, [])

    def run(self) -> tuple[A, list[str]]:
        return (self.value, self.log)


def logged_threshold(threshold: float, reason: str) -> Callable[[GenState], Writer[GenState]]:
    """Set threshold with logging.

    Example:
        ext = logged_threshold(0.12, "Starting exploration")
        result = ext(state)
        new_state, logs = result.run()
        print(logs)  # ["Step 10: Starting exploration (threshold 0.15 → 0.12)"]
    """
    def extension(state: GenState) -> Writer[GenState]:
        new_state = replace(state, threshold=threshold)
        msg = f"Step {state.step}: {reason} (threshold {state.threshold:.3f} → {threshold:.3f})"
        return Writer(new_state, [msg] if state.threshold != threshold else [])
    return extension


def compose_writers(*extensions: Callable[[GenState], Writer[GenState]]) -> Callable[[GenState], Writer[GenState]]:
    """Compose Writer extensions with automatic log accumulation."""
    def composed(state: GenState) -> Writer[GenState]:
        result = Writer.of(state)
        for ext in extensions:
            result = result.bind(ext)
        return result
    return composed


if __name__ == '__main__':
    print("=" * 60)
    print("USEFUL MONADIC PATTERNS")
    print("=" * 60)
    print()

    # Result monad
    print("1. RESULT MONAD - Safe operations")
    print("-" * 60)

    ext = safe_modify_threshold(2.5, max_val=0.30)
    state = GenState(
        step=10, entropy=2.0, threshold=0.15,
        selected_tokens=(), branching_factor=1, prompt_length=10
    )

    result = ext(state)
    if isinstance(result, Success):
        print(f"✓ Success: {state.threshold} → {result.value.threshold}")
    else:
        print(f"✗ Failed: {result.error}")
        print(f"  Using fallback: {result.get_or_else(state).threshold}")

    print()

    # Writer monad
    print("2. WRITER MONAD - Clean logging")
    print("-" * 60)

    pipeline = compose_writers(
        logged_threshold(0.12, "Phase 1: Exploration"),
        logged_threshold(0.20, "Phase 2: Refinement"),
        logged_threshold(1.0, "Phase 3: Commitment")
    )

    state = GenState(
        step=50, entropy=2.0, threshold=0.15,
        selected_tokens=(), branching_factor=1, prompt_length=10
    )

    result = pipeline(state)
    new_state, logs = result.run()

    print(f"Final threshold: {new_state.threshold}")
    print(f"Logs collected:")
    for log in logs:
        print(f"  {log}")

    print()
    print("=" * 60)
    print("WHEN TO USE:")
    print("=" * 60)
    print("""
✅ Result: Need graceful error handling without exceptions
✅ Writer: Need clean logging without side effects

Keep it simple - only use when you truly need these patterns!
""")

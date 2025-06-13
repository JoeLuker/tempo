"""Token value objects for the TEMPO generation system.

This module defines immutable value objects representing tokens and token sets.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass(frozen=True)
class Token:
    """Immutable value object representing a single token."""
    id: int
    text: str
    logit: float
    probability: float
    position: int
    
    def __post_init__(self):
        """Validate token properties."""
        if self.id < 0:
            raise ValueError(f"Token ID must be non-negative, got {self.id}")
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(f"Probability must be between 0 and 1, got {self.probability}")
        if self.position < 0:
            raise ValueError(f"Position must be non-negative, got {self.position}")


@dataclass(frozen=True)
class TokenSet:
    """Immutable value object representing a set of tokens at a single position."""
    tokens: List[Token]
    position: int
    is_parallel: bool = False
    
    def __post_init__(self):
        """Validate token set properties."""
        if not self.tokens:
            raise ValueError("TokenSet must contain at least one token")
        if self.position < 0:
            raise ValueError(f"Position must be non-negative, got {self.position}")
        # Ensure all tokens have the same position
        positions = {token.position for token in self.tokens}
        if len(positions) > 1:
            raise ValueError(f"All tokens in a TokenSet must have the same position, got {positions}")
        if positions and list(positions)[0] != self.position:
            raise ValueError(f"TokenSet position {self.position} doesn't match token positions {positions}")

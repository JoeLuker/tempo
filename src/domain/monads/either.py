"""Either monad for representing values with two possible types."""

from __future__ import annotations
from typing import TypeVar, Generic, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

L = TypeVar('L')  # Left type
R = TypeVar('R')  # Right type
T = TypeVar('T')
U = TypeVar('U')


class Either(ABC, Generic[L, R]):
    """
    Either monad representing a value that can be one of two types.
    By convention, Left represents the error case and Right the success case.
    """
    
    @abstractmethod
    def is_left(self) -> bool:
        """Check if this is a Left value."""
        pass
    
    @abstractmethod
    def is_right(self) -> bool:
        """Check if this is a Right value."""
        pass
    
    @abstractmethod
    def map(self, f: Callable[[R], T]) -> Either[L, T]:
        """Map over the Right value."""
        pass
    
    @abstractmethod
    def map_left(self, f: Callable[[L], T]) -> Either[T, R]:
        """Map over the Left value."""
        pass
    
    @abstractmethod
    def flat_map(self, f: Callable[[R], Either[L, T]]) -> Either[L, T]:
        """Monadic bind for Either."""
        pass
    
    @abstractmethod
    def fold(self, left_f: Callable[[L], T], right_f: Callable[[R], T]) -> T:
        """Fold the Either into a single value."""
        pass
    
    @abstractmethod
    def swap(self) -> Either[R, L]:
        """Swap Left and Right."""
        pass
    
    def get_or_else(self, default: R) -> R:
        """Get Right value or default."""
        return self.fold(lambda _: default, lambda x: x)
    
    def to_result(self):
        """Convert to Result monad."""
        from .result import Ok, Err
        return self.fold(lambda l: Err(l), lambda r: Ok(r))


@dataclass(frozen=True)
class Left(Either[L, R]):
    """Left variant of Either."""
    value: L
    
    def is_left(self) -> bool:
        return True
    
    def is_right(self) -> bool:
        return False
    
    def map(self, f: Callable[[R], T]) -> Either[L, T]:
        return Left(self.value)
    
    def map_left(self, f: Callable[[L], T]) -> Either[T, R]:
        return Left(f(self.value))
    
    def flat_map(self, f: Callable[[R], Either[L, T]]) -> Either[L, T]:
        return Left(self.value)
    
    def fold(self, left_f: Callable[[L], T], right_f: Callable[[R], T]) -> T:
        return left_f(self.value)
    
    def swap(self) -> Either[R, L]:
        return Right(self.value)


@dataclass(frozen=True)
class Right(Either[L, R]):
    """Right variant of Either."""
    value: R
    
    def is_left(self) -> bool:
        return False
    
    def is_right(self) -> bool:
        return True
    
    def map(self, f: Callable[[R], T]) -> Either[L, T]:
        return Right(f(self.value))
    
    def map_left(self, f: Callable[[L], T]) -> Either[T, R]:
        return Right(self.value)
    
    def flat_map(self, f: Callable[[R], Either[L, T]]) -> Either[L, T]:
        return f(self.value)
    
    def fold(self, left_f: Callable[[L], T], right_f: Callable[[R], T]) -> T:
        return right_f(self.value)
    
    def swap(self) -> Either[R, L]:
        return Left(self.value)


# Type alias
EitherT = Union[Left[L, R], Right[L, R]]


# Helper functions
def left(value: L) -> Left[L, Any]:
    """Create a Left value."""
    return Left(value)


def right(value: R) -> Right[Any, R]:
    """Create a Right value."""
    return Right(value)
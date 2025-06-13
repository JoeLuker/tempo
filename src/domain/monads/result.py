"""Result monad for error handling without exceptions."""

from __future__ import annotations
from typing import TypeVar, Generic, Callable, Union, Optional, Any, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
E = TypeVar('E')
U = TypeVar('U')


class Result(ABC, Generic[T, E]):
    """
    Result monad representing either success (Ok) or failure (Err).
    
    This provides a functional way to handle errors without exceptions,
    enabling railway-oriented programming.
    """
    
    @abstractmethod
    def is_ok(self) -> bool:
        """Check if this is an Ok result."""
        pass
    
    @abstractmethod
    def is_err(self) -> bool:
        """Check if this is an Err result."""
        pass
    
    @abstractmethod
    def unwrap(self) -> T:
        """Get the value, raising if this is an error."""
        pass
    
    @abstractmethod
    def unwrap_err(self) -> E:
        """Get the error, raising if this is Ok."""
        pass
    
    @abstractmethod
    def unwrap_or(self, default: T) -> T:
        """Get the value or return a default."""
        pass
    
    @abstractmethod
    def unwrap_or_else(self, f: Callable[[E], T]) -> T:
        """Get the value or compute it from the error."""
        pass
    
    @abstractmethod
    def map(self, f: Callable[[T], U]) -> Result[U, E]:
        """Transform the value if Ok."""
        pass
    
    @abstractmethod
    def map_err(self, f: Callable[[E], U]) -> Result[T, U]:
        """Transform the error if Err."""
        pass
    
    @abstractmethod
    def flat_map(self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Chain operations that return Results (monadic bind)."""
        pass
    
    @abstractmethod
    def and_then(self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Alias for flat_map for readability."""
        pass
    
    @abstractmethod
    def or_else(self, f: Callable[[E], Result[T, U]]) -> Result[T, U]:
        """Chain error recovery operations."""
        pass
    
    def __iter__(self):
        """Allow use in for comprehensions."""
        if self.is_ok():
            yield self.unwrap()
    
    def to_optional(self) -> Optional[T]:
        """Convert to Optional, losing error information."""
        return self.unwrap() if self.is_ok() else None


@dataclass(frozen=True)
class Ok(Result[T, E]):
    """Success variant of Result."""
    value: T
    
    def is_ok(self) -> bool:
        return True
    
    def is_err(self) -> bool:
        return False
    
    def unwrap(self) -> T:
        return self.value
    
    def unwrap_err(self) -> E:
        raise ValueError("Called unwrap_err on Ok value")
    
    def unwrap_or(self, default: T) -> T:
        return self.value
    
    def unwrap_or_else(self, f: Callable[[E], T]) -> T:
        return self.value
    
    def map(self, f: Callable[[T], U]) -> Result[U, E]:
        try:
            return Ok(f(self.value))
        except Exception as e:
            logger.debug(f"Exception in map: {e}")
            return Err(e)  # type: ignore
    
    def map_err(self, f: Callable[[E], U]) -> Result[T, U]:
        return Ok(self.value)
    
    def flat_map(self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        try:
            return f(self.value)
        except Exception as e:
            logger.debug(f"Exception in flat_map: {e}")
            return Err(e)  # type: ignore
    
    def and_then(self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        return self.flat_map(f)
    
    def or_else(self, f: Callable[[E], Result[T, U]]) -> Result[T, U]:
        return Ok(self.value)


@dataclass(frozen=True)
class Err(Result[T, E]):
    """Error variant of Result."""
    error: E
    
    def is_ok(self) -> bool:
        return False
    
    def is_err(self) -> bool:
        return True
    
    def unwrap(self) -> T:
        raise ValueError(f"Called unwrap on Err value: {self.error}")
    
    def unwrap_err(self) -> E:
        return self.error
    
    def unwrap_or(self, default: T) -> T:
        return default
    
    def unwrap_or_else(self, f: Callable[[E], T]) -> T:
        return f(self.error)
    
    def map(self, f: Callable[[T], U]) -> Result[U, E]:
        return Err(self.error)
    
    def map_err(self, f: Callable[[E], U]) -> Result[T, U]:
        try:
            return Err(f(self.error))
        except Exception as e:
            logger.debug(f"Exception in map_err: {e}")
            return Err(e)  # type: ignore
    
    def flat_map(self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        return Err(self.error)
    
    def and_then(self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        return Err(self.error)
    
    def or_else(self, f: Callable[[E], Result[T, U]]) -> Result[T, U]:
        try:
            return f(self.error)
        except Exception as e:
            logger.debug(f"Exception in or_else: {e}")
            return Err(e)  # type: ignore


# Result type alias for better type hints
ResultT = Union[Ok[T, E], Err[T, E]]


# Helper functions for Result creation
def ok(value: T) -> Ok[T, Any]:
    """Create an Ok result."""
    return Ok(value)


def err(error: E) -> Err[Any, E]:
    """Create an Err result."""
    return Err(error)


def from_optional(opt: Optional[T], error: E) -> Result[T, E]:
    """Convert Optional to Result."""
    return Ok(opt) if opt is not None else Err(error)


def collect_results(results: list[Result[T, E]]) -> Result[list[T], E]:
    """
    Collect a list of Results into a Result of list.
    Returns Ok with all values if all are Ok, otherwise returns first Err.
    """
    values = []
    for result in results:
        if result.is_err():
            return Err(result.unwrap_err())
        values.append(result.unwrap())
    return Ok(values)


def try_result(f: Callable[[], T], 
               error_type: Type[E] = Exception) -> Result[T, E]:
    """
    Execute a function and wrap the result in Result.
    Catches exceptions of specified type.
    """
    try:
        return Ok(f())
    except error_type as e:
        return Err(e)


class ResultMonad:
    """Monadic operations for Result."""
    
    @staticmethod
    def pure(value: T) -> Result[T, Any]:
        """Lift a value into Result context."""
        return Ok(value)
    
    @staticmethod
    def bind(result: Result[T, E], 
             f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Monadic bind operation."""
        return result.flat_map(f)
    
    @staticmethod
    def sequence(results: list[Result[T, E]]) -> Result[list[T], E]:
        """Convert list of Results to Result of list."""
        return collect_results(results)
    
    @staticmethod
    def traverse(f: Callable[[T], Result[U, E]], 
                 items: list[T]) -> Result[list[U], E]:
        """Map a Result-returning function over a list and collect results."""
        return collect_results([f(item) for item in items])
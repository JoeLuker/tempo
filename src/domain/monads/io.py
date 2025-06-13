"""IO monad for handling side effects functionally."""

from __future__ import annotations
from typing import TypeVar, Generic, Callable, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
U = TypeVar('U')
E = TypeVar('E')


@dataclass(frozen=True)
class IO(Generic[T]):
    """
    IO monad for wrapping side-effectful computations.
    The computation is not executed until explicitly run.
    """
    computation: Callable[[], T]
    
    def run(self) -> T:
        """Execute the IO action and return the result."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Running IO computation")
        return self.computation()
    
    def map(self, f: Callable[[T], U]) -> IO[U]:
        """Transform the result of the IO action."""
        return IO(lambda: f(self.run()))
    
    def flat_map(self, f: Callable[[T], IO[U]]) -> IO[U]:
        """Chain IO actions (monadic bind)."""
        return IO(lambda: f(self.run()).run())
    
    def and_then(self, other: IO[U]) -> IO[U]:
        """Execute this IO, ignore result, then execute other."""
        return self.flat_map(lambda _: other)
    
    def or_else(self, f: Callable[[Exception], IO[T]]) -> IO[T]:
        """Handle exceptions in IO computation."""
        def safe_computation():
            try:
                return self.run()
            except Exception as e:
                logger.debug(f"Exception in IO: {e}")
                return f(e).run()
        return IO(safe_computation)
    
    @staticmethod
    def pure(value: T) -> IO[T]:
        """Lift a pure value into IO context."""
        return IO(lambda: value)
    
    @staticmethod
    def lift(f: Callable[[], T]) -> IO[T]:
        """Lift a side-effectful function into IO."""
        return IO(f)
    
    @staticmethod
    def sequence(ios: list[IO[T]]) -> IO[list[T]]:
        """Convert list of IO actions to IO of list."""
        def run_all():
            return [io.run() for io in ios]
        return IO(run_all)


@dataclass(frozen=True) 
class IOResult(Generic[T, E]):
    """
    Combination of IO and Result monads for safe side effects.
    Represents an IO action that might fail.
    """
    computation: Callable[[], 'Result[T, E]']
    
    def run(self) -> 'Result[T, E]':
        """Execute the IO action and return the Result."""
        from .result import Result, Ok, Err, try_result
        try:
            return self.computation()
        except Exception as e:
            logger.debug(f"Exception in IOResult: {e}")
            return Err(e)  # type: ignore
    
    def map(self, f: Callable[[T], U]) -> IOResult[U, E]:
        """Map over successful result."""
        return IOResult(lambda: self.run().map(f))
    
    def map_error(self, f: Callable[[E], U]) -> IOResult[T, U]:
        """Map over error result."""
        return IOResult(lambda: self.run().map_err(f))
    
    def flat_map(self, f: Callable[[T], IOResult[U, E]]) -> IOResult[U, E]:
        """Chain IOResult actions."""
        def computation():
            result = self.run()
            if result.is_ok():
                return f(result.unwrap()).run()
            return result  # type: ignore
        return IOResult(computation)
    
    def recover(self, f: Callable[[E], IOResult[T, E]]) -> IOResult[T, E]:
        """Recover from errors."""
        def computation():
            result = self.run()
            if result.is_err():
                return f(result.unwrap_err()).run()
            return result
        return IOResult(computation)
    
    @staticmethod
    def pure(value: T) -> IOResult[T, Any]:
        """Lift a pure value into IOResult."""
        from .result import Ok
        return IOResult(lambda: Ok(value))
    
    @staticmethod
    def from_io(io: IO[T]) -> IOResult[T, Exception]:
        """Convert IO to IOResult, catching exceptions."""
        from .result import try_result
        return IOResult(lambda: try_result(io.run))
    
    @staticmethod
    def from_result(result: 'Result[T, E]') -> IOResult[T, E]:
        """Lift a Result into IOResult."""
        return IOResult(lambda: result)


# Helper functions
def io_pure(value: T) -> IO[T]:
    """Create a pure IO value."""
    return IO.pure(value)


def io_lift(f: Callable[[], T]) -> IO[T]:
    """Lift a function into IO."""
    return IO.lift(f)


def io_result_pure(value: T) -> IOResult[T, Any]:
    """Create a pure IOResult value."""
    return IOResult.pure(value)
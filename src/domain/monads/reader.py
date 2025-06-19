"""Reader monad for dependency injection."""

from typing import TypeVar, Generic, Callable
from dataclasses import dataclass

R = TypeVar('R')  # Reader environment type
T = TypeVar('T')  # Result type
U = TypeVar('U')


@dataclass(frozen=True)
class Reader(Generic[R, T]):
    """
    Reader monad for dependency injection.
    Represents a computation that depends on some shared environment.
    """
    run: Callable[[R], T]
    
    def map(self, f: Callable[[T], U]) -> Reader[R, U]:
        """Transform the result of the Reader."""
        return Reader(lambda env: f(self.run(env)))
    
    def flat_map(self, f: Callable[[T], Reader[R, U]]) -> Reader[R, U]:
        """Chain Reader computations (monadic bind)."""
        return Reader(lambda env: f(self.run(env)).run(env))
    
    def local(self, f: Callable[[R], R]) -> Reader[R, T]:
        """Run Reader with a modified environment."""
        return Reader(lambda env: self.run(f(env)))
    
    @staticmethod
    def pure(value: T) -> Reader[Any, T]:
        """Lift a pure value into Reader context."""
        return Reader(lambda _: value)
    
    @staticmethod
    def ask() -> Reader[R, R]:
        """Get the current environment."""
        return Reader(lambda env: env)
    
    @staticmethod
    def asks(f: Callable[[R], T]) -> Reader[R, T]:
        """Get a value derived from the environment."""
        return Reader(f)
    
    def __call__(self, env: R) -> T:
        """Allow calling Reader as a function."""
        return self.run(env)


# Reader monad transformer for Result
@dataclass(frozen=True)
class ReaderT(Generic[R, T]):
    """Reader transformer for Result monad."""
    run: Callable[[R], 'Result[T, Any]']
    
    def map(self, f: Callable[[T], U]) -> ReaderT[R, U]:
        """Map over successful result."""
        from .result import Result
        return ReaderT(lambda env: self.run(env).map(f))
    
    def flat_map(self, f: Callable[[T], ReaderT[R, U]]) -> ReaderT[R, U]:
        """Chain ReaderT computations."""
        from .result import Result
        def computation(env: R) -> Result[U, Any]:
            result = self.run(env)
            if result.is_ok():
                return f(result.unwrap()).run(env)
            return result  # type: ignore
        return ReaderT(computation)
    
    def local(self, f: Callable[[R], R]) -> ReaderT[R, T]:
        """Run with modified environment."""
        return ReaderT(lambda env: self.run(f(env)))
    
    @staticmethod
    def pure(value: T) -> ReaderT[Any, T]:
        """Lift a pure value into ReaderT."""
        from .result import Ok
        return ReaderT(lambda _: Ok(value))
    
    @staticmethod
    def ask() -> ReaderT[R, R]:
        """Get the current environment."""
        from .result import Ok
        return ReaderT(lambda env: Ok(env))
    
    @staticmethod
    def from_result(result: 'Result[T, Any]') -> ReaderT[Any, T]:
        """Lift a Result into ReaderT."""
        return ReaderT(lambda _: result)
    
    @staticmethod
    def from_reader(reader: Reader[R, T]) -> ReaderT[R, T]:
        """Lift a Reader into ReaderT."""
        from .result import Ok
        return ReaderT(lambda env: Ok(reader.run(env)))


# Helper functions
def reader_pure(value: T) -> Reader[Any, T]:
    """Create a pure Reader value."""
    return Reader.pure(value)


def ask() -> Reader[R, R]:
    """Get the current environment."""
    return Reader.ask()


def asks(f: Callable[[R], T]) -> Reader[R, T]:
    """Get a value from the environment."""
    return Reader.asks(f)
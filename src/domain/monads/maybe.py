"""Maybe monad for handling optional values functionally."""

from typing import TypeVar, Generic, Callable, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

T = TypeVar('T')
U = TypeVar('U')


class Maybe(ABC, Generic[T]):
    """
    Maybe monad for handling optional values without null checks.
    Represents a value that might be present (Some) or absent (Nothing).
    """
    
    @abstractmethod
    def is_some(self) -> bool:
        """Check if this contains a value."""
        pass
    
    @abstractmethod
    def is_nothing(self) -> bool:
        """Check if this is empty."""
        pass
    
    @abstractmethod
    def map(self, f: Callable[[T], U]) -> Maybe[U]:
        """Transform the value if present."""
        pass
    
    @abstractmethod
    def flat_map(self, f: Callable[[T], Maybe[U]]) -> Maybe[U]:
        """Monadic bind for Maybe."""
        pass
    
    @abstractmethod
    def filter(self, predicate: Callable[[T], bool]) -> Maybe[T]:
        """Filter the value based on a predicate."""
        pass
    
    @abstractmethod
    def get_or_else(self, default: T) -> T:
        """Get the value or a default."""
        pass
    
    @abstractmethod
    def or_else(self, alternative: Maybe[T]) -> Maybe[T]:
        """Return this or an alternative Maybe."""
        pass
    
    @abstractmethod
    def to_optional(self) -> Optional[T]:
        """Convert to Python Optional."""
        pass
    
    def __iter__(self):
        """Allow use in for comprehensions."""
        if self.is_some():
            yield self.get_or_else(None)  # type: ignore


@dataclass(frozen=True)
class Some(Maybe[T]):
    """Some variant containing a value."""
    value: T
    
    def is_some(self) -> bool:
        return True
    
    def is_nothing(self) -> bool:
        return False
    
    def map(self, f: Callable[[T], U]) -> Maybe[U]:
        return Some(f(self.value))
    
    def flat_map(self, f: Callable[[T], Maybe[U]]) -> Maybe[U]:
        return f(self.value)
    
    def filter(self, predicate: Callable[[T], bool]) -> Maybe[T]:
        return self if predicate(self.value) else Nothing()
    
    def get_or_else(self, default: T) -> T:
        return self.value
    
    def or_else(self, alternative: Maybe[T]) -> Maybe[T]:
        return self
    
    def to_optional(self) -> Optional[T]:
        return self.value


class Nothing(Maybe[T]):
    """Nothing variant representing absence of value."""
    
    def is_some(self) -> bool:
        return False
    
    def is_nothing(self) -> bool:
        return True
    
    def map(self, f: Callable[[T], U]) -> Maybe[U]:
        return Nothing()
    
    def flat_map(self, f: Callable[[T], Maybe[U]]) -> Maybe[U]:
        return Nothing()
    
    def filter(self, predicate: Callable[[T], bool]) -> Maybe[T]:
        return Nothing()
    
    def get_or_else(self, default: T) -> T:
        return default
    
    def or_else(self, alternative: Maybe[T]) -> Maybe[T]:
        return alternative
    
    def to_optional(self) -> Optional[T]:
        return None
    
    def __eq__(self, other):
        return isinstance(other, Nothing)
    
    def __hash__(self):
        return hash(None)


# Type alias
MaybeT = Union[Some[T], Nothing[T]]


# Helper functions
def some(value: T) -> Some[T]:
    """Create a Some value."""
    return Some(value)


def nothing() -> Nothing[Any]:
    """Create a Nothing value."""
    return Nothing()


def from_optional(opt: Optional[T]) -> Maybe[T]:
    """Convert Python Optional to Maybe."""
    return Some(opt) if opt is not None else Nothing()


def lift_maybe(f: Callable[[T], U]) -> Callable[[Maybe[T]], Maybe[U]]:
    """Lift a function to work with Maybe values."""
    return lambda maybe: maybe.map(f)
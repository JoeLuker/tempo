"""State monad for stateful computations."""

from typing import TypeVar, Generic, Callable
from dataclasses import dataclass

S = TypeVar('S')  # State type
T = TypeVar('T')  # Result type
U = TypeVar('U')


@dataclass(frozen=True)
class State(Generic[S, T]):
    """
    State monad for stateful computations.
    Represents a computation that transforms state and produces a value.
    """
    run: Callable[[S], tuple[T, S]]
    
    def map(self, f: Callable[[T], U]) -> State[S, U]:
        """Transform the result value."""
        def computation(state: S) -> tuple[U, S]:
            value, new_state = self.run(state)
            return f(value), new_state
        return State(computation)
    
    def flat_map(self, f: Callable[[T], State[S, U]]) -> State[S, U]:
        """Chain stateful computations (monadic bind)."""
        def computation(state: S) -> tuple[U, S]:
            value, new_state = self.run(state)
            return f(value).run(new_state)
        return State(computation)
    
    def then(self, other: State[S, U]) -> State[S, U]:
        """Execute this, ignore result, then execute other."""
        return self.flat_map(lambda _: other)
    
    @staticmethod
    def pure(value: T) -> State[S, T]:
        """Lift a pure value into State context."""
        return State(lambda state: (value, state))
    
    @staticmethod
    def get() -> State[S, S]:
        """Get the current state."""
        return State(lambda state: (state, state))
    
    @staticmethod
    def put(new_state: S) -> State[S, None]:
        """Set the state."""
        return State(lambda _: (None, new_state))
    
    @staticmethod
    def modify(f: Callable[[S], S]) -> State[S, None]:
        """Modify the state."""
        return State(lambda state: (None, f(state)))
    
    def eval_state(self, initial_state: S) -> T:
        """Run the computation and return only the value."""
        value, _ = self.run(initial_state)
        return value
    
    def exec_state(self, initial_state: S) -> S:
        """Run the computation and return only the final state."""
        _, final_state = self.run(initial_state)
        return final_state
    
    def run_state(self, initial_state: S) -> tuple[T, S]:
        """Run the computation and return both value and state."""
        return self.run(initial_state)


# State monad transformer for Result
@dataclass(frozen=True)
class StateT(Generic[S, T]):
    """State transformer for Result monad."""
    run: Callable[[S], 'Result[tuple[T, S], Any]']
    
    def map(self, f: Callable[[T], U]) -> StateT[S, U]:
        """Map over successful result."""
        from .result import Result
        def computation(state: S) -> Result[tuple[U, S], Any]:
            result = self.run(state)
            return result.map(lambda pair: (f(pair[0]), pair[1]))
        return StateT(computation)
    
    def flat_map(self, f: Callable[[T], StateT[S, U]]) -> StateT[S, U]:
        """Chain StateT computations."""
        from .result import Result
        def computation(state: S) -> Result[tuple[U, S], Any]:
            result = self.run(state)
            if result.is_ok():
                value, new_state = result.unwrap()
                return f(value).run(new_state)
            return result  # type: ignore
        return StateT(computation)
    
    @staticmethod
    def pure(value: T) -> StateT[S, T]:
        """Lift a pure value into StateT."""
        from .result import Ok
        return StateT(lambda state: Ok((value, state)))
    
    @staticmethod
    def get() -> StateT[S, S]:
        """Get the current state."""
        from .result import Ok
        return StateT(lambda state: Ok((state, state)))
    
    @staticmethod
    def put(new_state: S) -> StateT[S, None]:
        """Set the state."""
        from .result import Ok
        return StateT(lambda _: Ok((None, new_state)))
    
    @staticmethod
    def modify(f: Callable[[S], S]) -> StateT[S, None]:
        """Modify the state."""
        from .result import Ok
        return StateT(lambda state: Ok((None, f(state))))
    
    def eval_state(self, initial_state: S) -> 'Result[T, Any]':
        """Run and return only the value."""
        return self.run(initial_state).map(lambda pair: pair[0])
    
    def exec_state(self, initial_state: S) -> 'Result[S, Any]':
        """Run and return only the final state."""
        return self.run(initial_state).map(lambda pair: pair[1])


# Helper functions
def state_pure(value: T) -> State[Any, T]:
    """Create a pure State value."""
    return State.pure(value)


def get_state() -> State[S, S]:
    """Get the current state."""
    return State.get()


def put_state(new_state: S) -> State[S, None]:
    """Set the state."""
    return State.put(new_state)


def modify_state(f: Callable[[S], S]) -> State[S, None]:
    """Modify the state."""
    return State.modify(f)
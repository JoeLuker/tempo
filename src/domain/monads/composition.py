"""Utilities for composing monadic operations.

This module provides helper functions and combinators for working with monads.
"""

from typing import TypeVar, Callable, Any
from functools import reduce

from .result import Result, Ok, Err
from .maybe import Maybe, some, nothing
from .either import Either, left, right
from .io import IO
from .reader import Reader
from .state import State

T = TypeVar('T')
U = TypeVar('U')
E = TypeVar('E')
R = TypeVar('R')
S = TypeVar('S')
L = TypeVar('L')  # Left type for Either
V = TypeVar('V')  # Additional type variable


# Result combinators

def sequence_results(results: list[Result[T, E]]) -> Result[list[T], E]:
    """Convert a list of Results into a Result of list.
    
    Fails fast - returns first error encountered.
    """
    values = []
    for result in results:
        if result.is_err():
            return result  # type: ignore
        values.append(result.unwrap())
    return Ok(values)


def parallel_results(results: list[Result[T, E]]) -> Result[list[T], list[E]]:
    """Collect all results, accumulating errors."""
    successes = []
    errors = []
    
    for result in results:
        result.fold(
            lambda err: errors.append(err),
            lambda val: successes.append(val)
        )
    
    if errors:
        return Err(errors)
    return Ok(successes)


def traverse_result(
    f: Callable[[T], Result[U, E]], 
    items: list[T]
) -> Result[list[U], E]:
    """Apply a Result-returning function to each item and collect results."""
    return sequence_results([f(item) for item in items])


def partition_results(
    results: list[Result[T, E]]
) -> tuple[list[T], list[E]]:
    """Partition a list of Results into successes and failures."""
    successes = []
    failures = []
    
    for result in results:
        if result.is_ok():
            successes.append(result.unwrap())
        else:
            failures.append(result.unwrap_err())
    
    return successes, failures


# Maybe combinators

def sequence_maybes(maybes: list[Maybe[T]]) -> Maybe[list[T]]:
    """Convert a list of Maybes into a Maybe of list.
    
    Returns Nothing if any Maybe is Nothing.
    """
    values = []
    for maybe in maybes:
        if maybe.is_nothing():
            return nothing()
        values.append(maybe.get_or_else(None))  # type: ignore
    return some(values)


def cat_maybes(maybes: list[Maybe[T]]) -> list[T]:
    """Extract all Some values from a list of Maybes."""
    return [m.get_or_else(None) for m in maybes if m.is_some()]  # type: ignore


def first_some(maybes: list[Maybe[T]]) -> Maybe[T]:
    """Return the first Some value, or Nothing if all are Nothing."""
    for maybe in maybes:
        if maybe.is_some():
            return maybe
    return nothing()


# Either combinators

def sequence_eithers(eithers: list[Either[L, R]]) -> Either[L, list[R]]:
    """Convert a list of Eithers into an Either of list.
    
    Returns first Left encountered.
    """
    rights = []
    for either in eithers:
        if either.is_left():
            return either  # type: ignore
        rights.append(either.get_or_else(None))  # type: ignore
    return right(rights)


def partition_eithers(
    eithers: list[Either[L, R]]
) -> tuple[list[L], list[R]]:
    """Partition a list of Eithers into lefts and rights."""
    lefts = []
    rights = []
    
    for either in eithers:
        either.fold(
            lambda l: lefts.append(l),
            lambda r: rights.append(r)
        )
    
    return lefts, rights


# IO combinators

def sequence_io(ios: list[IO[T]]) -> IO[list[T]]:
    """Convert a list of IO actions into an IO of list."""
    return IO.sequence(ios)


def map_m_io(f: Callable[[T], IO[U]], items: list[T]) -> IO[list[U]]:
    """Map a function returning IO over a list."""
    return sequence_io([f(item) for item in items])


def for_m_io(items: list[T], f: Callable[[T], IO[Any]]) -> IO[None]:
    """Execute IO actions for side effects, discarding results."""
    def run_all():
        for item in items:
            f(item).run()
        return None
    return IO(run_all)


# Reader combinators

def local_reader(
    f: Callable[[R], R], 
    reader: Reader[R, T]
) -> Reader[R, T]:
    """Run a Reader with a locally modified environment."""
    return reader.local(f)


def compose_readers(
    *readers: Reader[R, Any]
) -> Reader[R, list[Any]]:
    """Compose multiple Readers into a single Reader."""
    def run_all(env: R) -> list[Any]:
        return [reader.run(env) for reader in readers]
    return Reader(run_all)


# State combinators

def sequence_state(states: list[State[S, T]]) -> State[S, list[T]]:
    """Convert a list of State actions into a State of list."""
    def run_all(initial_state: S) -> tuple[list[T], S]:
        results = []
        state = initial_state
        
        for state_action in states:
            value, state = state_action.run(state)
            results.append(value)
        
        return results, state
    
    return State(run_all)


def map_m_state(
    f: Callable[[T], State[S, U]], 
    items: list[T]
) -> State[S, list[U]]:
    """Map a State-returning function over a list."""
    return sequence_state([f(item) for item in items])


# General combinators

def compose(*functions: Callable) -> Callable:
    """Compose functions from right to left.
    
    compose(f, g, h)(x) = f(g(h(x)))
    """
    def composed(x):
        return reduce(lambda acc, f: f(acc), reversed(functions), x)
    return composed


def pipe(*functions: Callable) -> Callable:
    """Compose functions from left to right.
    
    pipe(f, g, h)(x) = h(g(f(x)))
    """
    def piped(x):
        return reduce(lambda acc, f: f(acc), functions, x)
    return piped


def curry(f: Callable) -> Callable:
    """Convert a function to curried form."""
    import functools
    
    @functools.wraps(f)
    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= f.__code__.co_argcount:
            return f(*args, **kwargs)
        return lambda *more_args, **more_kwargs: curried(
            *(args + more_args), 
            **{**kwargs, **more_kwargs}
        )
    
    return curried


# Kleisli composition for Result monad
def kleisli_result(
    f: Callable[[T], Result[U, E]], 
    g: Callable[[U], Result[V, E]]
) -> Callable[[T], Result[V, E]]:
    """Compose two Result-returning functions."""
    return lambda x: f(x).flat_map(g)


# Applicative style operations
def lift2_result(
    f: Callable[[T, U], V]
) -> Callable[[Result[T, E], Result[U, E]], Result[V, E]]:
    """Lift a binary function to work with Results."""
    def lifted(ra: Result[T, E], rb: Result[U, E]) -> Result[V, E]:
        if ra.is_ok() and rb.is_ok():
            return Ok(f(ra.unwrap(), rb.unwrap()))
        elif ra.is_err():
            return ra  # type: ignore
        else:
            return rb  # type: ignore
    return lifted


def lift2_maybe(
    f: Callable[[T, U], V]
) -> Callable[[Maybe[T], Maybe[U]], Maybe[V]]:
    """Lift a binary function to work with Maybes."""
    def lifted(ma: Maybe[T], mb: Maybe[U]) -> Maybe[V]:
        if ma.is_some() and mb.is_some():
            return some(f(
                ma.get_or_else(None),  # type: ignore
                mb.get_or_else(None)   # type: ignore
            ))
        return nothing()
    return lifted
"""Monadic types for functional programming in TEMPO."""

from .result import Result, Ok, Err, ResultT
from .maybe import Maybe, Some, Nothing, MaybeT
from .either import Either, Left, Right, EitherT
from .io import IO, IOResult
from .reader import Reader, ReaderT
from .state import State, StateT

__all__ = [
    # Result monad
    'Result', 'Ok', 'Err', 'ResultT',
    # Maybe monad
    'Maybe', 'Some', 'Nothing', 'MaybeT',
    # Either monad
    'Either', 'Left', 'Right', 'EitherT',
    # IO monad
    'IO', 'IOResult',
    # Reader monad
    'Reader', 'ReaderT',
    # State monad
    'State', 'StateT',
]
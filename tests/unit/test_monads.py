"""Unit tests for monadic types."""

import pytest
from typing import List

from src.domain.monads import (
    Result, Ok, Err, 
    Maybe, some, nothing,
    Either, left, right,
    IO, io_pure, io_lift,
    Reader, reader_pure,
    State, state_pure
)
from src.domain.monads.composition import (
    sequence_results, parallel_results, traverse_result,
    sequence_maybes, cat_maybes, first_some,
    sequence_eithers, partition_eithers,
    compose, pipe, curry, kleisli_result
)


class TestResult:
    """Tests for Result monad."""
    
    def test_ok_creation_and_access(self):
        result = Ok(42)
        assert result.is_ok()
        assert not result.is_err()
        assert result.unwrap() == 42
        assert result.unwrap_or(0) == 42
    
    def test_err_creation_and_access(self):
        result = Err("error")
        assert result.is_err()
        assert not result.is_ok()
        assert result.unwrap_err() == "error"
        assert result.unwrap_or(0) == 0
    
    def test_map_on_ok(self):
        result = Ok(5).map(lambda x: x * 2)
        assert result.is_ok()
        assert result.unwrap() == 10
    
    def test_map_on_err(self):
        result = Err("error").map(lambda x: x * 2)
        assert result.is_err()
        assert result.unwrap_err() == "error"
    
    def test_flat_map_on_ok(self):
        result = Ok(5).flat_map(lambda x: Ok(x * 2))
        assert result.unwrap() == 10
        
        result = Ok(5).flat_map(lambda x: Err("failed"))
        assert result.is_err()
        assert result.unwrap_err() == "failed"
    
    def test_flat_map_on_err(self):
        result = Err("error").flat_map(lambda x: Ok(x * 2))
        assert result.is_err()
        assert result.unwrap_err() == "error"
    
    def test_fold(self):
        ok_result = Ok(5)
        assert ok_result.fold(lambda e: 0, lambda v: v * 2) == 10
        
        err_result = Err("error")
        assert err_result.fold(lambda e: len(e), lambda v: 0) == 5
    
    def test_try_result(self):
        from src.domain.monads.result import try_result
        
        # Success case
        result = try_result(lambda: 42)
        assert result.is_ok()
        assert result.unwrap() == 42
        
        # Exception case
        result = try_result(lambda: 1 / 0)
        assert result.is_err()
        assert isinstance(result.unwrap_err(), ZeroDivisionError)
    
    def test_sequence_results(self):
        # All Ok
        results = [Ok(1), Ok(2), Ok(3)]
        sequenced = sequence_results(results)
        assert sequenced.is_ok()
        assert sequenced.unwrap() == [1, 2, 3]
        
        # Contains Err
        results = [Ok(1), Err("error"), Ok(3)]
        sequenced = sequence_results(results)
        assert sequenced.is_err()
        assert sequenced.unwrap_err() == "error"
    
    def test_parallel_results(self):
        # All Ok
        results = [Ok(1), Ok(2), Ok(3)]
        parallel = parallel_results(results)
        assert parallel.is_ok()
        assert parallel.unwrap() == [1, 2, 3]
        
        # Contains Errs
        results = [Ok(1), Err("error1"), Ok(3), Err("error2")]
        parallel = parallel_results(results)
        assert parallel.is_err()
        assert parallel.unwrap_err() == ["error1", "error2"]


class TestMaybe:
    """Tests for Maybe monad."""
    
    def test_some_creation_and_access(self):
        maybe = some(42)
        assert maybe.is_some()
        assert not maybe.is_nothing()
        assert maybe.get_or_else(0) == 42
        assert maybe.to_optional() == 42
    
    def test_nothing_creation_and_access(self):
        maybe = nothing()
        assert maybe.is_nothing()
        assert not maybe.is_some()
        assert maybe.get_or_else(0) == 0
        assert maybe.to_optional() is None
    
    def test_map_on_some(self):
        maybe = some(5).map(lambda x: x * 2)
        assert maybe.is_some()
        assert maybe.get_or_else(0) == 10
    
    def test_map_on_nothing(self):
        maybe = nothing().map(lambda x: x * 2)
        assert maybe.is_nothing()
    
    def test_flat_map_on_some(self):
        maybe = some(5).flat_map(lambda x: some(x * 2))
        assert maybe.get_or_else(0) == 10
        
        maybe = some(5).flat_map(lambda x: nothing())
        assert maybe.is_nothing()
    
    def test_filter(self):
        maybe = some(5).filter(lambda x: x > 3)
        assert maybe.is_some()
        assert maybe.get_or_else(0) == 5
        
        maybe = some(5).filter(lambda x: x > 10)
        assert maybe.is_nothing()
    
    def test_or_else(self):
        assert some(5).or_else(some(10)).get_or_else(0) == 5
        assert nothing().or_else(some(10)).get_or_else(0) == 10
    
    def test_from_optional(self):
        from src.domain.monads.maybe import from_optional
        
        assert from_optional(42).is_some()
        assert from_optional(42).get_or_else(0) == 42
        assert from_optional(None).is_nothing()
    
    def test_sequence_maybes(self):
        # All Some
        maybes = [some(1), some(2), some(3)]
        sequenced = sequence_maybes(maybes)
        assert sequenced.is_some()
        assert sequenced.get_or_else([]) == [1, 2, 3]
        
        # Contains Nothing
        maybes = [some(1), nothing(), some(3)]
        sequenced = sequence_maybes(maybes)
        assert sequenced.is_nothing()
    
    def test_cat_maybes(self):
        maybes = [some(1), nothing(), some(3), nothing(), some(5)]
        values = cat_maybes(maybes)
        assert values == [1, 3, 5]


class TestEither:
    """Tests for Either monad."""
    
    def test_left_creation_and_access(self):
        either = left("error")
        assert either.is_left()
        assert not either.is_right()
        assert either.get_or_else("default") == "default"
    
    def test_right_creation_and_access(self):
        either = right(42)
        assert either.is_right()
        assert not either.is_left()
        assert either.get_or_else(0) == 42
    
    def test_map_on_right(self):
        either = right(5).map(lambda x: x * 2)
        assert either.is_right()
        assert either.fold(lambda l: 0, lambda r: r) == 10
    
    def test_map_on_left(self):
        either = left("error").map(lambda x: x * 2)
        assert either.is_left()
        assert either.fold(lambda l: l, lambda r: 0) == "error"
    
    def test_map_left(self):
        either = left("error").map_left(lambda x: x.upper())
        assert either.is_left()
        assert either.fold(lambda l: l, lambda r: "") == "ERROR"
    
    def test_flat_map(self):
        either = right(5).flat_map(lambda x: right(x * 2))
        assert either.fold(lambda l: 0, lambda r: r) == 10
        
        either = right(5).flat_map(lambda x: left("failed"))
        assert either.is_left()
        assert either.fold(lambda l: l, lambda r: "") == "failed"
    
    def test_swap(self):
        either = right(42).swap()
        assert either.is_left()
        assert either.fold(lambda l: l, lambda r: 0) == 42
        
        either = left("error").swap()
        assert either.is_right()
        assert either.fold(lambda l: "", lambda r: r) == "error"
    
    def test_to_result(self):
        either = right(42)
        result = either.to_result()
        assert result.is_ok()
        assert result.unwrap() == 42
        
        either = left("error")
        result = either.to_result()
        assert result.is_err()
        assert result.unwrap_err() == "error"


class TestIO:
    """Tests for IO monad."""
    
    def test_pure(self):
        io = io_pure(42)
        assert io.run() == 42
    
    def test_lift(self):
        counter = [0]
        
        def increment():
            counter[0] += 1
            return counter[0]
        
        io = io_lift(increment)
        assert counter[0] == 0  # Not executed yet
        assert io.run() == 1
        assert counter[0] == 1  # Now executed
    
    def test_map(self):
        io = io_pure(5).map(lambda x: x * 2)
        assert io.run() == 10
    
    def test_flat_map(self):
        io = io_pure(5).flat_map(lambda x: io_pure(x * 2))
        assert io.run() == 10
    
    def test_and_then(self):
        results = []
        io1 = io_lift(lambda: results.append(1))
        io2 = io_lift(lambda: results.append(2))
        
        combined = io1.and_then(io2)
        combined.run()
        assert results == [1, 2]
    
    def test_or_else(self):
        io = io_lift(lambda: 1 / 0).or_else(lambda e: io_pure(0))
        assert io.run() == 0
    
    def test_sequence(self):
        counter = [0]
        
        def increment():
            counter[0] += 1
            return counter[0]
        
        ios = [io_lift(increment) for _ in range(3)]
        sequenced = IO.sequence(ios)
        
        assert counter[0] == 0
        result = sequenced.run()
        assert result == [1, 2, 3]
        assert counter[0] == 3


class TestReader:
    """Tests for Reader monad."""
    
    def test_pure(self):
        reader = reader_pure(42)
        assert reader.run("env") == 42
        assert reader("env") == 42  # Using __call__
    
    def test_ask(self):
        from src.domain.monads.reader import ask
        reader = ask()
        assert reader.run("environment") == "environment"
    
    def test_asks(self):
        from src.domain.monads.reader import asks
        reader = asks(lambda env: env["key"])
        assert reader.run({"key": "value"}) == "value"
    
    def test_map(self):
        reader = reader_pure(5).map(lambda x: x * 2)
        assert reader.run("env") == 10
    
    def test_flat_map(self):
        from src.domain.monads.reader import ask
        reader = ask().flat_map(lambda env: reader_pure(len(env)))
        assert reader.run("hello") == 5
    
    def test_local(self):
        from src.domain.monads.reader import ask
        reader = ask().local(lambda env: env.upper())
        assert reader.run("hello") == "HELLO"


class TestState:
    """Tests for State monad."""
    
    def test_pure(self):
        state = state_pure(42)
        value, new_state = state.run("initial")
        assert value == 42
        assert new_state == "initial"
    
    def test_get(self):
        from src.domain.monads.state import get_state
        state = get_state()
        value, new_state = state.run("current")
        assert value == "current"
        assert new_state == "current"
    
    def test_put(self):
        from src.domain.monads.state import put_state
        state = put_state("new")
        value, new_state = state.run("old")
        assert value is None
        assert new_state == "new"
    
    def test_modify(self):
        from src.domain.monads.state import modify_state
        state = modify_state(lambda s: s * 2)
        value, new_state = state.run(5)
        assert value is None
        assert new_state == 10
    
    def test_map(self):
        state = state_pure(5).map(lambda x: x * 2)
        value, new_state = state.run("state")
        assert value == 10
        assert new_state == "state"
    
    def test_flat_map(self):
        from src.domain.monads.state import get_state, put_state
        # Get current state, multiply by 2, set as new state
        state = get_state().flat_map(lambda s: put_state(s * 2).then(state_pure(s)))
        value, new_state = state.run(5)
        assert value == 5  # Original state
        assert new_state == 10  # Modified state


class TestComposition:
    """Tests for composition utilities."""
    
    def test_compose(self):
        add1 = lambda x: x + 1
        mul2 = lambda x: x * 2
        sub3 = lambda x: x - 3
        
        composed = compose(add1, mul2, sub3)
        assert composed(5) == 5  # (5 - 3) * 2 + 1 = 5
    
    def test_pipe(self):
        add1 = lambda x: x + 1
        mul2 = lambda x: x * 2
        sub3 = lambda x: x - 3
        
        piped = pipe(add1, mul2, sub3)
        assert piped(5) == 9  # ((5 + 1) * 2) - 3 = 9
    
    def test_curry(self):
        def add(x, y, z):
            return x + y + z
        
        curried = curry(add)
        assert curried(1)(2)(3) == 6
        assert curried(1, 2)(3) == 6
        assert curried(1)(2, 3) == 6
        assert curried(1, 2, 3) == 6
    
    def test_kleisli_result(self):
        def safe_div(x: float) -> Result[float, str]:
            if x == 0:
                return Err("Division by zero")
            return Ok(10 / x)
        
        def safe_sqrt(x: float) -> Result[float, str]:
            if x < 0:
                return Err("Square root of negative")
            return Ok(x ** 0.5)
        
        composed = kleisli_result(safe_div, safe_sqrt)
        
        result = composed(2)  # 10/2 = 5, sqrt(5) ≈ 2.236
        assert result.is_ok()
        assert abs(result.unwrap() - 2.236) < 0.001
        
        result = composed(0)  # Division by zero
        assert result.is_err()
        assert result.unwrap_err() == "Division by zero"
        
        result = composed(-2)  # 10/-2 = -5, sqrt(-5) = error
        assert result.is_err()
        assert result.unwrap_err() == "Square root of negative"
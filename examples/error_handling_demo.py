#!/usr/bin/env python3
"""
Error handling and logging demonstration.

This script demonstrates how to use the enhanced error handling and logging
systems in the TEMPO project.
"""

import os
import time
import random
import logging
from typing import List, Dict, Any

# Set up logging before imports to ensure proper configuration
logging.basicConfig(level=logging.INFO)

from src.utils.logger import get_logger, add_global_context
from src.utils.error_manager import (
    TempoError,
    ErrorCode,
    ErrorSeverity,
    ErrorContext,
    ModelError,
    TokenizationError,
    GenerationError,
    ConfigurationError,
    ResourceError,
    safe_execute,
)
from src.utils.exception_handlers import (
    handle_exceptions,
    handle_model_errors,
    handle_tokenization_errors,
    handle_generation_errors,
    retry,
    exception_context,
)
from src.utils.logging_utils import LoggingMixin


# Create a logger for this demo
logger = get_logger("error_demo")

# Add global context for all logs in this script
add_global_context(script="error_handling_demo.py", env="demo")


class DemoClass(LoggingMixin):
    """Demonstration class using LoggingMixin."""

    def __init__(self, name: str):
        """Initialize with a name."""
        super().__init__()
        self.name = name

        # Set up logging with debug mode
        self.setup_logging("demo_class", debug_mode=True)

        self.log("DemoClass initialized")

    def risky_operation(self, value: int) -> int:
        """An operation that might fail."""
        self.log(f"Performing risky operation with value {value}")

        # Use operation context manager for tracking
        with self.operation("risky_calculation"):
            # Randomly fail for demonstration
            if random.random() < 0.3:
                self.log("Operation failed", level="error")
                raise ValueError(f"Random failure with value {value}")

            result = value * 2
            self.log(f"Operation succeeded with result {result}")
            return result

    @handle_exceptions(
        error_message="Demo operation failed",
        error_code=ErrorCode.VALIDATION_ERROR,
        reraise=False,
        default_return=-1,
    )
    def safe_operation(self, value: int) -> int:
        """A safe operation that handles exceptions."""
        self.log(f"Performing safe operation with value {value}")

        # Add context for this specific operation
        with self.with_context(operation_type="calculation", input_value=value):
            # Randomly fail for demonstration
            if random.random() < 0.5:
                self.log("Operation might fail here", level="warning")
                raise ValueError(f"Random failure with value {value}")

            result = value * 3
            self.log_success(f"Operation succeeded with result {result}")
            return result

    @retry(
        max_attempts=3,
        retry_delay=0.5,
        backoff_factor=2.0,
        exceptions=(ValueError, KeyError),
    )
    def retry_operation(self, value: int) -> int:
        """An operation that will be retried on failure."""
        self.log(f"Attempt: retry operation with value {value}")

        # Succeed only on specific values for demonstration
        if value % 3 != 0:
            self.log("Operation failed, will retry", level="warning")
            raise ValueError(f"Value {value} is not divisible by 3")

        result = value * 5
        self.log_success(f"Retry operation succeeded with result {result}")
        return result


# Demonstration functions with various decorators


@handle_model_errors(error_message="Model demo failed")
def model_demo(model_id: str) -> bool:
    """Demonstrate model error handling."""
    logger.info(f"Loading model {model_id}")

    # Simulate model loading failure
    if "invalid" in model_id:
        raise Exception(f"Could not load model: {model_id}")

    logger.success(f"Model {model_id} loaded successfully")
    return True


@handle_tokenization_errors(error_message="Tokenization demo failed")
def tokenization_demo(text: str) -> List[int]:
    """Demonstrate tokenization error handling."""
    logger.info(f"Tokenizing text: {text[:20]}...")

    # Simulate tokenization
    if not text or len(text) < 5:
        raise ValueError("Text too short for tokenization")

    # Fake tokens
    tokens = [hash(char) % 1000 for char in text]

    logger.success(f"Tokenized {len(tokens)} tokens")
    return tokens


@handle_generation_errors()
def generation_demo(tokens: List[int], max_length: int) -> str:
    """Demonstrate generation error handling."""
    logger.info(f"Generating from {len(tokens)} tokens, max_length={max_length}")

    # Simulate generation
    with exception_context("token_generation", error_code=ErrorCode.GENERATION_FAILED):
        if max_length > 100:
            raise ResourceError(
                message="Max length too large", code=ErrorCode.RESOURCE_EXHAUSTED
            )

        # Fake generation
        time.sleep(0.1)  # Simulate processing time
        result = "".join([chr(65 + (token % 26)) for token in tokens])

        if len(result) > max_length:
            result = result[:max_length]

        logger.success(f"Generated {len(result)} characters")
        return result


def main():
    """Run the demo."""
    logger.info("Starting error handling and logging demo", demo_version="1.0")

    try:
        # Demonstrate LoggingMixin
        logger.info("--- LoggingMixin Demo ---")
        demo = DemoClass("test_demo")

        # Try risky operation a few times
        for i in range(3):
            try:
                result = demo.risky_operation(i)
                logger.info(f"Risky operation {i} result: {result}")
            except Exception as e:
                logger.error(f"Caught exception from risky operation: {e}")

        # Try safe operation that handles its own exceptions
        for i in range(3):
            result = demo.safe_operation(i)
            logger.info(f"Safe operation {i} result: {result}")

        # Try retry operation
        for i in range(1, 7):
            try:
                result = demo.retry_operation(i)
                logger.info(f"Retry operation {i} result: {result}")
            except Exception as e:
                logger.error(f"Retry finally failed for {i}: {e}")

        # Demonstrate decorators
        logger.info("\n--- Decorator Demo ---")

        # Model demo
        try:
            model_demo("valid_model")
        except TempoError as e:
            logger.error(f"Model demo error: {e}")

        try:
            model_demo("invalid_model")
        except TempoError as e:
            logger.error(f"Expected model error: {e.code}")

        # Tokenization demo
        try:
            tokens = tokenization_demo("This is a test of tokenization")
            logger.info(f"Got {len(tokens)} tokens")

            # Should fail
            tokenization_demo("")
        except TokenizationError as e:
            logger.error(f"Expected tokenization error: {e.code}")

        # Generation demo
        try:
            result = generation_demo([1, 2, 3, 4, 5], 50)
            logger.info(f"Generation result: {result}")

            # Should fail
            generation_demo([1, 2, 3], 200)
        except GenerationError as e:
            logger.error(f"Expected generation error: {e.code}")

        # Demonstrate safe_execute
        logger.info("\n--- safe_execute Demo ---")

        def might_fail(divisor):
            return 100 / divisor

        # Safe execution with default return
        result = safe_execute(
            lambda: might_fail(0),
            error_message="Division failed",
            error_code=ErrorCode.VALIDATION_ERROR,
            raise_on_error=False,
            default_return="N/A",
        )
        logger.info(f"Safe execution result: {result}")

        # Safe execution that raises
        try:
            result = safe_execute(
                lambda: might_fail(0),
                error_message="Division failed",
                error_code=ErrorCode.VALIDATION_ERROR,
                raise_on_error=True,
            )
        except TempoError as e:
            logger.error(f"Expected safe_execute error: {e.code}")

        logger.success("Demo completed successfully")

    except Exception as e:
        logger.critical(f"Unexpected error in demo: {e}", exc_info=True)

    finally:
        # Print metrics report
        metrics = logger.metrics_report()
        logger.info(
            "Logger metrics report:",
            total_logs=metrics["total_logs"],
            duration=f"{metrics['elapsed_seconds']:.2f}s",
        )


if __name__ == "__main__":
    main()

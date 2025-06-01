"""
Error management for TEMPO project.

This module provides a centralized error handling system with standardized
error classes, error codes, and error reporting functionality.
"""

import traceback
import inspect
import logging
import sys
from enum import Enum
from typing import Dict, List, Any, Optional, Type, Union, Callable
from dataclasses import dataclass

from src.utils.config_manager import config

# Configure module logger
logger = logging.getLogger("tempo-error")


class ErrorSeverity(Enum):
    """Severity levels for errors."""

    CRITICAL = "CRITICAL"  # Application cannot continue
    ERROR = "ERROR"  # Operation failed, but application can continue
    WARNING = "WARNING"  # Potentially problematic situation
    INFO = "INFO"  # Informational message about an error


class ErrorCategory(Enum):
    """Categories for errors to help with grouping and filtering."""

    INITIALIZATION = "INITIALIZATION"  # Errors during system startup/initialization
    CONFIGURATION = "CONFIGURATION"  # Configuration-related errors
    MODEL = "MODEL"  # Model-related errors (loading, inference)
    TOKENIZATION = "TOKENIZATION"  # Tokenization-related errors
    GENERATION = "GENERATION"  # Text generation errors
    API = "API"  # API-related errors
    VALIDATION = "VALIDATION"  # Input validation errors
    RESOURCE = "RESOURCE"  # Resource-related errors (memory, disk, etc.)
    SYSTEM = "SYSTEM"  # System-level errors (OS, hardware)
    UNKNOWN = "UNKNOWN"  # Default category


class ErrorCode(Enum):
    """
    Standard error codes for the TEMPO project.

    Format: CATEGORY_DESCRIPTION
    Example: MODEL_LOAD_FAILED
    """

    # Initialization errors
    INIT_FAILED = "INIT_FAILED"
    INIT_COMPONENT_MISSING = "INIT_COMPONENT_MISSING"

    # Configuration errors
    CONFIG_INVALID = "CONFIG_INVALID"
    CONFIG_MISSING = "CONFIG_MISSING"
    CONFIG_FILE_ERROR = "CONFIG_FILE_ERROR"

    # Model errors
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    MODEL_INFERENCE_FAILED = "MODEL_INFERENCE_FAILED"
    MODEL_SAVE_FAILED = "MODEL_SAVE_FAILED"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_INVALID = "MODEL_INVALID"
    MODEL_UNSUPPORTED = "MODEL_UNSUPPORTED"

    # Tokenization errors
    TOKENIZE_FAILED = "TOKENIZE_FAILED"
    DECODE_FAILED = "DECODE_FAILED"

    # Generation errors
    GENERATION_FAILED = "GENERATION_FAILED"
    GENERATION_TIMEOUT = "GENERATION_TIMEOUT"
    GENERATION_INVALID_PARAMS = "GENERATION_INVALID_PARAMS"
    GENERATION_TOO_LONG = "GENERATION_TOO_LONG"

    # API errors
    API_REQUEST_INVALID = "API_REQUEST_INVALID"
    API_RATE_LIMIT = "API_RATE_LIMIT"
    API_UNAUTHORIZED = "API_UNAUTHORIZED"
    API_FORBIDDEN = "API_FORBIDDEN"
    API_NOT_FOUND = "API_NOT_FOUND"
    API_SERVER_ERROR = "API_SERVER_ERROR"
    API_TIMEOUT = "API_TIMEOUT"

    # Resource errors
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_BUSY = "RESOURCE_BUSY"

    # System errors
    SYSTEM_ERROR = "SYSTEM_ERROR"
    SYSTEM_INTERRUPT = "SYSTEM_INTERRUPT"

    # General errors
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"

    @classmethod
    def get_category(cls, code: "ErrorCode") -> ErrorCategory:
        """Get the category for an error code."""
        code_str = code.value
        for category in ErrorCategory:
            if code_str.startswith(category.name):
                return category
        return ErrorCategory.UNKNOWN


@dataclass
class ErrorContext:
    """
    Context information for an error.

    Contains detailed information about the context in which an error occurred.
    """

    module: str
    function: str
    line_number: int
    file_path: str
    call_stack: List[str]
    args: Dict[str, Any] = None
    additional_info: Dict[str, Any] = None

    @classmethod
    def current(
        cls,
        include_args: bool = False,
        stack_depth: int = 2,
        args_filter: Optional[Callable[[str, Any], bool]] = None,
    ) -> "ErrorContext":
        """
        Create an ErrorContext from the current execution context.

        Args:
            include_args: Whether to include function arguments in the context
            stack_depth: How far up the stack to look for caller information
            args_filter: Optional function to filter arguments (name, value) -> bool

        Returns:
            ErrorContext: Context information for the current execution point
        """
        # Get the frame for the caller
        frame = inspect.currentframe()
        try:
            for _ in range(stack_depth):
                frame = frame.f_back
                if frame is None:
                    break

            if frame is None:
                return cls(
                    module="unknown",
                    function="unknown",
                    line_number=-1,
                    file_path="unknown",
                    call_stack=traceback.format_stack(),
                    args={},
                    additional_info={},
                )

            # Get information about the frame
            frameinfo = inspect.getframeinfo(frame)
            module = inspect.getmodule(frame)
            module_name = module.__name__ if module else "unknown"

            # Get call stack
            call_stack = traceback.format_stack()

            # Get arguments if requested
            args_dict = {}
            if include_args:
                try:
                    args_dict = {
                        name: value
                        for name, value in frame.f_locals.items()
                        if args_filter is None or args_filter(name, value)
                    }
                except Exception as e:
                    args_dict = {"error_getting_args": str(e)}

            return cls(
                module=module_name,
                function=frameinfo.function,
                line_number=frameinfo.lineno,
                file_path=frameinfo.filename,
                call_stack=call_stack,
                args=args_dict,
                additional_info={},
            )
        finally:
            # Make sure to clean up the frame reference to avoid reference cycles
            del frame


class TempoError(Exception):
    """
    Base exception class for TEMPO project.

    This class provides standardized error handling with detailed context and
    consistent formatting.
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        **kwargs,
    ):
        """
        Initialize a TempoError.

        Args:
            message: Error message
            code: Error code
            severity: Error severity level
            context: Error context information
            cause: Original exception that caused this error
            **kwargs: Additional context information to add to error
        """
        self.message = message
        self.code = code
        self.severity = severity
        self.cause = cause

        # Create context if not provided
        self.context = context or ErrorContext.current(include_args=True)

        # Add additional info to context
        if kwargs and self.context:
            if self.context.additional_info is None:
                self.context.additional_info = {}
            self.context.additional_info.update(kwargs)

        # Create the final error message
        full_message = f"{code.value}: {message}"
        if cause:
            full_message += f" (Caused by: {type(cause).__name__}: {str(cause)})"

        # Initialize the base Exception with the full message
        super().__init__(full_message)

        # Log the error based on severity
        self._log_error()

    def _log_error(self):
        """Log the error based on its severity."""
        log_message = self._format_for_logging()

        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif self.severity == ErrorSeverity.ERROR:
            logger.error(log_message)
        elif self.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        else:  # INFO or default
            logger.info(log_message)

    def _format_for_logging(self) -> str:
        """Format the error for logging."""
        parts = [f"ERROR [{self.code.value}] ({self.severity.value}): {self.message}"]

        if self.context:
            parts.append(
                f"Location: {self.context.module}.{self.context.function} ({self.context.file_path}:{self.context.line_number})"
            )

            if self.context.additional_info:
                parts.append("Additional Info:")
                for key, value in self.context.additional_info.items():
                    parts.append(f"  {key}: {value}")

        if self.cause:
            parts.append(f"Caused by: {type(self.cause).__name__}: {str(self.cause)}")

            if hasattr(self.cause, "__traceback__") and self.cause.__traceback__:
                tb_lines = traceback.format_tb(self.cause.__traceback__)
                parts.append("Cause Traceback:")
                for line in tb_lines:
                    parts.append(f"  {line.rstrip()}")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the error to a dictionary for serialization."""
        result = {
            "code": self.code.value,
            "message": self.message,
            "severity": self.severity.value,
        }

        if self.context:
            result["context"] = {
                "module": self.context.module,
                "function": self.context.function,
                "line_number": self.context.line_number,
                "file_path": self.context.file_path,
            }

            if self.context.additional_info:
                result["additional_info"] = self.context.additional_info

        if self.cause:
            result["cause"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause),
            }

        return result


# Specific error classes for different categories


class InitializationError(TempoError):
    """Error during initialization."""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.INIT_FAILED, **kwargs):
        super().__init__(message, code=code, severity=ErrorSeverity.CRITICAL, **kwargs)


class ConfigurationError(TempoError):
    """Error related to configuration."""

    def __init__(
        self, message: str, code: ErrorCode = ErrorCode.CONFIG_INVALID, **kwargs
    ):
        super().__init__(message, code=code, severity=ErrorSeverity.ERROR, **kwargs)


class ModelError(TempoError):
    """Error related to model operations."""

    def __init__(
        self, message: str, code: ErrorCode = ErrorCode.MODEL_LOAD_FAILED, **kwargs
    ):
        super().__init__(message, code=code, severity=ErrorSeverity.ERROR, **kwargs)


class TokenizationError(TempoError):
    """Error related to tokenization."""

    def __init__(
        self, message: str, code: ErrorCode = ErrorCode.TOKENIZE_FAILED, **kwargs
    ):
        super().__init__(message, code=code, severity=ErrorSeverity.ERROR, **kwargs)


class GenerationError(TempoError):
    """Error during text generation."""

    def __init__(
        self, message: str, code: ErrorCode = ErrorCode.GENERATION_FAILED, **kwargs
    ):
        super().__init__(message, code=code, severity=ErrorSeverity.ERROR, **kwargs)


class ResourceError(TempoError):
    """Error related to resources (memory, disk, etc.)."""

    def __init__(
        self, message: str, code: ErrorCode = ErrorCode.RESOURCE_EXHAUSTED, **kwargs
    ):
        super().__init__(message, code=code, severity=ErrorSeverity.ERROR, **kwargs)


class ValidationError(TempoError):
    """Error related to input validation."""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        additional_info = kwargs.pop("additional_info", {}) or {}
        if field:
            additional_info["field"] = field

        super().__init__(
            message,
            code=ErrorCode.VALIDATION_ERROR,
            severity=ErrorSeverity.WARNING,
            additional_info=additional_info,
            **kwargs,
        )


def safe_execute(
    func,
    error_message: str = "Function execution failed",
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    error_class: Type[TempoError] = TempoError,
    log_args: bool = False,
    raise_on_error: bool = True,
    default_return=None,
):
    """
    Execute a function safely with standardized error handling.

    Args:
        func: Function to execute
        error_message: Message to use if the function fails
        error_code: Error code to use if the function fails
        error_class: Error class to use if the function fails
        log_args: Whether to log function arguments in the error context
        raise_on_error: Whether to raise an exception on error
        default_return: Value to return on error if not raising

    Returns:
        The result of the function, or default_return on error if not raising

    Raises:
        TempoError: If the function raises an exception and raise_on_error is True
    """
    try:
        return func()
    except Exception as e:
        # Create error context
        context = ErrorContext.current(include_args=log_args, stack_depth=3)

        # Create error
        error = error_class(
            message=error_message, code=error_code, cause=e, context=context
        )

        # Raise or return default
        if raise_on_error:
            raise error

        return default_return

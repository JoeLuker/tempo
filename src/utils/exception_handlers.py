"""
Exception handling utilities for TEMPO project.

This module provides decorators and context managers for standardized
exception handling across the project.
"""

import time
import functools
import traceback
import inspect
from typing import Dict, Any, Optional, Union, List, Callable, Type, TypeVar, Tuple
from contextlib import contextmanager

from src.utils.error_manager import (
    TempoError, ErrorCode, ErrorSeverity, ErrorContext,
    ModelError, TokenizationError, GenerationError, InitializationError,
    ConfigurationError, ResourceError, ValidationError
)
from src.utils.logger import get_logger

# Create module logger
logger = get_logger("exception_handlers")

# Type variable for generic function return type
T = TypeVar('T')


def handle_exceptions(
    error_message: str = "An error occurred",
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    error_class: Type[TempoError] = TempoError,
    reraise: bool = True,
    log_level: str = "error",
    default_return: Any = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for handling exceptions in a standardized way.
    
    Args:
        error_message: Message to log when an exception occurs
        error_code: Error code to use for the exception
        error_class: Error class to use for the exception
        reraise: Whether to reraise the exception
        log_level: Log level to use for the exception
        default_return: Value to return on exception if not reraising
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Don't wrap TempoErrors, just reraise them
                if isinstance(e, TempoError):
                    if reraise:
                        raise
                    else:
                        # Log but don't reraise
                        return default_return
                
                # Create error context
                context = ErrorContext.current(include_args=True)
                
                # Create error message with function info
                func_name = getattr(func, "__name__", "unknown")
                module_name = getattr(func, "__module__", "unknown")
                full_error_message = f"{error_message} in {module_name}.{func_name}"
                
                # Create and log the error
                error = error_class(
                    message=full_error_message,
                    code=error_code,
                    cause=e,
                    context=context
                )
                
                # Log the error
                if log_level == "debug":
                    logger.debug(f"Handled exception: {error}")
                elif log_level == "info":
                    logger.info(f"Handled exception: {error}")
                elif log_level == "warning":
                    logger.warning(f"Handled exception: {error}")
                elif log_level == "error":
                    logger.error(f"Handled exception: {error}")
                elif log_level == "critical":
                    logger.critical(f"Handled exception: {error}")
                
                # Reraise or return default
                if reraise:
                    raise error from e
                else:
                    return default_return
                    
        return wrapper
    return decorator


# Specialized decorators for common error cases

def handle_model_errors(error_message: str = "Model operation failed", **kwargs):
    """
    Decorator for handling model-related errors.
    
    Args:
        error_message: Error message
        **kwargs: Additional arguments for handle_exceptions
    """
    # Use provided error_code or default to MODEL_INFERENCE_FAILED
    if 'error_code' not in kwargs:
        kwargs['error_code'] = ErrorCode.MODEL_INFERENCE_FAILED
    
    return handle_exceptions(
        error_message=error_message,
        error_class=ModelError,
        **kwargs
    )


def handle_tokenization_errors(error_message: str = "Tokenization failed", **kwargs):
    """
    Decorator for handling tokenization-related errors.
    
    Args:
        error_message: Error message
        **kwargs: Additional arguments for handle_exceptions
    """
    # Use provided error_code or default to TOKENIZE_FAILED
    if 'error_code' not in kwargs:
        kwargs['error_code'] = ErrorCode.TOKENIZE_FAILED
    
    return handle_exceptions(
        error_message=error_message,
        error_class=TokenizationError,
        **kwargs
    )


def handle_generation_errors(error_message: str = "Text generation failed", **kwargs):
    """
    Decorator for handling generation-related errors.
    
    Args:
        error_message: Error message
        **kwargs: Additional arguments for handle_exceptions
    """
    # Use provided error_code or default to GENERATION_FAILED
    if 'error_code' not in kwargs:
        kwargs['error_code'] = ErrorCode.GENERATION_FAILED
    
    return handle_exceptions(
        error_message=error_message,
        error_class=GenerationError,
        **kwargs
    )


def handle_initialization_errors(error_message: str = "Initialization failed", **kwargs):
    """
    Decorator for handling initialization-related errors.
    
    Args:
        error_message: Error message
        **kwargs: Additional arguments for handle_exceptions
    """
    # Use provided error_code or default to INIT_FAILED
    if 'error_code' not in kwargs:
        kwargs['error_code'] = ErrorCode.INIT_FAILED
    
    return handle_exceptions(
        error_message=error_message,
        error_class=InitializationError,
        **kwargs
    )


def handle_resource_errors(error_message: str = "Resource operation failed", **kwargs):
    """
    Decorator for handling resource-related errors.
    
    Args:
        error_message: Error message
        **kwargs: Additional arguments for handle_exceptions
    """
    # Use provided error_code or default to RESOURCE_EXHAUSTED
    if 'error_code' not in kwargs:
        kwargs['error_code'] = ErrorCode.RESOURCE_EXHAUSTED
    
    return handle_exceptions(
        error_message=error_message,
        error_class=ResourceError,
        **kwargs
    )


def retry(
    max_attempts: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    error_message: str = "Operation failed after multiple retries",
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    error_class: Type[TempoError] = TempoError
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying a function on exception.
    
    Args:
        max_attempts: Maximum number of attempts
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each retry
        exceptions: Exception type(s) to retry on
        error_message: Message for the final error
        error_code: Error code for the final error
        error_class: Error class for the final error
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            delay = retry_delay
            
            # Get function info for logging
            func_name = getattr(func, "__name__", "unknown")
            module_name = getattr(func, "__module__", "unknown")
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # Log the retry
                    if attempt < max_attempts:
                        logger.warning(
                            f"Retry {attempt}/{max_attempts} for {module_name}.{func_name} "
                            f"after error: {type(e).__name__}: {str(e)}", 
                            exc_info=True
                        )
                        
                        # Wait before retrying
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        # Last attempt failed, log the final error
                        logger.error(
                            f"All {max_attempts} attempts failed for {module_name}.{func_name}",
                            exc_info=True
                        )
            
            # If we get here, all attempts failed
            context = ErrorContext.current(include_args=True)
            error = error_class(
                message=f"{error_message} in {module_name}.{func_name} after {max_attempts} attempts",
                code=error_code,
                cause=last_exception,
                context=context
            )
            raise error from last_exception
            
        return wrapper
    return decorator


@contextmanager
def exception_context(
    operation_name: str,
    error_message: Optional[str] = None,
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    error_class: Type[TempoError] = TempoError
):
    """
    Context manager for handling exceptions in a standardized way.
    
    Args:
        operation_name: Name of the operation for logging
        error_message: Error message (defaults to "Failed during {operation_name}")
        error_code: Error code for the exception
        error_class: Error class for the exception
        
    Yields:
        None
    """
    try:
        logger.debug(f"Starting operation: {operation_name}")
        start_time = time.time()
        yield
        elapsed = time.time() - start_time
        logger.debug(f"Completed operation: {operation_name} in {elapsed:.3f}s")
    except Exception as e:
        # Don't wrap TempoErrors
        if isinstance(e, TempoError):
            raise
            
        elapsed = time.time() - start_time
        if error_message is None:
            error_message = f"Failed during {operation_name}"
            
        # Create error context
        context = ErrorContext.current(include_args=True)
        
        # Add operation info to context
        if context.additional_info is None:
            context.additional_info = {}
        context.additional_info.update({
            "operation": operation_name,
            "duration": f"{elapsed:.3f}s"
        })
        
        # Create and raise error
        error = error_class(
            message=error_message,
            code=error_code,
            cause=e,
            context=context
        )
        
        logger.error(f"Error in operation {operation_name} after {elapsed:.3f}s: {str(error)}")
        raise error from e
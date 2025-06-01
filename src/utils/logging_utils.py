"""
Logging utilities for TEMPO project.

This module provides common logging functionality to standardize logging across all modules.
This is a compatibility layer that uses the enhanced logger.py module internally.
"""

import os
import logging
from typing import Optional

from src.utils.config_manager import config, get_debug_mode

# Import the enhanced logger (lazy import to avoid circular imports)
_logger_module = None


def _get_logger_module():
    """Get the logger module, importing it if needed."""
    global _logger_module
    if _logger_module is None:
        import src.utils.logger as logger_module

        _logger_module = logger_module
    return _logger_module


def setup_logger(
    logger_name: str,
    log_file_name: Optional[str] = None,
    debug_mode: Optional[bool] = None,
) -> logging.Logger:
    """
    Set up a logger with file handler and formatter.

    Args:
        logger_name: Name of the logger
        log_file_name: Name of the log file (without directory path)
        debug_mode: Whether to enable debug mode (overrides config if provided)

    Returns:
        logging.Logger: Configured logger instance
    """
    # Get debug mode from config if not explicitly provided
    if debug_mode is None:
        debug_mode = get_debug_mode(logger_name)

    # Use logger_name as log file name if not provided
    if log_file_name is None:
        log_file_name = f"{logger_name}_debug.log"

    # Use the enhanced logger
    tempo_logger = _get_logger_module().get_logger(logger_name)

    # Set debug mode context
    if debug_mode:
        _get_logger_module().add_global_context(debug_mode=debug_mode)

    # Log initialization
    tempo_logger.info(f"{logger_name} initialized with debug_mode={debug_mode}")

    # Return the underlying logger for compatibility
    return tempo_logger.logger


class LoggingMixin:
    """
    Mixin class to provide consistent logging functionality.

    Usage:
        class MyClass(LoggingMixin):
            def __init__(self):
                super().__init__()
                self.setup_logging("my_class")

            def my_method(self):
                self.log("This is a log message")
    """

    def setup_logging(
        self,
        logger_name: str,
        log_file_name: Optional[str] = None,
        debug_mode: Optional[bool] = None,
    ):
        """
        Set up logging for this class.

        Args:
            logger_name: Name of the logger
            log_file_name: Name of the log file (without directory path)
            debug_mode: Whether to enable debug mode (overrides config if provided)
        """
        # Get debug mode from config if not explicitly provided
        if debug_mode is None:
            debug_mode = get_debug_mode(logger_name)

        # Get enhanced logger
        self._tempo_logger = _get_logger_module().get_logger(logger_name)
        # Get standard logger for compatibility
        self.logger = self._tempo_logger.logger
        self.debug_mode = debug_mode

        # Set debug mode in global context if enabled
        if debug_mode:
            with self._tempo_logger.context(debug_mode=debug_mode):
                self._tempo_logger.info(f"{logger_name} debug mode enabled")
        else:
            self._tempo_logger.info(f"{logger_name} debug mode disabled")

    def log(self, message: str, level: str = "info", **kwargs):
        """
        Log a message if debug mode is enabled.

        Args:
            message: Message to log
            level: Log level (info, debug, warning, error, critical)
            **kwargs: Additional context key-value pairs
        """
        if not hasattr(self, "debug_mode") or not self.debug_mode:
            return

        # Validate level
        valid_levels = ["info", "debug", "warning", "error", "critical"]
        if level not in valid_levels:
            raise ValueError(
                f"Invalid log level: {level}. Must be one of {valid_levels}"
            )

        # Add debug mode to context
        kwargs["debug_mode"] = self.debug_mode

        # Log with the appropriate level
        if level == "info":
            self._tempo_logger.info(message, **kwargs)
        elif level == "debug":
            self._tempo_logger.debug(message, **kwargs)
        elif level == "warning":
            self._tempo_logger.warning(message, **kwargs)
        elif level == "error":
            self._tempo_logger.error(message, **kwargs)
        elif level == "critical":
            self._tempo_logger.critical(message, **kwargs)

    def set_debug_mode(self, enabled: bool = True):
        """
        Enable or disable debug mode.

        Args:
            enabled: Whether to enable debug mode
        """
        old_mode = getattr(self, "debug_mode", False)
        self.debug_mode = enabled

        # Log mode change if it's different
        if hasattr(self, "_tempo_logger") and old_mode != enabled:
            if enabled:
                self._tempo_logger.info(f"Debug mode enabled", debug_mode=enabled)
            else:
                self._tempo_logger.info(f"Debug mode disabled", debug_mode=enabled)

    # Enhanced logging methods for convenience

    def log_trace(self, message: str, **kwargs):
        """Log a trace message with stack trace if debug mode is enabled."""
        if (
            hasattr(self, "debug_mode")
            and self.debug_mode
            and hasattr(self, "_tempo_logger")
        ):
            self._tempo_logger.trace(message, **kwargs)

    def log_success(self, message: str, **kwargs):
        """Log a success message."""
        if hasattr(self, "_tempo_logger"):
            self._tempo_logger.success(message, **kwargs)

    def log_exception(self, message: str, exc_info=True, **kwargs):
        """Log an exception with traceback."""
        if hasattr(self, "_tempo_logger"):
            self._tempo_logger.exception(message, exc_info=exc_info, **kwargs)

    def with_context(self, **kwargs):
        """Create a context manager for adding context to logs."""
        if hasattr(self, "_tempo_logger"):
            return self._tempo_logger.context(**kwargs)
        else:
            # Fallback to a no-op context manager
            class NoOpContextManager:
                def __enter__(self):
                    pass

                def __exit__(self, *args):
                    pass

            return NoOpContextManager()

    def operation(self, operation_name: str, **kwargs):
        """Create a context manager for tracking operations."""
        if hasattr(self, "_tempo_logger"):
            return self._tempo_logger.operation(operation_name, **kwargs)
        else:
            # Fallback to a no-op context manager
            class NoOpContextManager:
                def __enter__(self):
                    pass

                def __exit__(self, *args):
                    pass

            return NoOpContextManager()

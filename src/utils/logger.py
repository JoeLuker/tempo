"""
Enhanced logging system for TEMPO project.

This module provides a comprehensive logging system that extends the basic
logging_utils.py functionality with structured logging, rotating file handlers,
context tracking, and more.
"""

import os
import sys
import json
import time
import logging
import traceback
import threading
import functools
import inspect
from typing import Dict, Any, Optional, Union, List, Callable, Tuple
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from queue import Queue
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

from src.utils.config_manager import config
from src.utils.logging_utils import LoggingMixin, setup_logger


# Thread local storage for context tracking
_thread_local = threading.local()

# Global context that applies to all threads
_global_context = {}

# Log record formatter that includes all context information
class ContextAwareFormatter(logging.Formatter):
    """
    Custom formatter that adds context information to log records.
    
    This formatter adds both thread-local and global context to log messages.
    """
    
    def format(self, record):
        """Format the log record with context information."""
        # Add context information to the record
        self._add_context_to_record(record)
        
        # Format the record using the parent formatter
        return super().format(record)
    
    def _add_context_to_record(self, record):
        """Add context information to the log record."""
        # Get the thread-local context
        thread_context = getattr(_thread_local, 'context', {})
        
        # Create context dict by combining thread-local and global context
        context = {**_global_context, **thread_context}
        
        # Add context info to the record's __dict__
        for key, value in context.items():
            # Don't override existing record attributes
            if not hasattr(record, key):
                setattr(record, key, value)
        
        # Add a context_str attribute for displaying formatted context
        if context:
            context_items = []
            for key, value in context.items():
                # Handle different value types
                if isinstance(value, (str, int, float, bool, type(None))):
                    context_items.append(f"{key}={value}")
                else:
                    try:
                        # For complex objects, try to use their string representation
                        context_items.append(f"{key}={str(value)}")
                    except Exception:
                        context_items.append(f"{key}=<unprintable>")
            
            setattr(record, 'context_str', ' '.join(context_items))
        else:
            setattr(record, 'context_str', '')


class RotatingFileHandlerWithHeader(RotatingFileHandler):
    """
    RotatingFileHandler that writes a header when a new log file is created.
    
    This is useful for adding timestamps, version information, and other
    metadata to the beginning of log files.
    """
    
    def __init__(self, filename, **kwargs):
        """Initialize the handler with a custom header function."""
        self.header_function = kwargs.pop('header_function', None)
        super().__init__(filename, **kwargs)
        
        # Write header if file is new or empty
        if self.header_function and os.path.getsize(filename) == 0:
            self._write_header()
    
    def doRollover(self):
        """Write a header to the new log file after rollover."""
        super().doRollover()
        
        if self.header_function:
            self._write_header()
    
    def _write_header(self):
        """Write the header to the log file."""
        header = self.header_function()
        if header:
            self.stream.write(header + "\n")
            self.stream.flush()


class TempoLogger:
    """
    Enhanced logger for TEMPO project.
    
    Features:
    - Structured logging with context
    - Multiple output formats (text, JSON)
    - Rotating file handlers
    - Performance metrics
    - Context managers for tracking operations
    """
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_file: Log file path (defaults to name.log in config.logging.log_dir)
        """
        self.name = name
        self.start_time = time.time()
        
        # Create the underlying logger
        self.logger = logging.getLogger(name)
        
        # Set the log level from config
        log_level = getattr(logging, config.logging.log_level.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        
        # Remove any existing handlers to avoid duplicate logs
        if self.logger.handlers:
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)
        
        # Add handlers
        self._setup_handlers(log_file)
        
        # Setup metrics
        self.metrics = {
            'log_count_by_level': {level: 0 for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']},
            'start_time': self.start_time,
        }
    
    def _setup_handlers(self, log_file: Optional[str]):
        """
        Set up log handlers.
        
        Args:
            log_file: Log file path
        """
        # Formatter for console
        console_formatter = ContextAwareFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s%(context_str)s"
        )
        
        # Formatter for file (more detailed)
        file_formatter = ContextAwareFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s%(context_str)s"
        )
        
        # Create console handler if enabled
        if config.logging.console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Create file handler if enabled
        if config.logging.enable_file_logging:
            # Set up log file path
            if log_file is None:
                log_file = f"{self.name.replace('.', '_')}.log"
            
            log_path = Path(config.logging.log_dir) / log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Header function for log files
            def header_function():
                return f"--- Log started at {datetime.now().isoformat()} ---\n" + \
                      f"--- TEMPO Logger: {self.name} ---\n" + \
                      f"--- App Version: {getattr(config, 'version', 'unknown')} ---"
            
            # Create rotating file handler
            file_handler = RotatingFileHandlerWithHeader(
                filename=str(log_path),
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                header_function=header_function
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def _log(self, level: int, msg: str, *args, exc_info=None, stack_info=False, 
             stacklevel=1, extra=None, **kwargs):
        """
        Log a message with the specified level and context.
        
        Args:
            level: Log level
            msg: Message to log
            *args: Arguments for string formatting
            exc_info: Exception info
            stack_info: Whether to include stack info
            stacklevel: Stack level for finding caller
            extra: Extra info for the log record
            **kwargs: Context key-value pairs to add
        """
        # Update metrics
        level_name = logging.getLevelName(level)
        self.metrics['log_count_by_level'][level_name] = self.metrics['log_count_by_level'].get(level_name, 0) + 1
        
        # Add kwargs to thread-local context temporarily
        with self.context(**kwargs):
            # Log the message
            self.logger.log(level, msg, *args, exc_info=exc_info, stack_info=stack_info, 
                           stacklevel=stacklevel + 1, extra=extra)
    
    # Standard logging methods
    def debug(self, msg, *args, **kwargs):
        """Log a debug message."""
        self._log(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        """Log an info message."""
        self._log(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        """Log a warning message."""
        self._log(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        """Log an error message."""
        self._log(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        """Log a critical message."""
        self._log(logging.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg, *args, **kwargs):
        """Log an exception with traceback."""
        kwargs['exc_info'] = kwargs.get('exc_info', True)
        self._log(logging.ERROR, msg, *args, **kwargs)
    
    # Enhanced logging methods
    
    def success(self, msg, *args, **kwargs):
        """Log a success message (info level with success marker)."""
        self._log(logging.INFO, f"✓ SUCCESS: {msg}", *args, **kwargs)
    
    def trace(self, msg, *args, **kwargs):
        """Log a trace message (debug level with call stack)."""
        kwargs['stack_info'] = True
        self._log(logging.DEBUG, f"TRACE: {msg}", *args, **kwargs)
    
    def metrics_report(self):
        """
        Generate a metrics report for this logger.
        
        Returns:
            dict: Dictionary of logger metrics
        """
        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        
        # Build report
        report = {
            'name': self.name,
            'elapsed_seconds': elapsed,
            'log_count': self.metrics['log_count_by_level'],
            'total_logs': sum(self.metrics['log_count_by_level'].values()),
            'logs_per_second': sum(self.metrics['log_count_by_level'].values()) / max(1.0, elapsed),
        }
        
        return report
    
    # Context management
    
    @contextmanager
    def context(self, **kwargs):
        """
        Context manager for adding temporary context to logs.
        
        Args:
            **kwargs: Context key-value pairs
        """
        # Initialize thread-local context if needed
        if not hasattr(_thread_local, 'context'):
            _thread_local.context = {}
        
        # Save old context
        old_context = _thread_local.context.copy()
        
        # Update context with new values
        _thread_local.context.update(kwargs)
        
        try:
            yield
        finally:
            # Restore old context
            _thread_local.context = old_context
    
    @contextmanager
    def operation(self, operation_name: str, log_result: bool = True, log_exception: bool = True, 
                  level: int = logging.INFO):
        """
        Context manager for tracking and logging operations.
        
        Args:
            operation_name: Name of the operation
            log_result: Whether to log the result
            log_exception: Whether to log exceptions
            level: Log level for result logging
        """
        # Log operation start
        self._log(level, f"Starting operation: {operation_name}")
        start_time = time.time()
        
        try:
            # Run the operation
            yield
            
            # Log operation success
            elapsed = time.time() - start_time
            if log_result:
                self._log(level, f"Completed operation: {operation_name} in {elapsed:.3f}s", 
                        operation=operation_name, duration=elapsed, status="success")
        
        except Exception as e:
            # Log operation failure
            elapsed = time.time() - start_time
            if log_exception:
                self._log(logging.ERROR, f"Failed operation: {operation_name} after {elapsed:.3f}s - {str(e)}",
                        operation=operation_name, duration=elapsed, status="failed", 
                        error_type=type(e).__name__, error=str(e), exc_info=True)
            raise
    
    def function_logger(self, func=None, *, entry_level=logging.DEBUG, exit_level=logging.DEBUG, 
                       exception_level=logging.ERROR, log_args=False, log_result=False):
        """
        Decorator for logging function entry, exit, and exceptions.
        
        Args:
            func: Function to decorate
            entry_level: Log level for function entry
            exit_level: Log level for function exit
            exception_level: Log level for exceptions
            log_args: Whether to log function arguments
            log_result: Whether to log function result
            
        Returns:
            Decorated function
        """
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                # Prepare function info
                fn_name = fn.__name__
                module_name = fn.__module__
                
                # Log entry
                entry_msg = f"Entering {module_name}.{fn_name}"
                if log_args:
                    # Format arguments for logging (carefully to avoid huge outputs)
                    formatted_args = []
                    for i, arg in enumerate(args):
                        # Try to get argument name from function signature
                        params = list(inspect.signature(fn).parameters.keys())
                        if i < len(params):
                            arg_name = params[i]
                            formatted_args.append(f"{arg_name}={_format_value(arg)}")
                        else:
                            formatted_args.append(f"arg{i}={_format_value(arg)}")
                    
                    # Add keyword arguments
                    for k, v in kwargs.items():
                        formatted_args.append(f"{k}={_format_value(v)}")
                    
                    entry_msg += f" with args: {', '.join(formatted_args)}"
                
                self._log(entry_level, entry_msg, function=fn_name, module=module_name)
                
                # Call the function
                start_time = time.time()
                try:
                    result = fn(*args, **kwargs)
                    
                    # Log exit
                    elapsed = time.time() - start_time
                    exit_msg = f"Exiting {module_name}.{fn_name} after {elapsed:.3f}s"
                    if log_result:
                        exit_msg += f" with result: {_format_value(result)}"
                    
                    self._log(exit_level, exit_msg, function=fn_name, module=module_name, 
                            duration=elapsed)
                    
                    return result
                
                except Exception as e:
                    # Log exception
                    elapsed = time.time() - start_time
                    self._log(
                        exception_level,
                        f"Exception in {module_name}.{fn_name} after {elapsed:.3f}s: {type(e).__name__}: {str(e)}",
                        function=fn_name, module=module_name, duration=elapsed,
                        error_type=type(e).__name__, error=str(e), exc_info=True
                    )
                    raise
            
            return wrapper
        
        # Handle both @function_logger and @function_logger() syntax
        if func is None:
            return decorator
        return decorator(func)


# Utility functions

def _format_value(value, max_length=100):
    """
    Format a value for logging, truncating if too long.
    
    Args:
        value: Value to format
        max_length: Maximum string length
        
    Returns:
        str: Formatted value
    """
    # Handle common types
    if value is None:
        return 'None'
    elif isinstance(value, (int, float, bool)):
        return str(value)
    elif isinstance(value, str):
        if len(value) > max_length:
            return f'"{value[:max_length]}..."'
        return f'"{value}"'
    
    # For other types, use the str representation and truncate if needed
    try:
        result = str(value)
        if len(result) > max_length:
            return f"{result[:max_length]}..."
        return result
    except Exception:
        return f"<unprintable {type(value).__name__}>"


# Global context management

def add_global_context(**kwargs):
    """
    Add context that will be included in all log records.
    
    Args:
        **kwargs: Context key-value pairs
    """
    _global_context.update(kwargs)


def clear_global_context(keys=None):
    """
    Clear global context.
    
    Args:
        keys: Specific keys to clear, or None to clear all
    """
    global _global_context
    
    if keys is None:
        _global_context = {}
    else:
        for key in keys:
            if key in _global_context:
                del _global_context[key]


# Module-level functions

_loggers: Dict[str, TempoLogger] = {}

def get_logger(name: str) -> TempoLogger:
    """
    Get or create a logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        TempoLogger: Logger instance
    """
    if name not in _loggers:
        _loggers[name] = TempoLogger(name)
    
    return _loggers[name]


# Initialize with application metadata
add_global_context(
    app_version=getattr(config, 'version', 'unknown'),
    app_env=os.environ.get('TEMPO_ENV', 'development')
)
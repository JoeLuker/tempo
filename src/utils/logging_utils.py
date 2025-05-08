"""
Logging utilities for TEMPO project.

This module provides common logging functionality to standardize logging across all modules.
"""

import os
import logging
from typing import Optional

from src.utils.config import get_debug_mode, LOG_DIR, ENABLE_FILE_LOGGING


def setup_logger(logger_name: str, log_file_name: Optional[str] = None, debug_mode: Optional[bool] = None) -> logging.Logger:
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
    
    # Full path to log file
    log_file = os.path.join(LOG_DIR, log_file_name)
    
    # Configure logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers to avoid duplicate logs
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # Add file handler if enabled
    if ENABLE_FILE_LOGGING:
        # Clear the log file by opening in write mode first
        with open(log_file, "w") as f:
            pass
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
    
    # Log initialization
    logger.info(f"{logger_name} initialized with debug_mode={debug_mode}")
    
    return logger


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
    
    def setup_logging(self, logger_name: str, log_file_name: Optional[str] = None, debug_mode: Optional[bool] = None):
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
            
        self.logger = setup_logger(logger_name, log_file_name, debug_mode)
        self.debug_mode = debug_mode
        
        if debug_mode:
            print(f"{logger_name} debug mode enabled - logging to file at logs/{log_file_name or f'{logger_name}_debug.log'}")
        else:
            print(f"{logger_name} debug mode disabled")
    
    def log(self, message: str, level: str = "info"):
        """
        Log a message if debug mode is enabled.
        
        Args:
            message: Message to log
            level: Log level (info, debug, warning, error)
        """
        if not hasattr(self, 'debug_mode') or not self.debug_mode:
            return
            
        assert level in ["info", "debug", "warning", "error"], f"Invalid log level: {level}"
        
        if level == "info":
            self.logger.info(message)
        elif level == "debug":
            self.logger.debug(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
    
    def set_debug_mode(self, enabled: bool = True):
        """
        Enable or disable debug mode.
        
        Args:
            enabled: Whether to enable debug mode
        """
        self.debug_mode = enabled
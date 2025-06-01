"""
Rate limiting for TEMPO API.

This module provides rate limiting functionality for the API,
protecting against excessive usage and ensuring fair resource allocation.
"""

from fastapi import FastAPI
from src.utils.rate_limiter import RateLimitMiddleware


def add_rate_limiting(app: FastAPI, enabled: bool = True) -> None:
    """
    Add rate limiting to a FastAPI application.

    Args:
        app: FastAPI application
        enabled: Whether rate limiting is enabled
    """
    app.add_middleware(RateLimitMiddleware, enabled=enabled)

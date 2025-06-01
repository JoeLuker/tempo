"""
Rate limiting utilities for TEMPO API.

This module provides rate limiting functionality for the TEMPO API to protect
against excessive usage and ensure fair resource allocation.
"""

import time
from typing import Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.types import ASGIApp

from src.utils.api_errors import RateLimitError


class TokenBucket:
    """
    Token bucket algorithm implementation for rate limiting.

    The token bucket algorithm works by managing a bucket of tokens that refill at a
    constant rate. Each request consumes tokens, and if there are not enough tokens
    available, the request is rate limited.
    """

    def __init__(self, capacity: int, refill_rate: float, refill_time: int = 60):
        """
        Initialize a token bucket.

        Args:
            capacity: Maximum number of tokens in the bucket
            refill_rate: Number of tokens to add per refill period
            refill_time: Number of seconds between refills
        """
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.refill_time = refill_time
        self.last_refill = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            bool: True if tokens were consumed, False if not enough tokens
        """
        # Refill tokens based on elapsed time
        self._refill()

        # Check if enough tokens are available
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        return False

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        if elapsed > 0:
            # Calculate tokens to add based on elapsed time
            new_tokens = (elapsed / self.refill_time) * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now


class RateLimiter:
    """
    Rate limiter implementation for the API.

    This class provides a way to track and limit request rates based on client identifiers
    (like IP addresses or API keys) using the token bucket algorithm.
    """

    def __init__(self):
        """Initialize the rate limiter."""
        # Map client identifiers to token buckets
        self.buckets: Dict[str, TokenBucket] = {}

        # Default limits
        self.default_capacity = 60  # 60 requests
        self.default_refill_rate = 60  # 60 tokens per minute
        self.default_refill_time = 60  # 1 minute

        # Endpoint-specific limits
        self.endpoint_limits: Dict[str, Tuple[int, float, int]] = {
            # path: (capacity, refill_rate, refill_time)
            "/api/v2/generate": (10, 10, 60),  # 10 requests per minute for generation
        }

    def check_rate_limit(
        self, client_id: str, endpoint: str, tokens: int = 1
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if a client has exceeded the rate limit for an endpoint.

        Args:
            client_id: Client identifier (IP, API key, etc.)
            endpoint: API endpoint being accessed
            tokens: Number of tokens to consume (based on request complexity)

        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        # Create a bucket key that combines client and endpoint
        bucket_key = f"{client_id}:{endpoint}"

        # Get or create the token bucket for this client and endpoint
        if bucket_key not in self.buckets:
            # Get endpoint-specific limits or use defaults
            if endpoint in self.endpoint_limits:
                capacity, refill_rate, refill_time = self.endpoint_limits[endpoint]
            else:
                capacity = self.default_capacity
                refill_rate = self.default_refill_rate
                refill_time = self.default_refill_time

            self.buckets[bucket_key] = TokenBucket(capacity, refill_rate, refill_time)

        bucket = self.buckets[bucket_key]

        # Try to consume tokens
        is_allowed = bucket.consume(tokens)

        # Calculate rate limit information
        remaining = int(bucket.tokens)
        reset_time = bucket.last_refill + bucket.refill_time
        limit = bucket.capacity

        rate_limit_info = {
            "limit": limit,
            "remaining": remaining,
            "reset": reset_time,
            "retry_after": 0 if is_allowed else max(1, int(reset_time - time.time())),
        }

        return is_allowed, rate_limit_info


# Global rate limiter instance
_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    """
    Get the global rate limiter instance.

    Returns:
        RateLimiter: The global rate limiter instance
    """
    return _rate_limiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.

    This middleware applies rate limiting to all API requests, using client IP address
    or API key as the identifier.
    """

    def __init__(
        self,
        app: ASGIApp,
        client_identifier: Callable[[Request], str] = None,
        enabled: bool = True,
    ):
        """
        Initialize the rate limit middleware.

        Args:
            app: FastAPI application
            client_identifier: Function to extract client identifier from request
            enabled: Whether rate limiting is enabled
        """
        super().__init__(app)
        self.limiter = get_rate_limiter()
        self._client_identifier = client_identifier or self._default_identifier
        self.enabled = enabled

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process the request with rate limiting.

        Args:
            request: The incoming request
            call_next: The next middleware or route handler

        Returns:
            Response: The response from the API

        Raises:
            RateLimitError: If the client has exceeded the rate limit
        """
        # Skip rate limiting if disabled or for certain paths
        if not self.enabled or not self._should_rate_limit(request):
            return await call_next(request)

        # Get client identifier and endpoint
        client_id = self._client_identifier(request)
        endpoint = request.url.path

        # Determine tokens to consume based on endpoint and method
        tokens = self._get_token_cost(request)

        # Check rate limit
        is_allowed, rate_limit_info = self.limiter.check_rate_limit(
            client_id, endpoint, tokens
        )

        # Handle rate limiting
        if not is_allowed:
            error = RateLimitError(
                message="Rate limit exceeded. Please slow down your requests.",
                details=[
                    {
                        "message": f"Retry after {rate_limit_info['retry_after']} seconds",
                        "code": "rate_limit_exceeded",
                    }
                ],
            )
            return error.to_response(request)

        # Process the request
        response = await call_next(request)

        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(rate_limit_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_limit_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(int(rate_limit_info["reset"]))

        return response

    def _should_rate_limit(self, request: Request) -> bool:
        """
        Determine if rate limiting should be applied to this request.

        Args:
            request: The incoming request

        Returns:
            bool: Whether to apply rate limiting
        """
        # Skip rate limiting for documentation, health checks, etc.
        excluded_paths = [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/api/v2/health",
        ]
        return request.url.path not in excluded_paths and request.url.path.startswith(
            "/api/"
        )

    def _get_token_cost(self, request: Request) -> int:
        """
        Determine the token cost for this request.

        Args:
            request: The incoming request

        Returns:
            int: Number of tokens to consume
        """
        # Higher cost for complex operations like generation
        if request.url.path.endswith("/generate") and request.method == "POST":
            return 5

        # Default cost
        return 1

    def _default_identifier(self, request: Request) -> str:
        """
        Extract a default client identifier from the request.

        Args:
            request: The incoming request

        Returns:
            str: Client identifier (IP address by default)
        """
        # Try to get client IP from various headers
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        return request.client.host if request.client else "unknown"


def add_rate_limiting(app: FastAPI, enabled: bool = True):
    """
    Add rate limiting to a FastAPI application.

    Args:
        app: FastAPI application
        enabled: Whether rate limiting is enabled
    """
    app.add_middleware(RateLimitMiddleware, enabled=enabled)

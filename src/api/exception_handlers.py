"""
Exception handlers for TEMPO API.

This module provides exception handlers for the TEMPO API, ensuring standardized
error responses across all endpoints.
"""

from fastapi import FastAPI, Request, status
from fastapi.exceptions import HTTPException, RequestValidationError
from pydantic import ValidationError

from src.utils.api_errors import (
    APIError,
    ErrorType,
    ErrorResponse,
    ValidationError as APIValidationError,
)


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register exception handlers for the FastAPI app.

    Args:
        app: FastAPI application instance
    """

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        """Handle custom API errors."""
        # Get request ID if available
        request_id = getattr(request.state, "request_id", None)
        return exc.to_response(request, request_id)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Convert FastAPI HTTPExceptions to standardized responses."""
        # Map status code to error type
        error_type = ErrorType.SERVER_ERROR
        if exc.status_code == status.HTTP_404_NOT_FOUND:
            error_type = ErrorType.RESOURCE_NOT_FOUND
        elif exc.status_code == status.HTTP_401_UNAUTHORIZED:
            error_type = ErrorType.AUTHENTICATION_ERROR
        elif exc.status_code == status.HTTP_403_FORBIDDEN:
            error_type = ErrorType.FORBIDDEN_ERROR
        elif exc.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY:
            error_type = ErrorType.VALIDATION_ERROR
        elif exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            error_type = ErrorType.RATE_LIMIT_ERROR
        elif 400 <= exc.status_code < 500:
            error_type = ErrorType.REQUEST_ERROR

        # Create API error and return response
        api_error = APIError(
            message=exc.detail, status_code=exc.status_code, error_type=error_type
        )
        request_id = getattr(request.state, "request_id", None)
        return api_error.to_response(request, request_id)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        """Handle Pydantic validation errors from request parsing."""
        # Extract field information from validation errors
        details = []
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error.get("loc", []))
            msg = error.get("msg", "")
            err_type = error.get("type", "")

            details.append({"field": field, "message": msg, "code": err_type})

        # Create API validation error
        api_error = APIValidationError(
            message="Request validation failed", details=details
        )

        request_id = getattr(request.state, "request_id", None)
        return api_error.to_response(request, request_id)

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all uncaught exceptions."""
        import traceback
        import logging

        # Log the error with traceback for debugging
        logger = logging.getLogger("tempo-api")
        logger.error(f"Uncaught exception: {str(exc)}")
        logger.error(traceback.format_exc())

        # Create a server error
        api_error = APIError(
            message=f"An unexpected error occurred: {str(exc)}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_type=ErrorType.SERVER_ERROR,
        )

        request_id = getattr(request.state, "request_id", None)
        return api_error.to_response(request, request_id)

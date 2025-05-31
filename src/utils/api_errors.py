"""
Error handling utilities for TEMPO API.

This module provides standardized error handling for the TEMPO API, including
custom exception classes, error formatting, and response generation.
"""

from typing import Optional, Dict, Any, List, Type
from enum import Enum
from datetime import datetime
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from pydantic import BaseModel, Field


class ErrorType(str, Enum):
    """Standardized error types for API responses."""
    VALIDATION_ERROR = "validation_error"
    RESOURCE_NOT_FOUND = "resource_not_found"
    AUTHORIZATION_ERROR = "authorization_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    SERVER_ERROR = "server_error"
    MODEL_ERROR = "model_error"
    GENERATION_ERROR = "generation_error"
    REQUEST_ERROR = "request_error"
    FORBIDDEN_ERROR = "forbidden_error"
    TIMEOUT_ERROR = "timeout_error"
    DEPENDENCY_ERROR = "dependency_error"


class ErrorDetail(BaseModel):
    """Detailed information about a specific error."""
    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code") 


class ErrorResponse(BaseModel):
    """Standardized error response model."""
    status_code: int = Field(..., description="HTTP status code")
    error_type: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: str = Field(..., description="Error timestamp")
    path: Optional[str] = Field(None, description="Request path")
    details: Optional[List[ErrorDetail]] = Field(None, description="Detailed error information")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    
    class Config:
        schema_extra = {
            "example": {
                "status_code": 400,
                "error_type": "validation_error",
                "message": "Invalid request parameters",
                "timestamp": "2023-05-01T12:34:56.789Z",
                "path": "/api/v2/generate",
                "details": [
                    {
                        "field": "max_tokens",
                        "message": "Value must be greater than 0",
                        "code": "value_error"
                    }
                ],
                "request_id": "abcd1234-ef56-7890"
            }
        }


class APIError(Exception):
    """
    Base API error class with standardized formatting.
    
    This exception class allows for consistent error handling across the API.
    """
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type: ErrorType = ErrorType.SERVER_ERROR
    message: str = "An unexpected error occurred"
    
    def __init__(
        self, 
        message: Optional[str] = None,
        details: Optional[List[Dict[str, Any]]] = None,
        status_code: Optional[int] = None,
        error_type: Optional[ErrorType] = None,
    ):
        """
        Initialize an API error with custom parameters.
        
        Args:
            message: Custom error message
            details: Detailed error information
            status_code: HTTP status code
            error_type: Type of error
        """
        self.message = message or self.message
        self.details = details or []
        self.status_code = status_code or self.status_code
        self.error_type = error_type or self.error_type
        super().__init__(self.message)
    
    def to_response(self, request: Request, request_id: Optional[str] = None) -> JSONResponse:
        """
        Convert error to a standardized JSONResponse.
        
        Args:
            request: FastAPI request
            request_id: Optional request ID for tracking
            
        Returns:
            JSONResponse: Standardized error response
        """
        # Format details objects if they exist
        error_details = None
        if self.details:
            error_details = [
                ErrorDetail(
                    field=detail.get("field"),
                    message=detail.get("message", ""),
                    code=detail.get("code")
                )
                for detail in self.details
            ]
        
        # Create error response
        error_response = ErrorResponse(
            status_code=self.status_code,
            error_type=self.error_type,
            message=self.message,
            timestamp=datetime.now().isoformat(),
            path=request.url.path,
            details=error_details,
            request_id=request_id
        )
        
        # Return as JSON response
        return JSONResponse(
            status_code=self.status_code,
            content=error_response.dict(exclude_none=True)
        )


# Specific error classes 

class ValidationError(APIError):
    """Error for request validation failures."""
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_type = ErrorType.VALIDATION_ERROR
    message = "Validation error in request parameters"


class ResourceNotFoundError(APIError):
    """Error for resource not found."""
    status_code = status.HTTP_404_NOT_FOUND
    error_type = ErrorType.RESOURCE_NOT_FOUND
    message = "The requested resource was not found"


class AuthorizationError(APIError):
    """Error for authorization failures."""
    status_code = status.HTTP_403_FORBIDDEN
    error_type = ErrorType.AUTHORIZATION_ERROR
    message = "Not authorized to access this resource"


class AuthenticationError(APIError):
    """Error for authentication failures."""
    status_code = status.HTTP_401_UNAUTHORIZED
    error_type = ErrorType.AUTHENTICATION_ERROR
    message = "Authentication required"


class RateLimitError(APIError):
    """Error for rate limit exceeded."""
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    error_type = ErrorType.RATE_LIMIT_ERROR
    message = "Rate limit exceeded"


class ModelError(APIError):
    """Error for model-related failures."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type = ErrorType.MODEL_ERROR
    message = "Error in model processing"


class GenerationError(APIError):
    """Error for generation failures."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type = ErrorType.GENERATION_ERROR
    message = "Error during text generation"


class RequestError(APIError):
    """Error for invalid requests that pass validation."""
    status_code = status.HTTP_400_BAD_REQUEST
    error_type = ErrorType.REQUEST_ERROR
    message = "Invalid request"


class ModelNotAvailableError(APIError):
    """Error for when model is not available."""
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_type = ErrorType.DEPENDENCY_ERROR
    message = "Model is not available"


class TimeoutError(APIError):
    """Error for request timeouts."""
    status_code = status.HTTP_504_GATEWAY_TIMEOUT
    error_type = ErrorType.TIMEOUT_ERROR
    message = "Request timed out"


# Exception handlers for FastAPI

def register_exception_handlers(app):
    """
    Register exception handlers for the FastAPI app.
    
    Args:
        app: FastAPI application
    """
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        """Handle all API errors."""
        return exc.to_response(request)
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Convert HTTPExceptions to standardized responses."""
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
            message=exc.detail,
            status_code=exc.status_code,
            error_type=error_type
        )
        return api_error.to_response(request)
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all uncaught exceptions."""
        import traceback
        
        # Log the error with traceback for debugging
        print(f"Uncaught exception: {str(exc)}")
        traceback.print_exc()
        
        # Create a server error
        api_error = APIError(
            message=f"An unexpected error occurred: {str(exc)}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_type=ErrorType.SERVER_ERROR
        )
        return api_error.to_response(request)
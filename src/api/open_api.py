"""
OpenAPI schema customization for TEMPO API.

This module provides functions for customizing the OpenAPI schema,
including standardizing error responses and adding examples.
"""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

def custom_openapi_schema(app: FastAPI) -> dict:
    """
    Generate a custom OpenAPI schema for the API.
    
    Args:
        app: FastAPI application
        
    Returns:
        dict: The customized OpenAPI schema
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="TEMPO API",
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom schema components
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    if "schemas" not in openapi_schema["components"]:
        openapi_schema["components"]["schemas"] = {}
        
    # Add ErrorResponse schema
    openapi_schema["components"]["schemas"]["ErrorResponse"] = {
        "type": "object",
        "properties": {
            "status_code": {"type": "integer", "description": "HTTP status code"},
            "error_type": {"type": "string", "description": "Error type"},
            "message": {"type": "string", "description": "Error message"},
            "timestamp": {"type": "string", "description": "Error timestamp"},
            "path": {"type": "string", "description": "Request path"},
            "details": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string", "description": "Field that caused the error"},
                        "message": {"type": "string", "description": "Error message"},
                        "code": {"type": "string", "description": "Error code"}
                    },
                    "required": ["message"]
                },
                "description": "Detailed error information"
            },
            "request_id": {"type": "string", "description": "Request ID for tracking"}
        },
        "required": ["status_code", "error_type", "message", "timestamp"]
    }
    
    # Update all endpoints to include error responses
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if method.lower() in ["get", "post", "put", "delete", "patch"]:
                responses = openapi_schema["paths"][path][method]["responses"]
                
                # Add common error responses
                if "400" not in responses:
                    responses["400"] = {
                        "description": "Bad Request",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}
                    }
                if "422" not in responses:
                    responses["422"] = {
                        "description": "Validation Error",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}
                    }
                if "500" not in responses:
                    responses["500"] = {
                        "description": "Internal Server Error",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}
                    }
    
    return openapi_schema
"""
TEMPO API middleware.

This module contains middleware components for the TEMPO API, such as
request ID generation, logging, and other request processing functions.
"""

import uuid
from fastapi import Request, Response
from starlette.middleware.base import RequestResponseEndpoint


async def request_id_middleware(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    """
    Add a unique request ID to each request for tracking.

    Args:
        request: The incoming request
        call_next: The next middleware or route handler

    Returns:
        Response: The processed response with added request ID header
    """
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Process the request
    response = await call_next(request)

    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    return response

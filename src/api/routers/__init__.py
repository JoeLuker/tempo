"""
TEMPO API routers package.

This package contains all the API routers for different API versions and endpoints.
"""

from src.api.routers.v2 import router as v2_router
from src.api.routers.v1 import router as v1_router
from src.api.routers.common import router as common_router
from src.api.routers.docs import router as docs_router

__all__ = ["v2_router", "v1_router", "common_router", "docs_router"]
"""
TEMPO API application.

This module defines the main FastAPI application and includes all routers.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

from src.utils import config
from src.api.middleware import request_id_middleware
from src.api.exception_handlers import register_exception_handlers
from src.api.rate_limiting import add_rate_limiting
from src.api.routers import common_router, v2_router, v1_router, docs_router
from src.api.open_api import custom_openapi_schema

# Create the FastAPI application
app = FastAPI(
    title="TEMPO API",
    description="""
    TEMPO (Threshold-Enabled Multipath Parallel Output) is an experimental approach to text generation
    that processes multiple token possibilities simultaneously.
    
    This API provides endpoints for generating text with TEMPO, configuring model parameters,
    and monitoring system health.
    """,
    version=config.api.api_version,
    docs_url=None,  # Customize docs URL below
    redoc_url=None,  # Customize redoc URL below
    openapi_url=f"/api/{config.api.api_version}/openapi.json",
)

# Configure CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request ID middleware
app.middleware("http")(request_id_middleware)

# Register exception handlers for standardized error responses
register_exception_handlers(app)

# Add rate limiting if enabled
add_rate_limiting(app, enabled=not config.debug.global_debug)

# Custom OpenAPI schema
app.openapi = lambda: custom_openapi_schema(app)


# Documentation endpoints
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI endpoint."""
    return get_swagger_ui_html(
        openapi_url=f"/api/{config.api.api_version}/openapi.json",
        title="TEMPO API",
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5.10.0/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5.10.0/swagger-ui.css",
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """Custom ReDoc endpoint."""
    return get_redoc_html(
        openapi_url=f"/api/{config.api.api_version}/openapi.json",
        title="TEMPO API",
        redoc_js_url="https://unpkg.com/redoc@next/bundles/redoc.standalone.js",
    )


# Include all API routers
app.include_router(v2_router)
app.include_router(v1_router)
app.include_router(common_router)
app.include_router(docs_router)


def create_app() -> FastAPI:
    """
    Create and configure a new FastAPI application.

    This function exists primarily for testing, allowing the creation of
    isolated app instances. The global 'app' object should be used for
    normal operation.

    Returns:
        FastAPI: Configured FastAPI application
    """
    return app


# Run with: uvicorn src.api.app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn

    # Get port from environment or config
    port = int(os.environ.get("PORT", config.api.port))

    # Configure logging for uvicorn
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "use_colors": True,
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO"},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }

    # Run server
    uvicorn.run(
        "src.api.app:app", host=config.api.host, port=port, log_config=log_config
    )

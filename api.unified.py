#!/usr/bin/env python3
"""
TEMPO API Entry Point.

This is the main entry point for the TEMPO API server. It imports and runs
the FastAPI application from the src.api package.
"""

import os
import logging
import uvicorn
from src.utils import config

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("logs/api.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger("tempo-api")

    # Log startup
    logger.info("Starting TEMPO API server...")

    # Get host and port from config or environment
    host = os.environ.get("HOST", config.api.host)
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

    # Log config settings
    logger.info(f"API Version: {config.api.api_version}")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Debug Mode: {config.debug.global_debug}")
    logger.info(f"Default Model: {config.model.model_id}")

    # Run server
    logger.info("Starting uvicorn server...")

    # Import app after logging setup
    from src.api import app

    uvicorn.run("src.api:app", host=host, port=port, log_config=log_config)

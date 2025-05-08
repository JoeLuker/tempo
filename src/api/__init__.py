"""
TEMPO API package.

This package provides a FastAPI-based REST API for the TEMPO text generation system.
It consolidates the API functionality into a well-organized, modular structure.
"""

from src.api.app import create_app, app

__all__ = ["create_app", "app"]
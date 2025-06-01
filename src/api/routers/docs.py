"""
Documentation router for TEMPO API.

This module contains routes for API documentation and guides.
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, List, Any

from src.utils.api_docs import APIDocumentation

# Create router with documentation tag
router = APIRouter(prefix="/docs", tags=["Documentation"])


@router.get(
    "/sections",
    summary="List Documentation Sections",
    description="Returns a list of available documentation sections.",
    response_description="List of documentation sections.",
    status_code=status.HTTP_200_OK,
)
async def list_doc_sections():
    """List available documentation sections."""
    return {"sections": APIDocumentation.get_section_list()}


@router.get(
    "/section/{section_name}",
    summary="Get Documentation Section",
    description="Returns documentation content for a specific section.",
    response_description="Documentation content.",
    status_code=status.HTTP_200_OK,
)
async def get_doc_section(section_name: str):
    """
    Get documentation for a specific section.

    Args:
        section_name: Name of the documentation section

    Returns:
        dict: Documentation section content

    Raises:
        HTTPException: If section not found
    """
    section = APIDocumentation.get_section(section_name)
    if section["title"] == "Documentation Not Found":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Documentation section '{section_name}' not found",
        )
    return section

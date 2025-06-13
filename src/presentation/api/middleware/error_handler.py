"""Error handling middleware for the API."""

import logging
import traceback
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import torch

logger = logging.getLogger(__name__)


async def error_handler_middleware(request: Request, call_next):
    """Middleware to handle errors and convert them to appropriate HTTP responses."""
    try:
        response = await call_next(request)
        return response
    except HTTPException:
        # Re-raise HTTPExceptions as they already have proper status codes
        raise
    except ValueError as e:
        logger.error(f"Validation Error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=400,
            content={"detail": f"Invalid request: {str(e)}"}
        )
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA out of memory: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "GPU memory exceeded. Try reducing max_tokens or batch size."}
        )
    except RuntimeError as e:
        logger.error(f"Runtime Error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"Processing error: {str(e)}"}
        )
    except Exception as e:
        logger.error(f"Unexpected Error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(e)}"}
        )


def handle_generation_errors(func):
    """Decorator for handling errors in generation endpoints."""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValueError as e:
            logger.error(f"Validation Error during generation: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid generation parameters: {str(e)}"
            )
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory: {e}")
            raise HTTPException(
                status_code=500,
                detail="GPU memory exceeded. Try reducing max_tokens or batch size."
            )
        except RuntimeError as e:
            logger.error(f"Runtime Error during generation: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, 
                detail=f"Generation failed: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected Error during generation: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, 
                detail=f"An unexpected error occurred: {str(e)}"
            )
    
    return wrapper
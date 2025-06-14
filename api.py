"""Monadic API for TEMPO generation using functional programming patterns.

This module implements the TEMPO API using monadic design for better error handling,
composition, and dependency injection.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import traceback

from src.domain.monads import Result, Ok, Err, IO, IOResult, Reader
from src.domain.monads.composition import sequence_results
from src.application.services.monadic_generation_service import (
    MonadicGenerationService, 
    validate_request,
    log_result
)
from src.presentation.api.models.requests import GenerationRequest
from src.presentation.api.models.responses import GenerationResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tempo-monadic-api")


# Create FastAPI app
app = FastAPI(
    title="TEMPO Monadic API",
    description="Monadic API for TEMPO text generation using functional programming patterns",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Service singleton
class ServiceSingleton:
    """Singleton for the monadic generation service."""
    service: Optional[MonadicGenerationService] = None
    
    @classmethod
    def get_service(cls) -> Result[MonadicGenerationService, str]:
        """Get or create the generation service."""
        if cls.service is None:
            try:
                cls.service = MonadicGenerationService()
                logger.info("Initialized MonadicGenerationService")
                return Ok(cls.service)
            except Exception as e:
                error_msg = f"Failed to initialize service: {str(e)}"
                logger.error(error_msg)
                return Err(error_msg)
        return Ok(cls.service)


# Dependency injection
def get_generation_service() -> MonadicGenerationService:
    """Dependency injection for generation service."""
    result = ServiceSingleton.get_service()
    if result.is_err():
        raise HTTPException(status_code=500, detail=result.unwrap_err())
    return result.unwrap()


# API Models
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    service: str


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    details: Optional[Dict[str, Any]] = None
    traceback: Optional[str] = None


# Monadic endpoint handlers

def handle_generation_result(
    result: Result[GenerationResponse, str]
) -> GenerationResponse:
    """Convert a Result to an API response or raise an exception."""
    return result.fold(
        lambda err: _raise_http_error(400, err),
        lambda resp: resp
    )


def _raise_http_error(status_code: int, message: str):
    """Raise an HTTP exception with the given status code and message."""
    logger.error(f"API Error {status_code}: {message}")
    raise HTTPException(status_code=status_code, detail=message)


# API Endpoints

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint returning API information."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        service="TEMPO Monadic API"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # Check if service can be initialized
    service_result = ServiceSingleton.get_service()
    
    if service_result.is_err():
        raise HTTPException(
            status_code=503, 
            detail=f"Service unavailable: {service_result.unwrap_err()}"
        )
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        service="TEMPO Monadic API"
    )


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    service: MonadicGenerationService = Depends(get_generation_service)
) -> GenerationResponse:
    """Generate text using monadic TEMPO generation.
    
    This endpoint uses monadic composition for:
    - Request validation
    - Error handling
    - Generation pipeline
    """
    logger.info(f"Received generation request: {request.prompt[:50]}...")
    
    # Build monadic pipeline
    generation_pipeline = (
        validate_request(request)
        .flat_map(lambda req: service.generate_text(req))
    )
    
    # Execute pipeline and handle result
    return handle_generation_result(generation_pipeline)


@app.post("/generate/batch", response_model=List[GenerationResponse])
async def generate_batch(
    requests: List[GenerationRequest],
    service: MonadicGenerationService = Depends(get_generation_service)
) -> List[GenerationResponse]:
    """Generate text for multiple prompts in batch.
    
    Uses monadic sequencing to handle multiple generations.
    """
    logger.info(f"Received batch generation request with {len(requests)} prompts")
    
    # Validate all requests
    validation_results = [validate_request(req) for req in requests]
    validated_requests = sequence_results(validation_results)
    
    if validated_requests.is_err():
        raise HTTPException(
            status_code=400, 
            detail=f"Validation failed: {validated_requests.unwrap_err()}"
        )
    
    # Generate for each request
    generation_results = []
    for request in validated_requests.unwrap():
        result = service.generate_text(request)
        if result.is_err():
            # For batch, we could either fail fast or collect errors
            # Here we fail fast
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed: {result.unwrap_err()}"
            )
        generation_results.append(result.unwrap())
    
    return generation_results


@app.post("/generate/stream")
async def generate_stream(
    request: GenerationRequest,
    service: MonadicGenerationService = Depends(get_generation_service)
):
    """Stream generation results using Server-Sent Events.
    
    This demonstrates how to use monadic patterns with async streaming.
    """
    from fastapi.responses import StreamingResponse
    import asyncio
    import json
    
    async def event_generator():
        """Generate SSE events from the generation process."""
        try:
            # Validate request
            validation_result = validate_request(request)
            if validation_result.is_err():
                yield f"data: {json.dumps({'error': validation_result.unwrap_err()})}\n\n"
                return
            
            # For now, we'll do regular generation and stream the result
            # In a full implementation, this would stream tokens as they're generated
            result = service.generate_text(validation_result.unwrap())
            
            if result.is_ok():
                response = result.unwrap()
                # Stream the response in chunks
                yield f"data: {json.dumps({'status': 'started'})}\n\n"
                await asyncio.sleep(0.1)
                
                # Stream tokens (simplified - in reality would stream as generated)
                tokens = response.generated_text.split()
                for i, token in enumerate(tokens):
                    yield f"data: {json.dumps({'token': token, 'index': i})}\n\n"
                    await asyncio.sleep(0.05)  # Simulate streaming delay
                
                yield f"data: {json.dumps({'status': 'completed', 'full_response': response.dict()})}\n\n"
            else:
                yield f"data: {json.dumps({'error': result.unwrap_err()})}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


# Error handling middleware

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle uncaught exceptions using monadic error types."""
    error_message = str(exc)
    logger.error(f"Unhandled exception: {error_message}")
    logger.error(traceback.format_exc())
    
    return ErrorResponse(
        error=error_message,
        details={"type": type(exc).__name__},
        traceback=traceback.format_exc() if logger.isEnabledFor(logging.DEBUG) else None
    )


# Monadic API utilities

def with_timeout(timeout_seconds: float):
    """Decorator to add timeout to endpoint handlers."""
    import asyncio
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs), 
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"Request timeout after {timeout_seconds} seconds"
                )
        return wrapper
    return decorator


def with_rate_limit(max_calls: int, window_seconds: float):
    """Decorator to add rate limiting to endpoints."""
    from collections import defaultdict
    from datetime import datetime, timedelta
    import asyncio
    
    call_times = defaultdict(list)
    lock = asyncio.Lock()
    
    def decorator(func):
        @wraps(func)
        async def wrapper(request, *args, **kwargs):
            client_id = request.client.host
            now = datetime.now()
            
            async with lock:
                # Clean old entries
                call_times[client_id] = [
                    t for t in call_times[client_id] 
                    if now - t < timedelta(seconds=window_seconds)
                ]
                
                # Check rate limit
                if len(call_times[client_id]) >= max_calls:
                    raise HTTPException(
                        status_code=429,
                        detail=f"Rate limit exceeded: {max_calls} calls per {window_seconds} seconds"
                    )
                
                # Record this call
                call_times[client_id].append(now)
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


# Example of using monadic patterns for complex endpoints

@app.post("/analyze", response_model=Dict[str, Any])
async def analyze_generation(
    request: GenerationRequest,
    service: MonadicGenerationService = Depends(get_generation_service)
) -> Dict[str, Any]:
    """Analyze generation parameters and provide recommendations.
    
    This demonstrates complex monadic composition.
    """
    from src.domain.monads import Maybe, some, nothing
    
    def analyze_threshold(threshold: float) -> Result[str, str]:
        if threshold < 0.01:
            return Ok("Very selective - expect focused, deterministic output")
        elif threshold < 0.1:
            return Ok("Selective - balanced creativity and coherence")
        elif threshold < 0.3:
            return Ok("Moderate - increased diversity in outputs")
        else:
            return Ok("High - very creative but potentially less coherent")
    
    def analyze_max_tokens(tokens: int) -> Result[str, str]:
        if tokens < 50:
            return Ok("Short generation - suitable for completions")
        elif tokens < 200:
            return Ok("Medium generation - suitable for paragraphs")
        else:
            return Ok("Long generation - suitable for extended text")
    
    def recommend_settings(req: GenerationRequest) -> Result[Dict[str, Any], str]:
        # Analyze various parameters
        threshold_analysis = analyze_threshold(req.selection_threshold)
        tokens_analysis = analyze_max_tokens(req.max_tokens)
        
        recommendations = {
            "threshold_analysis": threshold_analysis.unwrap(),
            "tokens_analysis": tokens_analysis.unwrap(),
            "recommendations": []
        }
        
        # Add specific recommendations
        if req.selection_threshold > 0.2 and req.use_retroactive_removal:
            recommendations["recommendations"].append(
                "Consider lowering selection threshold when using retroactive removal"
            )
        
        if req.max_tokens > 500 and not req.use_mcts:
            recommendations["recommendations"].append(
                "Consider enabling MCTS for long generations"
            )
        
        return Ok(recommendations)
    
    # Build analysis pipeline
    analysis_pipeline = (
        validate_request(request)
        .flat_map(recommend_settings)
    )
    
    return handle_generation_result(analysis_pipeline)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# Monadic API Design for TEMPO

## Overview

The monadic API design introduces functional programming patterns to the TEMPO generation system, providing better error handling, composition, and dependency injection through the use of monads.

## Monadic Types

### Result Monad
Handles operations that might fail:
```python
def generate_text(request: GenerationRequest) -> Result[GenerationResponse, str]:
    return (
        validate_request(request)
        .flat_map(lambda req: service.generate_text(req))
    )
```

### Maybe Monad
Handles optional values without null checks:
```python
def get_system_content(self) -> Maybe[str]:
    return some(self.system_content) if self.system_content else nothing()
```

### Either Monad
Represents values with two possible types:
```python
def parse_config() -> Either[ConfigError, Config]:
    # Returns Left(error) or Right(config)
```

### IO Monad
Wraps side effects for pure functional composition:
```python
log_io = IO(lambda: logger.info("Processing request"))
log_io.run()  # Execute the side effect
```

### Reader Monad
Dependency injection through monadic composition:
```python
def process_with_deps() -> Reader[Dependencies, Result]:
    return ask().flat_map(lambda deps: use_deps(deps))
```

### State Monad
Stateful computations in a functional way:
```python
counter = get_state().flat_map(
    lambda n: put_state(n + 1).then(state_pure(n))
)
```

## Architecture

### 1. Monadic Generation Service
Located at `src/application/services/monadic_generation_service.py`

Features:
- Pipeline-based generation using Reader monad for dependency injection
- Result monad for comprehensive error handling
- IO monad for side effects (logging, timing)
- Composable generation steps

Example pipeline:
```python
generation_pipeline = (
    self._create_context(request)
    .flat_map(self._set_debug_mode)
    .flat_map(self._log_request)
    .flat_map(self._create_retroactive_remover)
    .flat_map(self._configure_rope_modifier)
    .flat_map(self._prepare_system_content)
    .flat_map(self._perform_generation)
    .flat_map(self._format_response)
)
```

### 2. Monadic API Endpoints
Located at `api_monadic.py`

Features:
- Result-based error handling
- Streaming support with monadic patterns
- Batch processing with error accumulation
- Rate limiting and timeout decorators

Example endpoint:
```python
@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    service: MonadicGenerationService = Depends(get_generation_service)
) -> GenerationResponse:
    generation_pipeline = (
        validate_request(request)
        .flat_map(lambda req: service.generate_text(req))
    )
    return handle_generation_result(generation_pipeline)
```

### 3. Composition Utilities
Located at `src/domain/monads/composition.py`

Utilities for working with monads:
- `sequence_results`: Convert list of Results to Result of list
- `parallel_results`: Collect all results, accumulating errors
- `traverse_result`: Map Result-returning function over list
- `kleisli_result`: Compose Result-returning functions
- `compose` / `pipe`: Function composition
- `curry`: Convert functions to curried form

## Benefits

### 1. Explicit Error Handling
```python
# Errors are explicit in the type system
def risky_operation() -> Result[Value, Error]:
    if success:
        return Ok(value)
    else:
        return Err(error)
```

### 2. Composable Operations
```python
# Chain operations that might fail
result = (
    validate_input(data)
    .flat_map(process_data)
    .flat_map(save_to_database)
    .map(format_response)
)
```

### 3. Dependency Injection
```python
# Dependencies are passed through Reader monad
computation = ReaderT(lambda deps: 
    deps.database.query()
        .flat_map(lambda data: deps.processor.process(data))
)
```

### 4. Pure Functions
```python
# Side effects are isolated in IO monad
def pure_computation(x: int) -> IO[None]:
    return IO(lambda: print(f"Result: {x * 2}"))
```

## Usage Examples

### Basic Generation
```python
from src.domain.monads import Result, Ok, Err

# Request validation and generation
result = service.generate_text(request)

# Handle result
response = result.fold(
    lambda err: handle_error(err),
    lambda resp: format_success(resp)
)
```

### Batch Processing
```python
# Process multiple requests
results = [validate_request(req) for req in requests]
validated = sequence_results(results)

if validated.is_ok():
    responses = [service.generate_text(req) for req in validated.unwrap()]
```

### Error Recovery
```python
# Retry with fallback
result = (
    try_primary_model(request)
    .or_else(lambda _: try_fallback_model(request))
    .or_else(lambda _: return_cached_response(request))
)
```

## Testing

The monadic design makes testing easier:

```python
def test_generation_pipeline():
    # Mock dependencies
    mock_deps = GenerationDependencies(
        model_repository=Mock(),
        response_formatter=Mock()
    )
    
    # Test individual steps
    context_result = service._create_context(request).run(mock_deps)
    assert context_result.is_ok()
    
    # Test error cases
    mock_deps.model_repository.side_effect = Exception()
    error_result = service._create_context(request).run(mock_deps)
    assert error_result.is_err()
```

## Running the Monadic API

```bash
# Start the monadic API server
uvicorn api_monadic:app --reload --port 8001

# The API will be available at http://localhost:8001
# API documentation at http://localhost:8001/docs
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /generate` - Generate text
- `POST /generate/batch` - Batch generation
- `POST /generate/stream` - Streaming generation
- `POST /analyze` - Analyze generation parameters

## Future Enhancements

1. **Async Monads**: Implement AsyncResult and AsyncIO for better async support
2. **Effect System**: Add typed effects for more granular side-effect control
3. **Monad Transformers**: Add more transformers for complex stacks
4. **Property-Based Testing**: Use monadic laws for property testing
5. **Performance Optimization**: Implement trampolining for deep recursion
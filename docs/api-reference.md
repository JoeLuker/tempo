# TEMPO API Reference

This document provides a complete reference for the TEMPO REST API. The API is built with FastAPI and provides endpoints for text generation with parallel token exploration.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. This may change in future versions.

## Rate Limiting

Default rate limits:
- 100 requests per minute per IP
- 5 concurrent requests maximum

Configure via environment variables:
- `TEMPO_API_RATE_LIMIT`: Requests per minute
- `TEMPO_API_MAX_CONCURRENT`: Maximum concurrent requests

## Endpoints

### Health Check

#### `GET /health`

Check if the API is running and the model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_id": "deepcogito/cogito-v1-preview-llama-3B",
  "version": "0.1.0"
}
```

**Status Codes:**
- `200`: Service is healthy
- `503`: Service unavailable (model not loaded)

---

### Generate Text

#### `POST /api/generate`

Generate text using TEMPO's parallel token exploration.

**Request Body:**
```json
{
  "prompt": "string",
  "max_tokens": 100,
  "selection_threshold": 0.1,
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.95,
  "repetition_penalty": 1.0,
  "use_retroactive_pruning": false,
  "attention_threshold": 0.01,
  "dynamic_threshold": false,
  "bezier_p1": 0.2,
  "bezier_p2": 0.8,
  "use_mcts": false,
  "mcts_simulations": 10,
  "mcts_c_puct": 1.0,
  "mcts_depth": 5,
  "enable_thinking": false,
  "default_mode": false,
  "allow_intraset_token_visibility": false,
  "use_custom_rope": true,
  "save_visualization": false,
  "seed": null
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Input text to continue generation from |
| `max_tokens` | integer | 100 | Maximum number of tokens to generate |
| `selection_threshold` | float | 0.1 | Minimum probability for token selection (0.0-1.0) |
| `temperature` | float | 0.8 | Sampling temperature (higher = more random) |
| `top_k` | integer | 50 | Limit to top K tokens |
| `top_p` | float | 0.95 | Nucleus sampling threshold |
| `repetition_penalty` | float | 1.0 | Penalty for repeated tokens |
| `use_retroactive_pruning` | boolean | false | Enable attention-based pruning |
| `attention_threshold` | float | 0.01 | Threshold for retroactive pruning |
| `dynamic_threshold` | boolean | false | Use dynamic threshold adjustment |
| `bezier_p1` | float | 0.2 | First Bezier control point |
| `bezier_p2` | float | 0.8 | Second Bezier control point |
| `use_mcts` | boolean | false | Enable Monte Carlo Tree Search |
| `mcts_simulations` | integer | 10 | Number of MCTS simulations |
| `mcts_c_puct` | float | 1.0 | MCTS exploration constant |
| `mcts_depth` | integer | 5 | Maximum MCTS depth |
| `enable_thinking` | boolean | false | Enable Cogito thinking mode |
| `default_mode` | boolean | false | Use standard generation (no TEMPO) |
| `allow_intraset_token_visibility` | boolean | false | Allow parallel tokens to see each other |
| `use_custom_rope` | boolean | true | Use custom RoPE modifications |
| `save_visualization` | boolean | false | Generate visualization data |
| `seed` | integer | null | Random seed for reproducibility |

**Response:**
```json
{
  "generated_text": "The [future/world] of [AI/technology] is [bright/promising]",
  "clean_text": "The future of AI is bright",
  "raw_generated_text": "The future world of AI technology is bright promising",
  "token_count": 8,
  "generation_time": 2.34,
  "visualization_data": {
    "token_tree": {...},
    "attention_maps": {...},
    "pruning_history": [...]
  },
  "metadata": {
    "model_id": "deepcogito/cogito-v1-preview-llama-3B",
    "selection_threshold": 0.1,
    "parallel_positions": 3,
    "total_candidates": 12,
    "pruned_tokens": 4
  }
}
```

**Status Codes:**
- `200`: Successful generation
- `400`: Invalid request parameters
- `422`: Validation error
- `500`: Internal server error
- `503`: Model not loaded

---

### Batch Generation

#### `POST /api/generate/batch`

Generate text for multiple prompts in a single request.

**Request Body:**
```json
{
  "prompts": ["prompt1", "prompt2", "prompt3"],
  "common_params": {
    "max_tokens": 100,
    "selection_threshold": 0.1,
    "temperature": 0.8
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "prompt": "prompt1",
      "generated_text": "...",
      "clean_text": "...",
      "success": true
    },
    {
      "prompt": "prompt2",
      "generated_text": "...",
      "clean_text": "...",
      "success": true
    },
    {
      "prompt": "prompt3",
      "error": "Generation failed: ...",
      "success": false
    }
  ],
  "total_time": 10.5,
  "successful": 2,
  "failed": 1
}
```

---

### Streaming Generation

#### `POST /api/generate/stream`

Generate text with server-sent events for real-time streaming.

**Request:** Same as `/api/generate`

**Response:** Server-Sent Events stream

```
data: {"token": "The", "position": 0, "alternatives": []}

data: {"token": "future", "position": 1, "alternatives": ["world", "next"]}

data: {"token": "of", "position": 2, "alternatives": []}

data: {"done": true, "clean_text": "The future of AI", "total_tokens": 4}
```

**Client Example:**
```javascript
const eventSource = new EventSource('/api/generate/stream', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({prompt: "Hello"})
});

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.done) {
    console.log("Complete:", data.clean_text);
    eventSource.close();
  } else {
    console.log("Token:", data.token);
  }
};
```

---

### Analyze Generation

#### `POST /api/analyze`

Analyze generation parameters without performing generation.

**Request Body:**
```json
{
  "prompt": "Test prompt",
  "selection_threshold": 0.1,
  "max_tokens": 100
}
```

**Response:**
```json
{
  "prompt_tokens": 3,
  "estimated_time": 2.5,
  "memory_usage_mb": 4500,
  "parameter_recommendations": {
    "selection_threshold": "Good for balanced exploration",
    "attention_threshold": "Consider 0.01-0.02 for this threshold",
    "max_parallel_tokens": 5
  },
  "warnings": [
    "High selection_threshold may produce many parallel tokens"
  ]
}
```

---

### Model Information

#### `GET /api/model/info`

Get information about the loaded model.

**Response:**
```json
{
  "model_id": "deepcogito/cogito-v1-preview-llama-3B",
  "model_type": "LlamaForCausalLM",
  "vocab_size": 128256,
  "max_position_embeddings": 131072,
  "hidden_size": 3072,
  "num_hidden_layers": 28,
  "device": "cuda",
  "quantization": null,
  "memory_usage_mb": 6144
}
```

---

### List Available Models

#### `GET /api/models`

List models available for loading.

**Response:**
```json
{
  "current_model": "deepcogito/cogito-v1-preview-llama-3B",
  "available_models": [
    "deepcogito/cogito-v1-preview-llama-3B",
    "gpt2",
    "gpt2-medium",
    "mistralai/Mistral-7B-Instruct-v0.2"
  ]
}
```

---

### Configuration

#### `GET /api/config`

Get current API configuration.

**Response:**
```json
{
  "api_version": "v2",
  "default_model": "deepcogito/cogito-v1-preview-llama-3B",
  "max_concurrent_requests": 5,
  "rate_limit": 100,
  "features": {
    "mcts_enabled": true,
    "streaming_enabled": true,
    "batch_enabled": true,
    "retroactive_pruning_enabled": true
  }
}
```

---

## Error Responses

All error responses follow this format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field": "Additional error context"
    }
  },
  "request_id": "uuid-string",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Error Codes

| Code | Description |
|------|-------------|
| `INVALID_PROMPT` | Prompt is empty or too long |
| `INVALID_PARAMETER` | Parameter value out of valid range |
| `MODEL_NOT_LOADED` | Model not initialized |
| `GENERATION_FAILED` | Error during text generation |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `TIMEOUT` | Request timed out |
| `INTERNAL_ERROR` | Unexpected server error |

## WebSocket API

### `WS /ws/generate`

WebSocket endpoint for bidirectional streaming generation.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/generate');
```

**Message Format:**
```json
{
  "type": "generate",
  "prompt": "Hello",
  "params": {
    "max_tokens": 100,
    "selection_threshold": 0.1
  }
}
```

**Response Messages:**
```json
// Token generated
{
  "type": "token",
  "token": "world",
  "position": 1,
  "alternatives": ["universe", "everyone"]
}

// Generation complete
{
  "type": "complete",
  "full_text": "Hello world",
  "clean_text": "Hello world",
  "stats": {...}
}

// Error
{
  "type": "error",
  "error": "Generation failed",
  "code": "GENERATION_FAILED"
}
```

## Examples

### cURL Examples

**Basic generation:**
```bash
curl -X POST "http://localhost:8000/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_tokens": 50,
    "selection_threshold": 0.1
  }'
```

**With retroactive pruning:**
```bash
curl -X POST "http://localhost:8000/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing",
    "max_tokens": 200,
    "selection_threshold": 0.08,
    "use_retroactive_pruning": true,
    "attention_threshold": 0.015
  }'
```

### Python Examples

```python
import requests

# Basic generation
response = requests.post(
    "http://localhost:8000/api/generate",
    json={
        "prompt": "Once upon a time",
        "max_tokens": 100,
        "selection_threshold": 0.1
    }
)
result = response.json()
print(result["clean_text"])

# Streaming generation
import sseclient

response = requests.post(
    "http://localhost:8000/api/generate/stream",
    json={"prompt": "Hello world"},
    stream=True
)

client = sseclient.SSEClient(response)
for event in client.events():
    data = json.loads(event.data)
    if data.get("done"):
        print("Complete:", data["clean_text"])
        break
    else:
        print("Token:", data["token"])
```

### JavaScript Examples

```javascript
// Basic generation
const response = await fetch('http://localhost:8000/api/generate', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    prompt: 'The meaning of life is',
    max_tokens: 100
  })
});
const result = await response.json();
console.log(result.clean_text);

// With error handling
try {
  const response = await fetch('http://localhost:8000/api/generate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      prompt: 'Hello',
      selection_threshold: 0.1
    })
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error.message);
  }
  
  const result = await response.json();
  console.log(result);
} catch (error) {
  console.error('Generation failed:', error);
}
```

## API Versioning

The API uses URL versioning. Current version: `v2`

- Latest version: `/api/generate`
- Specific version: `/api/v2/generate`
- Legacy version: `/api/v1/generate` (deprecated)

Version compatibility is maintained for 6 months after deprecation.

## Performance Tips

1. **Reuse connections**: Use connection pooling for multiple requests
2. **Batch when possible**: Use `/api/generate/batch` for multiple prompts
3. **Stream for long outputs**: Use streaming endpoints for better UX
4. **Cache results**: Implement client-side caching for repeated prompts
5. **Set appropriate timeouts**: Long generations may take 30+ seconds

## OpenAPI Specification

Full OpenAPI specification available at:
- JSON: `http://localhost:8000/openapi.json`
- Interactive docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
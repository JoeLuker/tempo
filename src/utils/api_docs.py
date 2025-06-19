"""
API documentation utilities for TEMPO API.

This module provides tools for generating comprehensive API documentation,
including examples, use cases, and detailed explanations of parameters.
"""

from typing import Any, Optional
from datetime import datetime


class APIDocumentation:
    """Manager for API documentation content."""

    # Documentation content - this would typically be stored in a database or files
    # For demonstration, we'll store it in memory
    docs = {
        "overview": {
            "title": "TEMPO API Overview",
            "content": """
# TEMPO API

TEMPO (Threshold-Enabled Multipath Parallel Output) is an experimental approach to language model text generation
that processes multiple token possibilities simultaneously, providing more coherent and diverse outputs.

## Key Features

- **Parallel Token Generation**: Process multiple possible next tokens at each step
- **Retroactive Pruning**: Refine token sets based on future token attention
- **Custom RoPE Modifications**: Optimized for parallel token positioning
- **Threshold-Based Selection**: Control generation diversity vs. coherence
- **Monte Carlo Tree Search**: Optional advanced search strategies

## API Versions

- **v2**: Current stable version with full features (use /api/v2/...)
- **v1**: Legacy version (deprecated)

## Core Endpoints

- `POST /api/v2/generate`: Generate text with TEMPO
- `GET /api/v2/models/list`: List available models
- `GET /api/health`: Check system health
            """,
        },
        "generate": {
            "title": "Text Generation Guide",
            "content": """
# TEMPO Text Generation Guide

This guide explains how to use the TEMPO API for text generation.

## Basic Generation

The simplest way to generate text is to provide a prompt:

```json
{
  "prompt": "Explain the difference between a llama and an alpaca",
  "max_tokens": 50
}
```

## Controlling Generation Quality

### Selection Threshold

The `selection_threshold` parameter (default: 0.1) controls how strict the initial token filtering is:

- **Lower values** (e.g., 0.05): More diverse but potentially less coherent outputs
- **Higher values** (e.g., 0.3): More focused but potentially more repetitive outputs

### Retroactive Pruning

Enable retroactive pruning to refine token sets based on future context:

```json
{
  "prompt": "Write a short story about",
  "pruning_settings": {
    "enabled": true,
    "attention_threshold": 0.01,
    "use_relative_attention": true
  }
}
```

## Advanced Parameters

### Dynamic Thresholds

Gradually increase selection threshold over generation:

```json
{
  "threshold_settings": {
    "use_dynamic_threshold": true,
    "final_threshold": 0.8,
    "bezier_points": [0.2, 0.8]
  }
}
```

### Monte Carlo Tree Search

Use MCTS for more strategic token selection:

```json
{
  "mcts_settings": {
    "use_mcts": true,
    "simulations": 20,
    "c_puct": 1.5,
    "depth": 3
  }
}
```
            """,
        },
        "examples": {
            "title": "API Usage Examples",
            "content": """
# TEMPO API Examples

## Simple Text Generation

```python
import requests
import json

api_url = "http://localhost:8000/api/v2/generate"

payload = {
    "prompt": "Write a poem about artificial intelligence",
    "max_tokens": 100,
    "selection_threshold": 0.1
}

response = requests.post(api_url, json=payload)
result = response.json()

print(result["generated_text"])
```

## Advanced Generation with Custom Settings

```python
import requests
import json

api_url = "http://localhost:8000/api/v2/generate"

payload = {
    "prompt": "Explain how quantum computers work",
    "max_tokens": 150,
    "selection_threshold": 0.05,
    "min_steps": 10,
    "threshold_settings": {
        "use_dynamic_threshold": true,
        "final_threshold": 0.8,
        "bezier_points": [0.2, 0.8]
    },
    "pruning_settings": {
        "enabled": true,
        "attention_threshold": 0.01,
        "use_relative_attention": true,
        "relative_threshold": 0.5
    },
    "advanced_settings": {
        "use_custom_rope": true,
        "disable_kv_cache": false,
        "debug_mode": false
    }
}

response = requests.post(api_url, json=payload)
result = response.json()

print(result["generated_text"])
```

## Command Line Example (using curl)

```bash
curl -X POST http://localhost:8000/api/v2/generate \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "Write a short story about a robot learning to paint",
    "max_tokens": 100,
    "selection_threshold": 0.1
  }'
```
            """,
        },
        "parameters": {
            "title": "API Parameter Reference",
            "content": """
# TEMPO API Parameter Reference

## Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | *required* | Text prompt to start generation |
| `max_tokens` | integer | 50 | Maximum number of tokens to generate |
| `selection_threshold` | float | 0.1 | Probability threshold for token selection |
| `min_steps` | integer | 0 | Minimum steps to generate before considering EOS |
| `model_name` | string | *system model* | Model to use for generation |

## Threshold Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_dynamic_threshold` | boolean | false | Use threshold that changes over generation steps |
| `final_threshold` | float | 1.0 | Final threshold value for dynamic thresholding |
| `bezier_points` | array | [0.2, 0.8] | Bezier control points for threshold curve |
| `use_relu` | boolean | false | Use ReLU transition instead of Bezier curve |
| `relu_activation_point` | float | 0.5 | Point where ReLU transition begins |

## Pruning Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | true | Use retroactive pruning |
| `attention_threshold` | float | 0.01 | Attention threshold for pruning |
| `use_relative_attention` | boolean | true | Use relative attention thresholds |
| `relative_threshold` | float | 0.5 | Threshold for relative attention |
| `use_multi_scale_attention` | boolean | true | Use multi-scale attention integration |
| `num_layers_to_use` | integer | null | Number of layers to use for attention |
| `use_lci_dynamic_threshold` | boolean | true | Use LCI dynamic thresholding |
| `use_sigmoid_threshold` | boolean | true | Use sigmoid decision boundary |
| `sigmoid_steepness` | float | 10.0 | Sigmoid steepness parameter |
| `pruning_mode` | string | "keep_token" | How to handle pruned positions |

## MCTS Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_mcts` | boolean | false | Use Monte Carlo Tree Search |
| `simulations` | integer | 10 | Number of MCTS simulations per step |
| `c_puct` | float | 1.0 | Exploration constant for MCTS |
| `depth` | integer | 5 | Maximum depth for MCTS simulations |

## Advanced Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_custom_rope` | boolean | true | Use custom RoPE modifications |
| `disable_kv_cache` | boolean | false | Disable KV caching |
| `disable_kv_cache_consistency` | boolean | false | Disable KV cache consistency checks |
| `allow_intraset_token_visibility` | boolean | false | Allow parallel tokens to see each other |
| `no_preserve_isolated_tokens` | boolean | false | Allow pruning isolated tokens |
| `show_token_ids` | boolean | false | Include token IDs in output |
| `system_content` | string | null | System message for chat models |
| `enable_thinking` | boolean | false | Enable deep thinking mode |
| `debug_mode` | boolean | false | Enable debug mode for logging |
            """,
        },
        "errors": {
            "title": "API Error Reference",
            "content": """
# TEMPO API Error Reference

## Error Response Format

All API errors follow this standardized format:

```json
{
  "status_code": 400,
  "error_type": "validation_error",
  "message": "Invalid request parameters",
  "timestamp": "2023-05-01T12:34:56.789Z",
  "path": "/api/v2/generate",
  "details": [
    {
      "field": "max_tokens",
      "message": "Value must be greater than 0",
      "code": "value_error"
    }
  ],
  "request_id": "abcd1234-ef56-7890"
}
```

## Common Error Types

| Status Code | Error Type | Description |
|-------------|------------|-------------|
| 400 | `request_error` | Invalid request that passed validation |
| 401 | `authentication_error` | Authentication required |
| 403 | `authorization_error` | Not authorized to access resource |
| 404 | `resource_not_found` | Resource not found |
| 422 | `validation_error` | Request validation failed |
| 429 | `rate_limit_error` | Rate limit exceeded |
| 500 | `server_error` | Server-side error |
| 500 | `model_error` | Model processing error |
| 500 | `generation_error` | Text generation failed |
| 503 | `dependency_error` | Model not available |
| 504 | `timeout_error` | Request timed out |

## Handling Rate Limits

When you receive a `429 Too Many Requests` error, the response includes a `Retry-After` header with the number of seconds to wait before trying again.

```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1620000000
Retry-After: 60
```
            """,
        },
    }

    @classmethod
    def get_section(cls, section_name: str) -> dict[str, Any]:
        """
        Get documentation content for a specific section.

        Args:
            section_name: Name of the documentation section

        Returns:
            Dict with title and content of the section
        """
        if section_name in cls.docs:
            return cls.docs[section_name]

        return {
            "title": "Documentation Not Found",
            "content": f"No documentation found for section: {section_name}",
        }

    @classmethod
    def get_all_sections(cls) -> dict[str, dict[str, str]]:
        """
        Get all documentation sections.

        Returns:
            Dict mapping section names to content
        """
        return cls.docs

    @classmethod
    def get_section_list(cls) -> list[dict[str, str]]:
        """
        Get a list of available documentation sections.

        Returns:
            List of section info dictionaries
        """
        return [
            {"name": name, "title": content["title"]}
            for name, content in cls.docs.items()
        ]

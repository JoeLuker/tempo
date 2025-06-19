# TEMPO Configuration Guide

This guide covers all configuration options for TEMPO, including environment variables, configuration files, and runtime parameters.

## Configuration Hierarchy

TEMPO uses a hierarchical configuration system with the following priority (highest to lowest):

1. **Runtime parameters** (CLI arguments or API request parameters)
2. **Environment variables** (prefixed with `TEMPO_`)
3. **Configuration file** (`config.json`)
4. **Default values** (hardcoded in source)

## Configuration File

Create a `config.json` file based on the provided template:

```bash
cp config.example.json config.json
```

### Complete Configuration Structure

```json
{
  "logging": {
    "enable_file_logging": true,
    "log_dir": "logs",
    "log_level": "INFO",
    "console_logging": true,
    "max_log_size_mb": 100,
    "backup_count": 5,
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  },
  "model": {
    "model_id": "deepcogito/cogito-v1-preview-llama-3B",
    "device": null,
    "quantization": null,
    "trust_remote_code": false,
    "use_fast_tokenizer": true,
    "revision": null,
    "low_cpu_mem_usage": true,
    "torch_dtype": "float16",
    "cache_dir": "~/.cache/huggingface",
    "local_files_only": false,
    "max_memory": null,
    "offload_folder": "offload",
    "offload_state_dict": false
  },
  "generation": {
    "max_length": 200,
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 0.8,
    "repetition_penalty": 1.1,
    "length_penalty": 1.0,
    "beam_width": 1,
    "use_dynamic_thresholding": true,
    "use_retroactive_pruning": true,
    "use_parallel_generation": true,
    "max_parallel_tokens": 5,
    "min_steps": 0,
    "early_stopping": true,
    "pad_token_id": null,
    "eos_token_id": null,
    "use_cache": true
  },
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "cors_origins": ["*"],
    "debug": false,
    "enable_docs": true,
    "api_version": "v2",
    "max_concurrent_requests": 5,
    "request_timeout": 300,
    "rate_limit": 100,
    "enable_metrics": true,
    "metrics_port": 9090
  },
  "debug": {
    "global_debug": false,
    "module_debug": {
      "token_generator": false,
      "attention_manager": false,
      "rope_modifier": false,
      "model_wrapper": false,
      "token_selector": false,
      "parallel_generator": false,
      "experiment_runner": false,
      "retroactive_pruner": false,
      "mcts_generator": false,
      "api": false,
      "cache_manager": false,
      "performance_tracker": false
    },
    "save_intermediate_states": false,
    "log_attention_maps": false,
    "profile_memory": false,
    "trace_execution": false
  },
  "performance": {
    "enable_profiling": false,
    "profile_output_dir": "profiles",
    "memory_efficient_mode": false,
    "batch_size": 1,
    "num_workers": 0,
    "prefetch_factor": 2,
    "persistent_workers": false,
    "pin_memory": true
  },
  "cache": {
    "enable_prompt_cache": true,
    "prompt_cache_size": 1000,
    "enable_kv_cache": true,
    "kv_cache_implementation": "default",
    "cache_dir": ".cache",
    "max_cache_size_gb": 10,
    "cache_eviction_policy": "lru"
  },
  "experimental": {
    "enable_flash_attention": false,
    "enable_xformers": false,
    "enable_torch_compile": false,
    "torch_compile_mode": "default",
    "enable_gradient_checkpointing": false,
    "mixed_precision": "no"
  }
}
```

## Environment Variables

All configuration options can be set via environment variables using the `TEMPO_` prefix and nested keys separated by underscores.

### Common Environment Variables

```bash
# Logging
export TEMPO_LOGGING_LOG_LEVEL=DEBUG
export TEMPO_LOGGING_LOG_DIR=/var/log/tempo
export TEMPO_LOGGING_ENABLE_FILE_LOGGING=true

# Model
export TEMPO_MODEL_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.2
export TEMPO_MODEL_DEVICE=cuda
export TEMPO_MODEL_QUANTIZATION=4bit
export TEMPO_MODEL_TORCH_DTYPE=float16

# Generation
export TEMPO_GENERATION_MAX_LENGTH=500
export TEMPO_GENERATION_TEMPERATURE=0.7
export TEMPO_GENERATION_USE_RETROACTIVE_PRUNING=true

# API
export TEMPO_API_PORT=8080
export TEMPO_API_HOST=0.0.0.0
export TEMPO_API_RATE_LIMIT=200

# Debug
export TEMPO_DEBUG_GLOBAL_DEBUG=true
export TEMPO_DEBUG_MODULE_DEBUG_TOKEN_GENERATOR=true

# Performance
export TEMPO_PERFORMANCE_ENABLE_PROFILING=true
export TEMPO_PERFORMANCE_MEMORY_EFFICIENT_MODE=true
```

### Special Environment Variables

```bash
# Override config file location
export TEMPO_CONFIG_FILE=/etc/tempo/config.json

# Force CPU-only mode
export CUDA_VISIBLE_DEVICES=""

# Set HuggingFace cache directory
export HF_HOME=/data/huggingface

# Enable offline mode
export HF_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## CLI Parameter Reference

### Generation Parameters

| Parameter | Type | Default | Environment Variable | Description |
|-----------|------|---------|---------------------|-------------|
| `--prompt` | str | "Once upon a time" | N/A | Input prompt |
| `--max-tokens` | int | 100 | `TEMPO_GENERATION_MAX_LENGTH` | Maximum tokens to generate |
| `--selection-threshold` | float | 0.1 | `TEMPO_GENERATION_SELECTION_THRESHOLD` | Token selection threshold |
| `--temperature` | float | 0.8 | `TEMPO_GENERATION_TEMPERATURE` | Sampling temperature |
| `--top-k` | int | 50 | `TEMPO_GENERATION_TOP_K` | Top-k sampling |
| `--top-p` | float | 0.95 | `TEMPO_GENERATION_TOP_P` | Nucleus sampling |
| `--repetition-penalty` | float | 1.0 | `TEMPO_GENERATION_REPETITION_PENALTY` | Repetition penalty |

### Pruning Parameters

| Parameter | Type | Default | Environment Variable | Description |
|-----------|------|---------|---------------------|-------------|
| `--use-retroactive-pruning` | flag | False | `TEMPO_GENERATION_USE_RETROACTIVE_PRUNING` | Enable retroactive pruning |
| `--attention-threshold` | float | 0.01 | `TEMPO_PRUNING_ATTENTION_THRESHOLD` | Attention threshold |
| `--dynamic-threshold` | flag | False | `TEMPO_PRUNING_DYNAMIC_THRESHOLD` | Dynamic threshold adjustment |
| `--bezier-p1` | float | 0.2 | `TEMPO_PRUNING_BEZIER_P1` | First Bezier control point |
| `--bezier-p2` | float | 0.8 | `TEMPO_PRUNING_BEZIER_P2` | Second Bezier control point |

### Advanced Parameters

| Parameter | Type | Default | Environment Variable | Description |
|-----------|------|---------|---------------------|-------------|
| `--use-mcts` | flag | False | `TEMPO_MCTS_ENABLED` | Enable MCTS generation |
| `--mcts-simulations` | int | 10 | `TEMPO_MCTS_SIMULATIONS` | MCTS simulations |
| `--enable-thinking` | flag | False | `TEMPO_COGITO_THINKING_ENABLED` | Cogito thinking mode |
| `--default-mode` | flag | False | `TEMPO_GENERATION_DEFAULT_MODE` | Standard generation |

## Performance Tuning

### Memory Optimization

For systems with limited memory:

```json
{
  "model": {
    "quantization": "4bit",
    "low_cpu_mem_usage": true,
    "torch_dtype": "float16",
    "offload_state_dict": true
  },
  "performance": {
    "memory_efficient_mode": true,
    "batch_size": 1
  },
  "cache": {
    "enable_kv_cache": false,
    "max_cache_size_gb": 2
  }
}
```

### Speed Optimization

For maximum generation speed:

```json
{
  "model": {
    "torch_dtype": "float16",
    "use_fast_tokenizer": true
  },
  "generation": {
    "use_cache": true,
    "beam_width": 1
  },
  "performance": {
    "enable_profiling": false,
    "pin_memory": true
  },
  "cache": {
    "enable_prompt_cache": true,
    "enable_kv_cache": true
  },
  "experimental": {
    "enable_flash_attention": true,
    "enable_torch_compile": true,
    "mixed_precision": "fp16"
  }
}
```

### Quality Optimization

For best generation quality:

```json
{
  "generation": {
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "use_retroactive_pruning": true,
    "use_dynamic_thresholding": true
  },
  "model": {
    "torch_dtype": "float32"
  }
}
```

## Device Configuration

### GPU Selection

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0
export TEMPO_MODEL_DEVICE=cuda:0

# Use multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1
export TEMPO_MODEL_DEVICE=cuda

# Force CPU
export CUDA_VISIBLE_DEVICES=""
export TEMPO_MODEL_DEVICE=cpu
```

### Apple Silicon (MPS)

```json
{
  "model": {
    "device": "mps",
    "torch_dtype": "float32"
  }
}
```

## Logging Configuration

### Log Levels

- `DEBUG`: Detailed information for debugging
- `INFO`: General information about execution
- `WARNING`: Warning messages
- `ERROR`: Error messages only
- `CRITICAL`: Critical errors only

### Per-Module Debugging

Enable debug logging for specific modules:

```bash
# Enable all debug logging
export TEMPO_DEBUG_GLOBAL_DEBUG=true

# Enable specific modules
export TEMPO_DEBUG_MODULE_DEBUG_TOKEN_GENERATOR=true
export TEMPO_DEBUG_MODULE_DEBUG_ATTENTION_MANAGER=true
export TEMPO_DEBUG_MODULE_DEBUG_RETROACTIVE_PRUNER=true
```

### Log Rotation

```json
{
  "logging": {
    "max_log_size_mb": 100,
    "backup_count": 5,
    "log_dir": "/var/log/tempo"
  }
}
```

## API Configuration

### CORS Settings

```json
{
  "api": {
    "cors_origins": [
      "http://localhost:5174",
      "https://myapp.com"
    ]
  }
}
```

### Rate Limiting

```json
{
  "api": {
    "rate_limit": 100,
    "rate_limit_window": 60,
    "rate_limit_strategy": "sliding_window"
  }
}
```

## Configuration Profiles

### Development Profile

```bash
export TEMPO_PROFILE=development
```

config-dev.json:
```json
{
  "logging": {"log_level": "DEBUG"},
  "api": {"debug": true, "enable_docs": true},
  "debug": {"global_debug": true}
}
```

### Production Profile

```bash
export TEMPO_PROFILE=production
```

config-prod.json:
```json
{
  "logging": {"log_level": "WARNING"},
  "api": {"debug": false, "enable_docs": false},
  "performance": {"memory_efficient_mode": true}
}
```

## Validation and Testing

### Validate Configuration

```python
from src.utils import TempoConfig

# Validate configuration file
config = TempoConfig.from_file("config.json")
errors = config.validate()
if errors:
    print("Configuration errors:", errors)
```

### Test Configuration

```bash
# Test configuration without starting the server
python -c "from src.utils import config; print(config.to_dict())"

# Verify model loading with configuration
python run_tempo.py --prompt "Test" --max-tokens 10 --dry-run
```

## Troubleshooting

### Common Issues

1. **Configuration not loading**: Check file path and JSON syntax
2. **Environment variables not working**: Ensure proper prefix (`TEMPO_`)
3. **Nested keys**: Use underscores for nesting (e.g., `TEMPO_MODEL_DEVICE`)
4. **Type errors**: Ensure correct types in JSON (strings need quotes)

### Debug Configuration Loading

```bash
# Enable configuration debug logging
export TEMPO_CONFIG_DEBUG=true

# Show final configuration
python -c "from src.utils import config; config.print_config()"
```

## Best Practices

1. **Use configuration files** for stable settings
2. **Use environment variables** for deployment-specific settings
3. **Use runtime parameters** for experimentation
4. **Version control** your configuration files
5. **Document** custom configurations
6. **Validate** configurations before deployment
7. **Use profiles** for different environments
8. **Monitor** performance impact of settings

## Migration Guide

### From v1 to v2

Configuration changes:
- `max_length` → `max_tokens`
- `use_rope_modification` → `use_custom_rope`
- `pruning_threshold` → `attention_threshold`

Migration script:
```python
import json

with open("config-v1.json") as f:
    old_config = json.load(f)

# Update keys
if "max_length" in old_config["generation"]:
    old_config["generation"]["max_tokens"] = old_config["generation"].pop("max_length")

# Save new config
with open("config-v2.json", "w") as f:
    json.dump(old_config, f, indent=2)
```
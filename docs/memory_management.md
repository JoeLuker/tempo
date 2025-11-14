# Memory Management in TEMPO

TEMPO now includes comprehensive memory controls to ensure generation stays within specified limits. This is crucial for running TEMPO on systems with constrained memory resources.

## Overview

Memory management in TEMPO controls three main consumers:

1. **Model Weights** (~3GB for cogito-v1-preview-llama-3B) - Fixed cost
2. **KV Cache** - Grows with sequence length and number of layers
3. **Parallel Token Batches** - Depends on number of parallel tokens processed

## Configuration

### Command Line

```bash
# Set maximum total memory usage (default: 36GB)
python3 run_tempo.py --prompt "Your prompt" --max-memory-gb 36.0

# Manually limit parallel tokens per step
python3 run_tempo.py --prompt "Your prompt" --max-parallel-tokens 20

# Limit maximum tokens in KV cache
python3 run_tempo.py --prompt "Your prompt" --max-cache-tokens 2048
```

### Configuration File

```yaml
# config.yaml
max_memory_gb: 36.0
max_parallel_tokens: 20  # Optional - auto-calculated if not set
max_cache_tokens: 2048   # Optional - for long sequences

prompt: "Your prompt here"
max_tokens: 100
selection_threshold: 0.1
```

```bash
python3 run_tempo.py --config config.yaml
```

## How It Works

### 1. Memory Monitor

The `MemoryMonitor` class tracks memory usage and enforces limits:

```python
from src.utils.memory_monitor import MemoryMonitor

# Initialize with 36GB limit
monitor = MemoryMonitor(max_memory_gb=36.0, device="cuda")

# Check current usage
monitor.log_memory_stats("After model load")

# Verify we're within limit
monitor.check_memory_limit("model loading")
```

### 2. Automatic Limit Calculation

If `max_parallel_tokens` is not specified, TEMPO automatically calculates a safe value:

```python
max_parallel_tokens = monitor.calculate_max_parallel_tokens(
    sequence_length=100  # max_tokens value
)
```

This calculation:
- Estimates memory needed per parallel token
- Accounts for hidden size, vocab size, sequence length
- Reserves 20% for overhead
- Caps between 1 and 50 tokens

### 3. KV Cache Limits

The KV cache manager enforces memory limits:

```python
from src.algorithms.generation.kv_cache_manager import KVCacheManager

cache_manager = KVCacheManager(
    num_layers=32,
    device="cuda",
    max_memory_gb=10.0,  # Dedicated cache limit
    max_cache_tokens=2048  # Maximum sequence length
)
```

If cache expansion would exceed limits, a `RuntimeError` is raised.

### 4. Parallel Token Limits

The parallel processor enforces per-step limits:

```python
from src.algorithms.generation.parallel_processor import ParallelProcessor

processor = ParallelProcessor(
    device="cuda",
    max_parallel_tokens=20  # Max parallel tokens per step
)

# Automatically trims to top-k by probability
token_set = processor.create_parallel_set(
    position=5,
    token_ids=[10, 20, 30, ...],  # 30 tokens
    probabilities=[0.3, 0.25, 0.15, ...]  # Sorted by prob
)
# Result: Only top 20 tokens kept
```

## Memory Estimation

### KV Cache Memory

```python
kv_memory_gb = monitor.estimate_kv_cache_memory(
    batch_size=1,
    sequence_length=100,
    num_layers=32,
    num_heads=32,
    head_dim=96,
    dtype=torch.float16
)
# Example: ~0.5GB for 100 tokens
```

### Parallel Batch Memory

```python
parallel_memory_gb = monitor.estimate_parallel_batch_memory(
    num_parallel_tokens=20,
    sequence_length=100,
    vocab_size=128256,
    hidden_size=3072,
    dtype=torch.float16
)
# Example: ~0.8GB for 20 parallel tokens
```

## Best Practices

### 1. Start with Auto-Calculation

Let TEMPO calculate `max_parallel_tokens` automatically:

```bash
python3 run_tempo.py --prompt "Your prompt" --max-memory-gb 36.0
# No need to specify --max-parallel-tokens
```

### 2. Monitor Memory Usage

Enable debug mode to see detailed memory logs:

```bash
python3 run_tempo.py --prompt "Your prompt" --max-memory-gb 36.0 --debug-mode
```

Output will show:
```
Memory: System 15.23/64.00GB (23.8%) GPU 8.45GB allocated, 9.12GB reserved
```

### 3. Adjust for Long Sequences

For sequences longer than 500 tokens, consider setting `max_cache_tokens`:

```bash
python3 run_tempo.py \
    --prompt "Your prompt" \
    --max-tokens 1000 \
    --max-memory-gb 36.0 \
    --max-cache-tokens 1024  # Limit cache growth
```

### 4. Balance Quality vs Memory

Lower `selection_threshold` = more parallel tokens = more memory:

```bash
# Higher quality, more memory
python3 run_tempo.py --selection-threshold 0.05 --max-memory-gb 36.0

# Lower memory, faster, slightly lower quality
python3 run_tempo.py --selection-threshold 0.15 --max-memory-gb 24.0
```

## Troubleshooting

### Memory Limit Exceeded

If you see:
```
RuntimeError: Cache memory limit would be exceeded: 38.50GB > 36.00GB
```

Solutions:
1. Increase `--max-memory-gb` if system allows
2. Reduce `--max-tokens` (shorter sequences)
3. Increase `--selection-threshold` (fewer parallel tokens)
4. Set `--max-parallel-tokens` to a lower value

### Out of Memory (OOM)

If CUDA runs out of memory:

```bash
# Reduce batch processing
python3 run_tempo.py --max-parallel-tokens 10 --max-memory-gb 20.0

# Enable memory cleanup
python3 run_tempo.py --disable-kv-cache --max-memory-gb 24.0
```

### Slow Generation

If generation is slower than expected:

- Check if `max_parallel_tokens` is too low (< 5)
- Verify memory limit isn't forcing excessive pruning
- Consider increasing `--max-memory-gb` if available

## Memory Thresholds

The memory monitor uses these thresholds:

- **Warning**: 85% of `max_memory_gb`
  - Logs warning message
  - Generation continues

- **Critical**: 95% of `max_memory_gb`
  - Raises `MemoryError`
  - Generation stops

These thresholds ensure safety margin for memory spikes.

## Testing Memory Controls

Run the test script to verify memory controls:

```bash
python3 test_memory_controls.py
```

This will:
1. Load the model
2. Calculate memory-safe limits
3. Run a test generation
4. Verify memory stayed under 36GB
5. Report success/failure

## Advanced Usage

### Custom Memory Monitor

```python
from src.utils.memory_monitor import MemoryMonitor

monitor = MemoryMonitor(max_memory_gb=36.0, device="cuda")

# Get current stats
stats = monitor.get_memory_stats()
print(f"GPU allocated: {stats.gpu_allocated_gb:.2f}GB")
print(f"Available: {stats.gpu_free_gb:.2f}GB")

# Free memory
monitor.free_memory()  # Runs gc.collect() and torch.cuda.empty_cache()

# Get memory config for components
config = monitor.get_memory_config()
# Returns: {
#     "max_memory_gb": 36.0,
#     "current_usage_gb": 8.5,
#     "available_gb": 27.5,
#     ...
# }
```

### Programmatic Control

```python
from src.experiments import ExperimentRunner, ArgumentParser

args = {
    "prompt": "The future of AI is",
    "max_tokens": 100,
    "selection_threshold": 0.1,
    "max_memory_gb": 36.0,
    "max_parallel_tokens": 20,  # Manual override
    "output_dir": "./output"
}

runner = ExperimentRunner(model=model_wrapper, tokenizer=tokenizer)
results = runner.run_experiment(args)
```

## Implementation Details

### Architecture

Memory controls are implemented across multiple layers:

```
┌─────────────────────────┐
│   run_tempo.py          │  CLI entry point
│   MemoryMonitor init    │  Logs stats, calculates limits
└──────────┬──────────────┘
           │
           v
┌─────────────────────────┐
│  ExperimentRunner       │  Passes limits to config
│  args["max_memory_gb"]  │
└──────────┬──────────────┘
           │
           v
┌─────────────────────────┐
│  GenerationConfig       │  Stores memory params
│  max_parallel_tokens    │
│  max_cache_tokens       │
└──────────┬──────────────┘
           │
           v
┌─────────────────────────────────────┐
│  Components enforce limits:         │
│  - KVCacheManager.check_limits()    │
│  - ParallelProcessor.create_set()   │
│  - MemoryMonitor.check_limit()      │
└─────────────────────────────────────┘
```

### Key Files

- **`src/utils/memory_monitor.py`**: Memory tracking and estimation
- **`src/algorithms/generation/kv_cache_manager.py`**: KV cache with limits
- **`src/algorithms/generation/parallel_processor.py`**: Parallel token limits
- **`src/domain/entities/parallel_generation.py`**: Config with memory params
- **`run_tempo.py`**: Integration and auto-calculation

## Future Enhancements

Planned improvements:

1. **Dynamic memory adjustment** during generation
2. **Per-layer memory tracking** for detailed analysis
3. **Memory-aware pruning strategies**
4. **Swap-based cache management** for ultra-long sequences
5. **Multi-GPU memory distribution**

## See Also

- [Configuration Guide](configuration-guide.md)
- [Architecture Documentation](architecture.md)
- [Quickstart Guide](quickstart.md)

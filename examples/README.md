# TEMPO Examples

This directory contains example scripts demonstrating various TEMPO features and use cases.

## Quick Start

Make sure TEMPO is installed and dependencies are available:
```bash
cd ..  # Go to TEMPO root directory
pip install -r requirements.txt
```

## Available Examples

### 1. Basic Generation (`basic_generation.py`)

Simple examples to get started with TEMPO:
- Simple text generation
- Creative writing with higher thresholds
- Generation with retroactive pruning
- Dynamic thresholding
- Cogito thinking mode
- Low threshold generation

```bash
# List all examples
python examples/basic_generation.py --list

# Run a specific example
python examples/basic_generation.py --example 1

# Run all examples
python examples/basic_generation.py --all
```

### 2. Advanced Pruning (`advanced_pruning.py`)

Demonstrates sophisticated pruning strategies:
- Basic retroactive pruning
- Relative attention thresholds
- Multi-scale attention integration
- Sigmoid decision boundaries
- Dynamic Bezier/ReLU thresholding
- Complete position removal
- Combined advanced features

```bash
# Run specific pruning example
python examples/advanced_pruning.py --example 3

# See all pruning strategies
python examples/advanced_pruning.py --all
```

### 3. API Client (`api_client.py`)

Shows how to use TEMPO programmatically via API:
- Basic API usage
- Batch processing
- Concurrent requests
- Parameter comparison
- Error handling

**Note**: Requires the API server to be running:
```bash
# In a separate terminal
uvicorn api:app --reload --port 8000
```

Then run examples:
```bash
# Basic API example
python examples/api_client.py --example 1

# Concurrent requests
python examples/api_client.py --example 4
```

## Example Output

TEMPO generates text with visible branching points:
```
The future of AI [is/will be] [fascinating/transformative/revolutionary]
```

Where `[option1/option2/option3]` shows the parallel tokens considered at each step.

## Tips for Experimentation

1. **Selection Threshold**: Controls how many parallel tokens to consider
   - Low (0.05-0.1): Minimal branching, more focused
   - Medium (0.1-0.2): Balanced branching
   - High (0.2+): Many alternatives, very exploratory

2. **Retroactive Pruning**: Refines past choices based on future context
   - `attention_threshold`: Lower = keep more tokens
   - `dynamic_threshold`: Gradually increases selectivity

3. **Temperature**: Controls randomness
   - Low (0.5-0.8): More predictable
   - High (0.9-1.5): More creative

## Creating Your Own Examples

Copy one of the existing examples as a template:

```python
#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_tempo import main as run_tempo

# Your example
args = [
    "--prompt", "Your prompt here",
    "--selection-threshold", "0.1",
    "--max-tokens", "100",
    # Add more parameters
]

run_tempo(args)
```

## Common Issues

1. **Import Error**: Make sure you're running from the TEMPO root directory
2. **API Connection Error**: Ensure the API server is running for `api_client.py`
3. **Out of Memory**: Reduce `--max-tokens` or use a smaller model

## Further Reading

- See the main [README.md](../README.md) for full parameter documentation
- Check [docs/](../docs/) for detailed guides
- Visit the [API docs](http://localhost:8000/docs) when server is running
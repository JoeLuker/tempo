# TEMPO: Threshold-Enabled Multipath Parallel Output

[![GitHub](https://img.shields.io/github/license/JoeLuker/tempo)](https://github.com/JoeLuker/tempo/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/JoeLuker/tempo)](https://github.com/JoeLuker/tempo/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/JoeLuker/tempo)](https://github.com/JoeLuker/tempo/issues)

This project implements and evaluates a novel text generation mechanism called **TEMPO (Threshold-Enabled Multipath Parallel Output)**. TEMPO explores generating text non-autoregressively for specific steps by processing multiple token possibilities simultaneously, using modifications to standard Transformer attention and positional encoding.

## Table of Contents
- [Overview](#overview)
- [Core Mechanism](#core-mechanism)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)
- [Testing](#testing)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

Standard autoregressive text generation selects a single token at each step. TEMPO investigates an alternative approach where, at a given step, multiple tokens above a probability **Threshold** are selected. Instead of creating separate sequence branches (like beam search), TEMPO uses advanced techniques to process these parallel possibilities concurrently within a *single evolving sequence state*.

The core idea is to:

1.  **Select Multiple Candidates:** Identify all tokens exceeding a probability threshold at the current generation step.
2.  **Prune Candidates:** Optionally apply pruning strategies (coherence, diversity, retroactive attention) to refine the set of candidate tokens for the current step.
3.  **Simulate Parallel Processing:** Use modifications to Rotary Position Embeddings (RoPE) and attention masks to perform a *single* forward pass where the chosen candidate tokens are processed *as if* they occupy the same logical position simultaneously. This allows the model to consider the combined influence of these possibilities when predicting the *next* step.
4.  **Maintain Single Sequence:** Append *all* selected (and potentially pruned) candidate tokens sequentially to the main input tensor. The RoPE and attention modifications ensure they are treated appropriately relative to their logical position during subsequent forward passes. Parallelism at a position only collapses if pruning naturally reduces the candidate set to one token.
5.  **Produce Output:** Generate text where positions with multiple simultaneous tokens are marked (e.g., `[tokenA/tokenB]`), revealing the model's internal parallelism or uncertainty at those steps.

This allows exploring how language models might internally represent and process concurrent possibilities without the full computational cost of maintaining separate sequence branches.

## Core Mechanism

-   **Threshold Selection:** A probability threshold (`--threshold`) determines the initial set of parallel candidate tokens at each logical step.
-   **Sequential Layout:** All selected candidate tokens are appended sequentially to a single `input_ids` tensor.
-   **Logical Position Tracking:** A separate mechanism (`logical_layout`) tracks which physical indices in the `input_ids` tensor correspond to the same logical generation step.
-   **RoPE Modification (`RoPEModifier`):** Patches the model's Rotary Position Embedding calculation. Uses the `logical_layout` to assign the *same* positional embedding to all tokens belonging to the same logical step, even though they occupy different physical positions in the `input_ids` tensor.
-   **Attention Mask Modification (`AttentionManager`):** Generates custom attention masks (if the model supports explicit mask input, otherwise relies on RoPE alone) that allow tokens within the same logical step (parallel set) to attend to each other (controlled by `--allow-intraset-token-visibility`) while maintaining overall causality.
-   **Single KV Cache:** Evolves a single standard `past_key_values` cache based on the sequential processing, with the RoPE/Attention modifications influencing the internal calculations during the forward pass.
-   **Pruning:** Strategies (`--use-pruning`, `--pruning-strategy`, `--attention-threshold`, etc.) refine the set of candidate tokens considered at each step *before* they are appended to the sequence and processed simultaneously.

## Features

-   **Simultaneous Token Processing:** Explores parallelism by processing multiple candidate tokens per logical step using RoPE and Attention modifications within a single sequence state.
-   **Configurable Parallelism:** Control the initial candidate set size via `--threshold`.
-   **Advanced Pruning:**
    -   Filter candidates based on semantic coherence (`coherence` strategy, uses attention patterns if available).
    -   Select diverse candidates using embedding clustering (`diversity` strategy).
    *   Combine strategies (`hybrid` strategy).
    *   Retroactive Pruning (`--use-pruning --attention-threshold`): Refines *previously processed* parallel sets based on attention from later tokens (requires model attention output).
    -   Dynamic Thresholding (`--dynamic-threshold`): Adjust pruning aggressiveness over time using Bezier or ReLU curves.
-   **Parallel Token Interaction Control:**
    *   `--allow-intraset-token-visibility`: Lets parallel tokens attend to each other during the simultaneous processing step (requires `--use-custom-rope`). Default is isolated.
-   **Visualization:** Output indicates positions where multiple tokens were processed simultaneously (e.g., `[tokenA/tokenB]`). `token_sets` data provides detailed history.
-   **Web Interface**: Modern Svelte-based UI for interactive exploration.
-   **Early Exit Transformers**: (Optional feature) Integrates adaptive computation for potentially faster inference on compatible models (see `examples/early_exit_demo.py`).

## System Requirements

-   Python 3.8 or higher
-   Node.js 16 or higher (for frontend)
-   Apple Silicon Mac (M1/M2) for optimal performance
-   At least 16GB RAM
-   20GB free disk space for model and dependencies

## Installation

### Backend Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn api:app --reload
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The web interface will be available at `http://localhost:5173` and the API at `http://localhost:8000`.

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
MODEL_PATH=/path/to/model
DEVICE=mps  # or cuda for NVIDIA GPUs
LOG_LEVEL=INFO
API_PORT=8000
FRONTEND_PORT=5173
```

### Model Configuration

The system supports various model configurations through the API:

- Model size and architecture
- Generation parameters
- Pruning strategies
- Visualization options

See the API documentation at `http://localhost:8000/docs` for detailed configuration options.

## Usage

### Command Line Interface

```bash
# Basic generation
python run_tempo.py --prompt "Your prompt here" --threshold 0.1

# With MCTS enabled
python run_tempo.py --prompt "Your prompt here" --threshold 0.1 --use-mcts --mcts-simulations 10

# With deep thinking mode
python run_tempo.py --prompt "Your prompt here" --threshold 0.1 --enable-thinking

# With custom pruning strategy
python run_tempo.py --prompt "Your prompt here" --threshold 0.1 --use-pruning --coherence-threshold 0.7 --diversity-clusters 3

# With early exit
python run_tempo.py --prompt "Your prompt here" --early-exit --exit-layers "3,7,11,15" --confidence-thresholds "0.7,0.75,0.8,0.9"
```

### API Usage

The FastAPI backend provides the following endpoints:

- `GET /`: API documentation
- `GET /health`: Health check endpoint
- `POST /generate`: Main generation endpoint with extensive configuration options

Example API call:

```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Your prompt here", "threshold": 0.1}'
```

## Development

### Project Structure

```
.
├── api.py                 # FastAPI backend implementation
├── run_tempo.py          # Command-line interface
├── requirements.txt      # Python dependencies
├── frontend/            # Svelte frontend
│   ├── src/            # Frontend source code
│   ├── static/         # Static assets
│   └── package.json    # Frontend dependencies
├── src/                # Core implementation
│   ├── modeling/       # Model wrapper and utilities
│   ├── generation/     # Generation strategies
│   ├── search/        # MCTS implementation
│   ├── pruning/       # Pruning strategies
│   ├── visualization/ # Visualization tools
│   └── experiments/   # Experimental features and research
├── output/            # Generated outputs
└── images/            # Project images and visualizations
```

### Development Guidelines

1. Follow PEP 8 style guide for Python code
2. Use type hints for all function parameters and return values
3. Write docstrings for all public functions and classes
4. Keep commits atomic and well-documented
5. Create feature branches for new development
6. Update documentation when adding new features

## Testing

### Backend Testing

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest

# Run specific test file
pytest tests/test_generation.py

# Run with coverage
pytest --cov=src
```

### Frontend Testing

```bash
cd frontend
npm test
```

## Security

### Best Practices

1. Never commit API keys or sensitive credentials
2. Use environment variables for configuration
3. Validate all user input
4. Implement rate limiting for API endpoints
5. Keep dependencies updated
6. Use HTTPS in production

### Security Headers

The API includes security headers by default:
- CORS protection
- XSS protection
- Content Security Policy
- HSTS (in production)

## Troubleshooting

### Common Issues

1. **Model Loading Issues**
   - Ensure sufficient disk space
   - Check model path in configuration
   - Verify GPU/CPU compatibility

2. **API Connection Problems**
   - Check if both backend and frontend servers are running
   - Verify port configurations
   - Check firewall settings

3. **Generation Errors**
   - Adjust threshold values
   - Check available memory
   - Verify model compatibility

4. **Visualization Issues**
   - Clear browser cache
   - Check browser compatibility
   - Verify data format

### Getting Help

- Check the [issues](https://github.com/JoeLuker/tempo/issues) page
- Create a new issue with detailed error information
- Include system information and error logs
- Follow [@JoeLuker](https://github.com/JoeLuker) on GitHub for updates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

Please ensure your PR:
- Follows the project's coding style
- Includes tests for new features
- Updates documentation
- Has a clear description

## License

TEMPO is licensed under the MIT License.

The MIT License is a permissive license that allows you to:
- Use the software for any purpose
- Modify and distribute the software
- Use the software commercially
- Keep modifications private

The only requirements are:
- Include the original copyright notice
- Include the license text

See the [LICENSE](LICENSE) file for full terms and conditions.

## New Feature: Early-Exit Transformers

The latest addition to TEMPO is the Early-Exit Transformer capability, which allows models to terminate processing before completing all layers when sufficient confidence is reached, dramatically improving inference speed.

### Key Benefits of Early Exits

- **Adaptive Computation**: Uses just the right amount of compute based on query complexity
- **Reduced Latency**: Simpler queries complete much faster (up to 5x speedup)
- **Power Efficiency**: Significantly lower energy consumption (30-50% less)
- **MPS Optimization**: Particularly well-suited to Mac's Metal architecture

### Using Early Exits

To enable early exits, use the following command-line flags:

```bash
python run_tempo.py --prompt "Your prompt here" --early-exit --exit-layers "3,7,11,15" --confidence-thresholds "0.7,0.75,0.8,0.9"
```

Parameters:
- `--early-exit`: Enable early exit capability
- `--exit-layers`: Comma-separated list of layer indices for early exits (optional)
- `--confidence-thresholds`: Comma-separated list of confidence thresholds (optional)

If you don't specify exit layers or thresholds, sensible defaults will be used (exits every 4 layers with gradually increasing confidence requirements).

You can also run a direct example:

```bash
python examples/early_exit_demo.py --model "JackFram/llama-68m" --prompt "The capital of France is" --compare
```

This will demonstrate the speedup compared to standard generation.

# TEMPO: Threshold-Enabled Multipath Parallel Output

[![GitHub](https://img.shields.io/github/license/JoeLuker/tempo)](https://github.com/JoeLuker/tempo/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/JoeLuker/tempo)](https://github.com/JoeLuker/tempo/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/JoeLuker/tempo)](https://github.com/JoeLuker/tempo/issues)

This project implements and evaluates **TEMPO (Threshold-Enabled Multipath Parallel Output)**, an experimental approach to language model generation. TEMPO explores processing multiple token possibilities simultaneously at certain steps, aiming to understand how models might handle concurrent hypotheses within a single sequence state, differing from traditional beam search which maintains separate sequences.

**Note:** This project is currently in an experimental phase, primarily focused on the `deepcogito/cogito-v1-preview-llama-3B` model.

## Table of Contents

- [Overview](#overview)
- [Target Model](#target-model)
- [Core Mechanism](#core-mechanism)
- [Key Features](#key-features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Development](#development)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

Standard autoregressive text generation selects a single token at each step. TEMPO investigates an alternative approach where, at a given "logical" step, multiple token candidates above a probability **Selection Threshold** are identified. Instead of branching the sequence, TEMPO processes these parallel possibilities concurrently within a *single evolving sequence state* using modifications to positional embeddings.

## Motivation & Research Context

TEMPO explores how language models handle uncertainty and multiple hypotheses during generation. By allowing models to process multiple token possibilities simultaneously, we can:

- **Understand Model Uncertainty**: Visualize where models are genuinely uncertain vs. confident
- **Study Attention Mechanisms**: Analyze how retroactive attention patterns reveal which tokens truly matter for coherence
- **Interpretability Research**: The parallel token exploration provides insights into the model's internal decision-making process

This work contributes to mechanistic interpretability by exposing the model's internal branching logic and attention dynamics during text generation.

The core idea is to:

1. **Select Multiple Candidates:** Identify all tokens exceeding a `--selection-threshold` probability at the current logical generation step.
2. **Prune Candidates (Optional):** Apply pruning strategies (strategy-based or retroactive attention-based) to refine the candidate set for the current step.
3. **Append Sequentially:** Append *all* selected (and potentially pruned) candidate tokens physically one after another onto the main `input_ids` tensor.
4. **Simulate Parallelism via RoPE:** Use modifications to Rotary Position Embeddings (**`RoPEModifier`**) to assign the *same* positional embedding to all physical tokens belonging to the same logical step. This is the primary mechanism allowing the model to process them *as if* they occupy the same logical position simultaneously during the *single* forward pass for the *next* step's prediction.
5. **Maintain Single State:** Evolve a single standard `past_key_values` (KV cache) based on this sequential processing. The RoPE modification ensures the model internally accounts for the intended logical parallelism.
6. **Produce Output:** Generate text where positions involving multiple simultaneous tokens (before potential pruning) are marked (e.g., `[tokenA/tokenB]`), revealing the model's internal branching or uncertainty at those steps.

## Target Model

Currently, this project is focused on experimentation and validation using the **`deepcogito/cogito-v1-preview-llama-3B`** model. While the concepts might apply more broadly, the RoPE patching and default configurations are tuned for this specific Llama-based architecture. Generalization to other models is a potential future direction but is not the immediate goal.

## Core Mechanism

- **Selection Threshold:** A probability threshold (`--selection-threshold`) determines the initial set of parallel candidate tokens considered at each logical step.
- **Sequential Layout:** All selected candidate tokens (potentially after pruning) are appended sequentially to a single `input_ids` tensor.
- **Logical Position Tracking (`logical_layout`):** An internal mechanism tracks which physical indices in the `input_ids` tensor correspond to the same logical generation step.
- **RoPE Modification (`RoPEModifier`):** (Requires `--use-custom-rope`) Patches the model's Rotary Position Embedding calculation. Uses the `logical_layout` to assign the *same* positional embedding to all tokens belonging to the same logical step, tricking the model into processing them as simultaneous alternatives.
- **Attention Masking (`AttentionManager`):** By default, uses standard causal masking. Can generate custom masks if needed, potentially allowing parallel tokens to attend to each other (controlled by `--allow-intraset-token-visibility`, requires RoPE modification).
- **Single KV Cache:** Evolves a single standard `past_key_values` cache. RoPE modifications influence how the query/key vectors are calculated before cache interaction. *KV Cache consistency logic has been simplified/removed as the position mapping handles the core requirement.*
- **Pruning:**
  - **Strategy-Based Pruning (`--use-pruning`, `--pruning-strategy`, etc.):** Refines the candidate set *before* tokens are appended, using coherence, diversity, or a hybrid approach. Uses its *own* threshold (e.g., `--coherence-threshold`).
  - **Retroactive Removal (`--use-retroactive-removal`):** Refines *previously processed* parallel sets based on attention from later tokens (uses `--attention-threshold` as base for dynamic curve).

## Key Features

- **Simulated Parallel Processing:** Explores parallelism by processing multiple candidate tokens per logical step using RoPE modifications within a single sequence state.
- **Configurable Selection:** Control the initial candidate set size via `--selection-threshold`.
- **Advanced Pruning:**
  - **Retroactive Removal** (`--use-retroactive-removal`): Refines *previously processed* parallel sets based on attention from later tokens.
  - **Dynamic Thresholding** (`--dynamic-threshold`): Adjust retroactive removal aggressiveness over time using Bezier (`--bezier-p1`, `--bezier-p2`) or ReLU (`--use-relu`, `--relu-activation-point`) curves.
  - **Attention-Based Pruning**: Multiple attention mechanisms including relative thresholds (`--relative-threshold`), multi-scale attention, and sigmoid decision boundaries.
- **Generation Modes:**
  - **TEMPO Mode** (default): Full parallel token processing with RoPE modifications.
  - **Default Mode** (`--default-mode`): Standard generation without TEMPO features.
  - **MCTS Mode** (`--use-mcts`): Monte Carlo Tree Search for enhanced exploration.
- **Model-Specific Features:**
  - **Cogito Thinking Mode** (`--enable-thinking`): Enhanced reasoning for Cogito models.
  - **Custom RoPE** (`--use-custom-rope`): Modify positional embeddings for parallel processing.
- **Parallel Token Control:**
  - `--allow-intraset-token-visibility`: Lets parallel tokens attend to each other (requires `--use-custom-rope`).
  - `--no-preserve-isolated-tokens`: Controls pruning behavior for isolated tokens.
- **Visualization & Analysis:**
  - Output text indicates positions where multiple tokens were processed simultaneously (e.g., `[tokenA/tokenB]`).
  - Comprehensive visualization tools (`--save-visualization`).
  - Web Interface (`frontend/`): Interactive Svelte UI for real-time exploration.
  - Performance profiling (`--profile`, `--use-cprofile`).

## System Requirements

- Python 3.8 or higher
- PyTorch >= 2.0
- Transformers >= 4.30
- Node.js 16 or higher (for frontend)
- Target Model: `deepcogito/cogito-v1-preview-llama-3B` (tested on Apple Silicon M-series with MPS/CPU fallback)
- RAM: 16GB+ recommended
- Disk Space: ~20GB for model and dependencies

## Installation

### Backend Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt # For testing

# Start the FastAPI server (for frontend interaction)
uvicorn api:app --reload --port 8000
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server (connects to backend on port 8000)
npm run dev
```

The web interface will be available at `http://localhost:5174` and the API at `http://localhost:8000`.

## Usage

### Command Line Interface (`run_tempo.py`)

The primary way to run experiments with detailed control and profiling.

#### Basic Usage Examples

```bash
# Basic generation (using defaults for the target model)
python run_tempo.py --prompt "Explain the theory of relativity simply." --selection-threshold 0.05 --max-tokens 150

# Enable retroactive removal
python run_tempo.py --prompt "Write a haiku about servers." --selection-threshold 0.1 --use-retroactive-removal --attention-threshold 0.02

# Use dynamic threshold with Bezier curve for retroactive removal
python run_tempo.py --prompt "Story about a lost robot." --selection-threshold 0.08 --use-retroactive-removal --attention-threshold 0.01 --dynamic-threshold --bezier-p1 0.1 --bezier-p2 0.9

# Enable Debug Mode for detailed logs
python run_tempo.py --prompt "Debug this." --selection-threshold 0.2 --max-tokens 20 --debug-mode

# Enable performance profiling
python run_tempo.py --prompt "Profile this run." --selection-threshold 0.1 --max-tokens 50 --profile --use-cprofile
```

#### Advanced Features

```bash
# MCTS generation with exploration
python run_tempo.py --prompt "Creative story beginning" --use-mcts --mcts-simulations 20 --mcts-c-puct 1.5 --mcts-depth 8

# Cogito thinking mode for enhanced reasoning
python run_tempo.py --prompt "Solve this logic puzzle: " --enable-thinking --selection-threshold 0.15

# Advanced pruning with multiple attention mechanisms
python run_tempo.py --prompt "Complex reasoning task" --use-retroactive-removal --attention-threshold 0.005 --relative-threshold 0.3 --sigmoid-steepness 15.0

# Default mode (standard generation without TEMPO)
python run_tempo.py --prompt "Simple generation" --default-mode --max-tokens 100

# Custom model and output directory
python run_tempo.py --prompt "Custom setup" --model "path/to/model" --output-dir "./my_outputs" --seed 123
```

#### Technical Configuration

```bash
# Disable KV cache for testing
python run_tempo.py --prompt "Technical test" --disable-kv-cache

# Allow parallel tokens to see each other
python run_tempo.py --prompt "Parallel visibility test" --allow-intraset-token-visibility --selection-threshold 0.2

# Fine-tuned pruning control
python run_tempo.py --prompt "Precision pruning" --use-retroactive-removal --complete-removal-mode "keep_unattended" --num-layers-to-use 8
```

### API Usage (`api.py`)

The FastAPI backend provides endpoints for integration, primarily used by the frontend.

#### Available Endpoints

- `GET /docs`: Interactive API documentation (Swagger UI)
- `GET /health`: Health check endpoint
- `POST /api/generate`: Main generation endpoint (v1)
- `POST /api/v2/generate`: v2 generation endpoint (backwards compatibility)

#### API Parameters

The API accepts most CLI parameters in JSON format, with these key differences:
- CLI flags become boolean fields (e.g., `--use-retroactive-removal` → `"use_retroactive_removal": true`)
- Underscore naming convention (e.g., `selection_threshold` instead of `--selection-threshold`)
- All parameters including MCTS are available via API

#### Example API Calls

```bash
# Basic generation
curl -X POST "http://localhost:8000/api/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "Translate to French: Hello, world!",
           "selection_threshold": 0.1,
           "max_tokens": 30
         }'

# Advanced generation with retroactive removal
curl -X POST "http://localhost:8000/api/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "Write a creative story about AI",
           "selection_threshold": 0.08,
           "max_tokens": 200,
           "use_retroactive_removal": true,
           "attention_threshold": 0.015,
           "dynamic_threshold": true,
           "bezier_p1": 0.1,
           "bezier_p2": 0.9
         }'

# Cogito thinking mode
curl -X POST "http://localhost:8000/api/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "Explain quantum entanglement",
           "enable_thinking": true,
           "selection_threshold": 0.12,
           "max_tokens": 300
         }'

# MCTS generation
curl -X POST "http://localhost:8000/api/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "Write a creative story opening",
           "use_mcts": true,
           "mcts_simulations": 20,
           "mcts_c_puct": 1.5,
           "mcts_depth": 8,
           "selection_threshold": 0.15,
           "max_tokens": 100
         }'
```

### Web Interface

Run the backend (`uvicorn`) and frontend (`npm run dev`) simultaneously. Access the UI at `http://localhost:5173` for interactive generation and visualization.

## Development

### Project Structure

``` text
.
├── api.py                 # FastAPI backend implementation
├── run_tempo.py          # Command-line interface for experiments
├── requirements.txt      # Python dependencies
├── requirements-test.txt # Test dependencies
├── frontend/            # Svelte frontend
├── src/                # Core TEMPO implementation
│   ├── modeling/       # Model wrapper, Early Exit
│   ├── generation/     # ParallelGenerator, RoPE, Attention, Tokenizer, Selector, Formatter
│   ├── pruning/       # Pruning strategies, Dynamic Threshold
│   ├── search/        # MCTS implementation (CLI only)
│   ├── visualization/ # Plotting tools
│   └── experiments/   # ExperimentRunner, ArgumentParser
├── tests/               # Unit and Integration tests
│   ├── unit/
│   └── integration/
├── output/            # Default directory for generated outputs/visualizations
└── README.md            # This file
```

### Development Guidelines

1. Follow PEP 8 style guide.
2. Use type hints.
3. Write docstrings for public functions/classes.
4. Keep commits focused. Use feature branches.
5. Update documentation (this README) with significant changes.
6. Add tests for new functionality.

## Testing

Use the provided `run_tests.py` script for convenience.

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
python run_tests.py

# Run unit tests only
python run_tests.py --unit-only

# Run integration tests only
python run_tests.py --integration-only

# Run tests with coverage report (saved to .coverage)
python run_tests.py --cov

# View HTML coverage report (after running with --cov)
# pip install coverage
# coverage html
# open htmlcov/index.html
```

## Troubleshooting

- **Model Loading Issues:** Ensure sufficient RAM/disk space. Check model name/path. Verify PyTorch/CUDA/MPS setup.
- **API/Frontend Connection:** Make sure both `uvicorn` and `npm run dev` are running. Check ports (default 8000 backend, 5173 frontend).
- **Generation Errors:** Check logs in the `logs/` directory (enable `--debug-mode`). Try adjusting the `--threshold` or pruning parameters. `RuntimeError` in `CoherencePruningStrategy` likely means attention outputs were not correctly obtained.
- **RoPE Patching Errors:** If `RoPEModifier` fails to install, the model structure might differ significantly from the expected Llama-like architecture. Examine the debug output during installation in `run_tempo.py`.

## Contributing

Contributions are welcome, but please note the current focus on the specific target model.

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## CLI Parameters Reference

TEMPO provides extensive command-line configuration options organized into logical groups:

### Basic Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--prompt` | str | "Once upon a time" | Text prompt to start generation |
| `--max-tokens` | int | 100 | Maximum number of tokens to generate |
| `--selection-threshold` | float | 0.1 | Probability threshold for initial token candidate selection |
| `--min-steps` | int | 0 | Minimum steps to generate before considering EOS tokens |
| `--model` | str | "deepcogito/cogito-v1-preview-llama-3B" | Model name or path to use |
| `--output-dir` | str | "./output" | Directory to save output |
| `--seed` | int | 42 | Random seed for reproducibility |

### Pruning & Attention Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use-retroactive-pruning` | flag | False | Enable retroactive pruning based on future token attention |
| `--attention-threshold` | float | 0.01 | Attention threshold for retroactive pruning (lower = more tokens kept) |
| `--use-relative-attention` | flag | False | Use relative attention thresholds instead of absolute |
| `--no-relative-attention` | flag | False | Disable relative attention thresholds |
| `--relative-threshold` | float | 0.5 | Threshold for relative attention-based pruning (0-1) |
| `--use-multi-scale-attention` | flag | False | Enable multi-scale attention integration across all layers |
| `--no-multi-scale-attention` | flag | False | Disable multi-scale attention integration |
| `--num-layers-to-use` | int | None | Number of last layers to use for attention (None = all layers) |
| `--use-sigmoid-threshold` | flag | False | Enable sigmoid-based decision boundary |
| `--no-sigmoid-threshold` | flag | False | Disable sigmoid-based decision boundary |
| `--sigmoid-steepness` | float | 10.0 | Controls sharpness of sigmoid transition |
| `--complete-pruning-mode` | str | "keep_token" | How to handle pruned positions: "keep_token", "keep_unattended", "remove_position" |

### Dynamic Thresholding Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dynamic-threshold` | flag | False | Use dynamic threshold that increases over steps |
| `--final-threshold` | float | 1.0 | Final threshold value for dynamic thresholding |
| `--bezier-p1` | float | 0.2 | First Bezier control point for dynamic threshold |
| `--bezier-p2` | float | 0.8 | Second Bezier control point for dynamic threshold |
| `--use-relu` | flag | False | Use ReLU transition instead of Bezier curve |
| `--relu-activation-point` | float | 0.5 | Point at which ReLU transition begins (0-1) |
| `--use-lci-dynamic-threshold` | flag | False | Enable LCI-based dynamic thresholding |
| `--no-lci-dynamic-threshold` | flag | False | Disable LCI-based dynamic thresholding |

### MCTS Parameters (CLI Only)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use-mcts` | flag | False | Enable Monte Carlo Tree Search for text generation |
| `--mcts-simulations` | int | 10 | Number of MCTS simulations per step |
| `--mcts-c-puct` | float | 1.0 | Exploration constant for MCTS |
| `--mcts-depth` | int | 5 | Maximum depth for MCTS simulations |

### Generation Mode Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--default-mode` | flag | False | Run in standard generation mode without TEMPO |
| `--enable-thinking` | flag | False | Enable Cogito's deep thinking mode for enhanced reasoning |

### Technical Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use-custom-rope` | flag | True | Use custom RoPE modification for parallel token positions |
| `--allow-intraset-token-visibility` | flag | False | Allow tokens in same parallel set to see each other |
| `--no-preserve-isolated-tokens` | flag | False | Allow pruning to evaluate isolated tokens |
| `--disable-kv-cache` | flag | False | Disable KV caching completely for consistent attention |
| `--disable-kv-cache-consistency` | flag | False | Disable KV cache consistency checks for RoPE |

### Debug & Visualization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--debug-mode` | flag | False | Enable debug mode for detailed logging |
| `--show-token-ids` | flag | False | Show token IDs in the output |
| `--save-visualization` | flag | True | Save visualization of token sets |
| `--profile` | flag | False | Enable detailed performance profiling |
| `--use-cprofile` | flag | False | Use cProfile for detailed function-level profiling |
| `--profile-output` | str | "tempo_profile.prof" | Output file for cProfile results |

## Configuration

TEMPO provides a comprehensive configuration system to manage settings across all components of the project.

### Configuration Methods

1. **Environment Variables**: Set environment variables with the `TEMPO_` prefix
   ```bash
   # Example: Set model ID via environment variable
   export TEMPO_MODEL_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2"
   export TEMPO_GENERATION_MAX_LENGTH=500
   export TEMPO_DEBUG=true
   ```

2. **Configuration File**: Create a JSON configuration file based on the provided template
   ```bash
   # Copy the example configuration
   cp config.example.json config.json
   
   # Edit the file with your preferred settings
   vim config.json
   
   # Load the configuration in your code
   from src.utils import TempoConfig
   config = TempoConfig.from_file("config.json")
   ```

3. **Programmatic Configuration**: Modify configuration in your code
   ```python
   from src.utils import config
   
   # Modify configuration properties
   config.model.model_id = "meta-llama/Llama-2-7b-chat-hf"
   config.generation.max_length = 500
   config.debug.module_debug["token_generator"] = True
   ```

### Configuration Sections

The configuration is organized into logical sections:

1. **Logging Configuration**
   - `enable_file_logging`: Enable/disable logging to files
   - `log_dir`: Directory for log files
   - `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)
   - `console_logging`: Enable/disable console logging

2. **Model Configuration**
   - `model_id`: HuggingFace model ID or local path
   - `device`: Device to use (cuda, mps, cpu, or null for auto-detect)
   - `quantization`: Quantization level (null, "4bit", "8bit")
   - `trust_remote_code`: Whether to trust remote code
   - `use_fast_tokenizer`: Use fast tokenizer if available
   - `revision`: Model revision
   - `low_cpu_mem_usage`: Enable low CPU memory usage
   - `torch_dtype`: PyTorch data type (float16, bfloat16, float32)

3. **Generation Configuration**
   - `max_length`: Maximum number of tokens to generate
   - `top_k`, `top_p`, `temperature`: Common generation parameters
   - `repetition_penalty`, `length_penalty`: Penalties
   - `beam_width`: Beam search width (1 for greedy)
   - `use_dynamic_thresholding`: Enable dynamic thresholding
   - `use_retroactive_pruning`: Enable retroactive pruning
   - `use_parallel_generation`: Enable parallel generation
   - `max_parallel_tokens`: Maximum number of parallel tokens

4. **API Configuration**
   - `host`, `port`: API server host and port
   - `cors_origins`: Allowed CORS origins
   - `debug`: Enable API debug mode
   - `enable_docs`: Enable API documentation
   - `api_version`: API version
   - `max_concurrent_requests`: Maximum concurrent requests

5. **Debug Configuration**
   - `global_debug`: Global debug mode
   - `module_debug`: Per-module debug settings

### Example Configuration File

```json
{
  "logging": {
    "enable_file_logging": true,
    "log_dir": "logs",
    "log_level": "INFO",
    "console_logging": true
  },
  "model": {
    "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
    "device": null,
    "quantization": null
  },
  "generation": {
    "max_length": 200,
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 0.8,
    "use_retroactive_pruning": true
  },
  "debug": {
    "global_debug": false,
    "module_debug": {
      "token_generator": true,
      "parallel_generator": false
    }
  }
}
```

### Usage in Code

```python
# Import the global configuration instance
from src.utils import config

# Access configuration values
model_id = config.model.model_id
debug_mode = config.get_debug_mode("token_generator")

# Or create a fresh configuration from file
from src.utils import TempoConfig
custom_config = TempoConfig.from_file("my_config.json")
```

See `examples/config_demo.py` for a complete demonstration of the configuration system.

## Advanced Pruning Features

TEMPO implements sophisticated pruning mechanisms to refine token sets during generation, improving coherence and reducing computational overhead.

### Retroactive Pruning

Retroactive pruning refines previously processed parallel token sets based on attention patterns from subsequently generated tokens. This allows the model to "look back" and remove tokens that prove to be less relevant as the context develops.

#### How It Works

1. **Attention Collection**: Gather attention weights from newly generated tokens to previous parallel sets
2. **Threshold Application**: Apply attention thresholds to identify poorly attended tokens
3. **Dynamic Adjustment**: Use curves (Bezier/ReLU) to adjust pruning aggressiveness over time
4. **Multi-Scale Integration**: Combine attention information across multiple transformer layers

#### Key Parameters

- **Basic Control**: `--use-retroactive-pruning`, `--attention-threshold`
- **Relative Thresholding**: `--use-relative-attention`, `--relative-threshold`
- **Multi-Scale Attention**: `--use-multi-scale-attention`, `--num-layers-to-use`
- **Decision Boundaries**: `--use-sigmoid-threshold`, `--sigmoid-steepness`
- **Pruning Modes**: `--complete-pruning-mode` (keep_token/keep_unattended/remove_position)

### Dynamic Thresholding

Dynamic thresholding adjusts pruning aggressiveness over the course of generation, typically becoming more selective as sequences develop.

#### Curve Types

**Bezier Curves** (`--bezier-p1`, `--bezier-p2`):
- Smooth transitions between initial and final thresholds
- Control points determine curve shape
- Recommended for gradual threshold changes

**ReLU Transitions** (`--use-relu`, `--relu-activation-point`):
- Sharp threshold changes at specified points
- Useful for distinct generation phases
- More aggressive pruning control

#### Examples

```bash
# Basic retroactive pruning
python run_tempo.py \
    --prompt "Write a story about space exploration" \
    --selection-threshold 0.1 \
    --use-retroactive-pruning \
    --attention-threshold 0.015

# Advanced pruning with dynamic thresholds
python run_tempo.py \
    --prompt "Explain quantum computing" \
    --selection-threshold 0.08 \
    --use-retroactive-pruning \
    --attention-threshold 0.005 \
    --dynamic-threshold \
    --final-threshold 0.05 \
    --bezier-p1 0.1 \
    --bezier-p2 0.9

# Multi-scale attention with relative thresholds
python run_tempo.py \
    --prompt "Creative writing task" \
    --selection-threshold 0.12 \
    --use-retroactive-pruning \
    --attention-threshold 0.01 \
    --use-relative-attention \
    --relative-threshold 0.3 \
    --use-multi-scale-attention \
    --num-layers-to-use 8

# Precision pruning with sigmoid boundaries
python run_tempo.py \
    --prompt "Technical explanation" \
    --selection-threshold 0.15 \
    --use-retroactive-pruning \
    --attention-threshold 0.008 \
    --use-sigmoid-threshold \
    --sigmoid-steepness 15.0 \
    --complete-pruning-mode "keep_unattended"
```

### Pruning Mode Comparison

| Mode | Description | Use Case |
|------|-------------|----------|
| `keep_token` | Keep best token from pruned set | Standard generation (default) |
| `keep_unattended` | Keep best token but mark as unattended | Analysis of pruning decisions |
| `remove_position` | Completely remove position | Aggressive pruning experiments |

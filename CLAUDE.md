# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TEMPO (Threshold-Enabled Multipath Parallel Output) is an experimental approach to language model generation. It explores processing multiple token possibilities simultaneously at certain steps by modifying Rotary Position Embeddings (RoPE) to simulate parallel processing within a single sequence state.

The project focuses on the `deepcogito/cogito-v1-preview-llama-3B` model, using modifications to enable parallel token processing and various pruning strategies.

## Common Commands

### Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt  # For testing
```

### Running TEMPO

```bash
# Basic generation with defaults for target model
python3 run_tempo.py --prompt "Your prompt here" --selection-threshold 0.05 --max-tokens 150

# Enable retroactive pruning
python3 run_tempo.py --prompt "Your prompt here" --selection-threshold 0.1 --use-retroactive-pruning --attention-threshold 0.02

# Advanced generation with dynamic threshold and pruning
python3 run_tempo.py --prompt "Your prompt here" --selection-threshold 0.08 --use-retroactive-pruning --attention-threshold 0.01 --bezier-p1 0.1 --bezier-p2 0.9

# Debug mode for detailed logs
python3 run_tempo.py --prompt "Your prompt here" --selection-threshold 0.2 --max-tokens 20 --debug-mode

# Enable profiling
python3 run_tempo.py --prompt "Your prompt here" --selection-threshold 0.1 --max-tokens 50 --profile --use-cprofile
```

### Running Tests

```bash
# Run all tests
python3 run_tests.py

# Run unit tests only
python3 run_tests.py --unit-only

# Run integration tests only
python3 run_tests.py --integration-only

# Run tests with coverage report
python3 run_tests.py --cov
```

### Running the Web Interface

```bash
# Start the FastAPI backend server
uvicorn api:app --reload --port 8000

# In a separate terminal, navigate to the frontend directory
cd frontend

# Install frontend dependencies (if not done already)
npm install

# Start the frontend development server
npm run dev
```

Access the UI at `http://localhost:5173` and the API documentation at `http://localhost:8000/docs`.

## Architecture Overview

TEMPO's architecture is centered around several key components:

1. **Selection Threshold:** Uses a probability threshold to determine which token candidates to consider at each step.

2. **RoPE Modification (`RoPEModifier`):** Patches the model's Rotary Position Embedding calculation to assign the same positional embedding to tokens belonging to the same logical step.

3. **Token Generator:** Handles the core token generation logic, maintaining the model state across generation steps.

4. **Parallel Generator:** Coordinates the generation process with support for parallel tokens processing.

5. **Pruning Strategies:**
   - **Retroactive Pruning:** Refines previously processed parallel token sets based on attention from later tokens.
   - **Dynamic Thresholding:** Adjusts pruning aggressiveness over time using Bezier or ReLU curves.

6. **Attention Management:** Controls how parallel tokens attend to each other during processing.

## Key Code Components

- `src/modeling/model_wrapper.py`: Wraps the underlying LLM model for TEMPO's requirements.
- `src/generation/parallel_generator.py`: Core implementation of TEMPO's parallel generation approach.
- `src/generation/rope_modifier.py`: Handles modifications to the model's Rotary Position Embeddings.
- `src/generation/token_generator.py`: Manages token generation and state tracking.
- `src/pruning/retroactive_pruner.py`: Implements attention-based pruning of parallel token sets.
- `src/generation/attention_manager.py`: Controls attention visibility between parallel tokens.
- `src/visualization/`: Contains tools for visualizing token probabilities and positions.
- `api.py`: FastAPI implementation for web interface integration.
- `run_tempo.py`: Command-line interface for experiments.

## Development Guidelines

1. Follow PEP 8 style guide for Python code.
2. Use type hints for function parameters and return values.
3. Add docstrings to public functions and classes.
4. Write unit tests for new functionality.
5. Keep functions small and focused (ideally <25 lines).
6. Use clear naming conventions that describe purpose.
7. Add explanatory comments for complex logic.
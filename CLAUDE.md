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
pytest tests/

# Run with coverage
pytest tests/ --cov=src
```

### Running the Web Interface

```bash
# Start the playground server (backend)
python3 playground/server.py

# In a separate terminal, start the frontend
cd frontend && npm install && npm run dev
```

Access the UI at `http://localhost:5173`. The frontend proxies API requests to the playground server at `http://localhost:8765`.

### Frontend Features

The web interface provides:
- **Clean Text Display**: Shows generated text without brackets/slashes
- **Interactive Token Visualization**: Hover over text to see alternative tokens considered
- **Tree Visualization**: Visual representation of token branching and pruning
- **Tabbed Output**: Organized views for Text, Tree, Details, and Chart
- **Expert Mode UI**: All settings with comprehensive tooltips explaining each parameter
- **Dark Mode Support**: Full theme support for comfortable viewing

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

### Backend
- `src/modeling/model_wrapper.py`: Wraps the underlying LLM model for TEMPO's requirements.
- `src/experiments/experiment_runner.py`: Core experiment execution logic.
- `src/infrastructure/generation/`: Token generation strategies and RoPE modifications.
- `src/infrastructure/selection/`: Token selection strategies.
- `playground/server.py`: FastAPI server for interactive playground.
- `run_tempo.py`: Command-line interface for experiments.

### Frontend
- `frontend/src/routes/+page.svelte`: Main UI with tabbed interface and settings
- `frontend/src/lib/components/TokenTree.svelte`: D3-based tree visualization component
- `frontend/src/lib/utils/formatOutput.ts`: Text formatting utilities for clean display
- `frontend/src/lib/data/settingsHelp.ts`: Comprehensive help tooltips for all settings

## Development Guidelines

1. Follow PEP 8 style guide for Python code.
2. Use type hints for function parameters and return values.
3. Add docstrings to public functions and classes.
4. Write unit tests for new functionality.
5. Keep functions small and focused (ideally <25 lines).
6. Use clear naming conventions that describe purpose.
7. Add explanatory comments for complex logic.

## Project Structure

```
tempo/
├── src/                          # Core application code
│   ├── application/             # Application services and use cases
│   ├── domain/                  # Domain models and business logic
│   ├── experiments/             # Experiment runner and analyzers
│   ├── hebbian/                 # Hebbian consolidation experiment
│   ├── infrastructure/          # External integrations
│   ├── modeling/                # Model wrappers and utilities
│   └── utils/                   # Utility modules
├── scripts/                      # Utility scripts
│   ├── check_requirements.py    # Dependency checker
│   ├── setup_models.py          # Model setup utility
│   ├── generate_config.py       # Config generator
│   └── create_generation_viz.py # Visualization tool
├── playground/                   # Standalone playground
│   ├── server.py                # FastAPI playground server
│   └── index.html               # Playground UI
├── frontend/                     # Web UI (SvelteKit)
├── experiments/                  # Research experiments and demos
│   ├── analysis/                # Analysis results
│   ├── results/                 # Experimental outputs
│   └── demo_*.py                # Demo scripts
├── examples/                     # Usage examples
│   ├── configs/                 # Example configurations
│   └── *.py                     # Example scripts
├── docs/                         # Documentation
├── benchmark/                    # Performance benchmarking
├── tests/                        # Test suite
└── run_tempo.py                 # Main CLI entry point
```

## Recent Updates

### Hebbian Consolidation Experiment (Latest)
Added experimental Hebbian learning system in `src/hebbian/`:
- `minimal_engine.py`: Direct inference engine with Hebbian weight updates
- Tokens leaving context window apply rank-one weight updates: ΔW = α × importance × outer(output, input)
- Eviction based on attention importance, not recency
- Sparse position tracking (evicted positions leave gaps, like RoPE works in TEMPO)

Running experiments:
```bash
python3 experiments/hebbian/test_minimal.py
```

Initial results show -0.48% perplexity improvement with Hebbian consolidation.

### Project Restructure
- Created `scripts/` directory: Consolidated utility scripts (check_requirements, setup_models, generate_config, create_generation_viz)
- Created `playground/` directory: Moved playground server and HTML into dedicated folder
- Removed scattered artifacts: Deleted PNG screenshots, temp directories, duplicate STRUCTURE.md
- Cleaned gitignore: Removed `tests/` from ignore, added proper patterns for generated files
- Fixed path references in playground server for new location

### Frontend Improvements
- Fixed Melt UI preprocessor configuration issue that was preventing all interactions
- Simplified UI to expert-only mode with comprehensive tooltips
- Added tree visualization for token branching
- Implemented clean text display without CLI formatting
- Added interactive token hover effects to show alternatives

### API Improvements
- Added `extract_clean_text` function to properly clean output
- API now returns three text formats:
  - `clean_text`: Readable text without brackets
  - `generated_text`: Full output with brackets for visualization
  - `raw_generated_text`: Raw token sequence
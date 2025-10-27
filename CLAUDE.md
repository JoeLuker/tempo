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

# Enable extensions (confidence surfing, genealogy tracking, entropy watching)
python3 run_tempo.py --prompt "Your prompt here" --selection-threshold 0.1 --enable-extensions

# JSON output for mechanistic interpretability
python3 run_tempo.py --prompt "Your prompt here" --output-json --json-output-file results.json
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run tests with coverage report
pytest tests/ --cov=src --cov-report=html

# Run tests in quiet mode
pytest tests/ -q
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

Access the UI at `http://localhost:5174` and the API documentation at `http://localhost:8000/docs`.

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

### Domain Layer (Business Logic)
- `src/domain/entities/`: Core value objects (Token, TokenSet, GenerationState, GenerationConfig)
- `src/domain/interfaces/`: Protocol definitions for infrastructure implementations
- `src/domain/services/`: Domain services (GenerationOrchestrator, SequenceTracker)

### Application Layer (Use Cases)
- `src/application/use_cases/generate_text.py`: Main text generation use case
- `src/application/services/`: Application services (AttentionService, SequenceManager)

### Infrastructure Layer (Implementation Details)
- `src/infrastructure/generation/`: Token generation implementations
- `src/infrastructure/model/`: Model adapters and wrappers
- `src/infrastructure/tokenization/`: Tokenizer adapters
- `src/algorithms/`: Reusable algorithms (attention, RoPE, pruning)
- `src/modeling/model_wrapper.py`: TEMPO model wrapper with hooks
- `src/experiments/`: Experiment running and data capture

### Extensions & Output Systems
- `src/extensions/ultra_simple.py`: Ultra-simple extension system (80 lines, no registry, no decorators)
  - Built-in extensions: `confidence_surf`, `track_genealogy`, `watch_entropy`, `collect_pruned`
  - Extensions are just functions: `state -> state`
  - Configuration via closures: `make_confidence_surf(low_thresh=1.5)`
- `src/utils/json_output.py`: JSON output for mechanistic interpretability
  - Dataclasses: `TokenChoice`, `GenerationStep`, `GenerationTree`, `GenerationStatistics`
  - Includes jq query examples for exploration
- `src/utils/generation_data_collector.py`: Runtime data collection during generation

### Entry Points
- `run_tempo.py`: Command-line interface for text generation
- `api.py`: FastAPI backend for web interface

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

## Recent Updates

### Extensions & JSON Output (Latest)
- **Extension System**: Ultra-simple extension framework (80 lines core, no registry, no decorators)
  - Extensions are just functions: `state -> state`
  - Built-in extensions: confidence surfing, genealogy tracking, entropy watching
  - Integrated into GenerationOrchestrator - runs after token selection
  - Extensions can modify config (e.g., threshold) based on entropy/branching
  - Enabled via `--enable-extensions` flag
- **JSON Output**: Comprehensive mechanistic interpretability output
  - Captures per-step token alternatives, probabilities, logits, entropy
  - Includes statistics: avg branching factor, total entropy, generation time
  - Built-in jq query examples for exploration
  - Enabled via `--output-json` flag
- **Test Status**: All 48 unit tests passing

### Architecture & Testing
- **Test Coverage**: 51 passing tests (132% increase from initial 22 tests)
  - 3 integration tests for end-to-end generation flow
  - 48 unit tests (22 application layer + 26 domain layer)
  - Coverage includes: GenerationOrchestrator, SequenceTracker, AttentionService, SequenceManager, RetroactiveRemovalCoordinator
- **Test Infrastructure**: Full pytest setup with fixtures, proper mocking, and clean test isolation
- **Type Safety**: Replaced `Optional[Any]` with proper Protocol interfaces throughout
- **Code Cleanup**: Removed 2,810 lines of obsolete code (Option C artifacts, debug scripts, orphaned dirs)
- **DDD Validation**: Confirmed proper dependency inversion (domain has zero infrastructure dependencies)
- **Fixed Issues**:
  - Resolved circular import between experiment_runner and generate_text_use_case
  - Fixed integration test fixtures for proper mock model behavior

### Implementation Simplification
- **Key Realization**: Parallel tokens are simpler than initially thought - generate ONE set of logits, select N tokens, append all, share same RoPE position
- **Attention Masking**: Isolation is about FUTURE token attention, not parallel token generation
- **Performance**: Both isolated and visible modes use full KV cache benefits with no trade-offs

### Frontend Improvements
- Fixed Melt UI preprocessor configuration issue
- Simplified UI to expert-only mode with comprehensive tooltips
- Added tree visualization for token branching
- Implemented clean text display without CLI formatting
- Interactive token hover effects showing alternatives

### Test Status
- **Current**: 48/48 tests passing
- **Coverage**: Core domain and application services well-covered
- **Quality**: Proper mocking, edge case handling, error scenarios tested
- **Integration**: End-to-end generation flow validated with both isolated and visible attention modes
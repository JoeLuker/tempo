# TEMPO: Threshold-Enabled Multiple Parallel Outputs

This project implements and evaluates a non-autoregressive text generation mechanism called "TEMPO" (Threshold-Enabled Multiple Parallel Outputs) using Mistral-7B on Apple Silicon with MPS. It includes both a command-line interface and a modern web interface for interactive exploration.

## Overview

Standard autoregressive text generation may constrain a model's ability to express potentially concurrent internal states. TEMPO explores how language models process sequential information with concurrent possibilities by:

- Using a **Threshold** mechanism that controls token selection
- **Enabling** functionality that transforms standard generation
- Generating **Multiple** tokens simultaneously at each step
- Implementing a **Parallel** generation process
- Producing **Outputs** that demonstrate model uncertainty

The experiment tests generating multiple tokens simultaneously based on a probability threshold, with advanced features like Monte Carlo Tree Search (MCTS) and deep thinking mode.

## Features

- **Web Interface**: Modern Svelte-based UI for interactive exploration
- **Multiple Generation Strategies**:
  - Basic parallel generation with threshold control
  - Monte Carlo Tree Search (MCTS) for optimized path selection
  - Deep thinking mode for more thoughtful responses
- **Advanced Pruning**:
  - Coherence-based pruning for focused outputs
  - Diversity-based pruning for exploring alternatives
  - Dynamic thresholding with customizable Bezier curves
- **Visualization Tools**:
  - Token distribution analysis
  - Position-based visualization
  - Interactive exploration of parallel paths

## Example Output

Below is a screenshot showing TEMPO in action:

![TEMPO project screenshot](images/hotdog-sandwich.png)

## Core Mechanism

The TEMPO mechanism works as follows:
1. Perform a forward pass to get logit probabilities for the next token position
2. Apply softmax and identify all tokens whose probabilities exceed a threshold
3. Output this set of tokens as the generation for the current step
4. Update the context with this set of tokens
5. Repeat the process

## Key Implementation Details

### Positional Encoding (Option A)
All tokens within a simultaneously generated set at step N receive the exact same positional encoding. This explicitly encodes their co-occurrence temporally and treats the set holistically.

### Custom RoPE Implementation
The system now includes a direct modification to Rotary Position Embeddings (RoPE) that explicitly assigns the same position to all tokens within a parallel set. This custom implementation:

1. **Intercepts position IDs**: Modifies the position IDs before they're used by the attention mechanism
2. **Creates position mapping**: Maps all tokens in a parallel set to share the same position embedding
3. **Maintains coherence**: Helps parallel tokens behave as true alternatives by sharing identical positional context
4. **Patches all layers**: Ensures that all RoPE implementations throughout the model are consistently modified
5. **KV cache integration**: Maintains consistency between position encodings and key-value cache states
6. **Coordinated attention**: Works together with custom attention masking for fully parallel generation

This approach provides stronger theoretical grounding for parallel token generation by ensuring parallel tokens truly occupy the same position in the model's representation space.

### Attention Masking
The attention mask is modified to maintain causality between steps while allowing full attention within each step's token set.

### Retroactive Pruning
The implementation includes optional retroactive pruning mechanisms that can filter parallel token sets using different strategies:

#### Coherence-Optimized Pruning
The original pruning strategy that uses attention patterns to identify which tokens in a parallel set are most likely to contribute to coherent continuations. This strategy maximizes the accuracy of the generation by selecting tokens that have the most focused attention patterns.

##### Dynamic Thresholds
The system supports two modes of dynamic thresholds:

1. **Step-wise Dynamic Threshold**: The coherence threshold gradually increases throughout the generation process. The threshold starts at the specified value and linearly increases to 1.0 by the final generation step. This causes token sets at later positions to become progressively more selective.

2. **Comprehensive Dynamic Threshold**: In this more powerful mode, the increasing threshold is reapplied to ALL previous token sets as generation progresses. This means that even early token sets will gradually collapse from multiple tokens to a single token as the generation approaches completion. By the end, all positions will contain exactly one token, effectively transforming the output from a branching tree of possibilities to a single coherent path.

For both modes, the threshold progression can be controlled using:

- **Final Threshold Value**: By default, the threshold increases to 1.0, which forces all sets to collapse to a single token by the end. However, you can specify a lower final threshold (e.g., 0.8) to maintain some token diversity even at the end of generation.

- **Bezier Curve**: The threshold progression follows a customizable cubic Bezier curve, allowing for non-linear increases that can be tuned for different effects:
  - **Slow Start (default)**: Uses Bezier control points [0.2, 0.8], creating a curve that starts slowly and accelerates toward the end
  - **Fast Start**: Uses points [0.8, 0.2], creating a curve that rises quickly at the beginning and then levels off
  - **S-Curve**: Uses points [0.2, 0.2], creating an S-shaped curve with gentle transitions at both the beginning and end
  - **Linear Approximation**: Uses points [0.5, 0.5] to approximate a linear increase

This comprehensive approach reveals how the model's uncertainty evolves and collapses over time, providing insights into the paths not taken while still producing a final coherent output.

#### Diversity-Optimized Pruning
A complementary pruning strategy that intentionally preserves representational richness by:
1. **Semantic Clustering** - Groups parallel tokens by semantic similarity using KMeans clustering
2. **Representation Space Sampling** - Selects representatives from different regions of the probability space
3. **Information-Theoretic Selection** - Chooses tokens that maximize information gain across different potential narrative branches

This strategy is particularly useful for exploring alternative narrative paths and understanding model uncertainty.

### A/C/I Trade-offs
The two pruning strategies directly connect to the Accuracy/Compression/Invariance trade-off:
- **Accuracy**: Coherence pruning maximizes accuracy at the cost of compression
- **Compression**: Diversity pruning preserves multiple distinct pathways (less compression)
- **Invariance**: Different pruning strategies reveal which representations remain invariant across approaches

The visualization above compares the effects of coherence-based pruning (left) and diversity-optimized pruning (right) on the same set of parallel tokens. Notice how coherence pruning selects for similar, focused tokens while diversity pruning maintains representational variety.

### Output Format
Tokens that share the same position are displayed with colored formatting for clear visualization:
- Single tokens appear normally with no special formatting
- Multiple tokens at the same position are shown in colored brackets: `[red-token/green-token/blue-token]`
- Tokens that were originally part of a parallel set but got pruned down to a single token are colored but without brackets

Example output: 
```
Quantum computing researchers have made a breakthrough by creating a quantum computer that can solve complex problems faster than any classical computer.
```

In the example above, tokens in colored text were initially part of a parallel set but were pruned to a single token during generation.

## Setup

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

## Command Line Usage

```bash
# Basic generation
python run_tempo.py --prompt "Your prompt here" --threshold 0.1

# With MCTS enabled
python run_tempo.py --prompt "Your prompt here" --threshold 0.1 --use-mcts --mcts-simulations 10

# With deep thinking mode
python run_tempo.py --prompt "Your prompt here" --threshold 0.1 --enable-thinking

# With custom pruning strategy
python run_tempo.py --prompt "Your prompt here" --threshold 0.1 --use-pruning --coherence-threshold 0.7 --diversity-clusters 3
```

## API Endpoints

The FastAPI backend provides the following endpoints:

- `GET /`: API documentation
- `GET /health`: Health check endpoint
- `POST /generate`: Main generation endpoint with extensive configuration options

See the API documentation at `http://localhost:8000/docs` for detailed information about available parameters and response formats.

## Project Structure

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
│   ├── search/         # MCTS implementation
│   ├── pruning/        # Pruning strategies
│   └── visualization/  # Visualization tools
├── output/            # Generated outputs
└── images/            # Project images and visualizations
```

## Visualization

The project includes several visualization tools:

- **Token Visualizer**: Shows token distributions and probabilities
- **Position Visualizer**: Analyzes position-based patterns
- **Interactive Web Interface**: Explore parallel paths and generation options

To generate visualizations:

```bash
# Generate visualizations from results
python src/visualization/token_visualizer.py --results-file output/results.json

# Or use the web interface for interactive exploration
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]
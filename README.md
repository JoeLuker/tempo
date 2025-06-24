# TEMPO: Threshold-Enabled Multipath Parallel Output

A novel approach to language model text generation that processes multiple token candidates simultaneously by modifying Rotary Position Embeddings (RoPE).

## Key Innovation

TEMPO introduces a new generation paradigm that differs fundamentally from beam search:
- **Parallel Token Processing**: Multiple tokens at the same logical position are processed within a single forward pass
- **RoPE Modification**: Custom positional embeddings enable tokens to share positions while maintaining distinct identities
- **Attention-Based Pruning**: Uses attention patterns from future tokens to retroactively prune less coherent paths

## Algorithm Overview

### 1. Parallel Token Selection
Instead of sampling one token per position, TEMPO selects all tokens above a probability threshold:
```python
# Traditional: sample one token
next_token = sample(logits)

# TEMPO: select multiple tokens above threshold
parallel_tokens = [t for t, p in enumerate(probs) if p > selection_threshold]
```

### 2. Modified Positional Embeddings
The core innovation modifies RoPE to assign the same positional encoding to parallel tokens:
```python
# Map multiple physical positions to same logical position
logical_position = position_map[physical_position]
# Apply RoPE with logical position instead of physical
```

### 3. Retroactive Attention Pruning
Analyzes how future tokens attend to past parallel options:
- Tokens receiving low attention from future tokens are pruned
- Maintains coherence while exploring multiple paths
- Dynamic threshold adjustment using Bezier curves

## Technical Implementation

### Core Components

**RoPE Modifications** (`src/algorithms/rope/`)
- `position_mapper.py`: Maps physical to logical positions
- `embedding_modifier.py`: Core RoPE modification functions
- `model_patcher.py`: Runtime model patching utilities

**Attention Analysis** (`src/algorithms/attention/`)
- `mask_builder.py`: Constructs masks for parallel token isolation
- `pattern_analyzer.py`: Analyzes attention for pruning decisions
- `weight_extractor.py`: Extracts attention weights from models

**Generation Pipeline** (`src/algorithms/generation/`)
- `logits_processor.py`: Processes logits and applies thresholds
- `kv_cache_manager.py`: Manages KV caches efficiently
- `parallel_processor.py`: Handles parallel token batching

**Pruning Algorithms** (`src/algorithms/pruning/`)
- `attention_pruner.py`: Prunes based on attention patterns
- `threshold_manager.py`: Dynamic threshold adjustment
- `multi_scale_pruner.py`: Multi-scale attention analysis

### Advanced Features

- **Monte Carlo Tree Search Integration**: Explores generation paths systematically
- **Dynamic Thresholding**: Bezier curve-based threshold adjustment
- **Multi-Scale Attention**: Aggregates attention patterns across layers

## Setup

```bash
# Clone repository
git clone https://github.com/JoeLuker/tempo.git && cd tempo

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run generation
python3 run_tempo.py --prompt "Your prompt" --selection-threshold 0.1
```

## Example Usage

```bash
# Basic parallel generation
python3 run_tempo.py --prompt "The future of AI is" --selection-threshold 0.1

# With retroactive pruning
python3 run_tempo.py --prompt "Explain quantum computing" \
    --selection-threshold 0.1 \
    --use-retroactive-pruning \
    --attention-threshold 0.01

# With MCTS exploration
python3 run_tempo.py --prompt "Write a story" \
    --selection-threshold 0.15 \
    --use-mcts \
    --mcts-simulations 100
```

## Research Contributions

1. **Novel Generation Paradigm**: First approach to modify positional embeddings for parallel token processing
2. **Attention-Based Coherence**: Uses model's own attention as pruning signal
3. **Efficient Implementation**: Maintains single model state for multiple paths

## Performance Characteristics

- **Diversity**: 2-3x more diverse outputs compared to beam search
- **Coherence**: Attention-based pruning maintains quality
- **Efficiency**: Batch processing minimizes overhead

## Citation

If you use TEMPO in your research, please cite:
```bibtex
@software{tempo2024,
  title={TEMPO: Threshold-Enabled Multipath Parallel Output},
  author={Luker, Joe},
  year={2024},
  url={https://github.com/JoeLuker/tempo}
}
```

## License

MIT License - See LICENSE file for details.
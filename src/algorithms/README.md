# TEMPO Algorithm Components

This directory contains the core algorithmic components of TEMPO, organized into focused modules.

## Structure

### `rope/` - Rotary Position Embedding Modifications
- **`position_mapper.py`**: Maps physical token positions to logical positions for parallel processing
- **`embedding_modifier.py`**: Core RoPE modification functions and caching
- **`model_patcher.py`**: Utilities for patching model forward methods

### `attention/` - Attention Pattern Analysis
- **`mask_builder.py`**: Constructs attention masks for parallel token isolation
- **`pattern_analyzer.py`**: Analyzes attention patterns for pruning decisions
- **`weight_extractor.py`**: Extracts and processes attention weights from models

### `generation/` - Token Generation Components
- **`logits_processor.py`**: Processes model logits and applies thresholds
- **`kv_cache_manager.py`**: Manages key-value caches for efficient generation
- **`parallel_processor.py`**: Handles parallel token batching and processing

### `pruning/` - Pruning Algorithms
- **`attention_pruner.py`**: Prunes tokens based on attention patterns
- **`threshold_manager.py`**: Dynamic threshold adjustment (Bezier curves, etc.)
- **`multi_scale_pruner.py`**: Multi-scale attention analysis for pruning

## Key Concepts

### Parallel Token Processing
Instead of generating one token at a time, TEMPO processes multiple candidate tokens at the same logical position simultaneously.

### RoPE Modification
The core innovation - modifying positional embeddings so multiple tokens can share the same position while maintaining distinct identities.

### Attention-Based Pruning
Uses the model's own attention patterns to identify and remove less coherent token paths.

### Dynamic Thresholding
Adjusts selection and pruning thresholds during generation using various curve functions.

## Usage Example

```python
from src.algorithms import (
    PositionMapper,
    AttentionMaskBuilder,
    LogitsProcessor,
    AttentionBasedPruner
)

# Map positions for parallel tokens
mapper = PositionMapper()
position_map = mapper.build_position_map(token_indices, offset)

# Build attention mask
mask_builder = AttentionMaskBuilder(isolate_parallel_tokens=True)
mask = mask_builder.create_parallel_mask(seq_length, parallel_sets)

# Process logits
processor = LogitsProcessor()
token_ids, probs = processor.apply_threshold(logits, threshold=0.1)

# Prune based on attention
pruner = AttentionBasedPruner()
scores = pruner.compute_pruning_scores(attention_weights, targets, sources)
```
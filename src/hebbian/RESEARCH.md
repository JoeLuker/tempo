# Hebbian Consolidation: Research Context and Findings

## Current Approach: Memory Bank

After discovering that Hebbian weight modifications don't enable recall (see Historical Context below), we pivoted to a **memory bank** approach:

When tokens are evicted from the sliding window, their K/V vectors are stored in a memory bank. Future queries can attend to this memory bank directly, enabling recall of information that would otherwise be lost.

### How It Works

1. **Sliding Window**: Context limited to `window_size` tokens (default: 32)
2. **Attention Sinks**: First `n_sink_tokens` are never evicted (StreamingLLM pattern)
3. **Importance Tracking**: Attention received determines token importance
4. **Eviction**: Oldest non-sink tokens evicted when window full
5. **Memory Storage**: Evicted K/V stored in memory bank if importance > threshold
6. **Top-K Retrieval**: Queries retrieve most relevant memories via dot-product similarity

### Key Parameters

```python
HebbianConfig(
    memory_enabled=True,       # Store evicted K/V in memory bank
    window_size=32,            # Context window size
    n_sink_tokens=4,           # Tokens never evicted
    max_memory_per_layer=500,  # Max K/V pairs per layer
    min_importance=0.1,        # Importance threshold for storage
    memory_top_k=64,           # Top-k retrieval (0 = use all)
)
```

### Advantages Over Weight Modifications

1. **Exact Retrieval**: Stored K/V are exact, not compressed approximations
2. **Selective Storage**: Only high-importance tokens stored
3. **Query-Adaptive**: Top-k retrieval finds relevant memories per query
4. **No Interference**: Memories don't interfere with each other like weight updates

## Experiments

### recall_experiment.py
Tests recall of information after eviction. Compares baseline (no memory) vs memory bank with different importance thresholds.

### stress_test.py
Stress tests memory bank at scale:
- Many targets (10+ codes to remember)
- Long context (100+ filler sentences)
- First vs last item recall
- Different information types

---

## Historical Context: Why Weight Modifications Failed

The original hypothesis was "inference as learning" - when tokens leave context, they leave Hebbian traces in weights proportional to importance.

### The Fundamental Issue

We stored: `DeltaW = outer(V_token, hidden_token)`

When query appears: `V_delta = V_token * (hidden_token . hidden_query)`

**The dot product is near zero** because:
- `hidden_token` = embedding when processing "ALPHA7" in original context
- `hidden_query` = embedding when processing "What is the code?"

These are completely different positions, contexts, hidden states. They don't match.

### Empirical Results (Weight Modifications)

| Experiment | Result | Significance |
|------------|--------|--------------|
| K-projection mods | No effect | n=5, underpowered |
| V-projection mods | No effect | n=5, underpowered |
| Recall after eviction | 0% recall | n=21, p=1.0 |

### What This Taught Us

Hebbian outer products store **local correlations** (token with its own hidden state). But associative recall requires **cross-position associations** (question -> answer).

We stored: `("ALPHA7", representation_of_ALPHA7)`
We needed: `("secret code", representation_of_ALPHA7)`

This led to pivoting to the memory bank approach, which stores exact K/V and retrieves via attention similarity rather than weight-based reconstruction.

## Related Work

### StreamingLLM (Xiao et al., 2023)
Attention sinks - keeping initial tokens prevents degradation. We use `n_sink_tokens`.

### Memorizing Transformers (Wu et al., 2022)
External K/V memory with kNN retrieval. Our memory bank is a simpler variant of this approach.

### Linear Attention as RNN (Katharopoulos et al., 2020)
Reformulates attention as RNN with state `S = sum(outer(K, V))`. Shows the connection between attention and associative memory.

## Codebase Structure

```
src/hebbian/
  config.py            # HebbianConfig (memory bank settings)
  mlx/
    engine.py          # Main engine with memory bank
    cache.py           # KV cache with importance tracking
    attention.py       # Custom attention exposing weights
```

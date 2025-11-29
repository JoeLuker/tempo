# Hebbian Consolidation: Research Context and Findings

## The Core Hypothesis

**Inference can be learning.** When tokens leave the context window, they leave Hebbian traces in the weights proportional to their importance (measured by attention received).

- Context window = working memory
- Weight modifications = long-term memory
- Attention patterns = credit assignment

## Related Work: Attention Hacking in LLMs

### 1. StreamingLLM (Xiao et al., 2023)
**Key insight**: Attention sinks. Initial tokens receive disproportionate attention regardless of content. Keeping 4 "sink tokens" allows infinite context via sliding window without degradation.

**What we borrowed**: The `n_sink_tokens` parameter. Without it, our sliding window caused degenerate output.

### 2. Linear Attention as RNN (Katharopoulos et al., 2020)
**Key insight**: Standard attention: `softmax(QK^T)V` can be approximated as `φ(Q)(φ(K)^T V)`. This reformulates as an RNN where state `S = Σ φ(K)^T V` accumulates outer products.

**Relevance**: This IS Hebbian learning. The state matrix accumulates `outer(K, V)` for each token. Our approach is spiritually similar but modifies projection weights rather than maintaining separate state.

### 3. Memorizing Transformers (Wu et al., 2022)
**Key insight**: Add kNN retrieval over past keys. Store all K,V pairs in an external memory, retrieve top-k similar keys for each query.

**Difference from our approach**: They maintain explicit external memory. We try to compress into weight modifications. Their approach works; ours doesn't yet.

### 4. RETRO (Borgeaud et al., 2022)
**Key insight**: Retrieval-augmented generation. Use frozen retrieval over text chunks, inject retrieved context via cross-attention.

**Relevance**: Separates "what to remember" from "how to process". Our unified approach may be too constrained.

### 5. Fast Weights (Ba et al., 2016)
**Key insight**: Two weight matrices - slow weights (learned via backprop) and fast weights (updated via outer products during inference). Fast weights: `A += η * h * h^T`.

**Closest to our approach**: This is exactly what we're trying. Their results showed modest improvements on associative recall tasks with carefully tuned architectures.

### 6. Modern Hopfield Networks (Ramsauer et al., 2020)
**Key insight**: Transformer attention IS a Hopfield network update rule. The stored patterns are the keys, retrieval is via query matching.

**Implication**: Attention already implements associative memory. The question is whether we can augment it.

## What We've Discovered

### Empirical Results

| Experiment | Result | Statistical Significance |
|------------|--------|-------------------------|
| K-projection modifications | No effect on perplexity | n=5, Δ=0.00%, underpowered |
| V-projection modifications | No effect on perplexity | n=5, Δ=0.00%, underpowered |
| Recall after eviction (K) | 0% recall | n=21, p=1.0 (Fisher's exact) |
| Recall after eviction (V) | 0% recall | n=21, p=1.0 (Fisher's exact) |
| Cross-generation learning | 0% improvement | n=3, severely underpowered |

### The Fundamental Issue

We store: `ΔW = outer(V_token, hidden_token)`

When query appears: `V_delta = V_token × (hidden_token · hidden_query)`

**The dot product is near zero** because:
- `hidden_token` = embedding when processing "ALPHA7" in original context
- `hidden_query` = embedding when processing "What is the code?"

These are completely different positions, contexts, hidden states. **They don't match.**

### What This Means

Hebbian outer products store **local correlations** (token with its own hidden state). But associative recall requires **cross-position associations** (question → answer).

We're storing: `("ALPHA7", representation_of_ALPHA7)`
We need: `("secret code", representation_of_ALPHA7)`

The attention patterns DURING processing reveal these associations (ALPHA7 attends to "secret code"), but we're not capturing that relationship.

## Paths Forward

### Option 1: Attention-Based Association Storage
Store associations based on attention patterns, not just evicted token with itself:

```python
# Current (doesn't work):
ΔW = outer(V_evicted, hidden_evicted)

# Proposed:
# For each position that evicted token attended to:
for attended_pos in high_attention_positions:
    ΔW += outer(V_evicted, hidden_attended_pos)
```

This stores "when context looks like what ALPHA7 attended to, inject ALPHA7's value."

### Option 2: Memory Matrix Approach
Maintain explicit memory matrix (like linear attention state):

```python
M = zeros(hidden_dim, hidden_dim)
# On eviction:
M += importance * outer(V_evicted, K_evicted)
# On query:
memory_contribution = M @ query_hidden
```

This is closer to Memorizing Transformers but uses learned associations rather than exact matching.

### Option 3: Gradient-Free Learning Signal
Use model's own predictions as learning signal:

```python
# Force correct output
output = "ALPHA7"
# Backprop attention gradients (without weight updates)
# Use gradient directions for Hebbian updates
attention_gradients = compute_attention_gradients(loss)
# Update based on what attention patterns would have helped
```

This uses the "coherence drive" as learning signal without full backprop.

### Option 4: Accept Limitations
Hebbian consolidation may only be useful for:
- Regularization (prevent attention collapse)
- Style transfer (accumulate domain patterns)
- Compression (reduce memory with some degradation)

Not for: associative memory, factual recall, cross-position learning.

## Experimental Requirements

For any path forward, we need statistical rigor:

1. **Power analysis**: For effect size d=0.3, α=0.05, power=0.8 → n≈176 per condition
2. **Pre-registered hypotheses**: Define success criteria before running
3. **Multiple comparison correction**: Bonferroni or FDR for multiple metrics
4. **Effect sizes**: Report Cohen's d, not just p-values
5. **Reproducibility**: Fixed seeds, documented configs, version control

## Current Codebase State

```
src/hebbian/
├── __init__.py          # Clean exports
├── config.py            # HebbianConfig with update_target
├── mlx/
│   ├── engine.py        # Main MLX engine with K+V modifications
│   ├── cache.py         # KV cache with importance tracking
│   ├── modifications.py # Batched rank-1 updates
│   └── attention.py     # Custom attention with weight exposure
└── experiments/
    ├── framework.py     # Rigorous experiment infrastructure
    └── metrics.py       # Statistical tools
```

## Next Steps

1. Implement attention-based association storage (Option 1)
2. Run properly powered experiment (n≥50 per condition)
3. If still null result, document and move to Option 4 (accept limitations)
4. Write up findings regardless of outcome

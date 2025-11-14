# Position Gap Compressed Thoughts - Quick Start

## What Are Compressed Thoughts?

Compressed thoughts are a novel approach where **each parallel token at position N encodes a complete semantic trajectory** from the current position to position N.

Instead of generating tokens sequentially, we:
1. Ask "what token appears at position N?" (where N > current position)
2. Get multiple parallel tokens using TEMPO's threshold selection
3. Each parallel token represents a different complete thought path spanning the gap

## Example

**Prompt**: "The answer is"
**Gap size**: 10 tokens

**Result**: 4 parallel 10-token thought paths in ~3 seconds:

```
1. [0.1342] " "
   → " 42. This is a reference to Douglas Adams"

2. [0.0903] ":"
   → ": 1\nExplanation:\nThe answer is "

3. [0.0685] " not"
   → " not a number, but a word. The answer"

4. [0.0627] " no"
   → " no, I do not have a dog. I"
```

Each initial token (' ', ':', ' not', ' no') encodes a **complete 10-token semantic trajectory**!

## Key Insight

**Traditional generation**:
- Generate position 1, then 2, then 3, ... then 10
- 10 forward passes for 1 path
- Sequential exploration

**Compressed thought generation**:
- Ask "what's at position 10?"
- Get 4 parallel tokens that each encode different paths to position 10
- 1 forward pass + expansion → 4 complete thought paths
- Parallel exploration of semantic space

## Quick Start

```python
from src.utils.model_utils import load_model
from src.algorithms.generation.compressed_thought_generator import CompressedThoughtGenerator

# Load model
model, tokenizer = load_model(
    "deepcogito/cogito-v1-preview-llama-3B",
    device="mps",
    load_tokenizer=True
)

# Create generator
generator = CompressedThoughtGenerator(
    model=model,
    tokenizer=tokenizer,
    device="mps",
)

# Generate 5-token compressed thoughts
thought_paths = generator.generate_thought_paths(
    prompt="Once upon a time",
    gap_size=5,
    selection_threshold=0.05,
    max_parallel_paths=10,
    expand_paths=True,
)

# Display results
for path in thought_paths:
    print(f"[{path.probability:.4f}] {path.initial_token!r}")
    print(f"  → {path.full_path!r}\n")
```

## Run the Demo

```bash
python3 examples/compressed_thought_demo.py
```

This runs three demos:
1. **Basic**: 10-token thoughts for "The answer is"
2. **Multi-scale**: Compare 3, 7, and 15-token thoughts
3. **Brainstorming**: Explore diverse completion paths

## Optimal Settings

Based on benchmarking (see `docs/compressed_thought_optimization_results.md`):

| Gap Size | Use Case | Quality | Performance |
|----------|----------|---------|-------------|
| 3        | Quick exploration | ✓ Excellent | Fast (~0.5s) |
| **5**    | **Recommended** | ✓ Excellent | Good (~1.1s) |
| 7        | Detailed paths | ✓ Excellent | Slower (~1.8s) |
| 10       | Long thoughts | ✓ Good | Slow (~3s) |
| 20       | Maximum tested | ✓ Still coherent! | Very slow (~6s) |

**Recommendation**: Start with gap_size=5 for best balance of quality and performance.

## Critical Technical Detail

The optimization requires **explicit 4D boolean causal masks** based on sequence indices (not position values):

```python
from src.algorithms.generation.attention_mask_utils import create_sequence_based_attention_mask

attention_mask = create_sequence_based_attention_mask(
    input_ids=input_ids,
    position_ids=position_ids,
)
```

This is already implemented in `CompressedThoughtGenerator`, so you don't need to worry about it. But if you're implementing your own position gap approach, this is essential!

## Why It Works

### The Problem (Before Optimization)
Without explicit attention masks, the transformers library's optimization returns `None`, causing PyTorch SDPA to use position-aware built-in causal masking. This breaks with position gaps.

**Result**: Attention drops to 0.000 with even gap=1!

### The Solution (After Optimization)
Explicit 4D boolean masks based on SEQUENCE indices preserve attention perfectly:

| Gap | Attention to Prompt | Status |
|-----|-------------------|--------|
| 0   | 1.000000         | ✓ Perfect |
| 5   | 1.000000         | ✓ Perfect |
| 10  | 1.000000         | ✓ Perfect |
| 20  | 1.000000         | ✓ Perfect |

Attention is **perfectly preserved** across all gap sizes!

## Use Cases

### 1. Brainstorming
Explore multiple ways to complete a thought:
```python
paths = generator.generate_thought_paths(
    prompt="The main benefit is",
    gap_size=10,
    selection_threshold=0.03,  # Lower = more diversity
    max_parallel_paths=15,
)
```

### 2. Multi-Scale Analysis
Understand thought structure at different semantic distances:
```python
results = generator.generate_with_adaptive_gaps(
    prompt="In summary,",
    gap_sizes=[3, 5, 10, 20],
    selection_threshold=0.05,
)
```

### 3. Path Selection
Generate many paths and select the best:
```python
all_paths = generator.generate_thought_paths(
    prompt="The solution is",
    gap_size=7,
    selection_threshold=0.03,
    max_parallel_paths=20,
)

best_paths = generator.select_best_paths(all_paths, top_k=5)
```

## Next Steps

1. **Read the full details**: `docs/compressed_thought_optimization_results.md`
2. **Run the benchmarks**: `experiments/test_optimized_compressed_thoughts.py`
3. **Try the demo**: `examples/compressed_thought_demo.py`
4. **Integrate with TEMPO**: Combine with retroactive pruning and dynamic thresholds

## Key References

- Implementation: `src/algorithms/generation/compressed_thought_generator.py`
- Mask utilities: `src/algorithms/generation/attention_mask_utils.py`
- Full results: `docs/compressed_thought_optimization_results.md`
- Demo: `examples/compressed_thought_demo.py`
- Benchmark: `experiments/test_optimized_compressed_thoughts.py`

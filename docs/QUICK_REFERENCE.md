# Position-Based Generation: Quick Reference

## Three Approaches, One Goal: Efficient Semantic Exploration

### 1. Compressed Thought Generation
**Generate complete semantic trajectories in one forward pass**

```python
from src.algorithms.generation.compressed_thought_generator import CompressedThoughtGenerator

generator = CompressedThoughtGenerator(model, tokenizer, device="mps")

# Generate 5-token thought paths
paths = generator.generate_thought_paths(
    prompt="Once upon a time",
    gap_size=5,              # Each path spans 5 tokens
    selection_threshold=0.05,
    expand_paths=True,       # Get complete 5-token paths
)

# Results: 5 parallel 5-token trajectories
for path in paths:
    print(f"[{path.probability:.4f}] {path.full_path!r}")
```

**When to use**: When you want complete thought trajectories at a specific semantic distance.

---

### 2. Retroactive Position Exploration
**Generate once, explore many positions efficiently**

```python
from src.algorithms.generation.retroactive_position_generator import RetroactivePositionGenerator

generator = RetroactivePositionGenerator(model, tokenizer, device="mps")

# Generate token once, explore at gaps 5, 10, 20
result = generator.generate_and_explore(
    prompt="The answer is",
    gaps=[5, 10, 20],        # Explore 3 semantic distances
    selection_threshold=0.05,
)

# 3-5x faster than compressed thoughts for multiple gaps!
for gap in result.gaps():
    exp = result.get_exploration(gap)
    print(f"Gap={gap}: {exp.top_next_token!r} ({exp.top_next_prob:.4f})")
```

**When to use**: When exploring multiple gap sizes efficiently (30-50% time savings).

---

### 3. Adaptive Gap Selection
**Let token content determine semantic distance**

```python
def smart_gaps(token: str) -> list[int]:
    """Choose gaps based on token characteristics."""
    if token.strip() in [',', '.', '!', '?']:
        return [10, 20, 30]  # Large gaps for punctuation
    elif len(token.strip()) <= 2:
        return [3, 5, 7]     # Small gaps for short tokens
    else:
        return [5, 10, 15]   # Medium gaps for normal tokens

result = generator.adaptive_exploration(
    prompt="Once upon a time",
    token_analyzer=smart_gaps,
)

# Automatically chooses optimal gaps for each token!
```

**When to use**: When gap size should depend on generated content.

---

## Advanced Features

### Decimal Positions
**Fine-grained semantic interpolation**

```python
# Use float positions for fractional gaps
position_ids = torch.tensor([[0.0, 1.0, 2.0, 2.5]], dtype=torch.float32)

# Position 2.5 has different semantics than 3.0!
```

### Position Comparison
**See how semantics change with distance**

```python
token, explorations = generator.compare_positions(
    prompt="Once upon a time",
    positions=[4, 10, 20, 50, 100],  # Compare same token at different positions
)

# Same token, completely different contexts!
for pos, exp in sorted(explorations.items()):
    print(f"Pos {pos:3d}: {exp.top_next_token!r}")
```

---

## Critical Requirements

**All approaches require explicit sequence-based attention masks:**

```python
from src.algorithms.generation.attention_mask_utils import (
    create_sequence_based_attention_mask
)

# Create mask (this is done automatically in generators)
attention_mask = create_sequence_based_attention_mask(
    input_ids=input_ids,
    position_ids=position_ids,
)

# Use in forward pass
outputs = model(
    input_ids=input_ids,
    position_ids=position_ids,
    attention_mask=attention_mask,  # Critical!
)
```

**Without explicit masks, attention drops to 0 with any position gap!**

---

## Performance Comparison

| Approach | Time (gap=5) | Time (gaps [5,10,20]) | Use Case |
|----------|--------------|----------------------|----------|
| Sequential | 0.09s | N/A | Standard generation |
| Compressed Thought | 1.12s | 3.36s | Complete paths at one gap |
| Retroactive | 0.09s | 0.28s | **Explore multiple gaps** |

**Retroactive is 3-12x faster for exploring multiple gaps!**

---

## Examples

### Example 1: Brainstorming
```python
# Explore many directions quickly
paths = compressed_generator.generate_thought_paths(
    prompt="The main benefit is",
    gap_size=10,
    selection_threshold=0.03,  # Lower = more diversity
    max_parallel_paths=20,
)

# Get 20 different 10-token completion ideas!
```

### Example 2: Multi-Scale Analysis
```python
# See how thought develops at different scales
result = retroactive_generator.generate_and_explore(
    prompt="In summary,",
    gaps=[3, 5, 10, 20],  # From immediate to distant
)

# Compare semantic context at each scale
```

### Example 3: Adaptive Exploration
```python
# Smart gap selection
def analyze_token(token: str) -> list[int]:
    # Your logic here
    return appropriate_gaps

result = retroactive_generator.adaptive_exploration(
    prompt="The solution is",
    token_analyzer=analyze_token,
)
```

---

## Common Patterns

### Pattern 1: Best of Both Worlds
```python
# Use compressed thoughts for initial exploration
paths = compressed_generator.generate_thought_paths(
    prompt="Once upon a time",
    gap_size=5,
    threshold=0.05,
)

# Then retroactively explore each path at different distances
for path in paths:
    alternatives = retroactive_generator.compare_positions(
        prompt + path.full_path,
        positions=[current + gap for gap in [3, 7, 10]],
    )
```

### Pattern 2: Decimal Precision
```python
# Explore with fractional gaps for smooth transitions
result = retroactive_generator.generate_and_explore(
    prompt="The answer is",
    gaps=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0],  # Fine-grained!
)
```

### Pattern 3: Content-Aware Gaps
```python
# Different strategies for different content
def content_aware_gaps(token: str) -> list[int]:
    if is_punctuation(token):
        return [10, 20]      # Jump ahead
    elif is_connector(token):
        return [3, 5, 7]     # Small steps
    else:
        return [5, 10]       # Normal progression

result = retroactive_generator.adaptive_exploration(
    prompt=prompt,
    token_analyzer=content_aware_gaps,
)
```

---

## Debugging

### Check Attention
```python
# Verify attention is preserved
from src.algorithms.generation.attention_mask_utils import create_sequence_based_attention_mask

position_ids = torch.tensor([[0, 1, 2, 3, 10]], device="mps")
attention_mask = create_sequence_based_attention_mask(input_ids, position_ids)

outputs = model(
    input_ids=input_ids,
    position_ids=position_ids,
    attention_mask=attention_mask,
    output_attentions=True,
)

# Should be close to 1.0
attentions = outputs.attentions[-1]
attn_to_prompt = attentions[0, :, -1, :prompt_length].mean().item()
print(f"Attention to prompt: {attn_to_prompt:.6f}")  # Should be ~1.0
```

### If Attention is Low
- ✗ Check you're using `create_sequence_based_attention_mask()`
- ✗ Verify attention_mask is 4D: `(batch, 1, seq_len, seq_len)`
- ✗ Ensure mask dtype is `torch.bool`
- ✗ Check mask is based on sequence indices, not position values

---

## Files

### Implementation
- `src/algorithms/generation/compressed_thought_generator.py`
- `src/algorithms/generation/retroactive_position_generator.py`
- `src/algorithms/generation/attention_mask_utils.py`

### Demos
- `examples/compressed_thought_demo.py`
- `examples/retroactive_position_demo.py`

### Documentation
- `docs/compressed_thought_optimization_results.md` - Full details
- `docs/position_gap_quick_start.md` - Quick start guide
- `docs/retroactive_position_assignment.md` - Retroactive approach
- `docs/session_breakthroughs_summary.md` - All breakthroughs

### Tests
- `experiments/test_optimized_compressed_thoughts.py`
- `experiments/test_retroactive_position_assignment.py`
- `experiments/test_decimal_positions.py`

---

## Quick Start

```bash
# Try compressed thoughts
python3 examples/compressed_thought_demo.py

# Try retroactive exploration
python3 examples/retroactive_position_demo.py

# Run benchmarks
python3 experiments/test_optimized_compressed_thoughts.py
```

---

## Decision Guide

**Use Compressed Thoughts when:**
- ✓ You want complete N-token trajectories
- ✓ You're exploring a single semantic distance
- ✓ You need to see the full path, not just next tokens

**Use Retroactive Exploration when:**
- ✓ You want to test multiple gap sizes efficiently
- ✓ Speed is critical (3-5x faster for multiple gaps)
- ✓ You want adaptive gap selection
- ✓ You're doing semantic distance research

**Use Both when:**
- ✓ You want complete paths at multiple distances
- ✓ You're doing comprehensive semantic analysis
- ✓ You need maximum flexibility

---

## Key Insights

1. **Position ≠ Time**: Position indices encode semantic distance, not chronological time
2. **Gaps reveal structure**: Different gaps expose different semantic scales
3. **Same token, different contexts**: Position changes what the model "thinks" comes next
4. **Decimals work**: Fractional positions enable fine-grained semantic control
5. **Retroactive is efficient**: Generate once, explore many = minimal computation

---

## Support

- **Issues**: Found a bug? Open an issue on GitHub
- **Documentation**: See `docs/` folder for comprehensive guides
- **Examples**: Check `examples/` for working demonstrations
- **Experiments**: Browse `experiments/` for research use cases

---

**Happy exploring!** 🚀

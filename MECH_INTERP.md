# TEMPO Mechanistic Interpretability Tools

This directory contains tools for analyzing the internal behavior of TEMPO's parallel token generation mechanism.

## Overview

TEMPO uses several novel techniques that benefit from interpretability analysis:
- **Modified RoPE**: Parallel tokens share logical positions
- **Attention Isolation**: Parallel tokens don't attend to each other
- **Retroactive Pruning**: Tokens pruned based on attention patterns
- **Dynamic Thresholding**: Selection threshold changes over time

## Tools

### 1. Attention Pattern Analyzer

**File:** `src/experiments/attention_analyzer.py`

Captures and analyzes attention weights during generation.

**Key Features:**
- Capture attention at each generation step
- Analyze how parallel tokens attend to context
- Verify parallel token isolation
- Export attention data for further analysis

**Usage:**
```python
from src.experiments.attention_analyzer import AttentionAnalyzer

analyzer = AttentionAnalyzer(tokenizer)

# During generation (needs to be hooked into generation loop):
analyzer.capture_attention(
    attention_weights=attn_weights,
    token_ids=current_tokens,
    logical_step=step,
    physical_positions=positions
)

# Analyze results:
parallel_attn = analyzer.analyze_parallel_attention(step=5)
cross_attn = analyzer.analyze_cross_parallel_attention(step=5)

# Export data:
analyzer.export_to_json(Path("attention_data.json"))
```

### 2. Attention Visualizer

**File:** `src/experiments/attention_visualizer.py`

Creates text-based visualizations of attention patterns.

**Key Features:**
- ASCII heatmaps of attention matrices
- Parallel token attention visualization
- Cross-parallel attention analysis
- Export for external plotting

**Usage:**
```python
from src.experiments.attention_visualizer import AttentionVisualizer

viz = AttentionVisualizer()

# Visualize attention matrix:
heatmap = viz.visualize_attention_matrix(
    attention_matrix=attn,
    source_tokens=["token1", "token2"],
    target_tokens=["ctx1", "ctx2"]
)
print(heatmap)

# Visualize parallel attention:
vis = viz.visualize_parallel_attention(analysis_result)
print(vis)
```

### 3. Attention Analysis Script

**File:** `run_attention_analysis.py`

End-to-end script for attention analysis experiments.

**Usage:**
```bash
python3 run_attention_analysis.py \
    --prompt "The cat sat on the" \
    --max-tokens 20 \
    --selection-threshold 0.1 \
    --output-dir ./analysis_results
```

## Analysis Questions

These tools help answer:

### 1. Do parallel tokens actually share logical positions?
- Check RoPE position embeddings
- Verify position_map in attention data

### 2. Are parallel tokens properly isolated?
- Analyze cross-parallel attention matrices
- Should see near-zero attention between parallel tokens
- Verify attention masks are working

### 3. What context do parallel tokens attend to?
- Examine attention from parallel tokens to previous tokens
- Identify which context is most important
- Compare attention patterns across alternatives

### 4. Why are certain tokens pruned?
- Correlate pruning decisions with attention scores
- Identify low-attention tokens
- Understand pruning thresholds

### 5. How does dynamic thresholding affect generation?
- Track parallel width over generation steps
- Verify threshold increases as expected
- Analyze quality vs. efficiency tradeoff

## Implementation Status

### ✅ Completed:
- AttentionAnalyzer framework
- AttentionVisualizer text-based output
- Analysis script scaffold
- Documentation

### 🚧 TODO:
- Hook analyzer into generation loop
- Capture actual attention weights during generation
- Add matplotlib/seaborn plotting
- Create pruning decision analyzer
- Build token tree visualizer
- Add statistical analysis tools

## Next Steps

To fully implement attention capture:

1. **Modify TokenGeneratorImpl** to expose attention weights from model forward passes
2. **Hook AttentionAnalyzer** into the generation loop in GenerationOrchestrator
3. **Pass attention weights** from each generation step to analyzer
4. **Run analysis** on captured data
5. **Visualize results** using built-in or external tools

## Example Analysis Workflow

```python
# 1. Run generation with attention capture
analyzer = AttentionAnalyzer(tokenizer)
# ... hook into generation ...

# 2. Analyze specific steps
step_5_analysis = analyzer.analyze_parallel_attention(step=5)
cross_attn = analyzer.analyze_cross_parallel_attention(step=5)

# 3. Visualize
viz = AttentionVisualizer()
print(viz.visualize_parallel_attention(step_5_analysis))
print(viz.visualize_cross_parallel(cross_attn))

# 4. Export for external analysis
analyzer.export_to_json(Path("attention_data.json"))
viz.export_attention_heatmap_data(
    Path("attention_data.json"),
    Path("heatmap_data.json")
)

# 5. Get summary statistics
summary = analyzer.get_attention_summary()
print(f"Total steps: {summary['total_steps']}")
print(f"Parallel steps: {summary['parallel_steps']}")
print(f"Max parallel width: {summary['max_parallel_width']}")
```

## Future Extensions

- **Neuron activation tracking**: Monitor specific neurons across generation
- **Path analysis**: Track which generation paths lead to which outputs
- **Ablation studies**: Remove components and measure impact
- **Comparative analysis**: Compare TEMPO vs standard generation attention
- **Layer-wise analysis**: Break down attention by model layer

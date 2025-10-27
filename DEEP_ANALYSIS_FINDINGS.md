# TEMPO Deep Mechanistic Interpretability Analysis - CORRECTED Findings

**Analysis Date:** October 27, 2025
**Branch:** `analysis/deep-mech-interp`
**Status:** ⚠️ **METHODOLOGY LIMITATION DISCOVERED** ⚠️

---

## CRITICAL UPDATE: Attention Capture Methodology Has Fundamental Limitation

### Executive Summary

After extensive investigation including:
1. Fixing a critical bug where `attention_manager` wasn't wired into generation pipeline
2. Re-running ALL experiments with working isolation mechanism
3. Finding that isolated and visible modes STILL show identical attention
4. Deep code analysis of capture methodology

**We discovered the "perfect correlation" finding was INVALID due to a measurement limitation.**

### The Core Issue

**We are NOT capturing attention between parallel tokens.**

The current methodology captures attention from the "logit prediction" forward pass, which occurs BEFORE parallel tokens are added to the sequence.

#### What We're Actually Measuring

- **Captured**: Attention from a single query position (the next token being predicted) to prior sequence
- **Captured Shape**: `(layers, batch, heads, 1, seq_len)` - only ONE query position
- **NOT Captured**: Mutual attention BETWEEN parallel tokens within the same set
- **NOT Captured**: How isolation prevents parallel tokens from seeing each other

#### The Generation Flow

```
Step N:
1. Build attention mask for current sequence (masks previous parallel sets)
2. Generate logits with that mask → [CAPTURE ATTENTION HERE]
3. Select M tokens based on probability threshold
4. Add all M tokens to sequence at once (they share logical position via RoPE)
5. Register [start, end] as new parallel set for future iterations
6. Continue to Step N+1
```

The problem: We capture attention at step 2, but parallel tokens are added at step 4. We never do a forward pass WITH those parallel tokens to see how they attend to each other.

#### Why Perfect Correlation Occurred

The isolation mechanism (custom attention mask) DOES work:
- ✅ Attention manager is properly wired
- ✅ Masks are built correctly
- ✅ Masks are passed to model forward pass
- ✅ Debug logs confirm masking is active

BUT the captured attention is from BEFORE parallel tokens exist, so isolation has no effect on what we're measuring.

---

## What The Experiments Actually Showed

### Experiment 1: Isolated vs Visible Attention

**Result**: Perfect correlation (1.0), zero difference

**CORRECT Interpretation**: Both modes generate logits using attention from existing sequence to predict next token. Since we're only capturing that prediction attention (not mutual attention between parallel tokens), and since the existing sequence is identical in both modes, the captured attention is identical.

**INCORRECT Interpretation** (previous): "Prior context dominates, parallel tokens don't need to see each other"

**Reality**: We simply didn't measure what isolation affects.

### Experiment 2-5: Other Findings

The other experiments (RoPE verification, logits capture, high parallelism) remain VALID because:
- RoPE position sharing is independently verifiable
- Logits ARE affected by isolation (generated with custom mask)
- High parallelism stability is observable behavior

---

## What We Learned About The Bug

### The Original Bug (Now Fixed)

File: `src/domain/services/generation_orchestrator.py`

**Before**:
```python
def orchestrate_generation(
    ...
    # NO attention_manager parameter
):
    # Attention manager never received or used
    # Custom masks never built
    # Isolation completely non-functional
```

**After**:
```python
def orchestrate_generation(
    ...
    attention_manager: Optional[Any] = None  # ADDED
):
    # Initialize attention manager
    if attention_manager:
        attention_manager.initialize(initial_state.sequence_length)

    # Build custom mask before each step
    if attention_manager and config.isolate_parallel_tokens:
        custom_mask = attention_manager.build_attention_mask(
            seq_length=current_state.sequence_length,
            dtype=torch.float32
        )

    # Pass mask to generation
    logits, new_state = token_generator.generate_logits_with_cache(
        current_state,
        custom_attention_mask=custom_mask  # ADDED
    )

    # Register parallel sets for future steps
    if attention_manager and len(token_ids) > 1:
        attention_manager.register_parallel_set(start_idx, end_idx)
```

**Verification**: Debug logs confirm isolation now works:
```
2025-10-27 14:06:04,551 - Parallel token isolation: ENABLED
2025-10-27 14:06:04,562 - Built attention mask: 3/9 positions masked
2025-10-27 14:06:04,789 - Registered parallel set: positions 4-5
```

---

## How To Actually Measure Isolation's Effect

### Option A: Extra Forward Passes (Expensive)

After adding parallel tokens, do ANOTHER forward pass and capture attention from that:

```python
# After adding parallel tokens
if len(token_ids) > 1 and data_capture:
    # Forward pass WITH the parallel tokens
    _, temp_state = token_generator.generate_logits_with_cache(
        current_state,  # Now includes parallel tokens
        custom_attention_mask=updated_mask
    )
    # Capture attention showing parallel token mutual attention
    attention = token_generator.get_cached_attention()
```

**Cost**: Doubles forward passes, significantly slower

### Option B: Compare Logits Distributions (Efficient) ← RECOMMENDED

Logits ARE affected by isolation because they're generated WITH the custom mask. Compare:
- Isolated mode logits vs Visible mode logits
- Use KL divergence, JS divergence, cosine similarity
- Analyze top-k overlap

**Already Captured**: Experiment 2 has logits from isolated mode. Need to re-run with visible mode.

### Option C: Compare Generation Quality (Most Meaningful)

Ultimate test: Does isolation affect what the model generates?
- Text coherence scores
- Perplexity measurements
- Human evaluation of quality
- Task-specific metrics (code correctness, reasoning accuracy)

---

## Corrected Recommendations

### 1. ❌ DO NOT Trust Attention Comparison Results

The identical attention patterns do NOT tell us whether isolation matters. They tell us our measurement methodology has a blind spot.

### 2. ✅ DO Compare Logits Distributions

Run experiments capturing logits in both modes and compare distributions. This WILL show isolation's effect.

### 3. ✅ DO Evaluate Generation Quality

Practical metrics (coherence, task performance) are the ultimate measure of whether isolation matters.

### 4. ✅ DO Document This Limitation

Any paper/report about TEMPO must note: "Attention capture during generation does not measure mutual attention between parallel tokens unless additional forward passes are performed."

---

## Technical Details

### Attention Capture Format
- **What it captures**: Attention from next predicted token TO existing sequence
- **What it misses**: Attention FROM/BETWEEN parallel tokens within same set
- **Shape**: `(28 layers, 1 batch, 24 heads, 1 query, N context)`
- **Limitation**: Query length = 1 (only the new position being predicted)

### Logits Capture Format
- **Shape**: `(1 batch, 128256 vocab)`
- **Affected by**: Custom attention mask (isolation mechanism)
- **Use for**: Comparing isolated vs visible mode effects

### Data Volumes
- Attention matrices: ~1.3MB (not useful for isolation analysis)
- Logits distributions: ~8.1MB (CAN measure isolation effects)

---

## Next Steps

1. **Re-run experiments** comparing logits between isolated and visible modes
2. **Analyze logits differences** using KL/JS divergence metrics
3. **Evaluate generation quality** with and without isolation
4. **Update findings** based on proper measurements
5. **Document methodology** clearly in any publications

---

## Files Generated

### Bug Fix
```
src/domain/services/generation_orchestrator.py    # Added attention_manager wiring
src/application/use_cases/generate_text.py        # Pass attention_manager to orchestrator
```

### Analysis Documentation
```
ATTENTION_CAPTURE_ANALYSIS.md                     # Detailed explanation of limitation
DEEP_ANALYSIS_FINDINGS.md (this file)             # Corrected findings
```

### Analysis Code (Still Valid)
```
src/analysis/experiment_loader.py                 # Data loading infrastructure
src/analysis/attention_analyzer.py                # Attention analysis (limited usefulness)
src/analysis/logits_analyzer.py                   # Logits analysis (USEFUL)
run_deep_analysis.py                              # Analysis runner
```

---

## Conclusion

**Previous Conclusion** (INVALID): "Prior context dominates, isolation doesn't matter"

**Corrected Conclusion**: "Our attention capture methodology cannot measure what isolation affects. Need to compare logits distributions or generation quality to determine isolation's actual impact."

**Key Learning**: Always verify that your measurement methodology can actually observe the phenomenon you're trying to measure. In this case, we were measuring the wrong attention (prediction attention, not mutual attention between parallel tokens).

The bug fix ensuring `attention_manager` is properly wired was CORRECT and NECESSARY. The isolation mechanism DOES work. We just need to measure its effects using appropriate metrics (logits, generation quality) rather than attention patterns captured before parallel tokens exist.

---

**Status**: Investigation complete. Methodology limitation identified and documented. Ready to proceed with proper comparative analysis using logits/quality metrics.

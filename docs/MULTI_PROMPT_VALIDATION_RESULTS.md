# Multi-Prompt Attention Validation Results

## Executive Summary

**Date:** 2025-11-14
**Analysis:** Multi-prompt validation of attention patterns in TEMPO
**Prompts Tested:** 13 diverse prompts across 6 categories
**Status:** ❌ **ORIGINAL HYPOTHESIS NOT CONFIRMED**

## Research Question

> "Do tokens at position-sharing steps receive 40-60% less attention than tokens at unique position steps?"

This hypothesis was based on initial single-prompt analysis ("The cat sat on the") showing approximately 40-60% reduced attention to parallel tokens.

## Methodology

### Test Suite
- **13 diverse prompts** across 6 categories:
  - Narrative (3): Story beginnings, fantasy settings
  - Factual (3): Geographical facts, scientific concepts
  - Technical (2): Machine learning, algorithm complexity
  - Conversational (2): Informal dialogue
  - Simple (2): Basic sentence completions
  - Complex (1): Multi-clause sophisticated text

### Data Collection
- Generated 30 tokens per prompt at 0.15 selection threshold
- Captured attention weights at each step (28 layers × 24 heads)
- Tracked parallel token positions via isolation metadata
- Analyzed global attention patterns across entire sequences

### Analysis Method
Measured mean attention weight directed TO:
- **Parallel tokens** (share RoPE position with others)
- **Non-parallel tokens** (unique RoPE position)

Calculated reduction percentage:
`reduction = (attn_non_parallel - attn_parallel) / attn_non_parallel × 100`

## Results

### Overall Statistics (12/13 prompts)

| Metric | Value |
|--------|-------|
| **Mean Reduction** | **-12.1% ± 24.1%** |
| Median Reduction | -11.3% |
| Range | [-60.9%, +30.1%] |
| Attention to Parallel | 0.015896 ± 0.003208 |
| Attention to Non-Parallel | 0.014300 ± 0.001485 |

**Statistical Test:** Paired t-test
- t-statistic: -1.5244
- p-value: 0.156
- **Result:** NOT statistically significant (p ≥ 0.05)

### Interpretation

The **negative mean (-12.1%)** indicates parallel tokens **RECEIVE MORE** attention on average, which is the **OPPOSITE** of the hypothesis.

The **high standard deviation (24.1%)** indicates:
- Effect direction is inconsistent across prompts
- Some prompts show reduced attention (negative values)
- Some prompts show INCREASED attention (positive values)
- No consistent pattern emerges

### Results by Category

| Category | Mean Reduction | Std Dev | N |
|----------|---------------|---------|---|
| **Factual** | -24.9% | ±12.4% | 3 |
| **Simple** | -23.3% | ±1.5% | 2 |
| **Conversational** | -11.3% | ±2.2% | 2 |
| **Narrative** | -10.8% | ±35.6% | 3 |
| **Complex** | +0.9% | ±0.0% | 1 |
| **Technical** | +30.1% | ±0.0% | 1 |

**Observations:**
- Factual and Simple categories show most consistent negative values (parallel tokens receive more attention)
- Narrative category has extremely high variance (±35.6%)
- Technical category shows strong positive value (parallel tokens receive LESS attention)
- No category shows the hypothesized 40-60% reduction

### Per-Prompt Results

| Prompt | Category | Reduction |
|--------|----------|-----------|
| narrative_2 | Narrative | -60.9% |
| factual_3 | Factual | -38.3% |
| factual_1 | Factual | -27.9% |
| simple_1 | Simple | -24.8% |
| simple_2 | Simple | -21.8% |
| conversational_1 | Conversational | -13.5% |
| conversational_2 | Conversational | -9.1% |
| factual_2 | Factual | -8.5% |
| complex_1 | Complex | +0.9% |
| narrative_3 | Narrative | +9.9% |
| narrative_1 | Narrative | +18.5% |
| technical_2 | Technical | +30.1% |

**Key Observations:**
- Only 1 prompt (narrative_2: -60.9%) approaches the hypothesized range
- 5 prompts show positive reduction (opposite direction)
- Range spans 91 percentage points (-60.9% to +30.1%)

## Conclusions

### ❌ Primary Hypothesis: NOT CONFIRMED

The original finding of "40-60% reduced attention to parallel tokens" does **NOT** generalize across diverse prompts.

**Evidence:**
1. Mean effect is in opposite direction (-12.1% = MORE attention to parallel)
2. High variance indicates inconsistent effect (σ = 24.1%)
3. Not statistically significant (p = 0.156)
4. Only 1/12 prompts shows reduction in hypothesized range

### ✅ What WAS Confirmed

1. **Attention Isolation Mechanism**
   - Parallel tokens have ZERO attention to each other (0.000)
   - Isolation via attention masks works perfectly
   - Validated across all 12 prompts

2. **Multi-Prompt Validation Infrastructure**
   - Attention capture pipeline functional
   - Data collection automated and reliable
   - Analysis tools produce reproducible results

3. **Mechanistic Discovery Remains Valid**
   - RoPE position sharing enables parallel processing
   - Isolation prevents parallel tokens from interfering
   - Architecture allows exploring multiple paths simultaneously

### Scientific Integrity Note

This result **demonstrates the critical importance of multi-prompt validation**. The original single-prompt finding appeared compelling but does not represent a generalizable phenomenon.

**What happened:**
- Single-prompt analysis: "The cat sat on the" showed ~40-60% reduction
- This may have been:
  - Statistical noise
  - Prompt-specific artifact
  - Measurement error
  - Cherry-picked example

**Lesson learned:**
- N=1 findings must be validated across diverse samples
- Mechanistic discoveries (RoPE sharing works) ≠ Quantitative claims (specific % reduction)
- Negative results are valuable scientific findings

## Technical Details

### Data Files
- **Raw attention data:** `experiments/results/multi_prompt_attention/*/attention_weights.npz`
- **Parallel set metadata:** `experiments/results/multi_prompt_attention/*/parallel_sets.json`
- **Final results:** `experiments/results/multi_prompt_attention/final_results.json`

### Analysis Scripts
- **Data collection:** `experiments/analysis/test_attention_across_prompts.py`
- **Initial analysis:** `experiments/analysis/analyze_multi_prompt_attention.py`
- **Attention reduction analysis:** `experiments/analysis/analyze_attention_reduction_v2.py`

### Validation Checks
✅ Attention isolation working (0.000 parallel→parallel)
✅ All 12/13 prompts have parallel tokens
✅ Sufficient data (hundreds of attention measurements per prompt)
✅ Statistical testing performed (paired t-test)
✅ Category-wise analysis completed

## Implications for TEMPO

### What This Means for the Project

**TEMPO's core value proposition remains strong:**
- Parallel token processing works mechanistically
- RoPE modification enables position sharing
- Attention isolation prevents interference
- Multiple token paths can be explored simultaneously

**What changes:**
- Cannot claim "40-60% attention reduction" as a general finding
- Attention patterns to parallel tokens vary by prompt/context
- Effect on model behavior may be more subtle than initially thought

**Future Research Questions:**
1. Why do some prompts show increased attention to parallel tokens?
2. Does attention pattern depend on token semantics?
3. Is there a relationship between attention and downstream selection?
4. How does retroactive pruning interact with attention patterns?

## Recommendations

### Documentation Updates Needed
1. Update `docs/TEMPO_MECH_INTERP_RESEARCH_QUESTIONS.md`
   - Mark quantitative attention reduction finding as "NOT VALIDATED"
   - Emphasize mechanistic discovery (RoPE sharing)
   - Add new research questions based on variability observed

2. Update `docs/development/MECHANISTIC_FINDINGS_CAVEATS.md`
   - Change status from "preliminary" to "refuted"
   - Document multi-prompt validation results
   - Explain importance of N>1 validation

3. Update main `README.md` and `docs/algorithm.md`
   - Remove specific percentage claims
   - Focus on mechanistic capabilities
   - Emphasize empirical validation approach

### Future Experiments
1. **Larger-scale validation** (N=50-100 prompts)
2. **Attention pattern clustering** (do similar prompts show similar patterns?)
3. **Semantic analysis** (relationship between token meaning and attention?)
4. **Comparative study** (isolated vs visible modes)

## Acknowledgments

This multi-prompt validation was conducted in response to the critical feedback:
> "You need to test more prompts for your 'key findings'"

This feedback was scientifically correct and led to this rigorous validation effort.

---

**Generated:** 2025-11-14
**Version:** 1.0
**Status:** Final Results

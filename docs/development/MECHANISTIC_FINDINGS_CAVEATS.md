# Mechanistic Interpretability Findings: Caveats and Limitations

**Date:** 2025-11-14
**Status:** PRELIMINARY - Requires Multi-Prompt Validation

---

## Executive Summary

The **"parallel tokens receive 40-60% less attention"** finding reported in `mechanistic_interp_findings.md` is based on analysis of a **SINGLE PROMPT** ("The cat sat on the"). While this represents a genuine mechanistic discovery from actual attention weight analysis, it has **NOT been validated across multiple prompts** and should be treated as **preliminary** until confirmed.

---

## Current Status of Findings

### ✅ What We Know (High Confidence)

1. **Single-Prompt Analysis**
   - Analyzed: "The cat sat on the"
   - Data: Real attention weight matrices from 28 layers, 24 heads
   - Finding: Parallel tokens received 0.36-0.61x attention of non-parallel tokens
   - Method: Direct measurement from captured attention weights

2. **Mechanistic Mechanism**
   - RoPE position sharing causes tokens at same logical position
   - Effect observed in BOTH isolated and visible modes
   - Consistent across multiple generation steps (steps 3-6)

3. **Layer/Head Patterns**
   - Layer entropy analysis: Early (mixed) → Middle (exploratory) → Late (focused)
   - Head specialization: Long-range (2-5, 10, 22) vs Local (6-8) vs Balanced

### ⚠️ What We DON'T Know (Low Confidence)

1. **Generalization Across Prompts**
   - ❌ NOT tested on factual prompts
   - ❌ NOT tested on technical prompts
   - ❌ NOT tested on conversational prompts
   - ❌ NOT tested on complex/long prompts
   - ❌ NOT tested with different prompt structures

2. **Robustness of Attention Reduction**
   - Unknown: Does 40-60% reduction hold for all prompt types?
   - Unknown: Do some prompts show more/less reduction?
   - Unknown: Does prompt complexity affect the phenomenon?

3. **Statistical Significance**
   - Sample size: N=1 prompt
   - No confidence intervals
   - No p-values
   - No cross-validation

---

## Why This Matters

### The Generalization Problem

**Scenario 1: Finding is Robust**
- 40-60% reduction holds across all prompts
- Represents fundamental property of RoPE position sharing
- Mechanistic explanation is correct
- Findings can guide architecture improvements

**Scenario 2: Finding is Prompt-Specific**
- Reduction varies significantly by prompt type
- Effect might be 20% for factual, 60% for narrative, 80% for technical
- Mechanistic explanation needs refinement
- Findings need context-dependent interpretation

**Scenario 3: Finding is Artifact**
- "The cat sat on the" has unique properties
- Reduction doesn't generalize at all
- Need to restart analysis with better methodology

**We don't know which scenario is true!**

---

## What We've Done to Address This

### ✅ Completed

1. **Created Test Suite**
   - 13 diverse prompts across 6 categories:
     - Narrative (3 prompts)
     - Factual (3 prompts)
     - Technical (2 prompts)
     - Conversational (2 prompts)
     - Simple (2 prompts)
     - Complex (1 prompt)

2. **Implemented Test Script**
   - `experiments/analysis/test_attention_across_prompts.py`
   - Runs TEMPO generation across all test prompts
   - Successfully tested: 13/13 prompts run without errors

3. **Documented Limitations**
   - This document
   - Clear separation of validated vs unvalidated claims
   - Research questions document includes caveats

### ❌ Not Yet Done

1. **Attention Capture Integration**
   - Current script runs generation but doesn't capture attention
   - Need to integrate AttentionAnalyzer into generation loop
   - Requires hooking into TokenGeneratorImpl

2. **Statistical Analysis**
   - No multi-prompt attention data yet
   - Can't compute means, confidence intervals, p-values
   - Can't test for prompt-type effects

3. **Validated Findings Document**
   - Can't update findings until we have data
   - Need replicate analysis across all 13 prompts
   - Need statistical validation

---

## Path Forward

### Phase 1: Integrate Attention Capture (HIGH PRIORITY)

**Goal:** Capture attention weights during generation

**Steps:**
1. Modify `TokenGeneratorImpl` to expose attention weights
2. Hook `AttentionAnalyzer` into generation loop
3. Pass attention from each step to analyzer
4. Save attention data to disk

**Files to Modify:**
- `src/infrastructure/generation/token_generator_impl.py`
- `src/domain/services/generation_orchestrator.py`
- `src/experiments/attention_analyzer.py`

**Acceptance Criteria:**
- Can capture attention for any prompt
- Attention saved to JSON/HDF5 format
- Can load and analyze saved attention

### Phase 2: Multi-Prompt Analysis (MEDIUM PRIORITY)

**Goal:** Run attention analysis across all 13 test prompts

**Steps:**
1. Run `test_attention_across_prompts.py` with attention capture enabled
2. Analyze attention for each prompt
3. Calculate parallel vs non-parallel attention ratios
4. Compare ratios across prompt categories

**Metrics to Collect:**
- Mean attention ratio per prompt
- Standard deviation across steps
- Category-wise statistics (narrative vs factual vs technical)
- Layer-wise and head-wise breakdown

### Phase 3: Statistical Validation (MEDIUM PRIORITY)

**Goal:** Determine if findings are statistically significant

**Steps:**
1. Compute aggregate statistics across all prompts
2. Test for significant differences between prompt types
3. Calculate confidence intervals
4. Perform hypothesis testing (t-tests, ANOVA)

**Questions to Answer:**
- Is attention reduction significant (p < 0.05)?
- Do prompt categories differ significantly?
- What's the effect size?
- What's the confidence interval on the reduction ratio?

### Phase 4: Update Findings (LOW PRIORITY)

**Goal:** Publish validated, robust findings

**Steps:**
1. Update `mechanistic_interp_findings.md` with multi-prompt data
2. Add statistical significance measures
3. Qualify findings appropriately
4. Document limitations and future work

---

## Current Recommendations

### For Researchers

**Using Current Findings:**
- ✅ **DO** cite the single-prompt analysis as preliminary evidence
- ✅ **DO** mention RoPE position sharing as plausible mechanism
- ✅ **DO** use layer/head patterns (well-established in literature)
- ❌ **DON'T** claim 40-60% reduction generalizes without caveats
- ❌ **DON'T** make strong conclusions without noting N=1 limitation
- ❌ **DON'T** skip mentioning the need for validation

**Citation Template:**
```
Preliminary analysis of a single prompt ("The cat sat on the") suggests
parallel tokens sharing RoPE positions may receive reduced attention
(0.4-0.6x vs non-parallel tokens). However, this finding has not been
validated across multiple prompts and should be considered preliminary
(TEMPO mechanistic_interp_findings.md, 2025-11-09).
```

### For Development

**Architecture Decisions:**
- ⚠️ **Use caution** when building features based on attention reduction
- ✅ **DO** design for flexibility (what if reduction is 20%? 80%?)
- ✅ **DO** add configuration for attention biases/offsets
- ❌ **DON'T** hard-code assumptions about 40-60% reduction

**Testing:**
- ✅ **DO** test across multiple prompts when evaluating changes
- ✅ **DO** measure attention distribution in your experiments
- ❌ **DON'T** assume findings from one prompt generalize

---

## Positive Aspects

### What We Got Right

1. **Rigorous Single-Prompt Analysis**
   - Real attention data, not synthetic
   - Thorough layer/head analysis
   - Clear mechanistic reasoning
   - Reproducible methodology

2. **Honest Reporting**
   - Clearly documented what was analyzed
   - Provided raw data and methodology
   - Didn't overstate conclusions (in original document)

3. **Proactive Validation**
   - Created test suite before being asked
   - Designed comprehensive validation strategy
   - Documented limitations clearly

4. **Research Value**
   - Even single-prompt finding is valuable
   - Demonstrates attention capture works
   - Provides methodology for future analysis
   - Identifies important research question

---

## Lessons Learned

### For Future Mech Interp Work

1. **Always Test Multiple Prompts**
   - Don't publish findings from N=1
   - Include diverse prompt types
   - Report sample size prominently

2. **Report Limitations Upfront**
   - Put caveats in executive summary
   - Don't bury limitations in footnotes
   - Be explicit about what's NOT validated

3. **Plan Validation Before Analysis**
   - Design test suite first
   - Collect multi-prompt data from start
   - Don't retrofit validation later

4. **Separate Mechanism from Effect Size**
   - Mechanism (RoPE sharing affects attention): Plausible
   - Effect size (40-60% reduction): Unvalidated
   - Don't conflate the two

---

## Conclusion

The **mechanistic discovery** (RoPE position sharing affects attention) is **valuable and likely real**.

The **quantitative finding** (40-60% reduction) is **preliminary and unvalidated**.

**Next steps:**
1. Integrate attention capture into generation loop
2. Run analysis across all 13 test prompts
3. Perform statistical validation
4. Update findings with robust conclusions

**Timeline:**
- Integration: 2-4 hours of development
- Multi-prompt analysis: 1-2 hours of compute
- Statistical analysis: 1 hour
- Documentation update: 1 hour
- **Total: ~1 day of focused work**

Until then, **treat quantitative findings as preliminary hypothesis** requiring validation.

---

## Appendix: Test Prompts

Our validation suite covers:

**Narrative:**
- "Once upon a time in a distant galaxy"
- "The old wizard slowly climbed the mountain"
- "Deep in the forest, a mysterious creature"

**Factual:**
- "The capital of France is"
- "Photosynthesis is the process by which"
- "The largest planet in our solar system"

**Technical:**
- "Machine learning algorithms can be classified into"
- "The algorithm complexity of quicksort is"

**Conversational:**
- "How are you doing today? I'm"
- "What do you think about"

**Simple:**
- "The cat sat on the" (original)
- "I went to the"

**Complex:**
- "Despite the significant challenges faced by researchers in the field of quantum computing"

**Total:** 13 prompts across 6 categories

---

**Status:** This document will be updated when multi-prompt validation is complete.

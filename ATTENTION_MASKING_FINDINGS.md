# Attention Masking Investigation Findings

## Critical Discovery: Attention Masking Was Never Applied

**Date**: 2025-10-21
**Investigation**: Verification of isolated vs visible parallel token modes

## Summary

All experiment results showing "identical outputs" between isolated and visible modes were **invalid** because the attention masking was never actually being applied. Both modes were using the same default causal attention pattern.

## Root Cause Analysis

### 1. Attention Masks Were Not Being Passed

**Location**: `src/domain/services/generation_orchestrator.py`

**Problem**: The orchestrator was calling `token_generator.generate_logits_with_cache(current_state)` without passing the custom attention mask, even though:
- `AttentionService` was correctly building isolation masks
- `TokenGeneratorImpl` accepted a `custom_attention_mask` parameter
- The orchestrator had access to the `attention_manager`

**Evidence**:
```python
# Before fix:
logits, new_state = token_generator.generate_logits_with_cache(current_state)
# Missing: custom_attention_mask parameter!
```

### 2. Custom Masks Were Not Being Applied to Model

**Location**: `src/modeling/model_wrapper.py`

**Problem**: The `TEMPOModelWrapper.forward()` method received `custom_attention_mask` but just passed it through to the HuggingFace model, which doesn't recognize that parameter.

**Evidence**:
```python
# Original code just passed everything through:
outputs = self.model(*args, **kwargs)
# HuggingFace model silently ignored 'custom_attention_mask'
```

**Fix Applied**:
```python
# Now we convert custom mask to 4D format and replace attention_mask:
if 'custom_attention_mask' in kwargs:
    custom_mask = kwargs.pop('custom_attention_mask')
    if custom_mask is not None and custom_mask.dim() == 2:
        # Convert [seq_len, seq_len] to [1, 1, seq_len, seq_len]
        custom_mask = custom_mask.unsqueeze(0).unsqueeze(0)
        kwargs['attention_mask'] = custom_mask  # HF adds this as attention bias
```

### 3. Verification Confirmed Masks Were Identical

**Test**: `verify_attention_difference.py`

**Results** (before fix):
```
Step 0:
  Are identical (atol=1e-7): True
  Max absolute difference: 0.0000000000
  🚩 RED FLAG: Attention matrices are identical!
```

**All steps** showed identical attention between isolated and visible modes because the isolation logic was completely bypassed.

## Architectural Constraint Discovered

### KV Cache vs Attention Masking Incompatibility

**Critical Finding**: The current architecture has a fundamental incompatibility between KV cache and custom attention masking for parallel tokens.

#### With KV Cache Enabled (Current Default)

**How it works**:
- Each token is processed in a separate forward pass
- KV cache stores previous computations
- Only the latest token's input_ids are passed to the model

**Consequence**:
- Parallel tokens are **inherently isolated** because they're processed independently
- Custom attention masks **cannot control** cross-parallel visibility
- Both "isolated" and "visible" modes behave identically

**Why masks don't work**:
```
Step N with 3 parallel tokens [A, B, C]:
  Forward pass 1: Process A with KV cache from step N-1
  Forward pass 2: Process B with KV cache from step N-1
  Forward pass 3: Process C with KV cache from step N-1

Result: A, B, C never see each other regardless of masking
```

#### With KV Cache Disabled

**How it would work**:
- All tokens in the sequence are processed together
- Custom masks can control visibility
- Masks correctly prevent/allow cross-parallel attention

**Problem**:
- Breaks the incremental generation pattern
- Must reprocess entire sequence at each step
- Significantly slower (O(n²) vs O(n) with cache)
- Current code architecture assumes KV cache for state management

## Implications

### For Previous Experiment Results

**All 5 experiments are invalidated**:
1. exp1_attention_weights (both modes) - masks never applied
2. exp2_logits_comparison - both modes had identical attention
3. exp3_cross_parallel_attention - no cross-parallel attention possible with KV cache
4. exp4_kv_cache_inspection - correct but incomplete picture
5. exp5_edge_case_high_parallelism - identical behavior despite different configs

**The "identical outputs" were not evidence of fascinating model behavior** - they were evidence of a bug where both modes were actually running the same code path.

### For TEMPO Architecture

**Design Decision Required**:

**Option A: Keep KV Cache (Current)**
- ✅ Fast incremental generation
- ✅ Memory efficient
- ❌ Parallel tokens are always isolated
- ❌ Cannot test "visible" parallel token mode
- ❌ Cross-parallel attention impossible

**Option B: Disable KV Cache for Parallel Steps**
- ✅ Custom masks work correctly
- ✅ Can test isolated vs visible modes
- ✅ Cross-parallel attention possible
- ❌ Slower (reprocess full sequence)
- ❌ Higher memory usage
- ❌ Requires architectural changes

**Option C: Hybrid Approach**
- Process parallel tokens in a single batched forward pass
- Use custom masks for that pass
- Split/merge KV caches appropriately
- ✅ Best of both worlds
- ❌ Most complex implementation
- ❌ Requires significant refactoring

## Fixes Applied

### 1. Orchestrator Integration

**File**: `src/domain/services/generation_orchestrator.py`

**Changes**:
- Added `attention_manager` parameter to `orchestrate_generation()`
- Build custom attention mask before each generation step
- Pass mask to token generator
- Register parallel token sets with attention manager

### 2. Model Wrapper Custom Mask Handling

**File**: `src/modeling/model_wrapper.py`

**Changes**:
- Intercept `custom_attention_mask` parameter
- Convert 2D mask to 4D format [1, 1, seq_len, seq_len]
- Replace HuggingFace `attention_mask` with custom mask
- Custom mask gets added as bias to attention scores

### 3. Use Case Wiring

**File**: `src/application/use_cases/generate_text.py`

**Changes**:
- Pass `attention_manager` from use case to orchestrator
- Ensures attention service is connected to generation pipeline

## Testing Status

### What Works Now

✅ Attention masks are built correctly by `AttentionService`
✅ Masks are passed through the full pipeline
✅ Model wrapper applies masks in HuggingFace-compatible format
✅ Can verify mask application via debug logging

### What Doesn't Work Yet

❌ KV cache prevents masks from having the intended effect
❌ Cannot actually test isolated vs visible modes with current architecture
❌ Need to disable KV cache, which breaks other assumptions

## Recommendations

### Immediate Actions

1. **Document the KV cache constraint** in TEMPO architecture docs
2. **Update experiment configs** to note KV cache requirement
3. **Add validation** to raise error if isolation is requested with KV cache
4. **Implement Option C** (hybrid approach) for proper parallel token handling

### For Experiments

1. **Do not run experiments** with current codebase expecting different behaviors
2. **Disable KV cache** in experiment configs (with performance trade-off)
3. **Fix sequence processing** to handle non-cached generation properly
4. **Re-design** parallel token generation to work with batched processing

## Conclusion

The investigation successfully identified why isolated and visible modes produced identical outputs: **the attention masking was completely non-functional due to missing integration points and architectural incompatibility with KV cache**.

The good news: We found and fixed the integration bugs.
The challenge: The architecture fundamentally cannot support both KV cache efficiency and custom attention masking without significant refactoring.

**Next Step**: Decide on architectural approach (A, B, or C above) before re-running any experiments.

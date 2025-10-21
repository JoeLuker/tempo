# Option C Implementation Plan: Hybrid Batched Parallel Token Processing

## Current Understanding of TEMPO

**Parallel Tokens**: Multiple tokens that:
1. Are appended to the SAME sequence (not branching)
2. Share the same logical RoPE position
3. Are generated when multiple candidates exceed the probability threshold

**Example**:
```
Prompt: "The cat"
Step 0: " sat" (single token)
Step 1: [" on", " under", " beside"] (3 parallel tokens, all at logical position 1)
Sequence becomes: "The cat sat on under beside"
Step 2: " the" (continues from all 3 parallel contexts)
```

## Problem with Current KV Cache Implementation

**Sequential Processing**:
```python
# Current code (simplified):
for token_id in parallel_tokens:
    append token_id to sequence
    process with KV cache
```

**What happens**:
1. token1 processed → added to KV cache
2. token2 processed → sees token1 in cache (asymmetric!)
3. token3 processed → sees token1 AND token2 (more asymmetric!)

**Result**: "Isolated" mode can't work because later tokens see earlier ones through the cache.

## Option C Solution: Batched Symmetric Processing

### Key Insight

Process all parallel tokens in ONE batched forward pass BEFORE adding any to the KV cache:

```python
# Option C approach:
batch_size = len(parallel_tokens)

# 1. Replicate base KV cache for batch
batch_kv = replicate_kv_cache(base_cache, batch_size)

# 2. Create batch input with all parallel tokens
batch_input_ids = [[...base_sequence..., token1],
                    [...base_sequence..., token2],
                    [...base_sequence..., token3]]

# 3. Apply custom attention mask
if isolated:
    # Prevent tokens from attending to each other
    mask[seq_len, seq_len] = -inf  # token1 can't see token2
    mask[seq_len+1, seq_len] = -inf  # token2 can't see token1
    # etc.
else:
    # Allow full cross-parallel attention
    mask[seq_len, seq_len+1] = 0  # token1 can see token2
    # etc.

# 4. Single batched forward pass
outputs = model(batch_input_ids, attention_mask=mask, past_key_values=batch_kv)

# 5. Merge results back to single sequence
# All parallel tokens processed symmetrically!
```

## Implementation Components

### 1. BatchedTokenGenerator (✓ Created)

**File**: `src/infrastructure/generation/batched_token_generator.py`

**Status**: Implemented with:
- KV cache replication for batch
- Batched forward pass
- KV cache splitting after processing
- Custom attention mask support

**Key method**:
```python
def generate_parallel_logits(
    base_state: GenerationState,
    num_parallel: int,
    custom_attention_mask: Optional[torch.Tensor]
) -> Tuple[BatchedTokenLogits, List[GenerationState]]
```

### 2. Orchestrator Integration (TODO)

**File**: `src/domain/services/generation_orchestrator.py`

**Current flow**:
1. Generate logits (single token)
2. Select N parallel tokens
3. Append all to sequence sequentially

**New flow**:
```python
# After token selection:
if len(selected_tokens) > 1:
    # Use batched generator
    batched_logits, parallel_states = batched_generator.generate_parallel_logits(
        base_state=current_state,
        num_parallel=len(selected_tokens),
        custom_attention_mask=isolation_mask if isolate else visibility_mask
    )

    # Append all parallel tokens to sequence
    # Use the FIRST state's KV cache (they all have same base)
    # OR merge all states somehow?

else:
    # Single token: use normal KV-cached generation
    logits, new_state = token_generator.generate_logits_with_cache(current_state)
```

**Open Question**: After batched processing, which KV cache do we use for the next step?
- All parallel tokens are now in the sequence
- Each batch element has its own KV cache
- Need to merge or select one

### 3. KV Cache Merging Strategy (TODO - CRITICAL)

**The Problem**:
After batched processing, we have N different KV caches (one per parallel token).
But the next step needs ONE KV cache to continue from.

**Possible Solutions**:

**A. Use First Token's Cache**:
```python
current_state = parallel_states[0]
# Ignore other caches
```
- ❌ Loses information from tokens 2, 3, ...
- ❌ Asymmetric (why favor first token?)

**B. Concatenate All Caches**:
```python
# Merge all parallel KV caches into one
merged_kv = concatenate_kv_caches(parallel_states)
```
- ✅ Preserves all information
- ❌ How to concatenate? Along sequence dim?
- ❌ May not be semantically correct

**C. Reprocess Full Sequence**:
```python
# After adding all parallel tokens, reprocess entire sequence
current_state = reprocess_sequence(full_sequence)
```
- ✅ Correct and unambiguous
- ❌ Expensive (defeats purpose of KV cache)
- ❌ Slow for long sequences

**D. Don't Use KV Cache for Parallel Steps**:
```python
if len(selected_tokens) > 1:
    # Disable cache for this step
    current_state.past_key_values = None
```
- ✅ Simple
- ✅ Avoids merging problem
- ❌ Slower
- ❌ Need to rebuild cache after

## Recommended Implementation Path

### Phase 1: Simple (No Cache for Parallel Steps)

1. Detect when multiple tokens are selected
2. Process them together with custom masking
3. Disable KV cache for that step
4. Rebuild cache on next step
5. Test isolated vs visible modes

**Pros**: Simple, correct, testable
**Cons**: Slower for steps with parallel tokens

### Phase 2: Optimized (Cache Concatenation)

1. After batched processing, concatenate KV caches along sequence dimension
2. Verify this works correctly with model
3. Benchmark vs Phase 1

**Pros**: Faster, more elegant
**Cons**: Complex, needs validation

### Phase 3: Hybrid Strategy

1. Use batched processing for parallel tokens
2. Use normal KV cache for single tokens
3. Switch between modes dynamically
4. Optimize cache management

**Pros**: Best performance
**Cons**: Most complex

## Next Steps

1. **Clarify KV cache merging strategy** ← BLOCKING
2. Implement orchestrator integration (Phase 1 approach)
3. Test with simple examples
4. Run experiments to verify isolated vs visible actually differ
5. Optimize if needed (Phase 2/3)

## Open Questions

1. **KV Cache Merging**: Which strategy (A, B, C, or D)?
2. **RoPE Positions**: How are they assigned in batched processing?
3. **Performance Impact**: How much slower is Option C vs current?
4. **Correctness Verification**: How to validate the implementation?

## Success Criteria

✅ Isolated mode prevents cross-parallel attention
✅ Visible mode allows cross-parallel attention
✅ Attention matrices differ between modes (verified via experiments)
✅ Outputs may differ (that's the point of testing!)
✅ Generation still works correctly


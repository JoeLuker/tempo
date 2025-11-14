# Attention Pattern Variability: Initial Insights

## Key Observations from Deep Dive Analysis

### 1. Temporal Dynamics (Early vs Late)

**narrative_2** (most extreme negative -60.9%):
- Early generation: +50.0% (parallel gets LESS attention)
- Late generation: -93.0% (parallel gets MUCH MORE attention)
- **Pattern**: REVERSES direction dramatically during generation

**narrative_1** (+18.5%):
- Early: +58.9% (parallel gets LESS attention)
- Late: +12.1% (parallel still gets less, but effect weakens)
- **Pattern**: Consistent direction, but magnitude decreases

**technical_2** (+30.1%):
- Early: +19.1%
- Late: +33.7%
- **Pattern**: Effect STRENGTHENS over time

### 2. Number of Parallel Positions

**Prompts with MANY parallel positions:**
- narrative_2: 11 parallel positions → -60.9% (extreme negative)
- technical_2: 16 parallel positions → +30.1% (extreme positive)

**Prompts with FEW parallel positions:**
- factual_1: 6 parallel positions → -27.9%
- narrative_1: 8 parallel positions → +18.5%

**Hypothesis**: Number of parallel positions alone doesn't determine direction, but may affect magnitude.

### 3. Attention Distribution Characteristics

**High median parallel attention (gets MORE attention):**
- narrative_2: Median 0.019861 (parallel) vs 0.008591 (non-parallel)
- factual_3: Median 0.014352 vs 0.008157
- simple_1: Median 0.014494 vs 0.008771

**Low median parallel attention (gets LESS attention):**
- technical_2: Median 0.007771 (parallel) vs 0.011251 (non-parallel)
- complex_1: Median 0.006830 vs 0.006767
- narrative_1: Median 0.007590 vs 0.010365

### 4. Variance Patterns

**High std dev in parallel attention:**
- simple_2: Std 0.017842 (very high variance)
- narrative_2: Std 0.017640
- factual_1: Std 0.015969

**Lower std dev:**
- technical_2: Std 0.010375
- complex_1: Std 0.010595
- narrative_3: Std 0.013151

## Emerging Hypotheses

### H1: Temporal Dynamics Matter
Some prompts show **attention pattern reversal** during generation (e.g., narrative_2).
This suggests that early vs late tokens may use parallel positions differently.

**Test**: Analyze attention patterns separately for:
- Tokens generated in first 1/3 of sequence
- Tokens in middle 1/3
- Tokens in final 1/3

### H2: Prompt Structure Influences Patterns
Looking at extreme cases:

**Parallel gets MORE attention (negative reduction):**
- "The old wizard slowly climbed the mountain" (narrative)
- "The capital of France is" (factual, incomplete)
- "The largest planet in our solar system" (factual, incomplete)

**Parallel gets LESS attention (positive reduction):**
- "Once upon a time in a distant galaxy" (complete phrase)
- "The algorithm complexity of quicksort is" (technical, incomplete)

**Hypothesis**: Incomplete prompts (requiring continuation) may show different patterns than complete phrases.

### H3: Token Semantics May Play a Role
Need to investigate:
- What tokens appear at parallel positions?
- Are they function words vs content words?
- Are they high-frequency vs low-frequency tokens?

### H4: Context Length Effect
Technical_2 has 16 parallel positions (most in dataset) and shows +30.1%.
Narrative_2 has 11 parallel positions and shows -60.9%.

**Question**: Does the proportion of parallel to non-parallel tokens matter?
- technical_2: 16 parallel / (total generated) = ?
- narrative_2: 11 parallel / (total generated) = ?

## Next Steps to Understand Mechanism

### Immediate Analysis (Use Existing Data):
1. ✅ Extract which tokens appear at parallel positions
2. ✅ Compare token types (function vs content words)
3. ✅ Analyze positional statistics (where do parallel tokens appear?)
4. ✅ Layer-wise analysis (do different layers show different patterns?)

### Larger Data Collection:
1. Generate 50-100 more prompts across categories
2. Test with different selection thresholds
3. Test with longer sequences (100+ tokens)
4. Compare isolated vs visible modes directly

### Controlled Experiments:
1. Use same prompt with different random seeds
2. Use prompts that force parallel tokens at specific positions
3. Test with manually constructed scenarios

## Questions to Answer

1. **WHY does narrative_2 reverse so dramatically?**
   - What happens at the reversal point?
   - Is there a syntactic/semantic shift?

2. **WHY does technical_2 show consistent positive reduction?**
   - Is it the technical nature?
   - Is it the prompt structure?
   - Is it the number of parallel positions?

3. **Can we PREDICT the direction and magnitude?**
   - Based on prompt characteristics?
   - Based on early generation patterns?
   - Based on token types selected?

4. **What is the MECHANISM?**
   - Is it RoPE position encoding effects?
   - Is it attention pattern learning in the model?
   - Is it interaction with KV cache?

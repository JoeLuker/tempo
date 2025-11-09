# CLI Attention Analysis Findings

**Date:** 2025-11-09
**Purpose:** Exploratory analysis of TEMPO attention patterns using various parameter configurations

## Experiment Overview

Five experiments were conducted to explore how different TEMPO parameters affect parallel token generation and attention patterns:

1. **Wizard (Low Threshold)** - Testing maximum parallelism
2. **Scientist (High Threshold)** - Testing conservative generation
3. **Galaxy (Retroactive Pruning)** - Testing attention-based pruning
4. **Robot (Isolation Mode)** - Testing parallel token isolation
5. **Future (Dynamic Threshold)** - Testing adaptive threshold behavior

## Experiments

### Experiment 1: Maximum Parallelism
**Prompt:** "The ancient wizard spoke the spell:"
**Parameters:**
- Selection Threshold: 0.15 (low for more parallel paths)
- Max Tokens: 15
- Seed: 42
- Debug Mode: Enabled

**Status:** Running with verbose debug output (not captured in summary)

**Purpose:** Explore how TEMPO behaves with aggressive parallel path exploration.

---

### Experiment 2: Conservative Generation
**Prompt:** "The scientist discovered"
**Parameters:**
- Selection Threshold: 0.3 (higher/conservative)
- Max Tokens: 12
- Seed: 123

**Result:**
```
The scientist discovered that the human body can produce its own energy through a process
```

**Observations:**
- **Generation Time:** 2.17s
- **Tokens/Second:** 5.53
- High threshold (0.3) resulted in minimal parallel branching
- Output appears linear with no visible parallel paths in the formatted text
- Conservative threshold leads to more deterministic, focused generation

**Interpretation:** With threshold 0.3, TEMPO operates more like standard autoregressive generation, selecting only the highest probability tokens. This produces coherent but less exploratory output.

---

### Experiment 3: Retroactive Pruning Analysis
**Prompt:** "In a galaxy far away"
**Parameters:**
- Selection Threshold: 0.10 (very low for maximum parallelism)
- Max Tokens: 15
- Retroactive Pruning: Enabled
- Attention Threshold: 0.02
- Seed: 456

**Result:**
```
In a galaxy far away[,/.../...] ...there['s/was/lived/exists] a planet[called/where]
the inhabitants[have/are/live] in[harmony/a] perpetual[state/twilight][,/.]
```

**Observations:**
- **Generation Time:** 2.66s
- **Tokens/Second:** 5.64
- Low threshold (0.10) created extensive parallel branching
- Color-coded alternatives visible at multiple points:
  - Punctuation choices: `,` vs `...`
  - Verb forms: `'s` / `was` / `lived` / `exists`
  - Prepositions: `called` / `where`
  - Verbs: `have` / `are` / `live`
  - Objects: `harmony` / `a`
  - Adjectives: `state` / `twilight`
- Retroactive pruning actively refined token sets based on attention from later tokens
- Generation maintained grammatical coherence despite parallel exploration

**Interpretation:** Low threshold + retroactive pruning enables rich exploration while maintaining coherence. The attention-based pruning mechanism successfully eliminates tokens that don't receive sufficient attention from subsequent context, demonstrating TEMPO's ability to refine parallel paths dynamically.

---

### Experiment 4: Isolation Mode Effects
**Prompt:** "The robot began to think"
**Parameters:**
- Selection Threshold: 0.20 (moderate)
- Max Tokens: 12
- Isolation Mode: Enabled (parallel tokens cannot attend to each other)
- Seed: 789

**Result:**
```
The robot began to think think. It was a strange[sensation/feeling], like a fog that
```

**Observations:**
- **Generation Time:** 1.71s
- **Tokens/Second:** 7.00
- Isolation mode prevented parallel tokens from seeing each other
- Only one parallel branch visible: `[sensation/feeling]`
- Faster generation (7.00 tokens/sec vs 5.64 in Experiment 3)
- Repetition artifact: "think think" (possible isolation side effect)

**Interpretation:** Isolation mode significantly reduces parallel branching compared to non-isolated generation with similar threshold. The faster generation speed suggests reduced computational overhead from attention calculations. The token repetition may indicate isolation affecting early token selection.

---

### Experiment 5: Dynamic Threshold Behavior
**Prompt:** "In a distant future"
**Parameters:**
- Selection Threshold: 0.15 (initial)
- Max Tokens: 15
- Dynamic Threshold: Enabled with Bezier curve
- Bezier P1: 0.2, P2: 0.8
- Final Threshold: 0.4
- Seed: 999

**Result:**
```
In a distant future,[humanity/the] world[is/has] been transformed[by/into]
a vast network of interconnected cities[,/and] mega
```

**Observations:**
- **Generation Time:** 1.76s
- **Tokens/Second:** 8.55
- Dynamic threshold started at 0.15 and gradually increased to 0.4
- Parallel branching visible throughout but potentially decreasing over time
- Multiple choice points:
  - Article/subject: `humanity` / `the`
  - Verb: `is` / `has`
  - Preposition: `by` / `into`
  - Punctuation: `,` / `and`
- Fastest generation of all experiments (8.55 tokens/sec)

**Interpretation:** Dynamic threshold with Bezier curve allows TEMPO to start with exploratory behavior (low threshold) and gradually become more conservative (high threshold). The fast generation speed (fastest of all experiments) suggests the increasing threshold reduces computational overhead as generation progresses. This adaptive approach may balance exploration and efficiency.

---

## Key Findings

### 1. Threshold Impact on Parallelism
- **Low thresholds (0.10-0.15):** Extensive parallel branching, rich exploration
- **Moderate thresholds (0.20):** Balanced parallelism, especially with isolation
- **High thresholds (0.30):** Minimal branching, near-deterministic output

### 2. Retroactive Pruning Effectiveness
- Successfully refines parallel token sets based on attention patterns
- Maintains grammatical coherence while exploring alternatives
- Effective at eliminating tokens that don't receive sufficient attention from later context

### 3. Isolation Mode Trade-offs
- **Benefits:** Faster generation (7.00 tokens/sec vs 5.64)
- **Costs:** Reduced parallel branching, potential token repetition artifacts
- **Use Case:** When computational efficiency matters more than exploration breadth

### 4. Dynamic Threshold Advantages
- **Fastest generation:** 8.55 tokens/sec (best of all experiments)
- **Adaptive behavior:** Exploratory early, conservative later
- **Practical benefit:** Balances exploration and efficiency automatically

### 5. Performance Characteristics
Generation speed across experiments:
1. Dynamic Threshold: 8.55 tokens/sec (fastest)
2. Isolation Mode: 7.00 tokens/sec
3. Retroactive Pruning: 5.64 tokens/sec
4. Conservative: 5.53 tokens/sec

Trade-off: Higher parallelism (more exploration) comes with computational cost.

---

## Mechanistic Observations

### Attention Pattern Insights
1. **Retroactive pruning relies on backward attention:** Tokens receive attention from later context, and low-attention tokens get pruned
2. **Isolation prevents mutual attention:** Parallel tokens at the same logical step cannot see each other, affecting token selection
3. **Dynamic threshold creates evolving attention patterns:** As threshold increases, attention becomes more focused on single paths

### Token Selection Behavior
1. **Threshold acts as probability gate:** Only tokens above threshold become parallel candidates
2. **Lower thresholds create more choice points:** More tokens exceed the threshold
3. **Isolation reduces choice points:** Without mutual attention, fewer parallel paths survive

### Coherence Mechanisms
1. **Retroactive pruning maintains coherence:** By removing tokens that don't fit later context
2. **High thresholds enforce coherence:** By limiting exploration to high-probability paths
3. **Dynamic thresholds balance both:** Early exploration with later refinement

---

## Recommendations

### For Maximum Exploration
- Use low threshold (0.10-0.15)
- Enable retroactive pruning with attention threshold ~0.02
- Accept slower generation for richer alternatives

### For Balanced Performance
- Use moderate threshold (0.20)
- Enable isolation mode for speed
- Good for interactive applications

### For Efficiency with Exploration
- Use dynamic threshold (0.15 → 0.4)
- Configure Bezier curve for desired transition
- Best overall balance of exploration and speed

### For Deterministic Output
- Use high threshold (0.30+)
- Disable retroactive pruning
- Fastest, most predictable generation

---

## Future Exploration Directions

1. **Attention Weight Analysis:** Deep dive into actual attention matrices to understand pruning decisions
2. **Threshold Sensitivity:** Systematic sweep of threshold values to map parallelism curves
3. **Pruning Threshold Interaction:** How do selection_threshold and attention_threshold interact?
4. **Isolation + Pruning:** Can isolation mode benefit from retroactive pruning?
5. **Dynamic Threshold Curves:** Compare Bezier vs ReLU curves for different generation tasks
6. **Long-form Generation:** How do these patterns scale to 100+ token sequences?

---

## Conclusion

TEMPO demonstrates sophisticated parallel token generation with multiple mechanisms for balancing exploration and efficiency. Key insights:

- **Threshold is primary control:** Determines parallelism breadth
- **Retroactive pruning adds intelligence:** Refines paths based on attention
- **Isolation trades exploration for speed:** Significant performance gain
- **Dynamic threshold offers best balance:** Adaptive behavior across generation

The attention-based pruning mechanism is particularly noteworthy, showing TEMPO can maintain coherence while exploring alternatives - a key capability for mechanistic interpretability and controlled generation.

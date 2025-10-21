# TEMPO Mechanistic Interpretability Experiments

This directory contains experimental configurations for investigating TEMPO's internal mechanisms.

## Experiment Goals

We're testing the hypothesis: **"The model handles same-position parallel tokens just fine"**

### Key Questions:
1. Do parallel tokens actually attend to each other in visible mode?
2. Are probability distributions identical or just top-k selections?
3. How does the model handle multiple tokens at the same RoPE position?
4. Why do both isolation modes produce identical results?

## Experiments

### Experiment 1: Attention Weight Capture
**Files:**
- `exp1_attention_weights_isolated.yaml`
- `exp1_attention_weights_visible.yaml`

**Goal:** Capture full attention matrices from both modes

**Measures:**
- Complete attention weights at each generation step
- Layer-wise attention patterns
- Storage in NumPy format for analysis

**Analysis:**
```python
# Compare attention patterns
isolated_attn = np.load('exp1_isolated/attention_weights.npz')
visible_attn = np.load('exp1_visible/attention_weights.npz')

# Check if patterns differ
```

---

### Experiment 2: Logits Distribution Comparison
**File:** `exp2_logits_comparison.yaml`

**Goal:** Compare full probability distributions between modes

**Measures:**
- Full vocabulary logits (not just top-k)
- KL divergence between distributions
- Cosine similarity of probability vectors
- Top-k overlap analysis

**Expected Outcomes:**
- If distributions are identical → prior context dominates
- If different but same top-k → subtle differences that don't affect selection
- If completely different → modes have fundamentally different behavior

---

### Experiment 3: Cross-Parallel Attention
**File:** `exp3_cross_parallel_attention.yaml`

**Goal:** Measure attention between parallel tokens in visible mode

**Measures:**
- Attention weights from parallel token i to parallel token j
- Statistics: mean, max, variance of cross-parallel attention
- Comparison to attention to prior context

**Key Metric:**
```
cross_parallel_score = mean(attention[parallel_i, parallel_j])
vs
prior_context_score = mean(attention[parallel_i, context])
```

**Hypothesis Test:**
- H0: cross_parallel_score ≈ 0 (tokens ignore siblings)
- H1: cross_parallel_score > 0 (tokens attend to siblings)

---

### Experiment 4: KV Cache Inspection
**File:** `exp4_kv_cache_inspection.yaml`

**Goal:** Verify RoPE position sharing and cache structure

**Measures:**
- RoPE position map: physical → logical positions
- KV cache contents at each step
- Verification that parallel tokens share logical positions

**Validates:**
- Parallel tokens are actually at same RoPE position
- Cache correctly stores all parallel tokens
- Next token properly queries all cached positions

---

### Experiment 5: Edge Case - High Parallelism
**File:** `exp5_edge_case_high_parallelism.yaml`

**Goal:** Test with maximum parallel width (threshold = 0.03)

**Measures:**
- Behavior with 10+ parallel tokens per step
- Whether modes still produce identical results
- Computational performance impact
- Potential numerical stability issues

**Expected:**
- Higher memory usage
- Longer generation time (especially isolated mode)
- Stress test of attention masking mechanism

---

## Running Experiments

### Prerequisites
```bash
# Ensure you're on the experiments/mech-interp branch
git checkout experiments/mech-interp

# Install dependencies (if needed)
pip install numpy scipy matplotlib
```

### Run Single Experiment
```bash
python3 run_experiment.py --config experiments/exp1_attention_weights_isolated.yaml
```

### Run All Experiments
```bash
./run_all_experiments.sh
```

### Analyze Results
```bash
python3 analyze_experiment_results.py --experiment exp1
```

---

## Expected Directory Structure

```
experiments/
├── README.md (this file)
├── exp1_attention_weights_isolated.yaml
├── exp1_attention_weights_visible.yaml
├── exp2_logits_comparison.yaml
├── exp3_cross_parallel_attention.yaml
├── exp4_kv_cache_inspection.yaml
├── exp5_edge_case_high_parallelism.yaml
├── results/
│   ├── exp1_isolated/
│   │   ├── attention_weights.npz
│   │   └── generation_log.json
│   ├── exp1_visible/
│   │   ├── attention_weights.npz
│   │   └── generation_log.json
│   ├── exp2_logits/
│   │   ├── logits_distributions.npz
│   │   ├── kl_divergence.json
│   │   └── analysis.txt
│   └── ...
└── analysis/
    ├── attention_comparison.py
    ├── logits_analysis.py
    └── visualization.py
```

---

## Implementation Status

### ✅ Completed:
- Experiment design
- YAML configurations
- Documentation

### 🚧 TODO:
- [ ] Implement attention capture in TokenGeneratorImpl
- [ ] Implement logits capture in generation strategy
- [ ] Create KV cache inspection hooks
- [ ] Build analysis scripts
- [ ] Create visualization tools
- [ ] Run experiments and collect data
- [ ] Analyze results and draw conclusions

---

## Expected Findings

### Scenario A: Prior Context Dominates
- **Evidence:** Identical logits distributions, zero cross-parallel attention
- **Conclusion:** Isolation doesn't matter because prior context determines everything

### Scenario B: Subtle Cross-Attention
- **Evidence:** Non-zero but small cross-parallel attention, nearly identical logits
- **Conclusion:** Parallel tokens do attend to each other, but it doesn't affect top-k selection

### Scenario C: Significant Interaction
- **Evidence:** High cross-parallel attention, different logits but same top-k
- **Conclusion:** Model uses sibling information but robustly arrives at same conclusions

### Scenario D: Fundamental Difference
- **Evidence:** Different logits, different attention patterns, but somehow same top-k
- **Conclusion:** Something deeper is going on (investigation needed)

---

## Next Steps After Data Collection

1. **Quantitative Analysis**
   - Statistical tests on attention differences
   - Distribution similarity metrics
   - Performance profiling

2. **Visualization**
   - Attention heatmaps
   - Probability distribution plots
   - Position embedding visualizations

3. **Conclusions**
   - Update ISOLATION_COMPARISON_FINDINGS.md
   - Recommend default isolation mode
   - Identify any surprising behaviors

4. **Publication**
   - Write up findings
   - Create visualizations
   - Share insights with community

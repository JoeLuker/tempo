# TEMPO Analysis Scaling Implementation Plan

## Goal
Scale mechanistic interpretability analysis from 1 prompt to 100+ prompts with statistical rigor.

## Three-Layer Optimization Stack

### Layer 1: Model Persistence (30 min implementation)
**Speedup:** 2x (200 min → 100 min for 100 experiments)

**Implementation:**
```python
class PersistentExperimentRunner:
    def __init__(self):
        self.model_wrapper = load_tempo_components_once()
        self.tokenizer = self.model_wrapper.tokenizer

    def run_experiment(self, prompt, seed, mode):
        # Clear KV cache
        self.model_wrapper.clear_kv_cache()

        # Run experiment with existing model
        result = generate_with_attention_capture(
            model=self.model_wrapper,
            prompt=prompt,
            seed=seed,
            isolated=mode == "isolated"
        )
        return result

    def run_suite(self, prompts, seeds, modes):
        all_results = []
        for prompt in prompts:
            for seed in seeds:
                for mode in modes:
                    result = self.run_experiment(prompt, seed, mode)
                    all_results.append(result)
        return all_results
```

**Files to modify:**
- Create: `experiments/persistent_runner.py`
- Modify: Wrap existing experiment runner with model reuse

**Testing:**
- Run 10 experiments sequentially
- Verify memory doesn't grow
- Confirm results match baseline

---

### Layer 2: Multi-Process Batching (2-3 hours implementation)
**Speedup:** 3-5x on top of Layer 1 (100 min → 20-30 min for 100 experiments)

**Implementation:**
```python
import multiprocessing as mp
from functools import partial

class ParallelExperimentSuite:
    def __init__(self, num_workers=5):
        self.num_workers = num_workers

    def worker_process(self, experiment_batch):
        # Each worker has its own persistent runner
        runner = PersistentExperimentRunner()  # Layer 1
        return [runner.run_experiment(*exp) for exp in experiment_batch]

    def run_all(self, prompts, seeds, modes):
        # Create experiment list
        experiments = [
            (prompt, seed, mode)
            for prompt in prompts
            for seed in seeds
            for mode in modes
        ]

        # Split into batches for workers
        batch_size = len(experiments) // self.num_workers
        batches = [
            experiments[i:i+batch_size]
            for i in range(0, len(experiments), batch_size)
        ]

        # Run in parallel
        with mp.Pool(self.num_workers) as pool:
            results = pool.map(self.worker_process, batches)

        # Flatten results
        return [r for batch in results for r in batch]
```

**Files to modify:**
- Create: `experiments/parallel_suite.py`
- Integrate with Layer 1's PersistentExperimentRunner

**Testing:**
- Run with num_workers=2, small experiment set
- Monitor memory per process
- Verify no result corruption

**Limitations:**
- Memory: 5 workers × ~8GB model = ~40GB RAM needed
- Solution: Reduce num_workers on smaller machines

---

### Layer 3: BatchedKVCache (4-6 hours implementation)
**Speedup:** 2-3x on top of Layer 2 (20-30 min → 10-15 min for 100 experiments)

**Implementation:**
```python
# 1. Integrate mlx_parallm's BatchedKVCache
from mlx_parallm.models import BatchedKVCache

class BatchedExperimentRunner:
    def __init__(self, batch_size=10):
        self.batch_size = batch_size
        self.model_wrapper = load_tempo_components_once()

        # Replace KVCache with BatchedKVCache
        self._convert_to_batched_kv_cache()

    def _convert_to_batched_kv_cache(self):
        # Modify model architecture to use BatchedKVCache
        for layer in self.model_wrapper.model.layers:
            if hasattr(layer, 'self_attn'):
                layer.self_attn.kv_cache = BatchedKVCache(
                    self.batch_size
                )

    def run_batch(self, prompts, seeds, mode):
        # Process multiple prompts simultaneously
        assert len(prompts) <= self.batch_size

        # Prepare batch inputs
        batch_inputs = self._prepare_batch(prompts, seeds)

        # Single forward pass for all prompts
        results = self.model_wrapper.generate_batch(
            batch_inputs,
            isolated=(mode == "isolated")
        )

        return results
```

**Files to modify:**
- Create: `src/algorithms/generation/batched_kv_cache.py`
- Modify: `src/modeling/model_wrapper.py` to support batched generation
- Modify: `src/infrastructure/model/attention_patcher.py` for batch attention capture

**Architecture changes:**
1. Copy model architecture files to use BatchedKVCache
2. Modify attention capture to handle batch dimension
3. Update data structures to support batch results

**Testing:**
- Run batch_size=2 with 2 simple prompts
- Verify both get correct independent results
- Check attention matrices have batch dimension

**Complexity:**
- Requires understanding TEMPO's model internals
- Must maintain isolation/visible mode compatibility
- Attention capture needs batch-aware indexing

---

## Integration Strategy

### Phase 1: Quick Win (Week 1)
- Implement Layer 1 only
- Run 100 experiments (10 prompts × 5 seeds × 2 modes)
- Time: ~100 minutes
- Deliverable: Multi-prompt statistical analysis

### Phase 2: Production Speed (Week 2)
- Add Layer 2 on top of Layer 1
- Run 100 experiments in ~25 minutes
- Deliverable: Fast replication capability

### Phase 3: Ultimate Optimization (Week 3-4)
- Add Layer 3 on top of Layer 1+2
- Run 100 experiments in ~12 minutes
- Deliverable: Publication-ready analysis suite

---

## Experiment Design

### Prompts (10 diverse categories)
```python
prompts = {
    "narrative": [
        "Once upon a time",
        "The hero embarked on"
    ],
    "factual": [
        "The scientist discovered",
        "According to the research"
    ],
    "technical": [
        "The algorithm processes",
        "The system architecture"
    ],
    "dialogue": [
        "She said to him",
        "He couldn't believe"
    ],
    "descriptive": [
        "The ancient civilization",
        "The magnificent landscape"
    ]
}
```

### Parameters
- Seeds: [42, 123, 456, 789, 999]
- Modes: ["isolated", "visible"]
- Max tokens: 10 (same as original)
- Threshold: 0.1 (same as original)

### Total Experiments
- 10 prompts × 5 seeds × 2 modes = **100 experiments**

---

## Expected Outcomes

### Statistical Rigor
- Mean parallel token down-weighting across prompts: μ ± 95% CI
- Variance between prompts: σ²
- Outlier detection: Which prompts deviate?
- Effect size consistency: Cohen's d per prompt

### New Insights
- Prompt-category effects (narrative vs technical)
- Seed sensitivity (how much does randomness matter?)
- Generalizability confidence (do findings hold universally?)

### Publication Quality
- 100 experiments >> 1 experiment
- Cross-prompt replication
- Robust statistical claims
- "We tested across 10 diverse prompts..." (credible)

---

## Resource Requirements

### Computational
- Layer 1: 1 × GPU (current setup)
- Layer 2: 5 × CPU cores (for 5 workers)
- Layer 3: 1 × GPU with batched processing

### Memory
- Layer 1: ~8GB (single model instance)
- Layer 2: ~40GB (5 workers × 8GB)
- Layer 3: ~12GB (batched KV cache overhead)

### Time Investment
- Layer 1: 30 min implementation + 100 min runtime = **2 hours total**
- Layer 2: 3 hours implementation + 25 min runtime = **3.5 hours total**
- Layer 3: 5 hours implementation + 12 min runtime = **5 hours total**

### Storage
- Attention matrices: ~300MB per experiment
- 100 experiments × 300MB = **30GB**
- With compression: ~10GB

---

## Success Criteria

### Layer 1
- ✅ 100 experiments complete in <120 minutes
- ✅ Memory stable (no leaks)
- ✅ Results match single-experiment baseline

### Layer 2
- ✅ 100 experiments complete in <30 minutes
- ✅ All workers complete successfully
- ✅ Results identical to Layer 1 (order-independent)

### Layer 3
- ✅ 100 experiments complete in <15 minutes
- ✅ Batched results match sequential results
- ✅ Attention capture works with batching

---

## Risk Mitigation

### Memory Issues
- **Risk:** 5 workers × 8GB = 40GB RAM
- **Mitigation:** Start with 2-3 workers, scale up
- **Fallback:** Use Layer 1 only on smaller machines

### BatchedKVCache Complexity
- **Risk:** Complex integration, hard to debug
- **Mitigation:** Implement incrementally, test at batch_size=2 first
- **Fallback:** Layer 1+2 gives 6.7x speedup already

### Result Validation
- **Risk:** Batching introduces bugs, wrong results
- **Mitigation:** Cross-validate 10% of batched results against sequential
- **Requirement:** Statistical tests to detect outliers

---

## Next Steps

1. **Immediate:** Implement Layer 1 (30 min)
2. **This week:** Run 100 experiments, analyze multi-prompt results
3. **Next week:** Implement Layer 2 if replication needed
4. **Future:** Implement Layer 3 for production analysis suite

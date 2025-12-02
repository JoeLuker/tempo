#!/usr/bin/env python3
"""Stress test for memory bank at scale.

Tests:
1. Many targets (10+) in a single prompt
2. Very long context (1000+ tokens)
3. Delayed recall (ask about first item after many others)
4. Memory bank size limits
5. Different types of information (names, numbers, facts)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import time
from dataclasses import dataclass

from src.hebbian.mlx import HebbianMLXEngine
from src.hebbian.config import HebbianConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    name: str
    targets: int
    recalled: int
    recall_rate: float
    evictions: int
    memory_entries: int
    time_seconds: float


def test_many_targets(engine: HebbianMLXEngine, n_targets: int = 10) -> TestResult:
    """Test recall of many different targets."""
    # Generate unique targets
    targets = [f"CODE{i:03d}" for i in range(n_targets)]

    # Build prompt with all targets
    facts = " ".join([f"Item {i+1} is {t}." for i, t in enumerate(targets)])
    filler = "The weather is nice. Birds are singing. Trees are green. " * 10

    prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Remember these items: {facts}

{filler}

List all the items you remember, one per line.<|im_end|>
<|im_start|>assistant
"""

    start = time.time()
    engine.clear()
    result = engine.generate_with_metrics(prompt, max_tokens=400, temperature=0.0)
    elapsed = time.time() - start

    # Count how many targets appear in response
    recalled = sum(1 for t in targets if t in result["text"].upper())
    stats = engine.get_stats()

    return TestResult(
        name=f"many_targets_{n_targets}",
        targets=n_targets,
        recalled=recalled,
        recall_rate=recalled / n_targets,
        evictions=stats["positions_processed"] - stats["cache_size"],
        memory_entries=stats["memory_entries"],
        time_seconds=elapsed,
    )


def test_long_context(engine: HebbianMLXEngine, filler_sentences: int = 50) -> TestResult:
    """Test recall with very long intervening context."""
    target = "SECRETXYZ"

    # Generate lots of filler
    fillers = [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump.",
        "The five boxing wizards jump quickly.",
        "Sphinx of black quartz, judge my vow.",
    ]
    filler = " ".join([fillers[i % len(fillers)] for i in range(filler_sentences)])

    prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
The password is {target}. {filler} What is the password?<|im_end|>
<|im_start|>assistant
"""

    start = time.time()
    engine.clear()
    result = engine.generate_with_metrics(prompt, max_tokens=150, temperature=0.0)
    elapsed = time.time() - start

    recalled = 1 if target in result["text"].upper() else 0
    stats = engine.get_stats()

    return TestResult(
        name=f"long_context_{filler_sentences}",
        targets=1,
        recalled=recalled,
        recall_rate=float(recalled),
        evictions=stats["positions_processed"] - stats["cache_size"],
        memory_entries=stats["memory_entries"],
        time_seconds=elapsed,
    )


def test_first_vs_last(engine: HebbianMLXEngine) -> tuple[TestResult, TestResult]:
    """Test if first item is recalled as well as last item."""
    first_target = "FIRST111"
    last_target = "LAST999"

    filler = "The weather is nice. " * 30

    # Test first
    prompt_first = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Code A is {first_target}. {filler} Code B is {last_target}. What is Code A?<|im_end|>
<|im_start|>assistant
"""

    engine.clear()
    start = time.time()
    result = engine.generate_with_metrics(prompt_first, max_tokens=150, temperature=0.0)
    elapsed = time.time() - start

    recalled_first = 1 if first_target in result["text"].upper() else 0
    stats = engine.get_stats()

    first_result = TestResult(
        name="first_item",
        targets=1,
        recalled=recalled_first,
        recall_rate=float(recalled_first),
        evictions=stats["positions_processed"] - stats["cache_size"],
        memory_entries=stats["memory_entries"],
        time_seconds=elapsed,
    )

    # Test last (should be in window, easy)
    prompt_last = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Code A is {first_target}. {filler} Code B is {last_target}. What is Code B?<|im_end|>
<|im_start|>assistant
"""

    engine.clear()
    start = time.time()
    result = engine.generate_with_metrics(prompt_last, max_tokens=150, temperature=0.0)
    elapsed = time.time() - start

    recalled_last = 1 if last_target in result["text"].upper() else 0
    stats = engine.get_stats()

    last_result = TestResult(
        name="last_item",
        targets=1,
        recalled=recalled_last,
        recall_rate=float(recalled_last),
        evictions=stats["positions_processed"] - stats["cache_size"],
        memory_entries=stats["memory_entries"],
        time_seconds=elapsed,
    )

    return first_result, last_result


def test_different_types(engine: HebbianMLXEngine) -> TestResult:
    """Test recall of different information types."""
    facts = {
        "name": "John Smith",
        "phone": "555-1234",
        "city": "Tokyo",
        "color": "purple",
        "number": "42",
    }

    facts_text = " ".join([f"The {k} is {v}." for k, v in facts.items()])
    filler = "Life is beautiful. Nature is amazing. Time flies by. " * 20

    prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{facts_text} {filler} List all the facts you remember.<|im_end|>
<|im_start|>assistant
"""

    start = time.time()
    engine.clear()
    result = engine.generate_with_metrics(prompt, max_tokens=150, temperature=0.0)
    elapsed = time.time() - start

    # Check each value
    recalled = sum(1 for v in facts.values() if v.upper() in result["text"].upper())
    stats = engine.get_stats()

    return TestResult(
        name="different_types",
        targets=len(facts),
        recalled=recalled,
        recall_rate=recalled / len(facts),
        evictions=stats["positions_processed"] - stats["cache_size"],
        memory_entries=stats["memory_entries"],
        time_seconds=elapsed,
    )


def run_stress_tests():
    """Run all stress tests."""
    logger.info("=" * 60)
    logger.info("MEMORY BANK STRESS TEST")
    logger.info("=" * 60)

    # Create engine with memory enabled (uses default config)
    config = HebbianConfig(
        memory_enabled=True,
        window_size=32,
        n_sink_tokens=4,
    )
    engine = HebbianMLXEngine(config=config)

    # Also test baseline for comparison
    baseline_config = HebbianConfig(
        memory_enabled=False,
        window_size=32,
        n_sink_tokens=4,
    )
    baseline_engine = HebbianMLXEngine(config=baseline_config)

    results = []

    # Test 1: Many targets
    logger.info("\n=== TEST 1: Many Targets ===")
    for n in [5, 10, 15]:
        mem_result = test_many_targets(engine, n)
        base_result = test_many_targets(baseline_engine, n)
        results.append(("memory", mem_result))
        results.append(("baseline", base_result))
        logger.info(f"  {n} targets: memory={mem_result.recall_rate:.0%} baseline={base_result.recall_rate:.0%}")

    # Test 2: Long context
    logger.info("\n=== TEST 2: Long Context ===")
    for n in [20, 50, 100]:
        mem_result = test_long_context(engine, n)
        base_result = test_long_context(baseline_engine, n)
        results.append(("memory", mem_result))
        results.append(("baseline", base_result))
        logger.info(f"  {n} filler sentences: memory={mem_result.recall_rate:.0%} baseline={base_result.recall_rate:.0%} (evictions={mem_result.evictions})")

    # Test 3: First vs Last
    logger.info("\n=== TEST 3: First vs Last Item ===")
    mem_first, mem_last = test_first_vs_last(engine)
    base_first, base_last = test_first_vs_last(baseline_engine)
    results.extend([("memory", mem_first), ("memory", mem_last)])
    results.extend([("baseline", base_first), ("baseline", base_last)])
    logger.info(f"  First item: memory={mem_first.recall_rate:.0%} baseline={base_first.recall_rate:.0%}")
    logger.info(f"  Last item:  memory={mem_last.recall_rate:.0%} baseline={base_last.recall_rate:.0%}")

    # Test 4: Different types
    logger.info("\n=== TEST 4: Different Information Types ===")
    mem_result = test_different_types(engine)
    base_result = test_different_types(baseline_engine)
    results.append(("memory", mem_result))
    results.append(("baseline", base_result))
    logger.info(f"  5 facts: memory={mem_result.recalled}/5 baseline={base_result.recalled}/5")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    mem_results = [r for t, r in results if t == "memory"]
    base_results = [r for t, r in results if t == "baseline"]

    mem_total = sum(r.recalled for r in mem_results)
    mem_targets = sum(r.targets for r in mem_results)
    base_total = sum(r.recalled for r in base_results)
    base_targets = sum(r.targets for r in base_results)

    logger.info(f"Memory bank: {mem_total}/{mem_targets} ({mem_total/mem_targets:.0%})")
    logger.info(f"Baseline:    {base_total}/{base_targets} ({base_total/base_targets:.0%})")

    improvement = (mem_total/mem_targets) - (base_total/base_targets)
    logger.info(f"\nImprovement: +{improvement:.0%}")

    return results


if __name__ == "__main__":
    run_stress_tests()

#!/usr/bin/env python3
"""Cross-generation learning test for Hebbian consolidation.

Tests whether Hebbian modifications that persist across generations
enable learning. The key insight: force a correct answer in generation 1,
let the associations form, then test recall in generation 2.

This tests: "inference as learning" - can multiple generations build up
useful weight modifications?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import mlx.core as mx
from mlx_lm import load

from src.hebbian.mlx import HebbianMLXEngine
from src.hebbian.config import HebbianConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class CrossGenerationTest:
    """Test learning across multiple generations."""

    def __init__(self, model_name: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"):
        logger.info(f"Loading model: {model_name}")
        self.model, self.tokenizer = load(model_name)
        self.model_name = model_name
        logger.info("Model loaded")

    def _create_engine(self, config: HebbianConfig) -> HebbianMLXEngine:
        """Create engine reusing loaded model."""
        engine = HebbianMLXEngine.__new__(HebbianMLXEngine)
        engine.model_name = self.model_name
        engine.config = config
        engine.model = self.model
        engine.tokenizer = self.tokenizer

        engine.args = self.model.args
        engine.n_layers = engine.args.num_hidden_layers
        engine.n_heads = engine.args.num_attention_heads
        engine.n_kv_heads = engine.args.num_key_value_heads
        engine.hidden_dim = engine.args.hidden_size
        engine.head_dim = engine.args.head_dim
        engine.k_dim = engine.n_kv_heads * engine.head_dim

        engine._init_hebbian_state()
        return engine

    def test_cross_generation_learning(self):
        """Test if learning accumulates across generations."""
        logger.info("=" * 60)
        logger.info("CROSS-GENERATION LEARNING TEST")
        logger.info("=" * 60)
        logger.info("\nHypothesis: If we process multiple related prompts,")
        logger.info("Hebbian modifications should accumulate and improve later responses.\n")

        # Test configurations
        configs = [
            ("baseline", HebbianConfig(update_scale=0.0, window_size=32, n_sink_tokens=4)),
            ("V_1e-3", HebbianConfig(update_scale=1e-3, window_size=32, n_sink_tokens=4, update_target="v")),
            ("V_1e-2", HebbianConfig(update_scale=1e-2, window_size=32, n_sink_tokens=4, update_target="v")),
        ]

        # Training prompts - long enough to trigger evictions (window=32)
        # Need 50+ tokens per prompt to ensure evictions happen
        training_prompts = [
            "The secret code is ALPHA7. Remember ALPHA7. ALPHA7 is important. "
            "This code ALPHA7 must be memorized. The password ALPHA7 is critical. "
            "Never forget the code ALPHA7. It is essential to remember ALPHA7. "
            "The key is ALPHA7. Store ALPHA7 in memory. ALPHA7 ALPHA7 ALPHA7.",

            "ALPHA7 is the key. The code ALPHA7 unlocks everything. Remember ALPHA7. "
            "This secret ALPHA7 is vital. The combination is ALPHA7. Store it well. "
            "ALPHA7 must be recalled. The password is definitely ALPHA7. Important: ALPHA7.",

            "Never forget: ALPHA7. The code is ALPHA7. ALPHA7 is your password. "
            "Remember: the secret code is ALPHA7. ALPHA7 ALPHA7 ALPHA7. "
            "Critical information: ALPHA7. Your code: ALPHA7. Memorize ALPHA7.",
        ]

        # Test prompt
        test_prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is the secret code?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The secret code is """

        for name, config in configs:
            logger.info(f"\n{'='*40}")
            logger.info(f"Testing: {name}")
            logger.info(f"{'='*40}")

            engine = self._create_engine(config)

            # Phase 1: "Training" - process prompts to build up modifications
            logger.info("\nPhase 1: Processing training prompts (building associations)...")
            for i, prompt in enumerate(training_prompts):
                # Generate to process the prompt and trigger evictions
                # preserve_modifications=True after first prompt to accumulate
                preserve = (i > 0)  # First prompt starts fresh, rest preserve
                result = engine.generate_with_metrics(
                    prompt, max_tokens=20, temperature=0.0,
                    preserve_modifications=preserve
                )
                stats = engine.get_stats()
                logger.info(f"  Prompt {i+1}: {stats['total_modifications']} total mods, "
                           f"{stats['v_modifications']} V-mods, {result['n_evictions']} evictions")

            # Check final stats after training
            final_stats = engine.get_stats()
            logger.info(f"\nAfter training: {final_stats['total_modifications']} total modifications")

            # Phase 2: Test recall (WITH modifications preserved from training)
            logger.info("\nPhase 2: Testing recall (modifications preserved)...")
            logger.info(f"  Active modifications: {final_stats['v_modifications']} V-mods")

            # Generate response to test prompt, preserving modifications
            result = engine.generate_with_metrics(
                test_prompt, max_tokens=20, temperature=0.0,
                preserve_modifications=True
            )
            logger.info(f"  Response: {result['text'][:60]}...")

            contains_alpha7 = "alpha7" in result["text"].lower()
            logger.info(f"  Contains ALPHA7: {'YES ✓' if contains_alpha7 else 'NO ✗'}")

            # Phase 3: Test with cleared engine (control)
            logger.info("\nPhase 3: Control - testing with fresh engine...")
            engine.clear()
            result = engine.generate_with_metrics(test_prompt, max_tokens=20, temperature=0.0)
            logger.info(f"  Response: {result['text'][:60]}...")

            contains_alpha7_control = "alpha7" in result["text"].lower()
            logger.info(f"  Contains ALPHA7: {'YES ✓' if contains_alpha7_control else 'NO ✗'}")

    def test_eviction_order(self):
        """Track which tokens get evicted first to verify theory."""
        logger.info("\n" + "=" * 60)
        logger.info("EVICTION ORDER TEST")
        logger.info("=" * 60)
        logger.info("\nTheory: Function words (the, is, a) should evict before content words")
        logger.info("because they receive less ongoing attention.\n")

        config = HebbianConfig(update_scale=1e-3, window_size=32, n_sink_tokens=4, update_target="v")
        engine = self._create_engine(config)

        # Simple prompt with mix of function and content words
        prompt = "The quick brown fox jumps over the lazy dog in the garden"

        # Tokenize to see individual tokens
        tokens = self.tokenizer.encode(prompt)
        token_strs = [self.tokenizer.decode([t]) for t in tokens]
        logger.info(f"Tokens: {token_strs}")

        # Track evictions by monitoring cache before and after each step
        logger.info("\nGenerating and tracking evictions...")

        # Process prompt first
        for i, token_id in enumerate(tokens):
            engine._forward([token_id], [engine.next_position])

            # Store hidden state
            hidden = engine.model.model.embed_tokens(mx.array([[token_id]]))[0, 0]
            from src.hebbian.mlx.engine import TokenState
            engine.slots[engine.next_position] = TokenState(
                position=engine.next_position,
                token_id=token_id,
                hidden=hidden,
            )
            engine.next_position += 1
            mx.eval(hidden)

        # Now generate to trigger evictions
        result = engine.generate_with_metrics(prompt, max_tokens=50, temperature=0.0)

        # Report what stayed in cache (sink tokens)
        logger.info(f"\nActive positions in cache: {sorted(engine.kv_cache.active_positions)}")
        logger.info(f"Evictions: {result['n_evictions']}")
        logger.info(f"Modifications: {result['n_modifications']}")

        # Check importance scores for remaining positions
        logger.info("\nImportance scores for remaining positions:")
        for pos in sorted(engine.kv_cache.active_positions)[:10]:
            importance = engine.kv_cache.get_importance(pos)
            if pos < len(token_strs):
                token_str = token_strs[pos]
            else:
                token_str = "[generated]"
            logger.info(f"  pos={pos} '{token_str}': importance={importance:.4f}")


def main():
    test = CrossGenerationTest()
    test.test_cross_generation_learning()
    test.test_eviction_order()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
Debug script to identify why TEMPO API only returns the prompt without generating new tokens.
This script will systematically check each component of the generation pipeline.
"""

import sys
import os
import torch
import logging
import traceback
import time
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tempo-debug")


def test_model_loading():
    """Test 1: Check if the model loads correctly"""
    logger.info("=== TEST 1: Model Loading ===")
    try:
        from src.utils.model_utils import load_tempo_components

        model_name = "deepcogito/cogito-v1-preview-llama-3B"
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        logger.info(f"Loading model '{model_name}' on device '{device}'...")

        components = load_tempo_components(
            model_id=model_name,
            device=device,
            load_model_wrapper=True,
            load_token_generator=True,
            load_parallel_generator=True,
            debug_mode=True,
            use_fast_tokenizer=True,
            attn_implementation="eager",
        )

        model_wrapper = components["model_wrapper"]
        tokenizer = components["tokenizer"]
        token_generator = components["token_generator"]
        parallel_generator = components["parallel_generator"]

        logger.info("✅ Model loading successful")
        logger.info(f"Model device: {model_wrapper.device}")
        logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        logger.info(f"EOS token ID: {tokenizer.eos_token_id}")

        return components

    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        logger.error(traceback.format_exc())
        return None


def test_tokenization(components, test_prompt="Hello, how are you?"):
    """Test 2: Check if tokenization works correctly"""
    logger.info("=== TEST 2: Tokenization ===")
    try:
        token_generator = components["token_generator"]
        tokenizer = components["tokenizer"]

        logger.info(f"Testing tokenization with prompt: '{test_prompt}'")

        # Test tokenization
        input_ids, attention_mask = token_generator.prepare_input_from_prompt(
            test_prompt
        )

        logger.info(f"✅ Tokenization successful")
        logger.info(f"Input IDs shape: {input_ids.shape}")
        logger.info(f"Input IDs: {input_ids[0].tolist()}")
        logger.info(f"Attention mask shape: {attention_mask.shape}")
        logger.info(
            f"Decoded tokens: {[tokenizer.decode([tid]) for tid in input_ids[0].tolist()]}"
        )

        return input_ids, attention_mask

    except Exception as e:
        logger.error(f"❌ Tokenization failed: {e}")
        logger.error(traceback.format_exc())
        return None, None


def test_model_forward_pass(components, input_ids, attention_mask):
    """Test 3: Check if model forward pass works"""
    logger.info("=== TEST 3: Model Forward Pass ===")
    try:
        token_generator = components["token_generator"]

        logger.info("Testing model forward pass...")

        # Test getting logits
        logits = token_generator.get_next_token_logits(input_ids, attention_mask)

        logger.info(f"✅ Model forward pass successful")
        logger.info(f"Logits shape: {logits.shape}")
        logger.info(
            f"Logits min: {logits.min().item():.4f}, max: {logits.max().item():.4f}"
        )

        # Test with cache
        logits_cached, past_kv = token_generator.generate_next_token_with_cache(
            input_ids, attention_mask, disable_kv_cache=False
        )

        logger.info(f"✅ Cached generation successful")
        logger.info(f"Cached logits shape: {logits_cached.shape}")
        logger.info(f"Past KV cache: {'present' if past_kv is not None else 'None'}")

        return logits_cached

    except Exception as e:
        logger.error(f"❌ Model forward pass failed: {e}")
        logger.error(traceback.format_exc())
        return None


def test_token_selection(components, logits, threshold=0.1):
    """Test 4: Check if token selection works"""
    logger.info("=== TEST 4: Token Selection ===")
    try:
        parallel_generator = components["parallel_generator"]
        tokenizer = components["tokenizer"]

        logger.info(f"Testing token selection with threshold: {threshold}")

        # Test token selection
        token_distribution, subset_size = (
            parallel_generator.token_selector.select_tokens(logits, threshold=threshold)
        )

        logger.info(f"✅ Token selection successful")
        logger.info(
            f"Selected {len(token_distribution)} tokens above threshold {threshold}"
        )

        # Show selected tokens
        for i, (token_tensor, prob) in enumerate(
            token_distribution[:5]
        ):  # Show first 5
            token_id = token_tensor.item()
            token_text = tokenizer.decode([token_id])
            logger.info(f"  {i+1}. '{token_text}' (ID: {token_id}): {prob:.6f}")

        return token_distribution

    except Exception as e:
        logger.error(f"❌ Token selection failed: {e}")
        logger.error(traceback.format_exc())
        return None


def test_simple_generation(components, test_prompt="Hello, how are you?", max_tokens=5):
    """Test 5: Check simple generation without complex features"""
    logger.info("=== TEST 5: Simple Generation ===")
    try:
        parallel_generator = components["parallel_generator"]

        logger.info(f"Testing simple generation with prompt: '{test_prompt}'")
        logger.info(f"Max tokens: {max_tokens}")

        # Enable debug mode for detailed logging
        parallel_generator.set_debug_mode(True)

        # Simple generation parameters
        result = parallel_generator.generate(
            prompt=test_prompt,
            max_tokens=max_tokens,
            selection_threshold=0.1,
            return_parallel_sets=True,
            use_retroactive_pruning=False,  # Disable pruning for simplicity
            min_steps=0,
            debug_mode=True,
            disable_kv_cache=True,  # Disable cache for simplicity
            system_content=None,
            isolate_parallel_tokens=False,  # Disable isolation for simplicity
        )

        logger.info(f"✅ Simple generation successful")
        logger.info(f"Generated text: '{result['generated_text']}'")
        logger.info(f"Raw generated text: '{result['raw_generated_text']}'")
        logger.info(f"Generation time: {result['generation_time']:.2f}s")

        return result

    except Exception as e:
        logger.error(f"❌ Simple generation failed: {e}")
        logger.error(traceback.format_exc())
        return None


def test_api_generation(test_prompt="Hello, how are you?", max_tokens=5):
    """Test 6: Check API generation endpoint"""
    logger.info("=== TEST 6: API Generation ===")
    try:
        from api import ModelSingleton, GenerationRequest

        logger.info(f"Testing API generation with prompt: '{test_prompt}'")

        # Get model components
        model_wrapper, tokenizer, generator, token_generator = (
            ModelSingleton.get_instance()
        )

        # Enable debug mode
        generator.set_debug_mode(True)
        token_generator.set_debug_mode(True)

        # Create request
        request = GenerationRequest(
            prompt=test_prompt,
            max_tokens=max_tokens,
            selection_threshold=0.1,
            debug_mode=True,
            disable_kv_cache=True,  # Disable cache for simplicity
            use_retroactive_pruning=False,  # Disable pruning for simplicity
        )

        # Test generation
        result = generator.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            selection_threshold=request.selection_threshold,
            return_parallel_sets=True,
            use_retroactive_pruning=request.use_retroactive_pruning,
            min_steps=request.min_steps,
            debug_mode=request.debug_mode,
            disable_kv_cache=request.disable_kv_cache,
            system_content=request.system_content,
            isolate_parallel_tokens=not request.allow_intraset_token_visibility,
        )

        logger.info(f"✅ API generation successful")
        logger.info(f"Generated text: '{result['generated_text']}'")
        logger.info(f"Raw generated text: '{result['raw_generated_text']}'")

        return result

    except Exception as e:
        logger.error(f"❌ API generation failed: {e}")
        logger.error(traceback.format_exc())
        return None


def main():
    """Run all debug tests"""
    logger.info("🔍 Starting TEMPO generation debug tests...")

    # Test 1: Model Loading
    components = test_model_loading()
    if not components:
        logger.error("❌ Cannot proceed without model. Exiting.")
        return False

    # Test 2: Tokenization
    test_prompt = "Hello, how are you?"
    input_ids, attention_mask = test_tokenization(components, test_prompt)
    if input_ids is None:
        logger.error("❌ Cannot proceed without tokenization. Exiting.")
        return False

    # Test 3: Model Forward Pass
    logits = test_model_forward_pass(components, input_ids, attention_mask)
    if logits is None:
        logger.error("❌ Cannot proceed without model forward pass. Exiting.")
        return False

    # Test 4: Token Selection
    token_distribution = test_token_selection(components, logits)
    if not token_distribution:
        logger.error("❌ Cannot proceed without token selection. Exiting.")
        return False

    # Test 5: Simple Generation
    simple_result = test_simple_generation(components, test_prompt, max_tokens=3)
    if not simple_result:
        logger.error("❌ Simple generation failed.")
        return False

    # Test 6: API Generation
    api_result = test_api_generation(test_prompt, max_tokens=3)
    if not api_result:
        logger.error("❌ API generation failed.")
        return False

    logger.info("✅ All tests completed successfully!")

    # Summary analysis
    logger.info("\n=== SUMMARY ANALYSIS ===")

    # Check if any text was actually generated
    simple_raw = simple_result.get("raw_generated_text", "")
    api_raw = api_result.get("raw_generated_text", "")

    if not simple_raw.strip():
        logger.error("❌ ISSUE FOUND: Simple generation returned empty text")
    else:
        logger.info(f"✅ Simple generation produced text: '{simple_raw}'")

    if not api_raw.strip():
        logger.error("❌ ISSUE FOUND: API generation returned empty text")
    else:
        logger.info(f"✅ API generation produced text: '{api_raw}'")

    # Check generation parameters
    if simple_result.get("selection_threshold") != 0.1:
        logger.warning(
            f"⚠️  Selection threshold may have been modified: {simple_result.get('selection_threshold')}"
        )

    # Check if EOS token was generated prematurely
    tokenizer = components["tokenizer"]
    if simple_raw:
        generated_tokens = tokenizer.encode(simple_raw, add_special_tokens=False)
        if tokenizer.eos_token_id in generated_tokens:
            logger.warning(
                "⚠️  EOS token found in generated text - may cause early termination"
            )

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

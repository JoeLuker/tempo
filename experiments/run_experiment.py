#!/usr/bin/env python3
"""Baseline experiment runner (loads model fresh each time).

This is the baseline implementation for comparison testing.
It loads the model fresh for each experiment, which is slow
but guarantees clean state.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import time

from src.utils.model_utils import load_tempo_components
from src.domain.entities.parallel_generation import GenerationConfig
from src.application.services.generation_service import GenerationService


def run_single_experiment(
    prompt: str,
    seed: int,
    isolated: bool = True,
    max_tokens: int = 10,
    selection_threshold: float = 0.1,
    capture_attention: bool = True,
    output_dir: Optional[Path] = None,
    model_id: str = "deepcogito/cogito-v1-preview-llama-3B",
    device: str = "mps",
    debug_mode: bool = False
) -> Dict:
    """
    Run a single experiment with fresh model load (baseline).

    This is the SLOW but REFERENCE implementation.
    Each call loads the model fresh.

    Args:
        prompt: Input prompt
        seed: Random seed
        isolated: Use isolation mode
        max_tokens: Max tokens to generate
        selection_threshold: Parallel token threshold
        capture_attention: Capture attention matrices
        output_dir: Where to save (None = don't save)
        model_id: Model to use
        device: Device (mps, cuda, cpu)
        debug_mode: Enable debug logging

    Returns:
        Dict with experiment results
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load components (FRESH EACH TIME - this is the baseline)
    components = load_tempo_components(
        model_id=model_id,
        device=device,
        load_model_wrapper=True,
        debug_mode=debug_mode
    )

    model_wrapper = components["model_wrapper"]
    tokenizer = components["tokenizer"]
    rope_modifier = components.get("rope_modifier")
    attention_manager = components.get("attention_manager")

    # Create generation service
    generation_service = GenerationService(
        model_wrapper=model_wrapper,
        rope_modifier=rope_modifier,
        attention_manager=attention_manager,
        debug_mode=debug_mode
    )

    # Create config
    config = GenerationConfig(
        prompt=prompt,
        max_tokens=max_tokens,
        selection_threshold=selection_threshold,
        isolate_parallel_tokens=isolated,
        seed=seed,
        disable_kv_cache=False,
        debug_mode=debug_mode
    )

    # Run generation
    start_time = time.time()
    result = generation_service.generate(config)
    generation_time = time.time() - start_time

    # Extract data (same format as persistent runner)
    output = {
        "generated_token_ids": _extract_token_ids(result, tokenizer),
        "generated_text": result.generated_text,
        "raw_generated_text": result.raw_generated_text,
        "generation_time": generation_time,
        "prompt": prompt,
        "seed": seed,
        "mode": "isolated" if isolated else "visible",
        "selection_threshold": selection_threshold,
        "max_tokens": max_tokens
    }

    # Capture attention if requested
    if capture_attention and attention_manager:
        output["attention_data"] = _extract_attention_data(attention_manager)

    # Capture parallel sets
    output["parallel_sets"] = _extract_parallel_sets(result)

    # Save if requested
    if output_dir:
        _save_results(output, output_dir)

    # Cleanup
    del model_wrapper
    del generation_service
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output


def _extract_token_ids(result, tokenizer) -> list:
    """Extract token IDs."""
    if hasattr(result, 'logical_layout'):
        token_ids = []
        for pos in result.logical_layout:
            token_ids.extend(pos.token_ids)
        return token_ids
    else:
        tokens = tokenizer.encode(result.raw_generated_text)
        return tokens


def _extract_attention_data(attention_manager) -> Dict:
    """Extract attention matrices."""
    attention_data = {}

    if hasattr(attention_manager, 'get_captured_attention'):
        captured = attention_manager.get_captured_attention()
        for step, data in captured.items():
            if 'attention_weights' in data:
                attention_data[f"step_{step}_attention"] = data['attention_weights']
            if 'positions' in data:
                attention_data[f"step_{step}_positions"] = data['positions']
            if 'logical_positions' in data:
                attention_data[f"step_{step}_logical"] = data['logical_positions']

    return attention_data


def _extract_parallel_sets(result) -> Dict:
    """Extract parallel sets."""
    parallel_sets = {
        "parallel_sets": [],
        "summary": {
            "total_parallel_steps": 0,
            "max_parallel_width": 0
        }
    }

    if hasattr(result, 'logical_layout'):
        for i, pos in enumerate(result.logical_layout):
            if len(pos.token_ids) > 1:
                parallel_sets["parallel_sets"].append({
                    "step": i,
                    "count": len(pos.token_ids),
                    "tokens": pos.token_ids,
                    "positions": list(range(i, i + len(pos.token_ids)))
                })

        parallel_sets["summary"]["total_parallel_steps"] = len(parallel_sets["parallel_sets"])
        if parallel_sets["parallel_sets"]:
            parallel_sets["summary"]["max_parallel_width"] = max(
                ps["count"] for ps in parallel_sets["parallel_sets"]
            )

    return parallel_sets


def _save_results(output: Dict, output_dir: Path):
    """Save results."""
    import json
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "prompt": output["prompt"],
        "seed": output["seed"],
        "mode": output["mode"],
        "selection_threshold": output["selection_threshold"],
        "max_tokens": output["max_tokens"],
        "generation_time": output["generation_time"]
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    if "attention_data" in output and output["attention_data"]:
        np.savez(
            output_dir / "attention_weights.npz",
            **output["attention_data"]
        )

    if "parallel_sets" in output:
        with open(output_dir / "parallel_sets.json", 'w') as f:
            json.dump(output["parallel_sets"], f, indent=2)


if __name__ == "__main__":
    print("Running baseline experiment...")
    result = run_single_experiment(
        prompt="The cat sat on the",
        seed=42,
        isolated=True,
        max_tokens=10,
        capture_attention=False
    )
    print(f"Generated: {result['generated_text']}")
    print(f"Time: {result['generation_time']:.2f}s")

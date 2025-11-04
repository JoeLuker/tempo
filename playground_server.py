#!/usr/bin/env python3
"""
FastAPI server for TEMPO interactive playground.
Direct programmatic access to TEMPO components.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.experiments.experiment_runner import ExperimentRunner

app = FastAPI(title="TEMPO Playground API")

# Global model cache (lazy loaded on first request)
_model = None
_tokenizer = None
_runner = None

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerationRequest(BaseModel):
    prompt: str
    selection_threshold: float = 0.25
    max_tokens: int = 20
    isolate: bool = False
    seed: int = 42


class TokenStep(BaseModel):
    step_number: int
    type: str  # 'prompt', 'single', 'parallel'
    tokens: List[str]
    probabilities: Optional[List[float]] = None


class GenerationResponse(BaseModel):
    steps: List[TokenStep]
    raw_text: str
    generation_time: float
    threshold: float
    isolate: bool


def convert_results_to_steps(result_dict: dict, prompt: str, tokenizer) -> List[TokenStep]:
    """Convert TEMPO results into frontend-friendly steps with real probabilities."""
    import re

    generated_text = result_dict.get('generated_text', '')
    all_original_token_sets = result_dict.get('all_original_token_sets', {})

    # Remove ANSI codes
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_text = ansi_escape.sub('', generated_text)

    # Remove prompt
    if prompt in clean_text:
        clean_text = clean_text.split(prompt, 1)[1]

    steps = []

    # Add prompt
    steps.append(TokenStep(
        step_number=0,
        type='prompt',
        tokens=[prompt]
    ))

    # Parse parallel sets [token1/token2]
    parallel_pattern = r'\[([^\]]+)\]'
    current_pos = 0
    step_num = 1
    logical_step = 0  # Track logical step for matching with token sets

    for match in re.finditer(parallel_pattern, clean_text):
        # Text before parallel set
        before = clean_text[current_pos:match.start()].strip()
        if before:
            steps.append(TokenStep(
                step_number=step_num,
                type='single',
                tokens=[before]
            ))
            step_num += 1
            logical_step += 1

        # Parallel tokens - extract from text
        parallel_text = match.group(1)
        tokens_from_text = [t.strip() for t in parallel_text.split('/')]

        # Try to get real probabilities from token sets
        probabilities = []
        if logical_step in all_original_token_sets:
            token_set = all_original_token_sets[logical_step]
            # token_set is a list of (token_id, probability) tuples

            # Decode token IDs to get token strings
            token_id_to_prob = {tid: prob for tid, prob in token_set}

            # Match tokens from text to their probabilities
            for token_text in tokens_from_text:
                # Find the token ID that decodes to this text
                found_prob = None
                for token_id, prob in token_set:
                    decoded = tokenizer.decode([token_id])
                    if decoded.strip() == token_text.strip():
                        found_prob = prob
                        break

                if found_prob is not None:
                    probabilities.append(found_prob)
                else:
                    # Fallback if we can't match
                    probabilities.append(0.1)

        # If we couldn't get real probabilities, use approximation
        if not probabilities or len(probabilities) != len(tokens_from_text):
            num_tokens = len(tokens_from_text)
            probabilities = [0.8 / (i + 1) for i in range(num_tokens)] if num_tokens > 0 else []
            prob_sum = sum(probabilities)
            if prob_sum > 0:
                probabilities = [p / prob_sum for p in probabilities]

        steps.append(TokenStep(
            step_number=step_num,
            type='parallel',
            tokens=tokens_from_text,
            probabilities=probabilities
        ))
        step_num += 1
        logical_step += 1

        current_pos = match.end()

    # Remaining text
    remaining = clean_text[current_pos:].strip()
    if remaining:
        steps.append(TokenStep(
            step_number=step_num,
            type='single',
            tokens=[remaining]
        ))

    return steps


@app.post("/api/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Run TEMPO generation directly."""

    try:
        # Lazy load model on first request
        global _model, _tokenizer, _runner
        if _runner is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
            model_name = 'deepcogito/cogito-v1-preview-llama-3B'

            print(f"Loading model {model_name} on device {device}...")
            _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            _model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device != 'cpu' else torch.float32
            ).to(device)

            _runner = ExperimentRunner(model=_model, tokenizer=_tokenizer, device=device)
            print(f"Model loaded successfully on {device}")

        # Build args dict
        args = {
            'prompt': request.prompt,
            'max_tokens': request.max_tokens,
            'selection_threshold': request.selection_threshold,
            'isolate': request.isolate,
            'seed': request.seed,
            'output_dir': './playground_temp',
            'debug_mode': False,
            'use_retroactive_removal': False,
            'disable_kv_cache': False,
            'enable_thinking': False,
            'use_mcts': False,
        }

        # Run experiment
        result_dict = _runner.run_experiment(args)

        # Convert to frontend format
        steps = convert_results_to_steps(result_dict, request.prompt, _tokenizer)

        return GenerationResponse(
            steps=steps,
            raw_text=result_dict.get('raw_generated_text', ''),
            generation_time=result_dict.get('generation_time', 0.0),
            threshold=request.selection_threshold,
            isolate=request.isolate
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/")
async def root():
    """Serve the playground UI."""
    return FileResponse('playground.html')


if __name__ == "__main__":
    import uvicorn
    print("="*80)
    print("🚀 Starting TEMPO Playground Server")
    print("="*80)
    print("\n📱 Open your browser to: http://localhost:8765")
    print("\n✨ Interactive playground ready!")
    print("="*80)
    uvicorn.run(app, host="0.0.0.0", port=8765)

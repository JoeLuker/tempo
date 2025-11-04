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
    # Basic parameters
    prompt: str
    max_tokens: int = 20
    selection_threshold: float = 0.25
    min_steps: int = 0
    seed: int = 42

    # Parallel token control
    isolate: bool = False  # Isolate parallel tokens from seeing each other

    # Retroactive pruning
    use_retroactive_removal: bool = False
    attention_threshold: float = 0.01

    # Dynamic threshold
    dynamic_threshold: bool = False
    final_threshold: float = 1.0
    bezier_p1: float = 0.2
    bezier_p2: float = 0.8
    use_relu: bool = False
    relu_activation_point: float = 0.5

    # MCTS parameters
    use_mcts: bool = False
    mcts_simulations: int = 10
    mcts_c_puct: float = 1.0
    mcts_depth: int = 5

    # Advanced options
    disable_kv_cache: bool = False
    enable_thinking: bool = False  # Cogito model's deep thinking mode
    show_token_ids: bool = False


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
    # Note: all_original_token_sets is keyed by logical step (includes ALL steps, not just parallel)
    # Steps 0-N are: prompt processing, token 1, token 2, ... parallel set, etc.
    # We need to find which logical step each parallel set corresponds to
    parallel_pattern = r'\[([^\]]+)\]'
    current_pos = 0
    step_num = 1

    # Find all steps that have >1 token (these are parallel sets)
    parallel_logical_steps = [step for step, tokens in all_original_token_sets.items() if len(tokens) > 1]
    parallel_set_index = 0

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

        # Parallel tokens - extract from text
        parallel_text = match.group(1)
        tokens_from_text = [t.strip() for t in parallel_text.split('/')]

        # Try to get real probabilities from token sets
        probabilities = []
        if parallel_set_index < len(parallel_logical_steps):
            logical_step = parallel_logical_steps[parallel_set_index]
            token_set = all_original_token_sets[logical_step]
            # token_set is a list of (token_id, probability) tuples

            # If we have the same number of tokens, just use the probabilities in order
            if len(token_set) == len(tokens_from_text):
                probabilities = [prob for _, prob in token_set]
            else:
                # Try to match by decoding
                for token_text in tokens_from_text:
                    found_prob = None
                    for token_id, prob in token_set:
                        decoded = tokenizer.decode([token_id]).strip()
                        if decoded == token_text:
                            found_prob = prob
                            break

                    if found_prob is not None:
                        probabilities.append(found_prob)
                    else:
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
        parallel_set_index += 1

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
            # Basic parameters
            'prompt': request.prompt,
            'max_tokens': request.max_tokens,
            'selection_threshold': request.selection_threshold,
            'min_steps': request.min_steps,
            'seed': request.seed,

            # Parallel token control
            'isolate': request.isolate,

            # Retroactive pruning
            'use_retroactive_removal': request.use_retroactive_removal,
            'attention_threshold': request.attention_threshold,

            # Dynamic threshold
            'dynamic_threshold': request.dynamic_threshold,
            'final_threshold': request.final_threshold,
            'bezier_p1': request.bezier_p1,
            'bezier_p2': request.bezier_p2,
            'use_relu': request.use_relu,
            'relu_activation_point': request.relu_activation_point,

            # MCTS parameters
            'use_mcts': request.use_mcts,
            'mcts_simulations': request.mcts_simulations,
            'mcts_c_puct': request.mcts_c_puct,
            'mcts_depth': request.mcts_depth,

            # Advanced options
            'disable_kv_cache': request.disable_kv_cache,
            'enable_thinking': request.enable_thinking,
            'show_token_ids': request.show_token_ids,

            # Internal
            'output_dir': './playground_temp',
            'debug_mode': False,
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

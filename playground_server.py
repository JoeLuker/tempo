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


def convert_results_to_steps(result_dict: dict, prompt: str) -> List[TokenStep]:
    """Convert TEMPO results into frontend-friendly steps."""
    import re

    generated_text = result_dict.get('generated_text', '')

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

        # Parallel tokens
        parallel_text = match.group(1)
        tokens = [t.strip() for t in parallel_text.split('/')]
        steps.append(TokenStep(
            step_number=step_num,
            type='parallel',
            tokens=tokens
        ))
        step_num += 1

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
        # Create experiment runner
        runner = ExperimentRunner()

        # Build args dict
        args = {
            'prompt': request.prompt,
            'max_tokens': request.max_tokens,
            'selection_threshold': request.selection_threshold,
            'isolate': request.isolate,
            'seed': request.seed,
            'output_dir': './playground_temp',
            'model': 'deepcogito/cogito-v1-preview-llama-3B',
            'use_custom_rope': True,
            'debug_mode': False,
            'use_retroactive_removal': False,
            'disable_kv_cache': False,
            'enable_thinking': False,
            'use_mcts': False,
        }

        # Run experiment
        result_dict = runner.run_experiment(args)

        # Convert to frontend format
        steps = convert_results_to_steps(result_dict, request.prompt)

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

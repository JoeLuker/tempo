#!/usr/bin/env python3
"""
FastAPI server for TEMPO interactive playground.
Direct programmatic access to TEMPO components with clean structured output.
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


class TokenNode(BaseModel):
    """A single token node with all its metadata."""
    id: str  # Unique ID in format "step_index" (e.g., "0_0", "1_0", "2_1")
    token_id: int  # Actual tokenizer token ID
    text: str  # Decoded text
    probability: float  # Softmax probability
    logical_step: int  # Which generation step this belongs to
    is_parallel: bool  # Whether this is part of a parallel set
    parent_ids: List[str]  # IDs of parent nodes (usually one, multiple for convergence)


class GenerationResponse(BaseModel):
    nodes: List[TokenNode]  # All tokens as flat list of nodes
    edges: List[tuple[str, str]]  # Parent-child relationships as (parent_id, child_id)
    prompt: str
    raw_text: str
    generation_time: float
    threshold: float


def convert_to_structured_graph(result_dict: dict, prompt: str, tokenizer) -> dict:
    """
    Convert TEMPO results directly into a structured graph.
    No parsing, no regex, no text matching - use the actual token data.
    """
    all_original_token_sets = result_dict.get('all_original_token_sets', {})

    if not all_original_token_sets:
        return {"nodes": [], "edges": []}

    nodes = []
    edges = []

    # Track node IDs for edge creation
    # node_map: logical_step -> [list of node IDs at that step]
    node_map: Dict[int, List[str]] = {}

    # Process each logical step
    for logical_step in sorted(all_original_token_sets.keys()):
        token_set = all_original_token_sets[logical_step]
        is_parallel = len(token_set) > 1

        step_node_ids = []

        # Create a node for each token in this step
        for idx, (token_id, probability) in enumerate(token_set):
            node_id = f"{logical_step}_{idx}"
            text = tokenizer.decode([token_id])

            # Determine parent IDs
            parent_ids = []
            if logical_step > 0:
                prev_step = logical_step - 1
                if prev_step in node_map:
                    # For parallel tokens, connect to primary parent
                    # For convergence (multiple prev, single current), connect all
                    if is_parallel or len(token_set) == 1:
                        # Connect to all parents from previous step
                        parent_ids = node_map[prev_step]
                    else:
                        # Single token after single token - direct lineage
                        parent_ids = [node_map[prev_step][0]]

            nodes.append(TokenNode(
                id=node_id,
                token_id=token_id,
                text=text,
                probability=probability,
                logical_step=logical_step,
                is_parallel=is_parallel,
                parent_ids=parent_ids
            ))

            step_node_ids.append(node_id)

            # Create edges
            for parent_id in parent_ids:
                edges.append((parent_id, node_id))

        node_map[logical_step] = step_node_ids

    return {"nodes": nodes, "edges": edges}


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

        # Convert to structured graph
        graph_data = convert_to_structured_graph(result_dict, request.prompt, _tokenizer)

        return GenerationResponse(
            nodes=graph_data["nodes"],
            edges=graph_data["edges"],
            prompt=request.prompt,
            raw_text=result_dict.get('raw_generated_text', ''),
            generation_time=result_dict.get('generation_time', 0.0),
            threshold=request.selection_threshold
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

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Tuple
import torch
import time
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from src.modeling.model_wrapper import TEMPOModelWrapper
from src.generation.parallel_generator import ParallelGenerator
from src.pruning.diversity_pruner import DiversityPruner
from src.pruning.retroactive_pruner import RetroactivePruner
from src.visualization.token_visualizer import TokenVisualizer
from src.visualization.position_visualizer import PositionVisualizer
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tempo-api")


# Singleton for model and components to keep them in memory
class ModelSingleton:
    model = None
    tokenizer = None
    generator = None
    initialized = False

    @classmethod
    def get_instance(cls, device="mps"):
        """Get or initialize model instance, maintaining the singleton invariant"""
        if not cls.initialized:
            logger.info("Loading model for the first time...")
            cls._initialize_model(device)
            cls.initialized = True

        # INVARIANT: After initialization, all components must exist
        assert cls.model is not None, "Model initialization failed"
        assert cls.tokenizer is not None, "Tokenizer initialization failed"
        assert cls.generator is not None, "Generator initialization failed"

        return cls.model, cls.tokenizer, cls.generator

    @classmethod
    def _initialize_model(cls, device):
        """Initialize model components with proper error handling"""
        try:
            # Determine the device and precision
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
            dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16 if device != "cpu" else torch.float32
            )

            logger.info(f"Using device: {device} with {dtype}")

            # Load model
            # model_name = "deepcogito/cogito-v1-preview-qwen-14B"
            model_name = "deepcogito/cogito-v1-preview-llama-3B"

            logger.info(f"Loading model {model_name} on {device}...")

            # Load tokenizer only once with caching
            cls.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if cls.tokenizer.pad_token is None:
                cls.tokenizer.pad_token = cls.tokenizer.eos_token

            # INVARIANT: Tokenizer must be properly initialized
            assert hasattr(cls.tokenizer, "encode"), "Tokenizer missing encode method"
            assert hasattr(cls.tokenizer, "decode"), "Tokenizer missing decode method"

            # First load the config to modify it
            config = AutoConfig.from_pretrained(model_name)

            # Disable sliding window attention for Qwen models to fix compatibility issues
            if hasattr(config, "sliding_window") and config.sliding_window is not None:
                logger.info(
                    f"Disabling sliding window attention (was set to {config.sliding_window})"
                )
                config.sliding_window = None

            # Load model with optimized settings and modified config
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else device,
                low_cpu_mem_usage=True,
                attn_implementation="eager",  # Use eager attention for better compatibility
            )

            # Optimize model for inference
            if hasattr(model, "eval"):
                model.eval()

            # Wrap model with TEMPO wrapper
            wrapped_model = TEMPOModelWrapper(model)
            cls.model = wrapped_model

            # INVARIANT: Model must be properly initialized
            assert hasattr(cls.model, "forward"), "Model missing forward method"
            assert hasattr(cls.model, "config"), "Model missing config attribute"

            # Create diversity pruner
            diversity_pruner = DiversityPruner(
                model=wrapped_model,
                tokenizer=cls.tokenizer,
                diversity_clusters=3,
                device=device,
                debug_mode=False,
            )

            # Create retroactive pruner
            retroactive_pruner = RetroactivePruner(
                model=wrapped_model,
                tokenizer=cls.tokenizer,
                attention_threshold=0.01,
                device=device,
                debug_mode=False,
            )

            # Create ParallelGenerator
            cls.generator = ParallelGenerator(
                model=wrapped_model,
                tokenizer=cls.tokenizer,
                pruner=diversity_pruner,  # Use diversity pruner as the default
                device=device,
                has_custom_attention=True,
            )

            # Make retroactive pruner available to the generator
            cls.retroactive_pruner = retroactive_pruner

            # INVARIANT: Generator must be properly initialized
            assert hasattr(
                cls.generator, "generate"
            ), "Generator missing generate method"

            logger.info("Model, tokenizer, and generator initialized successfully")

        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")


# Create FastAPI app
app = FastAPI(
    title="TEMPO API",
    description="API for TEMPO text generation with invariant guarantees",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request model with validators
class GenerationRequest(BaseModel):
    # Core parameters
    prompt: str = Field(description="Text prompt to start generation")
    max_tokens: int = Field(
        default=50, ge=1, le=200, description="Maximum number of tokens to generate"
    )
    threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Probability threshold for token selection - lower values allow more diverse completion paths",
    )

    # Advanced generation settings
    min_steps: int = Field(
        default=0,
        ge=0,
        description="Minimum steps to generate before considering EOS tokens",
    )
    use_custom_rope: bool = Field(
        default=True,
        description="Use custom RoPE modifications for improved parallel token positioning",
    )
    disable_kv_cache: bool = Field(
        default=False,
        description="Disable KV caching for more consistent attention patterns (slower but more accurate)",
    )
    show_token_ids: bool = Field(
        default=False, description="Include token IDs in the formatted output"
    )
    system_content: Optional[str] = Field(
        default=None,
        description="Optional system message content for chat models to adjust generation behavior",
    )
    enable_thinking: bool = Field(
        default=False,
        description="Enable Cogito's deep thinking mode for more thoughtful responses",
    )

    # MCTS parameters
    use_mcts: bool = Field(
        default=False, description="Use Monte Carlo Tree Search for text generation"
    )
    mcts_simulations: int = Field(
        default=10, ge=1, le=50, description="Number of MCTS simulations per step"
    )
    mcts_c_puct: float = Field(
        default=1.0, ge=0.1, le=5.0, description="Exploration constant for MCTS"
    )
    mcts_depth: int = Field(
        default=5, ge=1, le=10, description="Maximum depth for MCTS simulations"
    )

    # Pruning options
    use_pruning: bool = Field(
        default=True,
        description="Use pruning to reduce token sets for more coherent generation",
    )
    use_diversity_pruning: bool = Field(
        default=True,
        description="Use diversity pruning to reduce token sets for more diverse generation",
    )
    use_retroactive_pruning: bool = Field(
        default=True,
        description="Use retroactive pruning to reduce token sets for more coherent generation",
    )
    coherence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Threshold for coherence pruning - higher values require greater coherence",
    )
    diversity_clusters: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of clusters for diversity pruning - more clusters = more diverse completions",
    )
    diversity_steps: int = Field(
        default=5,
        ge=0,
        description="Number of steps to use diversity pruning before switching to coherence (for hybrid strategy)",
    )
    bezier_points: List[float] = Field(
        default=[0.2, 0.8],
        description="Control points for dynamic threshold Bezier curve (when dynamic thresholding is used)",
    )
    dynamic_threshold: bool = Field(
        default=True,
        description="Use dynamic thresholding that starts with diverse completions and gradually increases coherence",
    )
    attention_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Attention threshold for retroactive pruning (lower means more tokens kept)",
    )

    # Validator for prompt
    @field_validator("prompt")
    def prompt_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v.strip()

    # Validator for bezier points
    @field_validator("bezier_points")
    def validate_bezier_points(cls, v):
        if len(v) != 2:
            raise ValueError("Bezier points must contain exactly 2 values")
        if not all(0 <= p <= 1 for p in v):
            raise ValueError("Bezier points must be between 0 and 1")
        return v


# Response models with proper typing
class TokenInfo(BaseModel):
    token_text: str
    token_id: int
    probability: float


class StepInfo(BaseModel):
    position: int
    parallel_tokens: List[TokenInfo]
    pruned_tokens: List[TokenInfo]


class PruningInfo(BaseModel):
    strategy: str
    coherence_threshold: float
    diversity_clusters: int
    use_dynamic_threshold: bool
    diversity_steps: int
    pruning_time: float


class TimingInfo(BaseModel):
    generation_time: float
    pruning_time: float
    elapsed_time: float


class ModelInfo(BaseModel):
    model_name: str
    is_qwen_model: bool
    use_custom_rope: bool


class GenerationResponse(BaseModel):
    # Core output
    generated_text: str
    raw_generated_text: str

    # Token-level data
    steps: List[StepInfo]
    position_to_tokens: Dict[str, List[str]] = {}
    original_parallel_positions: List[int] = []

    # Performance and timing
    timing: TimingInfo

    # Pruning information
    pruning: Optional[PruningInfo] = None

    # Model information
    model_info: ModelInfo

    # Generation settings
    threshold: float
    max_tokens: int
    min_steps: int
    prompt: str

    # Advanced fields
    had_repetition_loop: bool = False
    system_content: Optional[str] = None

    # Token sets data for visualization (raw data from generator)
    token_sets: List[Tuple[int, List[Tuple[int, float]], List[Tuple[int, float]]]] = []

    # Raw token information for visualization
    tokens_by_position: Dict[str, Any] = {}
    final_pruned_sets: Dict[str, Any] = {}


# Dependency for getting model components
async def get_model_components():
    """Dependency to get model components with proper error handling"""
    try:
        model, tokenizer, generator = ModelSingleton.get_instance()
        return model, tokenizer, generator
    except Exception as e:
        logger.error(f"Failed to get model components: {str(e)}")
        raise HTTPException(status_code=503, detail="Model initialization failed")


# Create visualizer singletons
token_visualizer = TokenVisualizer()
position_visualizer = PositionVisualizer()

# Cache to store recent generation results for visualization
generation_cache = {}


@app.get("/")
async def root():
    return {"message": "TEMPO API is running", "status": "healthy"}


@app.get("/health")
async def health_check():
    """Health check endpoint to verify API and model status"""
    try:
        # Quick check if model is initialized
        if not ModelSingleton.initialized:
            return {"status": "initializing", "message": "Model is not yet initialized"}

        # Verify components exist
        model, tokenizer, generator = ModelSingleton.get_instance()

        return {
            "status": "healthy",
            "model_loaded": True,
            "model_name": "deepcogito/cogito-v1-preview-llama-3B",
            "device": generator.device if hasattr(generator, "device") else "unknown",
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest, components: Tuple = Depends(get_model_components)
):
    """
    Generate text using TEMPO parallel generation.
    """
    # Extract components
    model, tokenizer, generator = components
    
    # Validate that components were properly initialized
    assert model is not None, "Model not initialized"
    assert tokenizer is not None, "Tokenizer not initialized"
    assert generator is not None, "Generator not initialized"

    # Extract request parameters
    prompt = request.prompt
    max_tokens = request.max_tokens
    threshold = request.threshold
    
    # Validate core parameters
    assert prompt, "Prompt cannot be empty"
    assert max_tokens > 0, "max_tokens must be positive"
    assert 0.0 <= threshold <= 1.0, "threshold must be between 0.0 and 1.0"

    try:
        # Log the request
        logger.info(f"Received generation request: prompt={prompt[:50]}..., max_tokens={max_tokens}")

        # Prepare timer
        start_time = time.time()

        # Determine which pruner to use based on request
        if request.use_retroactive_pruning:
            # Use retroactive pruning
            assert hasattr(ModelSingleton, "retroactive_pruner"), "Retroactive pruner not initialized"
            retroactive_pruner = ModelSingleton.retroactive_pruner
            
            # Configure retroactive pruner
            if hasattr(retroactive_pruner, "set_attention_threshold"):
                retroactive_pruner.set_attention_threshold(request.attention_threshold)
        else:
            retroactive_pruner = None

        # Check if MCTS should be used
        if request.use_mcts:
            # MCTS not implemented in API yet
            logger.warning("MCTS requested but not yet supported in API, falling back to standard generation")
        
        # Verify generator has the correct methods
        assert hasattr(generator, "generate"), "Generator missing generate method"

        # Generate text with error handling
        generation_result = generator.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            threshold=threshold,
            return_parallel_sets=True,
            use_pruning=request.use_pruning,
            min_steps=request.min_steps,
            show_token_ids=request.show_token_ids,
            use_custom_rope=request.use_custom_rope,
            disable_kv_cache=request.disable_kv_cache,
            system_content=request.system_content,
            retroactive_pruner=retroactive_pruner,
        )
        
        # Verify generation results
        assert "generated_text" in generation_result, "Generation result missing 'generated_text'"
        assert "raw_generated_text" in generation_result, "Generation result missing 'raw_generated_text'"

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"Generation completed in {elapsed_time:.2f}s")

        # Create response object
        response = GenerationResponse(
            generated_text=generation_result["generated_text"],
            raw_generated_text=generation_result["raw_generated_text"],
            steps=[],  # Will be populated below
            timing=TimingInfo(
                generation_time=generation_result.get("generation_time", elapsed_time),
                pruning_time=generation_result.get("pruning_time", 0.0),
                elapsed_time=elapsed_time,
            ),
            model_info=ModelInfo(
                model_name="deepcogito/cogito-v1-preview-llama-3B",
                is_qwen_model=generation_result.get("is_qwen_model", False),
                use_custom_rope=request.use_custom_rope,
            ),
            threshold=threshold,
            max_tokens=max_tokens,
            min_steps=request.min_steps,
            prompt=prompt,
            had_repetition_loop=generation_result.get("had_repetition_loop", False),
            system_content=request.system_content,
            token_sets=[],
        )

        # Process token sets if available
        if "token_sets" in generation_result:
            # Validate token sets format
            token_sets = generation_result["token_sets"]
            assert isinstance(token_sets, list), "token_sets must be a list"
            
            # Convert token sets to required format
            response.token_sets = [
                (position, [(int(tid), float(prob)) for tid, prob in token_ids_probs[0]], 
                 [(int(tid), float(prob)) for tid, prob in token_ids_probs[1]])
                for position, token_ids_probs, _ in token_sets
            ]

            # Build steps for the response
            steps = []
            for position, (orig_tokens, orig_probs), (pruned_tokens, pruned_probs) in token_sets:
                # Create token info objects
                parallel_tokens = [
                    TokenInfo(
                        token_text=tokenizer.decode([int(tid)]),
                        token_id=int(tid),
                        probability=float(prob),
                    )
                    for tid, prob in zip(orig_tokens, orig_probs)
                ]
                
                pruned_tokens_info = [
                    TokenInfo(
                        token_text=tokenizer.decode([int(tid)]),
                        token_id=int(tid),
                        probability=float(prob),
                    )
                    for tid, prob in zip(pruned_tokens, pruned_probs)
                ]
                
                # Create step info
                step = StepInfo(
                    position=position,
                    parallel_tokens=parallel_tokens,
                    pruned_tokens=pruned_tokens_info,
                )
                steps.append(step)
            
            response.steps = steps

        # Add pruning information if available
        if request.use_pruning:
            strategy_name = "coherence"
            if request.use_diversity_pruning:
                strategy_name = "diversity" 
            if request.use_diversity_pruning and request.use_retroactive_pruning:
                strategy_name = "hybrid"
                
            response.pruning = PruningInfo(
                strategy=strategy_name,
                coherence_threshold=request.coherence_threshold,
                diversity_clusters=request.diversity_clusters,
                use_dynamic_threshold=request.dynamic_threshold,
                diversity_steps=request.diversity_steps,
                pruning_time=generation_result.get("pruning_time", 0.0),
            )

        # Add position to tokens mapping if available
        if "position_to_tokens" in generation_result:
            response.position_to_tokens = generation_result["position_to_tokens"]

        # Add final pruned sets if available
        if "final_pruned_sets" in generation_result:
            response.final_pruned_sets = generation_result["final_pruned_sets"]

        # Add original parallel positions if available
        if "original_parallel_positions" in generation_result:
            response.original_parallel_positions = list(generation_result["original_parallel_positions"])

        # Validate the response is complete
        assert response.generated_text, "Generated text is missing in response"
        assert response.raw_generated_text, "Raw generated text is missing in response"
        assert response.timing, "Timing information is missing in response"

        return response

    except Exception as e:
        # Log the error with traceback
        logger.error(f"Error during generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# Run with: uvicorn api:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

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
from src.visualization.token_visualizer import TokenVisualizer
from src.visualization.position_visualizer import PositionVisualizer
import traceback
import uuid

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

            # Create ParallelGenerator without a default pruner
            cls.generator = ParallelGenerator(
                model=wrapped_model,
                tokenizer=cls.tokenizer,
                pruner=None,  # No default pruner
                device=device,
                has_custom_attention=True,
            )

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
    pruning_strategy: str = Field(
        default="coherence",
        description="Pruning strategy: coherence, diversity, or hybrid",
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
    final_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Final threshold value for dynamic thresholding (default: 1.0)",
    )
    use_relu: bool = Field(
        default=False,
        description="Use ReLU transition instead of Bezier curve for dynamic threshold",
    )
    relu_activation_point: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Point at which ReLU transition begins (0-1)",
    )
    attention_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Attention threshold for retroactive pruning (lower means more tokens kept)",
    )

    # Parallel tokens options
    allow_intraset_token_visibility: bool = Field(
        default=False,
        description="Allow tokens within the same parallel set to see each other during generation",
    )
    no_preserve_isolated_tokens: bool = Field(
        default=False,
        description="Allow pruning to evaluate isolated tokens even when tokens are isolated",
    )

    # Advanced retroactive pruning options
    no_relative_attention: bool = Field(
        default=False,
        description="Disable relative attention thresholds",
    )
    relative_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for relative attention-based pruning (0-1)",
    )
    no_multi_scale_attention: bool = Field(
        default=False,
        description="Disable multi-scale attention integration",
    )
    num_layers_to_use: Optional[int] = Field(
        default=None,
        description="Number of last layers to use for attention (None means use all layers)",
    )
    no_lci_dynamic_threshold: bool = Field(
        default=False,
        description="Disable LCI-based dynamic thresholding",
    )
    no_sigmoid_threshold: bool = Field(
        default=False,
        description="Disable sigmoid-based decision boundary",
    )
    sigmoid_steepness: float = Field(
        default=10.0,
        ge=1.0,
        description="Controls how sharp the sigmoid transition is",
    )
    complete_pruning_mode: str = Field(
        default="keep_token",
        description="How to handle pruned positions: 'keep_token' (keep best token), 'keep_unattended' (mark as unattended), 'remove_position' (remove position)",
    )

    # Advanced caching options
    disable_kv_cache_consistency: bool = Field(
        default=False,
        description="Disable KV cache consistency checks for RoPE modification",
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

    # Validator for pruning strategy
    @field_validator("pruning_strategy")
    def validate_pruning_strategy(cls, v):
        valid_strategies = ["coherence", "diversity", "hybrid"]
        if v not in valid_strategies:
            raise ValueError(
                f"Pruning strategy must be one of: {', '.join(valid_strategies)}"
            )
        return v

    # Validator for complete pruning mode
    @field_validator("complete_pruning_mode")
    def validate_complete_pruning_mode(cls, v):
        valid_modes = ["keep_token", "keep_unattended", "remove_position"]
        if v not in valid_modes:
            raise ValueError(
                f"Complete pruning mode must be one of: {', '.join(valid_modes)}"
            )
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
    final_threshold: float = 1.0
    use_relu: bool = False
    relu_activation_point: float = 0.5
    bezier_points: List[float] = Field(default=[0.2, 0.8])
    pruning_time: float


class RetroactivePruningInfo(BaseModel):
    attention_threshold: float
    use_relative_attention: bool = True
    relative_threshold: float = 0.5
    use_multi_scale_attention: bool = True
    num_layers_to_use: Optional[int] = None
    use_lci_dynamic_threshold: bool = True
    use_sigmoid_threshold: bool = True
    sigmoid_steepness: float = 10.0
    complete_pruning_mode: str = "keep_token"


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
    retroactive_pruning: Optional[RetroactivePruningInfo] = None

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
        logger.info(
            f"Received generation request: prompt={prompt[:50]}..., max_tokens={max_tokens}"
        )

        # Prepare timer
        start_time = time.time()

        # Create per-request pruner instances
        pruner = None
        retroactive_pruner = None

        # Configure pruner if needed
        if request.use_pruning:
            # Import the pruner classes
            from src.pruning.pruner import Pruner

            # Create a new pruner with the requested strategy for this request
            pruner = Pruner(
                model=model,
                tokenizer=tokenizer,
                strategy=request.pruning_strategy,
                coherence_threshold=request.coherence_threshold,
                diversity_clusters=request.diversity_clusters,
                diversity_steps=request.diversity_steps,
                device=generator.device,
                use_dynamic_threshold=request.dynamic_threshold,
                max_steps=max_tokens,
                bezier_points=request.bezier_points,
                final_threshold=request.final_threshold,
                use_relu=request.use_relu,
                relu_activation_point=request.relu_activation_point,
                debug_mode=False,  # Set to True for debugging
            )

            logger.info(
                f"Created per-request pruner with strategy: {request.pruning_strategy}"
            )

            # Create per-request retroactive pruner if needed
            if request.use_retroactive_pruning:
                from src.pruning.retroactive_pruner import RetroactivePruner

                # Create a new retroactive pruner for this request
                retroactive_pruner = RetroactivePruner(
                    model=model,
                    tokenizer=tokenizer,
                    attention_threshold=request.attention_threshold,
                    device=generator.device,
                    debug_mode=False,
                    dynamic_threshold_manager=(
                        pruner.threshold_manager
                        if hasattr(pruner, "threshold_manager")
                        else None
                    ),
                )

                # Configure advanced retroactive pruning parameters
                retroactive_pruner.use_relative_attention = (
                    not request.no_relative_attention
                )
                retroactive_pruner.relative_threshold = request.relative_threshold
                retroactive_pruner.use_multi_scale_attention = (
                    not request.no_multi_scale_attention
                )
                retroactive_pruner.num_layers_to_use = request.num_layers_to_use
                retroactive_pruner.use_lci_dynamic_threshold = (
                    not request.no_lci_dynamic_threshold
                )
                retroactive_pruner.use_sigmoid_threshold = (
                    not request.no_sigmoid_threshold
                )
                retroactive_pruner.sigmoid_steepness = request.sigmoid_steepness
                retroactive_pruner.complete_pruning_mode = request.complete_pruning_mode

                logger.info(
                    f"Created per-request retroactive pruner with attention threshold: {request.attention_threshold}"
                )

        # Configure RoPE modifier if needed
        if hasattr(generator, "rope_modifier") and generator.rope_modifier is not None:
            if request.disable_kv_cache_consistency:
                if hasattr(generator.rope_modifier, "enable_kv_cache_consistency"):
                    generator.rope_modifier.enable_kv_cache_consistency(False)
                    logger.info("Disabled KV cache consistency for RoPE modifier")

        # Check if MCTS should be used
        if request.use_mcts:
            # MCTS not implemented in API yet
            logger.warning(
                "MCTS requested but not yet supported in API, falling back to standard generation"
            )

        # Prepare system content for thinking mode
        system_content = request.system_content
        if request.enable_thinking and not system_content:
            system_content = "Enable deep thinking subroutine."
            logger.info("Enabled thinking mode with system content")

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
            disable_kv_cache=request.disable_kv_cache,
            system_content=system_content,
            pruner=pruner,  # Pass the per-request pruner
            retroactive_pruner=retroactive_pruner,  # Pass the per-request retroactive pruner
            isolate_parallel_tokens=not request.allow_intraset_token_visibility,
            preserve_all_isolated_tokens=(
                not request.no_preserve_isolated_tokens
                if not request.allow_intraset_token_visibility
                else None
            ),
        )

        # Verify generation results
        assert (
            "generated_text" in generation_result
        ), "Generation result missing 'generated_text'"
        assert (
            "raw_generated_text" in generation_result
        ), "Generation result missing 'raw_generated_text'"

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
            system_content=system_content,
            token_sets=[],
        )

        # Process token sets if available
        if "token_sets" in generation_result:
            # Validate token sets format
            token_sets = generation_result["token_sets"]
            assert isinstance(token_sets, list), "token_sets must be a list"

            # Convert token sets to required format
            response.token_sets = [
                (
                    position,
                    [(int(tid), float(prob)) for tid, prob in token_ids_probs[0]],
                    [(int(tid), float(prob)) for tid, prob in token_ids_probs[1]],
                )
                for position, token_ids_probs, _ in token_sets
            ]

            # Build steps for the response
            steps = []
            for (
                position,
                (orig_tokens, orig_probs),
                (pruned_tokens, pruned_probs),
            ) in token_sets:
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
            response.pruning = PruningInfo(
                strategy=request.pruning_strategy,
                coherence_threshold=request.coherence_threshold,
                diversity_clusters=request.diversity_clusters,
                use_dynamic_threshold=request.dynamic_threshold,
                diversity_steps=request.diversity_steps,
                final_threshold=request.final_threshold,
                use_relu=request.use_relu,
                relu_activation_point=request.relu_activation_point,
                bezier_points=request.bezier_points,
                pruning_time=generation_result.get("pruning_time", 0.0),
            )

            # Add retroactive pruning information if available
            if request.use_retroactive_pruning:
                response.retroactive_pruning = RetroactivePruningInfo(
                    attention_threshold=request.attention_threshold,
                    use_relative_attention=not request.no_relative_attention,
                    relative_threshold=request.relative_threshold,
                    use_multi_scale_attention=not request.no_multi_scale_attention,
                    num_layers_to_use=request.num_layers_to_use,
                    use_lci_dynamic_threshold=not request.no_lci_dynamic_threshold,
                    use_sigmoid_threshold=not request.no_sigmoid_threshold,
                    sigmoid_steepness=request.sigmoid_steepness,
                    complete_pruning_mode=request.complete_pruning_mode,
                )

        # Add position to tokens mapping if available
        if "position_to_tokens" in generation_result:
            response.position_to_tokens = generation_result["position_to_tokens"]

        # Add final pruned sets if available
        if "final_pruned_sets" in generation_result:
            response.final_pruned_sets = generation_result["final_pruned_sets"]

        # Add original parallel positions if available
        if "original_parallel_positions" in generation_result:
            response.original_parallel_positions = list(
                generation_result["original_parallel_positions"]
            )

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

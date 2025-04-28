from fastapi import FastAPI, HTTPException, Depends, APIRouter
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
from src.pruning import RetroactivePruner
from src.generation.token_generator import TokenGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tempo-api")


# Singleton for model and components to keep them in memory
class ModelSingleton:
    model_wrapper = None  # Store the wrapped model
    tokenizer = None
    generator = None  # This will be ParallelGenerator
    token_generator = None  # Store the shared TokenGenerator
    initialized = False

    @classmethod
    def get_instance(cls, device="mps"):
        """Get or initialize model instance, maintaining the singleton invariant"""
        if not cls.initialized:
            logger.info("Loading model for the first time...")
            cls._initialize_model(device)
            cls.initialized = True

        # INVARIANT: After initialization, all components must exist
        assert cls.model_wrapper is not None, "Model Wrapper initialization failed"
        assert cls.tokenizer is not None, "Tokenizer initialization failed"
        assert cls.generator is not None, "Generator initialization failed"
        assert cls.token_generator is not None, "TokenGenerator initialization failed"
        # Ensure the generator has its internal token_generator
        assert hasattr(cls.generator, 'token_generator') and cls.generator.token_generator is not None, "Singleton Generator missing internal TokenGenerator"

        return cls.model_wrapper, cls.tokenizer, cls.generator, cls.token_generator

    @classmethod
    def _initialize_model(cls, device):
        """Initialize model components with proper error handling"""
        try:
            # Determine the device and precision
            def get_device():
                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    return "mps"
                return "cpu"

            def get_device_dtype(device_str):
                if device_str == "cuda":
                    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                        return torch.bfloat16
                    else:
                        return torch.float16
                elif device_str == "mps":
                    return torch.float32
                else:  # cpu
                    return torch.float32

            device_str = get_device()
            dtype = get_device_dtype(device_str)

            logger.info(f"Using device: {device_str} with dtype: {dtype}")

            # Load model and tokenizer
            model_name = "deepcogito/cogito-v1-preview-llama-3B"
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto" if device_str == "cuda" else None,
                attn_implementation="eager"  # Force eager attention implementation
            )

            if device_str != "cuda":
                model = model.to(device_str)

            # Create model wrapper
            model_wrapper = TEMPOModelWrapper(model)
            logger.info(f"Model wrapper created for device: {model_wrapper.device}")

            # Create the SHARED TokenGenerator instance
            cls.token_generator = TokenGenerator(
                model=model_wrapper,
                tokenizer=tokenizer,
                device=device_str
            )
            logger.info(f"Shared TokenGenerator created for device: {cls.token_generator.device}")

            # Create ParallelGenerator, passing the SHARED token_generator
            cls.generator = ParallelGenerator(
                model=model_wrapper,
                tokenizer=tokenizer,
                device=device_str,
                use_custom_rope=True,
                debug_mode=False,
                token_generator=cls.token_generator  # Pass the shared instance
            )
            logger.info(f"ParallelGenerator created with shared TokenGenerator (ID: {id(cls.token_generator)})")

            # Store components
            cls.model_wrapper = model_wrapper
            cls.tokenizer = tokenizer

        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Model initialization failed: {str(e)}"
            )


# Create FastAPI app
app = FastAPI(
    title="TEMPO API",
    description="API for TEMPO text generation with invariant guarantees",
)

# Create API router with /api prefix
api_router = APIRouter(prefix="/api")

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
    selection_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Probability threshold for initial token candidate selection - lower values allow more potential paths",
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
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode for detailed logging and performance information",
    )

    # MCTS parameters
    use_mcts: bool = Field(
        default=False,
        description="Use Monte Carlo Tree Search for text generation",
    )
    mcts_simulations: int = Field(
        default=10,
        ge=1,
        description="Number of MCTS simulations per step",
    )
    mcts_c_puct: float = Field(
        default=1.0,
        ge=0.0,
        description="Exploration constant for MCTS",
    )
    mcts_depth: int = Field(
        default=5,
        ge=1,
        description="Maximum depth for MCTS simulations",
    )

    # Dynamic threshold parameters
    dynamic_threshold: bool = Field(
        default=False,
        description="Use dynamic threshold that increases over steps",
    )
    final_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Final threshold value for dynamic thresholding",
    )
    bezier_p1: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="First Bezier control point for dynamic threshold",
    )
    bezier_p2: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Second Bezier control point for dynamic threshold",
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

    # Pruning options
    use_retroactive_pruning: bool = Field(
        default=True,
        description="Use retroactive pruning to refine token sets based on future token attention",
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
        description="Disable KV cache consistency checks (faster but may cause issues)",
    )

    # Validator for prompt
    @field_validator("prompt")
    def prompt_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v

    # Validator for bezier points
    @field_validator("bezier_p1", "bezier_p2")
    def validate_bezier_points(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Bezier points must be between 0 and 1")
        return v

    # Validator for complete pruning mode
    @field_validator("complete_pruning_mode")
    def validate_complete_pruning_mode(cls, v):
        valid_modes = ["keep_token", "keep_unattended", "remove_position"]
        if v not in valid_modes:
            raise ValueError(f"Complete pruning mode must be one of {valid_modes}")
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
    selection_threshold: float
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
        # Returns model_wrapper, tokenizer, singleton_generator, shared_token_generator
        return ModelSingleton.get_instance()
    except Exception as e:
        logger.error(f"Failed to get model components: {str(e)}")
        raise HTTPException(status_code=503, detail="Model initialization failed")


# Create visualizer singletons
token_visualizer = TokenVisualizer()
position_visualizer = PositionVisualizer()

# Cache to store recent generation results for visualization
generation_cache = {}


@api_router.get("/")
async def root():
    return {"message": "TEMPO API is running", "status": "healthy"}


@api_router.get("/health")
async def health_check():
    """Health check endpoint to verify API and model status"""
    try:
        # Quick check if model is initialized
        if not ModelSingleton.initialized:
            return {"status": "initializing", "message": "Model is not yet initialized"}

        # Verify components exist
        model_wrapper, tokenizer, generator, token_generator = ModelSingleton.get_instance()

        return {
            "status": "healthy",
            "model_loaded": True,
            "model_name": "deepcogito/cogito-v1-preview-llama-3B",
            "device": generator.device if hasattr(generator, "device") else "unknown",
            "token_generator_initialized": token_generator is not None,
            "generator_has_token_generator": hasattr(generator, 'token_generator') and generator.token_generator is not None
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}


@api_router.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    components: Tuple = Depends(get_model_components)  # components = (model_wrapper, tokenizer, generator, token_generator)
):
    """
    Generate text using TEMPO parallel generation.
    """
    try:
        model_wrapper, tokenizer, generator, shared_token_generator = components

        # --- Propagate Debug Mode from Request ---
        # Set on the shared TokenGenerator
        shared_token_generator.set_debug_mode(request.debug_mode)
        # Set on the singleton ParallelGenerator
        if hasattr(generator, 'set_debug_mode'):
            generator.set_debug_mode(request.debug_mode)
        # Set on the Model Wrapper
        if hasattr(model_wrapper, 'set_debug_mode'):
            model_wrapper.set_debug_mode(request.debug_mode)

        # Log the request
        logger.info(
            f"Received generation request: prompt={request.prompt[:50]}..., max_tokens={request.max_tokens}, debug={request.debug_mode}"
        )
        start_time = time.time()

        # --- Pruner Creation ---
        retroactive_pruner = None
        if request.use_retroactive_pruning:
            try:
                # Create RetroactivePruner
                retroactive_pruner = RetroactivePruner(
                    model=model_wrapper,
                    tokenizer=tokenizer,
                    device=generator.device,
                    debug_mode=request.debug_mode,
                    use_relative_attention=not request.no_relative_attention,
                    relative_threshold=request.relative_threshold,
                    use_multi_scale_attention=not request.no_multi_scale_attention,
                    num_layers_to_use=request.num_layers_to_use,
                    use_sigmoid_threshold=not request.no_sigmoid_threshold,
                    sigmoid_steepness=request.sigmoid_steepness,
                    complete_pruning_mode=request.complete_pruning_mode,
                )
                # Set the SHARED token generator on the retroactive pruner
                if hasattr(retroactive_pruner, 'set_token_generator'):
                    retroactive_pruner.set_token_generator(shared_token_generator)
                    logger.info(f"Set shared TokenGenerator (ID: {id(shared_token_generator)}) on RetroactivePruner")
                else:
                    logger.warning("RetroactivePruner does not have set_token_generator method.")

                logger.info(f"Created retroactive pruner with threshold: {request.attention_threshold}")

            except ImportError as e:
                logger.error(f"Failed to import pruning components: {e}")
                raise HTTPException(status_code=500, detail="Server configuration error: Pruning components not available")
            except Exception as e:
                logger.error(f"Failed to initialize retroactive pruning: {e}")
                logger.error(traceback.format_exc())  # Log full traceback for init errors
                raise HTTPException(status_code=500, detail=f"Failed to initialize retroactive pruning: {str(e)}")

        # Configure RoPE modifier KV cache consistency if RoPE is enabled
        if generator.use_custom_rope and hasattr(generator, "rope_modifier") and generator.rope_modifier is not None:
            if hasattr(generator.rope_modifier, 'enable_kv_cache_consistency'):
                if request.disable_kv_cache_consistency:
                    logger.info("Note: RoPE modifier KV cache consistency setting ignored (likely deprecated).")
            # Set debug mode on RoPE modifier instance
            generator.rope_modifier.set_debug_mode(request.debug_mode)

        # Prepare system content
        system_content = request.system_content
        if request.enable_thinking and not system_content:
            system_content = "Enable deep thinking subroutine."

        try:
            # --- Call the SINGLETON's generator ---
            # Pass the newly created pruners for this request
            generation_result = generator.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                selection_threshold=request.selection_threshold,  # Initial selection threshold
                return_parallel_sets=True,  # Needed for visualization
                use_retroactive_pruning=request.use_retroactive_pruning,  # Control whether retroactive pruning is applied
                min_steps=request.min_steps,
                show_token_ids=request.show_token_ids,
                # debug_mode already set on the generator instance
                disable_kv_cache=request.disable_kv_cache,
                system_content=system_content,
                isolate_parallel_tokens=not request.allow_intraset_token_visibility,
                preserve_all_isolated_tokens=(
                    not request.no_preserve_isolated_tokens
                    if not request.allow_intraset_token_visibility
                    else None
                ),
                retroactive_pruner=retroactive_pruner,  # Pass the request-specific RetroactivePruner
                # New MCTS parameters
                use_mcts=request.use_mcts,
                mcts_simulations=request.mcts_simulations,
                mcts_c_puct=request.mcts_c_puct,
                mcts_depth=request.mcts_depth,
                # New dynamic threshold parameters
                dynamic_threshold=request.dynamic_threshold,
                final_threshold=request.final_threshold,
                bezier_p1=request.bezier_p1,
                bezier_p2=request.bezier_p2,
                use_relu=request.use_relu,
                relu_activation_point=request.relu_activation_point,
            )
        except ValueError as e:
            logger.error(f"Value error during generation: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=400, detail=f"Invalid generation parameters: {str(e)}")
        except RuntimeError as e:
            logger.error(f"Runtime error during generation: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory: {e}")
            raise HTTPException(status_code=500, detail="GPU memory exceeded. Try reducing max_tokens or batch size.")
        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Generation failed unexpectedly: {str(e)}")

        elapsed_time = time.time() - start_time
        logger.info(f"Generation completed in {elapsed_time:.2f}s")

        try:
            # Format response with proper error handling
            response = GenerationResponse(
                generated_text=generation_result["generated_text"],
                raw_generated_text=generation_result.get("raw_generated_text", ""),
                steps=[],  # Populated below
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
                selection_threshold=request.selection_threshold,
                max_tokens=request.max_tokens,
                min_steps=request.min_steps,
                prompt=request.prompt,
                had_repetition_loop=generation_result.get("had_repetition_loop", False),
                system_content=system_content,
                token_sets=[],  # Populated below
                position_to_tokens=generation_result.get("position_to_tokens", {}),
                original_parallel_positions=list(generation_result.get("original_parallel_positions", set())),
                final_pruned_sets=generation_result.get("final_pruned_sets", {}),
            )

            # Process token sets safely
            token_sets_data = generation_result.get("token_sets", [])
            if token_sets_data:
                steps_list = []
                formatted_token_sets = []
                for step_data in token_sets_data:
                    try:
                        if isinstance(step_data, tuple) and len(step_data) == 3:
                            position, original_data, pruned_data = step_data
                            if (isinstance(original_data, tuple) and len(original_data) == 2 and
                                isinstance(pruned_data, tuple) and len(pruned_data) == 2):

                                original_ids, original_probs = original_data
                                pruned_ids_raw, pruned_probs_raw = pruned_data

                                # Convert to basic types safely
                                original_pairs = [(int(tid), float(prob)) for tid, prob in zip(original_ids, original_probs)]
                                pruned_pairs = [(int(tid), float(prob)) for tid, prob in zip(pruned_ids_raw, pruned_probs_raw)]

                                formatted_token_sets.append((position, original_pairs, pruned_pairs))

                                # Build step info
                                try:
                                    parallel_tokens = [
                                        TokenInfo(
                                            token_text=tokenizer.decode([tid]),
                                            token_id=tid,
                                            probability=prob
                                        )
                                        for tid, prob in original_pairs
                                    ]
                                    pruned_tokens_info = [
                                        TokenInfo(
                                            token_text=tokenizer.decode([tid]),
                                            token_id=tid,
                                            probability=prob
                                        )
                                        for tid, prob in pruned_pairs
                                    ]
                                    steps_list.append(StepInfo(
                                        position=position,
                                        parallel_tokens=parallel_tokens,
                                        pruned_tokens=pruned_tokens_info
                                    ))
                                except Exception as e:
                                    logger.warning(f"Error processing tokens for step {position}: {e}")
                                    continue
                            else:
                                logger.warning(f"Skipping malformed token_set inner data: {step_data}")
                        else:
                            logger.warning(f"Skipping malformed token_set step data: {step_data}")
                    except Exception as e:
                        logger.warning(f"Error processing step data: {e}")
                        continue

                response.token_sets = formatted_token_sets
                response.steps = steps_list

            # Add pruning info safely
            if request.use_retroactive_pruning:
                response.retroactive_pruning = RetroactivePruningInfo(
                    attention_threshold=request.attention_threshold,
                    use_relative_attention=not request.no_relative_attention,
                    relative_threshold=request.relative_threshold,
                    use_multi_scale_attention=not request.no_multi_scale_attention,
                    num_layers_to_use=request.num_layers_to_use,
                    use_sigmoid_threshold=not request.no_sigmoid_threshold,
                    sigmoid_steepness=request.sigmoid_steepness,
                    complete_pruning_mode=request.complete_pruning_mode,
                )

            return response

        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Failed to format generation response: {str(e)}")

    except ValueError as e:
        logger.error(f"Validation Error during generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Generation parameter error: {str(e)}")
    except RuntimeError as e:
        logger.error(f"Runtime Error during generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected Error during generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


# Include the API router
app.include_router(api_router)

# Run with: uvicorn api:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

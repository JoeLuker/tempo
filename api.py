from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Tuple
import torch
import time
import json
import os
import io
import uuid
import logging
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from src.modeling.model_wrapper import TEMPOModelWrapper
from src.generation.parallel_generator import ParallelGenerator
from src.pruning.pruner import Pruner
from src.visualization.token_visualizer import TokenVisualizer
from src.visualization.position_visualizer import PositionVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if device != "cpu" else torch.float32
            
            logger.info(f"Using device: {device} with {dtype}")
            
            # Load Cogito model
            model_name = "deepcogito/cogito-v1-preview-qwen-14B"
            
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
                logger.info(f"Disabling sliding window attention (was set to {config.sliding_window})")
                config.sliding_window = None
                
            # Load model with optimized settings and modified config
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else device,
                low_cpu_mem_usage=True,
                attn_implementation="eager"  # Use eager attention for better compatibility
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
            
            # Create Pruner with default settings
            pruner = Pruner(
                model=wrapped_model,
                tokenizer=cls.tokenizer,
                strategy="hybrid",
                coherence_threshold=0.3,
                diversity_clusters=3,
                device=device,
                use_dynamic_threshold=True,
                max_steps=20,
                final_threshold=1.0,
                diversity_steps=5
            )
            
            # Create ParallelGenerator
            cls.generator = ParallelGenerator(
                model=wrapped_model,
                tokenizer=cls.tokenizer,
                pruner=pruner,
                device=device,
                has_custom_attention=True
            )
            
            # INVARIANT: Generator must be properly initialized
            assert hasattr(cls.generator, "generate"), "Generator missing generate method"
            
            logger.info("Model, tokenizer, and generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

# Create FastAPI app
app = FastAPI(title="TEMPO API", description="API for TEMPO text generation with invariant guarantees")

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
    prompt: str = Field(
        description="Text prompt to start generation"
    )
    max_tokens: int = Field(
        default=50, ge=1, le=200,
        description="Maximum number of tokens to generate"
    )
    threshold: float = Field(
        default=0.1, ge=0.0, le=1.0,
        description="Probability threshold for token selection - lower values allow more diverse completion paths"
    )
    
    # Advanced generation settings
    min_steps: int = Field(
        default=0, ge=0,
        description="Minimum steps to generate before considering EOS tokens"
    )
    use_custom_rope: bool = Field(
        default=True,
        description="Use custom RoPE modifications for improved parallel token positioning"
    )
    disable_kv_cache: bool = Field(
        default=False,
        description="Disable KV caching for more consistent attention patterns (slower but more accurate)"
    )
    show_token_ids: bool = Field(
        default=False,
        description="Include token IDs in the formatted output"
    )
    system_content: Optional[str] = Field(
        default=None,
        description="Optional system message content for chat models to adjust generation behavior"
    )
    enable_thinking: bool = Field(
        default=False,
        description="Enable Cogito's deep thinking mode for more thoughtful responses"
    )
    
    # Pruning options
    use_pruning: bool = Field(
        default=True,
        description="Use pruning to reduce token sets for more coherent generation"
    )
    pruning_strategy: str = Field(
        default="hybrid",
        description="Strategy for pruning parallel tokens: coherence focuses on text quality, diversity on exploration"
    )
    coherence_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Threshold for coherence pruning - higher values require greater coherence"
    )
    diversity_clusters: int = Field(
        default=3, ge=1, le=10,
        description="Number of clusters for diversity pruning - more clusters = more diverse completions"
    )
    diversity_steps: int = Field(
        default=5, ge=0,
        description="Number of steps to use diversity pruning before switching to coherence (for hybrid strategy)"
    )
    bezier_points: List[float] = Field(
        default=[0.2, 0.8],
        description="Control points for dynamic threshold Bezier curve (when dynamic thresholding is used)"
    )
    dynamic_threshold: bool = Field(
        default=True,
        description="Use dynamic thresholding that starts with diverse completions and gradually increases coherence"
    )
    
    # Validator for prompt
    @field_validator('prompt')
    def prompt_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v.strip()
    
    # Validator for pruning strategy
    @field_validator('pruning_strategy')
    def validate_pruning_strategy(cls, v):
        valid_strategies = ["coherence", "diversity", "hybrid"]
        if v not in valid_strategies:
            raise ValueError(f"Pruning strategy must be one of: {', '.join(valid_strategies)}")
        return v
    
    # Validator for bezier points
    @field_validator('bezier_points')
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
            "model_name": "deepcogito/cogito-v1-preview-qwen-14B",
            "device": generator.device if hasattr(generator, "device") else "unknown"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    components: Tuple = Depends(get_model_components)
):
    """Generate text using TEMPO with invariant guarantees"""
    start_time = time.time()
    model, tokenizer, generator = components
    
    try:
        logger.info(f"Processing generation request with threshold={request.threshold}, tokens={request.max_tokens}")
        
        # INVARIANT: Use defensive parameter validation despite Pydantic
        if request.threshold < 0 or request.threshold > 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {request.threshold}")
        
        if request.max_tokens < 1:
            raise ValueError(f"Max tokens must be positive, got {request.max_tokens}")
            
        # Create pruner with requested settings if needed
        if request.use_pruning:
            pruner = Pruner(
                model=model,
                tokenizer=tokenizer,
                strategy=request.pruning_strategy,
                coherence_threshold=request.coherence_threshold,
                diversity_clusters=request.diversity_clusters,
                device="mps",
                use_dynamic_threshold=request.dynamic_threshold,
                max_steps=request.max_tokens,
                final_threshold=1.0,
                diversity_steps=request.diversity_steps,
                bezier_points=request.bezier_points
            )
            generator.pruner = pruner
        
        try:
            # Run TEMPO generation
            results = generator.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                threshold=request.threshold,
                return_parallel_sets=True,
                use_pruning=request.use_pruning,
                min_steps=request.min_steps,
                show_token_ids=request.show_token_ids,
                debug_mode=False,  # We don't expose debug mode in the API
                disable_kv_cache=request.disable_kv_cache,
                system_content=request.system_content if request.enable_thinking else None
            )
            # Debug log to see the structure of results
            logger.info(f"Generator returned keys: {list(results.keys())}")
            
            # If token_sets missing but parallel_tokens exists, adapt the structure
            if "token_sets" not in results and "parallel_tokens" in results:
                logger.info("Adapting parallel_tokens to token_sets format")
                parallel_tokens = results.get("parallel_tokens", [])
                token_sets = []
                
                # Convert parallel_tokens to token_sets format
                for i, tokens in enumerate(parallel_tokens):
                    if isinstance(tokens, list) and len(tokens) > 0:
                        # Each entry should be position, original_tokens, pruned_tokens
                        token_sets.append((i, tokens, []))
                
                # Update results with the converted token_sets
                results["token_sets"] = token_sets
                
            # Process position_to_tokens and final_pruned_sets early if token_sets is missing
            if "token_sets" not in results:
                logger.info("Processing position_to_tokens for Cogito model output")
                # This will be handled comprehensively after error handling
        
        except RuntimeError as e:
            # Handle the specific "No tokens above threshold" error
            if "No tokens above threshold" in str(e):
                logger.error(f"Generation error: {str(e)}")
                # Make sure we provide a token_sets array, even if empty
                results = {
                    "generated_text": f"{request.prompt} [Generation stopped: No tokens above threshold. Try lowering the threshold value.]",
                    "token_sets": []
                }
            else:
                # Re-raise other runtime errors
                raise
        
        # INVARIANT: Generation results must contain required fields
        if "generated_text" not in results:
            logger.warning(f"Generation results missing 'generated_text' field")
            raise ValueError("Generation results missing 'generated_text' field")
            
        if "token_sets" not in results:
            logger.warning(f"Generation results missing 'token_sets' field")
            # Instead of raising an error, create token_sets from position_to_tokens and final_pruned_sets
            logger.info("Constructing token_sets from position_to_tokens and final_pruned_sets")
            
            # Check if we have position_to_tokens
            position_to_tokens = results.get("position_to_tokens", {})
            final_pruned_sets = results.get("final_pruned_sets", {})
            
            # Quick sanity check on the structures
            if not isinstance(position_to_tokens, dict):
                logger.warning(f"position_to_tokens is not a dictionary! Type: {type(position_to_tokens)}")
                position_to_tokens = {}
                
            if not isinstance(final_pruned_sets, dict):
                logger.warning(f"final_pruned_sets is not a dictionary! Type: {type(final_pruned_sets)}")
                # Convert to dictionary or empty dictionary if unexpected type
                try:
                    if isinstance(final_pruned_sets, list):
                        logger.info("Converting final_pruned_sets from list to dictionary")
                        converted_final_pruned_sets = {}
                        for i, item in enumerate(final_pruned_sets):
                            converted_final_pruned_sets[str(i)] = item
                        final_pruned_sets = converted_final_pruned_sets
                    else:
                        final_pruned_sets = {}
                except Exception as e:
                    logger.error(f"Failed to convert final_pruned_sets: {str(e)}")
                    final_pruned_sets = {}
            
            # Add detailed logging for debugging
            if position_to_tokens:
                logger.info(f"Found position_to_tokens with {len(position_to_tokens)} positions")
                
                # Log keys in position_to_tokens
                logger.info(f"Position_to_tokens keys: {list(position_to_tokens.keys())}")
                
                # Log the structure of the first position for debugging
                if position_to_tokens:
                    first_pos = next(iter(position_to_tokens))
                    if isinstance(position_to_tokens[first_pos], dict):
                        logger.info(f"Position {first_pos} has keys: {list(position_to_tokens[first_pos].keys())}")
                    else:
                        logger.info(f"Position {first_pos} value type: {type(position_to_tokens[first_pos])}")
                    logger.info(f"Example position_to_tokens structure: {position_to_tokens[first_pos]}")
            else:
                logger.warning("No position_to_tokens data found")
                
            if final_pruned_sets:
                logger.info(f"Found final_pruned_sets with {len(final_pruned_sets)} positions")
                # Log the structure of the first position for debugging
                if final_pruned_sets:
                    first_pos = next(iter(final_pruned_sets))
                    if isinstance(final_pruned_sets[first_pos], dict):
                        logger.info(f"Final pruned position {first_pos} has keys: {list(final_pruned_sets[first_pos].keys())}")
                    elif isinstance(final_pruned_sets[first_pos], list):
                        logger.info(f"Final pruned position {first_pos} is a list of length {len(final_pruned_sets[first_pos])}")
                        if final_pruned_sets[first_pos]:
                            logger.info(f"First element type: {type(final_pruned_sets[first_pos][0])}")
                    else:
                        logger.info(f"Final pruned position {first_pos} value type: {type(final_pruned_sets[first_pos])}")
                    logger.info(f"Example final_pruned_sets structure: {final_pruned_sets[first_pos]}")
            else:
                logger.warning("No final_pruned_sets data found")
            
            if position_to_tokens:
                # Convert position_to_tokens to token_sets format
                token_sets = []
                
                # Sort positions to ensure they're in order
                positions = sorted([int(pos) for pos in position_to_tokens.keys() if pos.isdigit()])
                
                for pos in positions:
                    pos_str = str(pos)
                    if pos_str in position_to_tokens:
                        # Format: (position, original_tokens, pruned_tokens)
                        original_tokens = []
                        
                        # Extract tokens and probabilities
                        tokens_data = position_to_tokens[pos_str]
                        
                        # Handle the case where tokens_data is a list of string tokens
                        if isinstance(tokens_data, list) and tokens_data and isinstance(tokens_data[0], str):
                            logger.info(f"Processing string tokens at position {pos}")
                            # For each string token, convert to token ID and add to original_tokens
                            for token_str in tokens_data:
                                try:
                                    # Encode the string token to get token ID
                                    token_ids = tokenizer.encode(token_str, add_special_tokens=False)
                                    if token_ids:
                                        # Use the first token ID if multiple are returned
                                        token_id = token_ids[0]
                                        # Use default probability since we don't have actual values
                                        original_tokens.append((token_id, 1.0))
                                except Exception as e:
                                    logger.warning(f"Failed to encode token '{token_str}': {str(e)}")
                                    continue
                        else:
                            # Handle different possible structures for non-string tokens
                            tokens = []
                            probs = []
                            
                            if isinstance(tokens_data, dict):
                                if "tokens" in tokens_data:
                                    tokens = tokens_data.get("tokens", [])
                                    probs = tokens_data.get("probs", [])
                                elif "top_tokens" in tokens_data:
                                    # Alternative structure sometimes used
                                    tokens = tokens_data.get("top_tokens", [])
                                    probs = tokens_data.get("top_probs", [])
                                elif "token_ids" in tokens_data:
                                    # Another possible structure
                                    tokens = tokens_data.get("token_ids", [])
                                    probs = tokens_data.get("probabilities", [])
                            elif isinstance(tokens_data, list):
                                # Handle list of (token, prob) pairs or similar
                                if tokens_data and isinstance(tokens_data[0], (list, tuple)) and len(tokens_data[0]) >= 2:
                                    # Extract tokens and probs from list of pairs
                                    tokens = [item[0] for item in tokens_data]
                                    probs = [item[1] for item in tokens_data]
                                elif tokens_data:
                                    # Just a list of tokens, use default prob
                                    tokens = tokens_data
                                    probs = [0.5] * len(tokens)  # Default probability
                                
                            # Create (token_id, prob) pairs with defensive programming
                            for i, token_id in enumerate(tokens):
                                prob = probs[i] if i < len(probs) else 0.0
                                # Ensure token_id is an int and prob is a float
                                try:
                                    token_id_int = int(token_id)
                                    prob_float = float(prob)
                                    original_tokens.append((token_id_int, prob_float))
                                except (ValueError, TypeError):
                                    logger.warning(f"Invalid token data at position {pos}: {token_id}, {prob}")
                                    continue
                            
                        # Extract pruned tokens if available
                        pruned_tokens = []
                        try:
                            if pos_str in final_pruned_sets:
                                pruned_data = final_pruned_sets[pos_str]
                                
                                # Handle string tokens in pruned data
                                if isinstance(pruned_data, list) and pruned_data and isinstance(pruned_data[0], str):
                                    logger.info(f"Processing pruned string tokens at position {pos}")
                                    for token_str in pruned_data:
                                        try:
                                            # Encode the string token to get token ID
                                            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
                                            if token_ids:
                                                # Use the first token ID if multiple are returned
                                                token_id = token_ids[0]
                                                # Use default probability
                                                pruned_tokens.append((token_id, 0.8))
                                        except Exception as e:
                                            logger.warning(f"Failed to encode pruned token '{token_str}': {str(e)}")
                                            continue
                                elif isinstance(pruned_data, dict) and "tokens" in pruned_data:
                                    p_tokens = pruned_data.get("tokens", [])
                                    p_probs = pruned_data.get("probs", [])
                                    
                                    for i, token_id in enumerate(p_tokens):
                                        prob = p_probs[i] if i < len(p_probs) else 0.0
                                        try:
                                            token_id_int = int(token_id)
                                            prob_float = float(prob)
                                            pruned_tokens.append((token_id_int, prob_float))
                                        except (ValueError, TypeError):
                                            logger.warning(f"Invalid pruned token data at position {pos}: {token_id}, {prob}")
                                            continue
                        except Exception as e:
                            logger.error(f"Error processing pruned tokens at position {pos}: {str(e)}")
                            # Continue with empty pruned tokens if there was an error
                            pruned_tokens = []
                            
                        token_sets.append((pos, original_tokens, pruned_tokens))
                
                results["token_sets"] = token_sets
            else:
                # If no position_to_tokens, create empty token_sets
                results["token_sets"] = []
                
        # Extract the token sets and format them for the response
        steps = []
        token_sets = results.get("token_sets", [])
        
        for position, original_tokens, pruned_tokens in token_sets:
            # INVARIANT: Token sets must be properly structured
            if not isinstance(position, int):
                raise ValueError(f"Invalid position type: {type(position)}")
                
            if not all(isinstance(t, tuple) and len(t) == 2 for t in original_tokens):
                raise ValueError("Invalid token format in original_tokens")
                
            if not all(isinstance(t, tuple) and len(t) == 2 for t in pruned_tokens):
                raise ValueError("Invalid token format in pruned_tokens")
            
            # Create token info objects for original tokens
            original_token_infos = []
            for token_id, prob in original_tokens:
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                original_token_infos.append(TokenInfo(
                    token_text=token_text,
                    token_id=token_id,
                    probability=float(prob)  # Ensure float type
                ))
                
            # Create token info objects for pruned tokens
            pruned_token_infos = []
            for token_id, prob in pruned_tokens:
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                pruned_token_infos.append(TokenInfo(
                    token_text=token_text,
                    token_id=token_id,
                    probability=float(prob)  # Ensure float type
                ))
                
            steps.append(StepInfo(
                position=position,
                parallel_tokens=original_token_infos,
                pruned_tokens=pruned_token_infos
            ))
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Prepare pruning info if used
        pruning_info = None
        if request.use_pruning:
            pruning_info = PruningInfo(
                strategy=request.pruning_strategy,
                coherence_threshold=request.coherence_threshold,
                diversity_clusters=request.diversity_clusters,
                use_dynamic_threshold=True,  # This is hardcoded in the API
                diversity_steps=request.diversity_steps,
                pruning_time=results.get("pruning_time", 0.0)
            )
            
        # Extract original parallel positions if available
        original_parallel_positions = []
        if "original_parallel_positions" in results and isinstance(results["original_parallel_positions"], (list, set)):
            original_parallel_positions = list(results["original_parallel_positions"])
        
        # Format the response with all available data
        response = GenerationResponse(
            # Core output
            generated_text=results.get("generated_text", ""),
            raw_generated_text=results.get("raw_generated_text", ""),
            
            # Token-level data
            steps=steps,
            position_to_tokens=results.get("position_to_tokens", {}),
            original_parallel_positions=original_parallel_positions,
            
            # Performance and timing
            timing=TimingInfo(
                generation_time=results.get("generation_time", elapsed_time),
                pruning_time=results.get("pruning_time", 0.0),
                elapsed_time=elapsed_time
            ),
            
            # Pruning information
            pruning=pruning_info,
            
            # Model information
            model_info=ModelInfo(
                model_name="deepcogito/cogito-v1-preview-qwen-14B",
                is_qwen_model=results.get("is_qwen_model", True),
                use_custom_rope=results.get("use_custom_rope", True)
            ),
            
            # Generation settings
            threshold=request.threshold,
            max_tokens=request.max_tokens,
            min_steps=request.min_steps,
            prompt=request.prompt,
            
            # Advanced fields
            had_repetition_loop=results.get("had_repetition_loop", False),
            system_content=results.get("system_content", None),
            
            # Token sets data for visualization (raw data from generator)
            token_sets=token_sets,
            
            # Raw token information for visualization
            tokens_by_position=results.get("tokens_by_position", {}),
            final_pruned_sets=results.get("final_pruned_sets", {})
        )
        
        logger.info(f"Generation completed in {elapsed_time:.2f}s, produced {len(steps)} steps")
        
        return response
        
    except ValueError as e:
        # Client errors (invalid parameters)
        logger.warning(f"Client error during generation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        # Server errors (model failures, etc.)
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

# Run with: uvicorn api:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
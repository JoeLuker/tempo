"""
Improved API schemas for TEMPO generation with clear naming and structure.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from enum import Enum


class PruningMode(str, Enum):
    """How to handle completely removed positions."""
    KEEP_BEST = "keep_best"  # Keep the highest probability token
    MARK_UNATTENDED = "mark_unattended"  # Mark position as unattended
    REMOVE_POSITION = "remove_position"  # Remove the position entirely


class GenerationConfig(BaseModel):
    """Core generation parameters."""
    max_tokens: int = Field(default=50, ge=1, le=500, description="Maximum tokens to generate")
    selection_threshold: float = Field(
        default=0.1, ge=0.0, le=1.0,
        description="Probability threshold for parallel path selection (lower = more paths)"
    )
    min_steps_before_eos: int = Field(
        default=0, ge=0,
        description="Minimum generation steps before allowing end-of-sequence tokens"
    )
    

class RoPEConfig(BaseModel):
    """Rotary Position Embedding configuration."""
    use_custom_rope: bool = Field(
        default=True,
        description="Enable custom RoPE modifications for parallel tokens"
    )
    disable_kv_cache: bool = Field(
        default=False,
        description="Disable KV cache for more accurate attention (slower)"
    )


class MCTSConfig(BaseModel):
    """Monte Carlo Tree Search configuration."""
    enabled: bool = Field(default=False, description="Enable MCTS-based generation")
    simulations_per_step: int = Field(default=10, ge=1, description="Number of simulations per step")
    exploration_constant: float = Field(default=1.0, ge=0.0, description="UCB exploration constant (c_puct)")
    max_depth: int = Field(default=5, ge=1, description="Maximum simulation depth")


class DynamicThresholdConfig(BaseModel):
    """Dynamic threshold adjustment configuration."""
    enabled: bool = Field(default=False, description="Enable dynamic threshold adjustment")
    final_threshold: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Target threshold at end of generation"
    )
    curve_type: str = Field(default="bezier", description="Curve type: 'bezier' or 'relu'")
    bezier_control_points: List[float] = Field(
        default=[0.2, 0.8],
        description="Bezier curve control points (2 values between 0 and 1)"
    )
    relu_activation_point: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Activation point for ReLU transition"
    )


class RetroactivePruningConfig(BaseModel):
    """Retroactive attention-based removal configuration."""
    enabled: bool = Field(default=True, description="Enable retroactive removal")
    attention_threshold: float = Field(
        default=0.01, ge=0.0, le=1.0,
        description="Minimum attention weight to retain token"
    )
    use_relative_attention: bool = Field(
        default=True,
        description="Compare attention relative to other tokens in the set"
    )
    relative_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Relative attention threshold"
    )
    layers_to_analyze: Optional[int] = Field(
        default=None,
        description="Number of last layers to analyze (None = all layers)"
    )
    pruning_mode: PruningMode = Field(
        default=PruningMode.KEEP_BEST,
        description="How to handle fully removed positions"
    )


class DebugOptions(BaseModel):
    """Debug and monitoring options."""
    debug_mode: bool = Field(default=False, description="Enable detailed logging")
    show_token_ids: bool = Field(default=False, description="Include token IDs in output")
    include_attention_weights: bool = Field(default=False, description="Include attention weights")
    profile_performance: bool = Field(default=False, description="Include performance metrics")


class GenerationRequestV2(BaseModel):
    """Simplified and organized generation request."""
    prompt: str = Field(description="Input text to continue from")
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt for chat models"
    )
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    rope: RoPEConfig = Field(default_factory=RoPEConfig)
    mcts: Optional[MCTSConfig] = None
    dynamic_threshold: Optional[DynamicThresholdConfig] = None
    pruning: Optional[RetroactivePruningConfig] = Field(default_factory=RetroactivePruningConfig)
    debug: DebugOptions = Field(default_factory=DebugOptions)
    
    @field_validator("prompt")
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v


# Response models with clear naming
class TokenChoice(BaseModel):
    """A single token option at a generation step."""
    token_id: int
    token_text: str
    probability: float
    was_selected: bool = Field(description="Whether this token was selected for the final output")


class GenerationStep(BaseModel):
    """A single step in the generation process."""
    position: int = Field(description="Position in the sequence")
    considered_tokens: List[TokenChoice] = Field(description="All tokens considered at this position")
    selected_tokens: List[TokenChoice] = Field(description="Tokens selected after removal")
    had_parallel_paths: bool = Field(description="Whether multiple paths were active at this position")


class GenerationMetadata(BaseModel):
    """Metadata about the generation process."""
    total_tokens_generated: int
    total_tokens_considered: int
    parallel_positions: List[int] = Field(description="Positions where parallel paths were explored")
    generation_time_seconds: float
    removal_time_seconds: float
    model_name: str
    device: str


class GenerationResponseV2(BaseModel):
    """Clear and structured generation response."""
    # Primary output
    text: str = Field(description="The final generated text")
    
    # Generation details
    steps: List[GenerationStep] = Field(description="Detailed generation steps")
    metadata: GenerationMetadata
    
    # Request echo
    prompt: str
    settings_used: Dict[str, Any] = Field(description="Actual settings used for generation")
    
    # Optional debug information
    debug_info: Optional[Dict[str, Any]] = None


class APIError(BaseModel):
    """Structured error response."""
    error_code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    request_id: Optional[str] = Field(default=None, description="Request tracking ID")
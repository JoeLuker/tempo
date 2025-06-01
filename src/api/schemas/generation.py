"""
Text generation schemas for TEMPO API.

This module defines Pydantic models for generation request and response schemas.
"""

from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Dict, Any, Optional, Union, Set, Literal


class PruningMode(str, Enum):
    """Supported pruning modes for the API."""

    KEEP_TOKEN = "keep_token"
    KEEP_UNATTENDED = "keep_unattended"
    REMOVE_POSITION = "remove_position"


class TokenInfo(BaseModel):
    """Information about a token, including text, ID, and probability."""

    token_text: str = Field(..., description="The decoded text of the token")
    token_id: int = Field(..., description="The token ID")
    probability: float = Field(..., description="The probability of the token")

    class Config:
        schema_extra = {
            "example": {"token_text": " the", "token_id": 262, "probability": 0.87654}
        }


class StepInfo(BaseModel):
    """Information about a generation step, including parallel and pruned tokens."""

    position: int = Field(..., description="The position in the generated sequence")
    parallel_tokens: List[TokenInfo] = Field(
        ..., description="The tokens considered at this position"
    )
    pruned_tokens: List[TokenInfo] = Field(
        ..., description="The tokens pruned at this position"
    )

    class Config:
        schema_extra = {
            "example": {
                "position": 3,
                "parallel_tokens": [
                    {"token_text": " the", "token_id": 262, "probability": 0.87654},
                    {"token_text": " a", "token_id": 263, "probability": 0.12345},
                ],
                "pruned_tokens": [
                    {"token_text": " an", "token_id": 264, "probability": 0.00001}
                ],
            }
        }


class ThresholdSettings(BaseModel):
    """Settings for dynamic thresholding."""

    use_dynamic_threshold: bool = Field(
        default=False, description="Use a threshold that changes over generation steps"
    )
    final_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Final threshold value for dynamic thresholding",
    )
    bezier_points: List[float] = Field(
        default=[0.2, 0.8],
        description="Bezier control points for dynamic threshold curve",
    )
    use_relu: bool = Field(
        default=False, description="Use ReLU transition instead of Bezier curve"
    )
    relu_activation_point: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Point at which ReLU transition begins (0-1)",
    )

    @validator("bezier_points")
    def validate_bezier_points(cls, v):
        """Validate that bezier points are between 0 and 1."""
        if len(v) != 2:
            raise ValueError("Bezier points must be a list of 2 values")
        if not all(0 <= point <= 1 for point in v):
            raise ValueError("All Bezier points must be between 0 and 1")
        return v


class MCTSSettings(BaseModel):
    """Settings for Monte Carlo Tree Search."""

    use_mcts: bool = Field(
        default=False, description="Use Monte Carlo Tree Search for text generation"
    )
    simulations: int = Field(
        default=10, ge=1, description="Number of MCTS simulations per step"
    )
    c_puct: float = Field(
        default=1.0, ge=0.0, description="Exploration constant for MCTS"
    )
    depth: int = Field(
        default=5, ge=1, description="Maximum depth for MCTS simulations"
    )


class RetroactivePruningSettings(BaseModel):
    """Settings for retroactive pruning."""

    enabled: bool = Field(
        default=True,
        description="Use retroactive pruning to refine token sets based on future token attention",
    )
    attention_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Attention threshold for pruning (lower means more tokens kept)",
    )
    use_relative_attention: bool = Field(
        default=True, description="Use relative attention thresholds"
    )
    relative_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for relative attention-based pruning (0-1)",
    )
    use_multi_scale_attention: bool = Field(
        default=True, description="Use multi-scale attention integration"
    )
    num_layers_to_use: Optional[int] = Field(
        default=None,
        description="Number of last layers to use for attention (None means use all layers)",
    )
    use_lci_dynamic_threshold: bool = Field(
        default=True, description="Use LCI-based dynamic thresholding"
    )
    use_sigmoid_threshold: bool = Field(
        default=True, description="Use sigmoid-based decision boundary"
    )
    sigmoid_steepness: float = Field(
        default=10.0, ge=1.0, description="Controls how sharp the sigmoid transition is"
    )
    pruning_mode: PruningMode = Field(
        default=PruningMode.KEEP_TOKEN, description="How to handle pruned positions"
    )


class AdvancedGenerationSettings(BaseModel):
    """Advanced settings for generation."""

    use_custom_rope: bool = Field(
        default=True, description="Use custom RoPE modifications for parallel tokens"
    )
    disable_kv_cache: bool = Field(
        default=False, description="Disable KV caching (slower but more consistent)"
    )
    disable_kv_cache_consistency: bool = Field(
        default=False, description="Disable KV cache consistency checks"
    )
    allow_intraset_token_visibility: bool = Field(
        default=False,
        description="Allow tokens within same parallel set to see each other",
    )
    no_preserve_isolated_tokens: bool = Field(
        default=False, description="Allow pruning isolated tokens"
    )
    show_token_ids: bool = Field(
        default=False, description="Include token IDs in formatted output"
    )
    system_content: Optional[str] = Field(
        default=None, description="System message content for chat models"
    )
    enable_thinking: bool = Field(
        default=False, description="Enable deep thinking mode"
    )
    debug_mode: bool = Field(
        default=False, description="Enable debug mode for detailed logging"
    )


class GenerationRequest(BaseModel):
    """
    Request model for text generation with TEMPO.

    Contains all parameters for configuring and controlling the generation process.
    """

    # Core parameters
    prompt: str = Field(
        ...,
        description="Text prompt to start generation",
        example="Explain the difference between a llama and an alpaca",
    )
    max_tokens: int = Field(
        default=50,
        ge=1,
        le=512,
        description="Maximum number of tokens to generate",
        example=50,
    )
    selection_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Probability threshold for token selection",
        example=0.1,
    )
    min_steps: int = Field(
        default=0,
        ge=0,
        description="Minimum steps to generate before considering EOS tokens",
        example=0,
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Model to use for generation (defaults to system model)",
        example="deepcogito/cogito-v1-preview-llama-3B",
    )

    # Group advanced settings in nested models for better organization
    threshold_settings: ThresholdSettings = Field(default_factory=ThresholdSettings)
    mcts_settings: MCTSSettings = Field(default_factory=MCTSSettings)
    pruning_settings: RetroactivePruningSettings = Field(
        default_factory=RetroactivePruningSettings
    )
    advanced_settings: AdvancedGenerationSettings = Field(
        default_factory=AdvancedGenerationSettings
    )

    # Validators
    @validator("prompt")
    def prompt_must_not_be_empty(cls, v):
        """Validate that prompt is not empty."""
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v

    @validator("model_name")
    def validate_model_name(cls, v):
        """Validate model name if provided."""
        if v is not None and not v.strip():
            raise ValueError("Model name cannot be empty if provided")
        return v

    # For backward compatibility with older clients
    @root_validator(pre=True)
    def map_legacy_fields(cls, values):
        """Map fields from old API format to new nested structure."""
        # Map threshold fields
        if "dynamic_threshold" in values:
            values.setdefault("threshold_settings", {})
            values["threshold_settings"]["use_dynamic_threshold"] = values.pop(
                "dynamic_threshold"
            )
        if "final_threshold" in values:
            values.setdefault("threshold_settings", {})
            values["threshold_settings"]["final_threshold"] = values.pop(
                "final_threshold"
            )
        if "bezier_p1" in values and "bezier_p2" in values:
            values.setdefault("threshold_settings", {})
            values["threshold_settings"]["bezier_points"] = [
                values.pop("bezier_p1"),
                values.pop("bezier_p2"),
            ]
        if "use_relu" in values:
            values.setdefault("threshold_settings", {})
            values["threshold_settings"]["use_relu"] = values.pop("use_relu")
        if "relu_activation_point" in values:
            values.setdefault("threshold_settings", {})
            values["threshold_settings"]["relu_activation_point"] = values.pop(
                "relu_activation_point"
            )

        # Map MCTS fields
        if "use_mcts" in values:
            values.setdefault("mcts_settings", {})
            values["mcts_settings"]["use_mcts"] = values.pop("use_mcts")
        if "mcts_simulations" in values:
            values.setdefault("mcts_settings", {})
            values["mcts_settings"]["simulations"] = values.pop("mcts_simulations")
        if "mcts_c_puct" in values:
            values.setdefault("mcts_settings", {})
            values["mcts_settings"]["c_puct"] = values.pop("mcts_c_puct")
        if "mcts_depth" in values:
            values.setdefault("mcts_settings", {})
            values["mcts_settings"]["depth"] = values.pop("mcts_depth")

        # Map retroactive pruning fields
        if "use_retroactive_pruning" in values:
            values.setdefault("pruning_settings", {})
            values["pruning_settings"]["enabled"] = values.pop(
                "use_retroactive_pruning"
            )
        if "attention_threshold" in values:
            values.setdefault("pruning_settings", {})
            values["pruning_settings"]["attention_threshold"] = values.pop(
                "attention_threshold"
            )
        if "no_relative_attention" in values:
            values.setdefault("pruning_settings", {})
            values["pruning_settings"]["use_relative_attention"] = not values.pop(
                "no_relative_attention"
            )
        if "relative_threshold" in values:
            values.setdefault("pruning_settings", {})
            values["pruning_settings"]["relative_threshold"] = values.pop(
                "relative_threshold"
            )
        if "no_multi_scale_attention" in values:
            values.setdefault("pruning_settings", {})
            values["pruning_settings"]["use_multi_scale_attention"] = not values.pop(
                "no_multi_scale_attention"
            )
        if "num_layers_to_use" in values:
            values.setdefault("pruning_settings", {})
            values["pruning_settings"]["num_layers_to_use"] = values.pop(
                "num_layers_to_use"
            )
        if "no_lci_dynamic_threshold" in values:
            values.setdefault("pruning_settings", {})
            values["pruning_settings"]["use_lci_dynamic_threshold"] = not values.pop(
                "no_lci_dynamic_threshold"
            )
        if "no_sigmoid_threshold" in values:
            values.setdefault("pruning_settings", {})
            values["pruning_settings"]["use_sigmoid_threshold"] = not values.pop(
                "no_sigmoid_threshold"
            )
        if "sigmoid_steepness" in values:
            values.setdefault("pruning_settings", {})
            values["pruning_settings"]["sigmoid_steepness"] = values.pop(
                "sigmoid_steepness"
            )
        if "complete_pruning_mode" in values:
            values.setdefault("pruning_settings", {})
            # Map string to enum
            mode = values.pop("complete_pruning_mode")
            try:
                values["pruning_settings"]["pruning_mode"] = PruningMode(mode)
            except ValueError:
                # Default to KEEP_TOKEN if invalid
                values["pruning_settings"]["pruning_mode"] = PruningMode.KEEP_TOKEN

        # Map advanced settings
        for field in [
            "use_custom_rope",
            "disable_kv_cache",
            "disable_kv_cache_consistency",
            "allow_intraset_token_visibility",
            "no_preserve_isolated_tokens",
            "show_token_ids",
            "system_content",
            "enable_thinking",
            "debug_mode",
        ]:
            if field in values:
                values.setdefault("advanced_settings", {})
                values["advanced_settings"][field] = values.pop(field)

        return values

    class Config:
        schema_extra = {
            "example": {
                "prompt": "Explain the difference between a llama and an alpaca",
                "max_tokens": 50,
                "selection_threshold": 0.1,
                "min_steps": 0,
                "threshold_settings": {
                    "use_dynamic_threshold": True,
                    "final_threshold": 0.9,
                    "bezier_points": [0.2, 0.8],
                    "use_relu": False,
                    "relu_activation_point": 0.5,
                },
                "pruning_settings": {
                    "enabled": True,
                    "attention_threshold": 0.01,
                    "use_relative_attention": True,
                    "relative_threshold": 0.5,
                },
                "advanced_settings": {
                    "use_custom_rope": True,
                    "disable_kv_cache": False,
                    "debug_mode": False,
                },
            }
        }


class TimingInfo(BaseModel):
    """Performance timing information for generation."""

    generation_time: float = Field(..., description="Time spent in generation process")
    pruning_time: float = Field(..., description="Time spent in pruning process")
    elapsed_time: float = Field(..., description="Total elapsed time for the request")


class ModelInfo(BaseModel):
    """Information about the model used for generation."""

    model_name: str = Field(..., description="The name or path of the model")
    is_qwen_model: bool = Field(..., description="Whether the model is a Qwen model")
    use_custom_rope: bool = Field(..., description="Whether custom RoPE was used")
    device: str = Field(..., description="Device used for computation")
    model_type: Optional[str] = Field(
        None, description="The model type (e.g., llama, gpt-neo, etc.)"
    )


class PruningInfo(BaseModel):
    """Information about pruning strategies used."""

    strategy: str = Field(..., description="The pruning strategy used")
    coherence_threshold: float = Field(
        ..., description="Coherence threshold for pruning"
    )
    diversity_clusters: int = Field(..., description="Number of diversity clusters")
    use_dynamic_threshold: bool = Field(
        ..., description="Whether dynamic threshold was used"
    )
    diversity_steps: int = Field(..., description="Number of diversity steps")
    final_threshold: float = Field(..., description="Final threshold value")
    use_relu: bool = Field(..., description="Whether ReLU transition was used")
    relu_activation_point: float = Field(..., description="ReLU activation point")
    bezier_points: List[float] = Field(..., description="Bezier control points")
    pruning_time: float = Field(..., description="Time spent in pruning")


class RetroactivePruningInfo(BaseModel):
    """Information about retroactive pruning."""

    attention_threshold: float = Field(
        ..., description="Attention threshold for pruning"
    )
    use_relative_attention: bool = Field(
        ..., description="Whether relative attention was used"
    )
    relative_threshold: float = Field(..., description="Relative attention threshold")
    use_multi_scale_attention: bool = Field(
        ..., description="Whether multi-scale attention was used"
    )
    num_layers_to_use: Optional[int] = Field(
        None, description="Number of layers used for attention"
    )
    use_lci_dynamic_threshold: bool = Field(
        ..., description="Whether LCI dynamic threshold was used"
    )
    use_sigmoid_threshold: bool = Field(
        ..., description="Whether sigmoid threshold was used"
    )
    sigmoid_steepness: float = Field(..., description="Sigmoid steepness parameter")
    pruning_mode: str = Field(..., description="Mode for handling pruned positions")


class Token(BaseModel):
    """Information about a token in the raw token data."""

    id: int = Field(..., description="Token ID")
    text: str = Field(..., description="Token text")
    probability: float = Field(..., description="Token probability")


class TokenSetData(BaseModel):
    """Raw token set data for visualization."""

    position: int = Field(..., description="Position in the sequence")
    original_tokens: List[Token] = Field(..., description="Original tokens considered")
    pruned_tokens: List[Token] = Field(
        ..., description="Tokens pruned from consideration"
    )


class GenerationResponse(BaseModel):
    """
    Response model for text generation with TEMPO.

    Contains the generated text, token information, and performance metrics.
    """

    # Core output
    generated_text: str = Field(
        ..., description="Complete generated text with formatting"
    )
    raw_generated_text: str = Field(
        ..., description="Raw generated text without formatting"
    )

    # Token-level data
    steps: List[StepInfo] = Field(
        default_factory=list, description="Information about each generation step"
    )
    position_to_tokens: Dict[str, List[str]] = Field(
        default_factory=dict, description="Mapping of positions to tokens"
    )
    original_parallel_positions: List[int] = Field(
        default_factory=list, description="Positions with parallel tokens"
    )

    # Performance and timing
    timing: TimingInfo = Field(..., description="Timing information for generation")

    # Pruning information
    pruning: Optional[PruningInfo] = Field(
        None, description="Information about pruning strategies"
    )
    retroactive_pruning: Optional[RetroactivePruningInfo] = Field(
        None, description="Information about retroactive pruning"
    )

    # Model information
    model_info: ModelInfo = Field(..., description="Information about the model used")

    # Generation settings
    selection_threshold: float = Field(
        ..., description="Threshold used for token selection"
    )
    max_tokens: int = Field(..., description="Maximum tokens generated")
    min_steps: int = Field(..., description="Minimum steps before considering EOS")
    prompt: str = Field(..., description="Input prompt")

    # Advanced fields
    had_repetition_loop: bool = Field(
        default=False, description="Whether repetition was detected"
    )
    system_content: Optional[str] = Field(None, description="System content used")

    # Raw data for visualization
    token_sets: List[TokenSetData] = Field(
        default_factory=list,
        description="Raw token sets data for visualization",
        exclude=True,  # Exclude from automatically generated OpenAPI schema
    )
    tokens_by_position: Dict[str, Any] = Field(
        default_factory=dict, description="Token information by position", exclude=True
    )
    final_pruned_sets: Dict[str, Any] = Field(
        default_factory=dict, description="Final pruned token sets", exclude=True
    )

    # For clients that need raw data, include separate TokenSetData objects
    raw_token_data: List[TokenSetData] = Field(
        default_factory=list, description="Raw token data for visualization"
    )

    class Config:
        schema_extra = {
            "example": {
                "generated_text": "Llamas and alpacas are both camelid species, but they have several key differences: llamas are larger, weighing 250-450 pounds compared to alpacas at 100-200 pounds.",
                "raw_generated_text": "Llamas and alpacas are both camelid species, but they have several key differences: llamas are larger, weighing 250-450 pounds compared to alpacas at 100-200 pounds.",
                "steps": [
                    {
                        "position": 0,
                        "parallel_tokens": [
                            {
                                "token_text": "Llamas",
                                "token_id": 123,
                                "probability": 0.7,
                            },
                            {"token_text": "The", "token_id": 456, "probability": 0.2},
                        ],
                        "pruned_tokens": [
                            {"token_text": "A", "token_id": 789, "probability": 0.1}
                        ],
                    }
                ],
                "timing": {
                    "generation_time": 0.456,
                    "pruning_time": 0.123,
                    "elapsed_time": 0.579,
                },
                "model_info": {
                    "model_name": "deepcogito/cogito-v1-preview-llama-3B",
                    "is_qwen_model": False,
                    "use_custom_rope": True,
                    "device": "cuda",
                    "model_type": "llama",
                },
                "selection_threshold": 0.1,
                "max_tokens": 50,
                "min_steps": 0,
                "prompt": "Explain the difference between a llama and an alpaca",
            }
        }

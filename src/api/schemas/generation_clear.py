"""
API schemas with crystal-clear naming for TEMPO generation.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class TokenInfo(BaseModel):
    """Information about a single token."""
    token_id: int
    token_text: str
    probability: float


class GenerationStepClear(BaseModel):
    """A single step in the generation process with clear naming."""
    position: int = Field(description="Position in the sequence")
    
    # All tokens that were considered at this position
    candidate_tokens: List[TokenInfo] = Field(
        description="All tokens considered at this position before filtering"
    )
    
    # Tokens that passed the selection criteria and continued to next step
    selected_tokens: List[TokenInfo] = Field(
        description="Tokens that were selected to continue generation"
    )
    
    # Tokens that were rejected/removed
    rejected_tokens: List[TokenInfo] = Field(
        description="Tokens that were rejected and did not continue"
    )
    
    # Helpful flags
    had_multiple_paths: bool = Field(
        description="Whether multiple tokens were selected at this position"
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Why tokens were rejected (e.g., 'below_threshold', 'insufficient_attention')"
    )


class GenerationResponseClear(BaseModel):
    """Clear generation response with unambiguous naming."""
    # The final generated text
    final_text: str = Field(description="The final generated text without any formatting")
    
    # Detailed generation process
    generation_steps: List[GenerationStepClear] = Field(
        description="Step-by-step details of the generation process"
    )
    
    # Summary statistics
    statistics: Dict[str, Any] = Field(
        description="Generation statistics",
        example={
            "total_tokens_generated": 50,
            "total_candidates_considered": 250,
            "positions_with_multiple_paths": 5,
            "average_candidates_per_position": 5.0,
            "generation_time_seconds": 1.23
        }
    )
    
    # Original request
    original_prompt: str
    settings_used: Dict[str, Any]
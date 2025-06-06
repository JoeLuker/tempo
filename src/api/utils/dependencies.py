"""
Dependency injection for API components.
"""
import logging
from typing import Optional
from functools import lru_cache

from src.api.services.generation_service import GenerationService
from src.utils.model_utils import load_tempo_components
from src.modeling.model_wrapper import TEMPOModelWrapper
from src.generation.parallel_generator import ParallelGenerator
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class ModelComponents:
    """Container for model components."""
    def __init__(self):
        self.model_wrapper: Optional[TEMPOModelWrapper] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.generator: Optional[ParallelGenerator] = None
        self.initialized: bool = False
    
    def initialize(self, model_name: str = "deepcogito/cogito-v1-preview-llama-3B", 
                  device: str = "auto"):
        """Initialize model components."""
        if self.initialized:
            return
            
        logger.info(f"Initializing model components for {model_name} on {device}")
        
        components = load_tempo_components(
            model_id=model_name,
            device=device,
            load_model_wrapper=True,
            load_token_generator=True,
            load_parallel_generator=True,
            debug_mode=False,
            use_fast_tokenizer=True,
            attn_implementation="eager"
        )
        
        self.model_wrapper = components["model_wrapper"]
        self.tokenizer = components["tokenizer"]
        self.generator = components["parallel_generator"]
        self.initialized = True
        
        logger.info("Model components initialized successfully")


# Global model components instance
_model_components = ModelComponents()


@lru_cache()
def get_model_components() -> ModelComponents:
    """Get or initialize model components."""
    if not _model_components.initialized:
        _model_components.initialize()
    return _model_components


def get_generation_service() -> GenerationService:
    """Get generation service with injected dependencies."""
    components = get_model_components()
    
    if not components.initialized:
        raise RuntimeError("Model components not initialized")
    
    return GenerationService(
        model_wrapper=components.model_wrapper,
        tokenizer=components.tokenizer,
        generator=components.generator
    )
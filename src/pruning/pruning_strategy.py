from abc import ABC, abstractmethod
import torch
from typing import List, Tuple, Optional

class PruningStrategy(ABC):
    """
    Abstract base class for implementing token pruning strategies.
    """
    
    def __init__(self, model, tokenizer, device: str = "mps"):
        """
        Initialize the pruning strategy.
        
        Args:
            model: The language model
            tokenizer: HuggingFace tokenizer
            device: Device to use for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    @abstractmethod
    def prune_tokens(
        self, 
        input_ids: torch.Tensor, 
        parallel_tokens: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Prune the parallel tokens based on the specific strategy.
        
        Args:
            input_ids: Current input token IDs
            parallel_tokens: List of (token_id, probability) tuples
            
        Returns:
            List[Tuple[int, float]]: Pruned list of (token_id, probability) tuples
        """
        pass
    
    def get_scored_tokens(
        self,
        input_ids: torch.Tensor, 
        parallel_tokens: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Return tokens with their normalized scores.
        This is used by dynamic threshold approaches.
        
        Args:
            input_ids: Current input token IDs
            parallel_tokens: List of (token_id, probability) tuples
            
        Returns:
            List[Tuple[int, float]]: List of (token_id, normalized_score) tuples
        """
        # Default implementation just returns tokens with score 1.0
        # Subclasses should override this for meaningful scoring
        return [(token_id, 1.0) for token_id, _ in parallel_tokens] 
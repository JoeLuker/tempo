"""Strategy interfaces for generation approaches in TEMPO.

This module defines the strategy pattern interfaces for different
generation approaches (standard, MCTS, etc.).
"""

from typing import Protocol, Optional, Any
from abc import abstractmethod

from ..entities.token import TokenSet
from ..entities.logits import TokenLogits
from ..entities.generation_state import GenerationState
from ..entities.parallel_generation import GenerationConfig, MCTSNode


class GenerationStrategy(Protocol):
    """Interface for generation strategies."""
    
    @abstractmethod
    def select_tokens(
        self,
        logits: TokenLogits,
        step: int,
        config: GenerationConfig,
        state: GenerationState
    ) -> TokenSet:
        """Select tokens based on the strategy.
        
        Args:
            logits: Raw logits from the model
            step: Current generation step
            config: Generation configuration
            state: Current generation state
            
        Returns:
            TokenSet containing selected tokens
        """
        ...
    
    @abstractmethod
    def should_terminate(
        self,
        token_set: TokenSet,
        state: GenerationState
    ) -> bool:
        """Check if generation should terminate.
        
        Args:
            token_set: Most recently generated token set
            state: Current generation state
            
        Returns:
            True if generation should stop
        """
        ...


class ThresholdStrategy(Protocol):
    """Interface for threshold calculation strategies."""
    
    @abstractmethod
    def calculate_threshold(self, step: int, max_steps: int) -> float:
        """Calculate the threshold for a given step.
        
        Args:
            step: Current generation step
            max_steps: Maximum number of steps
            
        Returns:
            Threshold value between 0 and 1
        """
        ...


class MCTSStrategy(Protocol):
    """Interface for MCTS-based generation strategies."""
    
    @abstractmethod
    def initialize_tree(self, root_logits: TokenLogits) -> MCTSNode:
        """Initialize MCTS tree from root logits.
        
        Args:
            root_logits: Logits at the root position
            
        Returns:
            Root node of the MCTS tree
        """
        ...
    
    @abstractmethod
    def select_node(self, root: MCTSNode, c_puct: float) -> MCTSNode:
        """Select a node for expansion using tree policy.
        
        Args:
            root: Root node of the tree
            c_puct: Exploration constant
            
        Returns:
            Selected node for expansion
        """
        ...
    
    @abstractmethod
    def expand_node(self, node: MCTSNode, logits: TokenLogits) -> list[MCTSNode]:
        """Expand a node by adding children.
        
        Args:
            node: Node to expand
            logits: Logits at this position
            
        Returns:
            List of child nodes added
        """
        ...
    
    @abstractmethod
    def simulate(
        self,
        node: MCTSNode,
        state: GenerationState,
        depth: int
    ) -> float:
        """Run simulation from a node to estimate value.
        
        Args:
            node: Starting node for simulation
            state: Current generation state
            depth: Maximum simulation depth
            
        Returns:
            Estimated value from simulation
        """
        ...
    
    @abstractmethod
    def backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate value up the tree.
        
        Args:
            node: Leaf node to start backpropagation
            value: Value to propagate
        """
        ...
    
    @abstractmethod
    def get_best_tokens(
        self,
        root: MCTSNode,
        threshold: float
    ) -> list[tuple[int, float]]:
        """Get best tokens from MCTS tree.
        
        Args:
            root: Root node of the tree
            threshold: Threshold for token selection
            
        Returns:
            List of (token_id, probability) tuples
        """
        ...


class PruningStrategy(Protocol):
    """Interface for token pruning strategies."""
    
    @abstractmethod
    def should_prune_token(
        self,
        token_id: int,
        attention_score: float,
        step: int,
        threshold: float
    ) -> bool:
        """Determine if a token should be pruned.
        
        Args:
            token_id: Token to consider for pruning
            attention_score: Attention score for the token
            step: Current generation step
            threshold: Pruning threshold
            
        Returns:
            True if token should be pruned
        """
        ...
    
    @abstractmethod
    def prune_token_set(
        self,
        token_set: TokenSet,
        attention_scores: list[float],
        step: int
    ) -> TokenSet:
        """Prune a token set based on attention scores.
        
        Args:
            token_set: Token set to prune
            attention_scores: Attention scores for each token
            step: Current generation step
            
        Returns:
            Pruned token set
        """
        ...

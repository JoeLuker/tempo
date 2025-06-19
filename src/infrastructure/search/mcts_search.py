"""MCTS implementation for token search.

This module implements the Monte Carlo Tree Search algorithm
for exploring token selection strategies.
"""

import math
import torch
from typing import Optional
import numpy as np

from ...domain.entities.parallel_generation import MCTSNode
from ...domain.entities.token import Token
from ...domain.entities.logits import TokenLogits
from ...domain.entities.generation_state import GenerationState
from ...domain.interfaces.generation_strategy import MCTSStrategy
from ...domain.interfaces.token_generation import TokenGeneratorInterface
from ...utils.logging_utils import LoggingMixin


class MCTSSearchStrategy(MCTSStrategy, LoggingMixin):
    """MCTS implementation for token generation."""
    
    def __init__(
        self,
        token_generator: TokenGeneratorInterface,
        top_k: int = 50,
        temperature: float = 1.0,
        debug_mode: bool = False
    ):
        """Initialize MCTS search strategy.
        
        Args:
            token_generator: Interface for generating token logits
            top_k: Number of top tokens to consider
            temperature: Temperature for sampling
            debug_mode: Whether to enable debug logging
        """
        super().__init__()
        self.setup_logging("mcts_search", "mcts_search.log", debug_mode)
        
        self.token_generator = token_generator
        self.top_k = top_k
        self.temperature = temperature
    
    def initialize_tree(self, root_logits: TokenLogits) -> MCTSNode:
        """Initialize MCTS tree from root logits.
        
        Args:
            root_logits: Logits at the root position
            
        Returns:
            Root node of the MCTS tree
        """
        # Create root node with dummy values
        root = MCTSNode(
            token_id=-1,  # Root has no token
            probability=1.0,
            value=0.0,
            visits=1  # Root is considered visited
        )
        
        # Get top-k tokens from logits
        probs = torch.softmax(root_logits.logits / self.temperature, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs[0], min(self.top_k, probs.size(-1)))
        
        # Create child nodes for top-k tokens
        for idx, prob in zip(top_k_indices.tolist(), top_k_probs.tolist()):
            child = MCTSNode(
                token_id=idx,
                probability=prob
            )
            root.add_child(child)
        
        self.log(f"Initialized MCTS tree with {len(root.children)} children")
        return root
    
    def select_node(self, root: MCTSNode, c_puct: float) -> MCTSNode:
        """Select a node for expansion using tree policy.
        
        Args:
            root: Root node of the tree
            c_puct: Exploration constant
            
        Returns:
            Selected node for expansion
        """
        current = root
        
        # Traverse down the tree using UCB1
        while current.children:
            # Check if node is fully expanded
            unvisited = [c for c in current.children if c.visits == 0]
            
            if unvisited:
                # Select random unvisited child
                return np.random.choice(unvisited)
            
            # All children visited, select by UCB1
            best_value = -float('inf')
            best_child = None
            
            for child in current.children:
                ucb_value = self._calculate_ucb1(child, c_puct)
                if ucb_value > best_value:
                    best_value = ucb_value
                    best_child = child
            
            if best_child is None:
                break
                
            current = best_child
        
        return current
    
    def expand_node(self, node: MCTSNode, logits: TokenLogits) -> list[MCTSNode]:
        """Expand a node by adding children.
        
        Args:
            node: Node to expand
            logits: Logits at this position
            
        Returns:
            List of child nodes added
        """
        if node.children:
            # Already expanded
            return node.children
        
        # Get top-k tokens from logits
        probs = torch.softmax(logits.logits / self.temperature, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs[0], min(self.top_k, probs.size(-1)))
        
        # Create child nodes
        children = []
        for idx, prob in zip(top_k_indices.tolist(), top_k_probs.tolist()):
            child = MCTSNode(
                token_id=idx,
                probability=prob
            )
            node.add_child(child)
            children.append(child)
        
        self.log(f"Expanded node {node.token_id} with {len(children)} children")
        return children
    
    def simulate(self, node: MCTSNode, state: GenerationState, depth: int) -> float:
        """Run simulation from a node to estimate value.
        
        Args:
            node: Starting node for simulation
            state: Current generation state
            depth: Maximum simulation depth
            
        Returns:
            Estimated value from simulation
        """
        # Simple rollout policy: sample tokens based on probabilities
        current_state = state
        total_log_prob = 0.0
        
        # Start from the node's token
        if node.token_id >= 0:
            total_log_prob += math.log(node.probability + 1e-8)
        
        # Simulate forward
        for _ in range(depth):
            # Generate logits
            logits = self.token_generator.generate_logits(current_state)
            
            # Sample token
            probs = torch.softmax(logits.logits / self.temperature, dim=-1)
            token_id = torch.multinomial(probs[0], 1).item()
            token_prob = probs[0, token_id].item()
            
            # Update log probability
            total_log_prob += math.log(token_prob + 1e-8)
            
            # Update state
            current_state = current_state.add_token(token_id)
            
            # Check for EOS
            # if token_id == eos_token_id:
            #     break
        
        # Convert log probability to value (higher is better)
        # Normalize by depth to avoid favoring shorter sequences
        value = total_log_prob / (depth + 1)
        
        return value
    
    def backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate value up the tree.
        
        Args:
            node: Leaf node to start backpropagation
            value: Value to propagate
        """
        current = node
        
        while current is not None:
            current.update(value)
            current = current.parent
    
    def get_best_tokens(self, root: MCTSNode, threshold: float) -> list[tuple[int, float]]:
        """Get best tokens from MCTS tree.
        
        Args:
            root: Root node of the tree
            threshold: Threshold for token selection
            
        Returns:
            List of (token_id, probability) tuples
        """
        # Calculate selection probabilities based on visit counts
        total_visits = sum(child.visits for child in root.children)
        
        if total_visits == 0:
            # No simulations run, use original probabilities
            return [
                (child.token_id, child.probability)
                for child in root.children
                if child.probability >= threshold
            ]
        
        # Calculate selection probabilities
        selected_tokens = []
        
        for child in root.children:
            if child.visits == 0:
                continue
                
            # Selection probability based on visit count
            selection_prob = child.visits / total_visits
            
            # Weight by original probability
            weighted_prob = selection_prob * child.probability
            
            if weighted_prob >= threshold:
                selected_tokens.append((child.token_id, weighted_prob))
        
        # Sort by probability
        selected_tokens.sort(key=lambda x: x[1], reverse=True)
        
        self.log(f"Selected {len(selected_tokens)} tokens from MCTS tree")
        return selected_tokens
    
    def _calculate_ucb1(self, node: MCTSNode, c_puct: float) -> float:
        """Calculate UCB1 value for a node.
        
        Args:
            node: Node to calculate UCB1 for
            c_puct: Exploration constant
            
        Returns:
            UCB1 value
        """
        if node.visits == 0:
            return float('inf')
        
        if node.parent is None:
            return 0.0
        
        exploitation = node.value / node.visits
        exploration = c_puct * math.sqrt(math.log(node.parent.visits) / node.visits)
        
        # Also consider the original probability
        prior_weight = c_puct * node.probability / (1 + node.visits)
        
        return exploitation + exploration + prior_weight
    
    def get_tree_statistics(self, root: MCTSNode) -> dict[str, any]:
        """Get statistics about the MCTS tree.
        
        Args:
            root: Root node of the tree
            
        Returns:
            Dictionary of tree statistics
        """
        def count_nodes(node):
            count = 1
            for child in node.children:
                count += count_nodes(child)
            return count
        
        def max_depth(node, current_depth=0):
            if not node.children:
                return current_depth
            return max(max_depth(child, current_depth + 1) for child in node.children)
        
        total_nodes = count_nodes(root) - 1  # Exclude root
        tree_depth = max_depth(root)
        
        # Visit distribution
        visit_counts = [child.visits for child in root.children]
        
        return {
            "total_nodes": total_nodes,
            "tree_depth": tree_depth,
            "root_children": len(root.children),
            "total_visits": sum(visit_counts),
            "max_visits": max(visit_counts) if visit_counts else 0,
            "min_visits": min(visit_counts) if visit_counts else 0,
            "avg_visits": sum(visit_counts) / len(visit_counts) if visit_counts else 0
        }

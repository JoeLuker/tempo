"""MCTS-based text generation use case.

This module implements Monte Carlo Tree Search for token selection
in the parallel generation process.
"""

import torch
from typing import Optional, Any

from ...domain.entities.parallel_generation import MCTSNode, MCTSState, GenerationConfig
from ...domain.entities.token import Token
from ...domain.entities.logits import TokenLogits
from ...domain.entities.generation_state import GenerationState
from ...domain.interfaces.token_generation import TokenGeneratorInterface
from ...domain.interfaces.generation_strategy import MCTSStrategy
from ...utils.logging_utils import LoggingMixin


class MCTSGenerationUseCase(LoggingMixin):
    """Use case for MCTS-based token generation."""
    
    def __init__(
        self,
        token_generator: TokenGeneratorInterface,
        debug_mode: bool = False
    ):
        """Initialize the MCTS generation use case.
        
        Args:
            token_generator: Interface for generating token logits
            debug_mode: Whether to enable debug logging
        """
        super().__init__()
        self.setup_logging("mcts_generation", "mcts.log", debug_mode)
        self.token_generator = token_generator
    
    def generate_with_mcts(
        self,
        state: GenerationState,
        config: GenerationConfig,
        strategy: MCTSStrategy
    ) -> list[tuple[int, float]]:
        """Generate tokens using MCTS.
        
        Args:
            state: Current generation state
            config: Generation configuration with MCTS parameters
            strategy: MCTS strategy implementation
            
        Returns:
            List of (token_id, probability) tuples selected by MCTS
        """
        self.log(f"Starting MCTS with {config.mcts_simulations} simulations")
        
        # Generate initial logits
        logits = self.token_generator.generate_logits(state)
        
        # Initialize MCTS tree
        root = strategy.initialize_tree(logits)
        
        # Run MCTS simulations
        for sim in range(config.mcts_simulations):
            # 1. Selection - traverse tree to find node to expand
            node = strategy.select_node(root, config.mcts_c_puct)
            
            # 2. Expansion - add children if not fully expanded
            if not node.children and node.visits > 0:
                # Generate logits for this node's state
                node_state = self._build_state_for_node(state, node)
                node_logits = self.token_generator.generate_logits(node_state)
                children = strategy.expand_node(node, node_logits)
                
                # Select a child for simulation
                if children:
                    node = children[0]  # Could use various selection strategies
            
            # 3. Simulation - run rollout from node
            sim_state = self._build_state_for_node(state, node)
            value = strategy.simulate(node, sim_state, config.mcts_depth)
            
            # 4. Backpropagation - update statistics
            strategy.backpropagate(node, value)
        
        # Extract best tokens based on visit counts
        best_tokens = strategy.get_best_tokens(root, config.selection_threshold)
        
        self.log(f"MCTS selected {len(best_tokens)} tokens")
        return best_tokens
    
    def run_mcts_simulation(
        self,
        initial_state: MCTSState,
        depth: int,
        c_puct: float,
        threshold: float
    ) -> list[tuple[int, float]]:
        """Run a single MCTS simulation.
        
        Args:
            initial_state: Initial MCTS state
            depth: Maximum simulation depth
            c_puct: Exploration constant
            threshold: Selection threshold
            
        Returns:
            List of (token_id, probability) tuples from simulation
        """
        current_state = initial_state.copy()
        simulation_tokens = []
        
        for d in range(depth):
            # Generate logits for current state
            gen_state = GenerationState(
                input_ids=current_state.input_ids,
                attention_mask=current_state.attention_mask,
                past_key_values=current_state.past_key_values
            )
            
            logits, new_gen_state = self.token_generator.generate_logits_with_cache(gen_state)
            
            # Update KV cache
            current_state.past_key_values = new_gen_state.past_key_values
            
            # Calculate token probabilities
            token_probs = torch.softmax(logits.logits, dim=-1)
            
            # Select token using UCB1
            token_values = self._calculate_ucb1_values(
                token_probs,
                current_state.logical_step,
                c_puct
            )
            
            # Select best token
            selected_idx = torch.argmax(token_values).item()
            selected_prob = token_probs[0, selected_idx].item()
            
            simulation_tokens.append((selected_idx, selected_prob))
            
            # Update state
            new_token = torch.tensor([[selected_idx]], device=current_state.input_ids.device)
            current_state.input_ids = torch.cat([current_state.input_ids, new_token], dim=1)
            current_state.attention_mask = torch.cat([
                current_state.attention_mask,
                torch.ones((1, 1), device=current_state.attention_mask.device)
            ], dim=1)
            current_state.logical_step += 1
            current_state.depth = d + 1
        
        return simulation_tokens
    
    def aggregate_mcts_results(
        self,
        all_simulations: list[list[tuple[int, float]]],
        threshold: float
    ) -> list[tuple[int, float]]:
        """Aggregate results from multiple MCTS simulations.
        
        Args:
            all_simulations: Results from all simulations
            threshold: Selection threshold
            
        Returns:
            Aggregated list of (token_id, probability) tuples
        """
        # Count occurrences and average probabilities
        token_stats: dict[int, list[float]] = {}
        
        for simulation in all_simulations:
            for token_id, prob in simulation:
                if token_id not in token_stats:
                    token_stats[token_id] = []
                token_stats[token_id].append(prob)
        
        # Calculate average probabilities
        token_avg_probs = {
            token_id: sum(probs) / len(probs)
            for token_id, probs in token_stats.items()
        }
        
        # Filter by threshold
        selected_tokens = [
            (token_id, avg_prob)
            for token_id, avg_prob in token_avg_probs.items()
            if avg_prob >= threshold
        ]
        
        # Sort by probability
        selected_tokens.sort(key=lambda x: x[1], reverse=True)
        
        return selected_tokens
    
    def _build_state_for_node(self, base_state: GenerationState, node: MCTSNode) -> GenerationState:
        """Build generation state for a specific MCTS node."""
        # This would traverse from node to root to build the full path
        # For now, simplified version
        path_tokens = []
        current = node
        
        while current.parent is not None:
            path_tokens.append(current.token_id)
            current = current.parent
        
        path_tokens.reverse()
        
        # Add path tokens to base state
        if path_tokens:
            new_tokens = torch.tensor([path_tokens], device=base_state.input_ids.device)
            new_input_ids = torch.cat([base_state.input_ids, new_tokens], dim=1)
            new_attention_mask = torch.cat([
                base_state.attention_mask,
                torch.ones((1, len(path_tokens)), device=base_state.attention_mask.device)
            ], dim=1)
            
            return GenerationState(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                past_key_values=base_state.past_key_values
            )
        
        return base_state
    
    def _calculate_ucb1_values(
        self,
        probabilities: torch.Tensor,
        total_visits: int,
        c_puct: float
    ) -> torch.Tensor:
        """Calculate UCB1 values for token selection."""
        values = torch.zeros_like(probabilities)
        
        for i in range(probabilities.size(-1)):
            if probabilities[0, i] > 0:
                # UCB1: Q + c * sqrt(ln(N)/n)
                # Using probability as Q value
                exploration = c_puct * torch.sqrt(
                    torch.log(torch.tensor(total_visits + 1)) / (probabilities[0, i] + 1e-8)
                )
                values[0, i] = probabilities[0, i] + exploration
        
        return values

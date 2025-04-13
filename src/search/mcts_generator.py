import torch
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import time

class MCTSNode:
    """Node in the MCTS search tree."""
    
    def __init__(self, parent=None, token_id=None, prob=0.0, state=None):
        """
        Initialize a node in the MCTS tree.
        
        Args:
            parent: Parent node
            token_id: Token ID for this node
            prob: Prior probability from the policy network
            state: State representation (sequence of token IDs)
        """
        self.parent = parent
        self.token_id = token_id
        self.prob = prob
        self.state = state if state is not None else []
        
        self.children = {}  # token_id -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.attention_scores = {}  # position -> attention score
        
    def value(self):
        """Get the mean value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
        
    def expanded(self):
        """Check if this node has been expanded."""
        return len(self.children) > 0
        
    def add_child(self, token_id, prob, state):
        """Add a child node."""
        self.children[token_id] = MCTSNode(
            parent=self,
            token_id=token_id,
            prob=prob,
            state=state
        )
        return self.children[token_id]
        
    def update(self, value):
        """Update the node statistics."""
        self.visit_count += 1
        self.value_sum += value
        
    def ucb_score(self, parent_visit_count, c_puct):
        """Calculate the UCB score for this node."""
        if self.visit_count == 0:
            return float('inf')
            
        # Exploration term
        exploration = c_puct * self.prob * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        
        # Exploitation term
        exploitation = self.value()
        
        return exploitation + exploration


class MCTSGenerator:
    """
    Text generator using Monte Carlo Tree Search integrated with TEMPO.
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        token_generator,
        retroactive_pruner=None,
        c_puct: float = 1.0,
        num_simulations: int = 10,
        attention_threshold: float = 0.01,
        device: str = "cuda",
        debug_mode: bool = False
    ):
        """
        Initialize the MCTS generator.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            token_generator: TokenGenerator instance for generating logits
            retroactive_pruner: RetroactivePruner instance for pruning based on attention
            c_puct: Exploration constant for UCB formula
            num_simulations: Number of MCTS simulations per step
            attention_threshold: Threshold for attention-based pruning
            device: Device to use for computations
            debug_mode: Whether to enable debug output
        """
        self.model = model
        self.tokenizer = tokenizer
        self.token_generator = token_generator
        self.retroactive_pruner = retroactive_pruner
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.attention_threshold = attention_threshold
        self.device = device
        self.debug_mode = debug_mode
        
        # Root node of the search tree
        self.root = None
        
        # Dictionary of all nodes (for efficient lookup)
        self.nodes = {}
        
        # For tracking the search process
        self.search_path = []
        
    def reset(self):
        """Reset the search tree."""
        self.root = None
        self.nodes = {}
        self.search_path = []
        
    def _state_to_key(self, state):
        """Convert a state (list of token IDs) to a hashable key."""
        return tuple(state)
        
    def _select_node(self, node):
        """
        Select a node to expand using the UCB formula.
        Uses the PUCT algorithm from AlphaZero.
        """
        if not node.expanded():
            return node
            
        # Select child with highest UCB score
        max_ucb = float('-inf')
        best_child = None
        
        for child in node.children.values():
            ucb = child.ucb_score(node.visit_count, self.c_puct)
            if ucb > max_ucb:
                max_ucb = ucb
                best_child = child
                
        # If no children above threshold, return this node
        if best_child is None:
            return node
            
        # Continue selection from the best child
        return self._select_node(best_child)
        
    def _expand_node(self, node, input_ids):
        """Expand a node by adding all possible next tokens."""
        if node.expanded():
            return
            
        # Get logits for the current state
        next_token_logits, _ = self.token_generator.get_next_token_logits_cached(
            input_ids, 
            torch.ones_like(input_ids),
            None  # No past key values for simplicity in MCTS
        )
        
        # Get top-k tokens and probabilities
        top_k = 5  # Adjust based on desired branching factor
        probs = torch.softmax(next_token_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        
        # Add children for each top token
        for i in range(top_k):
            token_id = top_indices[0, i].item()
            prob = top_probs[0, i].item()
            
            # Create the new state
            new_state = node.state + [token_id]
            
            # Add child to the node
            node.add_child(token_id, prob, new_state)
            
    def _simulate(self, node, input_ids, max_steps=5):
        """Run a simulation from this node to evaluate it."""
        # Start with the current node's state
        sim_state = node.state.copy()
        sim_input_ids = input_ids.clone()
        
        # Run simulation for a few steps
        for _ in range(max_steps):
            # Get next token probabilities
            next_token_logits, _ = self.token_generator.get_next_token_logits_cached(
                sim_input_ids, 
                torch.ones_like(sim_input_ids),
                None  # No past key values for simplicity in MCTS
            )
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs[0], 1).item()
            
            # Update state
            sim_state.append(next_token)
            
            # Update input_ids for next iteration
            next_token_tensor = torch.tensor([[next_token]], device=sim_input_ids.device)
            sim_input_ids = torch.cat([sim_input_ids, next_token_tensor], dim=1)
            
        # Evaluate the final state
        # For text generation, we could use:
        # 1. Model confidence (avg token probability)
        # 2. Coherence metrics
        # 3. External reward model
        
        # For now, use a simplified heuristic: average token probability
        value = torch.mean(probs[0]).item()
        
        return value
        
    def _backpropagate(self, node, value):
        """Backpropagate the value through the search path."""
        while node is not None:
            node.update(value)
            node = node.parent
            
    def _retroactively_prune(self, node, attention_mask):
        """Prune the tree based on attention patterns."""
        if self.retroactive_pruner is None or not node.expanded():
            return
            
        # Get cached attention from most recent token
        cached_attention, _ = self.token_generator.get_cached_attention()
        if cached_attention is None:
            return
            
        # Use retroactive pruner logic to decide which children to prune
        for token_id, child in list(node.children.items()):
            position = len(node.state)
            
            # Calculate attention score for this token/position
            attention_score = self._calculate_attention_score(cached_attention, position)
            
            # Store attention score
            node.attention_scores[position] = attention_score
            
            # Prune if below threshold
            if attention_score < self.attention_threshold:
                # Only prune if there are multiple children
                if len(node.children) > 1:
                    del node.children[token_id]
                    
    def _calculate_attention_score(self, cached_attention, position):
        """Calculate attention score for a specific position."""
        # Extract layers
        layers_to_use = min(3, len(cached_attention))
        attention_layers = cached_attention[-layers_to_use:]
        
        # Average attention across layers
        avg_layer_attention = torch.mean(torch.stack([layer for layer in attention_layers]), dim=0)
        
        # Extract attention for last token
        try:
            last_token_attn = avg_layer_attention[0, :, -1, :-1]  # [num_heads, seq_len-1]
            avg_attention = last_token_attn.mean(dim=0)  # [seq_len-1]
            
            # Normalize attention
            normalized_attn = avg_attention / (torch.sum(avg_attention) + 1e-10)
            
            # Get attention score for the position
            if position < len(normalized_attn):
                return normalized_attn[position].item()
        except Exception as e:
            if self.debug_mode:
                print(f"Error extracting attention: {e}")
                
        return 0.0
        
    def search(self, input_ids):
        """Run MCTS search to find the best next token."""
        # Initialize root node if needed
        if self.root is None:
            initial_state = input_ids[0].tolist()
            self.root = MCTSNode(state=initial_state)
            
        # Run simulations
        for _ in range(self.num_simulations):
            # Phase 1: Selection
            selected_node = self._select_node(self.root)
            
            # Phase 2: Expansion
            if not selected_node.expanded():
                # Create input_ids from the selected node's state
                selected_input_ids = torch.tensor([selected_node.state], device=self.device)
                self._expand_node(selected_node, selected_input_ids)
                
            # Phase 3: Simulation
            # Choose a child to simulate from
            if selected_node.expanded():
                # Randomly select a child for simulation
                child_id = np.random.choice(list(selected_node.children.keys()))
                child = selected_node.children[child_id]
                
                # Create input_ids for the child
                child_input_ids = torch.tensor([child.state], device=self.device)
                value = self._simulate(child, child_input_ids)
            else:
                # Simulate from the selected node
                selected_input_ids = torch.tensor([selected_node.state], device=self.device)
                value = self._simulate(selected_node, selected_input_ids)
                
            # Phase 4: Backpropagation
            self._backpropagate(selected_node, value)
            
        # Retroactively prune the tree using attention patterns
        self._retroactively_prune(self.root, None)
        
        # Select best child based on visit count
        best_child = None
        best_visit_count = -1
        
        for child in self.root.children.values():
            if child.visit_count > best_visit_count:
                best_visit_count = child.visit_count
                best_child = child
                
        # Return the best token ID
        if best_child is not None:
            return best_child.token_id
        
        # Fallback to sampling if no children
        next_token_logits, _ = self.token_generator.get_next_token_logits_cached(
            input_ids, 
            torch.ones_like(input_ids),
            None
        )
        probs = torch.softmax(next_token_logits, dim=-1)
        return torch.multinomial(probs[0], 1).item()
        
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text using MCTS.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated text
        """
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids)
        
        # Reset MCTS
        self.reset()
        
        # Initialize root node with prompt tokens
        initial_state = input_ids[0].tolist()
        self.root = MCTSNode(state=initial_state)
        
        # Generate tokens
        all_tokens = initial_state.copy()
        
        # Set up progress bar
        progress_bar = tqdm(range(max_tokens), desc="Generating with MCTS", unit="token")
        
        for _ in progress_bar:
            # Run MCTS search
            next_token = self.search(input_ids)
            
            # Add the new token
            all_tokens.append(next_token)
            
            # Update input_ids for next iteration
            next_token_tensor = torch.tensor([[next_token]], device=self.device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
            
            # Update progress
            progress_bar.set_postfix(token=next_token)
            
            # Update root for next iteration
            if next_token in self.root.children:
                self.root = self.root.children[next_token]
                self.root.parent = None  # Detach from parent to save memory
            else:
                # Create new root
                self.root = MCTSNode(state=all_tokens)
                
        # Decode all tokens
        generated_text = self.tokenizer.decode(all_tokens, skip_special_tokens=True)
        
        return generated_text 
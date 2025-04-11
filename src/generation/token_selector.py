import torch
import logging
import os
from typing import List, Tuple, Optional, Any, Set

class TokenSelector:
    """
    Responsible for selecting tokens above threshold and related operations.
    Optimized for tensor operations and special cases handling.
    """
    
    def __init__(self, tokenizer):
        """
        Initialize the token selector.
        
        Args:
            tokenizer: HuggingFace tokenizer
        """
        self.tokenizer = tokenizer
        
        # Cache EOS token ID since we check it frequently
        # Invariant: Tokenizer must have an eos_token_id
        if not hasattr(tokenizer, "eos_token_id"):
            raise ValueError("Tokenizer must have an eos_token_id")
        self.eos_token_id = tokenizer.eos_token_id
        
        # Setup logging
        self._setup_logger()
        
        # Debug mode
        self.debug_mode = False
    
    def _setup_logger(self):
        """Setup logging to file."""
        # Ensure logs directory exists
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logger
        self.logger = logging.getLogger("token_selector")
        self.logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers to avoid duplicate logs
        if self.logger.handlers:
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)
        
        # Create file handler
        log_file = os.path.join(log_dir, "token_selector_debug.log")
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
    def log(self, message, level="info"):
        """
        Log a message to the log file if debug mode is enabled.
        
        Args:
            message: Message to log
            level: Log level (info, debug, warning, error)
        """
        if not self.debug_mode:
            return
            
        if level == "info":
            self.logger.info(message)
        elif level == "debug":
            self.logger.debug(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
            
    def set_debug_mode(self, enabled: bool = True):
        """
        Enable or disable debug mode for more verbose output.
        
        Args:
            enabled: Whether to enable debug mode
        """
        self.debug_mode = enabled
        if enabled:
            print(f"TokenSelector debug mode enabled - logging to file at logs/token_selector_debug.log")
        else:
            print(f"TokenSelector debug mode disabled")
    
    def select_tokens_above_threshold(
        self, 
        next_token_logits: torch.Tensor, 
        threshold: float,
        max_tokens: int = 25
    ) -> Tuple[List[int], List[float]]:
        """
        Select tokens with probabilities above the threshold.
        Optimized for performance with batched tensor operations.
        
        Args:
            next_token_logits: Logits tensor for next token [batch_size, vocab_size]
            threshold: Probability threshold
            max_tokens: Maximum number of tokens to return
            
        Returns:
            tuple: (token_ids, token_probs)
        """
        # DIAGNOSTIC: Show logits shape and values
        is_first_token = next_token_logits.size(0) == 1 and hasattr(self, '_first_token_processed') == False
        if is_first_token:
            self._first_token_processed = True
            self.log(f"\nDIAGNOSTIC - First token logits:")
            self.log(f"  Logits shape: {next_token_logits.shape}")
            self.log(f"  Logits min: {next_token_logits.min().item():.4f}, max: {next_token_logits.max().item():.4f}")
            self.log(f"  Threshold: {threshold}")
            
        # Apply softmax to logits efficiently using torch.softmax
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        
        # Invariant: Probabilities must sum to approximately 1.0 after softmax
        prob_sum = next_token_probs.sum().item()
        if not (0.99 <= prob_sum <= 1.01):
            raise ValueError(f"Invariant violation: Token probabilities sum to {prob_sum}, not approximately 1.0 after softmax")
        
        # DIAGNOSTIC: Show probability distribution
        if is_first_token:
            self.log(f"\nDIAGNOSTIC - Token probability distribution:")
            self.log(f"  Probability sum: {prob_sum:.6f}")
            top_k = 10
            top_probs, top_indices = torch.topk(next_token_probs.squeeze(), top_k)
            self.log(f"  Top {top_k} token probabilities:")
            
            # Get token strings for top 10
            top_tokens = []
            for i in range(top_k):
                token_id = top_indices[i].item()
                token_text = self.tokenizer.decode([token_id])
                top_tokens.append(token_text)
                
            for i, (token_text, token_id, prob) in enumerate(zip(top_tokens, top_indices.tolist(), top_probs.tolist())):
                self.log(f"    {i+1}. '{token_text}' (ID: {token_id}): {prob:.6f}")
                
            # Also show sum of top 100 probabilities
            top_100_probs, _ = torch.topk(next_token_probs.squeeze(), 100)
            self.log(f"  Sum of top 100 token probabilities: {top_100_probs.sum().item():.6f}")
        
        # Convert to numpy or CPU tensors only when needed
        next_token_probs = next_token_probs.squeeze().cpu()
        
        # Find all probabilities above threshold efficiently
        indices_above_threshold = torch.nonzero(next_token_probs >= threshold).squeeze(-1)
        
        # DIAGNOSTIC: Show number of tokens above threshold
        if is_first_token:
            self.log(f"\nDIAGNOSTIC - Found {indices_above_threshold.numel()} tokens above threshold {threshold}")
        
        # If no tokens are above threshold, return empty lists
        if indices_above_threshold.numel() == 0:
            if is_first_token:
                self.log("DIAGNOSTIC - No tokens above threshold! Returning empty list.")
            return [], []
        
        # Sort by probability (highest first)
        indices_and_probs = [
            (idx.item(), next_token_probs[idx].item()) 
            for idx in indices_above_threshold
        ]
        sorted_indices_and_probs = sorted(indices_and_probs, key=lambda x: x[1], reverse=True)
        
        # Limit to max_tokens
        sorted_indices_and_probs = sorted_indices_and_probs[:max_tokens]
        
        # Extract token IDs and probabilities
        token_ids = [idx for idx, _ in sorted_indices_and_probs]
        token_probs = [prob for _, prob in sorted_indices_and_probs]
        
        # DIAGNOSTIC: Show selected tokens
        if is_first_token:
            self.log(f"\nDIAGNOSTIC - Selected {len(token_ids)} tokens above threshold {threshold}:")
            token_info = []
            for i, (tid, prob) in enumerate(zip(token_ids, token_probs)):
                token_text = self.tokenizer.decode([tid])
                token_info.append(f"    {i+1}. '{token_text}' (ID: {tid}): {prob:.6f}")
            self.log("\n".join(token_info))
            
        return token_ids, token_probs
    
    def select_tokens_above_threshold_excluding(
        self, 
        next_token_logits: torch.Tensor, 
        threshold: float,
        exclude_tokens: List[int],
        max_tokens: int = 25
    ) -> Tuple[List[int], List[float]]:
        """
        Select tokens above threshold while excluding specific tokens.
        Useful for avoiding repetition loops.
        
        Args:
            next_token_logits: Logits tensor for next token [batch_size, vocab_size]
            threshold: Probability threshold
            exclude_tokens: List of token IDs to exclude
            max_tokens: Maximum number of tokens to return
            
        Returns:
            tuple: (token_ids, token_probs)
        """
        # Apply softmax to logits efficiently
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        
        # Invariant: Probabilities must sum to approximately 1.0 after softmax
        prob_sum = next_token_probs.sum().item()
        if not (0.99 <= prob_sum <= 1.01):
            raise ValueError(f"Invariant violation: Token probabilities sum to {prob_sum}, not approximately 1.0 after softmax")
        
        # Convert to CPU tensors 
        next_token_probs = next_token_probs.squeeze().cpu()
        
        # Create a mask for excluded tokens
        if exclude_tokens:
            exclude_mask = torch.ones_like(next_token_probs, dtype=torch.bool)
            exclude_mask[exclude_tokens] = False
            # Apply the mask to probabilities
            masked_probs = next_token_probs * exclude_mask
        else:
            # No exclusions
            masked_probs = next_token_probs
        
        # Find all probabilities above threshold
        indices_above_threshold = torch.nonzero(masked_probs >= threshold).squeeze(-1)
        
        # If no tokens are above threshold, return empty lists
        if indices_above_threshold.numel() == 0:
            return self.select_top_tokens(next_token_logits, 5, exclude_tokens)
        
        # Invariant: There must be tokens above threshold after exclusion
        if indices_above_threshold.numel() == 0:
            raise ValueError("No tokens above threshold after exclusion. Cannot proceed with generation.")
        
        # Sort by probability (highest first)
        indices_and_probs = [
            (idx.item(), masked_probs[idx].item()) 
            for idx in indices_above_threshold
        ]
        sorted_indices_and_probs = sorted(indices_and_probs, key=lambda x: x[1], reverse=True)
        
        # Limit to max_tokens
        sorted_indices_and_probs = sorted_indices_and_probs[:max_tokens]
        
        # Extract token IDs and probabilities
        token_ids = [idx for idx, _ in sorted_indices_and_probs]
        token_probs = [prob for _, prob in sorted_indices_and_probs]
        
        return token_ids, token_probs
    
    def select_top_tokens(
        self, 
        next_token_logits: torch.Tensor, 
        top_k: int = 5,
        exclude_tokens: Optional[List[int]] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Select top-k tokens by probability.
        Useful as a fallback when threshold-based selection returns nothing.
        
        Args:
            next_token_logits: Logits tensor for next token
            top_k: Number of top tokens to select
            exclude_tokens: Optional list of token IDs to exclude
            
        Returns:
            tuple: (token_ids, token_probs)
        """
        # Apply softmax to get probabilities
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        
        # Invariant: Probabilities must sum to approximately 1.0 after softmax
        prob_sum = next_token_probs.sum().item()
        if not (0.99 <= prob_sum <= 1.01):
            raise ValueError(f"Invariant violation: Token probabilities sum to {prob_sum}, not approximately 1.0 after softmax")
        
        # Convert to CPU tensors
        next_token_probs = next_token_probs.squeeze().cpu()
        
        # Apply exclusion mask if needed
        if exclude_tokens:
            exclude_mask = torch.ones_like(next_token_probs, dtype=torch.bool)
            exclude_mask[exclude_tokens] = False
            # Apply the mask
            masked_probs = next_token_probs * exclude_mask
        else:
            masked_probs = next_token_probs
        
        # Get top-k token indices and probabilities
        top_k_probs, top_k_indices = torch.topk(masked_probs, min(top_k, masked_probs.size(0)))
        
        # Convert to Python lists
        token_ids = top_k_indices.tolist()
        token_probs = top_k_probs.tolist()
        
        return token_ids, token_probs
    
    def is_eos_token(self, token_id: int) -> bool:
        """
        Check if a token is an EOS token.
        
        Args:
            token_id: Token ID to check
            
        Returns:
            bool: True if token is EOS, False otherwise
        """
        return token_id == self.eos_token_id
    
    def filter_repetitive_tokens(
        self,
        token_ids: List[int],
        token_probs: List[float],
        last_n_tokens: List[int],
        repetition_penalty: float = 0.9
    ) -> Tuple[List[int], List[float]]:
        """
        Filter out tokens that would create repetitive sequences.
        
        Args:
            token_ids: List of candidate token IDs
            token_probs: Corresponding probabilities
            last_n_tokens: Last N tokens generated
            repetition_penalty: Penalty factor for repetitive tokens
            
        Returns:
            tuple: (filtered_token_ids, filtered_token_probs)
        """
        if not last_n_tokens or len(last_n_tokens) < 2:
            return token_ids, token_probs
        
        # Detect potential repetitions
        repetition_detected = False
        
        # Check for token-level repetition (same token repeated)
        if len(last_n_tokens) >= 3:
            if last_n_tokens[-1] == last_n_tokens[-2] == last_n_tokens[-3]:
                repetition_detected = True
        
        # If repetition detected, filter tokens
        if repetition_detected:
            # Identify the repeating token
            repeating_token = last_n_tokens[-1]
            
            # Filter out the repeating token completely
            filtered_indices = [i for i, token in enumerate(token_ids) if token != repeating_token]
            
            # If no tokens left after filtering, return original set with penalties
            if not filtered_indices:
                # Apply repetition penalty to probabilities
                penalized_probs = [prob * repetition_penalty if token == repeating_token else prob
                                  for token, prob in zip(token_ids, token_probs)]
                
                # Sort by penalized probabilities
                paired = list(zip(token_ids, penalized_probs))
                paired.sort(key=lambda x: x[1], reverse=True)
                
                # Unpack
                return [t for t, _ in paired], [p for _, p in paired]
            
            # Extract filtered tokens and probabilities
            filtered_token_ids = [token_ids[i] for i in filtered_indices]
            filtered_token_probs = [token_probs[i] for i in filtered_indices]
            
            return filtered_token_ids, filtered_token_probs
        
        # No repetition detected, return original tokens
        return token_ids, token_probs
    
    def decode_tokens(self, token_ids: List[int]) -> List[str]:
        """
        Decode token IDs to human-readable strings.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            List[str]: Decoded token texts
        """
        return [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids] 
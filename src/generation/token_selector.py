import torch
from typing import List, Tuple, Optional, Any, Set
import numpy as np
from src.utils.logging_utils import LoggingMixin


class TokenSelector(LoggingMixin):
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
        super().__init__()
        self.tokenizer = tokenizer

        # Cache EOS token ID since we check it frequently
        # Invariant: Tokenizer must have an eos_token_id
        if not hasattr(tokenizer, "eos_token_id"):
            raise ValueError("Tokenizer must have an eos_token_id")
        self.eos_token_id = tokenizer.eos_token_id

        # Setup logging using the mixin
        self.setup_logging("token_selector", "token_selector_debug.log")

    def select_tokens_above_threshold(
        self, next_token_logits: torch.Tensor, threshold: float, max_tokens: int = 25
    ) -> Tuple[List[int], List[float]]:
        """
        Select tokens with probabilities above the threshold.
        Optimized for performance with batched tensor operations.

        Args:
            next_token_logits: Logits tensor for next token [batch_size, vocab_size] or [batch_size, seq_len, vocab_size]
            threshold: Probability threshold
            max_tokens: Maximum number of tokens to return

        Returns:
            tuple: (token_ids, token_probs) as NumPy arrays
        """
        # DIAGNOSTIC: Show logits shape and values
        is_first_token = (
            next_token_logits.size(0) == 1
            and hasattr(self, "_first_token_processed") == False
        )
        if is_first_token:
            self._first_token_processed = True
            self.log(f"\nDIAGNOSTIC - First token logits:")
            self.log(f"  Logits shape: {next_token_logits.shape}")
            self.log(
                f"  Logits min: {next_token_logits.min().item():.4f}, max: {next_token_logits.max().item():.4f}"
            )
            self.log(f"  Threshold: {threshold}")

        # Ensure we're working with the right tensor shape
        # If we have a 3D tensor [batch_size, seq_len, vocab_size]
        # We want to focus on the last position of the sequence
        if len(next_token_logits.shape) == 3:
            # Get logits for the last position in the sequence
            next_token_logits = next_token_logits[:, -1, :]

        # Apply softmax to logits efficiently using torch.softmax
        next_token_probs = torch.softmax(next_token_logits, dim=-1)

        # Invariant: Probabilities must sum to approximately 1.0 after softmax
        prob_sum = next_token_probs.sum().item()
        if not (0.99 <= prob_sum <= 1.01):
            # If next_token_logits has a batch dimension (e.g., [batch_size, vocab_size])
            # summing all values will give batch_size * 1.0 instead of just 1.0
            # We need to sum only across the vocabulary dimension (dim=-1)
            if len(next_token_logits.shape) > 1:
                prob_sum_per_batch = next_token_probs.sum(dim=-1)
                # Check if each batch row sums to approximately 1.0
                if not torch.all(
                    (0.99 <= prob_sum_per_batch) & (prob_sum_per_batch <= 1.01)
                ):
                    raise ValueError(
                        f"Invariant violation: Token probabilities sum to {prob_sum}, not approximately 1.0 after softmax"
                    )
            else:
                raise ValueError(
                    f"Invariant violation: Token probabilities sum to {prob_sum}, not approximately 1.0 after softmax"
                )

        # DIAGNOSTIC: Show probability distribution
        if is_first_token:
            self.log(f"\nDIAGNOSTIC - Token probability distribution:")
            self.log(f"  Probability sum: {prob_sum:.6f}")
            top_k = 10
            # Ensure we're getting the right probabilities for multi-dimensional tensors
            if next_token_probs.dim() > 1:
                # For multi-dimensional tensor, get first batch only for diagnostic
                top_probs, top_indices = torch.topk(next_token_probs[0], top_k)
            else:
                top_probs, top_indices = torch.topk(next_token_probs, top_k)
            self.log(f"  Top {top_k} token probabilities:")

            # Get token strings for top 10
            top_tokens = []
            for i in range(top_k):
                # Handle possible multi-dimensional indices by flattening or accessing properly
                if top_indices.dim() > 1:
                    # For multi-dimensional tensor, get the i-th element correctly
                    token_id = (
                        top_indices[0, i].item()
                        if top_indices.size(0) > 0
                        else top_indices[i, 0].item()
                    )
                else:
                    # For 1D tensor
                    token_id = top_indices[i].item()
                token_text = self.tokenizer.decode([int(token_id)])
                top_tokens.append(token_text)

            for i, (token_text, token_id, prob) in enumerate(
                zip(
                    top_tokens,
                    top_indices.reshape(-1).tolist()[:top_k],
                    top_probs.reshape(-1).tolist()[:top_k],
                )
            ):
                self.log(f"    {i+1}. '{token_text}' (ID: {token_id}): {prob:.6f}")

            # Also show sum of top 100 probabilities
            if next_token_probs.dim() > 1:
                top_100_probs, _ = torch.topk(next_token_probs[0], 100)
            else:
                top_100_probs, _ = torch.topk(next_token_probs, 100)
            self.log(
                f"  Sum of top 100 token probabilities: {top_100_probs.sum().item():.6f}"
            )

        # Convert to CPU and ensure we have the right shape
        # If we have a batch dimension [batch_size, vocab_size], take the first batch
        if next_token_probs.dim() > 1:
            next_token_probs = next_token_probs[0].cpu()
        else:
            next_token_probs = next_token_probs.cpu()

        # Find all probabilities above threshold efficiently
        indices_above_threshold = torch.nonzero(next_token_probs >= threshold).squeeze(
            -1
        )

        # DIAGNOSTIC: Show number of tokens above threshold
        if is_first_token:
            self.log(
                f"\nDIAGNOSTIC - Found {indices_above_threshold.numel()} tokens above threshold {threshold}"
            )

        # If no tokens are above threshold, return empty arrays
        if indices_above_threshold.numel() == 0:
            if is_first_token:
                self.log(
                    "DIAGNOSTIC - No tokens above threshold! Showing top tokens anyway."
                )

                # Select top-k tokens for debugging
                if next_token_probs.dim() > 1:
                    top_k_probs, top_k_indices = torch.topk(
                        next_token_probs[0], min(5, next_token_probs.size(-1))
                    )
                else:
                    top_k_probs, top_k_indices = torch.topk(
                        next_token_probs, min(5, next_token_probs.size(-1))
                    )

                # Show token info
                tokens_info = []
                for i, (idx, prob) in enumerate(
                    zip(top_k_indices.tolist(), top_k_probs.tolist())
                ):
                    token_text = self.tokenizer.decode([int(idx)])
                    tokens_info.append(
                        f"    {i+1}. '{token_text}' (ID: {idx}): {prob:.6f}"
                    )

                self.log("\nTop 5 tokens (all below threshold):")
                self.log("\n".join(tokens_info))
                self.log(f"Current threshold: {threshold}")

            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        # Get filtered probabilities for tokens above threshold
        probs_above_threshold = next_token_probs[indices_above_threshold]

        # Use torch.topk to sort in a vectorized way (highest probabilities first)
        # This handles both sorting and limiting to max_tokens in one operation
        if indices_above_threshold.numel() <= max_tokens:
            # If we have fewer tokens than max_tokens, just sort them all
            sorted_probs, sorted_indices = torch.sort(
                probs_above_threshold, descending=True
            )
        else:
            # If we have more tokens than max_tokens, get the top max_tokens
            sorted_probs, sorted_indices = torch.topk(probs_above_threshold, max_tokens)

        # Map sorted_indices back to original token indices
        # sorted_indices are positions in indices_above_threshold, not the actual token IDs
        token_indices = indices_above_threshold[sorted_indices]

        # Convert tensors to NumPy arrays (more efficient than lists for large data)
        # Convert BFloat16 to float32 before converting to NumPy
        token_ids = token_indices.numpy().astype(np.int32)
        token_probs = sorted_probs.to(torch.float32).numpy().astype(np.float32)

        # DIAGNOSTIC: Show selected tokens
        if is_first_token:
            self.log(
                f"\nDIAGNOSTIC - Selected {len(token_ids)} tokens above threshold {threshold}:"
            )
            token_info = []
            for i, (tid, prob) in enumerate(zip(token_ids, token_probs)):
                token_text = self.tokenizer.decode(
                    [int(tid)], skip_special_tokens=False
                )
                token_info.append(
                    f"    {i+1}. '{token_text}' (ID: {int(tid)}): {prob:.6f}"
                )
            self.log("\n".join(token_info))

        return token_ids, token_probs

    def select_tokens_above_threshold_excluding(
        self,
        next_token_logits: torch.Tensor,
        threshold: float,
        exclude_tokens: List[int],
        max_tokens: int = 25,
    ) -> Tuple[List[int], List[float]]:
        """
        Select tokens above threshold while excluding specific tokens.
        Useful for avoiding repetition loops.

        Args:
            next_token_logits: Logits tensor for next token [batch_size, vocab_size] or [batch_size, seq_len, vocab_size]
            threshold: Probability threshold
            exclude_tokens: List of token IDs to exclude
            max_tokens: Maximum number of tokens to return

        Returns:
            tuple: (token_ids, token_probs) as NumPy arrays
        """
        # Ensure we're working with the right tensor shape
        # If we have a 3D tensor [batch_size, seq_len, vocab_size]
        # We want to focus on the last position of the sequence
        if len(next_token_logits.shape) == 3:
            # Get logits for the last position in the sequence
            next_token_logits = next_token_logits[:, -1, :]

        # Apply softmax to logits efficiently
        next_token_probs = torch.softmax(next_token_logits, dim=-1)

        # Invariant: Probabilities must sum to approximately 1.0 after softmax
        prob_sum = next_token_probs.sum().item()
        if not (0.99 <= prob_sum <= 1.01):
            # If next_token_logits has a batch dimension (e.g., [batch_size, vocab_size])
            # summing all values will give batch_size * 1.0 instead of just 1.0
            # We need to sum only across the vocabulary dimension (dim=-1)
            if len(next_token_logits.shape) > 1:
                prob_sum_per_batch = next_token_probs.sum(dim=-1)
                # Check if each batch row sums to approximately 1.0
                if not torch.all(
                    (0.99 <= prob_sum_per_batch) & (prob_sum_per_batch <= 1.01)
                ):
                    raise ValueError(
                        f"Invariant violation: Token probabilities sum to {prob_sum}, not approximately 1.0 after softmax"
                    )
            else:
                raise ValueError(
                    f"Invariant violation: Token probabilities sum to {prob_sum}, not approximately 1.0 after softmax"
                )

        # Convert to CPU and ensure we have the right shape
        # If we have a batch dimension [batch_size, vocab_size], take the first batch
        if next_token_probs.dim() > 1:
            next_token_probs = next_token_probs[0].cpu()
        else:
            next_token_probs = next_token_probs.cpu()

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

        # If no tokens are above threshold, return empty arrays
        if indices_above_threshold.numel() == 0:
            return self.select_top_tokens(next_token_logits, 5, exclude_tokens)

        # Invariant: There must be tokens above threshold after exclusion
        if indices_above_threshold.numel() == 0:
            raise ValueError(
                "No tokens above threshold after exclusion. Cannot proceed with generation."
            )

        # Get filtered probabilities for tokens above threshold
        probs_above_threshold = masked_probs[indices_above_threshold]

        # Use torch.topk to sort in a vectorized way (highest probabilities first)
        # This handles both sorting and limiting to max_tokens in one operation
        if indices_above_threshold.numel() <= max_tokens:
            # If we have fewer tokens than max_tokens, just sort them all
            sorted_probs, sorted_indices = torch.sort(
                probs_above_threshold, descending=True
            )
        else:
            # If we have more tokens than max_tokens, get the top max_tokens
            sorted_probs, sorted_indices = torch.topk(probs_above_threshold, max_tokens)

        # Map sorted_indices back to original token indices
        # sorted_indices are positions in indices_above_threshold, not the actual token IDs
        token_indices = indices_above_threshold[sorted_indices]

        # Convert tensors to NumPy arrays (more efficient than lists for large data)
        # Convert BFloat16 to float32 before converting to NumPy
        token_ids = token_indices.numpy().astype(np.int32)
        token_probs = sorted_probs.to(torch.float32).numpy().astype(np.float32)

        return token_ids, token_probs

    def select_top_tokens(
        self,
        next_token_logits: torch.Tensor,
        top_k: int = 5,
        exclude_tokens: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[float]]:
        """
        Select top-k tokens by probability.
        Useful as a fallback when threshold-based selection returns nothing.

        Args:
            next_token_logits: Logits tensor for next token [batch_size, vocab_size] or [batch_size, seq_len, vocab_size]
            top_k: Number of top tokens to select
            exclude_tokens: Optional list of token IDs to exclude

        Returns:
            tuple: (token_ids, token_probs) as NumPy arrays
        """
        # Ensure we're working with the right tensor shape
        # If we have a 3D tensor [batch_size, seq_len, vocab_size]
        # We want to focus on the last position of the sequence
        if len(next_token_logits.shape) == 3:
            # Get logits for the last position in the sequence
            next_token_logits = next_token_logits[:, -1, :]

        # Apply softmax to get probabilities
        next_token_probs = torch.softmax(next_token_logits, dim=-1)

        # Invariant: Probabilities must sum to approximately 1.0 after softmax
        prob_sum = next_token_probs.sum().item()
        if not (0.99 <= prob_sum <= 1.01):
            # If next_token_logits has a batch dimension (e.g., [batch_size, vocab_size])
            # summing all values will give batch_size * 1.0 instead of just 1.0
            # We need to sum only across the vocabulary dimension (dim=-1)
            if len(next_token_logits.shape) > 1:
                prob_sum_per_batch = next_token_probs.sum(dim=-1)
                # Check if each batch row sums to approximately 1.0
                if not torch.all(
                    (0.99 <= prob_sum_per_batch) & (prob_sum_per_batch <= 1.01)
                ):
                    raise ValueError(
                        f"Invariant violation: Token probabilities sum to {prob_sum}, not approximately 1.0 after softmax"
                    )
            else:
                raise ValueError(
                    f"Invariant violation: Token probabilities sum to {prob_sum}, not approximately 1.0 after softmax"
                )

        # Convert to CPU and ensure we have the right shape
        # If we have a batch dimension [batch_size, vocab_size], take the first batch
        if next_token_probs.dim() > 1:
            next_token_probs = next_token_probs[0].cpu()
        else:
            next_token_probs = next_token_probs.cpu()

        # Apply exclusion mask if needed
        if exclude_tokens:
            exclude_mask = torch.ones_like(next_token_probs, dtype=torch.bool)
            exclude_mask[exclude_tokens] = False
            # Apply the mask
            masked_probs = next_token_probs * exclude_mask
        else:
            masked_probs = next_token_probs

        # Get top-k token indices and probabilities
        top_k_probs, top_k_indices = torch.topk(
            masked_probs, min(top_k, masked_probs.size(0))
        )

        # Convert tensors to NumPy arrays (more efficient than lists for large data)
        # Convert BFloat16 to float32 before converting to NumPy
        token_ids = top_k_indices.numpy().astype(np.int32)
        token_probs = top_k_probs.to(torch.float32).numpy().astype(np.float32)

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

    def all_are_eos_tokens(self, token_ids: List[int]) -> bool:
        """
        Vectorized check if all tokens in a list are EOS tokens.

        Args:
            token_ids: NumPy array or list of token IDs to check

        Returns:
            bool: True if all tokens are EOS, False otherwise
        """
        if not isinstance(token_ids, (list, np.ndarray)) or len(token_ids) == 0:
            return False

        # Convert to tensor for vectorized comparison
        if isinstance(token_ids, np.ndarray):
            # If already a NumPy array, use directly
            return bool(np.all(token_ids == self.eos_token_id))
        else:
            # Convert list to tensor
            tokens_tensor = torch.tensor(token_ids, dtype=torch.long)
            return bool(torch.all(tokens_tensor == self.eos_token_id).item())

    def filter_repetitive_tokens(
        self,
        token_ids: List[int],
        token_probs: List[float],
        last_n_tokens: List[int],
        repetition_penalty: float = 0.9,
    ) -> Tuple[List[int], List[float]]:
        """
        Filter out tokens that would create repetitive sequences.

        Args:
            token_ids: NumPy array of candidate token IDs
            token_probs: NumPy array of corresponding probabilities
            last_n_tokens: List of last N tokens generated
            repetition_penalty: Penalty factor for repetitive tokens

        Returns:
            tuple: (filtered_token_ids, filtered_token_probs) as NumPy arrays
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

            # Create a mask for non-repeating tokens
            non_repeating_mask = token_ids != repeating_token

            # If all tokens would be filtered out, apply penalty instead
            if not np.any(non_repeating_mask):
                # Create penalty mask (1.0 for non-repeating, penalty for repeating)
                penalty_mask = np.where(
                    token_ids == repeating_token, repetition_penalty, 1.0
                )

                # Apply penalty to probabilities
                penalized_probs = token_probs * penalty_mask

                # Sort by penalized probabilities
                sorted_indices = np.argsort(-penalized_probs)  # Descending order

                # Return sorted arrays
                return token_ids[sorted_indices], penalized_probs[sorted_indices]

            # Filter using the mask
            filtered_token_ids = token_ids[non_repeating_mask]
            filtered_token_probs = token_probs[non_repeating_mask]

            return filtered_token_ids, filtered_token_probs

        # No repetition detected, return original arrays
        return token_ids, token_probs

    def decode_tokens(self, token_ids: List[int]) -> List[str]:
        """
        Decode token IDs to human-readable strings.

        Args:
            token_ids: NumPy array or list of token IDs

        Returns:
            List[str]: Decoded token texts
        """
        # Ensure we're working with Python integers for tokenizer compatibility
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()

        # Invariant: All token IDs must be integers
        if not all(isinstance(tid, (int, np.integer)) for tid in token_ids):
            raise ValueError(
                f"Invariant violation: Token IDs must be integers, got {[type(tid) for tid in token_ids]}"
            )

        return [
            self.tokenizer.decode([int(tid)], skip_special_tokens=False)
            for tid in token_ids
        ]

    def select_tokens(
        self, next_token_logits: torch.Tensor, threshold: float, max_tokens: int = 25
    ) -> Tuple[List[Tuple[torch.Tensor, float]], int]:
        """
        Select tokens with probabilities above the threshold.
        This is a wrapper around select_tokens_above_threshold that returns the format
        expected by ParallelGenerator.

        Args:
            next_token_logits: Logits tensor for next token
            threshold: Probability threshold
            max_tokens: Maximum number of tokens to return

        Returns:
            tuple: (token_data, subset_size)
                token_data: List of (token_id_tensor, probability) tuples
                subset_size: Number of tokens above threshold
        """
        if self.debug_mode:
            self.log(f"select_tokens called with threshold {threshold}")

        # Call the existing method to get token IDs and probabilities
        token_ids, token_probs = self.select_tokens_above_threshold(
            next_token_logits, threshold, max_tokens
        )

        # If no tokens were selected, fall back to the top token
        if len(token_ids) == 0:
            if self.debug_mode:
                self.log(
                    f"No tokens above threshold {threshold}, falling back to top token"
                )

            # Get the top token regardless of threshold
            top_ids, top_probs = self.select_top_tokens(next_token_logits, top_k=1)
            token_ids = top_ids
            token_probs = top_probs

            if self.debug_mode:
                token_text = self.tokenizer.decode(
                    [int(token_ids[0])], skip_special_tokens=False
                )
                self.log(
                    f"Selected top token: '{token_text}' (ID: {int(token_ids[0])}) with prob: {token_probs[0]:.6f}"
                )

        # Convert to the format expected by ParallelGenerator
        token_data = []
        for token_id, prob in zip(token_ids, token_probs):
            # Create tensor for token ID
            token_tensor = torch.tensor(token_id, device=next_token_logits.device)
            token_data.append((token_tensor, prob))

        subset_size = len(token_data)

        if self.debug_mode:
            self.log(f"Returning {subset_size} tokens above threshold {threshold}")

        return token_data, subset_size

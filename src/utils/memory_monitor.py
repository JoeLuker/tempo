"""Memory monitoring and control utilities for TEMPO.

This module provides tools to monitor and control memory usage to stay within limits.
Key memory consumers:
- Model weights (~3GB for 3B parameter model)
- KV cache (grows with sequence length and parallel tokens)
- Parallel token batches
- Attention matrices
"""

import torch
import psutil
import gc
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Current memory statistics."""
    total_gb: float
    used_gb: float
    available_gb: float
    percent_used: float

    # GPU-specific (if available)
    gpu_allocated_gb: Optional[float] = None
    gpu_reserved_gb: Optional[float] = None
    gpu_free_gb: Optional[float] = None


class MemoryMonitor:
    """Monitors and controls memory usage to stay within limits."""

    def __init__(self, max_memory_gb: float = 36.0, device: str = "cuda"):
        """Initialize memory monitor.

        Args:
            max_memory_gb: Maximum memory usage allowed (default 36GB)
            device: Device to monitor ("cuda", "cpu", "mps")
        """
        assert max_memory_gb > 0, "max_memory_gb must be positive"

        self.max_memory_gb = max_memory_gb
        self.device = device
        self.warning_threshold = 0.85  # Warn at 85% usage
        self.critical_threshold = 0.95  # Critical at 95% usage

        logger.info(f"Initialized MemoryMonitor with {max_memory_gb}GB limit")

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics.

        Returns:
            MemoryStats object with current usage
        """
        # System memory
        mem = psutil.virtual_memory()
        stats = MemoryStats(
            total_gb=mem.total / (1024**3),
            used_gb=mem.used / (1024**3),
            available_gb=mem.available / (1024**3),
            percent_used=mem.percent
        )

        # GPU memory if available
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            stats.gpu_allocated_gb = gpu_allocated
            stats.gpu_reserved_gb = gpu_reserved
            stats.gpu_free_gb = gpu_total - gpu_reserved

        return stats

    def get_current_usage_gb(self) -> float:
        """Get current memory usage in GB.

        Returns:
            Current memory usage in GB
        """
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        else:
            mem = psutil.virtual_memory()
            return mem.used / (1024**3)

    def get_available_memory_gb(self) -> float:
        """Get available memory in GB.

        Returns:
            Available memory in GB
        """
        current_usage = self.get_current_usage_gb()
        return self.max_memory_gb - current_usage

    def is_within_limit(self, buffer_gb: float = 2.0) -> bool:
        """Check if memory usage is within limit.

        Args:
            buffer_gb: Safety buffer in GB (default 2GB)

        Returns:
            True if within limit, False otherwise
        """
        current_usage = self.get_current_usage_gb()
        return current_usage <= (self.max_memory_gb - buffer_gb)

    def check_memory_limit(self, operation: str = "operation") -> None:
        """Check memory limit and raise warning if exceeded.

        Args:
            operation: Name of operation being performed

        Raises:
            MemoryError: If memory limit is critically exceeded
        """
        current_usage = self.get_current_usage_gb()
        usage_ratio = current_usage / self.max_memory_gb

        if usage_ratio >= self.critical_threshold:
            msg = f"CRITICAL: Memory usage {current_usage:.2f}GB/{self.max_memory_gb}GB ({usage_ratio*100:.1f}%) during {operation}"
            logger.error(msg)
            raise MemoryError(msg)
        elif usage_ratio >= self.warning_threshold:
            msg = f"WARNING: High memory usage {current_usage:.2f}GB/{self.max_memory_gb}GB ({usage_ratio*100:.1f}%) during {operation}"
            logger.warning(msg)

    def estimate_kv_cache_memory(
        self,
        batch_size: int,
        sequence_length: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16
    ) -> float:
        """Estimate memory required for KV cache.

        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Head dimension
            dtype: Data type (default float16)

        Returns:
            Estimated memory in GB
        """
        # Each layer has K and V cache: [batch, heads, seq, dim]
        bytes_per_element = 2 if dtype == torch.float16 else 4

        elements_per_layer = batch_size * num_heads * sequence_length * head_dim
        bytes_per_layer = elements_per_layer * bytes_per_element * 2  # K and V
        total_bytes = bytes_per_layer * num_layers

        return total_bytes / (1024**3)

    def estimate_parallel_batch_memory(
        self,
        num_parallel_tokens: int,
        sequence_length: int,
        vocab_size: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float16
    ) -> float:
        """Estimate memory for parallel token processing.

        Args:
            num_parallel_tokens: Number of parallel tokens
            sequence_length: Current sequence length
            vocab_size: Vocabulary size
            hidden_size: Hidden dimension size
            dtype: Data type

        Returns:
            Estimated memory in GB
        """
        bytes_per_element = 2 if dtype == torch.float16 else 4

        # Input embeddings: [parallel_tokens, seq_len, hidden]
        embedding_bytes = num_parallel_tokens * sequence_length * hidden_size * bytes_per_element

        # Logits: [parallel_tokens, vocab_size]
        logits_bytes = num_parallel_tokens * vocab_size * bytes_per_element

        # Attention: [parallel_tokens, num_heads, seq_len, seq_len] (approximate)
        # Using conservative estimate with 32 heads
        attention_bytes = num_parallel_tokens * 32 * sequence_length * sequence_length * bytes_per_element

        total_bytes = embedding_bytes + logits_bytes + attention_bytes
        return total_bytes / (1024**3)

    def calculate_max_parallel_tokens(
        self,
        sequence_length: int,
        vocab_size: int = 128256,
        hidden_size: int = 3072,
        available_memory_gb: Optional[float] = None
    ) -> int:
        """Calculate maximum parallel tokens that fit in available memory.

        Args:
            sequence_length: Current sequence length
            vocab_size: Vocabulary size (default for Llama 3B)
            hidden_size: Hidden size (default for Llama 3B)
            available_memory_gb: Available memory (if None, calculates from limit)

        Returns:
            Maximum number of parallel tokens
        """
        if available_memory_gb is None:
            available_memory_gb = self.get_available_memory_gb()

        # Reserve 20% for overhead and safety
        usable_memory_gb = available_memory_gb * 0.8

        # Estimate memory per parallel token
        memory_per_token = self.estimate_parallel_batch_memory(
            num_parallel_tokens=1,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            hidden_size=hidden_size
        )

        max_tokens = int(usable_memory_gb / memory_per_token)

        # Apply reasonable bounds
        max_tokens = max(1, min(max_tokens, 50))  # Between 1 and 50

        logger.debug(
            f"Max parallel tokens: {max_tokens} "
            f"(available: {available_memory_gb:.2f}GB, per token: {memory_per_token:.3f}GB)"
        )

        return max_tokens

    def free_memory(self) -> None:
        """Aggressively free unused memory."""
        gc.collect()

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.debug("Freed unused memory")

    def log_memory_stats(self, prefix: str = "") -> None:
        """Log current memory statistics.

        Args:
            prefix: Prefix for log message
        """
        stats = self.get_memory_stats()

        msg_parts = [f"{prefix}Memory:" if prefix else "Memory:"]
        msg_parts.append(f"System {stats.used_gb:.2f}/{stats.total_gb:.2f}GB ({stats.percent_used:.1f}%)")

        if stats.gpu_allocated_gb is not None:
            msg_parts.append(f"GPU {stats.gpu_allocated_gb:.2f}GB allocated, {stats.gpu_reserved_gb:.2f}GB reserved")

        logger.info(" ".join(msg_parts))

    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory configuration for other components.

        Returns:
            Dictionary with memory constraints
        """
        stats = self.get_memory_stats()
        available = self.get_available_memory_gb()

        return {
            "max_memory_gb": self.max_memory_gb,
            "current_usage_gb": self.get_current_usage_gb(),
            "available_gb": available,
            "warning_threshold_gb": self.max_memory_gb * self.warning_threshold,
            "critical_threshold_gb": self.max_memory_gb * self.critical_threshold,
            "device": self.device
        }


def get_memory_monitor(max_memory_gb: float = 36.0, device: str = "cuda") -> MemoryMonitor:
    """Factory function to create memory monitor.

    Args:
        max_memory_gb: Maximum memory in GB
        device: Device to monitor

    Returns:
        MemoryMonitor instance
    """
    return MemoryMonitor(max_memory_gb=max_memory_gb, device=device)

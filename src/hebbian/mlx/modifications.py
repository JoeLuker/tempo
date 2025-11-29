"""Batched Hebbian modifications for efficient MLX computation.

Hebbian updates are rank-1 modifications: ΔW = scale * outer(output, input).
Instead of applying these one at a time (slow Python loop), we store them
as stacked arrays and apply all modifications in a single batched matmul.

This achieves 10-100x speedup over naive per-modification loops.
"""

from dataclasses import dataclass
import mlx.core as mx


@dataclass
class HebbianModifications:
    """Batched storage of rank-1 Hebbian modifications for a single layer.

    Each modification is a rank-1 update to the K projection:
        K' = K + scale * outer(output_vec, input_vec) @ input
           = K + scale * output_vec * (input_vec · input)

    For efficient computation, we batch all modifications and apply them
    in two matrix multiplications instead of N individual outer products.

    Attributes:
        k_dim: Output dimension (K projection dimension)
        hidden_dim: Input dimension (hidden state dimension)
        max_mods: Maximum number of modifications to store before consolidation
    """

    k_dim: int
    hidden_dim: int
    max_mods: int = 100

    def __post_init__(self):
        """Initialize empty modification storage."""
        self._output_vecs: mx.array | None = None  # (n_mods, k_dim)
        self._input_vecs: mx.array | None = None  # (n_mods, hidden_dim)
        self._scales: mx.array | None = None  # (n_mods,)
        self._n_active: int = 0

    @property
    def n_active(self) -> int:
        """Number of active modifications."""
        return self._n_active

    def add(
        self,
        output_vec: mx.array,
        input_vec: mx.array,
        scale: float,
    ) -> None:
        """Add a rank-1 modification.

        Args:
            output_vec: Output direction vector of shape (k_dim,)
            input_vec: Input direction vector of shape (hidden_dim,)
            scale: Scaling factor for this modification
        """
        # Ensure vectors are 1D
        output_vec = output_vec.flatten()
        input_vec = input_vec.flatten()

        if self._output_vecs is None:
            # First modification - initialize arrays
            self._output_vecs = output_vec[None, :]  # (1, k_dim)
            self._input_vecs = input_vec[None, :]  # (1, hidden_dim)
            self._scales = mx.array([scale])
        else:
            # Append to existing arrays
            self._output_vecs = mx.concatenate(
                [self._output_vecs, output_vec[None, :]], axis=0
            )
            self._input_vecs = mx.concatenate(
                [self._input_vecs, input_vec[None, :]], axis=0
            )
            self._scales = mx.concatenate([self._scales, mx.array([scale])])

        self._n_active += 1

        # Evaluate to prevent lazy graph explosion
        if self._n_active % 10 == 0:
            mx.eval(self._output_vecs, self._input_vecs, self._scales)

    def apply(self, x: mx.array) -> mx.array:
        """Apply all modifications to compute the K projection delta.

        Computes: sum_i scale_i * output_i * (input_i · x)

        This is equivalent to applying all rank-1 modifications but done
        in two efficient batched matrix multiplications.

        Args:
            x: Input hidden states of shape (batch, seq_len, hidden_dim)

        Returns:
            delta: K projection delta of shape (batch, seq_len, k_dim)
        """
        if self._n_active == 0:
            # No modifications - return zeros
            return mx.zeros((x.shape[0], x.shape[1], self.k_dim), dtype=x.dtype)

        # Step 1: Compute dot products with all input vectors
        # x: (batch, seq, hidden_dim)
        # input_vecs.T: (hidden_dim, n_mods)
        # dots: (batch, seq, n_mods)
        dots = mx.matmul(x, mx.transpose(self._input_vecs))

        # Step 2: Scale the dot products
        # scaled_dots: (batch, seq, n_mods)
        scaled_dots = dots * self._scales

        # Step 3: Compute weighted sum of output vectors
        # scaled_dots: (batch, seq, n_mods)
        # output_vecs: (n_mods, k_dim)
        # delta: (batch, seq, k_dim)
        delta = mx.matmul(scaled_dots, self._output_vecs)

        return delta

    def clear(self) -> None:
        """Remove all modifications."""
        self._output_vecs = None
        self._input_vecs = None
        self._scales = None
        self._n_active = 0

    def prune_oldest(self, n_keep: int) -> None:
        """Keep only the N most recent modifications.

        Args:
            n_keep: Number of modifications to keep
        """
        if self._n_active <= n_keep:
            return

        # Keep only the last n_keep modifications
        self._output_vecs = self._output_vecs[-n_keep:]
        self._input_vecs = self._input_vecs[-n_keep:]
        self._scales = self._scales[-n_keep:]
        self._n_active = n_keep

        mx.eval(self._output_vecs, self._input_vecs, self._scales)

    def total_scale(self) -> float:
        """Get the sum of all modification scales."""
        if self._scales is None:
            return 0.0
        return float(mx.sum(self._scales))

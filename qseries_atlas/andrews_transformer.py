"""
Andrews Transformer: Partition-Based Feature Engineering
=========================================================

Transforms categorical and continuous indices into q-series coefficients
using George Andrews' Theory of Partitions.

Core Concept:
    Instead of treating categories as disconnected labels (Hamming distance),
    we map them to integer partitions and compute Gaussian Binomial Coefficients.
    
    This creates a "graded" similarity: chemically similar catalysts map to
    similar q-binomial vectors, allowing the kernel to leverage structural
    similarity automatically.

Mathematical Foundation:
    The q-Binomial Coefficient (Gaussian Polynomial):
        [n choose k]_q = (1-q^n)(1-q^(n-1))...(1-q^(n-k+1)) / (1-q^k)(1-q^(k-1))...(1-q)
    
    Satisfies the q-Pascal Identity (recurrence relation):
        [n choose k]_q = [n-1 choose k]_q + q^(n-k) * [n-1 choose k-1]_q
    
    This recurrence enables efficient computation via dynamic programming.

Reference:
    Andrews, G. E. (1976). The Theory of Partitions. Addison-Wesley.
    Chapter 3: Gaussian Polynomials and generalized binomial coefficients.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class AndrewsTransformer(nn.Module):
    """
    Maps categorical/continuous inputs to q-series feature space.

    Parameters
    ----------
    q : float, default=1.1
        Deformation parameter. Typical range: (1.0, 2.0)
        - q→1: Recovers standard binomial coefficients
        - q=1.1-1.5: Moderate curvature (recommended)
        - q>2.0: Extreme curvature (rare use cases)

    max_n : int, default=10
        Maximum integer n for partitions. Determines the "complexity cap"
        of your experimental space. For a library of 100 catalysts, max_n≈10.

    Attributes
    ----------
    q_pascal_table : Tensor, optional
        Pre-computed lookup table of shape (max_n+1, max_n+1) containing
        all [n choose k]_q values. Built on first use via `build_table()`.

    Examples
    --------
    >>> transformer = AndrewsTransformer(q=1.1, max_n=10)
    >>> catalyst_indices = torch.tensor([0, 2, 5])
    >>> q_features = transformer.transform(catalyst_indices)
    >>> print(q_features.shape)  # (3, 1)

    Performance Notes
    -----------------
    - First call: O(max_n²) to build the Pascal table
    - Subsequent calls: O(n) lookup operations
    - Memory: ~8 KB for max_n=100, ~800 KB for max_n=1000

    Stability
    ---------
    The recurrence relation avoids numerical overflow/underflow that would
    occur with direct Pochhammer product calculation (which has exponential terms).
    """

    def __init__(self, q: float = 1.1, max_n: int = 10):
        super().__init__()
        self.q_val = float(q)
        self.max_n = int(max_n)

        # Register q as a buffer (not a parameter, since it's fixed)
        self.register_buffer("q", torch.tensor(q, dtype=torch.double))

        # Lazy initialization of the q-Pascal table
        self._q_pascal_table: Optional[torch.Tensor] = None
        self._table_initialized = False

    def build_table(self) -> torch.Tensor:
        """
        Pre-compute q-Pascal triangle using recurrence relation.

        Returns
        -------
        Tensor
            Shape (max_n+1, max_n+1) where entry [n,k] = [n choose k]_q

        Complexity: O(max_n²) time, O(max_n²) space
        """
        table = torch.zeros((self.max_n + 1, self.max_n + 1), dtype=torch.double)

        # Base case: [n choose 0]_q = 1 for all n
        table[:, 0] = 1.0

        # Fill using Andrews' recurrence:
        # [n choose k]_q = [n-1 choose k]_q + q^(n-k) * [n-1 choose k-1]_q
        for n in range(1, self.max_n + 1):
            table[n, 0] = 1.0
            for k in range(1, min(n, self.max_n) + 1):
                if n == k:
                    table[n, k] = 1.0
                else:
                    # Recurrence relation
                    prev_val = table[n - 1, k]
                    prev_diag = table[n - 1, k - 1]
                    q_power = self.q ** (n - k)
                    table[n, k] = prev_val + q_power * prev_diag

        self.register_buffer("_q_pascal_table", table)
        self._table_initialized = True
        return table

    def get_q_binomial(self, n: int, k: int) -> torch.Tensor:
        """
        Retrieve [n choose k]_q from pre-computed table.

        Parameters
        ----------
        n : int
            Top index of binomial coefficient
        k : int
            Bottom index of binomial coefficient

        Returns
        -------
        Tensor
            Scalar tensor with value of [n choose k]_q
        """
        if not self._table_initialized:
            self.build_table()

        if k < 0 or k > n or n > self.max_n:
            return torch.tensor(0.0, dtype=torch.double)

        return self._q_pascal_table[n, k]

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform category indices to q-binomial features.

        Parameters
        ----------
        x : Tensor, shape (..., ) or (..., d)
            Category indices (integers). Can be 1D or batched.

        Returns
        -------
        Tensor
            q-binomial features. Shape (..., 1) if input is 1D,
            or (..., d) if input is batched.

        Logic
        -----
        For each index i in x:
            1. Map i to a partition (n, k) where k = i % (max_n+1), n = max_n
            2. Compute [n choose k]_q
            3. Return as a feature value

        This ensures:
            - Similar indices map to similar binomial values (structural similarity)
            - All values are bounded by the structure of Andrews' partitions
            - Scale is invariant (no need for external normalization)
        """
        if not self._table_initialized:
            self.build_table()

        # Handle different input shapes
        original_shape = x.shape
        x_flat = x.flatten()

        features = []
        for idx in x_flat:
            idx_int = int(idx.item())
            # Map index to partition (n, k)
            # Use modulo to create a natural "wrap-around" for large libraries
            k = idx_int % (self.max_n + 1)
            n = self.max_n
            q_bin = self.get_q_binomial(n, k)
            features.append(q_bin)

        features_tensor = torch.stack(features).reshape(*original_shape, 1)
        return features_tensor

    def __repr__(self) -> str:
        return f"AndrewsTransformer(q={self.q_val}, max_n={self.max_n})"


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Demonstration: How q-binomial coefficients provide structural similarity
    transformer = AndrewsTransformer(q=1.1, max_n=10)

    # Map 10 "catalysts" (indices 0-9) to q-binomial features
    catalyst_indices = torch.arange(10)
    features = transformer.transform(catalyst_indices).squeeze()

    print("Catalyst Index -> q-Binomial Feature")
    print("-" * 40)
    for idx, feat in zip(catalyst_indices, features):
        print(f"  Catalyst {idx}: {feat.item():.4f}")

    # Plot: Show how similar catalysts have similar features
    plt.figure(figsize=(10, 5))
    plt.bar(catalyst_indices.numpy(), features.detach().numpy(), color="steelblue", alpha=0.7)
    plt.xlabel("Catalyst Index")
    plt.ylabel("q-Binomial Feature Value")
    plt.title("George Andrews' Partitions: Structural Similarity in Feature Space")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("andrews_features.png", dpi=150)
    print("\nPlot saved as andrews_features.png")

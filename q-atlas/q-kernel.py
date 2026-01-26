# q-kernels.py
"""
q-kernels.py

Purpose
-------
Provide q-deformed covariance kernels for GPyTorch / BoTorch models in Atlas.

This file is written to be a "drop-in kernel module" consistent with existing
Atlas kernels (e.g., TanimotoKernel). The main deliverable is QGaussianKernel,
a stationary kernel that reduces to an RBF-like exponential kernel as q → 1.

Expected Inputs
---------------
QGaussianKernel.forward(x1, x2, diag=False, **params)
  x1 : torch.Tensor, shape (..., n1, d)
  x2 : torch.Tensor, shape (..., n2, d)
  diag : bool
    If True, return only diagonal entries. Requires x1 and x2 to be aligned.
  **params : passed through by GPyTorch, typically unused by custom kernels.

Expected Outputs
----------------
If diag=False:
  K : torch.Tensor or LazyTensor-like, shape (..., n1, n2)
If diag=True:
  Kdiag : torch.Tensor, shape (..., n1)

Examples
--------
Example 1: Basic kernel matrix
Example 2: Diagonal computation
Example 3: q≈1 behavior check (approaches exp(-u))
Example 4: q>1 heavier tails (slower decay)
Example 5: active_dims usage for mixed kernels

Possible Sources of Error
-------------------------
- Input dtype mismatch with global torch default dtype settings.
- Extremely small lengthscale causing numerical instability (handled by clamp).
- q close to 1 causing division sensitivity (handled by eps).
- Passing integer tensors (should be floating).
- diag=True with mismatched x1/x2 shapes in some GPyTorch paths.
"""

from __future__ import annotations

from typing import Optional

import gpytorch
import torch
from gpytorch.constraints import Interval, Positive


def q_exp_kernel_value(u: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute q-exponential kernel factor: (1 + (q-1)u)^(-1/(q-1)).

    Parameters
    ----------
    u : torch.Tensor
        Non-negative distance measure (typically squared distance / lengthscale^2).
    q : torch.Tensor
        q parameter (broadcastable to u). Should be > 0 and typically near [1, 2.5].
    eps : float
        Small positive number for stability near q≈1 or large u.

    Returns
    -------
    torch.Tensor
        Kernel factor in (0, 1], same shape as u.

    Possible Sources of Error
    -------------------------
    - If q is outside constraints, may create negative bases.
    - If u contains negative values, interpretation breaks.
    """
    # Stabilize denominator and exponent near q = 1.
    denom = (q - 1.0).clamp(min=eps)

    # Base must be positive; clamp prevents taking powers of non-positive numbers.
    base = (1.0 + (q - 1.0) * u).clamp_min(eps)

    # Compute power = -1/(q-1); negative gives decay with u.
    power = -1.0 / denom

    return base.pow(power)


class QGaussianKernel(gpytorch.kernels.Kernel):
    """
    QGaussianKernel

    Purpose
    -------
    Stationary kernel defined via a q-exponential transform of squared distance:
        u = ||x - y||^2 / ℓ^2
        k(x,y) = (1 + (q-1)u)^(-1/(q-1))

    Limit
    -----
    As q → 1, k(x,y) → exp(-u), i.e., an RBF-like kernel on squared distance.

    Parameters
    ----------
    q_init : float
        Initial q value (learnable). Constrained to a reasonable interval.
    **kwargs :
        Passed to gpytorch.kernels.Kernel (e.g., active_dims, batch_shape).

    Expected Inputs / Outputs
    -------------------------
    See module docstring.

    Possible Sources of Error
    -------------------------
    - Using integer tensors for x1/x2.
    - Passing inconsistent shapes for diag=True.
    """

    is_stationary = True

    def __init__(self, q_init: float = 1.2, **kwargs):
        # Initialize base Kernel state (handles active_dims, batch_shape, etc.).
        super().__init__(**kwargs)

        # Register q as a learnable parameter with constraints.
        # Use default dtype (your project sets torch.double globally).
        self.register_parameter("raw_q", torch.nn.Parameter(torch.tensor([q_init])))
        self.register_constraint("raw_q", Interval(0.8, 2.5))

        # Register lengthscale with positivity constraint.
        # This mirrors how many GPyTorch kernels store lengthscale.
        self.register_parameter(
            "raw_lengthscale",
            torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
        )
        self.register_constraint("raw_lengthscale", Positive())

    @property
    def q(self) -> torch.Tensor:
        """Return constrained q."""
        return self.raw_q_constraint.transform(self.raw_q)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params) -> torch.Tensor:
        """
        Compute covariance entries between x1 and x2.

        Parameters
        ----------
        x1, x2 : torch.Tensor
            Input features.
        diag : bool
            If True, compute diagonal only.

        Returns
        -------
        torch.Tensor
            Covariance entries.

        Possible Sources of Error
        -------------------------
        - If x1/x2 are not floating tensors, covar_dist may error or cast unexpectedly.
        - If lengthscale becomes too small, u can blow up; clamp prevents division by zero.
        """
        # Compute squared Euclidean distances using GPyTorch helper (respects active_dims).
        dist2 = self.covar_dist(x1, x2, square_dist=True, diag=diag, **params)

        # Normalize by lengthscale^2; clamp avoids division instability.
        u = dist2 / self.lengthscale.pow(2).clamp_min(1e-12)

        # Apply q-exponential kernel map.
        return q_exp_kernel_value(u=u, q=self.q, eps=1e-12)


def _example_usage() -> None:
    """
    Run minimal sanity examples.

    Expected Outputs
    ----------------
    Printed shapes and a few scalar kernel values.

    Possible Sources of Error
    -------------------------
    - If gpytorch is not installed, import will fail.
    - If default dtype differs from project assumptions, numeric values may differ slightly.
    """
    # Example 1: Basic kernel matrix.
    k = QGaussianKernel(q_init=1.2)
    x = torch.tensor([[0.0], [1.0], [2.0]])
    K = k(x, x).evaluate()
    print("Ex1 K shape:", K.shape)

    # Example 2: Diagonal.
    Kd = k(x, x, diag=True).evaluate()
    print("Ex2 diag shape:", Kd.shape)

    # Example 3: q≈1 behavior.
    k.raw_q.data[:] = 1.00001
    print("Ex3 q≈1 K01:", float(k(x, x).evaluate()[0, 1]))

    # Example 4: q>1 heavier tail.
    k.raw_q.data[:] = 1.8
    print("Ex4 q=1.8 K02:", float(k(x, x).evaluate()[0, 2]))

    # Example 5: active_dims in mixed feature vectors.
    # Suppose features are [cont0, cont1, fp0, fp1, ...]. Use active_dims=(0,1).
    x_mixed = torch.tensor([[0.0, 1.0, 1.0, 0.0], [1.0, 2.0, 0.0, 1.0]])
    k2 = QGaussianKernel(q_init=1.2, active_dims=(0, 1))
    print("Ex5 active_dims K:", k2(x_mixed, x_mixed).evaluate().shape)


if __name__ == "__main__":
    _example_usage()

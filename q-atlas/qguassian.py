"""
q_atlas/kernels/qgaussian.py

Purpose
-------
Define a q-deformed stationary kernel for Gaussian Processes.

How this improves "original Atlas"
----------------------------------
Atlas commonly uses Matérn kernels for continuous variables.
Matérn assumes a particular smoothness and often produces near-Gaussian tails.

A q-deformed kernel can model heavier tails (and sometimes longer-range correlations),
which may better represent "rare jumps" in experimental response surfaces,
phase-like transitions, or heavy-tailed noise patterns.

This kernel is written as a standard GPyTorch Kernel so it can plug into existing
BoTorch/GPyTorch GP models with minimal friction.

Expected Inputs
---------------
forward(x1, x2):
  x1 : torch.Tensor (..., n1, d)
  x2 : torch.Tensor (..., n2, d)
  diag : bool
    If True, return only diagonal.

Expected Outputs
----------------
K : torch.Tensor (..., n1, n2) or (..., n1) if diag=True

5 Examples
----------
Example 1: Kernel matrix on small tensors
Example 2: diag=True output
Example 3: q close to 1 behaves RBF-like
Example 4: larger q gives slower tail decay
Example 5: lengthscale sensitivity

Possible Sources of Error
-------------------------
- Inputs not floating tensors: type errors or unexpected casting.
- Extremely small lengthscale: numerical instability (clamped).
- q too close to 1 can create division issues (handled by eps).
- Very large distances can underflow/overflow (handled by clamp).
"""

import torch
import gpytorch
from gpytorch.constraints import Positive, Interval


class QGaussianKernel(gpytorch.kernels.Kernel):
    """q-exponential-style stationary kernel on squared Euclidean distance."""
    is_stationary = True

    def __init__(self, q_init: float = 1.2, **kwargs):
        super().__init__(**kwargs)

        self.register_parameter("raw_q", torch.nn.Parameter(torch.tensor([q_init], dtype=torch.float32)))
        self.register_constraint("raw_q", Interval(0.8, 2.5))

        self.register_parameter(
            "raw_lengthscale",
            torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
        )
        self.register_constraint("raw_lengthscale", Positive())

    @property
    def q(self) -> torch.Tensor:
        """q after applying constraints."""
        return self.raw_q_constraint.transform(self.raw_q)

    def forward(self, x1, x2, diag: bool = False, **params):
        """
        Compute covariance using:
          u = ||x-y||^2 / ℓ^2
          k = (1 + (q-1)u)^(-1/(q-1))

        This form approaches exp(-u) as q -> 1.
        """
        dist2 = self.covar_dist(x1, x2, square_dist=True, diag=diag, **params)
        u = dist2 / self.lengthscale.pow(2).clamp_min(1e-12)

        q = self.q
        eps = 1e-8
        a = (1.0 + (q - 1.0) * u).clamp_min(eps)
        p = -1.0 / (q - 1.0 + eps)
        return torch.pow(a, p)


# -------------------------
# Five runnable examples
# -------------------------
if __name__ == "__main__":
    k = QGaussianKernel(q_init=1.2)
    x1 = torch.tensor([[0.0], [1.0], [2.0]])
    x2 = torch.tensor([[0.0], [2.0]])

    # Example 1: full kernel matrix
    K = k(x1, x2).evaluate()
    print("Ex1", K.shape)

    # Example 2: diag output
    Kd = k(x1, x1, diag=True).evaluate()
    print("Ex2 diag", Kd.shape)

    # Example 3: q close to 1
    k.raw_q.data[:] = 1.0001
    print("Ex3 q≈1", k(x1, x1).evaluate()[0, 1].item())

    # Example 4: heavier tails with higher q
    k.raw_q.data[:] = 1.8
    print("Ex4 q=1.8", k(x1, x1).evaluate()[0, 2].item())

    # Example 5: lengthscale effect
    k.raw_lengthscale.data[:] = 0.1
    print("Ex5 small ℓ", k(x1, x1).evaluate()[0, 2].item())

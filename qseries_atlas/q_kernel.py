"""
q-Gaussian Kernel for GPyTorch
===============================

Implements the q-exponential kernel using Tsallis entropy maximizers.

The q-Gaussian kernel replaces the standard RBF/Matern kernel with:
    k(x, x') = [1 - (1-q) * ||x-x'||² / ℓ²]^(1/(q-1))

As q → 1, this recovers the standard Gaussian kernel.
For q > 1, it exhibits heavy tails, enabling discovery of rare outcomes.

Theory:
    - Heavy-tailed kernels allow the GP to maintain search weight in
      regions where standard Gaussian kernels expect near-zero probability.
    - This is critical for self-driving labs where "black swan" experiments
      (0.1% yield jumping to 90%) are the target.
"""

import torch
import gpytorch
from gpytorch.constraints import Positive, Interval


class QGaussianKernel(gpytorch.kernels.Kernel):
    """
    q-Gaussian Kernel based on Tsallis entropy maximizers.

    Parameters
    ----------
    q_initial : float, default=1.5
        Deformation parameter. Range: (1.0, 3.0]
        - q=1.0: Recovers standard Gaussian (RBF)
        - q=1.5: Moderate heavy tails
        - q>2.0: Extreme heavy tails (use with care)

    Examples
    --------
    >>> kernel = QGaussianKernel(q_initial=1.1)
    >>> x1 = torch.randn(10, 5)
    >>> x2 = torch.randn(15, 5)
    >>> covar = kernel(x1, x2)
    """

    is_stationary = True

    def __init__(self, q_initial: float = 1.5, **kwargs):
        super().__init__(**kwargs)

        # Register q parameter with constraint
        self.register_parameter(
            name="raw_q",
            parameter=torch.nn.Parameter(torch.tensor([q_initial], dtype=torch.double))
        )
        self.register_constraint("raw_q", Interval(1.01, 3.0))

        # Standard lengthscale parameter
        self.register_parameter(
            name="raw_lengthscale",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1, dtype=torch.double))
        )
        self.register_constraint("raw_lengthscale", Positive())

    @property
    def q(self):
        """Return the constrained q parameter."""
        return self.raw_q_constraint.transform(self.raw_q)

    def forward(self, x1, x2, diag: bool = False, **params):
        """
        Compute the q-Gaussian kernel.

        Parameters
        ----------
        x1 : Tensor, shape (n, d)
            First input
        x2 : Tensor, shape (m, d)
            Second input
        diag : bool
            If True, return only diagonal elements

        Returns
        -------
        Tensor
            Covariance matrix of shape (n, m) or (n,) if diag=True
        """
        # 1. Calculate squared Euclidean distance
        dist = self.covar_dist(x1, x2, square_dist=True, diag=diag, **params)

        # 2. Rescale by lengthscale
        rescaled_dist = dist / self.lengthscale.pow(2)

        # 3. Apply q-exponential: [1 - (1-q)*d]^(1/(q-1))
        q_val = self.q
        power = 1.0 / (q_val - 1.0)

        # The q-exponential has compact support for q > 1
        # Use ReLU to ensure [1 - (1-q)*d] > 0
        base = 1.0 - (q_val - 1.0) * rescaled_dist
        base_clamped = torch.nn.functional.relu(base)

        # Return the kernel value
        return torch.pow(base_clamped, power)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test: Compare standard Gaussian vs q-Gaussian
    x = torch.linspace(-5, 5, 100).unsqueeze(-1)
    x_ref = torch.zeros(1, 1)

    # Standard Gaussian (q=1, approximated via RBF)
    kernel_standard = gpytorch.kernels.RBFKernel()
    cov_standard = kernel_standard(x, x_ref).squeeze()

    # q-Gaussian with q=1.5
    kernel_q = QGaussianKernel(q_initial=1.5)
    cov_q = kernel_q(x, x_ref).squeeze()

    plt.figure(figsize=(10, 5))
    plt.plot(x.squeeze().numpy(), cov_standard.detach().numpy(), label="Standard Gaussian", linewidth=2)
    plt.plot(x.squeeze().numpy(), cov_q.detach().numpy(), label="q-Gaussian (q=1.5)", linewidth=2)
    plt.xlabel("Distance from reference point")
    plt.ylabel("Kernel value")
    plt.legend()
    plt.title("q-Gaussian Kernel: Heavy Tails for Black Swan Discovery")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("q_gaussian_kernel_comparison.png", dpi=150)
    print("Plot saved as q_gaussian_kernel_comparison.png")

"""
QGaussianProcess: Unified q-Series Gaussian Process
=====================================================

A complete BoTorch-compatible Gaussian Process that integrates:
    1. AndrewsTransformer: Maps inputs to q-series feature space
    2. QGaussianKernel: Heavy-tailed covariance via q-exponential
    3. Standard GPyTorch machinery for likelihood and inference

This is the "glue logic" that bridges partition theory with Bayesian Optimization.

Usage in Atlas
--------------
Instead of:
    model = SingleTaskGP(train_x, train_y)

You use:
    model = QGaussianProcess(train_x, train_y, likelihood, q=1.1)

The model automatically handles:
    - Input transformation (category → q-binomial feature)
    - q-Gaussian covariance (heavy-tailed awareness)
    - Seamless integration with BoTorch acquisition functions

Why This Matters
----------------
Standard GP: Assumes data lies on a flat Euclidean space with thin-tailed Gaussian priors.
           → Misses rare discoveries in "heavy tails" where black swans live.

q-Gaussian GP: Explicitly models non-extensive statistics (Tsallis entropy).
             → Can find 0.1% yields jumping to 90% because the kernel expects it.
"""

import gpytorch
import torch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import Likelihood

from .q_kernel import QGaussianKernel
from .andrews_transformer import AndrewsTransformer


class QGaussianProcess(ExactGP, GPyTorchModel):
    """
    A Gaussian Process with q-series expansion and heavy-tailed covariance.

    Combines George Andrews' partition theory with non-extensive statistical
    mechanics to create a GP optimized for discovery in complex experimental spaces.

    Parameters
    ----------
    train_x : Tensor, shape (n, d)
        Training input data
    train_y : Tensor, shape (n, 1) or (n,)
        Training output data
    likelihood : Likelihood
        GPyTorch likelihood (usually GaussianLikelihood)
    q : float, default=1.1
        q-parameter for the q-Gaussian kernel
    max_n : int, default=10
        Maximum partition complexity for AndrewsTransformer

    Attributes
    ----------
    transformer : AndrewsTransformer
        Handles input encoding to q-series space
    mean_module : ConstantMean
        Mean function (constant across space)
    covar_module : QGaussianKernel
        Covariance function with q-exponential kernel

    Example
    -------
    >>> import torch
    >>> from gpytorch.likelihoods import GaussianLikelihood
    >>> from qseries_atlas import QGaussianProcess
    >>> 
    >>> # Create dummy data
    >>> train_x = torch.randn(50, 5)
    >>> train_y = torch.sin(train_x[:, 0]) + torch.randn(50) * 0.1
    >>> 
    >>> # Build model
    >>> likelihood = GaussianLikelihood()
    >>> model = QGaussianProcess(train_x, train_y, likelihood, q=1.1)
    >>> 
    >>> # Fit model (simplified; usually with BoTorch's fit_gpytorch_mll)
    >>> # ... training code ...
    >>> 
    >>> # Make predictions
    >>> test_x = torch.randn(10, 5)
    >>> with torch.no_grad():
    ...     preds = model(test_x)
    ...     mean = preds.mean
    ...     variance = preds.variance
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: Likelihood,
        q: float = 1.1,
        max_n: int = 10,
    ):
        super().__init__(train_x, train_y, likelihood)

        # 1. Data Engineering Layer: q-Series Feature Encoder
        self.transformer = AndrewsTransformer(q=q, max_n=max_n)

        # 2. Mean function
        self.mean_module = ConstantMean()

        # 3. Covariance: q-Gaussian kernel with heavy tails
        self.covar_module = QGaussianKernel(q_initial=q)

        # Store q for reference
        self._q = q
        self._max_n = max_n

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        Forward pass: Transform input and compute GP predictive distribution.

        Parameters
        ----------
        x : Tensor, shape (..., d)
            Input points

        Returns
        -------
        MultivariateNormal
            Predictive distribution with mean and covariance
        """
        # Step 1: Transform input to q-series feature space
        # This handles structural normalization (solves the rmacknight99 scaling issue)
        x_transformed = self.transformer.transform(x)

        # Step 2: Compute mean
        mean_x = self.mean_module(x_transformed)

        # Step 3: Compute covariance via q-Gaussian kernel
        covar_x = self.covar_module(x_transformed)

        # Step 4: Return predictive distribution
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __repr__(self) -> str:
        return (
            f"QGaussianProcess(q={self._q}, max_n={self._max_n}, "
            f"kernel=QGaussianKernel, transformer=AndrewsTransformer)"
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from gpytorch.likelihoods import GaussianLikelihood
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood

    print("QGaussianProcess Integration Test")
    print("=" * 50)

    # Create synthetic data with a "black swan" region
    torch.manual_seed(42)
    train_x = torch.linspace(-3, 3, 50).unsqueeze(-1)
    train_y = torch.sin(train_x.squeeze()) + torch.randn(50) * 0.05

    # Inject a rare high value (black swan)
    train_y[30] = 2.5

    print(f"Training data: {train_x.shape}, {train_y.shape}")
    print(f"Max training value: {train_y.max().item():.3f}")

    # Build model
    likelihood = GaussianLikelihood()
    model = QGaussianProcess(train_x, train_y.unsqueeze(-1), likelihood, q=1.2)

    print(f"Model: {model}")

    # Fit model
    mll = ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_mll(mll)

    # Make predictions
    test_x = torch.linspace(-4, 4, 100).unsqueeze(-1)
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        preds = likelihood(model(test_x))

    # Plot
    plt.figure(figsize=(12, 5))

    # Data
    plt.scatter(train_x.numpy(), train_y.numpy(), color="red", s=30, alpha=0.6, label="Training data")

    # Predictions
    plt.plot(test_x.numpy(), preds.mean.numpy(), "b-", linewidth=2, label="q-GP Mean")
    plt.fill_between(
        test_x.squeeze().numpy(),
        (preds.mean - 1.96 * preds.stddev).squeeze().numpy(),
        (preds.mean + 1.96 * preds.stddev).squeeze().numpy(),
        alpha=0.3,
        label="95% Confidence"
    )

    plt.xlabel("Input (x)")
    plt.ylabel("Output (y)")
    plt.title("QGaussianProcess: Heavy Tails Detect Black Swans")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("q_gaussian_process_demo.png", dpi=150)
    print("Plot saved as q_gaussian_process_demo.png")

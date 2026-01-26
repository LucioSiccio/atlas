# qgps.py
"""
qgps.py

Purpose
-------
Define GP surrogate models that use q-deformed kernels for Atlas / BoTorch workflows.

This file provides:
  - QGaussianExactGP: an ExactGP for regression with QGaussianKernel.
  - QGaussianSingleTaskGP: a BoTorch SingleTaskGP configured with QGaussianKernel.
  - build_continuous_kernel: a small factory to select Matérn vs q-kernel.
  - _example_train_and_predict: a minimal end-to-end example.

These components are intended to be plugged into existing Atlas model-building code
at the "covar_module construction" point (e.g., cont_kernel_factory).

Expected Inputs
---------------
QGaussianExactGP(train_x, train_y, q_init=1.2, likelihood=None)
  train_x : torch.Tensor (n, d)
  train_y : torch.Tensor (n, 1) or (n,)
  q_init  : float

QGaussianSingleTaskGP(train_X, train_Y, ...)
  train_X : torch.Tensor (n, d)
  train_Y : torch.Tensor (n, 1)

build_continuous_kernel(kind, ...)
  kind : str in {"matern52", "qgaussian"}

Expected Outputs
----------------
- Models that implement .forward(x) -> MultivariateNormal.
- BoTorch models support .posterior(x) and acquisition optimization.

Examples
--------
Example 1: ExactGP construction
Example 2: SingleTaskGP construction
Example 3: A/B kernel factory selection
Example 4: Fit + posterior mean prediction
Example 5: Use active_dims for mixed kernels

Possible Sources of Error
-------------------------
- train_y shape mismatch (n,) vs (n,1) handled but inconsistent usage can still bite.
- Default dtype inconsistencies (project uses torch.double).
- Forgetting model.train()/eval() mode when fitting / predicting.
- Passing non-floating tensors.
- Kernel selection string misspelled.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel

# Local q-kernel module. Adjust the import path to match your repo layout.
# If you place QGaussianKernel in atlas/gps/kernels/qgaussian.py, import from there.
from q-kernels import QGaussianKernel  # type: ignore


class QGaussianExactGP(gpytorch.models.ExactGP):
    """
    QGaussianExactGP

    Purpose
    -------
    Exact GP regression model using QGaussianKernel wrapped with ScaleKernel.

    Parameters
    ----------
    train_x : torch.Tensor
        Training inputs, shape (n, d).
    train_y : torch.Tensor
        Training targets, shape (n,) or (n,1).
    q_init : float
        Initial q value.
    likelihood : Likelihood or None
        Observation noise model. Defaults to GaussianLikelihood.

    Expected Outputs
    ----------------
    forward(x) returns MultivariateNormal(mean, covar).

    Possible Sources of Error
    -------------------------
    - train_x/train_y length mismatch.
    - train_y not squeezable to 1D (e.g., shape (n, k) with k>1).
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        q_init: float = 1.2,
        likelihood: Optional[Likelihood] = None,
    ):
        # Default to Gaussian observation noise if none provided.
        likelihood = likelihood or GaussianLikelihood()

        # ExactGP expects 1D target vector; squeeze final dimension if present.
        super().__init__(train_x, train_y.squeeze(-1), likelihood)

        # Constant mean is a stable baseline mean model.
        self.mean_module = gpytorch.means.ConstantMean()

        # q-kernel wrapped in ScaleKernel adds outputscale (signal variance).
        self.covar_module = ScaleKernel(
            base_kernel=QGaussianKernel(q_init=q_init),
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """
        Compute latent GP distribution at x.

        Parameters
        ----------
        x : torch.Tensor
            Inputs to evaluate.

        Returns
        -------
        MultivariateNormal
            Latent distribution.

        Possible Sources of Error
        -------------------------
        - x dtype mismatch with model parameters.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def build_continuous_kernel(
    *,
    kind: str,
    batch_shape: torch.Size = torch.Size(),
    ard_num_dims: Optional[int] = None,
    active_dims: Optional[Sequence[int]] = None,
    q_init: float = 1.2,
) -> gpytorch.kernels.Kernel:
    """
    Kernel factory for continuous variables.

    Purpose
    -------
    Centralize selection of continuous kernel used inside mixed-kernel models.

    Parameters
    ----------
    kind : str
        "matern52" or "qgaussian"
    batch_shape : torch.Size
        Batch shape used by batched GPs (optional).
    ard_num_dims : int or None
        ARD number of dims for lengthscale (optional).
    active_dims : sequence[int] or None
        Feature indices used by this kernel (for mixed feature vectors).
    q_init : float
        Only used if kind="qgaussian".

    Returns
    -------
    gpytorch.kernels.Kernel

    Examples
    --------
    build_continuous_kernel(kind="matern52", ard_num_dims=3)
    build_continuous_kernel(kind="qgaussian", q_init=1.5, active_dims=[0,1])

    Possible Sources of Error
    -------------------------
    - Unknown kernel kind string.
    """
    if kind == "matern52":
        return MaternKernel(
            nu=2.5,
            batch_shape=batch_shape,
            ard_num_dims=ard_num_dims,
            active_dims=active_dims,
            lengthscale_constraint=GreaterThan(1e-4),
        )

    if kind == "qgaussian":
        return QGaussianKernel(
            q_init=q_init,
            batch_shape=batch_shape,
            ard_num_dims=ard_num_dims,
            active_dims=active_dims,
        )

    raise ValueError(f"Unknown continuous kernel kind: {kind}")


class QGaussianSingleTaskGP(SingleTaskGP):
    """
    QGaussianSingleTaskGP

    Purpose
    -------
    BoTorch SingleTaskGP configured to use QGaussianKernel for covariance.

    Expected Inputs
    ---------------
    train_X : torch.Tensor (n, d)
    train_Y : torch.Tensor (n, 1)

    Expected Outputs
    ----------------
    Standard BoTorch model with .posterior(), compatible with acquisition functions.

    Possible Sources of Error
    -------------------------
    - Passing train_Y as (n,) instead of (n,1) can break some BoTorch transforms.
    - If you use input/output transforms elsewhere in Atlas, ensure consistency.
    """

    def __init__(self, train_X: torch.Tensor, train_Y: torch.Tensor, q_init: float = 1.2):
        # Use a constrained noise model typical for regression.
        min_noise = 1e-6 if train_X.dtype == torch.double else 1e-5
        likelihood = GaussianLikelihood(
            noise_constraint=GreaterThan(min_noise, transform=None, initial_value=1e-3),
        )

        # Build q-kernel + outputscale.
        covar_module = ScaleKernel(QGaussianKernel(q_init=q_init))

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            covar_module=covar_module,
            outcome_transform=None,
            input_transform=None,
        )


def _example_train_and_predict() -> None:
    """
    Minimal end-to-end example: fit q-GP and predict on test points.

    Expected Outputs
    ----------------
    Prints posterior mean and variance at test points.

    Possible Sources of Error
    -------------------------
    - Running without botorch/gpytorch installed.
    - Using CPU vs GPU changes performance but not correctness.
    """
    torch.manual_seed(0)

    # Example dataset: 1D regression.
    train_X = torch.linspace(0, 1, 12).unsqueeze(-1)
    train_Y = (torch.sin(8 * train_X) - (train_X - 0.7) ** 2) + 0.05 * torch.randn_like(train_X)

    # Example 1: BoTorch model construction.
    model = QGaussianSingleTaskGP(train_X, train_Y, q_init=1.2)

    # Put model in training mode for hyperparameter fitting.
    model.train()
    model.likelihood.train()

    # Fit hyperparameters via marginal likelihood.
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Switch to eval mode for posterior computations.
    model.eval()
    model.likelihood.eval()

    # Test points for prediction.
    test_X = torch.tensor([[0.15], [0.55], [0.9]], dtype=train_X.dtype)

    # Compute posterior.
    post = model.posterior(test_X)
    print("Posterior mean:", post.mean.squeeze(-1))
    print("Posterior var :", post.variance.squeeze(-1))

    # Example 3: A/B kernel selection via factory.
    k_m = build_continuous_kernel(kind="matern52", ard_num_dims=1)
    k_q = build_continuous_kernel(kind="qgaussian", ard_num_dims=1, q_init=1.6)
    print("Factory kernels:", type(k_m).__name__, type(k_q).__name__)

    # Example 5: active_dims in a mixed feature tensor (cont in dims 0..1).
    X_mixed = torch.tensor([[0.0, 1.0, 1.0], [1.0, 2.0, 0.0]], dtype=train_X.dtype)
    k_q2 = build_continuous_kernel(kind="qgaussian", ard_num_dims=2, active_dims=[0, 1], q_init=1.2)
    print("active_dims K shape:", k_q2(X_mixed, X_mixed).evaluate().shape)


if __name__ == "__main__":
    _example_train_and_predict()

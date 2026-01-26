"""
q_atlas/models/qseries_gp.py

Purpose
-------
Wrap QGaussianKernel in an ExactGP model compatible with BoTorch.

How this improves "original Atlas"
----------------------------------
Atlas typically fits a GP surrogate using standard kernels.
This model lets you swap in q-deformed covariance while keeping the rest
of the optimization stack the same (training, acquisition, recommend).

That means you can A/B test:
- baseline surrogate (Matérn / RBF)
vs
- q-surrogate (QGaussianKernel)

Expected Inputs
---------------
__init__(train_X, train_Y):
  train_X : torch.Tensor (n, d)
  train_Y : torch.Tensor (n, 1) or (n,)

Expected Outputs
----------------
forward(x) returns a gpytorch.distributions.MultivariateNormal

5 Examples
----------
Example 1: Construct model
Example 2: Forward pass
Example 3: Posterior mean/variance via model(x)
Example 4: Different q_init
Example 5: Use ScaleKernel amplitude effect

Possible Sources of Error
-------------------------
- train_X shape mismatch vs train_Y length
- train_Y passed as (n,) but squeezed incorrectly (handled)
- dtype mismatch (BoTorch prefers float)
- not calling model.train()/eval() appropriately in real workflows
"""

import gpytorch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood

from q_atlas.kernels.qgaussian import QGaussianKernel


class QSeriesGP(ExactGP, GPyTorchModel):
    """Single-output ExactGP with q-deformed covariance."""
    _num_outputs = 1

    def __init__(self, train_X, train_Y, q_init: float = 1.2):
        likelihood = GaussianLikelihood()
        super().__init__(train_X, train_Y.squeeze(-1), likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(QGaussianKernel(q_init=q_init))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# -------------------------
# Five runnable examples
# -------------------------
if __name__ == "__main__":
    import torch

    # Example 1: construct
    X = torch.linspace(0, 1, 8).unsqueeze(-1)
    Y = torch.sin(6 * X) + 0.05 * torch.randn_like(X)
    model = QSeriesGP(X, Y, q_init=1.2)
    print("Ex1 model built")

    # Example 2: forward
    out = model(X)
    print("Ex2 MVN", out.mean.shape)

    # Example 3: posterior mean at new points
    Xt = torch.tensor([[0.25], [0.75]])
    print("Ex3 mean", model(Xt).mean)

    # Example 4: different q_init
    model2 = QSeriesGP(X, Y, q_init=1.8)
    print("Ex4 q=1.8 mean", model2(Xt).mean)

    # Example 5: outputscale sensitivity
    model.covar_module.outputscale = 2.0
    print("Ex5 scaled var", model(Xt).variance)

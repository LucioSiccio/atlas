import gpytorch
import torch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.models import ExactGP

class QGaussianProcess(ExactGP, GPyTorchModel):
    def __init__(self, train_x, train_y, likelihood, q=1.1, max_n=10):
        super().__init__(train_x, train_y, likelihood)
        
        # 1. The Andrews Transformer (Data Engineering Layer)
        # This converts raw input into the q-series expansion basis
        self.transformer = AndrewsTransformer(q=q, max_n=max_n)
        
        # 2. Mean and Covariance
        self.mean_module = gpytorch.means.ConstantMean()
        
        # 3. The q-Gaussian Kernel (The Search Engine)
        # Replaces RBF/Matern with the q-exponential for heavy-tail awareness
        self.covar_module = QGaussianKernel(q_init=q)

    def forward(self, x):
        # Apply q-series transformation to incoming data x
        # This solves the scaling issue noted in the genetic optimizer
        x_q = self.transformer.transform(x)
        
        mean_x = self.mean_module(x_q)
        covar_x = self.covar_module(x_q)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
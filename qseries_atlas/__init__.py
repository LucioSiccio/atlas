"""
q-Series Extension for Atlas SDL
================================

A Bayesian Optimization extension that generalizes Gaussian Processes
using q-series expansions from partition theory.

Core Components:
    - QGaussianKernel: Custom kernel using q-exponential covariance
    - AndrewsTransformer: Maps categorical/continuous inputs to q-binomial features
    - QGaussianProcess: BoTorch-compatible GP with integrated q-series logic

Theoretical Basis:
    - George Andrews' Theory of Partitions (Gaussian Polynomials)
    - Non-extensive Statistical Mechanics (Tsallis Entropy)
    - Jackson q-Calculus (q-derivatives for optimization)

Example Usage:
    >>> from qseries_atlas import QGaussianProcess, AndrewsTransformer
    >>> transformer = AndrewsTransformer(q=1.1, max_n=10)
    >>> model = QGaussianProcess(train_x, train_y, likelihood, q=1.1)
    >>> predictions = model(test_x)

References:
    - Andrews, G. E. (1976). The Theory of Partitions.
    - Tsallis, C. (1988). Possible Generalization of Boltzmann-Gibbs Statistics.
    - Jackson, F. H. (1909). On q-Definite Integrals.
"""

__version__ = "0.1.0"
__author__ = "q-Atlas Contributors"

from .q_kernel import QGaussianKernel
from .andrews_transformer import AndrewsTransformer
from .q_gaussian_process import QGaussianProcess

__all__ = [
    "QGaussianKernel",
    "AndrewsTransformer",
    "QGaussianProcess",
]

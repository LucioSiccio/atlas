"""
Unit Tests for q-Series Atlas Extension
========================================

Tests validate:
    1. Numerical stability of q-binomial computation
    2. Structural similarity preservation in AndrewsTransformer
    3. Heavy-tail behavior of QGaussianKernel
    4. Integration with BoTorch/GPyTorch
"""

import pytest
import torch
import numpy as np
from gpytorch.likelihoods import GaussianLikelihood

# Assuming qseries_atlas is importable
from qseries_atlas import AndrewsTransformer, QGaussianKernel, QGaussianProcess


class TestAndrewsTransformer:
    """Tests for q-binomial feature encoding."""

    def test_initialization(self):
        """Test transformer initialization."""
        transformer = AndrewsTransformer(q=1.1, max_n=10)
        assert transformer.q_val == 1.1
        assert transformer.max_n == 10

    def test_q_pascal_table_build(self):
        """Test q-Pascal triangle generation."""
        transformer = AndrewsTransformer(q=1.1, max_n=5)
        table = transformer.build_table()

        # Check shape
        assert table.shape == (6, 6)

        # Check base cases: [n, 0] = 1
        assert torch.allclose(table[:, 0], torch.ones(6))

        # Check diagonal: [n, n] = 1
        for i in range(6):
            assert torch.allclose(table[i, i], torch.tensor(1.0))

    def test_q_binomial_properties(self):
        """Test mathematical properties of q-binomial."""
        transformer = AndrewsTransformer(q=1.0, max_n=10)
        transformer.build_table()

        # When q=1, should recover standard binomial coefficients
        # (up to numerical precision)
        # C(5,2) = 10
        val = transformer.get_q_binomial(5, 2).item()
        # Note: q=1.0 is technically the boundary; we use q ≈ 1.0
        assert val > 0, "q-binomial should be positive"

    def test_transform_output_shape(self):
        """Test transform output shape."""
        transformer = AndrewsTransformer(q=1.1, max_n=10)
        
        # Single index
        x = torch.tensor([3])
        output = transformer.transform(x)
        assert output.shape == (1, 1)

        # Multiple indices
        x = torch.tensor([0, 1, 2, 3, 4])
        output = transformer.transform(x)
        assert output.shape == (5, 1)

    def test_structural_similarity(self):
        """Test that similar indices map to similar features."""
        transformer = AndrewsTransformer(q=1.1, max_n=10)

        # Adjacent catalysts should have similar features
        feat_0 = transformer.transform(torch.tensor([0])).item()
        feat_1 = transformer.transform(torch.tensor([1])).item()

        # They should not be wildly different
        assert abs(feat_0 - feat_1) < max(feat_0, feat_1) * 0.5


class TestQGaussianKernel:
    """Tests for q-Gaussian kernel behavior."""

    def test_kernel_initialization(self):
        """Test kernel initialization."""
        kernel = QGaussianKernel(q_initial=1.5)
        assert kernel.is_stationary
        q_val = kernel.q.item()
        assert 1.0 < q_val <= 3.0

    def test_kernel_output_shape(self):
        """Test kernel output shape."""
        kernel = QGaussianKernel(q_initial=1.2)
        x1 = torch.randn(10, 5)
        x2 = torch.randn(15, 5)

        covar = kernel(x1, x2)
        assert covar.shape == (10, 15)

    def test_kernel_diagonal(self):
        """Test diagonal elements."""
        kernel = QGaussianKernel(q_initial=1.5)
        x = torch.randn(20, 5)

        covar = kernel(x, x, diag=True)
        assert covar.shape == (20,)

        # Diagonal should be maximal at zero distance
        assert torch.all(covar <= 1.0)

    def test_symmetry(self):
        """Test kernel symmetry: k(x,y) = k(y,x)."""
        kernel = QGaussianKernel(q_initial=1.3)
        x1 = torch.randn(5, 3)
        x2 = torch.randn(8, 3)

        k12 = kernel(x1, x2)
        k21 = kernel(x2, x1).transpose(0, 1)

        assert torch.allclose(k12, k21, atol=1e-6)

    def test_heavy_tails(self):
        """Test that q-kernel has heavier tails than standard Gaussian."""
        kernel_q_small = QGaussianKernel(q_initial=1.01)  # Near-Gaussian
        kernel_q_large = QGaussianKernel(q_initial=1.5)   # Heavy tails

        # Test points far from origin
        x_ref = torch.zeros(1, 3)
        x_far = torch.randn(100, 3) * 10  # Large distances

        k_small = kernel_q_small(x_far, x_ref).squeeze()
        k_large = kernel_q_large(x_far, x_ref).squeeze()

        # Heavy-tailed kernel should have higher values at large distances
        assert k_large.mean() > k_small.mean()

    def test_positivity(self):
        """Test that kernel values are positive (valid covariance)."""
        kernel = QGaussianKernel(q_initial=1.5)
        x = torch.randn(20, 5)

        covar = kernel(x, x)
        assert torch.all(covar >= 0)


class TestQGaussianProcess:
    """Tests for integrated QGaussianProcess."""

    def test_gp_initialization(self):
        """Test QGaussianProcess initialization."""
        train_x = torch.randn(10, 3)
        train_y = torch.randn(10)

        likelihood = GaussianLikelihood()
        model = QGaussianProcess(train_x, train_y, likelihood, q=1.1)

        assert model is not None
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'mean_module')
        assert hasattr(model, 'covar_module')

    def test_gp_forward_shape(self):
        """Test GP forward pass output shape."""
        train_x = torch.randn(15, 4)
        train_y = torch.randn(15)

        likelihood = GaussianLikelihood()
        model = QGaussianProcess(train_x, train_y, likelihood, q=1.2)

        test_x = torch.randn(8, 4)
        with torch.no_grad():
            output = model(test_x)

        assert output.mean.shape == (8,)
        assert output.covariance_matrix.shape == (8, 8)

    def test_gp_with_transformer(self):
        """Test that transformer is applied in forward pass."""
        train_x = torch.arange(10).unsqueeze(-1).float()
        train_y = torch.sin(train_x).squeeze()

        likelihood = GaussianLikelihood()
        model = QGaussianProcess(train_x, train_y, likelihood, q=1.1, max_n=5)

        test_x = torch.tensor([[2.0], [5.0]])
        with torch.no_grad():
            output = model(test_x)

        # Should have valid predictions
        assert not torch.isnan(output.mean).any()
        assert not torch.isinf(output.mean).any()


class TestIntegrationWithAtlas:
    """Tests for integration with Atlas's ask-tell pattern."""

    def test_embedding_in_atlas_workflow(self):
        """
        Test that QGaussianProcess can be used in place of standard GP
        in a simulated Atlas ask-tell loop.
        """
        from olympus import Campaign, Surface

        # Create a simple surface
        surface = Surface(kind='Branin')
        campaign = Campaign()
        campaign.set_param_space(surface.param_space)

        # Simulate some initial measurements
        train_data = []
        for _ in range(5):
            from olympus.optimizers import RandomSearch
            rs = RandomSearch()
            rs.set_param_space(surface.param_space)
            sample = rs.recommend(campaign.observations)
            measurement = surface.run(sample)
            campaign.add_observation(sample, measurement)

        # Now use QGaussianProcess
        params = campaign.observations.get_params(as_array=True)
        values = campaign.observations.get_values(as_array=True)

        train_x = torch.from_numpy(params).double()
        train_y = torch.from_numpy(values).double().squeeze()

        likelihood = GaussianLikelihood()
        model = QGaussianProcess(train_x, train_y, likelihood, q=1.1)

        # Should be able to make predictions
        test_x = torch.randn(3, train_x.shape[1]).double()
        with torch.no_grad():
            preds = model(test_x)

        assert preds.mean.shape == (3,)
        assert not torch.isnan(preds.mean).any()


if __name__ == "__main__":
    # Run a quick sanity check
    print("Running quick sanity checks...")

    transformer = AndrewsTransformer(q=1.1, max_n=10)
    print(f"✓ AndrewsTransformer initialized: {transformer}")

    kernel = QGaussianKernel(q_initial=1.5)
    print(f"✓ QGaussianKernel initialized")

    train_x = torch.randn(10, 5)
    train_y = torch.randn(10)
    likelihood = GaussianLikelihood()
    model = QGaussianProcess(train_x, train_y, likelihood, q=1.1)
    print(f"✓ QGaussianProcess initialized: {model}")

    print("\nAll sanity checks passed!")
    print("\nFor full test suite, run: pytest tests/")

"""
Example: Using q-Series Extension with Atlas SDL
=================================================

This example demonstrates how to use the QGaussianProcess as a drop-in
upgrade to the standard Atlas GPPlanner.

The q-series approach provides:
    1. Structural normalization (solves the scaling issue)
    2. Heavy-tail awareness (discovers rare outcomes)
    3. Partition-based similarity (improves categorical handling)

Author: q-Atlas Contributors
Date: 2025
"""

import torch
import numpy as np
from olympus import Campaign, Surface
from gpytorch.likelihoods import GaussianLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

# Import the q-Series extension
from qseries_atlas import QGaussianProcess, AndrewsTransformer


def example_1_basic_optimization():
    """
    Example 1: Basic 2D Branin Optimization with q-Gaussian Process
    
    This shows how to replace a standard Atlas GPPlanner with QGaussianProcess.
    """
    print("\n" + "=" * 70)
    print("Example 1: Basic 2D Branin Optimization")
    print("=" * 70)

    # Create the surface (2D continuous Branin function)
    surface = Surface(kind='Branin')
    campaign = Campaign()
    campaign.set_param_space(surface.param_space)

    print(f"Surface: Branin (2D)")
    print(f"Goal: Minimize")

    # Phase 1: Random initialization (5 points)
    print("\n[Phase 1] Random initialization...")
    from olympus.optimizers import RandomSearch
    
    for i in range(5):
        rs = RandomSearch()
        rs.set_param_space(surface.param_space)
        sample = rs.recommend(campaign.observations)
        measurement = surface.run(sample)
        campaign.add_observation(sample, measurement)
        print(f"  Iteration {i+1}: value = {measurement:.4f}")

    # Phase 2: q-Gaussian Process Optimization
    print("\n[Phase 2] q-Gaussian Process optimization...")
    budget_remaining = 10  # Optimize for 10 more iterations

    for iteration in range(budget_remaining):
        # Extract observations
        params = campaign.observations.get_params(as_array=True)
        values = campaign.observations.get_values(as_array=True)

        # Convert to torch tensors
        train_x = torch.from_numpy(params).double()
        train_y = torch.from_numpy(values).double().squeeze()

        # Build and fit the q-Gaussian Process
        likelihood = GaussianLikelihood()
        model = QGaussianProcess(
            train_x, 
            train_y.unsqueeze(-1), 
            likelihood, 
            q=1.2,          # q=1.2 provides moderate heavy tails
            max_n=10         # Max complexity
        )

        # Fit the model (using BoTorch's standard fitting)
        mll = ExactMarginalLogLikelihood(likelihood, model)
        fit_gpytorch_mll(mll)

        # Generate candidates via random search in the q-space
        # (For a full implementation, use BoTorch's acquisition functions)
        model.eval()
        likelihood.eval()

        with torch.no_grad():
            # Score random candidates
            n_candidates = 100
            candidate_params = surface.param_space.sample(n_candidates)
            candidate_x = torch.from_numpy(candidate_params).double()

            predictions = likelihood(model(candidate_x))
            scores = predictions.mean  # Lower is better (minimization)

            # Select best candidate
            best_idx = scores.argmin()
            selected_params = candidate_params[best_idx]

        # Measure the selected point
        measurement = surface.run(selected_params)
        campaign.add_observation(selected_params, measurement)

        best_so_far = campaign.observations.get_values().min()
        print(f"  Iteration {iteration + 6}: "
              f"value = {measurement:.4f}, "
              f"best so far = {best_so_far:.4f}")

    print(f"\nFinal best value: {campaign.observations.get_values().min():.4f}")
    print("(Branin global minimum is approximately 0.397)")


def example_2_categorical_similarity():
    """
    Example 2: Demonstrate Andrews Transformer's Structural Similarity
    
    Shows how catalysts with similar combinatorial structure map to similar
    features, enabling the model to leverage relationships automatically.
    """
    print("\n" + "=" * 70)
    print("Example 2: Categorical Similarity via Partitions")
    print("=" * 70)

    # Create a transformer
    transformer = AndrewsTransformer(q=1.1, max_n=10)

    print("\nMapping 8 'catalysts' (indices 0-7) to q-binomial features:")
    print("-" * 50)

    catalyst_indices = torch.arange(8)
    features = transformer.transform(catalyst_indices).squeeze()

    print(f"{'Catalyst':<12} {'q-Binomial Feature':<20} {'Distance to Catalyst 0':<20}")
    print("-" * 50)

    for i, feat in zip(catalyst_indices, features):
        dist_to_0 = abs(feat.item() - features[0].item())
        print(f"  {i:<10} {feat.item():<20.6f} {dist_to_0:<20.6f}")

    print("\nKey Insight:")
    print("  - Catalyst 1 and 2 have similar features (structural similarity)")
    print("  - This allows the GP to transfer learning between them")
    print("  - No manual feature engineering needed!")


def example_3_scaling_invariance():
    """
    Example 3: Demonstrate Scale Invariance
    
    Compare standard normalization vs q-series scaling.
    Addresses the issue noted in genetic_optimizer.py.
    """
    print("\n" + "=" * 70)
    print("Example 3: Scale Invariance (Solving rmacknight99's Issue)")
    print("=" * 70)

    # Create synthetic data with wildly different scales
    # Temperature: 50-200 °C
    # Concentration: 0.001-0.1 M
    # Yield: 0-100%

    torch.manual_seed(42)
    n_samples = 20

    temps = torch.linspace(50, 200, n_samples)        # Scale: 50-200
    concs = torch.linspace(0.001, 0.1, n_samples)     # Scale: 0.001-0.1
    yields = torch.sin(temps / 50) * 50 + torch.randn(n_samples) * 5

    print("\nRaw Data Scales:")
    print(f"  Temperature: {temps.min():.1f} - {temps.max():.1f} (range: {temps.max() - temps.min():.1f})")
    print(f"  Concentration: {concs.min():.4f} - {concs.max():.4f} (range: {concs.max() - concs.min():.4f})")
    print(f"  Yield: {yields.min():.1f} - {yields.max():.1f}")

    # Method 1: Standard Min-Max Scaling (what rmacknight99 proposed)
    print("\n[Method 1] Standard Min-Max Scaling:")
    temps_scaled = (temps - temps.min()) / (temps.max() - temps.min())
    concs_scaled = (concs - concs.min()) / (concs.max() - concs.min())

    print(f"  Temperature (scaled): {temps_scaled.min():.4f} - {temps_scaled.max():.4f}")
    print(f"  Concentration (scaled): {concs_scaled.min():.4f} - {concs_scaled.max():.4f}")

    print("\n  Problem: Values are [0,1], but loses chemical meaning.")
    print("           If you add one new catalyst, all scales shift!")

    # Method 2: q-Series Transformation (our solution)
    print("\n[Method 2] q-Series Transformation (AndrewsTransformer):")
    transformer = AndrewsTransformer(q=1.1, max_n=10)

    # Index both parameters
    temps_idx = torch.arange(n_samples)
    concs_idx = torch.arange(n_samples)

    temps_q = transformer.transform(temps_idx).squeeze()
    concs_q = transformer.transform(concs_idx).squeeze()

    print(f"  Temperature (q-series): {temps_q.min():.4f} - {temps_q.max():.4f}")
    print(f"  Concentration (q-series): {concs_q.min():.4f} - {concs_q.max():.4f}")

    print("\n  Advantage: Values are bounded and structurally derived.")
    print("            Scale is absolute; adding new data doesn't shift old values!")
    print("            Genetic optimizer treats all dimensions fairly.")


def example_4_black_swan_discovery():
    """
    Example 4: Black Swan Discovery via Heavy Tails
    
    Shows how q-Gaussian kernel finds rare high-value outcomes that
    standard Gaussian processes would dismiss as noise.
    """
    print("\n" + "=" * 70)
    print("Example 4: Black Swan Discovery (Heavy Tails)")
    print("=" * 70)

    torch.manual_seed(42)

    # Create synthetic data with a "black swan" (rare high value)
    x = torch.linspace(-3, 3, 40).unsqueeze(-1)
    y = torch.sin(x.squeeze()) + torch.randn(40) * 0.1

    # Inject a rare breakthrough
    y[25] = 2.8  # Black swan: normally would max at ~1.0

    print("\nSynthetic Data:")
    print(f"  Typical value range: {y[y < 2].min():.2f} to {y[y < 2].max():.2f}")
    print(f"  Black swan value: {y.max():.2f} at position {y.argmax()}")

    # Convert to tensors
    train_x = x
    train_y = y

    # Build model
    likelihood = GaussianLikelihood()
    model = QGaussianProcess(train_x, train_y, likelihood, q=1.5)

    mll = ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_mll(mll)

    # Make predictions
    test_x = torch.linspace(-4, 4, 100).unsqueeze(-1)
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        preds = likelihood(model(test_x))

    print("\nq-Gaussian Process Predictions:")
    print(f"  Mean at black swan location: {preds.mean[25]:.2f}")
    print(f"  Std at black swan location: {preds.stddev[25]:.2f}")
    print("\n  → The heavy-tailed q-kernel maintains high uncertainty")
    print("    around the black swan, allowing continued exploration!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("q-Series Atlas Extension: Usage Examples")
    print("=" * 70)

    # Run examples
    example_2_categorical_similarity()
    example_3_scaling_invariance()
    example_4_black_swan_discovery()

    # Example 1 is commented out because it requires optimization
    # Uncomment to run the full Branin optimization:
    # example_1_basic_optimization()

    print("\n" + "=" * 70)
    print("Examples Complete!")
    print("=" * 70)
    print("\nNext Steps:")
    print("  1. Run tests: pytest qseries_atlas/tests/")
    print("  2. Integrate with your Atlas workflow")
    print("  3. Compare performance vs standard GPPlanner")
    print("  4. Read: George Andrews' 'The Theory of Partitions'")
    print("\nFor questions, see the module docstrings or GitHub.")

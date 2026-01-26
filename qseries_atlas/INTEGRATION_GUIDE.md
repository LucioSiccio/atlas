"""
Quick Integration Guide: q-Series Atlas Extension
==================================================

This guide shows how to integrate the qseries_atlas module into your existing
Atlas workflow without modifying any core Atlas code.

Approach: Hybrid (Non-invasive Extension)
"""

# ============================================================================
# Option 1: Drop-in Replacement for SingleTaskGP
# ============================================================================
# Use this when you want to try q-Gaussian in place of the standard GP

from olympus import Campaign, Surface
from qseries_atlas import QGaussianProcess
from gpytorch.likelihoods import GaussianLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch

def use_q_gaussian_in_atlas():
    """
    Replace standard GP with QGaussianProcess.
    Everything else in your Atlas workflow stays the same.
    """
    # Your existing code...
    surface = Surface(kind='Branin')
    campaign = Campaign()
    campaign.set_param_space(surface.param_space)
    
    # ... run initial design with Atlas as usual ...
    
    # When you need to fit the surrogate model:
    params = campaign.observations.get_params(as_array=True)
    values = campaign.observations.get_values(as_array=True)
    
    train_x = torch.from_numpy(params).double()
    train_y = torch.from_numpy(values).double().squeeze()
    
    # Instead of:
    #   model = SingleTaskGP(train_x, train_y)
    # Use:
    likelihood = GaussianLikelihood()
    model = QGaussianProcess(
        train_x, 
        train_y, 
        likelihood,
        q=1.1,      # Moderate heavy tails
        max_n=10    # Partition complexity
    )
    
    # Fit as usual
    mll = ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_mll(mll)
    
    # Now use with BoTorch acquisition functions exactly as before
    # No changes needed to the acquisition function code!


# ============================================================================
# Option 2: Structured Feature Encoding (Solving rmacknight99's Issue)
# ============================================================================
# Use this to solve the scaling problem noted in genetic_optimizer.py

from atlas.acquisition_optimizers.genetic_optimizer import GeneticOptimizer
from qseries_atlas import AndrewsTransformer

def improved_genetic_optimizer_with_q_transform():
    """
    The issue: genetic_optimizer.py doesn't scale inputs, so Temperature
    (100-200) is treated as 1000x more important than Concentration (0.001-0.1).
    
    The solution: Use AndrewsTransformer to project both into q-space,
    where they're structurally comparable.
    """
    # Initialize transformer once
    transformer = AndrewsTransformer(q=1.1, max_n=10)
    
    # In your genetic optimizer acquisition method, before passing to acqf:
    def improved_acquisition(self, x_raw):
        """
        Original issue: x is not scaled
        
        Improved version: Use AndrewsTransformer
        """
        # 1. Apply structural transformation
        x_transformed = transformer.transform(torch.from_numpy(x_raw))
        
        # 2. Now all dimensions are on the same "q-manifold"
        #    No need for manual min/max normalization!
        
        # 3. Pass to acquisition function
        acqf_value = self.acqf(x_transformed)
        
        return acqf_value
    
    # The beauty: This works for ANY optimizer, not just genetic


# ============================================================================
# Option 3: Comparison Benchmark (Prove q-Series is Better)
# ============================================================================
# Use this to validate the improvement in your specific use case

import matplotlib.pyplot as plt
import numpy as np

def benchmark_q_series_vs_standard():
    """
    Compare standard GP vs q-Gaussian GP on your own experimental data.
    """
    from botorch.models import SingleTaskGP
    
    # Load your campaign data
    params = ...  # Your experimental parameters
    values = ...  # Your experimental results
    
    train_x = torch.from_numpy(params).double()
    train_y = torch.from_numpy(values).double()
    
    test_x = ...  # Test points
    
    # Model 1: Standard GP
    model_std = SingleTaskGP(train_x, train_y.unsqueeze(-1))
    with torch.no_grad():
        preds_std = model_std(test_x)
    
    # Model 2: q-Gaussian GP
    likelihood = GaussianLikelihood()
    model_q = QGaussianProcess(train_x, train_y.unsqueeze(-1), likelihood, q=1.2)
    # (fit model_q here)
    with torch.no_grad():
        preds_q = likelihood(model_q(test_x))
    
    # Compare:
    # 1. Predictive uncertainty at rare high-value points
    # 2. Behavior at parameter boundaries
    # 3. Acquisition function values
    
    metrics = {
        'std_mean': preds_std.mean.std().item(),
        'std_var': preds_std.variance.mean().item(),
        'q_mean': preds_q.mean.std().item(),
        'q_var': preds_q.variance.mean().item(),
    }
    
    print(metrics)
    return metrics


# ============================================================================
# Option 4: Minimal Reproducible Example
# ============================================================================
# Use this to test the setup works in your environment

def minimal_example():
    """Tiny example that should run in <1 second"""
    import torch
    from gpytorch.likelihoods import GaussianLikelihood
    from qseries_atlas import QGaussianProcess, AndrewsTransformer
    
    print("Testing qseries_atlas installation...")
    
    # 1. Test transformer
    transformer = AndrewsTransformer(q=1.1, max_n=5)
    x = torch.tensor([0, 1, 2])
    features = transformer.transform(x)
    print(f"✓ AndrewsTransformer works: {features.shape}")
    
    # 2. Test GP
    train_x = torch.randn(10, 3)
    train_y = torch.randn(10)
    
    likelihood = GaussianLikelihood()
    model = QGaussianProcess(train_x, train_y, likelihood)
    print(f"✓ QGaussianProcess initialized: {model}")
    
    # 3. Test prediction
    test_x = torch.randn(5, 3)
    with torch.no_grad():
        preds = model(test_x)
    print(f"✓ Predictions work: mean shape = {preds.mean.shape}")
    
    print("\n✓✓✓ All systems nominal! ✓✓✓")


# ============================================================================
# Common Integration Patterns
# ============================================================================

# Pattern 1: A/B Testing (Run both models, compare)
def pattern_ab_testing():
    """
    Keep standard Atlas unchanged, run parallel q-series in test mode
    """
    # Your normal Atlas workflow
    # ... use standard GPPlanner ...
    
    # Parallel: log data and run q-Gaussian GP offline
    # Compare performance metrics later
    pass


# Pattern 2: Feature Engineering (Pre-process data)
def pattern_feature_engineering():
    """
    Use AndrewsTransformer as a pre-processing step
    """
    # Before feeding data to any model:
    transformer = AndrewsTransformer(q=1.1, max_n=10)
    
    # Transform categorical indices
    categorical_features = transformer.transform(category_indices)
    
    # Combine with continuous features
    continuous_features = continuous_params  # your numerical params
    
    # Concatenate
    all_features = torch.cat([categorical_features, continuous_features], dim=1)
    
    # Now use all_features with standard Atlas
    # Much better scaling!


# Pattern 3: Gradual Adoption (Hybrid)
def pattern_gradual_adoption():
    """
    Use standard Atlas for initial design
    Switch to q-Gaussian after sufficient data
    """
    # Phase 1: Standard Atlas (what you're doing now)
    planner = GPPlanner(goal='minimize', num_init_design=5)
    # ... run initial design ...
    
    # Phase 2: Once you have ~20-30 observations
    if len(campaign.observations.get_values()) > 20:
        # Switch to q-Gaussian
        params = campaign.observations.get_params(as_array=True)
        values = campaign.observations.get_values(as_array=True)
        
        train_x = torch.from_numpy(params).double()
        train_y = torch.from_numpy(values).double()
        
        likelihood = GaussianLikelihood()
        model = QGaussianProcess(train_x, train_y, likelihood, q=1.2)
        
        # Use q-Gaussian for remaining iterations
        # Compare final results to standard Atlas


# ============================================================================
# Troubleshooting
# ============================================================================

TROUBLESHOOTING = """
Q: How do I know which q value to use?
A: Start with q=1.1. If you notice the model is too conservative (low uncertainty),
   increase to q=1.3 or q=1.5. If too aggressive, decrease toward 1.01.

Q: Can I use this with categorical + continuous mixed spaces?
A: Yes. Use AndrewsTransformer for categorical, keep continuous as-is, concatenate.

Q: Will this work with my existing Atlas code without modifications?
A: Yes (Approach 3). Just replace the surrogate model, keep everything else.

Q: Is this production-ready?
A: It's research-grade. Thoroughly test on your use case before deploying.

Q: How do I cite this work?
A: Include references to Andrews (1976), Tsallis (1988), and Jackson (1909) papers.
   See README.md for full citations.

Q: Can I combine this with Atlas's other features (constraints, MOO, etc.)?
A: Yes. QGaussianProcess is a drop-in replacement for the surrogate GP only.
   All upper-level Atlas logic (constraints, acquisition functions) remains unchanged.
"""

# ============================================================================
# Next Steps
# ============================================================================

NEXT_STEPS = """
1. Run minimal_example() above to verify installation

2. Read the documentation:
   - qseries_atlas/README.md (overview)
   - .github/copilot-instructions.md (architecture)
   - George Andrews' "The Theory of Partitions" (math foundation)

3. Try one of the four options above:
   Option 1: Direct replacement (easiest)
   Option 2: Solve scaling issue (most impactful)
   Option 3: Benchmark (most rigorous)
   Option 4: Minimal example (quickest verification)

4. Run tests to ensure everything works:
   pytest qseries_atlas/tests/

5. Experiment with your own data
   Start with q=1.1, tune based on results

6. Compare against standard Atlas using the same data
   Document performance differences

7. Consider contributing improvements back if you find good patterns
"""

if __name__ == "__main__":
    print("q-Series Atlas Integration Guide")
    print("=" * 70)
    print("\nRun minimal_example() to verify installation:")
    minimal_example()
    print("\nNext steps:")
    print(NEXT_STEPS)

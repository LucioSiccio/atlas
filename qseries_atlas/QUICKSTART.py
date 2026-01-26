#!/usr/bin/env python3
"""
QUICK START CHECKLIST: q-Series Atlas Extension
================================================

Copy-paste this to verify everything is working.
Expected runtime: < 1 minute
"""

import sys
import torch

def check_imports():
    """Verify all imports work"""
    print("1. Checking imports...", end=" ")
    try:
        from qseries_atlas import QGaussianKernel, AndrewsTransformer, QGaussianProcess
        print("✓")
        return True
    except ImportError as e:
        print(f"✗ {e}")
        return False


def check_transformer():
    """Verify AndrewsTransformer works"""
    print("2. Testing AndrewsTransformer...", end=" ")
    try:
        from qseries_atlas import AndrewsTransformer
        transformer = AndrewsTransformer(q=1.1, max_n=10)
        x = torch.tensor([0, 1, 2, 3])
        features = transformer.transform(x)
        assert features.shape == (4, 1), f"Shape mismatch: {features.shape}"
        assert not torch.isnan(features).any(), "NaN detected"
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def check_kernel():
    """Verify QGaussianKernel works"""
    print("3. Testing QGaussianKernel...", end=" ")
    try:
        from qseries_atlas import QGaussianKernel
        kernel = QGaussianKernel(q_initial=1.2)
        x1 = torch.randn(5, 3)
        x2 = torch.randn(7, 3)
        covar = kernel(x1, x2)
        assert covar.shape == (5, 7), f"Shape mismatch: {covar.shape}"
        assert (covar >= 0).all(), "Negative kernel values detected"
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def check_gp():
    """Verify QGaussianProcess works"""
    print("4. Testing QGaussianProcess...", end=" ")
    try:
        from qseries_atlas import QGaussianProcess
        from gpytorch.likelihoods import GaussianLikelihood
        
        train_x = torch.randn(10, 4)
        train_y = torch.randn(10)
        
        likelihood = GaussianLikelihood()
        model = QGaussianProcess(train_x, train_y, likelihood, q=1.1)
        
        test_x = torch.randn(5, 4)
        with torch.no_grad():
            preds = model(test_x)
        
        assert preds.mean.shape == (5,), f"Mean shape: {preds.mean.shape}"
        assert not torch.isnan(preds.mean).any(), "NaN in predictions"
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def check_integration():
    """Verify Atlas integration"""
    print("5. Testing Atlas integration...", end=" ")
    try:
        from olympus import Campaign, Surface
        from qseries_atlas import QGaussianProcess
        from gpytorch.likelihoods import GaussianLikelihood
        
        surface = Surface(kind='Branin')
        campaign = Campaign()
        campaign.set_param_space(surface.param_space)
        
        # Dummy observations
        for _ in range(3):
            from olympus.optimizers import RandomSearch
            rs = RandomSearch()
            rs.set_param_space(surface.param_space)
            sample = rs.recommend(campaign.observations)
            measurement = surface.run(sample)
            campaign.add_observation(sample, measurement)
        
        # Use q-Gaussian
        params = campaign.observations.get_params(as_array=True)
        values = campaign.observations.get_values(as_array=True)
        
        train_x = torch.from_numpy(params).double()
        train_y = torch.from_numpy(values).double().squeeze()
        
        likelihood = GaussianLikelihood()
        model = QGaussianProcess(train_x, train_y, likelihood, q=1.1)
        
        test_x = torch.randn(2, train_x.shape[1]).double()
        with torch.no_grad():
            preds = model(test_x)
        
        assert preds.mean.shape == (2,), f"Prediction shape: {preds.mean.shape}"
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks"""
    print("\n" + "=" * 60)
    print("q-SERIES ATLAS EXTENSION: QUICK START CHECKLIST")
    print("=" * 60)
    print()
    
    checks = [
        check_imports,
        check_transformer,
        check_kernel,
        check_gp,
        check_integration,
    ]
    
    results = [check() for check in checks]
    
    print()
    print("=" * 60)
    if all(results):
        print("✓✓✓ ALL CHECKS PASSED ✓✓✓")
        print("\nYou're ready to use q-series atlas!")
        print("\nNext steps:")
        print("  1. Read qseries_atlas/README.md")
        print("  2. Check qseries_atlas/INTEGRATION_GUIDE.md")
        print("  3. Run: python qseries_atlas/examples.py")
        print("  4. Run: pytest qseries_atlas/tests/")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nTroubleshooting:")
        print("  1. Ensure qseries_atlas/ is in your Python path")
        print("  2. Check all dependencies are installed:")
        print("     - torch, gpytorch, botorch, olympus")
        print("  3. Run from the Atlas repo root directory")
        return 1


if __name__ == "__main__":
    sys.exit(main())

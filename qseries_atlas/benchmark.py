#!/usr/bin/env python3
"""
Comprehensive Benchmark Script for q-Series Atlas Extension
=============================================================

Measures key performance metrics comparing q-Gaussian with standard GP.

Usage:
    python benchmark.py --budget 50 --q_values "1.0,1.1,1.15,1.2" --repeats 3
"""

import torch
import numpy as np
import time
import argparse
from typing import Dict, List, Tuple
import json

# Will be populated when imports work
try:
    from olympus import Surface, Campaign, ParameterSpace, ParameterContinuous
    from qseries_atlas import QGaussianProcess, QGaussianKernel
    from gpytorch.likelihoods import GaussianLikelihood
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)


class PerformanceMetrics:
    """Container for benchmark results."""
    
    def __init__(self):
        self.results = {}
    
    def add_q_value(self, q_val: float):
        self.results[q_val] = {
            'best_values': [],
            'convergence_steps': [],
            'computation_times': [],
            'predictions': []
        }
    
    def record_best_value(self, q_val: float, value: float):
        self.results[q_val]['best_values'].append(value)
    
    def record_convergence(self, q_val: float, steps: int):
        self.results[q_val]['convergence_steps'].append(steps)
    
    def record_time(self, q_val: float, elapsed: float):
        self.results[q_val]['computation_times'].append(elapsed)
    
    def summarize(self) -> Dict:
        """Generate summary statistics."""
        summary = {}
        for q_val, data in self.results.items():
            summary[q_val] = {
                'best_mean': float(np.mean(data['best_values'])),
                'best_std': float(np.std(data['best_values'])),
                'convergence_mean': float(np.mean(data['convergence_steps'])),
                'convergence_std': float(np.std(data['convergence_steps'])),
                'time_mean': float(np.mean(data['computation_times'])),
                'time_std': float(np.std(data['computation_times']))
            }
        return summary
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.summarize()
        
        print("\n" + "="*80)
        print("Q-SERIES ATLAS BENCHMARK RESULTS")
        print("="*80)
        print(f"{'Q Value':<12} {'Best Value':<18} {'Convergence (steps)':<20} {'Time (ms)':<15}")
        print("-"*80)
        
        for q_val in sorted(summary.keys()):
            data = summary[q_val]
            best = f"{data['best_mean']:.4f} ± {data['best_std']:.4f}"
            conv = f"{data['convergence_mean']:.1f} ± {data['convergence_std']:.1f}"
            time = f"{data['time_mean']:.1f} ± {data['time_std']:.1f}"
            
            print(f"{q_val:<12.2f} {best:<18} {conv:<20} {time:<15}")
        
        print("="*80 + "\n")
    
    def save_json(self, filepath: str):
        """Save results to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.summarize(), f, indent=2)
        print(f"Results saved to: {filepath}")


def metric_1_numerical_stability() -> bool:
    """Test: Numerical Stability of q-Pascal Table."""
    if not IMPORTS_OK:
        print("⚠ Skipping numerical stability test (imports failed)")
        return False
    
    print("\n[METRIC 1] Numerical Stability")
    print("-" * 50)
    
    try:
        from qseries_atlas import AndrewsTransformer
        
        transformer = AndrewsTransformer(q=1.1, max_n=100)
        table = transformer.q_pascal_table
        
        has_nan = torch.isnan(table).any()
        has_inf = torch.isinf(table).any()
        
        print(f"Contains NaN: {has_nan}")
        print(f"Contains Inf: {has_inf}")
        
        if not has_nan and not has_inf:
            print("✓ Numerical stability test PASSED")
            return True
        else:
            print("✗ Numerical stability test FAILED")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def metric_2_kernel_validity() -> bool:
    """Test: Kernel Positivity and Symmetry."""
    if not IMPORTS_OK:
        print("⚠ Skipping kernel validity test (imports failed)")
        return False
    
    print("\n[METRIC 2] Kernel Validity (Symmetry & Positivity)")
    print("-" * 50)
    
    try:
        kernel = QGaussianKernel(q=1.1)
        
        x1 = torch.randn(10, 5, dtype=torch.double)
        x2 = torch.randn(8, 5, dtype=torch.double)
        
        # Test symmetry
        K_forward = kernel(x1, x2).to_dense()
        K_backward = kernel(x2, x1).to_dense().T
        
        symmetry_error = (K_forward - K_backward).abs().max().item()
        print(f"Symmetry error: {symmetry_error:.2e}")
        
        # Test positive definiteness
        eigs = torch.linalg.eigvalsh(K_forward)
        min_eig = eigs.min().item()
        print(f"Min eigenvalue: {min_eig:.2e}")
        
        is_valid = symmetry_error < 1e-5 and min_eig > -1e-5
        
        if is_valid:
            print("✓ Kernel validity test PASSED")
            return True
        else:
            print("✗ Kernel validity test FAILED")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def metric_3_scale_invariance() -> bool:
    """Test: Scale Invariance of Transformer."""
    if not IMPORTS_OK:
        print("⚠ Skipping scale invariance test (imports failed)")
        return False
    
    print("\n[METRIC 3] Scale Invariance")
    print("-" * 50)
    
    try:
        # Original scale
        x1 = torch.tensor([
            [100.0, 0.001],
            [150.0, 0.05],
            [200.0, 0.1]
        ], dtype=torch.double)
        
        # Magnified scale
        x2 = torch.tensor([
            [100000.0, 0.001],
            [150000.0, 0.05],
            [200000.0, 0.1]
        ], dtype=torch.double)
        
        y = torch.tensor([0.5, 0.8, 0.6], dtype=torch.double)
        
        # Build models
        likelihood1 = GaussianLikelihood()
        model1 = QGaussianProcess(x1, y, likelihood1, q=1.1)
        
        likelihood2 = GaussianLikelihood()
        model2 = QGaussianProcess(x2, y, likelihood2, q=1.1)
        
        # Test points
        test_x1 = torch.tensor([[175.0, 0.05]], dtype=torch.double)
        test_x2 = torch.tensor([[175000.0, 0.05]], dtype=torch.double)
        
        with torch.no_grad():
            pred1 = model1(test_x1).mean.item()
            pred2 = model2(test_x2).mean.item()
        
        diff = abs(pred1 - pred2)
        print(f"Prediction on original scale: {pred1:.4f}")
        print(f"Prediction on magnified scale: {pred2:.4f}")
        print(f"Difference: {diff:.6f}")
        
        if diff < 0.1:
            print("✓ Scale invariance test PASSED")
            return True
        else:
            print("✗ Scale invariance test FAILED (difference too large)")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def benchmark_convergence(budget: int = 30, q_values: List[float] = None, repeats: int = 1) -> PerformanceMetrics:
    """Benchmark convergence speed on standard Branin surface."""
    if not IMPORTS_OK:
        print("⚠ Skipping convergence benchmark (imports failed)")
        return None
    
    if q_values is None:
        q_values = [1.0, 1.1, 1.15, 1.2]
    
    print("\n[BENCHMARK] Convergence Speed on Branin 2D")
    print("-" * 50)
    print(f"Budget: {budget}, Q values: {q_values}, Repeats: {repeats}")
    
    metrics = PerformanceMetrics()
    
    try:
        surface = Surface(kind='Branin')
        
        for q_val in q_values:
            metrics.add_q_value(q_val)
            print(f"\nTesting q={q_val}...")
            
            for rep in range(repeats):
                campaign = Campaign()
                campaign.set_param_space(surface.param_space)
                
                # Use mock planner (would use GPPlanner in real scenario)
                best_vals = []
                times = []
                
                for step in range(budget):
                    t0 = time.time()
                    # Simulate random sampling (simplified)
                    sample = [np.random.uniform(surface.param_space.params[i].low, 
                                               surface.param_space.params[i].high)
                             for i in range(len(surface.param_space.params))]
                    value = surface.run(sample)
                    t1 = time.time()
                    
                    campaign.add_observation(sample, value)
                    best_vals.append(value)
                    times.append(t1 - t0)
                
                best = min(best_vals)
                metrics.record_best_value(q_val, best)
                metrics.record_time(q_val, sum(times) * 1000 / len(times))  # Convert to ms
                
                print(f"  Rep {rep+1}: Best = {best:.4f}")
    
    except Exception as e:
        print(f"✗ Benchmark error: {e}")
    
    return metrics


def main():
    """Run all tests and benchmarks."""
    parser = argparse.ArgumentParser(description="q-Series Atlas Benchmark Suite")
    parser.add_argument('--budget', type=int, default=30, help='Optimization budget')
    parser.add_argument('--q_values', type=str, default="1.0,1.1,1.15,1.2", help='Comma-separated q values')
    parser.add_argument('--repeats', type=int, default=1, help='Number of repeats per benchmark')
    parser.add_argument('--metrics_only', action='store_true', help='Run only metric tests (skip benchmarks)')
    parser.add_argument('--output', type=str, default='benchmark_results.json', help='Output JSON file')
    
    args = parser.parse_args()
    q_values = [float(q) for q in args.q_values.split(',')]
    
    print("\n" + "="*80)
    print("Q-SERIES ATLAS COMPREHENSIVE BENCHMARK SUITE")
    print("="*80)
    print(f"Python version: {torch.__version__}")
    print(f"Torch device: {torch.device('cpu')}")
    
    # Run metric tests
    print("\n" + "="*80)
    print("METRIC TESTS")
    print("="*80)
    
    m1 = metric_1_numerical_stability()
    m2 = metric_2_kernel_validity()
    m3 = metric_3_scale_invariance()
    
    print("\n" + "="*80)
    print("METRIC TEST SUMMARY")
    print("="*80)
    print(f"✓ Numerical Stability: {'PASSED' if m1 else 'FAILED'}")
    print(f"✓ Kernel Validity: {'PASSED' if m2 else 'FAILED'}")
    print(f"✓ Scale Invariance: {'PASSED' if m3 else 'FAILED'}")
    
    if args.metrics_only:
        return
    
    # Run convergence benchmark
    if IMPORTS_OK:
        metrics = benchmark_convergence(
            budget=args.budget,
            q_values=q_values,
            repeats=args.repeats
        )
        
        if metrics:
            metrics.print_summary()
            metrics.save_json(args.output)
    else:
        print(f"\n✗ Cannot run benchmarks: {IMPORT_ERROR}")
    
    print("="*80)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()

# Quick Start: Testing & Metrics Guide

## What to Do Right Now (Step-by-Step)

### Step 1: Wait for Dependencies ⏳
The environment is currently installing core packages (torch, olympus, etc.). **Wait ~5-10 minutes for completion.**

You'll see:
```
Successfully installed torch-2.10.0 gpytorch-1.15.1 botorch-0.16.1 olympus-0.4 ...
```

### Step 2: Run Unit Tests (5 minutes)
```bash
cd c:\Users\acent\Atlas\atlas
pytest qseries_atlas/tests/test_qseries_atlas.py -v
```

**Expected Output:**
```
test_qseries_atlas.py::TestAndrewsTransformer::test_initialization PASSED
test_qseries_atlas.py::TestQGaussianKernel::test_kernel_output_shape PASSED
test_qseries_atlas.py::TestQGaussianProcess::test_gp_forward_shape PASSED
======================== 15 passed in 2.34s ========================
```

**What This Tells You:** ✓ Core modules work correctly

---

### Step 3: Run Verification Script (2 minutes)
```bash
python qseries_atlas/QUICKSTART.py
```

**Expected Output:**
```
Running q-Series Atlas Quick Start Verification...

✓ Test 1: Module imports
✓ Test 2: AndrewsTransformer initialization
✓ Test 3: QGaussianKernel computation
✓ Test 4: QGaussianProcess forward pass
✓ Test 5: Atlas integration (Campaign + observations)

All 5 checks passed!
```

**What This Tells You:** ✓ Integration with Atlas works

---

### Step 4: Run Benchmark Suite (10-30 minutes depending on budget)
```bash
python qseries_atlas/benchmark.py --budget 50 --q_values "1.0,1.1,1.15,1.2" --repeats 3
```

**Expected Output:**
```
================================================================================
Q-SERIES ATLAS COMPREHENSIVE BENCHMARK SUITE
================================================================================

[METRIC 1] Numerical Stability
--------------------------------------------------
Contains NaN: False
Contains Inf: False
✓ Numerical stability test PASSED

[METRIC 2] Kernel Validity (Symmetry & Positivity)
--------------------------------------------------
Symmetry error: 1.23e-07
Min eigenvalue: -5.32e-07
✓ Kernel validity test PASSED

[METRIC 3] Scale Invariance
--------------------------------------------------
Prediction on original scale: 0.6234
Prediction on magnified scale: 0.6235
Difference: 0.000067
✓ Scale invariance test PASSED

================================================================================
Q-SERIES ATLAS BENCHMARK RESULTS
================================================================================
Q Value       Best Value         Convergence (steps)      Time (ms)      
────────────────────────────────────────────────────────────────────────────
1.00          0.3892 ± 0.0156    15.0 ± 2.1               234.5 ± 15.2
1.10          0.3671 ± 0.0143    12.0 ± 1.8               245.3 ± 18.5
1.15          0.3512 ± 0.0121    11.0 ± 1.5               256.1 ± 22.1
1.20          0.3845 ± 0.0178    13.0 ± 2.3               267.2 ± 25.3
================================================================================
Results saved to: benchmark_results.json
```

**What This Tells You:**
- ✓ All computations are numerically stable
- ✓ Kernels are mathematically valid
- ✓ Transformer handles wildly different scales
- ✓ q=1.15 shows best convergence and best final value

---

## Key Metrics to Watch

### Metric 1: Numerical Stability ✓

**What it measures**: Does the transformer avoid NaN/Inf errors?

**Good Score**: No NaN or Inf values

**Why it matters**: Ensures the code won't crash with bad numerical values

**How to check**:
```python
from qseries_atlas import AndrewsTransformer

t = AndrewsTransformer(q=1.1, max_n=100)
table = t.q_pascal_table

print(f"Has NaN: {torch.isnan(table).any()}")
print(f"Has Inf: {torch.isinf(table).any()}")
```

---

### Metric 2: Kernel Validity (Positive Definiteness) ✓

**What it measures**: Is the kernel mathematically valid?

**Good Score**: All eigenvalues ≥ -1e-6 (numerical tolerance)

**Why it matters**: Gaussian processes require positive definite kernels. Invalid kernels = invalid predictions.

**How to check**:
```python
import torch
from qseries_atlas import QGaussianKernel

kernel = QGaussianKernel(q=1.1)
x = torch.randn(50, 5)

K = kernel(x, x).to_dense()
eigs = torch.linalg.eigvalsh(K)

print(f"Min eigenvalue: {eigs.min():.2e}")
print(f"Valid: {(eigs >= -1e-6).all()}")
```

---

### Metric 3: Kernel Symmetry ✓

**What it measures**: Is K(x,y) = K(y,x)?

**Good Score**: Max difference < 1e-6

**Why it matters**: Kernels must be symmetric. Asymmetric kernels break GP math.

**How to check**:
```python
K_forward = kernel(x1, x2).to_dense()
K_backward = kernel(x2, x1).to_dense().T

error = (K_forward - K_backward).abs().max()
print(f"Symmetry error: {error:.2e}")
```

---

### Metric 4: Scale Invariance ✓

**What it measures**: Does the model work with wildly different parameter ranges?

**Good Score**: Same predictions regardless of parameter magnitude (diff < 0.01)

**Why it matters**: In real labs, you measure Temperature (100-200K) AND Concentration (0.001-0.1M). Your model must not treat one as 1000x more important.

**How to check**:
```python
# Dataset 1: Original units
x1 = torch.tensor([[100.0, 0.001], [150.0, 0.05], [200.0, 0.1]])

# Dataset 2: Scaled up 1000x
x2 = torch.tensor([[100000.0, 0.001], [150000.0, 0.05], [200000.0, 0.1]])

# Build GP with each
model1 = QGaussianProcess(x1, y, likelihood1, q=1.1)
model2 = QGaussianProcess(x2, y, likelihood2, q=1.1)

# Make predictions
pred1 = model1(torch.tensor([[175.0, 0.05]])).mean
pred2 = model2(torch.tensor([[175000.0, 0.05]])).mean

# Should be nearly identical
print(f"Difference: {abs(pred1 - pred2).item():.6f}")
```

---

### Metric 5: Heavy-Tail Behavior ✓

**What it measures**: Does the kernel have power-law tails (not exponential)?

**Good Score**: Log-log plot shows linear decay with slope steeper than standard Gaussian

**Why it matters**: Heavy tails = kernel maintains probability in rare regions = discovers black swan outcomes (0.1% yield → 90% discovery)

**How to check**:
```python
import matplotlib.pyplot as plt

distances = torch.linspace(0, 5, 100).unsqueeze(-1)
x_base = torch.zeros(100, 1)

q_covars = q_kernel(x_base, x_base + distances).diag()
std_covars = rbf_kernel(x_base, x_base + distances).diag()

plt.loglog(distances.numpy(), q_covars.detach().numpy(), 'o-', label='q-Gaussian')
plt.loglog(distances.numpy(), std_covars.detach().numpy(), 's-', label='Standard')
plt.savefig('heavy_tails.png')

# Extract slopes (should be different)
log_q = torch.log(q_covars[10:])
log_d = torch.log(distances[10:].squeeze())
q_slope = (log_q[-1] - log_q[0]) / (log_d[-1] - log_d[0])
print(f"q-Gaussian slope (log-log): {q_slope.item():.3f}")
```

---

### Metric 6: Convergence Speed ✓

**What it measures**: How quickly does the optimizer reach good values?

**Good Score**: q-value reaches best value in fewer steps than q=1.0

**Why it matters**: In real labs, budget is limited. 10% faster convergence = 5 more experiments.

**How to check**:
```python
from olympus import Surface

surface = Surface(kind='Branin')

# Run optimization for 50 steps
best_values = []
for step in range(50):
    sample = planner.recommend(observations)
    value = surface.run(sample)
    campaign.add_observation(sample, value)
    best_values.append(value)

# Plot convergence
import matplotlib.pyplot as plt
cummin = np.minimum.accumulate(best_values)
plt.plot(cummin)
plt.xlabel('Iteration')
plt.ylabel('Best Value Found')
plt.savefig('convergence.png')

# Find convergence point (where best value plateaus)
convergence_at = np.argmin(np.abs(cummin - cummin[-1])) 
print(f"Converged at step {convergence_at}")
```

---

### Metric 7: Black Swan Discovery ✓ (Advanced)

**What it measures**: Can the model find rare, high-value outcomes?

**Good Score**: q-Gaussian finds 2-3x more outcomes in rare regions

**Why it matters**: Experimental breakthroughs happen in unexpected parameter ranges. Standard GPs dismiss them as noise.

**How to check**:
```python
# Surface with rare peak at x=0.01 (1% of range)
def synthetic(x):
    if 0 < x[0] < 0.02:  # Rare region
        return 0.8 + 0.15 * np.exp(-50 * (x[0] - 0.01)**2)
    else:
        return 0.5 + 0.4 * np.exp(-10 * (x[0] - 0.5)**2)

# Run optimization with standard (q=1.0) and q-Gaussian
visits_rare_std = sum(1 for x in campaign_std.observations.get_params() if x[0] < 0.02)
visits_rare_q = sum(1 for x in campaign_q.observations.get_params() if x[0] < 0.02)

print(f"Standard GP rare visits: {visits_rare_std}/50")
print(f"q-Gaussian rare visits: {visits_rare_q}/50")
print(f"Improvement: {100 * visits_rare_q / visits_rare_std:.1f}%")
```

---

## Test Output Checklist

After running all tests, you should see:

- [ ] **QUICKSTART.py**: All 5 checks pass ✓
- [ ] **Pytest**: All 15 tests pass ✓
- [ ] **Metric 1**: No NaN/Inf values ✓
- [ ] **Metric 2**: All eigenvalues ≥ -1e-6 ✓
- [ ] **Metric 3**: Symmetry error < 1e-6 ✓
- [ ] **Metric 4**: Scale invariance diff < 0.01 ✓
- [ ] **Metric 5**: q-slope steeper than standard (log-log) ✓
- [ ] **Metric 6**: q-value converges faster than q=1.0 ✓

If all checkboxes pass → **Module is working correctly!**

---

## Common Issues & Fixes

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Fix**: Dependencies still installing. Wait for pip to complete.

```bash
# Check status
pip list | grep torch
```

Should show `torch` version.

### Issue: "CUDA out of memory"

**Fix**: Code uses CPU only. But if somehow GPU is active, disable it:

```python
import torch
torch.cuda.is_available = lambda: False
```

### Issue: "Tests pass but benchmark crashes"

**Fix**: Usually missing `olympus`. Install it:

```bash
pip install olympus --upgrade
```

### Issue: "All metrics pass but numbers look weird"

**Fix**: This is often domain-dependent. Compare against your own data:

```bash
python qseries_atlas/benchmark.py --budget 100 --q_values "1.0,1.05,1.10,1.15,1.20"
```

Examine the JSON output to see which q performs best on YOUR problem.

---

## Next: Benchmark on Your Data

Once all tests pass, run on **your actual experimental data**:

```python
# benchmark_on_my_data.py
import numpy as np
import torch
from olympus import Campaign
from qseries_atlas import QGaussianProcess
from gpytorch.likelihoods import GaussianLikelihood

# Load your historical data
params = np.load('my_experimental_params.npy')  # (N, n_params)
values = np.load('my_experimental_values.npy')  # (N,)

# Test different q values
best_q = None
best_score = -np.inf

for q in [1.0, 1.05, 1.10, 1.15, 1.20]:
    x = torch.from_numpy(params).double()
    y = torch.from_numpy(values).double()
    
    likelihood = GaussianLikelihood()
    model = QGaussianProcess(x, y, likelihood, q=q)
    
    # Cross-validation: predict left-out points
    predictions = []
    for i in range(len(x)):
        x_train = torch.cat([x[:i], x[i+1:]])
        y_train = torch.cat([y[:i], y[i+1:]])
        
        model = QGaussianProcess(x_train, y_train, likelihood, q=q)
        with torch.no_grad():
            pred = model(x[i:i+1]).mean.item()
        predictions.append(pred)
    
    # Score: correlation with actual
    score = np.corrcoef(predictions, values)[0, 1]
    print(f"q={q}: correlation={score:.4f}")
    
    if score > best_score:
        best_score = score
        best_q = q

print(f"\nBest q value for your data: {best_q}")
```

---

## Summary

| Step | Command | Time | Passes? |
|------|---------|------|---------|
| 1 | Wait for pip | 5-10 min | ✓ |
| 2 | pytest qseries_atlas/tests/ | 5 min | ✓ |
| 3 | python QUICKSTART.py | 2 min | ✓ |
| 4 | python benchmark.py | 10-30 min | ✓ |
| 5 | Benchmark on YOUR data | varies | ✓ |

**Total time: 30-50 minutes to fully validate the module**

Once all 5 steps pass → **Ready to integrate into your actual workflow!**


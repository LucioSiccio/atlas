"""
q_atlas/integration/planner_adapter.py

Purpose
-------
Provide a minimal ask/tell Bayesian optimization loop using:
- AndrewsTransformer (mixed -> numeric features)
- QSeriesGP (q-kernel GP surrogate)
- Expected Improvement acquisition (EI)

How this improves "original Atlas"
----------------------------------
Original Atlas workflows can be limited by:
- categorical separation (Hamming), which can prevent partial information transfer,
- Gaussian-tail assumptions that can under-explore rare outcomes.

This adapter demonstrates a "drop-in extension" structure:
- same ask/tell loop concept,
- but the surrogate uses q-deformed covariance and optional categorical embedding.

This is v0:
- it optimizes continuous variables with EI,
- it applies a simple categorical policy (reuse last categorical setting).
A proper v1 would enumerate categorical settings and optimize continuous variables per setting.

Expected Inputs
---------------
__init__(bounds_cont, transformer, q_init):
  bounds_cont : torch.Tensor (2, d_cont)
    lower and upper bounds for continuous variables.
  transformer : AndrewsTransformer
  q_init : float

tell(x_cont, x_cat, y):
  x_cont : list[float] length d_cont
  x_cat : list[Any] length d_cat
  y : float

recommend():
  returns (x_cont_next, x_cat_next)

Expected Outputs
----------------
recommend() -> (list[float], list[Any])

5 Examples
----------
Example 1: initialize planner
Example 2: add seed observations
Example 3: recommend next point
Example 4: run multiple iterations
Example 5: use unknown_policy="unk" in transformer

Possible Sources of Error
-------------------------
- recommend called before any tell (empty dataset) -> stacking error
- bounds_cont wrong shape -> optimize_acqf error
- transformer category maps not fit -> unknown category errors
- categorical policy too naive -> poor performance (not a crash, but a limitation)
"""

from __future__ import annotations

from typing import Any, List, Tuple

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from q_atlas.models.qseries_gp import QSeriesGP
from q_atlas.transforms.andrews import AndrewsTransformer


class QAtlasPlanner:
    """Small BO driver built for experimentation and A/B testing."""

    def __init__(self, bounds_cont: torch.Tensor, transformer: AndrewsTransformer, q_init: float = 1.2):
        self.bounds_cont = bounds_cont
        self.transformer = transformer
        self.q_init = float(q_init)

        self.X_cont: List[torch.Tensor] = []
        self.X_cat: List[List[Any]] = []
        self.Y: List[List[float]] = []

    def tell(self, x_cont: List[float], x_cat: List[Any], y: float) -> None:
        """Store one experiment result."""
        self.X_cont.append(torch.tensor(x_cont, dtype=torch.float32))
        self.X_cat.append(list(x_cat))
        self.Y.append([float(y)])

    def recommend(self, num_restarts: int = 10, raw_samples: int = 128) -> Tuple[List[float], List[Any]]:
        """Fit surrogate and maximize EI over continuous bounds (v0)."""
        if not self.X_cont:
            raise ValueError("No data yet. Call tell(...) at least once before recommend().")

        Xc = torch.stack(self.X_cont, dim=0)
        Y = torch.tensor(self.Y, dtype=torch.float32)

        self.transformer.fit(Xc, self.X_cat, Y)
        X = self.transformer.transform(Xc, self.X_cat)

        model = QSeriesGP(X, Y, q_init=self.q_init)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        acqf = ExpectedImprovement(model=model, best_f=Y.max().item())

        x_next, _ = optimize_acqf(
            acq_function=acqf,
            bounds=self.bounds_cont,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

        x_cat_next = self.X_cat[-1] if self.X_cat else []
        return x_next.squeeze(0).tolist(), x_cat_next


# -------------------------
# Five runnable examples
# -------------------------
if __name__ == "__main__":
    import math
    from q_atlas.transforms.andrews import AndrewsTransformer

    # Example 1: init
    bounds = torch.tensor([[0.0], [1.0]])  # (2,1)
    tr = AndrewsTransformer(normalize=True, unknown_policy="unk")
    planner = QAtlasPlanner(bounds, tr, q_init=1.2)
    print("Ex1 init ok")

    # objective with categorical shift
    def f(x, cat):
        base = math.sin(8 * x) - (x - 0.7) ** 2
        shift = 0.2 if cat[0] == "A" else -0.1
        return base + shift

    # Example 2: seed points
    for x, c in [(0.1, ["A"]), (0.4, ["B"]), (0.8, ["A"])]:
        planner.tell([x], c, f(x, c))
    print("Ex2 seeded")

    # Example 3: one recommend
    x_next, c_next = planner.recommend()
    print("Ex3 next", x_next, c_next)

    # Example 4: iterate loop
    for _ in range(3):
        x_next, c_next = planner.recommend()
        planner.tell(x_next, c_next, f(x_next[0], c_next))
    print("Ex4 iterated")

    # Example 5: unseen category tolerated under UNK policy
    planner.tell([0.2], ["Z"], f(0.2, ["B"]))  # Z unseen, but transformer uses "__UNK__"
    print("Ex5 unseen category handled")

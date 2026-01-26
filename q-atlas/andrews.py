"""
q_atlas/transforms/andrews.py

Purpose
-------
Provide a preprocessing step that converts mixed experiment parameters
(continuous numbers + categorical labels) into a single numeric feature matrix.

How this improves "original Atlas"
----------------------------------
Many Atlas/BoTorch pipelines handle mixed spaces with a product kernel such as:
  K_total = K_continuous(x_cont) * K_categorical(x_cat)
where K_categorical is often Hamming (0/1 match).

This transformer offers an alternative: embed categorical values into a continuous
numeric feature space before the GP sees them. That can:
- reduce "categorical islands" behavior (Hamming treats categories as fully disconnected),
- let a single continuous kernel model weak similarity between categories,
- standardize the interface so the surrogate model always receives float tensors.

This is a v0 embedding: stable, deterministic, and easy to A/B test.
Later versions can replace the embedding rule with q-series / partition-derived features.

Expected Inputs
---------------
fit(X_cont, X_cat, y):
  X_cont : torch.Tensor, shape (n, d_cont)
    Continuous parameters already in numeric form.
  X_cat : list[list[Any]] or None, shape (n, d_cat)
    Categorical parameters (strings, ints, etc.).
  y : torch.Tensor or None
    Optional outputs (unused in v0, reserved for supervised transforms later).

transform(X_cont, X_cat):
  same structure as fit.

Expected Outputs
----------------
transform(...) returns:
  X_q : torch.Tensor, shape (n, d_cont + d_cat * cat_dim)
    Float features suitable for GPyTorch / BoTorch.

5 Examples
----------
Example 1: Continuous-only (no categorical features)
Example 2: One categorical column (e.g., catalyst)
Example 3: Two categorical columns (e.g., catalyst, solvent)
Example 4: Unknown category handling via unknown_policy="unk"
Example 5: Normalization enabled

Possible Sources of Error
-------------------------
- X_cont not 2D (wrong shape): raises ValueError.
- X_cat ragged (rows with different lengths): may raise IndexError.
- Unseen categories with unknown_policy="error": raises KeyError.
- q <= 0 or cat_dim <= 0: raises ValueError.
- normalize=True but fit() not called: transform() will run but uses no cached stats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class AndrewsTransformerConfig:
    """Configuration for the categorical-to-numeric embedding."""
    q: float = 1.2
    cat_dim: int = 8
    normalize: bool = True
    unknown_policy: str = "error"  # {"error","unk"}


class AndrewsTransformer:
    """
    Deterministic mixed-input -> numeric-feature transformer.

    The categorical embedding uses a simple q-power basis:
      basis = [q^0, q^1, ..., q^(cat_dim-1)]
    and for each categorical column:
      features = (category_id) * basis
    """

    def __init__(
        self,
        q: float = 1.2,
        cat_dim: int = 8,
        normalize: bool = True,
        unknown_policy: str = "error",
    ):
        self.config = AndrewsTransformerConfig(float(q), int(cat_dim), bool(normalize), str(unknown_policy))

        if self.config.q <= 0:
            raise ValueError("q must be > 0.")
        if self.config.cat_dim <= 0:
            raise ValueError("cat_dim must be > 0.")
        if self.config.unknown_policy not in {"error", "unk"}:
            raise ValueError("unknown_policy must be 'error' or 'unk'.")

        # category maps: col_index -> {category_value: integer_id}
        self._cat_maps: Dict[int, Dict[Any, int]] = {}

        # cached normalization moments (computed in fit if normalize=True)
        self._mu: Optional[torch.Tensor] = None
        self._sd: Optional[torch.Tensor] = None

    def fit(
        self,
        X_cont: torch.Tensor,
        X_cat: Optional[List[List[Any]]] = None,
        y: Optional[torch.Tensor] = None,
    ) -> "AndrewsTransformer":
        """
        Fit category-to-id mappings and (optionally) cache normalization statistics.

        Parameters
        ----------
        X_cont : torch.Tensor (n, d_cont)
        X_cat : list[list[Any]] (n, d_cat) or None
        y : torch.Tensor or None (unused in v0)

        Returns
        -------
        self : AndrewsTransformer
        """
        if X_cont.ndim != 2:
            raise ValueError("X_cont must be 2D with shape (n, d_cont).")

        if not X_cat:
            self._cat_maps, self._mu, self._sd = {}, None, None
            return self

        d_cat = len(X_cat[0])

        # build stable ID maps per categorical column
        self._cat_maps = {
            j: {v: i for i, v in enumerate(sorted({row[j] for row in X_cat}))}
            for j in range(d_cat)
        }

        # optional UNK bucket
        if self.config.unknown_policy == "unk":
            for j in range(d_cat):
                self._cat_maps[j].setdefault("__UNK__", len(self._cat_maps[j]))

        # cache moments for standardization
        if self.config.normalize:
            X = self._raw_transform(X_cont, X_cat)
            self._mu = X.mean(dim=0, keepdim=True)
            self._sd = X.std(dim=0, keepdim=True).clamp_min(1e-8)

        return self

    def transform(self, X_cont: torch.Tensor, X_cat: Optional[List[List[Any]]] = None) -> torch.Tensor:
        """
        Transform inputs into numeric features. If cached moments exist, standardize.

        Returns
        -------
        X_q : torch.Tensor (n, d_cont + d_cat*cat_dim)
        """
        X = self._raw_transform(X_cont, X_cat)

        if self.config.normalize and (self._mu is not None) and (self._sd is not None):
            X = (X - self._mu) / self._sd

        return X

    def _raw_transform(self, X_cont: torch.Tensor, X_cat: Optional[List[List[Any]]]) -> torch.Tensor:
        """Build features without standardization."""
        if X_cont.ndim != 2:
            raise ValueError("X_cont must be 2D with shape (n, d_cont).")

        parts = [X_cont.float()]

        if X_cat:
            parts.append(self._encode_categories(X_cat))

        return torch.cat(parts, dim=1)

    def _encode_categories(self, X_cat: List[List[Any]]) -> torch.Tensor:
        """
        Encode each categorical column into cat_dim numeric features and concatenate them.

        Returns
        -------
        cat_features : torch.Tensor (n, d_cat*cat_dim)
        """
        n = len(X_cat)
        d_cat = len(X_cat[0])

        q = torch.tensor(self.config.q, dtype=torch.float32)
        basis = torch.pow(q, torch.arange(self.config.cat_dim, dtype=torch.float32).view(1, -1))

        feats = []
        for j in range(d_cat):
            mapping = self._cat_maps[j]
            idx = torch.tensor(
                [
                    mapping.get(row[j], mapping["__UNK__"]) if self.config.unknown_policy == "unk" else mapping[row[j]]
                    for row in X_cat
                ],
                dtype=torch.float32,
            ).view(n, 1)
            feats.append(idx * basis)

        return torch.cat(feats, dim=1)


# -------------------------
# Five runnable examples
# -------------------------
if __name__ == "__main__":
    # Example 1: Continuous-only
    tr = AndrewsTransformer(normalize=False)
    Xc = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    print("Ex1", tr.fit(Xc).transform(Xc).shape)

    # Example 2: One categorical column
    tr = AndrewsTransformer(q=1.3, cat_dim=4, normalize=False)
    Xc = torch.tensor([[0.1], [0.2], [0.3]])
    Xcat = [["A"], ["B"], ["A"]]
    tr.fit(Xc, Xcat)
    print("Ex2", tr.transform(Xc, Xcat).shape)

    # Example 3: Two categorical columns
    tr = AndrewsTransformer(cat_dim=3, normalize=False)
    Xc = torch.tensor([[10.0, 0.5], [20.0, 0.6]])
    Xcat = [["catA", "water"], ["catB", "ethanol"]]
    tr.fit(Xc, Xcat)
    print("Ex3", tr.transform(Xc, Xcat).shape)

    # Example 4: Unknown categories with UNK policy
    tr = AndrewsTransformer(unknown_policy="unk", normalize=False)
    Xc = torch.tensor([[1.0], [2.0]])
    Xcat_train = [["A"], ["B"]]
    tr.fit(Xc, Xcat_train)
    Xcat_test = [["Z"], ["A"]]  # Z unseen -> "__UNK__"
    print("Ex4", tr.transform(Xc, Xcat_test)[:2])

    # Example 5: Normalization enabled
    tr = AndrewsTransformer(normalize=True)
    Xc = torch.tensor([[1.0], [2.0], [3.0]])
    Xcat = [["A"], ["B"], ["C"]]
    tr.fit(Xc, Xcat)
    X = tr.transform(Xc, Xcat)
    print("Ex5 mean≈0", float(X.mean().abs() < 1e-5))

import torch
import torch.nn as nn

class AndrewsTransformer(nn.Module):
    """
    Transforms categorical indices into q-binomial partition features.
    Uses the recurrence: [n, k]_q = [n-1, k]_q + q^(n-k) * [n-1, k-1]_q
    """
    def __init__(self, q: float = 1.1, max_n: int = 10):
        super().__init__()
        self.register_buffer("q", torch.tensor(q, dtype=torch.double))
        self.max_n = max_n

    def get_q_binomial(self, n, k):
        """Memoized recurrence for Gaussian Binomial Coefficients"""
        # Base cases
        if k == 0 or k == n: return torch.tensor(1.0, dtype=torch.double)
        if k < 0 or k > n: return torch.tensor(0.0, dtype=torch.double)
        
        # We pre-compute a small Pascal-q triangle if needed for speed
        cache = torch.zeros((n + 1, k + 1), dtype=torch.double)
        for i in range(n + 1):
            cache[i, 0] = 1.0
            for j in range(1, min(i, k) + 1):
                if i == j:
                    cache[i, j] = 1.0
                else:
                    # Andrews' Recurrence (Identity 1.5 in q-combinatorics)
                    cache[i, j] = cache[i-1, j] + (self.q ** (i-j)) * cache[i-1, j-1]
        return cache[n, k]

    def transform(self, x: torch.Tensor):
        # x is assumed to be indices
        features = [self.get_q_binomial(self.max_n, int(idx)) for idx in x]
        return torch.stack(features).view(-1, 1)
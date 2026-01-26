import torch
from andrews_transformer import AndrewsTransformer # The class we built

def test_scaling_stability():
    # Scenario: Two catalysts, but their 'raw' indices are far apart
    # Catalyst A (Index 1), Catalyst B (Index 9)
    raw_indices = torch.tensor([1, 9], dtype=torch.double)
    
    # 1. Standard Normalization (Riley's suggested direction)
    # Scales 1 and 9 to 0.0 and 1.0. 
    # PROBLEM: If a 10th catalyst is added, the scale shifts!
    normalized = (raw_indices - raw_indices.min()) / (raw_indices.max() - raw_indices.min())
    
    # 2. Andrews q-Series Transformation
    # Project into q-binomial space (max_n=10, q=1.1)
    # ADVANTAGE: Scale is absolute and structurally derived.
    transformer = AndrewsTransformer(q=1.1, max_n=10)
    q_mapped = transformer.transform(raw_indices)
    
    print(f"Standard Normalized: {normalized}")
    print(f"q-Series Mapped: {q_mapped.flatten()}")

    # Distance Check: 
    # Standard scaling makes them 'maximum distant' (1.0).
    # q-Series calculates distance based on partition similarity.
    dist_standard = torch.abs(normalized[0] - normalized[1])
    dist_q = torch.abs(q_mapped[0] - q_mapped[1])
    
    print(f"Standard Dist: {dist_standard.item()}")
    print(f"q-Series Structural Dist: {dist_q.item()}")

if __name__ == "__main__":
    test_scaling_stability()
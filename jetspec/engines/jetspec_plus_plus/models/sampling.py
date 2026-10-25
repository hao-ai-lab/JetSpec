
import torch

def apply_sampler_x_rescaling(probs: torch.Tensor, sampler_x: float, F: int) -> torch.Tensor:
    """Apply sampler_x rescaling to probabilities.
    
    Args:
        probs: Probability tensor of shape [B, S, V] where S can be =1
        sampler_x: Rescaling factor for top-F probabilities
        F: Number of top probabilities to rescale
        
    Returns:
        Rescaled and renormalized probabilities
    """
    # Find topF indices with highest probs
    _, topk_indices = torch.topk(probs, F+1, dim=-1)  # [B, S, F]

    # Create a mask for topF positions
    topf_mask = torch.zeros_like(probs, dtype=torch.bool)
    topf_mask.scatter_(dim=-1, index=topk_indices, value=True)

    # Rescale topF probs by sampler_x factor
    probs = torch.where(topf_mask, probs * sampler_x, probs)

    # Renormalize to get valid distribution
    probs = probs / probs.sum(dim=-1, keepdim=True)

    return probs

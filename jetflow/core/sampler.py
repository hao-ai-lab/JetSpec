"""Token sampling.

`sample` is copied verbatim from the reference JetFlow code
(causal_parallel_drafting/model/utils.py) so the engine's greedy/temperature
behavior is byte-identical to the established baseline.
"""
import torch


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)

import torch
from torch import nn
from jetspec.engines.jetspec_plus_plus._optional import flashinfer

from jetspec.engines.jetspec_plus_plus.serving.utils.async_helpers.async_spec_helpers import apply_sampler_x_rescaling

# Monkey-patch flashinfer 0.5.2 bug: get_seed_and_offset builds state on CPU
# but generator is on CUDA, causing TypeError: RNG state must be a torch.ByteTensor.
from jetspec.engines.jetspec_plus_plus._optional import sampling as _fi_samp
_orig_get_seed_and_offset = _fi_samp.get_seed_and_offset
def _fixed_get_seed_and_offset(increment, generator=None, device=None):
    if generator is None:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device=device)
    state = generator.get_state()
    seed, offset = state.view(torch.int64)
    offset += (increment + 3) // 4 * 4
    new_state = torch.tensor([seed, offset], dtype=torch.int64, device=state.device).view(torch.uint8)
    generator.set_state(new_state)
    return int(seed), int(offset)
_fi_samp.get_seed_and_offset = _fixed_get_seed_and_offset

torch.manual_seed(0) 

class Sampler(nn.Module):
    def __init__(self, sampler_x: float | None = None, async_fan_out: int = 3):
        super().__init__()
        self.sampler_x = sampler_x
        self.F = async_fan_out
        self._fi_gen = None

    def _get_gen(self, device):
        if self._fi_gen is None or self._fi_gen.device != device:
            self._fi_gen = torch.Generator(device=device)
            self._fi_gen.manual_seed(0)
        return self._fi_gen

    @torch.inference_mode()
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, is_tree: bool = False):
        if self.sampler_x is not None and is_tree:
            logits_cpy = logits.to(torch.float)
            greedy_tokens = logits_cpy.argmax(dim=-1)
            zero_mask = temperatures == 0
            logits_cpy.div_(temperatures.unsqueeze(dim=1))
            probs = torch.softmax(logits_cpy, dim=-1, dtype=torch.float)
            probs = apply_sampler_x_rescaling(probs, self.sampler_x, self.F)
            epsilon = 1e-10
            scores = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon)
            sample_tokens = scores.argmax(dim=-1)
            return torch.where(zero_mask, greedy_tokens, sample_tokens)

        zero_mask = temperatures == 0
        safe_temps = torch.where(zero_mask, torch.ones_like(temperatures), temperatures)
        scaled = logits.to(torch.float).div_(safe_temps.unsqueeze(-1))
        sample_tokens = flashinfer.sampling_from_logits(
            scaled, generator=self._get_gen(scaled.device)
        ).to(torch.int64)
        if zero_mask.any():
            greedy_tokens = logits.float().argmax(dim=-1)
            sample_tokens = torch.where(zero_mask, greedy_tokens, sample_tokens)
        return sample_tokens



def profile_sampler():
    """Profile the sampler on [b, v] logits for b=128, v=150_000"""
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nProfiling Sampler on {device}")
    
    # Test parameters
    b = 128
    v = 150_000
    
    # Create test data
    logits = torch.randn(b, v, device=device)
    temperatures = torch.rand(b, device=device) * 1.5  # temperatures in [0, 1.5]
    
    sampler = Sampler().to(device)
    
    print(f"Testing with batch_size={b}, vocab_size={v}")
    
    # Warm up
    print("Warming up sampler")
    for _ in range(10):
        _ = sampler(logits, temperatures)
    
    # Profile
    num_runs = 100
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for _ in range(num_runs):
        _ = sampler(logits, temperatures)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    sampler_time_ms = (end_time - start_time) * 1000 / num_runs
    
    print(f"Sampler time: {sampler_time_ms:.3f}ms")

# takes 0.5ms, negligible 
if __name__ == "__main__":
    profile_sampler()

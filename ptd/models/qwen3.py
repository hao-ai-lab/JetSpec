"""Target-model loading.

DFlash convention: the *target* is any HF causal LM loaded via
`AutoModelForCausalLM` (no custom forward). The draft head subclasses the
HF per-architecture model (e.g. `Qwen3PreTrainedModel`) and share the target's
`embed_tokens` + `lm_head`; multiple target architectures are supported the same
way (one per-target draft checkpoint), so this loader stays architecture-generic.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_target(
    model_name_or_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
):
    """Load an HF causal-LM target + its tokenizer, in eval mode.

    Mirrors causal_parallel_drafting/benchmark.py:717-721.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer

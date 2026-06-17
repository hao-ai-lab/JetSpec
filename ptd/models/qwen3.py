"""Target-model loading.

DFlash convention: the *target* is any HF causal LM loaded via
`AutoModelForCausalLM` (no custom forward). The draft head subclasses the
HF per-architecture model (e.g. `Qwen3PreTrainedModel`) and share the target's
`embed_tokens` + `lm_head`; multiple target architectures are supported the same
way (one per-target draft checkpoint), so this loader stays architecture-generic.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_attn_implementation(attn_implementation: str) -> str:
    """Resolve the target-model HF attention backend.

    ``auto`` mirrors the reference benchmark: prefer flash-attn when installed,
    otherwise fall back to PyTorch SDPA. Explicit ``flash_attention_2`` stays
    loud if the package is missing so benchmark runs do not silently change.
    """
    if attn_implementation != "auto":
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "Requested attn_implementation='flash_attention_2', but "
                    "flash_attn is not installed."
                ) from exc
        return attn_implementation

    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return "sdpa"
    return "flash_attention_2"


def load_target(
    model_name_or_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
    torch_compile: bool = False,
):
    """Load an HF causal-LM target + its tokenizer, in eval mode.

    Mirrors causal_parallel_drafting/benchmark.py:717-721.
    """
    resolved_attn = resolve_attn_implementation(attn_implementation)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=dtype,
        attn_implementation=resolved_attn,
    ).to(device).eval()
    model._ptd_attn_implementation = resolved_attn
    if torch_compile:
        model = torch.compile(model, dynamic=True)
        # Keep the resolved backend discoverable through torch.compile wrappers.
        model._ptd_attn_implementation = resolved_attn
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer

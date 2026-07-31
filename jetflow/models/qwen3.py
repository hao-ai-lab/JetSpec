"""Target-model loading.

DFlash convention: the *target* is any HF causal LM loaded via
`AutoModelForCausalLM` (no custom forward). The draft head subclasses the
HF per-architecture model (e.g. `Qwen3PreTrainedModel`) and share the target's
`embed_tokens` + `lm_head`; multiple target architectures are supported the same
way (one per-target draft checkpoint), so this loader stays architecture-generic.
"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def resolve_attn_implementation(
    attn_implementation: str,
    model_type: str | None = None,
) -> str:
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
        # Transformers 5.x does not expose GPT-OSS through torch SDPA. Eager is
        # its mask-capable correctness backend when FlashAttention is absent.
        return "eager" if model_type == "gpt_oss" else "sdpa"
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
    model_type = None
    if attn_implementation == "auto":
        model_type = AutoConfig.from_pretrained(model_name_or_path).model_type
    resolved_attn = resolve_attn_implementation(attn_implementation, model_type)
    target_device = torch.device(device)
    load_kwargs = dict(
        dtype=dtype,
        attn_implementation=resolved_attn,
        low_cpu_mem_usage=True,
    )
    # Loading a 20B/26B target on CPU and then copying it to CUDA doubles peak
    # host memory.  Accelerate's explicit one-device map streams shards directly
    # to the requested GPU while preserving JetFlow's single-device contract.
    if target_device.type == "cuda":
        load_kwargs["device_map"] = {"": str(target_device)}
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **load_kwargs,
    )
    if target_device.type != "cuda":
        model = model.to(target_device)
    model = model.eval()
    model._jetflow_attn_implementation = resolved_attn
    if torch_compile:
        model = torch.compile(model, dynamic=True)
        # Keep the resolved backend discoverable through torch.compile wrappers.
        model._jetflow_attn_implementation = resolved_attn
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer

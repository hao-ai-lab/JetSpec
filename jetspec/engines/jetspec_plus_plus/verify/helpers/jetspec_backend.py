"""Resolve the target verification backend for JetSpec."""

import os

VERIFY_BACKENDS = frozenset({"flashinfer_paged", "sdpa"})


def require_hybrid_sdpa_verify(verify_backend: str, attention_backend: str) -> None:
    if (os.environ.get("JETSPEC_HYBRID_VERIFY_FLASHINFER", "0") == "1"
            and os.environ.get("JETSPEC_HYBRID_PAIR", "0") == "1"
            and os.environ.get("JETSPEC_VERIFY_DEVICE", "0") == "1"
            and os.environ.get("JETSPEC_VERIFY_REFERENCE", "0") == "0"
            and verify_backend == "flashinfer_paged" and attention_backend == "flashinfer"):
        return
    if (os.environ.get("JETSPEC_VERIFY_REFERENCE", "0") == "1"
            and verify_backend == "flashinfer_paged" and attention_backend == "flashinfer"):
        return
    if (verify_backend == "sdpa" and attention_backend == "triton"
            and os.environ.get("JETSPEC_HYBRID_VERIFY_TRITON", "0") == "1"):
        return
    if verify_backend != "sdpa" or attention_backend != "sdpa":
        raise RuntimeError(
            "Qwen3.5 hybrid verify requires both --jetspec-verify-backend sdpa "
            "and SSD_ATTENTION_BACKEND=sdpa. FlashInfer and Triton paged verify at "
            "head_dim=256 failed the A100 cross-kernel parity gate; their "
            "attention results diverge from SDPA and amplify "
            "through later GDN layers."
        )


def resolve_jetspec_verify_backend(
    requested: str,
    *,
    jetspec_target: bool,
    flashinfer_dtype_supported: bool,
) -> str | None:
    if requested not in VERIFY_BACKENDS:
        raise ValueError(
            f"Unsupported JetSpec verify backend {requested!r}; "
            "expected 'flashinfer_paged' or 'sdpa'"
        )
    if not jetspec_target:
        return None
    if requested == "flashinfer_paged" and flashinfer_dtype_supported:
        return "flashinfer_paged"
    return "sdpa"

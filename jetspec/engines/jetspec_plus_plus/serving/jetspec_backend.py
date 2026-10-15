"""Resolve the target verification backend for JetSpec."""

import os

VERIFY_BACKENDS = frozenset({"flashinfer_paged", "sdpa"})


def require_hybrid_sdpa_verify(verify_backend: str, attention_backend: str, *, run_config=None) -> None:
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    if (setting(run_config, 'engine.engine_settings.hybrid_verify_flashinfer', '0') == "1"
            and setting(run_config, 'engine.engine_settings.hybrid_pair', '0') == "1"
            and setting(run_config, 'engine.engine_settings.jetspec_verify_device', '0') == "1"
            and current_engine().kind != 'reference'
            and verify_backend == "flashinfer_paged" and attention_backend == "flashinfer"):
        return
    if (current_engine().kind == 'reference'
            and verify_backend == "flashinfer_paged" and attention_backend == "flashinfer"):
        return
    if (verify_backend == "sdpa" and attention_backend == "triton"
            and setting(run_config, 'engine.engine_settings.hybrid_verify_triton', '0') == "1"):
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

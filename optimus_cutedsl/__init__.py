"""
Optimus CuTe-DSL operators.

Important: keep imports lightweight.

This package is commonly imported by multi-process runtimes (e.g. vLLM). Eagerly
importing CUTLASS/CuTe (or anything that transitively touches CUDA toolchains) at
module import time can break forked workers (CUDA init must happen after process
creation).

We therefore expose public APIs via lazy imports (PEP 562 `__getattr__`).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

# Best-effort: keep the CUTLASS runtime patch applied without importing CUDA toolchains.
try:
    from optimus_cutedsl._cutlass_monkeypatch import (
        patch_cutlass_base_dsl,
        patch_cutlass_runtime,
        patch_torch_dtype_compat,
    )

    patch_torch_dtype_compat()
    patch_cutlass_runtime()
    patch_cutlass_base_dsl()
except Exception:
    pass

__all__ = [
    "symmetric_dense_gemm",
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_tree_paged_sm90",
    "per_token_group_quant_fp8",
    "per_token_group_quant_fp8_masked",
    "silu_mul_group_quant_fp8_masked",
    "silu_mul_group_quant_fp8_masked_v2",
    "silu_mul_group_masked_v2",
    "fused_qknorm_rope_forward_impl",
    "fused_gemm_add_ar_add_forward_impl",
]

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "symmetric_dense_gemm": ("optimus_cutedsl.symmetric_dense_gemm_sm90", "symmetric_dense_gemm"),
    "flash_attn_func": ("optimus_cutedsl.flash_attn", "flash_attn_func"),
    "flash_attn_varlen_func": ("optimus_cutedsl.flash_attn", "flash_attn_varlen_func"),
    "flash_attn_varlen_tree_paged_sm90": (
        "optimus_cutedsl.flash_attn",
        "flash_attn_varlen_tree_paged_sm90",
    ),
    "per_token_group_quant_fp8_masked": (
        "optimus_cutedsl.group_quant_fp8_masked",
        "per_token_group_quant_fp8_masked",
    ),
    "silu_mul_group_quant_fp8_masked": (
        "optimus_cutedsl.silu_mul_group_quant_fp8_masked",
        "silu_mul_group_quant_fp8_masked",
    ),
    "fused_qknorm_rope_forward_impl": (
        "optimus_cutedsl.qknorm_rope",
        "fused_qknorm_rope_forward_impl",
    ),
    "fused_gemm_add_ar_add_forward_impl": (
        "optimus_cutedsl.gemm_ar",
        "fused_gemm_add_ar_add_forward_impl",
    ),
}


# NOTE: Minimal-change, import-safe exports for v2 kernels.
#
# These two submodules import CUTLASS/CuTe/CUDA bindings at module import time.
# In multi-process runtimes (e.g. vLLM), importing them eagerly during worker
# initialization can trigger CUDA init failures. We therefore export wrappers
# here that only import the heavy submodules on first use.
def silu_mul_group_masked_v2(*args: Any, **kwargs: Any) -> Any:
    from optimus_cutedsl._cutlass_monkeypatch import apply_patches

    apply_patches()
    from optimus_cutedsl.silu_mul_group_masked_v2 import silu_mul_group_masked_v2 as _impl

    return _impl(*args, **kwargs)


def silu_mul_group_quant_fp8_masked_v2(*args: Any, **kwargs: Any) -> Any:
    from optimus_cutedsl._cutlass_monkeypatch import apply_patches

    apply_patches()
    from optimus_cutedsl.silu_mul_group_quant_fp8_masked_v2 import (
        silu_mul_group_quant_fp8_masked_v2 as _impl,
    )

    return _impl(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        mod_name, attr = _EXPORTS[name]
        mod = import_module(mod_name)
        value = getattr(mod, attr)
        globals()[name] = value  # cache
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))

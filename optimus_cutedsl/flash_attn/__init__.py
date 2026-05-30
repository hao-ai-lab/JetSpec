"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_tree_paged_sm90",
]

_CUTE_PATCHED = False


def _ensure_cute_compile_patched() -> None:
    """Patch `cutlass.cute.compile` lazily to avoid import-time CUDA/toolchain side effects."""
    global _CUTE_PATCHED
    if _CUTE_PATCHED:
        return
    # Ensure CuTe fake-tensor runtime is patched as well.
    try:
        mp = import_module("optimus_cutedsl._cutlass_monkeypatch")
        apply = getattr(mp, "apply_patches", None)
        if callable(apply):
            apply()
    except Exception:
        # Best-effort: if CUTLASS isn't installed yet, we'll fail later when actually used.
        pass
    cute = import_module("cutlass.cute")
    cute_compile_patched = import_module(
        "optimus_cutedsl.flash_attn.cute_dsl_utils"
    ).cute_compile_patched
    cute.compile = cute_compile_patched
    _CUTE_PATCHED = True


def __getattr__(name: str) -> Any:
    if name in {"flash_attn_func", "flash_attn_varlen_func"}:
        _ensure_cute_compile_patched()
        interface = import_module("optimus_cutedsl.flash_attn.interface")
        value = getattr(interface, name)
        globals()[name] = value  # cache
        return value
    if name == "flash_attn_varlen_tree_paged_sm90":
        _ensure_cute_compile_patched()
        module = import_module("optimus_cutedsl.flash_attn.flash_fwd_sm90_paged_tree")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))

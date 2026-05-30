"""optimus — reachability seam for the vendored tail-tree attention kernel.

The `optimus_cutedsl` package (vendored at the repo root) provides the SM90 /
Hopper paged tail-tree attention kernel ``flash_attn_varlen_tree_paged_sm90`` —
the kernel behind the paper's wall-clock TPS. It is **opt-in**: ``import ptd`` and
the default SDPA tree-verify path never touch it, so the package installs and
imports cleanly on hardware that can't run the kernel.

Resolve the kernel only through this module. It gates on GPU capability *before*
importing, so a non-Hopper box gets a clear error instead of a deep CUTLASS
ImportError (importing the kernel pulls ``cutlass.cute``). Install the kernel
dependencies with ``pip install -e '.[optimus]'`` (Hopper hardware).
"""
from __future__ import annotations

# SM90 == Hopper (H100/H200). The tail-tree kernel has no SM100/Blackwell variant,
# so a B200 (SM100) can import the package but cannot launch the kernel.
_REQUIRED_MAJOR_CC = 9


def optimus_available() -> bool:
    """Whether the optimus tail-tree kernel can be imported here (never raises).

    Does not check GPU capability — use :func:`load_optimus_tree_kernel` for the
    gated resolve. Intended for capability probes and test skips.
    """
    try:
        from optimus_cutedsl.flash_attn import (  # noqa: F401
            flash_attn_varlen_tree_paged_sm90,
        )
    except Exception:
        return False
    return True


def load_optimus_tree_kernel():
    """Return the optimus SM90 paged tail-tree attention callable.

    Raises a clear error on non-Hopper hardware *before* importing the kernel
    (the import otherwise pulls ``cutlass.cute`` and fails opaquely), and a
    helpful install hint if the optimus extra is not installed.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("tree_attn_kernel='optimus' requires a CUDA device.")
    major, _ = torch.cuda.get_device_capability(0)
    if major != _REQUIRED_MAJOR_CC:
        raise RuntimeError(
            f"optimus tail-tree kernel is Hopper (SM90) only; this device is "
            f"SM{major}0. Run on H100/H200, or use the default SDPA tree path."
        )
    try:
        from optimus_cutedsl.flash_attn import flash_attn_varlen_tree_paged_sm90
    except ImportError as exc:
        raise ImportError(
            "optimus kernel deps missing — install with: pip install -e '.[optimus]' "
            "(nvidia-cutlass-dsl==4.3.0.dev0 + cuda-python, on Hopper hardware)."
        ) from exc
    return flash_attn_varlen_tree_paged_sm90

"""Temporary compatibility patches for CUTLASS/CuTe python runtime files.

The current `nvidia-cutlass-dsl` drop that we ship with Optimus is missing
implementations of ``mark_layout_dynamic`` and ``mark_compact_shape_dynamic``
for ``cutlass.cute.runtime._FakeTensor``.  The TVM-FFI compilation path uses
``make_fake_tensor`` to describe kernel arguments, so the missing methods
cause attribute errors during compilation.  To unblock development we inject
lightweight implementations that emulate the subset of the behaviour we need.
"""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path
from typing import Iterable, Sequence, Tuple


_CUTLASS_RUNTIME_PATH = (
    Path(sysconfig.get_paths()["purelib"])
    / "nvidia_cutlass_dsl"
    / "python_packages"
    / "cutlass"
    / "cute"
    / "runtime.py"
)
_CUTLASS_BASE_DSL_PATH = (
    Path(sysconfig.get_paths()["purelib"])
    / "nvidia_cutlass_dsl"
    / "python_packages"
    / "cutlass"
    / "base_dsl"
    / "dsl.py"
)

_CUTLASS_OLD_BLOCK = """        if hasattr(tensor, "__dlpack_device__") and not hasattr(tensor, "__dlpack__"):
            self._dlpack_data = tensor.__dlpack_device__()
        else:
            try:
                # we expect no stream sync. Because torch has different default behavior
                # for stream parameter on different version.
                # we need to explicitly pass -1 to achieve no sync effects.
                self._dlpack_data = tensor.__dlpack__(stream=-1)
            except Exception:
                self._dlpack_data = tensor.__dlpack__()
        if enable_tvm_ffi:
            import tvm_ffi

            self._tvm_ffi_tensor = tvm_ffi.from_dlpack(tensor)
            self._dlpack_data = self._tvm_ffi_tensor.__dlpack__()"""

_CUTLASS_OLD_BLOCK_WITH_BLANK = """        if hasattr(tensor, "__dlpack_device__") and not hasattr(tensor, "__dlpack__"):
            self._dlpack_data = tensor.__dlpack_device__()
        else:
            try:
                # we expect no stream sync. Because torch has different default behavior
                # for stream parameter on different version.
                # we need to explicitly pass -1 to achieve no sync effects.
                self._dlpack_data = tensor.__dlpack__(stream=-1)
            except Exception:
                self._dlpack_data = tensor.__dlpack__()

        if enable_tvm_ffi:
            import tvm_ffi

            self._tvm_ffi_tensor = tvm_ffi.from_dlpack(tensor)
            self._dlpack_data = self._tvm_ffi_tensor.__dlpack__()"""

_CUTLASS_NEW_BLOCK = """        if hasattr(tensor, "__dlpack_device__") and not hasattr(tensor, "__dlpack__"):
            self._dlpack_data = tensor.__dlpack_device__()
        elif enable_tvm_ffi:
            import tvm_ffi

            self._tvm_ffi_tensor = tvm_ffi.from_dlpack(tensor)
            self._dlpack_data = self._tvm_ffi_tensor.__dlpack__()
        else:
            try:
                # we expect no stream sync. Because torch has different default behavior
                # for stream parameter on different version.
                # we need to explicitly pass -1 to achieve no sync effects.
                self._dlpack_data = tensor.__dlpack__(stream=-1)
            except Exception:
                self._dlpack_data = tensor.__dlpack__()"""
_CUTLASS_BASE_DSL_REPLACEMENTS = (
    (
        """        with ir.Context(), self.get_ir_location(location):""",
        """        with ir.Context() as ctx, self.get_ir_location(location):
            # Disable threading as a temporary workaround for segfault issue.
            ctx.enable_multithreading(False)""",
    ),
    (
        """        with ir.Context(), self.get_location(frame):""",
        """        with ir.Context() as ctx, self.get_location(frame):
            # Disable threading as a temporary workaround for segfault issue.
            ctx.enable_multithreading(False)""",
    ),
    (
        """        with ir.Context(), self.get_ir_location() as loc:""",
        """        with ir.Context() as ctx, self.get_ir_location() as loc:
            # Disable threading as a temporary workaround for segfault issue.
            ctx.enable_multithreading(False)""",
    ),
    (
        """        with ir.Context(), self.get_location():""",
        """        with ir.Context() as ctx, self.get_location():
            # Disable threading as a temporary workaround for segfault issue.
            ctx.enable_multithreading(False)""",
    ),
)


def patch_torch_dtype_compat() -> bool:
    """Backfill torch dtypes expected by CUTLASS TVM-FFI bindings."""
    try:
        import torch
    except Exception:
        return False

    # Some torch builds don't expose these symbols, while TVM-FFI dtype maps
    # still expect them during module import.
    fallback_map = {
        "uint16": "int32",
        "uint32": "int64",
        "uint64": "int64",
        "float8_e4m3fn": "float16",
        "float8_e4m3fnuz": "float16",
        "float8_e5m2": "float16",
        "float8_e5m2fnuz": "float16",
    }
    for name, fallback in fallback_map.items():
        if not hasattr(torch, name):
            setattr(torch, name, getattr(torch, fallback))
    return True


def _deduce_stride_order(stride: Sequence[int]) -> Tuple[int, ...]:
    """Reproduce torch.Tensor.dim_order() for compact layouts."""
    indexed = list(enumerate(stride))
    # Sort by stride magnitude from outermost (largest stride) to innermost.
    indexed.sort(key=lambda pair: pair[1], reverse=True)
    return tuple(idx for idx, _ in indexed)


def _needs_patch(fake_tensor_cls) -> bool:
    try:
        from cutlass.cute.typing import Tensor as _CuteTensor  # type: ignore
    except Exception:
        return False
    base_layout = getattr(_CuteTensor, "mark_layout_dynamic", None)
    base_compact = getattr(_CuteTensor, "mark_compact_shape_dynamic", None)
    return (
        getattr(fake_tensor_cls, "mark_layout_dynamic", None) is base_layout
        or getattr(fake_tensor_cls, "mark_compact_shape_dynamic", None) is base_compact
    )


def patch_fake_tensor_dynamic_methods() -> None:
    """Install ``mark_layout_dynamic``/``mark_compact_shape_dynamic`` on _FakeTensor.

    Returns:
        True if the patch is applied or determined to be unnecessary (no need to
        retry in this process). False if the patch couldn't be evaluated because
        CUTLASS/CuTe isn't importable (callers may retry later).
    """
    try:
        from cutlass.cute.runtime import _FakeTensor  # type: ignore
        from cutlass.cute.typing import SymInt, sym_int64
    except Exception:
        return False

    if getattr(_FakeTensor, "_optimus_dynamic_patch", False):
        return True
    if not _needs_patch(_FakeTensor):
        return True

    # ``cutlass.cute.typing.SymInt`` is only used for ``isinstance`` checks.
    SymIntTuple = (SymInt,)

    def _coerce_sym_int(value, divisibility: int):
        if isinstance(value, SymIntTuple):
            return value
        return sym_int64(divisibility=divisibility)

    def mark_layout_dynamic(self, leading_dim: int | None = None):
        if leading_dim is None:
            stride_ones = [idx for idx, val in enumerate(self._stride) if val == 1]
            if len(stride_ones) != 1:
                raise ValueError("Unable to deduce leading_dim for non-compact layout.")
            leading_dim = stride_ones[0]
        if not (0 <= leading_dim < len(self._stride)):
            raise ValueError(f"leading_dim {leading_dim} out of range.")
        if self._stride[leading_dim] != 1:
            raise ValueError(
                f"Leading dimension {leading_dim} must have stride 1, "
                f"got {self._stride[leading_dim]}."
            )
        self._optimus_leading_dim = leading_dim
        return self

    def mark_compact_shape_dynamic(
        self,
        mode: int,
        stride_order: Iterable[int] | None = None,
        divisibility: int = 1,
    ):
        if not (0 <= mode < len(self._shape)):
            raise ValueError(f"mode {mode} out of range for shape rank {len(self._shape)}.")
        if stride_order is None:
            stride_order = _deduce_stride_order(self._stride)
        else:
            stride_order = tuple(stride_order)
        if mode not in stride_order:
            raise ValueError(f"mode {mode} missing from stride_order {stride_order}.")

        shape = list(self._shape)
        shape[mode] = _coerce_sym_int(shape[mode], divisibility)
        self._shape = tuple(shape)

        dynamic_prefix = stride_order[: stride_order.index(mode)]
        stride = list(self._stride)
        for dim in dynamic_prefix:
            stride[dim] = _coerce_sym_int(stride[dim], divisibility)
        self._stride = tuple(stride)
        self._optimus_stride_order = stride_order
        return self

    _FakeTensor.mark_layout_dynamic = mark_layout_dynamic  # type: ignore[attr-defined]
    _FakeTensor.mark_compact_shape_dynamic = mark_compact_shape_dynamic  # type: ignore[attr-defined]
    _FakeTensor._optimus_dynamic_patch = True  # type: ignore[attr-defined]
    return True


def patch_cutlass_runtime() -> bool:
    if not os.path.isfile(_CUTLASS_RUNTIME_PATH):
        return False
    try:
        content = Path(_CUTLASS_RUNTIME_PATH).read_text(encoding="utf-8")
    except OSError:
        return False

    if _CUTLASS_NEW_BLOCK in content:
        return True

    if _CUTLASS_OLD_BLOCK in content:
        updated = content.replace(_CUTLASS_OLD_BLOCK, _CUTLASS_NEW_BLOCK, 1)
    elif _CUTLASS_OLD_BLOCK_WITH_BLANK in content:
        updated = content.replace(_CUTLASS_OLD_BLOCK_WITH_BLANK, _CUTLASS_NEW_BLOCK, 1)
    else:
        return True

    try:
        Path(_CUTLASS_RUNTIME_PATH).write_text(updated, encoding="utf-8")
    except OSError:
        return False
    return True


def patch_cutlass_base_dsl() -> bool:
    if not os.path.isfile(_CUTLASS_BASE_DSL_PATH):
        return False
    try:
        content = Path(_CUTLASS_BASE_DSL_PATH).read_text(encoding="utf-8")
    except OSError:
        return False

    updated = content
    changed = False

    for old_block, new_block in _CUTLASS_BASE_DSL_REPLACEMENTS:
        if new_block in updated:
            continue
        if old_block in updated:
            updated = updated.replace(old_block, new_block, 1)
            changed = True

    if changed:
        try:
            Path(_CUTLASS_BASE_DSL_PATH).write_text(updated, encoding="utf-8")
        except OSError:
            return False

    # Consider this patch successful only when threading-disable marker exists.
    # This avoids silently reporting success on unrecognized CUTLASS variants.
    return "ctx.enable_multithreading(False)" in updated


_OPTIMUS_PATCHES_APPLIED = False


def apply_patches() -> None:
    """Apply all Optimus monkeypatches (idempotent, cached per-process)."""
    global _OPTIMUS_PATCHES_APPLIED
    if _OPTIMUS_PATCHES_APPLIED:
        return
    done = (
        patch_torch_dtype_compat()
        and patch_fake_tensor_dynamic_methods()
        and patch_cutlass_runtime()
        and patch_cutlass_base_dsl()
    )
    # Only mark as applied when we've either successfully patched, or determined
    # that patching is unnecessary. If CUTLASS/CuTe isn't importable yet, allow
    # a future retry in the same process.
    if done:
        _OPTIMUS_PATCHES_APPLIED = True


# NOTE:
# Do NOT apply patches at import time.
#
# This package is often imported in multi-process runtimes (e.g. vLLM). Import-time
# side effects that transitively import/initialize CUDA toolchains can break forked
# workers (CUDA init must typically happen after process creation).
#
# Call `apply_patches()` explicitly (we do this from lazy import shims in
# `optimus_cutedsl/__init__.py` and `optimus_cutedsl/flash_attn/__init__.py`).

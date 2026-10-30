"""Weight-only NVFP4 (W4A16) GEMM for small M on Blackwell, in Triton.

The pack keeps `quantize_weight`'s E2M1 codes, E4M3 block scales and FP32
global descale. Only the nibble order inside each 4-byte group changes, so
that one `cvt.rn.f16x2.e2m1x2` per byte yields the even-K or odd-K
weights directly. The kernel computes `x @ dequantize(pack).T` with FP32
accumulation; the activation is never quantized.

Triton is imported on the first CUDA call; without it the pack, its round trip
and the reference stay usable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .format import NVFP4Tensor, quantize_weight, swizzle_scales

# Nibble order inside each 4-byte group: bytes hold codes (0,2), (4,6), (1,3), (5,7).
_GROUP_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
_GROUP_INVERSE = tuple(_GROUP_ORDER.index(i) for i in range(8))
DECODES = ("f16", "bf16", "int", "triton")
MAX_M = 128
# f16 decode: the activation is scaled by 2**-4 before its f16 conversion.
X_SHIFT = 4


@dataclass(frozen=True)
class W4A16Config:
    block_n: int = 128
    block_k: int = 128
    split_k: int = 1
    num_warps: int = 4
    num_stages: int = 4
    decode: str = "bf16"
    prefetch: int = 0
    tma: bool = False
    ablate: int = 0  # diagnostics only: 1 skips the decode, 2 skips the MMA (wrong results)

    def __post_init__(self):
        if self.decode not in DECODES:
            raise ValueError(f"decode must be one of {DECODES}")
        if self.block_k % 32 or self.block_k & (self.block_k - 1):
            raise ValueError("block_k must be a power of two and a multiple of 32")
        if self.block_n & (self.block_n - 1) or self.block_n < 16:
            raise ValueError("block_n must be a power of two of at least 16")
        if self.split_k < 1:
            raise ValueError("split_k must be positive")
        if self.prefetch not in (0, 1, 2):
            raise ValueError("prefetch must be 0 (Triton pipeliner), 1 or 2 tiles ahead")
        if self.tma and self.prefetch:
            raise ValueError("tma loads use the Triton pipeliner (prefetch=0)")


@dataclass
class W4A16Pack:
    """Kernel-order packed codes `uint8 [N,K/2]`, E4M3 scales `[N,K/16]`, FP32 descale."""

    data: torch.Tensor
    scales: torch.Tensor
    global_scale: torch.Tensor
    states: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.data.shape[0], self.data.shape[1] * 2)

    @property
    def nbytes(self) -> int:
        return self.data.numel() + self.scales.numel()


def _permute_codes(data: torch.Tensor, order) -> torch.Tensor:
    data = data.view(torch.uint8)
    rows, half = data.shape
    if half % 4:
        raise ValueError("W4A16 needs K divisible by 16")
    codes = torch.stack((data & 15, data >> 4), dim=-1).reshape(rows, half // 4, 8)
    codes = codes[..., list(order)]
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).reshape(rows, half).contiguous()


def pack_from_nvfp4(value: NVFP4Tensor) -> W4A16Pack:
    """Re-lay out an `NVFP4Tensor`; codes, scales and descale are unchanged."""
    return W4A16Pack(_permute_codes(value.data, _GROUP_ORDER),
                     value.block_scales.contiguous(), value.global_scale.float().reshape(1).contiguous())


def pack_w4a16(weight_bf16: torch.Tensor, *, chunk_rows: int = 256) -> W4A16Pack:
    """Load-time conversion through `quantize_weight` (may synchronize on CUDA)."""
    return pack_from_nvfp4(quantize_weight(weight_bf16, chunk_rows=chunk_rows))


def to_nvfp4(pack: W4A16Pack) -> NVFP4Tensor:
    """Inverse re-layout, for checks and the `dequantize` reference."""
    data = _permute_codes(pack.data, _GROUP_INVERSE)
    return NVFP4Tensor(data, pack.scales, swizzle_scales(pack.scales), pack.global_scale)


def num_sms(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def block_m(m: int) -> int:
    if not 1 <= m <= MAX_M:
        raise ValueError(f"W4A16 supports 1 <= M <= {MAX_M}")
    return max(16, 1 << (m - 1).bit_length())


def _tuned(block_n, block_k, split_k, num_warps, bucket=16):
    if bucket == 128:  # a 128-token x tile at block_k 256 exceeds shared memory
        return W4A16Config(block_n=block_n, block_k=128, split_k=split_k, num_warps=num_warps,
                           num_stages=3, tma=True)
    return W4A16Config(block_n=block_n, block_k=block_k, split_k=split_k, num_warps=num_warps,
                       num_stages=4 if block_k == 128 else 3, tma=True)


# GB200 sweep winners (TMA, bf16 decode) keyed by (N, K, M bucket); M 16 winners
# also serve bucket 32 and M 64 winners bucket 128. Other shapes use the heuristic.
_WINNERS = {(6144, 4096): ((64, 256, 4, 4), (128, 128, 6, 8)),
            (4096, 4096): ((128, 256, 8, 4), (64, 256, 4, 4)),
            (24576, 4096): ((64, 256, 1, 4), (128, 256, 2, 8)),
            (4096, 12288): ((128, 256, 8, 4), (128, 256, 8, 8)),
            (151936, 4096): ((128, 256, 1, 4), (128, 128, 1, 4)),
            (4096, 20480): ((128, 256, 8, 4), (128, 256, 8, 8))}
TUNED: dict[tuple[int, int, int], W4A16Config] = {
    (n, k, bucket): _tuned(*(small if bucket <= 32 else large), bucket=bucket)
    for (n, k), (small, large) in _WINNERS.items() for bucket in (16, 32, 64, 128)}


def default_config(n: int, k: int, m: int, sms: int = 148) -> W4A16Config:
    tuned = TUNED.get((n, k, block_m(m)))
    if tuned is not None:
        return tuned
    block_n = 128
    block_k = 256 if k % 256 == 0 and block_m(m) < 128 else 128
    tiles = -(-n // block_n)
    k_tiles = k // block_k
    split = 1
    while tiles * split < 2 * sms and split * 2 <= max(1, k_tiles // 2):
        split *= 2
    return W4A16Config(block_n=block_n, block_k=block_k, split_k=split,
                       num_stages=4 if block_k == 128 else 3, tma=k % 32 == 0)


def _check(x: torch.Tensor, pack: W4A16Pack, config: W4A16Config):
    n, k = pack.shape
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[1] != k:
        raise ValueError(f"x must be bf16 [M, {k}]")
    if x.stride(1) != 1:
        raise ValueError("x must be contiguous along K")
    if k % config.block_k:
        raise ValueError(f"K={k} must be divisible by block_k={config.block_k}")
    if config.split_k > k // config.block_k:
        raise ValueError("split_k exceeds the number of K tiles")
    block_m(x.shape[0])
    if not x.is_cuda or pack.data.device != x.device:
        raise RuntimeError("w4a16_linear needs CUDA tensors on one device")


class _State:
    """Split-K workspace and arrival counters, allocated before capture."""

    def __init__(self, pack: W4A16Pack, m: int, config: W4A16Config, device):
        n, _ = pack.shape
        tiles = -(-n // config.block_n)
        split = config.split_k
        self.workspace = torch.empty((split * m * n,) if split > 1 else (1,),
                                     dtype=torch.float32, device=device)
        self.counters = torch.zeros((tiles,), dtype=torch.int32, device=device)
        self.dummy_bias = torch.zeros((1,), dtype=torch.bfloat16, device=device)
        self.q_desc = self.s_desc = None
        if config.tma:
            from triton.tools.tensor_descriptor import TensorDescriptor
            self.q_desc = TensorDescriptor.from_tensor(pack.data, [config.block_n, config.block_k // 2])
            if config.block_k // 16 >= 16:
                self.s_desc = TensorDescriptor.from_tensor(pack.scales.view(torch.uint8),
                                                           [config.block_n, config.block_k // 16])
        self.kernel = None  # the last compiled Triton kernel, for inspection


def _state(pack: W4A16Pack, m: int, config: W4A16Config, device) -> _State:
    key = (str(device), m, config)
    state = pack.states.get(key)
    if state is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"warm up W4A16 M={m} {config} before CUDA graph capture")
        state = pack.states[key] = _State(pack, m, config, device)
    return state


def w4a16_linear(x: torch.Tensor, pack: W4A16Pack, bias: torch.Tensor | None = None,
                 out: torch.Tensor | None = None, *, config: W4A16Config | None = None) -> torch.Tensor:
    """`x @ dequantize(pack).T (+ bias)` in BF16, FP32 accumulation, M <= 128.

    Graph-safe after `w4a16_warmup` for the same M and config: no host sync and,
    when `out` is given, no allocation. Calls sharing one pack must not overlap.
    """
    n, k = pack.shape
    m = x.shape[0] if x.ndim == 2 else 0
    if config is None:
        config = default_config(n, k, max(m, 1), num_sms(x.device) if x.is_cuda else 148)
    _check(x, pack, config)
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    elif out.shape != (m, n) or out.dtype != torch.bfloat16 or out.stride(1) != 1:
        raise ValueError(f"out must be bf16 [{m}, {n}] contiguous along N")
    if bias is not None and (bias.shape != (n,) or bias.dtype != torch.bfloat16 or not bias.is_contiguous()):
        raise ValueError(f"bias must be contiguous bf16 [{n}]")
    from ._w4a16_kernel import _w4a16_kernel
    state = _state(pack, m, config, x.device)
    grid = (-(-n // config.block_n), config.split_k)
    x_desc = None
    if config.tma:
        if x.data_ptr() % 16 or (x.stride(0) * 2) % 16:
            raise ValueError("tma configs need a 16-byte aligned x with a 16-byte aligned row stride")
        from triton.tools.tensor_descriptor import TensorDescriptor
        x_desc = TensorDescriptor(x, list(x.shape), list(x.stride()), [block_m(m), config.block_k])
    state.kernel = _w4a16_kernel[grid](
        x, pack.data, pack.scales.view(torch.uint8), pack.global_scale,
        bias if bias is not None else state.dummy_bias, out, state.workspace, state.counters,
        x_desc, state.q_desc, state.s_desc, m, n, k, x.stride(0), out.stride(0),
        HAS_BIAS=bias is not None, SPLIT_K=config.split_k, BLOCK_M=block_m(m),
        BLOCK_N=config.block_n, BLOCK_K=config.block_k, DECODE=DECODES.index(config.decode),
        X_SCALE=2. ** -X_SHIFT, OUT_SHIFT=2. ** X_SHIFT, NUM_STAGES=config.num_stages,
        PREFETCH=config.prefetch, USE_TMA=config.tma, SCALE_TMA=state.s_desc is not None,
        ABLATE=config.ablate, num_warps=config.num_warps)
    return out


def w4a16_warmup(pack: W4A16Pack, m: int, *, config: W4A16Config | None = None,
                 x: torch.Tensor | None = None, bias: torch.Tensor | None = None,
                 out: torch.Tensor | None = None, iterations: int = 2) -> W4A16Config:
    """Compile and allocate for one exact M before capture; returns the config used."""
    n, k = pack.shape
    device = pack.data.device
    if config is None:
        config = default_config(n, k, m, num_sms(device))
    if x is None:
        x = torch.zeros((m, k), dtype=torch.bfloat16, device=device)
    for _ in range(iterations):
        w4a16_linear(x, pack, bias, out, config=config)
    return config


def reference(x: torch.Tensor, pack: W4A16Pack) -> torch.Tensor:
    """FP32 `x @ dequantize(pack).T`, the numerical yardstick."""
    from .format import dequantize
    return x.float() @ dequantize(to_nvfp4(pack), dtype=torch.float32).T

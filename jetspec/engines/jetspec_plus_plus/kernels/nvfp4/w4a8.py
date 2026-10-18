"""W4A8 small-M GEMM on Blackwell: MXFP4 weights x MXFP8 activations on the tensor core.

The weight pack holds E2M1 codes (low nibble = even K) and one UE8M0 scale per
32 K elements, the OCP MX layout. Each call quantizes the BF16 activation to
FP8 E4M3 with one UE8M0 scale per row and per 32 K elements. The GEMM is the K1
swap-AB kernel (`_k1_kernel.py`) built for `tcgen05.mma.kind::mxf8f6f4`. The
weights are the A operand (128 rows per CTA) and the tokens are the B operand.
Its TMA unpacks the FP4 weights into byte containers in shared memory, as the
mixed FP8 x FP4 MMA kind needs. HBM traffic stays 4 bits per weight.

Triton and CuTe DSL are imported on the first CUDA call. The pack, the torch
quantizers and `fq_w4a8_reference` work on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

from .format import pack_e2m1, swizzle_scales, unpack_e2m1

MAX_M = 256
TOKEN_TILES = (8, 16, 32, 64, 80, 128, 256)
SPLITS = (1, 2, 3, 4, 6, 8)
# E8M0 bytes are clamped so that the scale and its inverse are normal FP32 powers of two.
_BYTE_MIN, _BYTE_MAX = 1, 253
# Element format maxima: E2M1 6 = 0.75 * 2**3, E4M3 448 = 0.875 * 2**9. The scale
# exponent is ceil(log2(amax / max)), read from the FP32 bits of amax (no division).
_FP4 = (2, 0x400000)
_FP8 = (8, 0x600000)
_SMEM_BYTES = 232448


def _e8m0(amax: torch.Tensor, element) -> torch.Tensor:
    """Smallest power-of-two scale with amax / scale <= the element maximum, as a UE8M0 byte."""
    shift, fraction_limit = element
    bits = amax.float().contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    byte = exponent - shift + ((bits & 0x7FFFFF) > fraction_limit).to(torch.int32)
    return byte.clamp(_BYTE_MIN, _BYTE_MAX).to(torch.uint8)


def e8m0_to_float(byte: torch.Tensor) -> torch.Tensor:
    """Exact 2**(byte - 127) for bytes in the clamped range."""
    return (byte.to(torch.int32) << 23).view(torch.float32)


def _inverse(byte: torch.Tensor) -> torch.Tensor:
    return ((254 - byte.to(torch.int32)) << 23).view(torch.float32)


@dataclass(frozen=True)
class W4A8Config:
    tile_n: int = 16
    cta_k: int = 128
    split: int = 1
    stages: int = 0  # 0 = the most stages that fit in shared memory
    reduction: str = "auto"
    pdl: bool = True
    backend: str = "k1"  # "k1": quantizer + CuTe K1 kernel; "triton": one fused tl.dot_scaled launch
    prefetch: bool = False  # k1 backend: warp 7 prefetches the CTA's weight slab into L2 at launch
    warps: int = 4  # triton backend only
    fused: bool = False  # k1 backend: the GEMM quantizes each activation k-tile itself (no quantizer launch)

    def __post_init__(self):
        if self.backend not in ("k1", "triton"):
            raise ValueError("backend must be k1 or triton")
        if self.tile_n not in TOKEN_TILES:
            raise ValueError(f"tile_n must be one of {TOKEN_TILES}")
        if self.cta_k not in (128, 256):
            raise ValueError("MX block-scaled tiles need cta_k 128 or 256")
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if not 0 <= self.stages <= 12:
            raise ValueError("stages must be 0 (auto) or 1 to 12")
        if self.reduction not in ("auto", "dsmem", "workspace"):
            raise ValueError("reduction must be auto, dsmem or workspace")
        if self.fused and (self.backend != "k1" or self.tile_n > 128):
            raise ValueError("the fused quantizer needs the k1 backend and a token tile of at most 128")

    def resolved_reduction(self) -> str:
        if self.split == 1:
            return "dsmem"
        if self.reduction == "auto":
            return "dsmem" if self.tile_n % self.split == 0 else "workspace"
        if self.reduction == "dsmem" and self.tile_n % self.split:
            raise ValueError("DSMEM reduction needs split to divide the token tile")
        return self.reduction


def smem_bytes(config: W4A8Config, stages: int) -> int:
    """Mirror `_k1_kernel._smem_bytes` for FP4 x FP8 with UE8M0 scales per 32."""
    def align(value, alignment):
        return (value + alignment - 1) // alignment * alignment
    cluster_split = config.split if config.resolved_reduction() == "dsmem" else 1
    mailbox = max(cluster_split - 1, 0) * 128 * (config.tile_n // cluster_split)
    cursor = align(16 * stages + 16 + 16 + 8 + 8 + 4, 8) + 8 + mailbox * 4
    sf_k = config.cta_k // 32
    for size in (128 * config.cta_k * stages, config.tile_n * config.cta_k * stages,
                 128 * sf_k * stages, config.tile_n * sf_k * stages,
                 max(128, config.tile_n) * sf_k * stages):
        cursor = align(cursor, 128) + size
    return cursor


def max_stages(config: W4A8Config, ctas_per_sm: int = 1) -> int:
    """Most pipeline stages that let `ctas_per_sm` CTAs share one SM's shared memory."""
    budget = (_SMEM_BYTES + 1024) // ctas_per_sm - 1024
    fitting = [s for s in range(1, 13) if smem_bytes(config, s) + 1024 <= budget]
    if not fitting:
        raise ValueError(f"no stage count fits shared memory for {config}")
    return fitting[-1]


def resolved_stages(config: W4A8Config) -> int:
    return config.stages or max_stages(config)


# GB200 cold-L2 graph sweep winners, keyed by (N, K, token tile).
TUNED: dict[tuple[int, int, int], W4A8Config] = {
    (24576, 4096, 16): W4A8Config(16, 128, 1, 5),
    (24576, 4096, 64): W4A8Config(64, 128, 1, 4),
    (4096, 12288, 16): W4A8Config(16, 256, 4, 5),
    (151936, 4096, 16): W4A8Config(16, 128, 1, 5),
    (151936, 4096, 64): W4A8Config(64, 128, 1, 4),
    (4096, 20480, 16): W4A8Config(16, 256, 4, 5),
    # B200 small-M sweep winners (serving): per-call graph time over 9 cold weights, separate
    # quantizer. Comments: W4A8 us / BF16 us at the bucket's top M (bucket 8: the worse of
    # M 1 and 4), then the split heuristic it replaces. A tile below the bucket repeats over M
    # with one scale atom per tile.
    (6144, 4096, 8): W4A8Config(8, 256, 2),  # qkv 8.2/13.0 = 0.63, was 0.86
    (6144, 4096, 16): W4A8Config(16, 128, 2),  # 8.9/13.9 = 0.64, was 0.89
    (6144, 4096, 32): W4A8Config(32, 256, 3),  # 9.8/14.3 = 0.68, was 1.08
    (6144, 4096, 64): W4A8Config(32, 128, 1),  # 10.5/13.4 = 0.78, was 1.51
    (6144, 4096, 80): W4A8Config(32, 128, 1),  # 10.7/16.4 = 0.65, was 1.35
    (4096, 4096, 32): W4A8Config(16, 128, 2),  # o 8.7/10.9 = 0.79, was 0.88
    (4096, 4096, 64): W4A8Config(16, 256, 1),  # 9.8/11.2 = 0.88, was 1.10
    (4096, 4096, 80): W4A8Config(32, 128, 1),  # 10.2/11.4 = 0.90, was 1.16
    (2048, 4096, 8): W4A8Config(8, 128, 4),  # kv 6.8/13.9 = 0.49, was 0.69
    (2048, 4096, 16): W4A8Config(16, 256, 4),  # 7.3/10.0 = 0.73, was 1.05
    (2048, 4096, 32): W4A8Config(16, 128, 4),  # 7.5/10.0 = 0.75, was 1.32
    (2048, 4096, 64): W4A8Config(16, 128, 2),  # 8.4/10.2 = 0.82, was 1.84
    (2048, 4096, 80): W4A8Config(80, 128, 6, reduction="workspace"),  # 8.9/10.5 = 0.85, was 1.89
    (10240, 4096, 8): W4A8Config(8, 128, 1),  # 10.0/20.0 = 0.50, was 0.66
    (10240, 4096, 16): W4A8Config(16, 128, 1),  # 10.3/19.6 = 0.52, was 0.73
    (10240, 4096, 32): W4A8Config(32, 128, 1),  # 10.9/19.8 = 0.55, was 0.89
    (10240, 4096, 64): W4A8Config(32, 128, 1, 5),  # 14.8/18.0 = 0.82, was 1.22
    (10240, 4096, 80): W4A8Config(80, 128, 1),  # 13.9/21.4 = 0.65, was 1.10
    (12288, 4096, 8): W4A8Config(8, 128, 1),  # 10.5/19.4 = 0.54, was 0.69
    (12288, 4096, 16): W4A8Config(16, 128, 1),  # 10.7/21.4 = 0.50, was 0.69
    (12288, 4096, 32): W4A8Config(32, 128, 1),  # 11.4/21.6 = 0.53, was 0.83
    (12288, 4096, 64): W4A8Config(32, 128, 1, 5),  # 15.3/21.9 = 0.70, was 1.02
    (12288, 4096, 80): W4A8Config(80, 128, 1),  # 13.9/22.8 = 0.61, was 1.05
    (24576, 4096, 8): W4A8Config(8, 128, 1, 6),  # gate_up 15.3/36.0 = 0.42, was 0.51
    (24576, 4096, 32): W4A8Config(32, 128, 1, 5),  # 16.6/36.0 = 0.46, was 0.54
    (24576, 4096, 80): W4A8Config(80, 128, 1, 4),  # 21.9/45.1 = 0.49, was 0.55
    (4096, 12288, 32): W4A8Config(32, 128, 4, reduction="workspace"),  # down 11.8/23.7 = 0.50, was 0.53
    (4096, 12288, 64): W4A8Config(64, 128, 4, reduction="workspace"),  # 13.4/24.1 = 0.56, GB200 6/4 was 0.63
    (4096, 12288, 80): W4A8Config(80, 128, 4, reduction="workspace"),  # 14.6/24.4 = 0.60, was 0.71
    (4096, 20480, 32): W4A8Config(32, 128, 4, reduction="workspace"),  # w_fc 14.8/32.3 = 0.46, was 0.48
    (4096, 20480, 64): W4A8Config(64, 128, 4, reduction="workspace"),  # 17.5/36.2 = 0.48, GB200 4/7 was 0.55
    (4096, 20480, 80): W4A8Config(80, 128, 4, reduction="workspace"),  # 18.9/36.6 = 0.52, was 0.58
    (1024, 4096, 32): W4A8Config(16, 128, 8, reduction="workspace"),  # 7.3/11.2 = 0.65, was 0.73
    (1024, 4096, 64): W4A8Config(16, 128, 4),  # 7.5/9.8 = 0.77, was 1.14
    (1024, 4096, 80): W4A8Config(32, 128, 6, reduction="workspace"),  # 8.0/12.1 = 0.66, was 0.96
    # B200 back-to-back sweep winners above 80 rows (serving), pinned for the routed entries below.
    (24576, 4096, 128): W4A8Config(128, 128, 1),
    (4096, 12288, 128): W4A8Config(128, 128, 4, reduction="workspace"),
    (4096, 20480, 128): W4A8Config(128, 128, 4, reduction="workspace"),
    (4096, 20480, 256): W4A8Config(128, 128, 2, reduction="workspace"),
}


# Draft GEMMs with more than SMALL_M rows take the kernel only for the (N, K, token bucket)
# entries below. Each one ran at most 0.95x of BF16 `F.linear` at both of its bucket's M
# (96 and 128, or 140 and 256) in the B200 back-to-back sweep, 9 calls over 3 weights,
# serving run m256t3; ratios are W4A8 us / BF16 us per call. Every other large-M GEMM stays
# on the BF16 copy of the MX-rounded weights. Measured but not routed (M 96, 128 | 140, 256):
#   qkv 6144x4096 0.87 1.10 | 1.18 1.25    o 4096x4096 1.00 1.08 | 1.22 1.23
#   gate_up 24576x4096 bucket 256: 1.07 1.04    down 4096x12288 bucket 256: 0.93 0.96
#   2048x4096 0.98 1.02 | 1.05 1.20    10240x4096 1.02 1.10 | 1.28 1.55
#   12288x4096 0.87 0.96 | 1.39 1.35
SMALL_M = 80
LARGE_M_ROUTED: frozenset[tuple[int, int, int]] = frozenset({
    (24576, 4096, 128),  # gate_up: 34.6/41.4 us = 0.84 at M 96, 35.5/40.3 = 0.88 at M 128
    (4096, 12288, 128),  # down: 16.4/24.4 = 0.67, 18.0/24.4 = 0.74
    (4096, 20480, 128),  # w_fc: 21.4/39.1 = 0.55, 23.7/37.3 = 0.63
    (4096, 20480, 256),  # w_fc: 32.1/42.8 = 0.75 at M 140, 38.7/46.0 = 0.84 at M 256
})

# Draft GEMMs with at most SMALL_M rows take the BF16 copy for the (N, K, token tile) entries
# below, where BF16 `F.linear` measured faster. The serving B200 sweep covered every census site
# shape (6144, 4096, 2048, 10240, 12288, 24576, 1024 x 4096; 4096 x 12288; 4096 x 20480;
# 151936 x 4096) at M 1, 4, 16, 32, 64 and 80: with TUNED above, the kernel ran at 0.27 to 0.90
# of BF16 in every bucket, so no entry is needed. Shapes outside the sweep keep the kernel.
SMALL_M_BF16_FASTER: frozenset[tuple[int, int, int]] = frozenset()

# B200 padded-N measurements at M 1 and 16: only these token buckets beat BF16.
PADDED_N_ROUTED: frozenset[tuple[int, int, int]] = frozenset({
    (48, 5120, 8), (48, 5120, 16), (96, 5120, 8), (96, 5120, 16),
})


def token_tile(m: int) -> int:
    return next(tile for tile in TOKEN_TILES if m <= tile)


def routes_to_kernel(n: int, k: int, m: int) -> bool:
    """True when the draft lane should run this GEMM on the kernel rather than its BF16 copy."""
    if n % 128:
        return (n, k, token_tile(m)) in PADDED_N_ROUTED
    if m <= SMALL_M:
        return (n, k, token_tile(m)) not in SMALL_M_BF16_FASTER
    return (n, k, token_tile(m)) in LARGE_M_ROUTED


def default_config(n: int, k: int, m: int, sms: int = 148) -> W4A8Config:
    """Read the tuned table, else the small-M split heuristic (M <= 80) or the large-M default."""
    tile = token_tile(m)
    tuned = TUNED.get((n, k, tile))
    if tuned is not None:
        return tuned
    tiles = -(-n // 128)
    k_tiles = k // 128
    if m > SMALL_M:
        return _large_default(tiles, k_tiles, tile, sms)
    split = 1
    for candidate in (2, 4, 8):
        if 4 * tiles * split >= 3 * sms or k_tiles // candidate < 4:
            break
        split = candidate
    return W4A8Config(tile_n=tile, cta_k=128, split=split)


def _large_default(tiles: int, k_tiles: int, bucket: int, sms: int) -> W4A8Config:
    """Above SMALL_M: a 128-row token tile repeated over the bucket and the largest
    workspace split that keeps every CTA in one wave. The B200 sweep (serving) measured
    256-row tiles and the DSMEM split-K reduction of 128-row tiles 1.3 to 4x slower."""
    ctas = tiles * (bucket // 128)
    fitting = [s for s in SPLITS if ctas * s <= sms and k_tiles // s >= 4]
    split = max(fitting, default=1)
    return W4A8Config(tile_n=128, cta_k=128, split=split, reduction="workspace" if split > 1 else "auto")


@dataclass
class W4A8Pack:
    """E2M1 codes `uint8 [N,K/2]`, logical UE8M0 scales `uint8 [N,K/32]` and their 128x4 layout."""

    data: torch.Tensor
    scales: torch.Tensor
    swizzled_scales: torch.Tensor
    states: dict = field(default_factory=dict, repr=False)
    logical_n: int | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (self.logical_n or self.data.shape[0], self.data.shape[1] * 2)

    @property
    def nbytes(self) -> int:
        """Bytes the GEMM reads per call: packed codes plus physical scales."""
        return self.data.numel() + self.swizzled_scales.numel()


@dataclass(frozen=True)
class W4A8ActivationLayout:
    """Producer contract for one exact M and the selected K1 GEMM configuration.

    Scales are UE8M0 bytes: each byte encodes 2**(byte-127) for 32 K
    values. Physical storage consists of 128-row by 4-block (512-byte)
    scale atoms; atom_rows is the number of logical rows owned by each.
    """

    config: W4A8Config
    data_shape: tuple[int, int]
    scale_shape: tuple[int, int]
    token_tile: int
    atom_rows: int
    data_dtype: torch.dtype = torch.float8_e4m3fn
    scale_dtype: torch.dtype = torch.uint8
    scale_format: str = 'ue8m0'
    scale_k: int = 32
    scale_atom_shape: tuple[int, int] = (128, 4)


def activation_layout(pack: W4A8Pack, m: int, *,
                      config: W4A8Config | None = None) -> W4A8ActivationLayout:
    """Describe producer buffers for ``w4a8_linear(..., x_fp8=..., x_scale=...)``.

    Uses exactly the GEMM's default config unless one is supplied. Allocate
    contiguous buffers on pack.data.device and use the returned config at the
    consuming call. CPU packs can inspect the layout without Triton or CUDA.
    """
    if not 1 <= m <= MAX_M:
        raise ValueError(f'W4A8 supports M from 1 to {MAX_M}')
    n, k = pack.shape
    config = config or default_config(n, k, m, _sms(pack.data.device) if pack.data.is_cuda else 148)
    if config.backend != 'k1' or config.fused:
        raise ValueError('pre-quantized activations require K1 with fused=False')
    atoms = -(-m // config.tile_n) * -(-config.tile_n // 128)
    return W4A8ActivationLayout(config, (m, k), (atoms * 128, -(-(k // 32) // 4) * 4),
                                config.tile_n, min(config.tile_n, 128))


def _check_weight(weight: torch.Tensor) -> None:
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise TypeError("W4A8 packs a bfloat16 [N,K] weight")
    n, k = weight.shape
    if n <= 0 or k <= 0 or k % 128:
        raise ValueError("W4A8 needs positive N and K divisible by 128")


@torch.no_grad()
def pack_w4a8(weight: torch.Tensor, *, chunk_rows: int = 256) -> W4A8Pack:
    """Load-time MXFP4 conversion: per 32 K, scale 2**ceil(log2(amax/6)), E2M1 with RNE."""
    _check_weight(weight)
    n, k = weight.shape
    padded_n = -(-n // 128) * 128
    data = torch.zeros((padded_n, k // 2), dtype=torch.uint8, device=weight.device)
    scales = torch.ones((padded_n, k // 32), dtype=torch.uint8, device=weight.device)
    for start in range(0, n, chunk_rows):
        blocks = weight[start:start + chunk_rows].float().reshape(-1, k // 32, 32)
        if not bool(torch.isfinite(blocks).all()):
            raise ValueError("W4A8 weight must be finite")
        byte = _e8m0(blocks.abs().amax(-1), _FP4)
        values = blocks * _inverse(byte).unsqueeze(-1)
        stop = min(start + chunk_rows, n)
        data[start:stop].copy_(pack_e2m1(values.reshape(-1, k)))
        scales[start:stop].copy_(byte)
    return W4A8Pack(data, scales, swizzle_scales(scales), logical_n=n)


@torch.no_grad()
def dequantize_w4a8(pack: W4A8Pack, *, chunk_rows: int = 256) -> torch.Tensor:
    """FP32 weight `[N,K]` exactly as the MMA sees it."""
    n, k = pack.shape
    out = torch.empty((n, k), dtype=torch.float32, device=pack.data.device)
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        codes = unpack_e2m1(pack.data[start:stop]).reshape(-1, k // 32, 32)
        scale = e8m0_to_float(pack.scales[start:stop]).unsqueeze(-1)
        out[start:stop].copy_((codes * scale).reshape(-1, k))
    return out


@torch.no_grad()
def quantize_activation_mx8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch oracle of the in-call quantizer: FP8 E4M3 `[M,K]` and UE8M0 bytes `[M,K/32]`."""
    if x.ndim != 2 or x.shape[1] % 32:
        raise ValueError("activation must be [M,K] with K divisible by 32")
    blocks = x.float().reshape(x.shape[0], -1, 32)
    byte = _e8m0(blocks.abs().amax(-1), _FP8)
    values = (blocks * _inverse(byte).unsqueeze(-1)).reshape(x.shape)
    return values.to(torch.float8_e4m3fn), byte


def dequantize_activation_mx8(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    m, k = values.shape
    return (values.float().reshape(m, -1, 32) * e8m0_to_float(scales).unsqueeze(-1)).reshape(m, k)


@torch.no_grad()
def fq_w4a8_reference(x: torch.Tensor, w) -> torch.Tensor:
    """FP32 `dq(x) @ dq(w).T`; `w` is a `W4A8Pack` or a bfloat16 weight to pack first."""
    pack = w if isinstance(w, W4A8Pack) else pack_w4a8(w)
    weight = dequantize_w4a8(pack)
    return dequantize_activation_mx8(*quantize_activation_mx8(x)).to(weight.device) @ weight.T


_COMPILE_COUNT = 0


def compile_count() -> int:
    """Cumulative CuTe compilations in this process, including workspace reductions."""
    return _COMPILE_COUNT


def state_key(m: int, config: W4A8Config, bias: bool) -> tuple:
    """K1 compiles once per token bucket, configuration and bias presence."""
    return (token_tile(m), config, bias) if config.backend == "k1" else (m, config, False)


class _State:
    """Compiled kernels and owned buffers for one pack, token bucket and config."""

    def __init__(self, pack: W4A8Pack, m: int, config: W4A8Config, bias: bool):
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_stream, make_ptr
        from ._w4a8_kernel import W4A8Kernel, reduce_workspace
        from ._w4a8_quant import launch_quant

        global _COMPILE_COUNT
        logical_n, k = pack.shape
        n = pack.data.shape[0]
        device = pack.data.device
        if k % config.cta_k:
            raise ValueError("K must be divisible by cta_k")
        self.cuda, self.cutlass, self.make_ptr = cuda, cutlass, make_ptr
        self.launch_quant = launch_quant
        self.m, self.n, self.k, self.config = m, n, k, config
        self.logical_n = logical_n
        self.stages = resolved_stages(config)
        self.workspace = config.split > 1 and config.resolved_reduction() == "workspace"
        self.q = torch.zeros((m, k), dtype=torch.float8_e4m3fn, device=device)
        # One 128-row scale atom per token tile (per 128 rows of a larger tile).
        atoms = -(-m // config.tile_n) * -(-config.tile_n // 128)
        self.sfb = torch.zeros((atoms * 128, -(-(k // 32) // 4) * 4), dtype=torch.uint8, device=device)
        self.alpha = torch.ones((2,), dtype=torch.float32, device=device)
        self.partials = (torch.empty((config.split, m, n), dtype=torch.float32, device=device)
                         if self.workspace else None)
        self.output = (torch.empty((m, n), dtype=torch.bfloat16, device=device)
                       if n != logical_n else None)
        self.padded_bias = (torch.zeros(n, dtype=torch.bfloat16, device=device)
                            if bias and n != logical_n else None)
        self.bias = bias
        kernel = W4A8Kernel(mma_tiler_mnk=(128, config.tile_n, config.cta_k), num_ab_stage=self.stages,
                          num_sfb_tmem_stage=1, sf_vec_size=32,
                          split_k=1 if self.workspace else config.split,
                          workspace_split=config.split if self.workspace else 1,
                          use_scale=False, use_bias=bias and not self.workspace, use_pdl=config.pdl,
                          l2_prefetch=config.prefetch, fused_quant=config.fused)

        def ptr(tensor, dtype, align=16):
            return make_ptr(dtype, tensor.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=align)
        self.ptr = ptr
        placeholder = torch.empty((m, n), dtype=torch.bfloat16, device=device)
        self.fixed = (ptr(pack.data, cutlass.Float4E2M1FN), ptr(pack.swizzled_scales, cutlass.Float8E8M0FNU),
                      ptr(self.q, cutlass.Float8E4M3FN), ptr(self.sfb, cutlass.Float8E8M0FNU))
        self.problem = tuple(cutlass.Int32(v) for v in (n, m, k, 1))
        self.prefetch = (cutlass.Int64(pack.data.data_ptr()), cutlass.Int64(pack.swizzled_scales.data_ptr()),
                         cutlass.Int32(k // 2), cutlass.Int32(128 * pack.swizzled_scales.shape[1]))
        output = ptr(self.partials, cutlass.Float32) if self.workspace else ptr(placeholder, cutlass.BFloat16)
        bias_ptr = ptr(placeholder, cutlass.BFloat16) if bias and not self.workspace else None
        self.compiled = cute.compile(kernel, *self.fixed, output, None, bias_ptr, None,
                                     self._activation(placeholder), self.prefetch,
                                     self.problem, make_fake_stream(), options="--generate-line-info")
        _COMPILE_COUNT += 1
        self.reduce = None
        if self.workspace:
            self.reduce = cute.compile(reduce_workspace, ptr(self.partials, cutlass.Float32),
                                       ptr(placeholder, cutlass.BFloat16), ptr(self.alpha, cutlass.Float32, 4),
                                       cutlass.Int32(m * n), config.split, 0, make_fake_stream())
            _COMPILE_COUNT += 1
        self.warmed = False

    def _activation(self, x):
        """The fused kernel's BF16 activation pointer (in the prologue slot), else None."""
        if not self.config.fused:
            return None
        return (self.ptr(x, self.cutlass.BFloat16),) * 5

    def quantize(self, x):
        self.launch_quant(x, self.q, self.sfb, trigger=self.config.pdl, tile=self.config.tile_n)

    def gemm(self, out, bias=None, x=None, *, x_fp8=None, x_scale=None):
        cutlass, ptr = self.cutlass, self.ptr
        m = out.shape[0]
        problem = (self.problem[0], cutlass.Int32(m), self.problem[2], self.problem[3])
        target = out
        if self.output is not None:
            target = self.output[:m]
            if bias is not None:
                self.padded_bias[:self.logical_n].copy_(bias)
                bias = self.padded_bias
        stream = self.cuda.CUstream(torch.cuda.current_stream(out.device).cuda_stream)
        activation = self._activation(x)
        fixed = self.fixed
        if x_fp8 is not None:
            fixed = (*self.fixed[:2], ptr(x_fp8, cutlass.Float8E4M3FN),
                     ptr(x_scale, cutlass.Float8E8M0FNU))
        if self.workspace:
            self.compiled(*fixed, ptr(self.partials, cutlass.Float32), None, None, None, activation,
                          self.prefetch, problem, stream)
            if bias is None:
                self.reduce(ptr(self.partials, cutlass.Float32), ptr(target, cutlass.BFloat16),
                            ptr(self.alpha, cutlass.Float32, 4), cutlass.Int32(m * self.n), stream)
            else:
                from ._w4a8_quant import reduce_with_bias
                partials = self.partials.view(-1)[:self.config.split * m * self.n].view(
                    self.config.split, m, self.n)
                reduce_with_bias(partials, bias, target)
        else:
            bias_ptr = ptr(bias, cutlass.BFloat16) if self.bias else None
            self.compiled(*fixed, ptr(target, cutlass.BFloat16), None, bias_ptr, None, activation,
                          self.prefetch, problem, stream)
        if self.output is not None and out.data_ptr() != target.data_ptr():
            out.copy_(target[:, :self.logical_n])
        return out


def _check_call(x, pack, bias, out):
    n, k = pack.shape
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.shape[1] != k:
        raise ValueError("x must be a contiguous bfloat16 [M,K] matching the pack")
    if not 1 <= x.shape[0] <= MAX_M:
        raise ValueError(f"W4A8 supports M from 1 to {MAX_M}")
    if not x.is_cuda or x.device != pack.data.device:
        raise ValueError("x and the pack must share one CUDA device")
    if bias is not None and (bias.shape != (n,) or bias.dtype != torch.bfloat16
                             or not bias.is_contiguous() or bias.device != x.device):
        raise ValueError("bias must be a contiguous bfloat16 [N] on the device")
    if out is not None and (out.shape != (x.shape[0], n) or out.dtype != torch.bfloat16
                            or not (out.is_contiguous() or out.stride() == (pack.data.shape[0], 1))
                            or out.device != x.device
                            or out.data_ptr() % 16):
        raise ValueError("out must be a contiguous 16-byte aligned bfloat16 [M,N]")


def _state(pack: W4A8Pack, m: int, config: W4A8Config, bias: bool):
    key = state_key(m, config, bias)
    state = pack.states.get(key)
    if state is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("call w4a8_warmup for this M before CUDA graph capture")
        if torch.cuda.get_device_capability(pack.data.device)[0] != 10:
            raise RuntimeError("W4A8 needs Blackwell sm100/sm103 block-scaled MMA")
        if config.backend == "triton":
            from ._w4a8_triton import TritonState
            state = TritonState(pack, m, config, pack.data.device)
        else:
            state = _State(pack, key[0], config, bias)
        pack.states[key] = state
    return state


def w4a8_linear(x: torch.Tensor, pack: W4A8Pack, bias: torch.Tensor | None = None,
                out: torch.Tensor | None = None, *, config: W4A8Config | None = None,
                x_fp8: torch.Tensor | None = None, x_scale: torch.Tensor | None = None) -> torch.Tensor:
    """`x @ W.T (+ bias)` for bfloat16 `x [M,K]`, M 1 to 256; returns bfloat16 `[M,N]`.

    The activation quantization runs inside the call. A call with `out` given
    after `w4a8_warmup` does not allocate or synchronize, so it can be captured.
    State belongs to the pack, so calls on one pack must not overlap across streams.

    Supplying both x_fp8 and x_scale skips activation quantization and borrows
    the producer's buffers without copying or rebinding the pack's state.
    activation_layout(pack, M, config=...) describes their exact dtype/shape.
    x still supplies BF16 shape/device validation; its values are not read on
    this path. Keep producer buffers alive and immutable until the GEMM ends,
    including graph replay. The producer and GEMM must be stream-ordered.
    """
    if (x_fp8 is None) != (x_scale is None):
        raise ValueError('supply both x_fp8 and x_scale, or neither')
    _check_call(x, pack, bias, out)
    m = x.shape[0]
    config = config or default_config(*pack.shape, m, _sms(x.device))
    if x_fp8 is not None:
        layout = activation_layout(pack, m, config=config)
        if (tuple(x_fp8.shape) != layout.data_shape or x_fp8.dtype != layout.data_dtype
                or tuple(x_scale.shape) != layout.scale_shape or x_scale.dtype != layout.scale_dtype
                or any(t.device != x.device or not t.is_contiguous() or t.data_ptr() % 16
                       for t in (x_fp8, x_scale))):
            raise ValueError('pre-quantized activation must match the K1 config, shape, device and scale atoms')
    if config.fused and x.data_ptr() % 16:  # the fused producer reads x in 16-byte vectors
        config = replace(config, fused=False)
    state = _state(pack, m, config, bias is not None)
    if out is None:
        out = (state.output[:m, :pack.shape[0]] if state.output is not None
               else torch.empty((m, pack.shape[0]), dtype=torch.bfloat16, device=x.device))
    if config.backend == "triton":
        return state(x, out, bias)
    if config.fused:
        return state.gemm(out, bias, x)
    if x_fp8 is not None:
        return state.gemm(out, bias, x_fp8=x_fp8, x_scale=x_scale)
    state.quantize(x)
    return state.gemm(out, bias)


def w4a8_gemm_only(pack: W4A8Pack, out: torch.Tensor, *, config: W4A8Config | None = None) -> torch.Tensor:
    """Diagnostics: rerun the GEMM on the activation that the last call of this M quantized."""
    m = out.shape[0]
    config = config or default_config(*pack.shape, m, _sms(out.device))
    return pack.states[state_key(m, config, False)].gemm(out)


def w4a8_warmup(pack: W4A8Pack, m: int | None = None, *, config: W4A8Config | None = None, x=None, out=None,
                bias=None, iterations: int = 3) -> W4A8Pack:
    """Warm one token bucket, or every default bucket when M is omitted, before capture."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("warm up before CUDA graph capture")
    if m is None:
        if config is not None or x is not None or out is not None:
            raise ValueError("all-bucket warmup takes only pack, bias and iterations")
        for bucket in TOKEN_TILES:
            w4a8_warmup(pack, bucket, bias=bias, iterations=iterations)
        return pack
    device = pack.data.device
    x = torch.randn((m, pack.shape[1]), dtype=torch.bfloat16, device=device) if x is None else x
    for _ in range(iterations):
        w4a8_linear(x, pack, bias, out, config=config)
    return pack


_SM_COUNTS: dict = {}


def _sms(device) -> int:
    index = torch.device(device).index or 0
    if index not in _SM_COUNTS:
        _SM_COUNTS[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return _SM_COUNTS[index]

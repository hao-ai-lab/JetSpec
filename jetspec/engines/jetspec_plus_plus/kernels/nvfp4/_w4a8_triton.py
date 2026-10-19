"""Single-launch Triton W4A8 kernel for `w4a8.py` (backend "triton"); imported on the first CUDA call.

Each program owns 128 weight rows (the MMA M side) and one K range. Per K tile
it loads the E2M1 codes with TMA, quantizes its BF16 activation tile to FP8
E4M3 with UE8M0 scales in registers (bit for bit `quantize_activation_mx8`),
and issues `tl.dot_scaled(e2m1, e4m3)`. Split-K partials are summed in a
fixed order by the last program of each weight tile, found with an arrival
counter, so one call is one launch.
"""
import triton
import triton.language as tl


@triton.jit
def _w4a8_dot_kernel(X, WD, S, Bias, Out, Workspace, Counters, M, N, K,
                     HAS_BIAS: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, NUM_STAGES: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    SB: tl.constexpr = BLOCK_K // 32
    k_tiles = K // BLOCK_K
    first = pid_k * k_tiles // SPLIT_K
    last = (pid_k + 1) * k_tiles // SPLIT_K
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_ok = rows < N
    tokens = tl.arange(0, BLOCK_M)
    token_ok = tokens < M
    s_ptr = S + rows.to(tl.int64)[:, None] * (K // 32) + tl.arange(0, SB)[None, :]
    x_ptr = X + tokens[:, None] * K + tl.arange(0, BLOCK_K)[None, :]
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for tile in tl.range(first, last, num_stages=NUM_STAGES):
        w = WD.load([pid_n * BLOCK_N, tile * (BLOCK_K // 2)])
        w_scale = tl.load(s_ptr + tile * SB)
        x = tl.load(x_ptr + tile * BLOCK_K, mask=token_ok[:, None], other=0.).to(tl.float32)
        blocks = tl.reshape(x, (BLOCK_M, SB, 32))
        bits = tl.max(tl.abs(blocks), axis=2).to(tl.int32, bitcast=True)
        byte = ((bits >> 23) & 0xFF) - 8 + ((bits & 0x7FFFFF) > 0x600000).to(tl.int32)
        byte = tl.minimum(tl.maximum(byte, 1), 253)
        inverse = ((254 - byte) << 23).to(tl.float32, bitcast=True)
        q = tl.reshape((blocks * inverse[:, :, None]).to(tl.float8e4nv), (BLOCK_M, BLOCK_K))
        acc = tl.dot_scaled(w, w_scale, "e2m1", tl.trans(q), byte.to(tl.uint8), "e4m3", acc)
    out_ptr = Out + tokens[None, :] * N + rows[:, None]
    mask = row_ok[:, None] & token_ok[None, :]
    if SPLIT_K == 1:
        result = acc
        if HAS_BIAS:
            result += tl.load(Bias + rows, mask=row_ok, other=0.).to(tl.float32)[:, None]
        tl.store(out_ptr, result.to(tl.bfloat16), mask=mask)
    else:
        part = tokens[None, :] * N + rows[:, None]
        tl.store(Workspace + pid_k * M * N + part, acc, mask=mask)
        tl.debug_barrier()
        arrived = tl.atomic_add(Counters + pid_n, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT_K - 1:
            total = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for split in tl.static_range(SPLIT_K):
                total += tl.load(Workspace + split * M * N + part, mask=mask, other=0., cache_modifier=".cg")
            if HAS_BIAS:
                total += tl.load(Bias + rows, mask=row_ok, other=0.).to(tl.float32)[:, None]
            tl.store(out_ptr, total.to(tl.bfloat16), mask=mask)
            tl.atomic_xchg(Counters + pid_n, 0, sem="relaxed", scope="gpu")


class TritonState:
    """Workspace, counters and the weight descriptor for one pack, M and config."""

    def __init__(self, pack, m, config, device):
        import torch
        from triton.tools.tensor_descriptor import TensorDescriptor
        logical_n, k = pack.shape
        n = pack.data.shape[0]
        self.m, self.n, self.k, self.config = m, n, k, config
        self.logical_n = logical_n
        self.output = (torch.empty((m, n), dtype=torch.bfloat16, device=device)
                       if n != logical_n else None)
        self.padded_bias = (torch.zeros(n, dtype=torch.bfloat16, device=device)
                            if n != logical_n else None)
        self.block_m = max(16, 1 << (m - 1).bit_length())
        self.workspace = torch.empty((config.split * m * n,) if config.split > 1 else (1,),
                                     dtype=torch.float32, device=device)
        self.counters = torch.zeros((n // 128,), dtype=torch.int32, device=device)
        self.dummy_bias = torch.zeros((1,), dtype=torch.bfloat16, device=device)
        self.w_desc = TensorDescriptor.from_tensor(pack.data, [128, config.cta_k // 2])
        self.scales = pack.scales
        self.kernel = None

    def __call__(self, x, out, bias=None):
        config = self.config
        target = self.output if self.output is not None else out
        if bias is not None and self.padded_bias is not None:
            self.padded_bias[:self.logical_n].copy_(bias)
            bias = self.padded_bias
        self.kernel = _w4a8_dot_kernel[(self.n // 128, config.split)](
            x, self.w_desc, self.scales, bias if bias is not None else self.dummy_bias, target,
            self.workspace, self.counters, self.m, self.n, self.k,
            HAS_BIAS=bias is not None, SPLIT_K=config.split, BLOCK_M=self.block_m, BLOCK_N=128,
            BLOCK_K=config.cta_k, NUM_STAGES=config.stages or 3, num_warps=config.warps)
        if self.output is not None and out.data_ptr() != target.data_ptr():
            out.copy_(target[:, :self.logical_n])
        return out

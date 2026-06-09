"""NanoEngine — single-stream AR decode over a paged KV cache (nano_vllm N0).

The owned-substrate analogue of `ptd.engine.llm.LLM.generate()`: plain
greedy/temperature decode (prefill the prompt once, then single-token steps
reusing the cache), with the HF `DynamicCache` swapped for `PagedKVCache`. The
paging is invisible to the model forward (`ModelRunner` is reused as-is), so the
loop is the same — this is the N0 lossless gate: token-identical to `LLM.generate`
on the same model/device/dtype (fp32 exact on CPU; bf16 on b200, modulo the
documented SDPA-reduction-order borderline flips).

N0 is vanilla AR (no tree). N1 layers tree spec on the same paged cache, using
`PagedKVCache.gather` for the accepted-path KV (the paged `_select_kv_cache`).
N2a adds `generate_batch`: continuous-batched AR over the *multi-sequence* paged
pool (one shared cache, per-`seq_id` block tables), token-identical to running
`generate` on each prompt alone (the N2a lossless gate). N2b adds
`generate_tree_batch`: batched per-sequence TREE-spec over the same pool (each seq
builds its own tree, one batched verify forward under a padded per-seq 4D ancestor
mask, per-seq accept + ref-count-safe gather), token-identical to running
`generate_tree` on each prompt alone (the N2b lossless gate).
"""
import torch
from transformers import DynamicCache

from ptd.models.qwen3 import load_target
from ptd.engine.llm import SamplingParams
from ptd.engine.model_runner import ModelRunner
from ptd.engine.sampler import sample
from ptd.nano_vllm.paged_kv_cache import PagedKVCache
from ptd.nano_vllm.scheduler import Scheduler, SequenceRequest

# A3-BUCKET: tree-N bucket sizes for the compiled verify stack. `torch.compile(
# dynamic=False)` specializes `_stack` on the concrete node count N, so a variable-N
# decode recompiles per distinct N. We snap N UP to the next bucket by padding (pad
# rows get -inf qq_bias so real rows never attend them and tree_accept never reads
# them — see `_pad_tree_to_bucket`), so the compiled stack only ever sees these few N
# values. Buckets were chosen from the measured N distribution on a real gsm8k decode
# (Qwen3-8B, budget=255, width=7, block_size=16, epoch6 head): the crossproduct heap
# fills to the budget cap EVERY round, so N is degenerate at 255 — but the smaller
# buckets keep early/short-context or smaller-budget runs from recompiling too, and
# 256 covers the budget=255 steady state (the dominant case). Padding adds at most
# `bucket - N` dummy rows (here 1), a negligible verify-forward cost.
_TREE_BUCKETS = (64, 128, 192, 256)

# A3-GRAPH backends. The compiled verify/AR stacks (A3-INT/A3-HIDDEN) ride two flag sets:
#   - `_KERNEL_BACKENDS`: routes prefill + the verify forward through the paged tree-attn
#     custom_op substrate (the eager kernel and both compiled layers all need this).
#   - `_COMPILED_BACKENDS`: ALSO swaps the per-round verify forward for the compiled
#     read-only `CompiledVerifyStack` (bypassing `model.__call__`).
#   - `_CUDAGRAPH_BACKENDS`: a strict superset of compiled — it ALSO captures one CUDA
#     graph per tree-N bucket around the compiled stack and replays it per round
#     (A3-GRAPH), collapsing the per-kernel launch storm that compile can't remove at
#     B=1. It is opt-in: "triton_paged_tree_compiled" stays the compiled-non-graph oracle
#     (byte-for-byte unchanged), so the cudagraph path can be diffed against it.
#   - `_LOGICAL_KV_BACKENDS` (L5 no-gather): supersets of compiled/cudagraph that
#     keep committed tree KV where the verify wrote it and pass per-layer logical
#     slot maps to the kernel instead of running the O(context) per-round
#     `cache.gather`. The gather-path backends stay byte-identical oracles.
_KERNEL_BACKENDS = ("triton_paged_tree", "triton_paged_tree_compiled",
                    "triton_paged_tree_cudagraph",
                    "triton_paged_tree_compiled_nogather",
                    "triton_paged_tree_cudagraph_nogather")
_COMPILED_BACKENDS = ("triton_paged_tree_compiled", "triton_paged_tree_cudagraph",
                      "triton_paged_tree_compiled_nogather",
                      "triton_paged_tree_cudagraph_nogather")
_CUDAGRAPH_BACKENDS = ("triton_paged_tree_cudagraph",
                       "triton_paged_tree_cudagraph_nogather")
_LOGICAL_KV_BACKENDS = ("triton_paged_tree_compiled_nogather",
                        "triton_paged_tree_cudagraph_nogather")


def _bucket_for_n(n: int) -> int:
    """Smallest bucket >= n (A3-BUCKET). For n beyond the largest bucket, rounds up to
    the next multiple of that largest bucket (keeps the stack from per-N recompiling on
    an unexpectedly large tree; still a bounded, coarse set of shapes)."""
    for b in _TREE_BUCKETS:
        if n <= b:
            return b
    # n > max bucket: round up to the next multiple of the largest bucket so the
    # shape set stays small and predictable rather than per-N.
    step = _TREE_BUCKETS[-1]
    return ((n + step - 1) // step) * step


class NanoEngine:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        block_size: int = 16,
        attn_implementation: str = "sdpa",
        attn_backend: str = "sdpa",
    ):
        self.model, self.tokenizer = load_target(
            model_name_or_path, device, dtype, attn_implementation
        )
        self.runner = ModelRunner(self.model)
        self.device = device
        self.dtype = dtype
        self.block_size = block_size
        self.eos_token_ids = self._resolve_eos()
        # N3 attention backend (opt-in). "sdpa" (default) is byte-identical to the
        # pre-N3 engine. "triton_paged_tree" swaps the RUNTIME attention interface
        # for the paged tree-attention kernel (the model still loads with sdpa
        # weights/format; only the interface HF dispatches to is replaced). Affects
        # N0/N1/N2a; N2b stays on SDPA regardless (see generate_tree_batch).
        self.attn_backend = attn_backend
        if attn_backend == "triton_paged_tree":
            from ptd.nano_vllm.paged_attn_backend import register_ptd_paged_tree

            register_ptd_paged_tree()
            self.model.config._attn_implementation = "ptd_paged_tree"
        elif attn_backend in _COMPILED_BACKENDS:
            # A3-INT/A3-HIDDEN: same custom_op attention backend as
            # "triton_paged_tree" (prefill + the eager kernel fallback both ride it),
            # PLUS compiled read-only stacks that replace the per-round verify forward
            # and the AR decode forward.
            #   - `compiled_verify`: logits-only verify (need_hidden=False).
            #   - `compiled_verify_hidden`: the DraftHead path (need_hidden=True),
            #     built lazily on first use because `target_layer_ids` only arrive
            #     with `generate_tree`'s args (keyed by tap set so a different head
            #     gets its own compiled graph).
            #   - `compiled_ar`: a compiled N=1 single-node verify stack so the AR
            #     decode `generate()` compares compiled-vs-compiled with the tree path.
            # A3-GRAPH ("triton_paged_tree_cudagraph"): additionally wraps the tree
            # verify stacks in per-bucket CUDA graphs (built lazily in generate_tree
            # once the pool is reserved + the bucket set known). `_use_cudagraph` gates
            # that extra layer; everything else is identical to the compiled backend.
            from ptd.nano_vllm.paged_attn_backend import register_ptd_paged_tree
            from ptd.nano_vllm.compiled_verify_stack import CompiledVerifyStack

            register_ptd_paged_tree()
            self.model.config._attn_implementation = "ptd_paged_tree"
            self.compiled_verify = CompiledVerifyStack(self.model, block_size=self.block_size)
            self._compiled_verify_hidden = {}        # target_layer_ids -> stack
            self.compiled_ar = CompiledVerifyStack(self.model, block_size=self.block_size)
            self._use_cudagraph = attn_backend in _CUDAGRAPH_BACKENDS
            self._graphed_verify = {}                # (need_hidden, tap_set) -> GraphedVerify

    def _resolve_eos(self) -> set:
        """EOS ids from tokenizer + generation_config (mirrors `LLM._resolve_eos`)."""
        ids = set()
        gc = getattr(self.model, "generation_config", None)
        for src in (
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(gc, "eos_token_id", None) if gc is not None else None,
        ):
            if src is None:
                continue
            if isinstance(src, (list, tuple, set)):
                ids.update(int(x) for x in src)
            else:
                ids.add(int(src))
        return ids

    @torch.inference_mode()
    def generate(self, prompt, sampling_params: SamplingParams = None) -> dict:
        """Greedy/temperature decode over the paged cache. `prompt` is a str
        (tokenized raw) or an already-tokenized `input_ids` tensor (1, T). Returns
        {token_ids, text}. Token-identical to `LLM.generate` (the N0 gate)."""
        sp = sampling_params or SamplingParams()
        if isinstance(prompt, str):
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        else:
            input_ids = prompt.to(self.device)
        prompt_len = input_ids.shape[1]
        cache = PagedKVCache(
            block_size=self.block_size, device=torch.device(self.device), dtype=self.dtype
        )
        # N3 kernel path: route the single seq (id 0) through the paged tree-attn
        # kernel. Prefill runs on the kernel too (qq_bias=None -> pure causal, since
        # context_len = seq_len - S = 0). attention_mask=None: the kernel masks.
        # getattr default keeps object.__new__-built engines (test fixtures) on SDPA.
        # A3-HIDDEN: the compiled backend also rides the kernel substrate for prefill,
        # then runs the decode forward (N=1) through the compiled AR stack so
        # `decode_cuda_speedup` compares compiled-vs-compiled with the tree path.
        backend = getattr(self, "attn_backend", "sdpa")
        kernel = backend in _KERNEL_BACKENDS
        compiled = backend in _COMPILED_BACKENDS
        if kernel:
            cache._paged_handoff = True
            cache._handoff_seq_ids = [0]
            cache._ptd_attn_meta = {"seq_ids": [0], "qq_bias": None}

        # --- prefill: process the whole prompt once, sample the first token ---
        # (attention_mask stays None for both paths: SDPA builds its own causal
        # mask from cache_position; the kernel masks internally.)
        pos = torch.arange(prompt_len, device=self.device).unsqueeze(0)
        logits, cache, _ = self.runner.forward(input_ids, cache, pos)
        ar_graph = None
        if compiled:
            # A3-BUCKET: pre-grow the pool to the whole-run length ONCE so the compiled
            # AR (N=1) stack's pool-shape guard never trips mid-decode (only seq_lens_k
            # value grows, not the pool block-count shape). +1 for the lone decode slot.
            cache.reserve_capacity(prompt_len + sp.max_new_tokens + 1)
            # Step 1 (path-to-fork-tps): CUDA-graph the N=1 AR forward. The compiled AR
            # stack is fused but still launches ~36 layers' kernels eagerly per token;
            # capturing a B=1 graph (GraphedVerify, bucket {1}) collapses that launch
            # storm into one cudaGraphLaunch — the same win the tree-verify path banks.
            # Built here (after reserve_capacity, so the k/v pools are address-stable) and
            # only for the cudagraph backend; the compiled-non-graph path is left untouched
            # as the losslessness oracle.
            if getattr(self, "_use_cudagraph", False):
                from ptd.nano_vllm.graph_capture import GraphedVerify
                cfg = self.model.config
                head_dim = (getattr(cfg, "head_dim", None)
                            or cfg.hidden_size // cfg.num_attention_heads)
                nl = cfg.num_hidden_layers
                ar_graph = GraphedVerify(
                    self.compiled_ar,
                    [cache.pool(i)[0] for i in range(nl)],
                    [cache.pool(i)[1] for i in range(nl)],
                    block_table_width=cache.reserved_block_table_width,
                    head_dim=head_dim, hidden_size=cfg.hidden_size,
                    device=torch.device(self.device), dtype=self.dtype,
                    buckets=(1,),
                )
        next_tok = sample(logits[:, -1:, :], sp.temperature)  # (1, 1)
        out_ids = [int(next_tok.item())]
        cur = prompt_len

        # --- decode: single-token steps reusing the paged cache (no reprocess) ---
        for _ in range(sp.max_new_tokens - 1):
            if out_ids[-1] in self.eos_token_ids:
                break
            pos = torch.tensor([[cur]], device=self.device)
            if compiled:
                # A3-HIDDEN: compiled N=1 decode — reuse the verify stack with a
                # single node (qq_bias=None -> pure causal: the kernel makes the
                # cached prefix [0, cur) always-visible and the lone node attends
                # itself). Reserve the one slot (the stack scatters its post-RoPE
                # K/V in-graph), then run the compiled AR stack. Byte-equivalent to
                # the eager kernel decode forward; compiled so the AR baseline is
                # fused like the tree verify (a fair decode_cuda_speedup).
                # Pin the block_table width (fixed by reserve_capacity) so the compiled
                # AR stack's block_table shape guard stays stable as the seq lengthens.
                bts, node_blks, node_offs, slk = cache.reserve_tree_slots(
                    0, 1, cur, block_table_width=cache.reserved_block_table_width)
                dummy = torch.zeros(1, 1, self.model.config.hidden_size,
                                    device=self.device, dtype=self.dtype)
                cos, sin = self.model.model.rotary_emb(dummy, pos)
                cu = torch.tensor([0, 1], device=self.device, dtype=torch.int32)
                nlayers = self.model.config.num_hidden_layers
                k_pools = [cache.pool(i)[0] for i in range(nlayers)]
                v_pools = [cache.pool(i)[1] for i in range(nlayers)]
                if ar_graph is not None:
                    # Step 1: replay the captured B=1 graph. qq_bias is a (1,1) zero —
                    # the lone node attends itself with +0 bias, identical to the
                    # qq_bias=None causal path for N=1 (no inter-node masking, prefix
                    # always-visible). seq_lens_k (slk) grows each round and is staged,
                    # so the graph reads the lengthening prefix; node_blks/offs stage the
                    # new slot so the in-graph scatter lands this token's K/V correctly.
                    qq0 = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
                    logits = ar_graph.replay(
                        1, next_tok, cos, sin, bts, cu, slk, qq0,
                        node_blks, node_offs, 1,
                    )
                else:
                    logits = self.compiled_ar(
                        next_tok, cos, sin, k_pools, v_pools, bts, cu, slk,
                        None, node_blks, node_offs,
                    )
            else:
                logits, cache, _ = self.runner.forward(next_tok, cache, pos)
            next_tok = sample(logits[:, -1:, :], sp.temperature)
            out_ids.append(int(next_tok.item()))
            cur += 1

        text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        return {"token_ids": out_ids, "text": text}

    def _pad_tree_to_bucket(self, seq_step, posN, qq_bias, N: int, B: int):
        """Pad the N real tree rows up to bucket size B (A3-BUCKET). Returns
        `(seq_step_b (1,B), posN_b (1,B), qq_bias_b (B,B))`. No-op (returns the
        inputs) when B == N.

        Padding is lossless for the committed tokens AND the tapped hidden because:
          - `qq_bias_b[i, N:] = -inf` for every real row i < N: real queries assign
            -inf score to pad keys, so the softmax over `[prefix | tree]` is identical
            to the unbucketed (N,N) bias — real rows' attention output is unchanged.
          - `qq_bias_b[N:, :] = -inf` for every pad row: pad queries attend nothing
            (all keys -inf). Their attention output is the kernel's all-masked value
            (irrelevant — we slice `[:N]` off), and crucially they are never read by
            `tree_accept`, which walks child indices 0..N-1 only (so a pad row is never
            `current` and its logit/hidden is never inspected).
          - pad rows get a dummy token (0) and the last real RoPE position; neither
            feeds back into any real row (the -inf bias isolates them).
        The pad rows' post-RoPE K/V is scattered into the reserved slots
        `[past_len+N, past_len+B)`, which `gather(keep)` never selects and then frees."""
        if B == N:
            return seq_step, posN, qq_bias
        pad = B - N
        dev = self.device
        seq_step_b = torch.cat(
            [seq_step, torch.zeros((1, pad), dtype=seq_step.dtype, device=dev)], dim=1)
        # Pad RoPE positions with the last real position (value is masked out anyway).
        posN_b = torch.cat([posN, posN[:, -1:].expand(1, pad)], dim=1)
        neg_inf = torch.full((), float("-inf"), dtype=qq_bias.dtype, device=dev)
        qq_bias_b = neg_inf.expand(B, B).clone()           # every pad interaction -inf
        qq_bias_b[:N, :N] = qq_bias                         # real block unchanged
        return seq_step_b, posN_b, qq_bias_b

    def _get_compiled_verify_hidden(self, target_layer_ids):
        """Lazily build (and cache) the `need_hidden=True` compiled verify stack for
        a given tap set. A3-HIDDEN: `target_layer_ids` only arrive with
        `generate_tree`'s args, so we can't bake the tap set at engine construction;
        we cache one compiled stack per distinct tap tuple (different heads tap
        different layers and need their own DCE'd graph). Robust to the test fixtures
        that bypass `__init__` (no `_compiled_verify_hidden` dict)."""
        from ptd.nano_vllm.compiled_verify_stack import CompiledVerifyStack

        cache = getattr(self, "_compiled_verify_hidden", None)
        if cache is None:
            cache = {}
            self._compiled_verify_hidden = cache
        key = tuple(target_layer_ids)
        stack = cache.get(key)
        if stack is None:
            stack = CompiledVerifyStack(
                self.model, block_size=self.block_size,
                need_hidden=True, target_layer_ids=target_layer_ids,
            )
            cache[key] = stack
        return stack

    def _get_graphed_verify(self, stack, paged_cache, block_table_width,
                            need_hidden, target_layer_ids, logical_kv_bind=None):
        """Lazily build (and cache) the per-bucket `GraphedVerify` wrapping `stack`
        (A3-GRAPH). Built on first use because it needs the LIVE post-`reserve_capacity`
        k/v pools + the pinned block-table width, which only exist once `generate_tree`
        has prefilled and reserved. The per-bucket graph itself is captured lazily inside
        `replay` the first time each bucket is seen (so the warmup scatter uses real
        reserved slots) — captured ONCE per bucket, never recaptured within a decode.

        Keyed by (need_hidden, tap set); REBUILT when the live pool or block-table width
        changes. A captured graph hard-codes the pool's device addresses and the staged
        block-table column count, both fixed by THIS prompt's `reserve_capacity`. A new
        prompt builds a fresh `PagedKVCache` (new pool addresses) and may reserve a
        different width, so the old prompt's graphs are unusable. Rather than accumulate
        one `GraphedVerify` per prompt (each pinning that prompt's freed KV pools +
        captured graphs alive — an unbounded leak), we keep a SINGLE entry per
        (need_hidden, tap) and replace it (dropping the stale one, freeing its pool refs
        and graphs) whenever the pool tensor or width differs. Within one decode the pool
        + width are constant, so the entry is built once and every round replays it — the
        gate (no per-ROUND recapture) holds; a new prompt rebuilds once (per decode)."""
        from ptd.nano_vllm.graph_capture import GraphedVerify

        cache = getattr(self, "_graphed_verify", None)
        if cache is None:
            cache = {}
            self._graphed_verify = cache
        key = (bool(need_hidden), tuple(target_layer_ids) if target_layer_ids else ())
        # The live pool tensor identity + width tag this prompt's reservation; if they
        # changed (new prompt / new reserve_capacity), the cached graphs read stale
        # addresses, so rebuild (and drop the old GraphedVerify so its pool refs free).
        pool0 = paged_cache.pool(0)[0]
        # L5: the logical slot buffers are graph-read addresses too — a new decode's
        # fresh buffers make old graphs semantically incompatible (not just
        # address-stale), so they join the rebuild tag.
        lk_tag = id(logical_kv_bind[0][0]) if logical_kv_bind is not None else None
        tag = (id(pool0), int(block_table_width), lk_tag)
        gv = cache.get(key)
        if gv is None or gv.pool_tag != tag:
            nlayers = self.model.config.num_hidden_layers
            k_pools = [paged_cache.pool(i)[0] for i in range(nlayers)]
            v_pools = [paged_cache.pool(i)[1] for i in range(nlayers)]
            gv = GraphedVerify(
                stack, k_pools, v_pools, block_table_width=block_table_width,
                head_dim=self.compiled_verify.head_dim,
                hidden_size=self.model.config.hidden_size,
                device=torch.device(self.device), dtype=self.dtype,
                buckets=_TREE_BUCKETS,
                logical_kv_bind=logical_kv_bind,
            )
            gv.pool_tag = tag
            cache[key] = gv
        return gv

    @torch.inference_mode()
    def generate_tree(self, prompt, tree_drafter, block_size: int = 4, tree_width: int = 2,
                      budget: int = 15, algo: str = "crossproduct", algo_kwargs: dict = None,
                      target_layer_ids=None, sampling_params: SamplingParams = None,
                      return_stats: bool = False, prompt_info: dict = None,
                      profile_table: dict = None) -> dict:
        """Tree speculative decode over a PERSISTENT paged KV cache (nano_vllm N1).

        The owned-substrate analogue of `ptd.engine.llm.LLM._generate_tree_kv_cached`:
        each round the tree drafter emits per-depth logits, the tree algorithm builds
        a DraftTree, the target verifies all nodes in ONE forward under a 4D ancestor
        mask (appending only the tree nodes against the cached prefix — no prefix
        recompute), `tree_accept` takes the longest greedy-agreeing root-to-leaf path
        plus a correction, and the accepted path's KV is GATHERED back into a linear
        prefix (dropping the rejected branches). The only difference from the
        `DynamicCache` reference is the cache class: `PagedKVCache.gather` replaces
        `_select_kv_cache`. Lossless by construction (commits the verify forward's own
        greedy along the accepted path) — token-identical to `LLM._generate_tree_kv_cached`
        and to plain greedy `generate()` in fp32 (the N1 gate); bf16 may flip a
        borderline argmax after ~tens of exact tokens (cached prefix KV vs a fresh
        recompute differ in SDPA reduction order). Returns {token_ids, text, tpf}.

        `algo` / `algo_kwargs` / `prompt_info` / `profile_table` / `target_layer_ids`
        mirror `LLM.generate_tree`; all bundled algorithms recover crossproduct at
        their identity knobs, so the choice is lossless regardless."""
        # tree contract (engine -> tree, one-way): import only the public ptd.tree API
        from ptd.tree import get_algorithm, build_ancestor_matrix, tree_accept

        sp = sampling_params or SamplingParams()
        if isinstance(prompt, str):
            committed = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        else:
            committed = prompt.to(self.device)
        D = max(1, block_size - 1)
        algo_obj = get_algorithm(algo, **(algo_kwargs or {}))
        dtype = self.dtype
        neg = torch.finfo(dtype).min
        need_hidden = target_layer_ids is not None and block_size > 1
        new_ids, rounds = [], 0
        accept_lengths, tree_sizes = [], []   # per-round (acc+1) and node count (return_stats)
        target_hidden = None

        backend = getattr(self, "attn_backend", "sdpa")
        # A3-INT compiled verify rides the kernel substrate (prefill populates the
        # pool with post-RoPE prefix KV via the kernel path; the eager kernel is the
        # need_hidden fallback). `kernel` therefore covers both; `compiled` gates the
        # extra step of swapping the verify forward for the compiled read-only stack.
        kernel = backend in _KERNEL_BACKENDS
        compiled = backend in _COMPILED_BACKENDS
        logical_kv = backend in _LOGICAL_KV_BACKENDS   # L5 no-gather (compiled-only)

        # --- prefill: populate the persistent paged cache with the prompt's KV ---
        cache = PagedKVCache(
            block_size=self.block_size, device=torch.device(self.device), dtype=self.dtype
        )
        if kernel:
            # Prefill runs on the kernel too (single seq 0, qq_bias=None -> pure
            # causal over the prompt: context_len = seq_len - S = 0).
            cache._paged_handoff = True
            cache._handoff_seq_ids = [0]
            cache._ptd_attn_meta = {"seq_ids": [0], "qq_bias": None}
        pos = torch.arange(committed.shape[1], device=self.device).unsqueeze(0)
        logits, cache, full_hidden = self.runner.forward(
            committed, cache, pos,
            output_hidden_states=need_hidden, target_layer_ids=target_layer_ids,
        )
        if full_hidden is not None:
            target_hidden = full_hidden          # prompt context; anchor (first_tok) fed via noise
        if compiled:
            if getattr(self, "_use_cudagraph", False) and _bucket_for_n(budget) > _TREE_BUCKETS[-1]:
                # A3-GRAPH's GraphedVerify staging buffers are sized to the largest
                # STATIC bucket (`_TREE_BUCKETS[-1]`); a budget whose bucket exceeds it
                # can't be replayed (replay would copy an oversized tree into the fixed
                # staging). Fail loud and point at the compiled NON-graph backend, which
                # pads to `_bucket_for_n(budget)` and handles arbitrary budgets.
                raise ValueError(
                    f"attn_backend='triton_paged_tree_cudagraph' supports tree budget "
                    f"<= {_TREE_BUCKETS[-1]} (the largest CUDA-graph staging bucket); got "
                    f"budget={budget}. Use attn_backend='triton_paged_tree_compiled' for "
                    f"larger budgets."
                )
            # A3-BUCKET: pre-grow the pool to the whole-run high-water mark ONCE so the
            # compiled stack's pool-shape guard never trips. Peak occupancy in a round
            # is prefix + the B reserved tree nodes (before gather compacts back), so
            # reserve prompt + max_new_tokens + the largest bucket a tree of this `budget`
            # can hit (`_bucket_for_n(budget)`, not the static top bucket -- otherwise a
            # caller passing budget>255 undershoots the reservation and `reserve_tree_slots`
            # raises mid-decode). After this only the `seq_lens_k` value changes per round;
            # every pool/block-table shape stays fixed -> recompiles stop after the bucket
            # set is traced.
            Bmax = _bucket_for_n(budget)
            if logical_kv:
                # L5 no-gather: without gather's compaction, every committed token can
                # in the worst case pin its own retained block, so the POOL must be
                # sized for prompt + 16*(max_new + tree_depth) + Bmax tokens — while
                # the kernel-visible block-table WIDTH keeps today's (much smaller)
                # prompt + max_new + Bmax bound (it drives the dynamo guard + graph
                # staging and must not inflate with the pool). freeze_pool() turns any
                # later silent `_grow_pool` (a torch.cat = pool relocation under live
                # CUDA graphs) into a loud error.
                prompt_len = committed.shape[1]
                # Pool = the prefix (per-layer ids, via total_tokens) + the no-compaction
                # worst case as LAYER-SHARED ids (every committed token pinning its own
                # retained block, + one in-flight round) — shared because
                # reserve_logical_slots hands every layer the same block ids; sizing
                # this through total_tokens would multiply it ~num_layers-fold (174GB
                # at max_new=2048, the G2 OOM). Width keeps the kernel-visible bound.
                cache.reserve_capacity(
                    prompt_len,
                    block_table_tokens=prompt_len + sp.max_new_tokens + Bmax,
                    extra_shared_blocks=(sp.max_new_tokens + block_size
                                         + (Bmax + self.block_size - 1) // self.block_size + 1),
                )
                cache.freeze_pool()
                nlayers_lk = self.model.config.num_hidden_layers
                max_slots = sp.max_new_tokens + block_size + Bmax
                # Address-stable for the decode: graphs/compiled calls read these in
                # place (the pools' "REUSED IN PLACE" contract); the engine mutates
                # them between replays. Window = committed-after-prompt + this round's
                # B in-flight nodes; starts is write-once at prompt_len. ONE row shared
                # by all layers (layer-shared slot ids).
                slots_buf = torch.zeros((1, max_slots), dtype=torch.long,
                                        device=self.device)
                starts_buf = torch.tensor([prompt_len], dtype=torch.int32,
                                          device=self.device)
                lens_buf = torch.zeros((1,), dtype=torch.int32, device=self.device)
                slots_rows = [slots_buf] * nlayers_lk
                wlen = 0
                bts0 = cache.prefix_block_tables(cache.reserved_block_table_width)
                node_offs0 = {}   # bucket B -> (B,) arange % cache-block-size
            else:
                cache.reserve_capacity(committed.shape[1] + sp.max_new_tokens + Bmax)
        first_tok = sample(logits[:, -1:, :], sp.temperature)
        new_ids.append(int(first_tok.item()))
        committed = torch.cat([committed, first_tok.view(1, 1)], dim=1)   # anchor; NOT yet cached
        if int(first_tok.item()) in self.eos_token_ids:
            new_ids = new_ids[: sp.max_new_tokens]
            return {"token_ids": new_ids, "text": self.tokenizer.decode(new_ids, skip_special_tokens=True), "tpf": 0.0}

        # Invariant each round: cache.get_seq_length() == committed.shape[1] - 1 ==
        # target_hidden.shape[1] (when need_hidden). The cache trails `committed` by
        # the anchor (= tree root), which enters the cache via the verify forward.
        while len(new_ids) < sp.max_new_tokens:
            draft_logits = tree_drafter.propose_logits(committed, D, target_hidden=target_hidden).to(self.device)  # (1, D, V)
            tree = algo_obj.build(int(committed[0, -1]), draft_logits, block_size, tree_width, budget, self.device,
                                  prompt_info=prompt_info, profile_table=profile_table)
            N = tree.num_nodes
            # logical path: the cache's seq bookkeeping stays frozen at prompt_len
            # (no append/gather advances it); the engine-tracked window length is
            # the source of truth. Same VALUE as the gather path's get_seq_length()
            # (== committed.shape[1] - 1), so posN/RoPE are identical either way.
            past_len = (prompt_len + wlen) if logical_kv \
                else cache.get_seq_length()                            # == committed.shape[1] - 1
            # feed only the tree nodes (node 0 = anchor/root); the prefix KV is cached.
            seq_step = tree.token_ids.view(1, -1)                      # (1, N)
            depths = tree.depth.tolist()
            posN = torch.tensor([[past_len + d for d in depths]], device=self.device)   # RoPE: depth-based
            cache_pos = torch.arange(past_len, past_len + N, device=self.device)        # contiguous append slots
            if kernel:
                # Kernel path: prefix [0, past_len) is always-visible (handled by the
                # kernel); the N tree nodes attend per the ancestor mask folded in as
                # the fp32 (0/-inf) qq_bias. No dense 4D mask — attention_mask=None.
                anc = build_ancestor_matrix(tree).to(device=self.device, dtype=torch.bool)
                qq_bias = torch.where(
                    anc, torch.zeros((), dtype=torch.float32, device=self.device),
                    torch.full((), float("-inf"), dtype=torch.float32, device=self.device),
                )
                if compiled:
                    # A3-INT/A3-HIDDEN compiled verify: bypass model.__call__ + the
                    # per-layer Python and run the fused read-only stack. Reserve this
                    # round's node slots in the pool's block table (the allocation half
                    # of append, no scatter — the compiled stack scatters the post-RoPE
                    # node K/V in-graph), then call the compiled stack directly. After
                    # it returns, the node KV lives in the pool exactly where the eager
                    # kernel's `update` would have put it, so `cache.gather(keep)`
                    # below works byte-for-byte unchanged. When `need_hidden` (the real
                    # DraftHead path) the stack ALSO returns the tapped target_hidden,
                    # byte-matching extract_context_feature over `target_layer_ids`.
                    #
                    # A3-BUCKET: pad the N real tree rows up to a fixed bucket B so the
                    # compiled stack only ever sees the few `_TREE_BUCKETS` node counts
                    # (no per-N recompile). The pad rows get -inf qq_bias both ways, so
                    # real rows never attend pad keys and pad rows are never `current`
                    # in tree_accept (which walks real child indices 0..N-1 only) — the
                    # logits/hidden we slice back to [:N] are bit-identical to the
                    # unbucketed stack. We reserve B slots (the compiled stack scatters
                    # all B rows' KV in-graph); `gather(keep)` below references only
                    # `past_len + accepted_path` (< past_len + N), and decrefs ALL old
                    # blocks (incl. the transient pad slots), so the pad KV never
                    # survives the round.
                    B = _bucket_for_n(N)
                    seq_step_b, posN_b, qq_bias_b = self._pad_tree_to_bucket(
                        seq_step, posN, qq_bias, N, B)
                    # Pin the block_table width too (the second compiled-stack shape guard
                    # besides the pool) — reserve_capacity fixed the per-layer reservation.
                    if logical_kv:
                        # L5: block-aligned transient reservation — no block-table
                        # extension, no incref; nodes live at arbitrary pool slots the
                        # kernel reaches through the logical window. Stage this round's
                        # slots at window positions [wlen, wlen+B); the window length
                        # covers committed + in-flight nodes (the fork's
                        # persisted_len + qlen pattern). Block tables stay the frozen
                        # prefill ones (values ignored for window positions).
                        nb_lk, round_blocks = cache.reserve_logical_slots(B)
                        offs = node_offs0.get(B)
                        if offs is None:
                            offs = torch.arange(B, device=self.device) % self.block_size
                            node_offs0[B] = offs
                        slots_buf[:, wlen:wlen + B] = nb_lk * self.block_size + offs
                        lens_buf.fill_(wlen + B)
                        node_blks = [nb_lk] * len(bts0)   # layer-shared ids, one row
                        node_offs = [offs] * len(bts0)
                        bts = bts0
                        slk = torch.tensor([past_len + B], device=self.device,
                                           dtype=torch.int32)
                    else:
                        bts, node_blks, node_offs, slk = cache.reserve_tree_slots(
                            0, B, past_len, block_table_width=cache.reserved_block_table_width)
                    dummy = torch.zeros(1, B, self.model.config.hidden_size,
                                        device=self.device, dtype=self.dtype)
                    cos, sin = self.model.model.rotary_emb(dummy, posN_b)
                    cu = torch.tensor([0, B], device=self.device, dtype=torch.int32)
                    nlayers = self.model.config.num_hidden_layers
                    k_pools = [cache.pool(i)[0] for i in range(nlayers)]
                    v_pools = [cache.pool(i)[1] for i in range(nlayers)]
                    use_graph = getattr(self, "_use_cudagraph", False)
                    # L5: the logical metadata rides every verify call on the no-gather
                    # path. The graphed path binds the buffers IN PLACE at construction
                    # (like the pools — the engine mutates them before replay), so
                    # replay's signature is unchanged; the direct path passes them as
                    # kwargs each call.
                    lk_kwargs = dict(
                        logical_kv_slots=slots_rows, logical_kv_starts=starts_buf,
                        logical_kv_lens=lens_buf) if logical_kv else {}
                    lk_bind = (slots_rows, starts_buf, lens_buf) if logical_kv else None
                    if need_hidden:
                        stack = self._get_compiled_verify_hidden(target_layer_ids)
                        if use_graph:
                            # A3-GRAPH: replay the bucket-B captured graph. Copies this
                            # round's inputs into the persistent buffers + replays the
                            # whole captured forward (incl. the in-graph node-KV scatter)
                            # over the live pool; logits/new_hidden are sliced to [:N]
                            # inside replay, token-identical to the direct stack call.
                            gv = self._get_graphed_verify(
                                stack, cache, cache.reserved_block_table_width,
                                need_hidden=True, target_layer_ids=target_layer_ids,
                                logical_kv_bind=lk_bind)
                            logits, new_hidden = gv.replay(
                                B, seq_step_b, cos, sin, bts, cu, slk,
                                qq_bias_b, node_blks, node_offs, N)
                        else:
                            logits, new_hidden = stack(
                                seq_step_b, cos, sin, k_pools, v_pools, bts, cu, slk,
                                qq_bias_b, node_blks, node_offs, **lk_kwargs,
                            )
                            logits = logits[:, :N, :]            # drop the B-N pad rows
                            new_hidden = new_hidden[:, :N, :]
                    else:
                        if use_graph:
                            gv = self._get_graphed_verify(
                                self.compiled_verify, cache,
                                cache.reserved_block_table_width,
                                need_hidden=False, target_layer_ids=None,
                                logical_kv_bind=lk_bind)
                            logits = gv.replay(
                                B, seq_step_b, cos, sin, bts, cu, slk,
                                qq_bias_b, node_blks, node_offs, N)
                        else:
                            logits = self.compiled_verify(
                                seq_step_b, cos, sin, k_pools, v_pools, bts, cu, slk,
                                qq_bias_b, node_blks, node_offs, **lk_kwargs,
                            )
                            logits = logits[:, :N, :]            # drop the B-N pad rows
                        new_hidden = None
                else:
                    # Eager kernel path (the non-compiled "triton_paged_tree" backend;
                    # also remains the oracle the compiled need_hidden path is gated
                    # against in tests).
                    cache._handoff_seq_ids = [0]
                    cache._ptd_attn_meta = {"seq_ids": [0], "qq_bias": qq_bias}
                    logits, cache, new_hidden = self.runner.forward(
                        seq_step, cache, posN, attention_mask=None, cache_position=cache_pos,
                        output_hidden_states=need_hidden, target_layer_ids=target_layer_ids,
                    )
            else:
                # 4D additive mask: queries = N nodes; keys = past_len cached prefix
                # (all visible) + N nodes (ancestor mask, incl self).
                allowed = torch.zeros(N, past_len + N, dtype=torch.bool, device=self.device)
                allowed[:, :past_len] = True
                allowed[:, past_len:] = build_ancestor_matrix(tree).bool()
                mask = torch.where(allowed, torch.zeros((), dtype=dtype, device=self.device),
                                   torch.full((), neg, dtype=dtype, device=self.device)).view(1, 1, N, past_len + N)
                logits, cache, new_hidden = self.runner.forward(
                    seq_step, cache, posN, attention_mask=mask, cache_position=cache_pos,
                    output_hidden_states=need_hidden, target_layer_ids=target_layer_ids,
                )
            target_logits = logits                                    # (1, N, V) — every row is a tree node
            if sp.temperature == 0.0:
                # L2 (path-to-fork-tps): GPU-resident greedy accept — one .item()
                # (accepted_len) instead of the oracle's posterior.tolist() +
                # child-map python walk; the path/correction stay device tensors
                # so the gather/hidden/commit steps below never re-upload them.
                # max_depth = block_size (the tree's depth budget) keeps the
                # parent-walk loop short and sync-free. temperature>0 stays on
                # the CPU oracle (gpu_tree_accept is greedy-only).
                from ptd.tree._core.accept import gpu_tree_accept
                greedy = target_logits.argmax(dim=-1).squeeze(0)      # (N,) device
                path_t, acc, corr_t = gpu_tree_accept(
                    tree.token_ids, greedy, tree.parent_indices, tree.depth,
                    max_depth=block_size,
                )
            else:
                accepted_path, acc, correction = tree_accept(tree, target_logits, sp.temperature)
                path_t = torch.tensor(accepted_path, device=self.device)
                corr_t = torch.tensor(correction, device=self.device)
            if logical_kv:
                # L5 slot-commit (replaces the O(context) gather): the accepted nodes'
                # KV stays at the slots the verify wrote; only the WINDOW MAP is
                # rewritten — copy the accepted entries (advanced indexing copies, so
                # the overlapping write-back is safe; path_t[0]=0 maps wlen->wlen) down
                # to the committed region, then advance the window. O(acc+1) work.
                kept = slots_buf[:, wlen + path_t]
                slots_buf[:, wlen:wlen + int(kept.shape[1])] = kept
                # Free policy: a round's reservation is block-aligned, so block j of
                # EVERY layer holds exactly nodes [16j, 16j+16) of THIS round — a block
                # survives iff it holds an accepted node; pure-rejected (incl. pad)
                # blocks recycle now. Dead slots inside kept blocks leak until the
                # per-decode cache drops (~bounded by reserve_capacity's formula).
                # path_t is already on host for the EOS/commit step below — one small
                # DtoH, same data the round syncs anyway.
                path_list = path_t.tolist()
                kept_j = {p // self.block_size for p in path_list}
                freed_idx = [j for j in range(len(round_blocks))
                             if j not in kept_j]
                cache.release_round_blocks(round_blocks, freed_idx)
                wlen += len(path_list)
            else:
                # GATHER: keep prefix + accepted path (root + accepted nodes); drop the
                # rejected branches' KV. path_t = [0(root), …acc nodes], tree-ordered;
                # their cache slots are past_len + path -> contiguous positions after gather.
                keep = torch.cat([
                    torch.arange(past_len, device=self.device),
                    past_len + path_t,
                ])
                cache.gather(keep)                # cache length -> past_len + (acc + 1)
            if new_hidden is not None:
                # append [root | accepted nodes] hidden (the correction has none yet —
                # it is the next anchor, fed via noise). Restores the invariant.
                target_hidden = torch.cat([target_hidden, new_hidden[:, path_t, :]], dim=1)
            accepted = tree.token_ids[path_t[1:]] if acc > 0 \
                else torch.empty(0, dtype=tree.token_ids.dtype, device=self.device)
            block = torch.cat([accepted, corr_t.reshape(1)])
            committed = torch.cat([committed, block.view(1, -1)], dim=1)
            rounds += 1
            accept_lengths.append(int(block.numel()))   # acc + 1, matches reference accept-len
            tree_sizes.append(int(N))
            for t in block.tolist():
                new_ids.append(int(t))
                if int(t) in self.eos_token_ids:
                    break
            if new_ids and new_ids[-1] in self.eos_token_ids:
                break
        new_ids = new_ids[: sp.max_new_tokens]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        out = {"token_ids": new_ids, "text": text, "tpf": (len(new_ids) / rounds if rounds else 0.0)}
        if return_stats:
            out["accept_lengths"] = accept_lengths   # per-round (acc+1)
            out["tree_sizes"] = tree_sizes           # per-round node count
            out["rounds"] = rounds
        return out

    @torch.inference_mode()
    def generate_batch(self, prompts: list, sampling_params: SamplingParams = None) -> list:
        """Continuous-batched greedy/temperature AR over the shared multi-seq paged
        cache (nano_vllm N2a). Returns a list of `{token_ids, text}` aligned to
        `prompts`, each token-identical to `generate(prompt)` run alone — the N2a
        lossless gate.

        Each `prompts[i]` is a str (tokenized raw) or a tokenized `(1, T)` tensor.
        The `Scheduler` admits every prompt into one fixed `PagedKVCache` pool
        (per-`seq_id` block tables); a `SequenceRequest` carries each sequence's
        state. The loop is the pad-to-max batched forward of the N2a design: prefill
        each admitted prompt (its prefix KV lands in the pool under its `seq_id`),
        then each decode step reconstructs every live sequence's dense KV from the
        pool, pads to the batch max length, runs ONE forward under a 4D padding mask
        (so attention only sees each sequence's real positions — padded KV is masked,
        not attended), appends the new token's KV back into the pool per `seq_id`,
        and samples each sequence's next token. A sequence drops out of the batch on
        EOS or once it hits its token budget; the survivors keep decoding.

        Lossless because each sequence sees exactly the KV / RoPE positions / causal
        visibility it would see decoding alone — the only difference is the pooled
        storage and the per-step pad+mask, both of which are masked-out no-ops for
        attention. fp32 bitwise-equal on CPU; bf16 carries the same SDPA
        reduction-order caveat as N0/N1."""
        sp = sampling_params or SamplingParams()
        # Tokenize / normalize prompts to input_id lists.
        prompt_ids = []
        for p in prompts:
            if isinstance(p, str):
                ids = self.tokenizer(p, return_tensors="pt").input_ids[0].tolist()
            else:
                ids = p.to(self.device).view(-1).tolist()
            prompt_ids.append([int(t) for t in ids])

        cache = PagedKVCache(
            block_size=self.block_size, max_batch_size=max(2, len(prompt_ids)),
            device=torch.device(self.device), dtype=self.dtype,
        )
        scheduler = Scheduler(cache, max_batch_size=max(2, len(prompt_ids)))
        for seq_id, ids in enumerate(prompt_ids):
            scheduler.admit_request(SequenceRequest(
                seq_id=seq_id, input_ids=list(ids),
                max_new_tokens=sp.max_new_tokens, temperature=sp.temperature,
            ))
        scheduler.step()                              # FCFS admit all into the pool

        num_layers = self.model.config.num_hidden_layers
        results = {sid: {"token_ids": []} for sid in range(len(prompt_ids))}

        # --- prefill: each admitted prompt forwards once (batch=1) into the pool
        # under its seq_id; sample its first token. (Prefills are independent, so a
        # per-seq forward is simplest and identical to the single-stream prefill.)
        for sid, ids in enumerate(prompt_ids):
            input_ids = torch.tensor([ids], device=self.device)
            self._prefill_into_pool(cache, sid, input_ids, num_layers)
            logits = self._last_prefill_logits
            next_tok = int(sample(logits[:, -1:, :], sp.temperature).item())
            results[sid]["token_ids"].append(next_tok)
            scheduler.mark_decode_step(sid, next_tok, logits[:, -1:, :])

        # Active = sequences still decoding (not finished). Drop on EOS or budget.
        active = [sid for sid in range(len(prompt_ids))
                  if not self._is_finished(results[sid]["token_ids"], sp)]

        # --- decode: batched single-token steps over the live sequences ---------
        while active:
            toks = [results[sid]["token_ids"][-1] for sid in active]
            logits = self._batched_decode_forward(cache, active, toks, num_layers, sp.temperature)
            for i, sid in enumerate(active):
                next_tok = int(sample(logits[i:i + 1, -1:, :], sp.temperature).item())
                results[sid]["token_ids"].append(next_tok)
                scheduler.mark_decode_step(sid, next_tok, logits[i:i + 1, -1:, :])
            active = [sid for sid in active
                      if not self._is_finished(results[sid]["token_ids"], sp)]

        out = []
        for sid in range(len(prompt_ids)):
            ids = results[sid]["token_ids"][: sp.max_new_tokens]
            out.append({"token_ids": ids,
                        "text": self.tokenizer.decode(ids, skip_special_tokens=True)})
        return out

    def _is_finished(self, token_ids: list, sp: SamplingParams) -> bool:
        """A sequence is done once it hit EOS or its `max_new_tokens` budget."""
        return (len(token_ids) >= sp.max_new_tokens
                or (token_ids and token_ids[-1] in self.eos_token_ids))

    def _prefill_into_pool(self, cache: PagedKVCache, seq_id: int,
                           input_ids: torch.Tensor, num_layers: int) -> None:
        """Forward a prompt once (batch=1) and transfer its prefix KV into the
        shared pool under `seq_id`. HF's `update` can't route a per-row `seq_id`
        through a batched forward, so we prefill into a throwaway `DynamicCache`
        and copy each layer's KV into the pool via `append(..., seq_id=seq_id)`.
        Stashes the prompt's logits for the caller to sample the first token."""
        pos = torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0)
        scratch = DynamicCache()
        logits, scratch, _ = self.runner.forward(input_ids, scratch, pos)
        for layer_idx in range(num_layers):
            keys = scratch.layers[layer_idx].keys        # (1, H, T, D)
            values = scratch.layers[layer_idx].values
            cache.append(keys, values, layer_idx, seq_id=seq_id)
        self._last_prefill_logits = logits

    def _batched_decode_forward(self, cache: PagedKVCache, seq_ids: list,
                                tokens: list, num_layers: int, temperature: float):
        """One pad-to-max batched decode step over `seq_ids` (the N2a forward).

        Reconstructs every sequence's dense KV from the pool, pads to the batch max
        cached length, builds a 4D additive mask that exposes each sequence's real
        prefix + its own new token (padded KV masked out), forwards once, then
        appends the new token's KV back into the pool per `seq_id`. Returns the
        batched logits `(B, 1, V)`.

        On the N3 kernel path (`attn_backend == "triton_paged_tree"`) there is no
        pad+mask+copy-back: `update` appends each row's new-token KV to its own seq
        and the kernel reads `[0, past_i + 1)` straight from the pool (each seq is a
        pure decode -> qq_bias=None). Returns the same `(B, 1, V)` logits."""
        B = len(seq_ids)
        if getattr(self, "attn_backend", "sdpa") == "triton_paged_tree":
            return self._batched_decode_forward_kernel(cache, seq_ids, tokens)
        seq_lens = [cache.get_seq_length(0, seq_id=s) for s in seq_ids]
        max_len = max(seq_lens)
        neg = torch.finfo(self.dtype).min

        # Build a padded DynamicCache from each seq's logical KV (right-padded with
        # zeros to max_len; the padding columns are masked out below).
        batched = DynamicCache()
        for layer_idx in range(num_layers):
            k_pad = torch.zeros((B, cache._num_heads, max_len, cache._head_dim),
                                dtype=self.dtype, device=self.device)
            v_pad = torch.zeros_like(k_pad)
            for i, s in enumerate(seq_ids):
                gk, gv = cache._logical_kv(layer_idx, seq_id=s)   # (1, H, S_i, D)
                S_i = gk.shape[2]
                k_pad[i, :, :S_i, :] = gk[0]
                v_pad[i, :, :S_i, :] = gv[0]
            batched.update(k_pad, v_pad, layer_idx)

        # New token per seq + its RoPE position (= the seq's cached length).
        input_ids = torch.tensor([[t] for t in tokens], device=self.device)
        position_ids = torch.tensor([[s] for s in seq_lens], device=self.device)
        # 4D additive mask (B, 1, Q=1, KV=max_len+1): real prefix cols [0, S_i)
        # allowed, padded cols [S_i, max_len) masked, the new-token col (max_len)
        # is self-visible. cache_position=max_len places the new KV uniformly.
        mask = torch.full((B, 1, 1, max_len + 1), neg, dtype=self.dtype, device=self.device)
        for i, S_i in enumerate(seq_lens):
            mask[i, 0, 0, :S_i] = 0.0
            mask[i, 0, 0, max_len] = 0.0
        cache_position = torch.tensor([max_len], device=self.device)
        logits, batched, _ = self.runner.forward(
            input_ids, batched, position_ids,
            attention_mask=mask, cache_position=cache_position,
        )
        # Append each seq's new-token KV (col max_len) back into the pool.
        for layer_idx in range(num_layers):
            keys = batched.layers[layer_idx].keys        # (B, H, max_len+1, D)
            values = batched.layers[layer_idx].values
            for i, s in enumerate(seq_ids):
                k_new = keys[i:i + 1, :, max_len:max_len + 1, :]
                v_new = values[i:i + 1, :, max_len:max_len + 1, :]
                cache.append(k_new, v_new, layer_idx, seq_id=s)
        return logits

    def _batched_decode_forward_kernel(self, cache: PagedKVCache, seq_ids: list,
                                       tokens: list):
        """N3 kernel decode step: forward the REAL pooled cache, no pad/mask/copy-back.

        Each seq is a pure decode (one query row): `update` routes each batch row's
        new-token KV to its own `seq_id` (per `_handoff_seq_ids`) and the kernel
        reads `[0, past_i + 1)` from the pool with qq_bias=None (causal decode).
        position_ids = per-seq past length; cache_position is irrelevant on this
        path (the cache appends by seq order, not cache_position). Returns `(B, 1, V)`."""
        seq_lens = [cache.get_seq_length(0, seq_id=s) for s in seq_ids]
        cache._paged_handoff = True
        cache._handoff_seq_ids = list(seq_ids)
        cache._ptd_attn_meta = {"seq_ids": list(seq_ids), "qq_bias": None}
        input_ids = torch.tensor([[t] for t in tokens], device=self.device)
        position_ids = torch.tensor([[s] for s in seq_lens], device=self.device)
        logits, _, _ = self.runner.forward(
            input_ids, cache, position_ids, attention_mask=None,
        )
        return logits

    @torch.inference_mode()
    def generate_tree_batch(self, prompts: list, tree_drafter, block_size: int = 4,
                            tree_width: int = 2, budget: int = 15, algo: str = "crossproduct",
                            algo_kwargs: dict = None, sampling_params: SamplingParams = None) -> list:
        """Batched per-sequence TREE-spec decode over the shared multi-seq paged
        cache (nano_vllm N2b). Returns a list of `{token_ids, text, tpf}` aligned to
        `prompts`, each token-identical to single-stream `generate_tree(prompt)` run
        alone — the N2b lossless gate.

        Always uses the SDPA path regardless of `self.attn_backend`: the N3 kernel
        path is a follow-on for N2b. It pads queries to `S = max_N`, so `total_q =
        B*max_N` no longer matches Unit-2's ragged `qq_bias` (sum N_i) and padding-
        node KV would pollute the per-seq pool; that needs a padded `qq_bias` or
        query compaction (out of scope here).

        This is the tree-spec analogue of `generate_batch` (N2a) and the batched
        analogue of `generate_tree` (N1). Each round, every live sequence builds its
        OWN draft tree (`get_algorithm(algo).build` on its drafter logits — the N1
        per-seq path, with possibly different node counts N_i), the trees are padded
        to `max_N` and verified in ONE batched forward under the design's padded 4D
        mask, and each sequence's accepted root-to-leaf path is taken by `tree_accept`
        on its own logit slice. Each sequence's KV is gathered independently
        (ref-count-safe `PagedKVCache` per-seq append), so dropping a finished
        sequence never perturbs the survivors.

        Per-seq isolation comes from the additive mask `(B, 1, max_N, max_len + max_N)`
        (`max_len = max(past_len[i])`): for seq i, query j < N_i sees its real prefix
        columns `[0, past_len[i])`, its own tree nodes `[max_len, max_len + N_i)`
        filtered by the ancestor matrix, and nothing else (padding prefix columns,
        other seqs' tree columns, and padding tree columns are all `-inf`); padding
        queries j >= N_i are fully masked. RoPE positions are per-seq depth-relative
        (`past_len[i] + depth[i, j]`), `cache_position` is uniform — exactly the N1
        single-stream geometry replicated per row, so each sequence's attention graph
        is isomorphic to verifying its tree alone.

        Lossless by construction (commits the verify forward's own greedy along each
        accepted path); fp32 bitwise-equal to single-stream `generate_tree` on CPU,
        bf16 carries the same SDPA reduction-order caveat as N0/N1/N2a.

        `algo` / `algo_kwargs` mirror `generate_tree`; all bundled algorithms recover
        crossproduct at their identity knobs, so the choice is lossless regardless.
        Hidden-state-conditioned (DraftHead) drafting is N1-only for now: this batched
        route runs the no-hidden path (`target_hidden=None`), which is what the stub
        drafters that gate it exercise."""
        # tree contract (engine -> tree, one-way): import only the public ptd.tree API
        from ptd.tree import get_algorithm, build_ancestor_matrix, tree_accept

        sp = sampling_params or SamplingParams()
        # Tokenize / normalize prompts to input_id lists.
        prompt_ids = []
        for p in prompts:
            if isinstance(p, str):
                ids = self.tokenizer(p, return_tensors="pt").input_ids[0].tolist()
            else:
                ids = p.to(self.device).view(-1).tolist()
            prompt_ids.append([int(t) for t in ids])
        n_seq = len(prompt_ids)

        D = max(1, block_size - 1)
        algo_obj = get_algorithm(algo, **(algo_kwargs or {}))
        num_layers = self.model.config.num_hidden_layers
        cache = PagedKVCache(
            block_size=self.block_size, max_batch_size=max(2, n_seq),
            device=torch.device(self.device), dtype=self.dtype,
        )
        scheduler = Scheduler(cache, max_batch_size=max(2, n_seq))
        for seq_id, ids in enumerate(prompt_ids):
            scheduler.admit_request(SequenceRequest(
                seq_id=seq_id, input_ids=list(ids),
                max_new_tokens=sp.max_new_tokens, temperature=sp.temperature,
            ))
        scheduler.step()                              # FCFS admit all into the pool

        # Per-seq decode state. `committed` is the full token stream incl. the anchor
        # (= each round's tree root); the pool trails it by that anchor (the root
        # enters via the verify forward, exactly as in N1). `new_ids` accumulates the
        # generated tokens; `rounds` counts verify forwards for tpf.
        committed = {}
        new_ids = {sid: [] for sid in range(n_seq)}
        rounds = {sid: 0 for sid in range(n_seq)}

        # --- prefill: each prompt forwards once into the pool under its seq_id;
        # sample its first token (the anchor / first tree root). Mirrors the N2a
        # prefill, plus N1's anchor bookkeeping. ---------------------------------
        for sid, ids in enumerate(prompt_ids):
            input_ids = torch.tensor([ids], device=self.device)
            self._prefill_into_pool(cache, sid, input_ids, num_layers)
            logits = self._last_prefill_logits
            first_tok = int(sample(logits[:, -1:, :], sp.temperature).item())
            new_ids[sid].append(first_tok)
            # committed = prompt + anchor; the anchor is NOT yet cached (it is the
            # tree root, fed via the next verify forward).
            committed[sid] = torch.cat(
                [input_ids, torch.tensor([[first_tok]], device=self.device)], dim=1)

        # Active = sequences still decoding (not finished on the first token).
        active = [sid for sid in range(n_seq)
                  if not self._is_finished(new_ids[sid], sp)]

        # --- decode: each round builds every live seq's tree, verifies in one
        # batched forward, then per-seq tree_accept + ref-count-safe pool append. --
        while active:
            trees, ancestors, past_lens = [], [], []
            for sid in active:
                draft_logits = tree_drafter.propose_logits(
                    committed[sid], D, target_hidden=None).to(self.device)   # (1, D, V)
                tree = algo_obj.build(
                    int(committed[sid][0, -1]), draft_logits, block_size, tree_width,
                    budget, self.device)
                trees.append(tree)
                ancestors.append(build_ancestor_matrix(tree).bool())
                past_lens.append(cache.get_seq_length(seq_id=sid))   # == committed-1
            max_N = max(t.num_nodes for t in trees)

            # ONE batched verify forward over the padded per-seq trees.
            tree_kv = self._batched_tree_verify_forward(
                cache, active, trees, ancestors, past_lens, max_N, num_layers)
            logits = tree_kv["logits"]                # (B, max_N, V)

            still = []
            for i, sid in enumerate(active):
                tree = trees[i]
                N = tree.num_nodes
                logits_i = logits[i:i + 1, :N, :]     # (1, N, V) — only real nodes
                accepted_path, acc, correction = tree_accept(tree, logits_i, sp.temperature)
                rounds[sid] += 1
                # Append [root | accepted nodes] tree-node KV to the pool (the N1
                # gather's keep set, applied per-seq). The pool grows from
                # past_len[i] to past_len[i] + (acc + 1); the correction has no KV
                # yet (it becomes the next round's anchor / root).
                self._append_tree_path_kv(cache, sid, tree_kv, i, accepted_path, num_layers)
                accepted = tree.token_ids[torch.tensor(accepted_path[1:], device=self.device)] \
                    if acc > 0 else torch.empty(0, dtype=tree.token_ids.dtype, device=self.device)
                block = torch.cat([accepted, torch.tensor([correction], device=self.device)])
                committed[sid] = torch.cat([committed[sid], block.view(1, -1)], dim=1)
                for t in block.tolist():
                    new_ids[sid].append(int(t))
                    if int(t) in self.eos_token_ids:
                        break
                if not self._is_finished(new_ids[sid], sp):
                    still.append(sid)
            active = still

        out = []
        for sid in range(n_seq):
            ids = new_ids[sid][: sp.max_new_tokens]
            r = rounds[sid]
            out.append({"token_ids": ids,
                        "text": self.tokenizer.decode(ids, skip_special_tokens=True),
                        "tpf": (len(ids) / r if r else 0.0)})
        return out

    def _batched_tree_verify_forward(self, cache: PagedKVCache, seq_ids: list,
                                     trees: list, ancestors: list, past_lens: list,
                                     max_N: int, num_layers: int) -> dict:
        """One padded batched tree-verify forward over `seq_ids` (the N2b forward).

        Reconstructs every sequence's dense prefix KV from the pool, pads to the
        batch max prefix length `max_len`, appends each seq's `N_i` tree-node columns
        (right-padded to `max_N`), and forwards once under the design's 4D additive
        mask so each sequence's tree nodes see ONLY that seq's prefix + their tree
        ancestors. Returns `{"logits": (B, max_N, V), "cache": DynamicCache,
        "max_len": max_len}`; the caller slices each seq's accepted-path KV out of
        the verify cache and appends it to the pool via `_append_tree_path_kv`.

        Geometry (per seq i, the N1 single-stream verify replicated per row):
          - input_ids[i, j]    = tree_i.token_ids[j]              (j < N_i; pad 0)
          - position_ids[i, j] = past_len[i] + depth_i[j]         (RoPE, depth-based)
          - cache_position[j]  = max_len + j                      (uniform append slot)
          - mask[i, 0, j, k]:  0 on real prefix cols [0, past_len[i]); 0 on tree cols
            [max_len, max_len + N_i) where ancestor_i[j, k - max_len]; -inf elsewhere
            (padding prefix cols, other seqs' / padding tree cols). Padding queries
            j >= N_i are fully -inf (never attend)."""
        B = len(seq_ids)
        max_len = max(past_lens)
        neg = torch.finfo(self.dtype).min
        kv_len = max_len + max_N

        # Build a padded DynamicCache from each seq's logical prefix KV (right-padded
        # with zeros to max_len; padding columns are masked out below). The verify
        # forward appends the max_N tree columns onto this, giving (B, H, kv_len, D).
        batched = DynamicCache()
        for layer_idx in range(num_layers):
            k_pad = torch.zeros((B, cache._num_heads, max_len, cache._head_dim),
                                dtype=self.dtype, device=self.device)
            v_pad = torch.zeros_like(k_pad)
            for i, s in enumerate(seq_ids):
                gk, gv = cache._logical_kv(layer_idx, seq_id=s)   # (1, H, past_len[i], D)
                P_i = gk.shape[2]
                k_pad[i, :, :P_i, :] = gk[0]
                v_pad[i, :, :P_i, :] = gv[0]
            batched.update(k_pad, v_pad, layer_idx)

        # Per-seq tree nodes (token_ids / RoPE positions), right-padded to max_N.
        input_ids = torch.zeros((B, max_N), dtype=torch.long, device=self.device)
        position_ids = torch.zeros((B, max_N), dtype=torch.long, device=self.device)
        for i, s in enumerate(seq_ids):
            tree, N, past = trees[i], trees[i].num_nodes, past_lens[i]
            input_ids[i, :N] = tree.token_ids
            position_ids[i, :N] = past + tree.depth.to(self.device)

        # 4D additive mask (B, 1, max_N, kv_len): per-seq prefix + ancestor isolation.
        mask = torch.full((B, 1, max_N, kv_len), neg, dtype=self.dtype, device=self.device)
        for i, s in enumerate(seq_ids):
            N, past = trees[i].num_nodes, past_lens[i]
            mask[i, 0, :N, :past] = 0.0                         # real prefix: visible
            tree_block = torch.where(                           # tree cols: ancestor relation
                ancestors[i],
                torch.zeros((), dtype=self.dtype, device=self.device),
                torch.full((), neg, dtype=self.dtype, device=self.device))
            mask[i, 0, :N, max_len:max_len + N] = tree_block    # (N, N) ancestor-masked
            # rows j >= N (padding queries) stay fully -inf; cols outside the two
            # blocks (padding prefix, other seqs' / padding tree cols) stay -inf.
        cache_position = torch.arange(max_len, max_len + max_N, device=self.device)
        logits, batched, _ = self.runner.forward(
            input_ids, batched, position_ids,
            attention_mask=mask, cache_position=cache_position,
        )
        return {"logits": logits, "cache": batched, "max_len": max_len}

    def _append_tree_path_kv(self, cache: PagedKVCache, seq_id: int, tree_kv: dict,
                             batch_idx: int, accepted_path: list, num_layers: int) -> None:
        """Append seq `seq_id`'s accepted-path tree-node KV from the verify cache
        into the pool (the per-seq, ref-count-safe analogue of N1's `gather`).

        `accepted_path` is `[0(root), …acc accepted nodes]` in tree order; their KV
        sits at columns `max_len + node_idx` of the verify cache. After this append
        the pool length for `seq_id` is `past_len + (acc + 1)`, restoring the N1
        invariant (pool == committed minus the next anchor)."""
        max_len = tree_kv["max_len"]
        verify = tree_kv["cache"]
        cols = max_len + torch.tensor(accepted_path, device=self.device)   # (acc+1,)
        for layer_idx in range(num_layers):
            keys = verify.layers[layer_idx].keys          # (B, H, max_len + max_N, D)
            values = verify.layers[layer_idx].values
            k_path = keys[batch_idx:batch_idx + 1, :, cols, :]    # (1, H, acc+1, D)
            v_path = values[batch_idx:batch_idx + 1, :, cols, :]
            cache.append(k_path, v_path, layer_idx, seq_id=seq_id)

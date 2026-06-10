"""DraftHead-backed drafters — the real (checkpoint-gated) speculative drafters.

`DraftHeadDrafter` (chain) and `DraftHeadTreeDrafter` (tree) wrap a loaded DFlash
draft head (`ptd.models.draft_head.DFlashDraftModel`). The head owns neither
`embed_tokens` nor `lm_head` — it shares the *target's* (the DFlash convention),
so these drafters take the target module too and call `target.model.embed_tokens`
/ `target.lm_head` exactly as the reference `benchmark.py` does.

Kept in a separate module (not appended to `draft.py`) so the stub file stays
free of `transformers` / checkpoint imports; the real head pulls in `DynamicCache`
and the loaded model.

`_forward_head` mirrors `causal_parallel_drafting/benchmark.py` lines ~153-207
(chain, `tree_width=1`): build a noise embedding from `[anchor, mask_id*(block-1)]`,
run the head conditioned on the anchor's `target_hidden`, slice the `block_size-1`
real-prediction positions (gated on `draft_shift` — never hardcode), and apply the
target's `lm_head`. The recompute design feeds the running committed-context hidden
each round (the engine owns the decode loop; no draft-side KV reuse here).
"""
import torch
from transformers import DynamicCache

from ptd.draft import Drafter, TreeDrafter


class _DraftHeadForward:
    """Shared head-forward helper for the chain + tree drafters.

    Both drafters need the same single-step head forward; this holds the head /
    target / config and exposes `_forward_head` returning raw `(1, depth, V)`
    logits. The two public drafters compose it (chain argmaxes, tree returns raw).
    """

    def __init__(self, head, target, block_size: int, target_layer_ids, draft_shift: bool = False):
        self.head = head
        self.target = target
        self.block_size = block_size
        self.target_layer_ids = target_layer_ids
        self.draft_shift = draft_shift
        # The head's device + dtype are the source of truth: every tensor we
        # build (anchor row, mask-id placeholders, position ids) goes here so
        # embed_tokens / fc / lm_head never hit a device or dtype mismatch.
        self.device = next(head.parameters()).device
        self.dtype = next(head.parameters()).dtype
        self.mask_token_id = head.mask_token_id

    def block_output_ids(self, context_ids: torch.Tensor, fill_tokens=None) -> torch.Tensor:
        block_size = self.block_size
        anchor = context_ids[0, -1].view(1, 1).to(self.device)
        max_fill = block_size - 1
        if fill_tokens is None:
            fill = torch.full(
                (1, max_fill), self.mask_token_id, dtype=anchor.dtype, device=self.device
            )
        else:
            path = torch.as_tensor(fill_tokens, dtype=anchor.dtype, device=self.device).reshape(1, -1)
            if path.shape[1] > max_fill:
                path = path[:, :max_fill]
            mask_len = max_fill - int(path.shape[1])
            mask_fill = torch.full(
                (1, mask_len), int(self.mask_token_id), dtype=anchor.dtype, device=self.device
            )
            fill = torch.cat([path, mask_fill], dim=1)
        return torch.cat([anchor, fill], dim=1)

    def noise_embedding(self, context_ids: torch.Tensor, fill_tokens=None) -> torch.Tensor:
        block_output_ids = self.block_output_ids(context_ids, fill_tokens=fill_tokens)
        return self.target.model.embed_tokens(block_output_ids)  # (1, block_size, H)

    def _forward_head(
        self,
        context_ids: torch.Tensor,
        target_hidden: torch.Tensor,
        depth: int,
        fill_tokens=None,
    ) -> torch.Tensor:
        """Run the head once and return `(1, depth, V)` draft logits.

        `context_ids` (1, T): committed context; its last token is the anchor (the
        speculative-block root). `target_hidden` (1, ctx_len, dim_concat): the
        tapped target hidden states the head conditions on (the K/V context).
        `depth`: number of real-prediction positions to return (= block_size - 1).
        """
        if target_hidden is None:
            raise ValueError(
                "DraftHead drafters require target_hidden; pass it from the "
                "ModelRunner forward (output_hidden_states + target_layer_ids)."
            )
        block_size = self.block_size
        # block_output_ids = [anchor, fill...] of length block_size. Normal draft
        # calls fill with masks; conditioned calls fill with path tokens then masks.
        noise_embedding = self.noise_embedding(context_ids, fill_tokens=fill_tokens)

        ctx_len = target_hidden.shape[1]
        # The head's K/V context is [ctx_tokens ; block positions]; rotary needs
        # absolute positions over that concatenation (benchmark.py:163 draft_pos_ids).
        position_ids = torch.arange(ctx_len + block_size, device=self.device).unsqueeze(0)

        hidden = self.head(
            target_hidden=target_hidden.to(device=self.device, dtype=self.dtype),
            noise_embedding=noise_embedding,
            position_ids=position_ids,
            past_key_values=DynamicCache(),
            use_cache=False,
            is_causal=self.head.resolve_causal_head("auto"),
        )  # (1, block_size, H)

        # Draft-logit slice: gated on draft_shift, never hardcode (the I-DLM bug).
        #   in-place (draft_shift=False): positions 1..block_size-1 are predictions
        #     -> slice(-block_size+1, None)
        #   shift (draft_shift=True): positions 0..block_size-2 are predictions
        #     -> slice(0, block_size-1)
        draft_slice = slice(0, block_size - 1) if self.draft_shift else slice(-block_size + 1, None)
        draft_logits = self.target.lm_head(hidden[:, draft_slice, :])  # (1, block_size-1, V)
        return draft_logits[:, :depth, :]


class DraftHeadDrafter(Drafter):
    """Chain drafter backed by a trained DFlash draft head.

    One head forward proposes `k = block_size - 1` next tokens (argmax of the
    per-position draft logits). The engine's chain verify loop accepts the longest
    target-agreeing prefix — lossless regardless of draft quality.
    """

    def __init__(self, head, target, block_size: int, target_layer_ids, draft_shift: bool = False):
        self._fwd = _DraftHeadForward(head, target, block_size, target_layer_ids, draft_shift)
        self.head = head
        self.target = target
        self.block_size = block_size
        self.target_layer_ids = target_layer_ids
        self.draft_shift = draft_shift

    @torch.inference_mode()
    def propose(
        self,
        context_ids: torch.Tensor,
        k: int,
        target_hidden: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        draft_logits = self._fwd._forward_head(context_ids, target_hidden, k)  # (1, k, V)
        return draft_logits.squeeze(0).argmax(dim=-1)  # (k,)


class DraftHeadTreeDrafter(TreeDrafter):
    """Tree drafter backed by a trained DFlash draft head.

    Emits the raw per-depth draft logits `(1, depth, V)` from one head forward;
    the tree algorithm turns them into a DraftTree and the engine verifies all
    nodes under a 4D ancestor mask. Lossless regardless of draft quality.
    """

    def __init__(self, head, target, block_size: int, target_layer_ids, draft_shift: bool = False):
        self._fwd = _DraftHeadForward(head, target, block_size, target_layer_ids, draft_shift)
        self.head = head
        self.target = target
        self.block_size = block_size
        self.target_layer_ids = target_layer_ids
        self.draft_shift = draft_shift

    @torch.inference_mode()
    def propose_logits(
        self,
        context_ids: torch.Tensor,
        depth: int,
        target_hidden: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._fwd._forward_head(context_ids, target_hidden, depth)  # (1, depth, V)

    @torch.inference_mode()
    def propose_logits_conditioned(
        self,
        context_ids: torch.Tensor,
        path_tokens,
        target_hidden: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._fwd._forward_head(
            context_ids,
            target_hidden,
            self.block_size - 1,
            fill_tokens=path_tokens,
        )  # (1, block_size-1, V)


_DRAFT_HEAD_CTX_BUCKETS = (512, 1024, 2048, 4096, 8192, 16384, 32768)


def _draft_head_bucket_for_ctx_len(ctx_len: int, buckets=_DRAFT_HEAD_CTX_BUCKETS) -> int:
    """Smallest context-capacity bucket >= ctx_len.

    Beyond the largest configured bucket, round up to a multiple of that largest
    bucket instead of specializing on every new context length.
    """
    n = int(ctx_len)
    for b in tuple(int(x) for x in buckets):
        if n <= b:
            return b
    step = int(tuple(buckets)[-1])
    return ((n + step - 1) // step) * step


class CompiledDraftHead:
    """Bucketed compiled wrapper around the DraftHead forward path.

    The eager `_forward_head` sees `target_hidden.shape[1] == real_ctx_len`, so
    tracing it directly would specialize/recapture as decode context grows. This
    wrapper pads `target_hidden` to a coarse capacity bucket, builds position ids
    with the real context positions followed by dummy masked pad positions, and
    supplies an additive mask that prevents real draft queries from attending to
    padded context keys. The compiled callable therefore sees only bucket shapes.
    """

    def __init__(
        self,
        head,
        target,
        block_size: int,
        target_layer_ids,
        draft_shift: bool = False,
        ctx_buckets=_DRAFT_HEAD_CTX_BUCKETS,
        compile: bool = True,
        fullgraph: bool = True,
    ):
        self.head = head
        self.target = target
        self.block_size = int(block_size)
        self.target_layer_ids = target_layer_ids
        self.draft_shift = draft_shift
        self.ctx_buckets = tuple(int(b) for b in ctx_buckets)
        if not self.ctx_buckets:
            raise ValueError("ctx_buckets must be non-empty")
        self.compile = bool(compile)
        self.fullgraph = bool(fullgraph)
        self.device = next(head.parameters()).device
        self.dtype = next(head.parameters()).dtype
        self.mask_token_id = head.mask_token_id
        if self.mask_token_id is None:
            raise ValueError("DraftHead mask_token_id is required")
        self._fwd = _DraftHeadForward(head, target, self.block_size, target_layer_ids, draft_shift)
        self._compiled_by_capacity = {}

    def bucket_for_ctx_len(self, ctx_len: int) -> int:
        return _draft_head_bucket_for_ctx_len(ctx_len, self.ctx_buckets)

    def pad_target_hidden(self, target_hidden: torch.Tensor, capacity: int) -> torch.Tensor:
        target_hidden = target_hidden.to(device=self.device, dtype=self.dtype)
        real_ctx_len = int(target_hidden.shape[1])
        capacity = int(capacity)
        if real_ctx_len > capacity:
            raise ValueError(f"target_hidden ctx_len={real_ctx_len} exceeds capacity={capacity}")
        if real_ctx_len == capacity:
            return target_hidden
        pad = torch.zeros(
            target_hidden.shape[0],
            capacity - real_ctx_len,
            target_hidden.shape[2],
            dtype=self.dtype,
            device=self.device,
        )
        return torch.cat([target_hidden, pad], dim=1)

    def _position_ids(self, real_ctx_len: int, capacity: int) -> torch.Tensor:
        real_ctx_len = int(real_ctx_len)
        capacity = int(capacity)
        pos = torch.empty(
            (1, capacity + self.block_size),
            dtype=torch.long,
            device=self.device,
        )
        if real_ctx_len > 0:
            pos[:, :real_ctx_len] = torch.arange(real_ctx_len, device=self.device)
        if capacity > real_ctx_len:
            pos[:, real_ctx_len:capacity] = max(real_ctx_len - 1, 0)
        pos[:, capacity:] = torch.arange(
            real_ctx_len,
            real_ctx_len + self.block_size,
            device=self.device,
        )
        return pos

    def _attention_mask(self, real_ctx_len: int, capacity: int) -> torch.Tensor:
        real_ctx_len = int(real_ctx_len)
        capacity = int(capacity)
        key_len = capacity + self.block_size
        mask = torch.zeros(
            (1, 1, self.block_size, key_len),
            dtype=self.dtype,
            device=self.device,
        )
        neg = torch.finfo(self.dtype).min
        if capacity > real_ctx_len:
            mask[:, :, :, real_ctx_len:capacity] = neg
        if self.head.resolve_causal_head("auto"):
            q = torch.arange(self.block_size, device=self.device).view(self.block_size, 1)
            k = torch.arange(self.block_size, device=self.device).view(1, self.block_size)
            block = torch.where(
                k <= q,
                torch.zeros((), dtype=self.dtype, device=self.device),
                torch.full((), neg, dtype=self.dtype, device=self.device),
            )
            mask[:, :, :, capacity:] = block.view(1, 1, self.block_size, self.block_size)
        return mask

    def noise_embedding(self, context_ids: torch.Tensor, fill_tokens=None) -> torch.Tensor:
        return self._fwd.noise_embedding(context_ids, fill_tokens=fill_tokens)

    def _forward_fixed(
        self,
        noise_embedding: torch.Tensor,
        target_hidden_padded: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        block_size = self.block_size
        hidden = self.head(
            target_hidden=target_hidden_padded,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )
        draft_slice = slice(0, block_size - 1) if self.draft_shift else slice(-block_size + 1, None)
        return self.target.lm_head(hidden[:, draft_slice, :])

    def _call_for_capacity(self, capacity: int):
        capacity = int(capacity)
        if not self.compile:
            return self._forward_fixed
        if capacity not in self._compiled_by_capacity:
            self._compiled_by_capacity[capacity] = torch.compile(
                self._forward_fixed,
                fullgraph=self.fullgraph,
                dynamic=False,
            )
        return self._compiled_by_capacity[capacity]

    @torch.inference_mode()
    def forward_padded(
        self,
        context_ids: torch.Tensor,
        target_hidden_padded: torch.Tensor,
        real_ctx_len: int,
        depth: int,
        fill_tokens=None,
    ) -> torch.Tensor:
        capacity = int(target_hidden_padded.shape[1])
        real_ctx_len = int(real_ctx_len)
        if real_ctx_len > capacity:
            raise ValueError(f"real_ctx_len={real_ctx_len} exceeds capacity={capacity}")
        target_hidden_padded = target_hidden_padded.to(device=self.device, dtype=self.dtype)
        noise_embedding = self.noise_embedding(context_ids, fill_tokens=fill_tokens)
        position_ids = self._position_ids(real_ctx_len, capacity)
        attention_mask = self._attention_mask(real_ctx_len, capacity)
        logits = self._call_for_capacity(capacity)(
            noise_embedding,
            target_hidden_padded,
            position_ids,
            attention_mask,
        )
        return logits[:, :depth, :]

    @torch.inference_mode()
    def __call__(
        self,
        context_ids: torch.Tensor,
        target_hidden: torch.Tensor,
        depth: int,
        fill_tokens=None,
        **kwargs,
    ) -> torch.Tensor:
        if target_hidden is None:
            raise ValueError(
                "CompiledDraftHead requires target_hidden; pass it from the "
                "ModelRunner forward (output_hidden_states + target_layer_ids)."
            )
        real_ctx_len = int(target_hidden.shape[1])
        capacity = self.bucket_for_ctx_len(real_ctx_len)
        target_hidden_padded = self.pad_target_hidden(target_hidden, capacity)
        return self.forward_padded(
            context_ids,
            target_hidden_padded,
            real_ctx_len,
            depth,
            fill_tokens=fill_tokens,
        )

    def propose_logits(
        self,
        context_ids: torch.Tensor,
        depth: int,
        target_hidden: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        return self(context_ids, target_hidden, depth)

    def propose_logits_conditioned(
        self,
        context_ids: torch.Tensor,
        path_tokens,
        target_hidden: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        return self(
            context_ids,
            target_hidden,
            self.block_size - 1,
            fill_tokens=path_tokens,
        )

    def propose(
        self,
        context_ids: torch.Tensor,
        k: int,
        target_hidden: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        draft_logits = self(context_ids, target_hidden, k)
        return draft_logits.squeeze(0).argmax(dim=-1)


class GraphedDraftHead:
    """CUDA-graph replay wrapper around `CompiledDraftHead`.

    One graph is captured lazily per context-capacity bucket. Per-round tensors
    are copied into persistent buffers before replay, so changing `ctx_len` within
    a bucket changes the staged `position_ids` and `attention_mask` without forcing
    a new capture.
    """

    def __init__(
        self,
        head,
        target,
        block_size: int,
        target_layer_ids,
        draft_shift: bool = False,
        ctx_buckets=_DRAFT_HEAD_CTX_BUCKETS,
        compile: bool = True,
    ):
        device = next(head.parameters()).device
        if device.type != "cuda":
            raise RuntimeError("GraphedDraftHead requires a CUDA DraftHead")
        self.compiled = CompiledDraftHead(
            head,
            target,
            block_size,
            target_layer_ids,
            draft_shift=draft_shift,
            ctx_buckets=ctx_buckets,
            compile=compile,
        )
        self.device = self.compiled.device
        self.dtype = self.compiled.dtype
        self.block_size = self.compiled.block_size
        self._buffers = {}
        self.graphs = {}
        self.outputs = {}
        self._pool = None

    def bucket_for_ctx_len(self, ctx_len: int) -> int:
        return self.compiled.bucket_for_ctx_len(ctx_len)

    def _buffers_for_capacity(self, capacity: int, target_hidden_dim: int, noise_hidden_dim: int):
        capacity = int(capacity)
        if capacity not in self._buffers:
            self._buffers[capacity] = {
                "noise_embedding": torch.zeros(
                    (1, self.block_size, noise_hidden_dim),
                    dtype=self.dtype,
                    device=self.device,
                ),
                "target_hidden": torch.zeros((1, capacity, target_hidden_dim), dtype=self.dtype, device=self.device),
                "position_ids": torch.zeros(
                    (1, capacity + self.block_size),
                    dtype=torch.long,
                    device=self.device,
                ),
                "attention_mask": torch.zeros(
                    (1, 1, self.block_size, capacity + self.block_size),
                    dtype=self.dtype,
                    device=self.device,
                ),
            }
        return self._buffers[capacity]

    def _copy_inputs(self, buffers, noise_embedding, target_hidden_padded, position_ids, attention_mask):
        buffers["noise_embedding"].copy_(noise_embedding.to(device=self.device, dtype=self.dtype))
        buffers["target_hidden"].copy_(target_hidden_padded)
        buffers["position_ids"].copy_(position_ids)
        buffers["attention_mask"].copy_(attention_mask)

    def _stage_inputs(self, context_ids: torch.Tensor, target_hidden: torch.Tensor, fill_tokens=None):
        real_ctx_len = int(target_hidden.shape[1])
        capacity = self.compiled.bucket_for_ctx_len(real_ctx_len)
        target_hidden_padded = self.compiled.pad_target_hidden(target_hidden, capacity)
        noise_embedding = self.compiled.noise_embedding(context_ids, fill_tokens=fill_tokens)
        position_ids = self.compiled._position_ids(real_ctx_len, capacity)
        attention_mask = self.compiled._attention_mask(real_ctx_len, capacity)
        buffers = self._buffers_for_capacity(
            capacity,
            int(target_hidden_padded.shape[2]),
            int(noise_embedding.shape[2]),
        )
        self._copy_inputs(buffers, noise_embedding, target_hidden_padded, position_ids, attention_mask)
        return capacity, buffers

    def _call_compiled(self, capacity: int, buffers):
        return self.compiled._call_for_capacity(capacity)(
            buffers["noise_embedding"],
            buffers["target_hidden"],
            buffers["position_ids"],
            buffers["attention_mask"],
        )

    @torch.inference_mode()
    def _capture_bucket(self, capacity: int, buffers):
        self._call_compiled(capacity, buffers)
        self._call_compiled(capacity, buffers)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            out = self._call_compiled(capacity, buffers)
        if self._pool is None:
            self._pool = graph.pool()
        self.graphs[capacity] = graph
        self.outputs[capacity] = out
        torch.cuda.synchronize()

    @torch.inference_mode()
    def __call__(
        self,
        context_ids: torch.Tensor,
        target_hidden: torch.Tensor,
        depth: int,
        fill_tokens=None,
        **kwargs,
    ) -> torch.Tensor:
        if target_hidden is None:
            raise ValueError("GraphedDraftHead requires target_hidden")
        capacity, buffers = self._stage_inputs(context_ids, target_hidden, fill_tokens=fill_tokens)
        if capacity not in self.graphs:
            self._capture_bucket(capacity, buffers)
        self.graphs[capacity].replay()
        return self.outputs[capacity][:, :depth, :]

    def propose_logits(
        self,
        context_ids: torch.Tensor,
        depth: int,
        target_hidden: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        return self(context_ids, target_hidden, depth)

    def propose_logits_conditioned(
        self,
        context_ids: torch.Tensor,
        path_tokens,
        target_hidden: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        return self(
            context_ids,
            target_hidden,
            self.block_size - 1,
            fill_tokens=path_tokens,
        )

    def propose(
        self,
        context_ids: torch.Tensor,
        k: int,
        target_hidden: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        draft_logits = self(context_ids, target_hidden, k)
        return draft_logits.squeeze(0).argmax(dim=-1)

# Graphed draft head with a K / V cache of the context, so each round only runs the draft block

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from jetspec.methods.jetspec.draft_head import rotate_half


def rope(x, cos, sin):
    # x: (1, H, S, D), cos and sin: (1, S, D)
    return x * cos.unsqueeze(1) + rotate_half(x) * sin.unsqueeze(1)


class CachedGraphedDraftHead:
    def __init__(self, graphed, chunk=64):
        # graphed: the GraphedDraftHead it replaces, we reuse its head, buckets, mask and positions
        # chunk: number of context tokens projected per replay of the projection graph
        self.c = graphed.compiled
        self.head = self.c.head
        self.target = self.c.target
        self.block_size = self.c.block_size
        self.draft_shift = self.c.draft_shift
        self.device = self.c.device
        self.dtype = self.c.dtype
        self.attn = ALL_ATTENTION_FUNCTIONS[self.head.config._attn_implementation]

        L = len(self.head.layers)
        Hkv = self.head.config.num_key_value_heads
        D = self.head.config.head_dim

        # K and V of every context token, per layer, shape (L, cmax + 1, Hkv, D)
        # row cmax is a trash row, the padding rows of the last chunk are written there
        self.cmax = self.c.ctx_buckets[-1]
        self.K = torch.zeros(L, self.cmax + 1, Hkv, D, dtype=self.dtype, device=self.device)
        self.V = torch.zeros_like(self.K)
        self.cached = 0

        # projection graph inputs, one chunk of target_hidden, its positions and its rows in K and V
        self.chunk = chunk
        self.th = None
        self.pos = torch.zeros(1, chunk, dtype=torch.long, device=self.device)
        self.dst = torch.full((chunk,), self.cmax, dtype=torch.long, device=self.device)
        self.ar = torch.arange(chunk, device=self.device)
        self.proj_fn = None
        self.proj_graph = None

        # one block graph per context bucket, with its buffers and its compiled fn
        self.ar_block = torch.arange(self.block_size, device=self.device).view(1, -1)
        self.graphs = {}
        self.outs = {}
        self.bufs = {}
        self.fns = {}

        # pool shared by the block graphs
        self.pool = None

    def reset_cache(self):
        # the engine calls it at each generate
        self.cached = 0

    def _project(self, th, pos):
        # th: (1, chunk, Dt) rows of target_hidden, pos: (1, chunk) their positions
        # only returns new tensors, writing K inside the compiled fn would make Inductor copy the whole store

        # (1, chunk, Dt) -> (1, chunk, H)
        h = self.head.hidden_dim_adapter(th)
        h = self.head.fc(h)
        h = self.head.hidden_norm(h)
        cos, sin = self.head.rotary_emb(h, pos)

        # k is normed then roped like in Qwen3DFlashAttention, v is only projected
        ks = []
        vs = []
        for layer in self.head.layers:
            at = layer.self_attn
            k = at.k_proj(h).view(1, self.chunk, -1, at.head_dim)
            k = at.k_norm(k).transpose(1, 2)
            k = rope(k, cos, sin).transpose(1, 2)
            v = at.v_proj(h).view(1, self.chunk, -1, at.head_dim)
            ks.append(k[0])
            vs.append(v[0])

        # (L, chunk, Hkv, D)
        return torch.stack(ks), torch.stack(vs)

    def _project_and_store(self):
        k, v = self.proj_fn(self.th, self.pos)

        # several padding rows hit the trash row cmax, which one wins doesn't matter
        self.K.index_copy_(1, self.dst, k)
        self.V.index_copy_(1, self.dst, v)

    def _project_rows(self, target_hidden, start, n):
        # target_hidden: (1, real, Dt), rows [start, start + n) are projected, n <= chunk

        # first call, same steps as GraphedDraftHead._capture_bucket
        if self.proj_graph is None:
            Dt = target_hidden.shape[-1]
            self.th = torch.zeros(1, self.chunk, Dt, dtype=self.dtype, device=self.device)
            self.proj_fn = torch.compile(self._project, fullgraph=True, dynamic=False)

            # dst is all cmax here, so the warm-up only writes the trash row
            self._project_and_store()
            self._project_and_store()
            torch.cuda.synchronize()

            # own pool, so a replay never writes over the block graph outputs
            self.proj_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.proj_graph):
                self._project_and_store()
            torch.cuda.synchronize()

        # the rows of th past n keep old values, dst sends them to the trash row
        self.th[:, :n].copy_(target_hidden[:, start:start + n])
        torch.add(self.ar, start, out=self.pos[0])
        self.dst.copy_(self.pos[0])
        self.dst[n:].fill_(self.cmax)
        self.proj_graph.replay()

    def _block(self, cap, noise, pos, mask):
        # cap: context bucket, noise: (1, B, H), pos: (1, B) block positions
        # mask: (1, 1, B, cap + B), set by CompiledDraftHead._attention_mask
        B = self.block_size
        h = noise
        cos, sin = self.head.rotary_emb(h, pos)

        for i, layer in enumerate(self.head.layers):
            at = layer.self_attn
            res = h
            x = layer.input_layernorm(h)

            # q, k, v of the block, (1, heads, B, D)
            q = at.q_proj(x).view(1, B, -1, at.head_dim)
            q = at.q_norm(q).transpose(1, 2)
            q = rope(q, cos, sin)
            k = at.k_proj(x).view(1, B, -1, at.head_dim)
            k = at.k_norm(k).transpose(1, 2)
            k = rope(k, cos, sin)
            v = at.v_proj(x).view(1, B, -1, at.head_dim).transpose(1, 2)

            # context rows [0, cap) from the store, (1, Hkv, cap, D)
            kc = self.K[i, :cap].transpose(0, 1).unsqueeze(0)
            vc = self.V[i, :cap].transpose(0, 1).unsqueeze(0)

            # rows past real hold old values but the mask sets them to finfo.min, (1, Hkv, cap + B, D)
            k = torch.cat([kc, k], dim=2)
            v = torch.cat([vc, v], dim=2)

            # is_causal=False like _forward_fixed, the mask holds the padding and the block causality
            o, _ = self.attn(at, q, k, v, mask, dropout=0.0, scaling=at.scaling,
                             sliding_window=at.sliding_window, is_causal=False)
            h = res + at.o_proj(o.reshape(1, B, -1))

            res = h
            h = layer.post_attention_layernorm(h)
            h = res + layer.mlp(h)

        # which positions are predictions depends on draft_shift, same slice as _forward_fixed
        h = self.head.norm(h)
        if self.draft_shift:
            sl = slice(0, B - 1)
        else:
            sl = slice(-B + 1, None)
        return self.target.lm_head(h[:, sl, :])

    def _capture_block(self, cap, noise, real):
        # same steps as GraphedDraftHead._capture_bucket, the block graphs share self.pool
        B = self.block_size
        buf = {}
        buf["noise"] = noise.clone()
        buf["pos"] = self.c._position_ids(real, cap)[:, -B:].clone()
        buf["mask"] = self.c._attention_mask(real, cap)
        buf["real"] = real
        fn = torch.compile(self._block, fullgraph=True, dynamic=False)

        fn(cap, buf["noise"], buf["pos"], buf["mask"])
        fn(cap, buf["noise"], buf["pos"], buf["mask"])
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            out = fn(cap, buf["noise"], buf["pos"], buf["mask"])
        if self.pool is None:
            self.pool = g.pool()
        torch.cuda.synchronize()

        self.graphs[cap] = g
        self.outs[cap] = out
        self.bufs[cap] = buf
        self.fns[cap] = fn

    @torch.inference_mode()
    def __call__(self, context_ids, target_hidden, depth, fill_tokens=None, **kwargs):
        # target_hidden: (1, real, Dt), append-only within a generate
        real = target_hidden.shape[1]

        # a shorter context is a new generate, the cache starts over
        if real < self.cached:
            self.cached = 0

        # past cmax, index_copy_ would write outside K
        if real > self.cmax:
            raise ValueError(f"context {real} > cmax {self.cmax}")

        # project the new context rows, one chunk per replay
        while self.cached < real:
            n = min(self.chunk, real - self.cached)
            self._project_rows(target_hidden, self.cached, n)
            self.cached += n

        cap = self.c.bucket_for_ctx_len(real)
        noise = self.c.noise_embedding(context_ids, fill_tokens=fill_tokens)
        if cap not in self.graphs:
            self._capture_block(cap, noise, real)

        # block positions are real .. real + B - 1
        buf = self.bufs[cap]
        buf["noise"].copy_(noise)
        torch.add(self.ar_block, real, out=buf["pos"])

        # the mask only changes on the context columns that became real since the last call
        if real > buf["real"]:
            buf["mask"][:, :, :, buf["real"]:real].fill_(0)

        # if real went back, the mask is rebuilt
        if real < buf["real"]:
            buf["mask"].copy_(self.c._attention_mask(real, cap))
        buf["real"] = real

        self.graphs[cap].replay()
        return self.outs[cap][:, :depth, :]

    def propose_logits(self, context_ids, depth, target_hidden=None, **kwargs):
        return self(context_ids, target_hidden, depth)

    def propose_logits_conditioned(self, context_ids, path_tokens, target_hidden=None, **kwargs):
        return self(context_ids, target_hidden, self.block_size - 1, fill_tokens=path_tokens)

    def propose(self, context_ids, k, target_hidden=None, **kwargs):
        return self(context_ids, target_hidden, k).squeeze(0).argmax(dim=-1)

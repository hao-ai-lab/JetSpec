# Tail Tree Attention

This directory keeps two roles separate:

1. `reference_ops.py`
   A pure PyTorch semantic reference for tree-masked decode attention.
2. `reference_tests.py`
   A correctness harness that compares the reference against `F.scaled_dot_product_attention`
   and against the optimized SM90 kernel through a bridge function.

## Reference Semantics

Tree decode attention here means:

- every query node attends to the full prefix `[0, prefix_len)`;
- every query node attends to tree KV position `prefix_len + j` only when `ancestor[i, j] == 1`;
- GQA maps query head `h` to KV head `h * H_KV // H`.

The reference files materialize the logic in the simplest way possible and are meant to be the
ground-truth definition of behavior, not a fast path.

## Optimized Path

The optimized implementation lives in:

- `optimus_cutedsl.flash_attn.flash_fwd_sm90_paged_tree`
- exported entry: `optimus_cutedsl.flash_attn_varlen_tree_paged_sm90`

That kernel is intentionally designed around the real decode layout:

- paged KV cache, not contiguous KV tensors;
- a compact tree mask `(N, N)` or `(B, N, N)`, not a full `(N, prefix_len + N)` mask;
- a single SM90 paged attention kernel, not an extra cleanup kernel;
- dense prefix and dense causal hot paths kept intact;
- only the last 1 to 2 tail page blocks go through the tree-mask specialization;
- `pack_gqa=True` is supported.

## Why The Bridge Is Not A Hot Path

`reference_tests.py` still exposes:

```python
cuda_tree_attention(query, key, value, ancestor, prefix_len, sm_scale)
```

with contiguous `key` and `value` tensors. To call the real paged kernel, the bridge repacks KV
into paged layout and synthesizes a sequential `page_table`.

That repack is acceptable for correctness checks, but it is not the intended deployment path.
A production decode stack should already own:

- `k_paged`
- `v_paged`
- `page_table`
- `context_lens`
- `cu_seqlens_q`

and pass them directly into `flash_attn_varlen_tree_paged_sm90`.

## Current SM90 Optimization Notes

The current optimized kernel focuses on the tree tail only:

- it keeps the original dense paged causal main loop untouched;
- it stages the `(128, 128)` tail tree-mask tile into shared memory as `uint8`;
- it avoids materializing a giant dense mask for the prefix region;
- it preserves the original one-kernel execution model.

The shared tile is a latency optimization for the last masked page blocks only. It is not meant to
change the semantics of the dense prefix path.

## Suggested Usage

For correctness:

```bash
PYTHONPATH=/data/tree_attention/optimus_jit/src \
pytest /data/tree_attention/optimus_jit/src/optimus_cutedsl/flash_attn/tail_tree_attn/reference_tests.py -q -s
```

For performance work, use the paged SM90 entry directly and avoid the contiguous-KV bridge in
`reference_tests.py`.

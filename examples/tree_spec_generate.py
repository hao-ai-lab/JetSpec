"""Trained-head tree-speculative decode on the owned `nano_vllm` engine.

Shows the contribution end to end via the public API: a trained DFlash draft
head proposes multi-token, tree-structured drafts, and the `NanoEngine`'s
compiled tree-attention path verifies the whole tree in one batched forward,
accepting the longest target-greedy-agreeing root-to-leaf path. Lossless by
construction (see README "Results"), faster than autoregressive decode.

Usage:

    python examples/tree_spec_generate.py [model] [draft_head]

Needs a CUDA GPU and a trained draft head. Defaults to Qwen3-8B with the
published head `Snyhlxde/ptd-qwen3-8b-distill-epoch6-3e-4-no-gamma`.
"""
import sys

from ptd import load_draft_head, DraftHeadTreeDrafter
from ptd.nano_vllm import NanoEngine, SamplingParams

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
DRAFT_HEAD = sys.argv[2] if len(sys.argv) > 2 else "Snyhlxde/ptd-qwen3-8b-distill-epoch6-3e-4-no-gamma"


def main():
    # The compiled tree-spec path (the contribution). The plain "triton_paged_tree"
    # backend runs the same tree-verify without torch.compile.
    engine = NanoEngine(MODEL, attn_backend="triton_paged_tree_compiled")
    head = load_draft_head(DRAFT_HEAD)
    drafter = DraftHeadTreeDrafter(
        head, target=engine.model, block_size=head.block_size,
        target_layer_ids=head.target_layer_ids, draft_shift=False,
    )

    # The head was trained on the chat-formatted distribution; template the prompt
    # so the hidden-state conditioning stays on-distribution (enable_thinking=False).
    prompt = engine.tokenizer.apply_chat_template(
        [{"role": "user", "content": "The three primary colors are"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)

    out = engine.generate_tree(
        prompt, drafter, block_size=head.block_size, tree_width=7, budget=63,
        algo="crossproduct", target_layer_ids=head.target_layer_ids,
        sampling_params=SamplingParams(0.0, 64),
    )
    print(out["text"])
    print(f"\ntokens-per-forward: {out['tpf']:.2f}")


if __name__ == "__main__":
    main()

"""J-pass self-refinement on top of DraftHeadTreeDrafter.

Every real round of drafting — chain or tree — calls `propose_logits` exactly
once (`dflash_generate`'s chain accept loop, and `LLM.generate_tree`'s single
`tree_drafter.propose_logits(...)` per round; `propose_logits_conditioned` is
only used by the separate CUDA-graph engine, not this reference path). Both
callers treat the return as one opaque `(1, depth, V)` tensor, so refining it
before return needs no change anywhere else: chain argmaxes it as usual, and
every registered tree-building algorithm builds its tree from it as usual.

J=1 is the existing behaviour exactly: one head forward, unfilled block
positions filled with MASK (`_DraftHeadForward.block_output_ids`, the
`fill_tokens=None` case). For J>1, each further pass re-runs the head over
the SAME context, feeding the previous pass's own argmax back as this pass's
`fill_tokens` instead of MASK — the block's own next Jacobi fixed-point
iterate. `_forward_head`'s cache crops back to the context length after every
call, so repeated same-position calls only recompute the block region; the
context K/V is not reprojected on passes 2..J.

Caveat this measures rather than hides: `mem-b2a1` and every other artifact
in this repo were trained with `sample_clean_prefix_lengths` drawing UNCLEAN
slots from independent random tokens sampled earlier in the sequence (real
vocabulary, uniformly by position) — never from the drafter's own predictions,
and never from MASK. J=1 (MASK-filled) and J>=2 (self-prediction-filled) are
therefore both somewhat out-of-distribution for how this checkpoint was
trained, in two different directions. Treat J>1 results as measuring whether
self-refinement transfers despite that gap, not as a mode this checkpoint was
tuned for.
"""
import torch

from jetspec.draft_head_adapter import DraftHeadTreeDrafter


class SelfRefiningDraftHeadTreeDrafter(DraftHeadTreeDrafter):
    def __init__(self, head, target, block_size: int, target_layer_ids,
                draft_shift: bool = False, j: int = 1):
        super().__init__(head, target, block_size, target_layer_ids, draft_shift=draft_shift)
        if isinstance(j, bool) or not isinstance(j, int) or j < 1:
            raise ValueError(f"self-refine j must be a positive integer, got {j!r}")
        self.j = j

    @torch.inference_mode()
    def propose_logits(self, context_ids: torch.Tensor, depth: int,
                       target_hidden: torch.Tensor = None, **kwargs) -> torch.Tensor:
        fill_tokens = None
        logits = None
        for _ in range(self.j):
            logits = self._fwd._forward_head(context_ids, target_hidden, depth, fill_tokens=fill_tokens)
            fill_tokens = logits.squeeze(0).argmax(dim=-1)  # this pass's guess -> next pass's input
        return logits

"""The single place the engine touches the model forward.

Today: a thin pass-through to the HF target with an explicit `position_ids` and
an HF `DynamicCache`. The draft-head forward and the tree-mask path (a 4D additive
ancestor mask passed to the same HF forward) build on this; a dedicated
tree-attention kernel later swaps out the mask path.
"""
import torch


class ModelRunner:
    def __init__(self, model):
        self.model = model

    @torch.inference_mode()
    def forward(self, input_ids, past_key_values, position_ids=None):
        """One target forward step. Returns (logits, past_key_values).

        `position_ids` is passed explicitly (not inferred) so the offline loop's
        positions are deterministic and match HF greedy generation exactly.
        """
        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return out.logits, out.past_key_values

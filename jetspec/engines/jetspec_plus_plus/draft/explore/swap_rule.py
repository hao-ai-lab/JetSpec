"""Two-expert swap rule, selected by method.explore.swap_rule."""
from types import SimpleNamespace
import torch
from jetspec.engines.jetspec_plus_plus.draft.method.refresh import rule_f
from jetspec.engines.jetspec_plus_plus.draft.explore import register

SLOT_BUCKETS = ('1', '2-3', '4-7', '8-15')
KHAT_BUCKETS = ('0-2', '3-7', '8-15')
RULE_TINY = float(torch.finfo(torch.float32).tiny)

from jetspec.engines.jetspec_plus_plus.draft.method.ops import reference_op

@reference_op("policy_swap_decision")
def policy_swap_decision(logits, draft_logits, ids, khat, slot_counts, active, params,
                         pin_slot=None, bias_table=None, pin_mask=None):
    """Fixed-shape two-expert rule from tau_lab 6908a729, without host transfers.

    Draft input is full logits or retained (top ids, fp32 probabilities). Margin
    adds to the slot/k-hat bias; scan searches later read slots after a refusal.
    A pin on the initial slot blocks the row, as in the lab refresh hook.
    """
    rows, width = ids.shape
    n = min(params['n'], logits.shape[-1])
    decoder_p = logits.float().softmax(-1).reshape(-1, logits.shape[-1])
    if isinstance(draft_logits, tuple):
        draft_top = tuple(value[:, 1:].reshape(-1, n) for value in draft_logits)
    else:
        ranked = draft_logits[:, 1:].float().softmax(-1).topk(n, -1)
        draft_top = ranked.indices.reshape(-1, n), ranked.values.reshape(-1, n)
    drafts = ids[:, 1:].reshape(-1)
    slots = torch.arange(1, width, device=ids.device)
    slot_bucket = (slots >= 2).long() + (slots >= 4).long() + (slots >= 8).long()
    khat_bucket = (khat >= 3).long() + (khat >= 8).long()
    table = bias_table
    if table is None:
        table = torch.tensor([[params['bias'][s][k] for k in KHAT_BUCKETS] for s in SLOT_BUCKETS],
                             dtype=torch.float64, device=ids.device)
    bias = table[slot_bucket[None, :], khat_bucket[:, None]] + params['margin']
    choice = _rule_choice(SimpleNamespace(**dict(params, n=n)), decoder_p, draft_top,
                          drafts, bias.reshape(-1), exclude_draft=True)
    new = choice.x
    used = torch.minimum(khat, rule_f(logits.argmax(-1), ids, slot_counts))
    target = (used + 1).clamp(max=width - 1)
    hit = choice.ok.reshape(rows, width - 1)
    hit &= (slots[None, :] <= slot_counts[:, None]) & (slots[None, :] > used[:, None])
    if not params['scan']:
        hit &= slots[None, :] == target[:, None]
    if pin_slot is not None:
        hit &= slots[None, :] != pin_slot[:, None]
    if pin_mask is not None:
        hit &= ~pin_mask[:, 1:]
    found = hit.any(-1)
    where = torch.where(found, hit.long().argmax(-1) + 1, target)
    selected = new.reshape(rows, width - 1).gather(1, (where - 1)[:, None])[:, 0]
    apply = found & active & (used + 1 < width)
    if pin_slot is not None:
        apply &= pin_slot != target
    if pin_mask is not None:
        apply &= ~pin_mask.gather(1, target[:, None])[:, 0]
    return used, where, selected, apply








def _refresh_swap_state():
    from .refresh import RefreshSwap
    return RefreshSwap

METHOD_KEYS = ("explore.swap_rule",)
register(METHOD_KEYS[0], refresh_swap_state=_refresh_swap_state)

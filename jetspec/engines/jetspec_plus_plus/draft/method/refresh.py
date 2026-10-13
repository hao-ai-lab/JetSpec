"""Refresh cutoffs, agreement, swap rules and visibility math."""
from types import SimpleNamespace

import torch

from jetspec.engines.jetspec_plus_plus.draft.method.ops import reference_op

SLOT_BUCKETS = ('1', '2-3', '4-7', '8-15')
KHAT_BUCKETS = ('0-2', '3-7', '8-15')
RULE_TINY = float(torch.finfo(torch.float32).tiny)


@reference_op("refresh_cutoff")
def refresh_cutoff(probabilities, threshold, *, start=1, final=None):
    """First slot below the cumulative probability threshold; equality continues."""
    chained = torch.cumprod(probabilities.double(), dim=1)
    below = chained < float(threshold)
    first = below.to(torch.int8).argmax(dim=1) + start
    final = probabilities.shape[1] + start - 1 if final is None else final
    return torch.where(below.any(dim=1), first, torch.full_like(first, final))


def refresh_khat_threshold(again=False, *, environ):
    if again and environ.get('JETSPEC_REFRESH_KHAT_T2') is not None:
        value = float(environ['JETSPEC_REFRESH_KHAT_T2'])
        if not 0.0 < value < 1.0:
            raise ValueError('JETSPEC_REFRESH_KHAT_T2 must be in (0, 1)')
        return value
    value = float(environ.get('JETSPEC_REFRESH_KHAT_T', '0.1'))
    if not 0.0 < value < 1.0:
        raise ValueError('JETSPEC_REFRESH_KHAT_T must be in (0, 1)')
    return value


def refresh_khat_cap(again=False, *, environ):
    if not again or environ.get('JETSPEC_REFRESH_KHAT_CAP2') is None:
        return None
    cap = int(environ['JETSPEC_REFRESH_KHAT_CAP2'])
    if not 0 <= cap <= 15:
        raise ValueError('JETSPEC_REFRESH_KHAT_CAP2 must be in 0..15')
    return cap


@reference_op("rule_f")
def rule_f(decoder_argmax, ids, slot_counts):
    """Spec rule F: ``agree[r, j] = decoder_argmax[r, j-1] == ids[r, j]`` for j = 1..n_r-1
    (n_r = slot_counts[r], the decoder rows computed); F[r] = the number of leading agreements,
    0..n_r-1.  ``decoder_argmax`` [R, >= W-1] holds the argmax of block slot j's decoder row (the
    prediction for slot j + 1).  (tau_lab.rule_f, 464e69ce.)"""
    width = ids.shape[1]
    slot_counts = torch.as_tensor(slot_counts, dtype=torch.long, device=ids.device).reshape(-1, 1)
    agree = decoder_argmax[:, :width - 1].to(ids.device) == ids[:, 1:]
    valid = torch.arange(1, width, device=ids.device)[None, :] < slot_counts
    return torch.cumprod((agree & valid).long(), dim=1).sum(dim=1)


@reference_op("margin_swap")
def margin_swap(logits, current, margin, *, exclude_current):
    """Per row: the decoder's top-1 (``exclude_current``: the best token other than
    ``current``) and whether p(top-1) - p(current) >= margin under the row's softmax.
    (tau_lab.margin_swap, 464e69ce.)"""
    probabilities = torch.softmax(logits.float(), dim=-1)
    scores = logits.float().clone()
    if exclude_current:
        scores.scatter_(1, current[:, None], float('-inf'))
    new = scores.argmax(dim=-1)
    gain = probabilities.gather(1, new[:, None])[:, 0] - probabilities.gather(1, current[:, None])[:, 0]
    return new, (gain >= margin) & (new != current)


@reference_op("swap_decision")
def swap_decision(logits, ids, khat, slot_counts, active, pin_slot=None):
    """The cutoff and the swap from the refresh-context decoder logits, on device.

    ``logits`` [R, W-1, V]: decoder row j of row r (the distribution for block slot j + 1), valid
    for j < slot_counts[r]; ``ids`` [R, W] the block (the next pass's input); ``khat`` [R] the
    draft cutoff.  Returns ``(khat_used, slot, new, apply)``: khat_used = min(khat, F); slot =
    khat_used + 1 (clamped to W - 1); ``apply`` marks the rows whose ``ids[r, slot]`` becomes
    ``new[r]``.  A row with a pinned slot (``pin_slot`` >= 0) is not swapped again.
    """
    rows, width = ids.shape
    f = rule_f(logits.argmax(dim=-1), ids, slot_counts)
    khat_used = torch.minimum(khat, f.to(khat.device))
    target = khat_used.to(ids.device) + 1
    slot = target.clamp(max=width - 1)
    eligible = active & (target <= width - 1)
    if pin_slot is not None:
        eligible = eligible & (pin_slot < 0)
    # decoder row khat_used is block slot khat_used's output: the distribution for slot
    # khat_used + 1; it was computed since khat_used <= min(khat, W - 2)
    row_logits = logits[torch.arange(rows, device=logits.device),
                        (slot - 1).clamp(0, logits.shape[1] - 1).to(logits.device)]
    current = ids.gather(1, slot[:, None])[:, 0]
    new, apply = margin_swap(row_logits, current.to(row_logits.device), 0.0, exclude_current=True)
    return khat_used, slot, new.to(ids.device), apply.to(ids.device) & eligible


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




def _rule_read(tokens, ids, probabilities, floor):
    """One expert's probability of each of ``tokens`` [M, K] (float64): its top-n value (``ids`` /
    ``probabilities`` [M, n]) where the token is in its top-n, else its ``floor`` [M]."""
    match = tokens[..., None] == ids[:, None, :]
    return torch.where(match.any(dim=-1), probabilities.double().gather(1, match.long().argmax(dim=-1)),
                       floor[:, None])


def _rule_choice(rule, decoder_p, draft_top, drafts, bias, *, exclude_draft):
    """The rule at M slots, on ``decoder_p``'s device: ``decoder_p`` [M, V] is the decoder's fp32 softmax there,
    ``draft_top`` the drafter's top-n (ids, fp32 probabilities) [M, n] (``_ranked``), ``drafts`` [M] the block's
    tokens d and ``bias`` [M] the bias of each slot's cell.  ``exclude_draft`` (the refresh swap): x* is the best
    candidate other than d.  Returns [M] tensors: ``x`` (x*), ``ok`` (x* exists, differs from d and its gain over d
    is at least the bias; pins aside), ``score_x`` / ``score_d`` and the probabilities those scores read
    (``p_s_x``, ``p_s_d``, ``p_d_x``, ``p_d_d``: a token outside an expert's top-n reads its floor)."""
    device = decoder_p.device
    decoder_ids, decoder_top = _ranked(decoder_p, rule.n)
    draft_ids, draft_p = (tensor.to(device) for tensor in draft_top)
    drafts, bias = drafts.to(device), bias.to(device)
    floors = rule.floor_factor * decoder_top[:, -1].double(), rule.floor_factor * draft_p[:, -1].double()


    candidates = torch.cat((decoder_ids, draft_ids), dim=1)
    valid = candidates != drafts[:, None] if exclude_draft else torch.ones_like(candidates, dtype=torch.bool)
    score = experts(candidates)[2].masked_fill(~valid, float('-inf'))
    tied = valid & (score == score.amax(dim=1, keepdim=True))
    # ties: the decoder's argmax, then the lower token id
    argmax = decoder_p.argmax(dim=-1)
    lowest = torch.where(tied, candidates, torch.iinfo(torch.long).max).amin(dim=1)
    x = torch.where((tied & (candidates == argmax[:, None])).any(dim=1), argmax, lowest)
    (p_s_x, p_d_x, score_x), (p_s_d, p_d_d, score_d) = (
        tuple(value[:, 0] for value in experts(token[:, None])) for token in (x, drafts))
    if rule.margin_space == 'log':
        gain = score_x - score_d
    else:
        gain = p_s_x * p_d_x.pow(rule.alpha) - p_s_d * p_d_d.pow(rule.alpha)
    return SimpleNamespace(x=x, ok=tied.any(dim=1) & (x != drafts) & (gain >= bias), score_x=score_x,
                           score_d=score_d, p_s_x=p_s_x, p_s_d=p_s_d, p_d_x=p_d_x, p_d_d=p_d_d)


@reference_op("refresh_visibility")
def inject_mask_reference(mask, khat, start, rows, reset):
    row_bytes = (start + rows) // 8
    index = torch.arange(rows * row_bytes, device=mask.device)
    query = index // row_bytes
    entry = (index % row_bytes) * 8 - start
    region = (entry >= 0) & (entry < rows)
    own = region & (entry // 16 == query // 16)
    limit = torch.minimum(khat.clamp(0, 15)[query // 16], query % 16 - 1)
    count = (limit - entry % 16 + 1).clamp(0, 8)
    packed = torch.where(own, (1 << count) - 1, 0).to(torch.uint8)
    target = mask[:index.numel()]
    prior = torch.where(region, 0, target).to(torch.uint8) if reset else target
    target.copy_(prior | packed)

"""E04_LAB_*: default-off tau-research knobs on the eager served rows path.

Offline tau research, not a serving configuration: each knob changes how the adapter refresh
and the anchor decoder are used, so one idea is one served read.  Speed is not a goal; every
function here is an eager reference implementation.  With every E04_LAB_* name unset nothing
in this module runs except ``lab_active()`` (an environment read) at its call sites.

Knobs (registered in ``env.py``):

* ``E04_LAB_REFRESH_CUTOFF`` = ``draft`` (default) | ``decoder_f`` | ``oracle``: the per-row
  refresh cutoff ``khat`` that ``inject_refresh`` uses.  ``draft`` keeps the draft-model rule
  (``refresh_khat_anchor0``); ``decoder_f`` uses min(khat_draft, F), F the decoder's first
  disagreement in the refresh context (``rule_f``); ``oracle`` uses min(khat_draft, true k),
  true k the leading drafts equal to a reference continuation (``oracle_true_k``).
* ``E04_LAB_SWAP`` = ``off`` (default) or a ``+``-joined set of ``slot1_pre_j0``,
  ``first_wrong_refresh`` and ``pre_verify``, with ``E04_LAB_SWAP_MARGIN`` (float, default 0.0):
  each mode replaces one draft token per row (per occasion) by the decoder's top-1 under a
  margin gate (``slot1_pre_j0``, ``refresh_hook``, ``pre_verify``).  A token swapped before or
  during the passes is pinned for the rest of the round; pins accumulate across modes and no
  mode ever re-swaps a pinned slot.
* ``E04_LAB_SWAP_MARGIN_SLOT1`` / ``_REFRESH`` / ``_PREVERIFY`` (floats; unset: the mode reads
  E04_LAB_SWAP_MARGIN): each, when set, is the margin of slot1_pre_j0 / first_wrong_refresh /
  pre_verify alone (``swap_margin(mode)``, ``MODE_MARGINS``) and needs its mode in E04_LAB_SWAP.
  SHADOW_AR reads no margin.  The record's ``swap_margins`` holds each active mode's margin.
* ``E04_LAB_PREVERIFY_ROUNDS=n`` (1..15, default 1: one swap): the pre_verify swap repeated up
  to n rounds per row, pipelined and exact rounds alike.  Each round after the first runs on the
  rows that swapped in the round before, batched: the adapter over each such row's edited block
  (one full call), the refresh-context decoder, and the first unpinned qualifying slot is swapped
  and pinned; a row stops when no slot qualifies (``pre_verify``).  Needs pre_verify in
  E04_LAB_SWAP.
* ``E04_LAB_REF_TOKENS``: the reference output tokens the ``oracle`` cutoff reads.
* ``E04_LAB_PASS_LABEL`` = unset | ``J2``: every pass after J0 reads the J2 thresholds.
* ``E04_LAB_REFRESH_EVERY=1``: the adapter refresh (with the cutoff and swap hooks) runs before
  every pass after J0, not only at ``JETSPEC_ADAPTER_REFRESH``'s passes (``refresh_schedule``).
* ``E04_LAB_EXTRA_PASSES=n`` (0..8): n passes after the recipe's, each preceded by a refresh with
  the hooks; their threshold label is the recipe's last pass label (``PackedBlocks``
  ``lab_label_passes``).
* ``E04_LAB_SHADOW_AR=n`` (0..15): before J0, slots 1..n are decoded greedily by the adapter +
  decoder ("shadow") one slot at a time, pinned and put into the J0 input (``shadow_ar_decode``).
* ``E04_LAB_EXACT=1``: the knobs above act in the exact anchor0 round too
  (``_exact_anchor0_rows``: the first round after prefill and every miss redraft; every round under
  JETSPEC_MISS_REDRAFT_FORCE=1).  Its adapter is the round-start staging call over the exact
  committed prefix with no tail (``exact_adapter_rows``), and the decoder reads no tail (zero tail
  slots).  The refresh runs before the passes ``refresh_schedule`` gives for the round's own pass
  count and injects entries 0..khat_used eagerly (``_exact_refresh``); the round's record is a
  W280_RECORD line keyed like a proposal record, its ``e04_lab`` carrying ``exact=1``.
* ``E04_LAB_SWAP_KHAT_MAX=n`` / ``E04_LAB_SWAP_SLOT_MAX=n`` (1..15; unset: no gate): a
  first_wrong_refresh swap applies only on a row whose draft k-hat at that refresh is < n / only
  at a slot <= n; every refresh with that swap records the swaps each gate held back
  (``gated_khat``, ``gated_slot``).  Pipelined and exact rounds alike.
* ``E04_LAB_ROWS_MIN=n`` (1..16; unset: no floor): after the joint row rule (E04_ROWRULE=joint),
  when fewer than n pairs passed tau, the top n pairs by joint score s are published, ranked as
  the rule ranks (-s, k, pool rank; s > 0), up to JETSPEC_ROWS_CAP rows (``joint_rows``).  The
  rule's own rows stay first and unchanged; the floor appends the next pairs of its ranking.
* ``E04_LAB_SWAP_ROWS=1``: after the rule (and the floor), each decoder-written slot s (1..W) of
  the verified block (``Block.e04_lab_written``, below) adds the pair (s - 1, the decoder's top
  offer at row s - 1: its first post-exclusion offer) unless it is already published, in slot
  order, up to JETSPEC_ROWS_CAP rows (``joint_rows``).  Not after a miss (the rule scores k = 0
  alone and the verified block is not the served one).  Needs E04_ROWRULE=joint.
* ``E04_LAB_KSCORE_WRITTEN`` = unset | ``disagree`` | ``draft_p`` | ``both``: the BLEND k score of
  the verified block (``live_rows.e04_kscore_blend`` and ``e04_kscore_blend_device``) at its
  decoder-written slots i: ``disagree`` sets the argmax indicator term to 0 (E04_KSCORE=blend_dec),
  ``draft_p`` replaces p_top1[i] by the draft model's probability of the written token d_i at the
  final pass of the block's row (``Block.e04_lab_written_p``), ``both`` does both (``written_terms``,
  ``written_drafter``, ``written_argmax``).  Unwritten slots and the coefficients are unchanged;
  after a miss the score is not run.
  E04_LAB_SWAP_ROWS and E04_LAB_KSCORE_WRITTEN need a source of decoder-written slots
  (E04_LAB_SWAP or E04_LAB_SHADOW_AR).
* ``E04_LAB_SEED_EXACT=1`` (needs JETSPEC_MISS_REDRAFT=1, JETSPEC_MISS_REDRAFT_FORCE=1 and
  E04_LAB_EXACT=1): a forced exact round whose round-ahead rows hit starts from the hit row's final
  block (its published tokens: after its pipelined passes, swaps and pre_verify rounds) instead of the
  J0 initialization (the context draws, SHADOW_AR and slot1_pre_j0).  The hit is ``select_live_block``'s
  decision on the draft rank, which published the rows and learns the verdict: it hands the exact round
  that decision and the row it selected (``round_ahead``), so the ranks exchange nothing new
  (``round_ahead_seed``).  The correction stays pinned in slot 0 and the row's lab pins are dropped, so
  the exact S may revise any slot >= 1.  A round-ahead miss and the admission round draft from J0 as
  before.
* ``E04_LAB_SEED_PASSES=m`` (0..4, default 0; needs E04_LAB_SEED_EXACT=1): the drafter passes a seeded
  round runs from the seed on the exact context before pre_verify, refreshed on ``refresh_schedule``'s
  schedule for m passes.  m = 0 runs none: pre_verify (with E04_LAB_PREVERIFY_ROUNDS) on the exact
  adapter alone, and the block keeps the seed row's own per-slot drafter outputs (``zero_pass_blocks``);
  m = 0 refuses E04_LAB_EXTRA_PASSES and E04_LAB_KSCORE_WRITTEN=draft_p / both, which need a pass.  A
  seeded block's passes run is the seed row's plus the round's, its threshold label capped at the
  round's recipe count (JETSPEC_MISS_REDRAFT_PASSES, as E04_LAB_EXTRA_PASSES caps it).  Unseeded exact
  rounds keep JETSPEC_MISS_REDRAFT_PASSES.
* ``E04_LAB_RECORD_TOPK=n`` (1..32; unset: off): records only.  Every refresh record gains both experts'
  top-n at each slot of the block it read and the rows' keys (``refresh_hook``), and the pre_verify record
  each read's decoder top-n and the final pass's drafter top-n (``pre_verify``); see Records.  Served
  tokens, rows, keys, pins, swaps and every other record field are the same with it set or unset.  It
  records at the adapter refresh, so it needs JETSPEC_ADAPTER_REFRESH.
* ``E04_LAB_SWAP_RULE_FILE=<path>`` (unset: the swaps above): a two-expert rule decides the first_wrong_refresh
  swap and, while pre_verify is in E04_LAB_SWAP, the pre_verify swap (slot1_pre_j0 is unchanged).  The file is a
  JSON object, read once per path and validated at load (``SwapRule``: an unknown or repeated key, a value of the
  wrong type or range and an incomplete bias table refuse); every key is optional, with these defaults:
  ``n`` 8 (1..32), ``floor_factor`` 0.5 (in (0, 1]), ``alpha`` 0.0 (finite, >= 0), ``margin_space`` "prob" or
  "log", ``pin`` 1 or 0 and ``bias`` {slot bucket "1", "2-3", "4-7", "8-15": {k-hat bucket "0-2", "3-7", "8-15":
  number}}, 0.0 in every cell.  At slot i of a row (``_rule_choice``), the candidates are the union of the
  decoder's and the drafter's top-n there.  p_S is the decoder's fp32 softmax (the read's row i - 1) and p_D the
  drafter's at T = 1, from the logits of the pass whose block the refresh reads (at pre_verify, each row's final
  pass); a token outside one expert's top-n reads that expert's floor, floor_factor x its n-th probability.  The
  score is log p_S(x) + alpha log p_D(x) (float64, a probability of 0 reading 2^-126 in the log), x* the
  candidate with the highest score (ties: the decoder's argmax, then the lower token id), and the gain of x* over
  the block's d_i is score(x*) - score(d_i) in "log" space and p_S(x*) p_D(x*)^alpha - p_S(d_i) p_D(d_i)^alpha in
  "prob" space.  At alpha 0 the prob gain is today's margin p_S(x*) - p_S(d_i); for alpha > 0 it is the gap
  between the two tokens' unnormalized product-of-experts weights, which shrink as alpha grows (p_D <= 1), so a
  prob-space bias belongs to its alpha.  The two spaces pick the same x* and differ only in that test (at bias 0
  both accept x* when its weight is at least d_i's).  bias(i, k) is the cell of slot i and of the row's draft
  k-hat k.
  - first_wrong_refresh (``_rule_refresh``): x* is the best candidate other than d_i, as today's swap token is
    (``margin_swap`` excludes the draft; the two readings differ only when d_i ties for the best score or under
    a negative bias).  The swap slot F is the first slot from the slot after the cutoff, khat_used + 1, through
    the read's last, min(khat_draft, 14) + 1, that is unpinned and whose gain >= bias(F, khat_draft); a row
    whose slot khat_used + 1 is pinned is not swapped, as today.  It needs E04_LAB_REFRESH_CUTOFF=decoder_f:
    slot khat_used + 1 is then the decoder's first disagreement (or the read's last slot), every unpinned slot
    before it holds both experts' argmax (the pass's own, which the decoder agrees with), and F lies past the
    inject, whose entries are 0..khat_used (F = khat_used + 1 at alpha 0 and bias 0; later when the drafter's
    term or a bias holds that slot back).  E04_LAB_SWAP_KHAT_MAX / _SLOT_MAX hold back the rule's swap at F as
    they hold back today's.
  - pre_verify: x* ranges over every candidate, and F is the first unpinned slot i >= 1 with x* != d_i and
    gain >= bias(i, k), k the drafter's own k-hat over the final block (``live_rows.refresh_khat_anchor0`` on the
    final pass's top-1 probabilities at JETSPEC_REFRESH_KHAT_T), as today's pre_verify scans.
  - pin 1 writes x* at F and pins it, as today.  pin 0 (the first_wrong_refresh swap; pre_verify's swaps stay
    pinned, no pass follows them) writes x* into the next pass's input unpinned: that pass's query at F reads
    x* there, the pass input at slots 0..F - 1 and the injected entries 0..khat_used (F itself is past the
    inject), and its own argmax at F becomes slot F's token, so the drafter may rewrite x*.  The slot is not
    decoder-written, and a later refresh may swap it again.
  With every default the swaps are today's first_wrong_refresh and pre_verify at margin 0, token for token.
  The rule reads the refresh's decoder read and the logits that read already sees (no adapter call, no decoder
  read of its own).  It needs first_wrong_refresh or pre_verify in E04_LAB_SWAP and JETSPEC_DRAFTS=15, and
  refuses E04_LAB_SWAP_MARGIN_REFRESH / _PREVERIFY (its bias is their margin) and, with pre_verify,
  E04_LAB_SEED_PASSES=0 (no pass, no drafter distribution).
* ``E04_LAB_AFTER_SWAP_FILL`` = ``keep`` (default) | ``mask`` | ``noise`` | ``context`` (tl10): at a
  first_wrong_refresh swap that writes slot s of a row, the next pass's input at that row's unpinned slots s + 1..W
  is the drafter's mask token, token ids uniform over the drafter's vocabulary less the mask token (one draw per
  block slot from a CPU generator seeded by ``E04_LAB_FILL_SEED`` (default 0; needs ``noise``), the request, the
  round, the exact flag, the pass and the row: ``_fill_generator``), or the round's J0 random-context draws there
  (``context``: the drafter trainer's default Jacobi-forcing noise source).  The slots stay unpinned: that pass's
  argmax replaces them (``_after_swap_fill``).  The Jacobi mask-free refusal is lifted for those slots of that pass
  only (``mask_outside_fill``: the round's guard reads the hook's output, the filled slots exempt); every other slot
  and pass, and every other path, keeps it.  Needs first_wrong_refresh in E04_LAB_SWAP.  No adapter call or decoder
  read is added: the refresh's inject (entries 0..khat_used) is computed before the fill and never reaches it.
  ``decoder`` (tl11): each unpinned slot j > s that the refresh's decoder read covers (j <= min(khat_draft, W - 2)
  + 1) takes the read's argmax for slot j (decoder row j - 1, the dense argmax ``rule_f`` reads); slots past the
  read keep their tokens.  The written slots stay unpinned and are not decoder-written slots (``lab.written``), so
  the next pass may change any of them; at every refresh that swaps (R1 and R2 under E04_LAB_EXTRA_PASSES=1).
* ``E04_LAB_PIN_AGREED=1`` (tl10): the first_wrong_refresh refresh under decoder_f also pins slots 1..khat_used of
  every row (the run J0 and the decoder agreed on; a row with no disagreement in its window pins all of it) to the
  block the refresh read, slots already pinned keeping their pins (``_pin_agreed``).  These pins are kept apart
  (``LabRound.agreed``) and are not decoder-written slots (``LabRound.written_pins``).  Needs first_wrong_refresh
  and E04_LAB_REFRESH_CUTOFF=decoder_f.
* ``E04_LAB_TAIL_PASSES=t`` (0..2, tl10): t drafter-only passes after the recipe's last (and after
  E04_LAB_EXTRA_PASSES'), no refresh and no decoder read before them, the lab pins applied as at every pass, their
  threshold label the recipe's last pass label (``refresh_schedule``); a round whose rows all converged stops as
  today.  The recipe knobs do not express this: serve_read's ``--sweep-j 2`` runs J0 R1 J1 J2 but its 3-pass
  blocks read the J2 thresholds.

Decoder-written slots: a block's lab pins once its round ends (slot1_pre_j0, SHADOW_AR,
first_wrong_refresh and pre_verify).  ``packed_refinement`` reads the round's pins after its
final edit and every published block carries its row's mask (``PackedBlocks.lab_written`` ->
``Block.e04_lab_written``, a bool per block slot); under draft_p / both each row's final-pass
logits are kept and the block also carries p(d_i) at its written slots (``e04_lab_written_p``,
fp32 softmax, NaN elsewhere).  The draft rank selects the served row (``select_live_block``)
and stages it as the next round's pending block, so the next round's k score and rows read the
mask without any protocol change.  Pipelined and exact rounds alike.

Records (W280_RECORD, the per-round ``e04_lab``; every field below is written whenever any
E04_LAB_* name is set):
* ``rows_min``, ``swap_rows``, ``kscore_written``, ``seed_exact``, ``seed_passes``: the knobs above;
* ``rows`` (E04_ROWRULE=joint rounds): ``tau`` and ``temperature``, ``rule_mass`` (the P(k) the
  rule read for each scored row k = 0..W: the BLEND mass, after E04_LAB_KSCORE_WRITTEN; [1.0]
  for k = 0 alone after a miss), ``pair_scores`` (the joint score of every published pair, in
  ``published_keys`` order), ``rows_min`` and ``floor_added``, and ``swap_rows`` (the pairs
  E04_LAB_SWAP_ROWS added); the scores are the rule's arithmetic on the host copy of the offers;
* ``verified``: the verified block's decoder-written slots (``written``) and, when it carries
  them, the draft model's p of each written token (``written_p``); absent for a block drafted
  without the lab (the admission rows, a miss redraft without E04_LAB_EXACT);
* per swap (every mode and every pre_verify round): ``top_ids`` / ``top_p``, the swap decoder's
  top-5 ids and fp32 probabilities (the full softmax, the current token included) at the
  swapped slot; an in-pass swap (slot1_pre_j0, first_wrong_refresh) also ``next_argmax``: the
  argmax of the pass whose input carries it, at its slot, i.e. the token the pin then replaced.
  That pass read the pinned token, so ``next_argmax`` is not the no-pin counterfactual; that
  token is not available without an extra pass and is not recorded;
* per refresh: ``block_ids``, every drafted row's block [anchor, d1..dW] at that refresh, before
  its swap;
* per exact round (``exact=1``): ``seed`` (``seeded``, the seed row's ``key`` [k, correction] and
  ``block``, ``passes``: the drafter passes the round ran per published row, and ``seed_passes``: the
  seed row's own), ``s_reads`` (the pre_verify reads run: one per pre_verify round executed, each a
  batched adapter + decoder read), ``swaps_by_mode`` (the round's swaps per E04_LAB_SWAP mode) and
  ``final_block`` (each published row's final block); a forced round (a redraft under
  JETSPEC_MISS_REDRAFT_FORCE=1) adds ``forced``: ``round_ahead`` (hit or miss) and, on a hit,
  ``served_key``, the key of the round-ahead row the verdict matched;
* E04_LAB_RECORD_TOPK=n, only while it is set: ``record_topk`` (n).  Each top-n is an fp32 softmax at
  T = 1, ids and probabilities in descending probability, laid out [rows][slots 1..B-1][n] (list index
  j - 1 is block slot j).  Per refresh: ``keys`` (each drafted row's [k, anchor]: row i is
  published_keys[i], [-1, -1] a padding row), ``draft_top_ids`` / ``draft_top_p`` and ``dec_top_ids`` /
  ``dec_top_p``, beside its ``khat_draft`` and ``khat_used`` (the cutoff it used).  The drafter's are over
  the logits of the pass whose block the refresh read (the pass before it: J0 before the first refresh).
  Every unpinned slot of that block is that pass's argmax, so that pass produced the slot's token (a
  frozen row's next input is its own last argmax too); a lab-pinned slot (slot1_pre_j0, SHADOW_AR, an
  earlier refresh swap: ``swaps`` and ``shadow_ar_ids`` name them) holds a decoder-written token no pass
  produced, and its entry is the same pass's distribution there, whose argmax the pin replaced.  The
  decoder's are the refresh's decoder read (``refresh_decoder_logits``: row j - 1 is slot j), which covers
  slots 1..min(khat_draft, B - 2) + 1, later slots null; ``dec_top_*`` are null at a refresh with no
  decoder read (no anchor decoder loaded: the refresh runs it whenever one is, for F).  ``pre_verify``
  adds ``keys``, ``draft_top_ids`` / ``draft_top_p`` (each row's final pass, the pass its tokens froze
  at, i.e. the block pre_verify read; null in a round that ran no pass, E04_LAB_SEED_PASSES=0) and
  ``reads``: per pre_verify read, ``rows`` (the rows it read) and its ``dec_top_ids`` / ``dec_top_p``
  [rows read][slots 1..B-1][n].  Each refresh and each pre_verify read moves its records home in one
  host transfer.
* E04_LAB_SWAP_RULE_FILE, only while it is set: ``swap_rule``, the file's ``sha256`` and its parsed ``params``
  (the defaults filled in), once per round record; each swap the rule made (first_wrong_refresh and
  pre_verify) adds ``rule``: ``x_star``, ``score_x`` / ``score_d`` (the scores of x* and of the replaced token),
  ``p_s_x`` / ``p_s_d`` / ``p_d_x`` / ``p_d_d`` (the probabilities those scores read: a floor outside an
  expert's top-n), the ``bias`` of its cell, the row's ``khat_draft`` (the bucket's k-hat) and ``pinned``.
  An unpinned swap's ``next_argmax`` is the token the pass then wrote at its slot.
* tl10, only while a tl10 knob is off its default: ``after_swap_fill``, ``fill_seed`` (null unless noise),
  ``pin_agreed``, ``tail_passes`` and ``passes_run`` (the passes each published row ran this round); each
  first_wrong_refresh swap under a fill adds ``fill`` (``kind``, ``slots``: the slots filled) and each refresh under
  E04_LAB_PIN_AGREED adds ``agreed_pins`` (each row's newly pinned slots).  The decoder fill's ``fill`` adds
  ``changed`` (tl11): the filled slots whose decoder token differs from the block's.

Under ``E04_LAB_REFRESH_EVERY`` or ``E04_LAB_EXTRA_PASSES`` the round runs its whole pass
budget: a converged row still stops (its tokens freeze), but the round no longer ends when
every row has converged, so a later refresh swap can still reopen a row.

The knobs serve the eager N=1 flat rows path under ``JETSPEC_ROW_LAYOUT=anchor0`` with 15 (or
31) drafts; they refuse graphed paths, ``JETSPEC_DECODER_LOOP=1``,
``JETSPEC_DECODER_PREFIX_MERGE=1`` and ``JETSPEC_DECODER_LMHEAD_KN=1`` (``check_regime``).
They act on the rounds ``_flat_rows_and_passes`` drafts: not on the first round after prefill
(``_exact_anchor0_rows``) nor on a miss redraft (``propose_fallback``) unless E04_LAB_EXACT=1.
A lab read with N > 1 drafts each request through the N=1 path instead of
``_flat_rows_and_passes_batch``.

Call graph with every lab name unset (AGENTS.md trap 6): ``lab_active()`` runs in
``_flat_rows_and_passes``, in the N > 1 routing of ``_propose_flat_bucketed`` and at draft-side
admission; ``exact_lab()`` in ``_exact_anchor0_rows``; ``check_regime()`` in
``KthRowsDrafter.__init__`` and ``AnchorDecoder.load``; each returns after reading the
environment and needs no collaborator; ``heads.kpred.pass_name`` reads E04_LAB_PASS_LABEL.
``packed_refinement``, ``PackedBlocks`` and ``_adapter_refresh`` gain default-None arguments or
attributes only.  tl06 adds no unconditional call: ``written_terms`` and ``joint_rows`` run only
while ``lab_active()``; ``_assemble_rows``, ``e04_kscore_blend`` and ``e04_kscore_blend_device``
gain a default-None ``written`` argument, ``packed_refinement`` default-None / False
``lab_written``, ``lab_written_p`` and ``lab_pass`` arguments, ``PackedBlocks`` default-None
``lab_written`` / ``lab_written_p`` attributes.  tl07: ``lab_active()`` also runs at the draft rank's
miss redraft (``DraftRunner._advance_jetspec_rows``), which hands the exact round its round-ahead
decision only while it is true; ``DraftRunner._jetspec_fallback``, ``propose_fallback`` and
``_exact_anchor0_rows`` gain a default-None ``round_ahead`` argument; ``round_ahead_seed`` and
``zero_pass_blocks`` run only under E04_LAB_EXACT.  tl08 adds no unconditional call: ``record_topk()``
runs only while ``lab_active()``; the refresh hook's calls (``_adapter_refresh``, ``_exact_refresh``) also
hand it the pass's logits (``draft_logits``); ``packed_refinement`` gains a default-None ``lab_top``
argument, passed (``LabRound.observe_top``) only under E04_LAB_RECORD_TOPK with pre_verify.  tl09 adds no
unconditional call: ``swap_rule()`` (an environment read while unset) runs only while ``lab_active()``, in
``check_regime``, ``LabRound.record``, ``LabRound.observe_top``, ``refresh_hook``'s first_wrong_refresh swap and
``pre_verify``; the round's call sites read it only with pre_verify in E04_LAB_SWAP, to pass ``lab_top`` under
the rule too.  ``LabRound.pin`` gains default ``pinned`` / ``rule`` arguments.  tl10 adds no unconditional call:
``after_swap_fill()``, ``pin_agreed()``, ``tail_passes()`` and ``tl10_on()`` (environment reads) run only while
``lab_active()``; ``packed_refinement``'s ``lab_written`` may be a callable (``LabRound.written_pins``, passed only
under E04_LAB_PIN_AGREED).  tl11 adds no unconditional call: ``_decoder_fill`` runs only under
E04_LAB_AFTER_SWAP_FILL=decoder, from ``_after_swap_fill``'s new default-None ``decoder_argmax`` argument.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from types import SimpleNamespace

import torch

NAMES = ('E04_LAB_REFRESH_CUTOFF', 'E04_LAB_SWAP', 'E04_LAB_SWAP_MARGIN',
         'E04_LAB_REF_TOKENS', 'E04_LAB_PASS_LABEL', 'E04_LAB_REFRESH_EVERY',
         'E04_LAB_EXTRA_PASSES', 'E04_LAB_SHADOW_AR', 'E04_LAB_EXACT', 'E04_LAB_SWAP_KHAT_MAX',
         'E04_LAB_SWAP_SLOT_MAX', 'E04_LAB_SWAP_MARGIN_SLOT1', 'E04_LAB_SWAP_MARGIN_REFRESH',
         'E04_LAB_SWAP_MARGIN_PREVERIFY', 'E04_LAB_PREVERIFY_ROUNDS', 'E04_LAB_ROWS_MIN',
         'E04_LAB_SWAP_ROWS', 'E04_LAB_KSCORE_WRITTEN', 'E04_LAB_SEED_EXACT', 'E04_LAB_SEED_PASSES',
         'E04_LAB_RECORD_TOPK', 'E04_LAB_SWAP_RULE_FILE', 'E04_LAB_AFTER_SWAP_FILL', 'E04_LAB_FILL_SEED',
         'E04_LAB_PIN_AGREED', 'E04_LAB_TAIL_PASSES')
CUTOFFS = ('draft', 'decoder_f', 'oracle')
FILLS = ('keep', 'mask', 'noise', 'context', 'decoder')  # E04_LAB_AFTER_SWAP_FILL (tl10; decoder tl11)
MAX_TAIL_PASSES = 2
KSCORE_WRITTEN = ('disagree', 'draft_p', 'both')
MAX_ROWS_MIN = 16
TOP_N = 5  # the swap decoder's ranked ids and probabilities recorded per swap
SWAP_MODES = ('slot1_pre_j0', 'first_wrong_refresh', 'pre_verify')  # the canonical order
# each swap mode's own margin, overriding E04_LAB_SWAP_MARGIN for that mode while set
MODE_MARGINS = {'slot1_pre_j0': 'E04_LAB_SWAP_MARGIN_SLOT1',
                'first_wrong_refresh': 'E04_LAB_SWAP_MARGIN_REFRESH',
                'pre_verify': 'E04_LAB_SWAP_MARGIN_PREVERIFY'}
MAX_EXTRA_PASSES = 8
MAX_SHADOW_AR = 15
MAX_SWAP_GATE = 15
MAX_PREVERIFY_ROUNDS = 15
MAX_SEED_PASSES = 4
MAX_RECORD_TOPK = 32
# E04_LAB_SWAP_RULE_FILE (SwapRule): the defaults of its keys, and its bias table's slot and draft k-hat buckets
RULE_DEFAULTS = dict(n=8, floor_factor=0.5, alpha=0.0, margin_space='prob', pin=1)
RULE_SLOT_BUCKETS = ('1', '2-3', '4-7', '8-15')
RULE_KHAT_BUCKETS = ('0-2', '3-7', '8-15')
RULE_SPACES = ('prob', 'log')
RULE_TINY = float(torch.finfo(torch.float32).tiny)  # a probability of 0 reads 2^-126 in the score's log
# The graphed regimes the lab has no capture for (the JETSPEC_DECODER_TRUE_TAIL refusal list,
# plus the graphed rows, refresh and adapter paths).
GRAPH_FLAGS = ('JETSPEC_ROWS_GRAPHS', 'JETSPEC_ROW_STAGE_GRAPH', 'JETSPEC_ROWS_FUSED_GRAPH',
               'JETSPEC_ROWS_STATIC', 'JETSPEC_ROWS_STATIC_CHAIN', 'JETSPEC_REFRESH_GRAPH',
               'JETSPEC_C1_GRAPH', 'JETSPEC_F_GRAPHS', 'JETSPEC_DECODER_GRAPH_LEAN',
               'JETSPEC_DECODER_GRAPH_WINDOW', 'JETSPEC_DECODER_ATTN_LEN_LONG',
               'JETSPEC_DECODER_ATTN_LEN')


def lab_active():
    """Any E04_LAB_* name present in the environment (an explicit default counts: it turns
    the per-round record fields on without changing a token)."""
    return any(name in os.environ for name in NAMES)


def refresh_cutoff():
    value = os.environ.get('E04_LAB_REFRESH_CUTOFF', 'draft')
    if value not in CUTOFFS:
        raise ValueError('E04_LAB_REFRESH_CUTOFF must be draft, decoder_f or oracle')
    return value


def swap_modes():
    """E04_LAB_SWAP as a tuple of modes in the canonical order (empty for ``off``): ``off`` or a
    ``+``-joined set of distinct modes, in any order."""
    value = os.environ.get('E04_LAB_SWAP', 'off')
    if value == 'off':
        return ()
    parts = value.split('+')
    if any(part not in SWAP_MODES for part in parts) or len(set(parts)) != len(parts):
        raise ValueError('E04_LAB_SWAP must be off or a +-joined set of distinct modes from '
                         'slot1_pre_j0, first_wrong_refresh, pre_verify')
    return tuple(mode for mode in SWAP_MODES if mode in parts)


def swap_mode():
    """The swap set as one canonical label (``off``, or the modes joined by ``+``)."""
    modes = swap_modes()
    return '+'.join(modes) if modes else 'off'


def refresh_every():
    value = os.environ.get('E04_LAB_REFRESH_EVERY', '0')
    if value not in ('0', '1'):
        raise ValueError('E04_LAB_REFRESH_EVERY must be 0 or 1')
    return value == '1'


def _count(name, maximum):
    value = os.environ.get(name, '0')
    if not (value.isascii() and value.isdecimal()) or not 0 <= int(value) <= maximum:
        raise ValueError(f'{name} must be an integer in 0..{maximum}')
    return int(value)


def extra_passes():
    return _count('E04_LAB_EXTRA_PASSES', MAX_EXTRA_PASSES)


def shadow_ar():
    return _count('E04_LAB_SHADOW_AR', MAX_SHADOW_AR)


def exact_lab():
    """E04_LAB_EXACT=1: the lab hooks act in the exact anchor0 round (``_exact_anchor0_rows``)."""
    value = os.environ.get('E04_LAB_EXACT', '0')
    if value not in ('0', '1'):
        raise ValueError('E04_LAB_EXACT must be 0 or 1')
    return value == '1'


def _gate(name):
    """An E04_LAB_SWAP_*_MAX bound: None while unset, else an integer in 1..15."""
    value = os.environ.get(name)
    if value is None:
        return None
    if not (value.isascii() and value.isdecimal()) or not 1 <= int(value) <= MAX_SWAP_GATE:
        raise ValueError(f'{name} must be unset or an integer in 1..{MAX_SWAP_GATE}')
    return int(value)


def swap_khat_max():
    """E04_LAB_SWAP_KHAT_MAX: the first_wrong_refresh swap needs the row's draft k-hat below it."""
    return _gate('E04_LAB_SWAP_KHAT_MAX')


def swap_slot_max():
    """E04_LAB_SWAP_SLOT_MAX: the first_wrong_refresh swap needs its slot at or below it."""
    return _gate('E04_LAB_SWAP_SLOT_MAX')




def swap_margin(mode=None):
    """E04_LAB_SWAP_MARGIN; for a swap ``mode``, that mode's E04_LAB_SWAP_MARGIN_<MODE>
    (``MODE_MARGINS``) while it is set, else E04_LAB_SWAP_MARGIN."""
    name = None if mode is None else MODE_MARGINS[mode]
    if name is None or name not in os.environ:
        name, value = 'E04_LAB_SWAP_MARGIN', float(os.environ.get('E04_LAB_SWAP_MARGIN', '0.0'))
    else:
        value = float(os.environ[name])
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def preverify_rounds():
    """E04_LAB_PREVERIFY_ROUNDS: the pre_verify swap's rounds per row, 1..15 (1: one swap)."""
    value = os.environ.get('E04_LAB_PREVERIFY_ROUNDS', '1')
    if not (value.isascii() and value.isdecimal()) or not 1 <= int(value) <= MAX_PREVERIFY_ROUNDS:
        raise ValueError(f'E04_LAB_PREVERIFY_ROUNDS must be an integer in 1..{MAX_PREVERIFY_ROUNDS}')
    return int(value)


def rows_min():
    """E04_LAB_ROWS_MIN: None while unset, else the joint rule's row floor n in 1..16."""
    value = os.environ.get('E04_LAB_ROWS_MIN')
    if value is None:
        return None
    if not (value.isascii() and value.isdecimal()) or not 1 <= int(value) <= MAX_ROWS_MIN:
        raise ValueError(f'E04_LAB_ROWS_MIN must be unset or an integer in 1..{MAX_ROWS_MIN}')
    return int(value)


def swap_rows():
    """E04_LAB_SWAP_ROWS=1: a row (s - 1, top offer) per decoder-written slot s of the verified block."""
    value = os.environ.get('E04_LAB_SWAP_ROWS', '0')
    if value not in ('0', '1'):
        raise ValueError('E04_LAB_SWAP_ROWS must be 0 or 1')
    return value == '1'


def kscore_written():
    """E04_LAB_KSCORE_WRITTEN: None while unset, else disagree, draft_p or both."""
    value = os.environ.get('E04_LAB_KSCORE_WRITTEN')
    if value is not None and value not in KSCORE_WRITTEN:
        raise ValueError('E04_LAB_KSCORE_WRITTEN must be unset, disagree, draft_p or both')
    return value


def seed_exact():
    """E04_LAB_SEED_EXACT=1: a forced exact round whose round-ahead rows hit starts from the hit row."""
    value = os.environ.get('E04_LAB_SEED_EXACT', '0')
    if value not in ('0', '1'):
        raise ValueError('E04_LAB_SEED_EXACT must be 0 or 1')
    return value == '1'


def seed_passes():
    """E04_LAB_SEED_PASSES: the drafter passes a seeded exact round runs from the seed, 0..4."""
    return _count('E04_LAB_SEED_PASSES', MAX_SEED_PASSES)


def record_topk():
    """E04_LAB_RECORD_TOPK: None while unset, else the n of the refresh and pre_verify top-n records, 1..32."""
    value = os.environ.get('E04_LAB_RECORD_TOPK')
    if value is None:
        return None
    if not (value.isascii() and value.isdecimal()) or not 1 <= int(value) <= MAX_RECORD_TOPK:
        raise ValueError(f'E04_LAB_RECORD_TOPK must be unset or an integer in 1..{MAX_RECORD_TOPK}')
    return int(value)


def after_swap_fill():
    """E04_LAB_AFTER_SWAP_FILL: keep (default), mask, noise, context or decoder: what the next pass reads at the
    unpinned slots after a first_wrong_refresh swap (``refresh_hook``)."""
    value = os.environ.get('E04_LAB_AFTER_SWAP_FILL', 'keep')
    if value not in FILLS:
        raise ValueError('E04_LAB_AFTER_SWAP_FILL must be keep, mask, noise, context or decoder')
    return value


def fill_seed():
    """E04_LAB_FILL_SEED: the noise fill's seed, a non-negative integer (default 0)."""
    value = os.environ.get('E04_LAB_FILL_SEED', '0')
    if not (value.isascii() and value.isdecimal()):
        raise ValueError('E04_LAB_FILL_SEED must be a non-negative integer')
    return int(value)


def pin_agreed():
    """E04_LAB_PIN_AGREED=1: a first_wrong_refresh refresh under decoder_f also pins slots 1..khat_used."""
    value = os.environ.get('E04_LAB_PIN_AGREED', '0')
    if value not in ('0', '1'):
        raise ValueError('E04_LAB_PIN_AGREED must be 0 or 1')
    return value == '1'


def tail_passes():
    """E04_LAB_TAIL_PASSES: drafter-only passes after the recipe's last pass (no refresh), 0..2."""
    return _count('E04_LAB_TAIL_PASSES', MAX_TAIL_PASSES)


def tl10_on():
    """Any tl10 knob at a value other than its default (an explicit default records nothing new)."""
    return after_swap_fill() != 'keep' or pin_agreed() or tail_passes() > 0




class SwapRule:
    """E04_LAB_SWAP_RULE_FILE: the two-expert swap rule's parameters, validated at load (the module docstring
    has the rule).  ``params`` holds every key with the defaults filled in, ``sha256`` the file's bytes and
    ``bias`` the table as a float64 [slot bucket, k-hat bucket] tensor."""

    def __init__(self, path):
        def refuse(problem):
            raise ValueError(f'E04_LAB_SWAP_RULE_FILE {path}: {problem}')

        def unique(pairs):
            keys = [key for key, _ in pairs]
            if len(set(keys)) != len(keys):
                refuse(f'repeats a key: {sorted({key for key in keys if keys.count(key) > 1})}')
            return dict(pairs)


        try:
            with open(path, 'rb') as handle:
                raw = handle.read()
        except OSError as error:
            raise ValueError(f'E04_LAB_SWAP_RULE_FILE {path}: cannot read it ({error})') from None
        try:
            data = json.loads(raw.decode('utf-8'), object_pairs_hook=unique, parse_constant=constant)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f'E04_LAB_SWAP_RULE_FILE {path}: not UTF-8 JSON ({error})') from None
        if not isinstance(data, dict):
            refuse('must hold a JSON object')
        unknown = sorted(set(data) - set(RULE_DEFAULTS) - {'bias'})
        if unknown:
            refuse(f'unknown keys {unknown} (the keys: {", ".join(RULE_DEFAULTS)}, bias)')
        params = {**RULE_DEFAULTS, **{key: value for key, value in data.items() if key != 'bias'}}
        n, pin = params['n'], params['pin']
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_RECORD_TOPK:
            refuse(f'n must be an integer in 1..{MAX_RECORD_TOPK}')
        floor_factor, alpha = _rule_number(params['floor_factor']), _rule_number(params['alpha'])
        if floor_factor is None or not 0.0 < floor_factor <= 1.0:
            refuse('floor_factor must be a number in (0, 1]')
        if alpha is None or alpha < 0.0:
            refuse('alpha must be a finite number >= 0')
        if not isinstance(params['margin_space'], str) or params['margin_space'] not in RULE_SPACES:
            refuse('margin_space must be "prob" or "log"')
        if isinstance(pin, bool) or not isinstance(pin, int) or pin not in (0, 1):
            refuse('pin must be 1 or 0')
        bias = data.get('bias', {bucket: dict.fromkeys(RULE_KHAT_BUCKETS, 0.0) for bucket in RULE_SLOT_BUCKETS})
        if not isinstance(bias, dict) or set(bias) != set(RULE_SLOT_BUCKETS):
            refuse(f'bias must be an object keyed by exactly the slot buckets {list(RULE_SLOT_BUCKETS)}')
        table = {}
        for bucket in RULE_SLOT_BUCKETS:
            cells = bias[bucket]
            if not isinstance(cells, dict) or set(cells) != set(RULE_KHAT_BUCKETS):
                refuse(f'bias[{bucket!r}] must be an object keyed by exactly the k-hat buckets '
                       f'{list(RULE_KHAT_BUCKETS)}')
            table[bucket] = {}
            for khat in RULE_KHAT_BUCKETS:
                table[bucket][khat] = _rule_number(cells[khat])
                if table[bucket][khat] is None:
                    refuse(f'bias[{bucket!r}][{khat!r}] must be a finite number')
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.n, self.floor_factor, self.alpha, self.pin = n, floor_factor, alpha, pin
        self.margin_space = params['margin_space']
        self.params = dict(n=n, floor_factor=floor_factor, alpha=alpha, margin_space=self.margin_space, pin=pin,
                           bias=table)
        self.bias = torch.tensor([[table[bucket][khat] for khat in RULE_KHAT_BUCKETS] for bucket in RULE_SLOT_BUCKETS],
                                 dtype=torch.float64)

    def bias_at(self, slots, khat):
        """The bias of each (block slot, draft k-hat) cell: slot buckets 1, 2..3, 4..7, 8..15 and k-hat buckets
        0..2, 3..7, 8..15; float64 on ``slots``' device."""
        slot_bucket = (slots >= 2).long() + (slots >= 4).long() + (slots >= 8).long()
        khat_bucket = (khat >= 3).long() + (khat >= 8).long()
        return self.bias.to(slots.device)[slot_bucket, khat_bucket.to(slots.device)]

    def record(self):
        return dict(sha256=self.sha256, params=self.params)


_SWAP_RULES = {}


def swap_rule():
    """E04_LAB_SWAP_RULE_FILE: None while unset, else the parsed rule (``SwapRule``).  The file is read once per
    path, so every round of a read runs the rule its record's sha256 names."""
    path = os.environ.get('E04_LAB_SWAP_RULE_FILE')
    if path is None:
        return None
    if path not in _SWAP_RULES:
        _SWAP_RULES[path] = SwapRule(path)
    return _SWAP_RULES[path]


def pass_label_override():
    """``J2`` when E04_LAB_PASS_LABEL=J2, else None."""
    value = os.environ.get('E04_LAB_PASS_LABEL')
    if value is not None and value != 'J2':
        raise ValueError('E04_LAB_PASS_LABEL must be unset or J2')
    return value




def check_regime(*, graphed=False):
    """Refuse the E04_LAB_* knobs where they have no implementation (the pattern of the
    JETSPEC_DECODER_TRUE_TAIL refusal).  A no-op while every E04_LAB_* name is unset."""
    if not lab_active():
        return
    cutoff, swaps = refresh_cutoff(), swap_modes()
    every, extra, shadow = refresh_every(), extra_passes(), shadow_ar()
    exact, khat_max, slot_max = exact_lab(), swap_khat_max(), swap_slot_max()
    floor, added_rows, written = rows_min(), swap_rows(), kscore_written()
    seeded, seeded_passes = seed_exact(), seed_passes()
    topk = record_topk()
    rule = swap_rule()  # E04_LAB_SWAP_RULE_FILE: read and validated here, at load
    fill, agreed, tail = after_swap_fill(), pin_agreed(), tail_passes()  # tl10
    fill_seed()
    for mode in (None,) + SWAP_MODES:
        swap_margin(mode)
    preverify_rounds()
    pass_label_override()
    if graphed or any(os.environ.get(name) == '1' for name in GRAPH_FLAGS) or \
            os.environ.get('JETSPEC_CHAIN_GRAPH', '0') != '0':
        raise ValueError('E04_LAB_* knobs do not support graphed paths')
    if os.environ.get('JETSPEC_DECODER_LOOP') == '1':
        raise ValueError('E04_LAB_* knobs do not support JETSPEC_DECODER_LOOP=1')
    for name in ('JETSPEC_DECODER_PREFIX_MERGE', 'JETSPEC_DECODER_LMHEAD_KN'):
        if os.environ.get(name, '0') == '1':
            raise ValueError(f'E04_LAB_* knobs do not support {name}=1')
    if cutoff == 'oracle' and not os.environ.get('E04_LAB_REF_TOKENS'):
        raise ValueError('E04_LAB_REFRESH_CUTOFF=oracle requires E04_LAB_REF_TOKENS')
    if (cutoff != 'draft' or swaps or every or extra or shadow or exact or floor is not None or added_rows
            or written is not None or topk is not None or tail) and (
            os.environ.get('JETSPEC_ROW_LAYOUT', 'kth') != 'anchor0'
            or os.environ.get('JETSPEC_DRAFTS') not in ('15', '31')):
        raise ValueError('E04_LAB_REFRESH_CUTOFF / _SWAP / _REFRESH_EVERY / _EXTRA_PASSES / _SHADOW_AR '
                         '/ _EXACT / _ROWS_MIN / _SWAP_ROWS / _KSCORE_WRITTEN / _RECORD_TOPK / _TAIL_PASSES serve '
                         'JETSPEC_ROW_LAYOUT=anchor0 with JETSPEC_DRAFTS=15 or 31 only')
    if (floor is not None or added_rows) and os.environ.get('E04_ROWRULE') != 'joint':
        raise ValueError('E04_LAB_ROWS_MIN / E04_LAB_SWAP_ROWS add rows after the joint row rule: they '
                         'need E04_ROWRULE=joint')
    if (added_rows or written is not None) and not (swaps or shadow):
        raise ValueError('E04_LAB_SWAP_ROWS / E04_LAB_KSCORE_WRITTEN read the decoder-written slots: they '
                         'need E04_LAB_SWAP or E04_LAB_SHADOW_AR')
    if written is not None and os.environ.get('E04_KSCORE') not in (
            ('blend', 'blend_dec') if written == 'draft_p' else ('blend_dec',)):
        raise ValueError('E04_LAB_KSCORE_WRITTEN changes the BLEND k score: draft_p needs E04_KSCORE=blend '
                         'or blend_dec, disagree and both need E04_KSCORE=blend_dec (the argmax term)')
    if (khat_max is not None or slot_max is not None) and 'first_wrong_refresh' not in swaps:
        raise ValueError('E04_LAB_SWAP_KHAT_MAX / E04_LAB_SWAP_SLOT_MAX gate the first_wrong_refresh '
                         'swap only')
    for mode, name in MODE_MARGINS.items():
        if name in os.environ and mode not in swaps:
            raise ValueError(f'{name} is the {mode} swap\'s margin: it needs {mode} in E04_LAB_SWAP')
    if 'E04_LAB_PREVERIFY_ROUNDS' in os.environ and 'pre_verify' not in swaps:
        raise ValueError('E04_LAB_PREVERIFY_ROUNDS repeats the pre_verify swap: it needs pre_verify in '
                         'E04_LAB_SWAP')
    if seeded and not (exact and os.environ.get('JETSPEC_MISS_REDRAFT', '0') == '1'
                       and os.environ.get('JETSPEC_MISS_REDRAFT_FORCE', '0') == '1'):
        raise ValueError('E04_LAB_SEED_EXACT seeds the forced exact round from its round-ahead hit: it needs '
                         'JETSPEC_MISS_REDRAFT=1, JETSPEC_MISS_REDRAFT_FORCE=1 and E04_LAB_EXACT=1')
    if 'E04_LAB_SEED_PASSES' in os.environ and not seeded:
        raise ValueError('E04_LAB_SEED_PASSES counts the seeded round\'s drafter passes: it needs '
                         'E04_LAB_SEED_EXACT=1')
    if seeded and not seeded_passes and (extra or written in ('draft_p', 'both')):
        raise ValueError('E04_LAB_SEED_PASSES=0 runs no drafter pass in a seeded round: E04_LAB_EXTRA_PASSES '
                         'and E04_LAB_KSCORE_WRITTEN=draft_p / both need one')
    if rule is not None:
        if not {'first_wrong_refresh', 'pre_verify'} & set(swaps):
            raise ValueError('E04_LAB_SWAP_RULE_FILE decides the first_wrong_refresh and pre_verify swaps: it '
                             'needs one of them in E04_LAB_SWAP')
        from jetspec.engines.jetspec_plus_plus.draft.block_geometry import block_width
        if block_width() != 16:
            raise ValueError('E04_LAB_SWAP_RULE_FILE: its bias table buckets block slots 1..15 and k-hat 0..15, '
                             'so it serves JETSPEC_DRAFTS=15 only')
        if 'first_wrong_refresh' in swaps and cutoff != 'decoder_f':
            raise ValueError('E04_LAB_SWAP_RULE_FILE scans the first_wrong_refresh swap from the decoder\'s first '
                             'disagreement: it needs E04_LAB_REFRESH_CUTOFF=decoder_f')
        for mode in ('first_wrong_refresh', 'pre_verify'):
            if MODE_MARGINS[mode] in os.environ:
                raise ValueError(f'{MODE_MARGINS[mode]} is inert under E04_LAB_SWAP_RULE_FILE: the rule\'s bias '
                                 f'table is the {mode} swap\'s margin')
        if seeded and not seeded_passes and 'pre_verify' in swaps:
            raise ValueError('E04_LAB_SEED_PASSES=0 runs no drafter pass in a seeded round: '
                             'E04_LAB_SWAP_RULE_FILE\'s pre_verify reads the final pass\'s drafter distribution')
    # tl10: E04_LAB_AFTER_SWAP_FILL / _FILL_SEED / _PIN_AGREED act at the first_wrong_refresh swap only
    if fill != 'keep' and 'first_wrong_refresh' not in swaps:
        raise ValueError('E04_LAB_AFTER_SWAP_FILL fills the next pass\'s input after the first_wrong_refresh swap: '
                         'it needs first_wrong_refresh in E04_LAB_SWAP')
    if 'E04_LAB_FILL_SEED' in os.environ and fill != 'noise':
        raise ValueError('E04_LAB_FILL_SEED seeds the noise fill: it needs E04_LAB_AFTER_SWAP_FILL=noise')
    if agreed and ('first_wrong_refresh' not in swaps or cutoff != 'decoder_f'):
        raise ValueError('E04_LAB_PIN_AGREED pins the agreed run at the first_wrong_refresh refresh under decoder_f: '
                         'it needs first_wrong_refresh in E04_LAB_SWAP and E04_LAB_REFRESH_CUTOFF=decoder_f')
    if seeded and not seeded_passes and tail:
        raise ValueError('E04_LAB_SEED_PASSES=0 runs no drafter pass in a seeded round: E04_LAB_TAIL_PASSES '
                         'adds passes after the round\'s own')
    refresh_off = os.environ.get('JETSPEC_ADAPTER_REFRESH', 'off') == 'off'
    if cutoff != 'draft' and refresh_off:
        raise ValueError('E04_LAB_REFRESH_CUTOFF acts only under JETSPEC_ADAPTER_REFRESH')
    if 'first_wrong_refresh' in swaps and refresh_off:
        raise ValueError('E04_LAB_SWAP=first_wrong_refresh acts only under JETSPEC_ADAPTER_REFRESH')
    if (every or extra) and refresh_off:
        raise ValueError('E04_LAB_REFRESH_EVERY / E04_LAB_EXTRA_PASSES refresh before their passes: '
                         'they need JETSPEC_ADAPTER_REFRESH')
    if topk is not None and refresh_off:
        raise ValueError('E04_LAB_RECORD_TOPK records at the adapter refresh (the lab refresh hook): it needs '
                         'JETSPEC_ADAPTER_REFRESH')


# --- the refresh-context decoder window (spec note 2026-09-29 section 3) -----------------------

def refresh_window_layout(tail_tokens, tail_counts, slot_counts, *, length, device=None):
    """The window of ``refresh_decoder_logits``: tail slots 0..tail_tokens-1, then row r's block
    slots 0..slot_counts[r]-1 for each row in order.

    Returns ``(allowed [N, N] bool, positions [N], row_of [N], slot_of [N])``: row_of is -1 on a
    tail token.  Tail slot t sits at ``length + t`` and sees tail slots <= t; block slot s of row r
    sits at ``length + tail_counts[r] + s`` (``refresh_window``'s positions) and sees tail slots
    < tail_counts[r] and its own row's block slots <= s.  No token reads another row; every
    token sees itself (the diagonal the decoder's causal attention needs).  The committed
    prefix, visible to every token, is outside this matrix.  The exact round (E04_LAB_EXACT) has no
    tail: ``tail_tokens`` 0 and every row's tail count 0, so block slot s sits at ``length + s``.
    """
    tail_counts = torch.as_tensor(tail_counts, dtype=torch.long, device=device).reshape(-1)
    slot_counts = torch.as_tensor(slot_counts, dtype=torch.long, device=device).reshape(-1)
    device = tail_counts.device
    if tail_counts.shape != slot_counts.shape:
        raise ValueError('tail_counts and slot_counts need one entry per row')
    if bool((tail_counts < (1 if tail_tokens else 0)).any()) or bool((tail_counts > tail_tokens).any()):
        raise ValueError('every row needs 1..tail_tokens staged tail slots (none in the exact round)')
    if bool((slot_counts < 1).any()):
        raise ValueError('every row needs at least one block slot')
    rows = int(tail_counts.numel())
    total = int(slot_counts.sum())
    row_index = torch.repeat_interleave(torch.arange(rows, device=device), slot_counts)
    starts = torch.cumsum(slot_counts, 0) - slot_counts
    slot = torch.arange(total, device=device) - starts[row_index]
    tail = torch.arange(tail_tokens, device=device)
    row_of = torch.cat((torch.full((tail_tokens,), -1, dtype=torch.long, device=device), row_index))
    slot_of = torch.cat((tail, slot))
    positions = torch.cat((length + tail, length + tail_counts[row_index] + slot))
    query_tail = row_of[:, None] < 0
    key_tail = row_of[None, :] < 0
    span = torch.cat((tail + 1, tail_counts[row_index]))  # tail keys a query may read: t < span
    allowed = torch.where(
        key_tail, slot_of[None, :] < span[:, None],
        (~query_tail) & (row_of[None, :] == row_of[:, None]) & (slot_of[None, :] <= slot_of[:, None]))
    return allowed, positions, row_of, slot_of


class _WindowCache:
    """Prefix K/V per layer index, extended by each call's keys; ``append`` keeps them."""

    def __init__(self, kv, append):
        self.kv = dict(kv)
        self.append = append

    def update(self, key, value, layer_idx, cache_kwargs=None):
        if layer_idx in self.kv:
            prefix_key, prefix_value = self.kv[layer_idx]
            key = torch.cat((prefix_key, key), dim=-2)
            value = torch.cat((prefix_value, value), dim=-2)
        if self.append:
            self.kv[layer_idx] = (key, value)
        return key, value


def _additive_mask(allowed, like):
    return like.new_zeros(allowed.shape).masked_fill_(~allowed, torch.finfo(like.dtype).min)[None, None]


@torch.inference_mode()
def refresh_decoder_logits(decoder, sequence, *, length, tail_hidden, tail_counts, block_hidden,
                           slot_counts):
    """The anchor decoder over each row's refreshed block, in the refresh context.

    * prefix: the decoder's K/V over the TRUE tap-33 states of committed positions
      0..length-1.  The sequence's cached decoder prefix is read up to its current length (the
      round start builds it to length-4) and the remaining true positions are run through the
      prefix layers into a private copy, so the cached prefix is never changed.
    * tail: ``tail_hidden`` [T, H] (the staged block's tap-33 slice) at positions length..,
      causal, computed once and shared by every row.  None in the exact round (E04_LAB_EXACT),
      whose rows read no tail (every tail count 0).
    * row r: ``block_hidden[r, :slot_counts[r]]`` at ``length + tail_counts[r] + slot``, reading
      the prefix, tail slots < tail_counts[r] and its own earlier slots (with the diagonal).

    Returns ``(logits [N, V], row_of [N], slot_of [N])`` over the packed block tokens; the logits
    of block slot j are the decoder's distribution for block slot j + 1.
    """
    parameter = next(decoder.layers.parameters())
    hidden_size = decoder.hidden_size
    tail_counts = torch.as_tensor(tail_counts, dtype=torch.long).reshape(-1).cpu()
    slot_counts = torch.as_tensor(slot_counts, dtype=torch.long).reshape(-1).cpu()
    tail_tokens = int(tail_counts.max())
    if tail_tokens > (0 if tail_hidden is None else int(tail_hidden.shape[0])):
        raise ValueError('a row reads more tail slots than the staged block holds')
    taps = sequence.taps
    if int(taps.shape[0]) < length:
        raise ValueError('the decoder refresh prefix needs the true taps of every committed position')
    existing = getattr(decoder, '_prefixes', {}).get(id(sequence))
    base = int(existing.length) if existing is not None and int(existing.length) <= length else 0
    kv = ({index: (pair[0][..., :base, :], pair[1][..., :base, :])
           for index, pair in existing.kv.items()} if base else {})
    cache = _WindowCache(kv, append=True)
    if base < length:
        hidden = taps[base:length, 4, :].to(device=parameter.device, dtype=parameter.dtype)[None]
        positions = torch.arange(base, length, device=parameter.device, dtype=torch.long)
        keys = torch.arange(length, device=parameter.device)
        mask = _additive_mask(keys[None, :] <= positions[:, None], hidden)
        rope = decoder.rotary(hidden, positions[None])
        for layer in decoder.prefix_layers:
            hidden = layer(hidden, attention_mask=mask, position_ids=positions[None],
                           position_embeddings=rope, past_key_values=cache,
                           cache_position=positions, use_cache=True)
    allowed, positions, row_of, slot_of = refresh_window_layout(
        tail_tokens, tail_counts, slot_counts, length=length)
    rows = torch.arange(int(slot_counts.numel()))
    block_index = torch.repeat_interleave(rows, slot_counts)
    tail = None if tail_hidden is None else tail_hidden[:tail_tokens, -hidden_size:]
    block = block_hidden[block_index.to(block_hidden.device), slot_of[tail_tokens:].to(block_hidden.device),
                         -hidden_size:]
    if tail is None:
        hidden = block.to(parameter)[None]
    else:
        hidden = torch.cat((tail.to(parameter), block.to(parameter)), dim=0)[None]
    positions = positions.to(parameter.device)
    full = torch.cat((torch.ones(allowed.shape[0], length, dtype=torch.bool), allowed), dim=1)
    mask = _additive_mask(full.to(parameter.device), hidden)
    rope = decoder.rotary(hidden, positions[None])
    frozen = _WindowCache(cache.kv, append=False)
    for layer in decoder.layers:
        hidden = layer(hidden, attention_mask=mask, position_ids=positions[None],
                       position_embeddings=rope, past_key_values=frozen, use_cache=False)
    decoded = decoder.norm(hidden)[0, tail_tokens:]
    return decoder.lm_head(decoded), row_of[tail_tokens:], slot_of[tail_tokens:]


def dense_rows(values, row_of, slot_of, rows, width, fill):
    """Scatter packed per-token values into [rows, width] (``fill`` elsewhere)."""
    out = torch.full((rows, width), fill, dtype=values.dtype, device=values.device)
    out[row_of.to(values.device), slot_of.to(values.device)] = values
    return out


# --- cutoffs -----------------------------------------------------------------------------------

from jetspec.engines.jetspec_plus_plus.draft.method.refresh import rule_f


def oracle_true_k(ids, reference, starts):
    """The number of leading drafts d_1.. of each row equal to the reference continuation:
    row r's block slot j sits at absolute position ``starts[r] + j`` and is compared with
    ``reference[starts[r] + j]``.  A position past the reference's end ends the count (the
    reference holds no token there)."""
    width = ids.shape[1]
    reference = torch.as_tensor(reference, dtype=torch.long).reshape(-1).to(ids.device)
    starts = torch.as_tensor(starts, dtype=torch.long).reshape(-1, 1).to(ids.device)
    positions = starts + torch.arange(1, width, device=ids.device)[None, :]
    inside = positions < reference.numel()
    expected = reference[positions.clamp(max=max(int(reference.numel()) - 1, 0))]
    match = inside & (ids[:, 1:] == expected)
    return torch.cumprod(match.long(), dim=1).sum(dim=1)


class ReferenceTokens:
    """E04_LAB_REF_TOKENS: ``{"requests": {"<prompt index>": [output token ids]}}``.

    The key is the prompt's ``index`` in the prompts JSON (bench.py's ``prompt_indices``, which
    the draft side carries as ``JetSpecSequence.prompt_index``), spelled as a decimal string;
    the value is that request's full T=0 output token ids (the ``token_ids`` a knob-off serve_read
    wrote).  A missing key, a committed output longer than the reference, or a committed token
    that differs from it raises: the oracle never guesses.
    """

    def __init__(self, path):
        with open(path, encoding='utf-8') as handle:
            data = json.load(handle)
        requests = data.get('requests') if isinstance(data, dict) else None
        if not isinstance(requests, dict) or not requests:
            raise ValueError(f'{path}: E04_LAB_REF_TOKENS needs a non-empty "requests" object')
        self.path = str(path)
        self.requests = {}
        for key, tokens in requests.items():
            if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
                raise ValueError(f'{path}: request {key!r} must map to a list of token ids')
            self.requests[str(key)] = torch.tensor(tokens, dtype=torch.long)

    def full_sequence(self, sequence):
        """The reference [prompt, output] for this sequence, after checking its committed output."""
        key = str(int(sequence.prompt_index))
        if key not in self.requests:
            raise ValueError(f'E04_LAB_REF_TOKENS {self.path} has no request {key}')
        prompt_length = getattr(sequence, 'e04_lab_prompt_length', None)
        if prompt_length is None:
            raise ValueError('E04_LAB_REFRESH_CUTOFF=oracle requires the admission prompt length')
        committed = sequence.committed_ids.reshape(-1).cpu()
        generated = committed[int(prompt_length):]
        reference = self.requests[key]
        if generated.numel() > reference.numel():
            raise ValueError(f'request {key}: {generated.numel()} committed output tokens exceed the '
                             f'reference\'s {reference.numel()}')
        if not torch.equal(generated, reference[:generated.numel()]):
            first = int((generated != reference[:generated.numel()]).nonzero()[0])
            raise ValueError(f'request {key}: committed output token {first} differs from the reference')
        return torch.cat((committed[:int(prompt_length)], reference))


_REFERENCES = {}


def reference_tokens():
    path = os.environ.get('E04_LAB_REF_TOKENS')
    if not path:
        raise ValueError('E04_LAB_REFRESH_CUTOFF=oracle requires E04_LAB_REF_TOKENS')
    if path not in _REFERENCES:
        _REFERENCES[path] = ReferenceTokens(path)
    return _REFERENCES[path]


def lab_seq_id(sequence):
    """The request's seq id: the draft rank's (-1 on a synthetic warm-up state), else the one the
    lab's admission records before it exists (``e04_lab_seq_id``: the first, exact round), else -1."""
    return int(getattr(sequence, '_jetspec_seq_id', getattr(sequence, 'e04_lab_seq_id', -1)))


def live_sequence(sequence):
    """A served request (not a synthetic warm-up state, whose seq id is -1)."""
    return lab_seq_id(sequence) >= 0


# --- the round's lab state ---------------------------------------------------------------------

class LabRound:
    """One round's lab state: the per-row slot pins every pass applies (-1: none) and the record.
    A swap's ``pass_index`` is the pass whose input carries it (0 for slot1_pre_j0, the refresh's
    pass for first_wrong_refresh) or ``'pre_verify'``.  ``recipe_passes`` (the recipe's pass
    count) marks a refresh before an extra pass (E04_LAB_EXTRA_PASSES) in the record.  ``exact``
    (E04_LAB_EXACT): the exact anchor0 round's state; its record carries ``exact=1``."""

    def __init__(self, rows, width, device, recipe_passes=None, exact=False):
        self.pins = torch.full((rows, width), -1, dtype=torch.long, device=device)
        # rows swapped during this pass: packed_refinement reopens them (a row that had already
        # converged is frozen, and the pass that carries its swap would otherwise not publish)
        self.touched = torch.zeros(rows, dtype=torch.bool, device=device)
        self.recipe_passes = recipe_passes
        self.exact = bool(exact)
        self.refreshes = []
        self.swaps = []
        self.slot1 = None
        self.shadow = None
        self.final = None
        self.rows = None      # E04_ROWRULE=joint rounds: joint_rows' record
        self.verified = None  # the verified block's decoder-written slots (verified_record)
        # exact rounds (E04_LAB_SEED_EXACT, seed_round): whether the round starts from a seed, its record,
        # a forced round's round-ahead decision, the pre_verify reads run and the published final blocks
        self.seeded = False
        self.seed = None
        self.forced = None
        self.reads = 0
        self.final_block = None
        # E04_LAB_RECORD_TOPK: each row's [k, anchor] (the caller sets it) and each row's top-n at the pass its
        # tokens froze at (observe_top), for the refresh and pre_verify records
        self.keys = None
        self.final_top = None
        # E04_LAB_SWAP_RULE_FILE: each row's drafter top-n (the rule's n) at the pass its tokens froze at
        # (observe_top), the drafter's distribution pre_verify's rule reads
        self.rule_top = None
        # tl10: E04_LAB_PIN_AGREED's pins (a bool per slot, None until the first; not decoder-written), the
        # after-swap fill's sources (``fill_source``) and the slots the last refresh filled (the mask guard
        # exempts them in that refresh's pass only), and the passes each published row ran
        self.agreed = None
        self.fill_mask_token = self.fill_context = None
        self.filled = None
        self.passes_run = None




    def pin(self, pass_index, rows, slot, old, new, top=None, pinned=True, rule=None):
        """Pin ``new`` at (rows[i], slot[i]); record (pass, row, slot, old, new) and, with ``top``
        (the swap decoder's ranked (ids, probabilities) [len(rows), TOP_N] at each slot),
        ``top_ids`` / ``top_p``.  ``pinned`` False (E04_LAB_SWAP_RULE_FILE's pin 0): the swap is recorded,
        and its row reopened, without the pin; ``rule`` (one dict per swap): the rule's ``rule`` record."""
        if len(rows):
            if pinned:
                self.pins[rows, slot] = new
            if not isinstance(pass_index, str):
                self.touched[rows] = True
            ranked = [None] * len(rows) if top is None else list(zip(top[0].tolist(), top[1].tolist()))
            for values, top_row in zip(zip(rows.tolist(), slot.tolist(), old.tolist(), new.tolist()), ranked):
                self.swaps.append(dict(pass_index=pass_index if isinstance(pass_index, str)
                                       else int(pass_index), row=values[0], slot=values[1],
                                       old=values[2], new=values[3]))
                if top_row is not None:
                    self.swaps[-1].update(top_ids=top_row[0], top_p=top_row[1])
            for swap, values in zip(self.swaps[-len(rows):], [] if rule is None else rule):
                swap['rule'] = values



    def record(self):
        out = dict(cutoff=refresh_cutoff(), swap=swap_mode(), swap_margin=swap_margin(),
                   swap_margins={mode: swap_margin(mode) for mode in swap_modes()},
                   pass_label=pass_label_override(), refresh_every=int(refresh_every()),
                   extra_passes=extra_passes(), shadow_ar=shadow_ar(),
                   swap_khat_max=swap_khat_max(), swap_slot_max=swap_slot_max(),
                   preverify_rounds=preverify_rounds(),
                   refreshes=self.refreshes, swaps=self.swaps)
        out.update(rows_min=rows_min(), swap_rows=int(swap_rows()), kscore_written=kscore_written())
        out.update(seed_exact=int(seed_exact()), seed_passes=seed_passes())
        if record_topk() is not None:
            out['record_topk'] = record_topk()  # E04_LAB_RECORD_TOPK: only while set (records only)
        if swap_rule() is not None:
            out['swap_rule'] = swap_rule().record()  # E04_LAB_SWAP_RULE_FILE: only while set
        if tl10_on():
            # tl10: the knobs and the passes each published row ran, only while one is off its default
            out.update(after_swap_fill=after_swap_fill(),
                       fill_seed=fill_seed() if after_swap_fill() == 'noise' else None,
                       pin_agreed=int(pin_agreed()), tail_passes=tail_passes(), passes_run=self.passes_run)
        if self.exact:
            out['exact'] = 1
            # E04_LAB_SEED_EXACT: the seed, the S reads, the swaps per mode and the published blocks
            out.update(seed=self.seed, s_reads=self.reads, final_block=self.final_block,
                       swaps_by_mode={mode: sum(_swap_mode_of(swap) == mode for swap in self.swaps)
                                      for mode in SWAP_MODES})
            if self.forced is not None:
                out['forced'] = self.forced
        if self.slot1 is not None:
            out['slot1_pre_j0'] = self.slot1
        if self.shadow is not None:
            out['shadow_ar_ids'] = self.shadow
        if self.final is not None:
            out['pre_verify'] = self.final
        if self.rows is not None:
            out['rows'] = self.rows
        if self.verified is not None:
            out['verified'] = self.verified
        return out




def _swap_mode_of(swap):
    """A recorded swap's E04_LAB_SWAP mode: slot1_pre_j0 pins at pass 0, a refresh swap at its pass >= 1."""
    if swap['pass_index'] == 'pre_verify':
        return 'pre_verify'
    return 'slot1_pre_j0' if swap['pass_index'] == 0 else 'first_wrong_refresh'


def _per_row(rows, chosen, values, device):
    """A [rows] list with ``values`` at ``chosen`` and -1 elsewhere (the record's per-row form)."""
    out = torch.full((rows,), -1, dtype=torch.long, device=device)
    if len(chosen):
        out[chosen] = values.to(device)
    return out.tolist()


def _top(probabilities, rows):
    """The swap decoder's top TOP_N (ids, probabilities) of ``probabilities`` [N, V] at ``rows``."""
    values, ids = probabilities[rows.to(probabilities.device)].topk(
        min(TOP_N, probabilities.shape[-1]), dim=-1)
    return ids, values


# --- tl08: E04_LAB_RECORD_TOPK, both experts' top-n at every refresh (records only) ---------------------

def _ranked(probabilities, n):
    """The top ``n`` (ids, probabilities) of ``probabilities`` [..., V] (an fp32 softmax at T = 1) in
    descending probability, on its device."""
    ranked = probabilities.topk(min(n, probabilities.shape[-1]), dim=-1)
    return ranked.indices, ranked.values


def _home(tensors):
    """A record's one host transfer: ``tensors`` (top-n ids and fp32 probabilities, index tensors) packed as
    float64, which holds every id and every fp32 value exactly, and split back into their shapes on the host."""
    device = tensors[0].device
    packed = torch.cat([tensor.reshape(-1).to(device=device, dtype=torch.float64) for tensor in tensors]).cpu()
    parts = packed.split([tensor.numel() for tensor in tensors])
    return [part.reshape(tensor.shape) for part, tensor in zip(parts, tensors)]


def _refresh_topk(n, keys, draft_logits, read, *, rows, width):
    """E04_LAB_RECORD_TOPK at one refresh: the rows' ``keys`` and both experts' top-n at slots 1..width-1 of
    the block ``rows`` x ``width`` the refresh read (list index j - 1 is slot j), in one host transfer.

    ``draft_logits`` [rows, width, V] are the logits of the pass whose block the refresh read: the drafter's.
    ``read`` is the refresh's decoder read ``(logits, row_of, slot_of)`` (packed; its row for block slot s is
    the distribution for slot s + 1), or None without one: the decoder's, a slot past the read None."""
    if draft_logits is None or tuple(draft_logits.shape[:2]) != (rows, width):
        raise ValueError('E04_LAB_RECORD_TOPK reads the logits of the pass whose block the refresh read, at '
                         'every block slot')
    tensors = list(_ranked(torch.softmax(draft_logits[:, 1:].float(), dim=-1), n))
    if read is not None:
        tensors += _ranked(torch.softmax(read[0].float(), dim=-1), n)
    home = _home(tensors)
    out = dict(keys=keys, draft_top_ids=home[0].long().tolist(), draft_top_p=home[1].tolist(),
               dec_top_ids=None, dec_top_p=None)
    if read is not None:
        out['dec_top_ids'] = [[None] * (width - 1) for _ in range(rows)]
        out['dec_top_p'] = [[None] * (width - 1) for _ in range(rows)]
        for row, slot, top_ids, top_p in zip(read[1].tolist(), read[2].tolist(), home[2].long().tolist(),
                                             home[3].tolist()):
            out['dec_top_ids'][row][slot], out['dec_top_p'][row][slot] = top_ids, top_p
    return out


# --- tl09: E04_LAB_SWAP_RULE_FILE, the two-expert swap rule ------------------------------------------------

from jetspec.engines.jetspec_plus_plus.draft.method.refresh import _rule_read


from jetspec.engines.jetspec_plus_plus.draft.method.refresh import _rule_choice


def _rule_values(choice, bias):
    """A swap record's float values at each slot of ``_rule_choice``'s result, [M, 7] in ``_rule_records``' order."""
    return torch.stack((choice.score_x, choice.score_d, choice.p_s_x, choice.p_s_d, choice.p_d_x, choice.p_d_d,
                        bias.to(choice.score_x.device)), dim=1)


def _rule_records(values, khat, x, pinned):
    """The ``rule`` record of each swap the rule made: x*, ``values`` [swaps, 7] (``_rule_values``), the row's
    draft k-hat ``khat`` [swaps] and ``pinned``, in one host transfer."""
    names = ('score_x', 'score_d', 'p_s_x', 'p_s_d', 'p_d_x', 'p_d_d', 'bias')
    home = _home([values, khat, x])
    return [dict(x_star=token, **dict(zip(names, row)), khat_draft=k, pinned=bool(pinned))
            for row, k, token in zip(home[0].tolist(), home[1].long().tolist(), home[2].long().tolist())]


def _rule_refresh(rule, logits, row_of, slot_of, *, ids, khat, khat_used, pins, draft_logits, candidates):
    """E04_LAB_SWAP_RULE_FILE at one refresh: the rule (``_rule_choice``, the draft excluded) at every slot of the
    refresh's decoder read (``logits`` packed: block slot s's row is the distribution for slot s + 1), the drafter's
    top-n from ``draft_logits`` [R, B, V] (the pass whose block ``ids`` the refresh read), the bias of each slot and
    the row's draft k-hat ``khat``.  Per row of ``candidates`` (the rows whose slot khat_used + 1 is free), F is the
    first slot >= khat_used + 1 the read covers that is unpinned and qualifies.  Returns ``(where, x, found, values,
    khat)`` per candidate: F (khat_used + 1 where none qualifies), x* there, whether a slot qualified, its record
    values (``_rule_values``) and the row's draft k-hat."""
    rows, width = ids.shape
    if draft_logits is None or tuple(draft_logits.shape[:2]) != (rows, width):
        raise ValueError('E04_LAB_SWAP_RULE_FILE reads the logits of the pass whose block the refresh read, at '
                         'every block slot')
    device = logits.device
    row_of, slot_of = row_of.to(device), slot_of.to(device)
    slot = slot_of + 1
    draft = _ranked(torch.softmax(draft_logits[:, 1:].float(), dim=-1), rule.n)
    draft = tuple(tensor.to(device)[row_of, slot_of] for tensor in draft)
    row_khat = khat.to(device)[row_of]
    bias = rule.bias_at(slot, row_khat)
    choice = _rule_choice(rule, torch.softmax(logits.float(), dim=-1), draft, ids.to(device)[row_of, slot], bias,
                          exclude_draft=True)
    scan = choice.ok & (pins.to(device)[row_of, slot] < 0) & (slot > khat_used.to(device)[row_of])
    hit = dense_rows(scan, row_of, slot_of, rows, width, False)[candidates.to(device)]
    found = hit.any(dim=1)
    where = torch.where(found, hit.long().argmax(dim=1) + 1, khat_used.to(device)[candidates.to(device)] + 1)
    at = dense_rows(torch.arange(len(row_of), device=device), row_of, slot_of, rows, width, -1)[
        candidates.to(device), where - 1]
    return where, choice.x[at], found, _rule_values(choice, bias)[at], row_khat[at]


from jetspec.engines.jetspec_plus_plus.draft.method.refresh import margin_swap


def adapter_block(adapter_head, embed_tokens, sequence, *, length, tail_counts, ids, slots, pack):
    """The adapter's rows for block slots 0..slots-1 of every row of ``ids`` [R, B], in the
    refresh geometry (``_adapter_refresh``'s calls: the cached tail under JETSPEC_REFRESH_PACK=2,
    else the uncached window).  A slot reads the prefix, the row's tail slots < tail_counts[r]
    and the row's earlier slots only, so the slots past ``slots`` do not enter.  None when the
    staged tail is not at ``length`` or the adapter holds no cache for this prefix."""
    staged = sequence.staged
    if staged.start_position != length or staged.token_ids is None:
        return None
    rows = ids.shape[0]
    embedded = embed_tokens(ids)
    block = None
    if pack == '2':
        from jetspec.engines.jetspec_plus_plus.draft.live_rows import refresh_token_bucket
        block = adapter_head.refresh_cached(
            block_embeddings=embedded, length=length, tail_counts=tail_counts, prefix_key=sequence,
            row_slots=[slots] * rows,
            slot_counts=torch.full((rows,), slots, dtype=torch.long, device=ids.device),
            pad_to=refresh_token_bucket(rows * slots))
    if block is None:
        window = adapter_head.refresh_window(
            tail_embeddings=embed_tokens(staged.token_ids.to(ids.device))[0],
            block_embeddings=embedded, length=length, tail_counts=tail_counts, prefix_key=sequence,
            row_slots=[slots] * rows)
        block = None if window is None else window[1]
    return block


@torch.inference_mode()
def exact_adapter_rows(drafter, sequence, ids):
    """E04_LAB_EXACT: the adapter's rows for every slot of each row of ``ids`` [R, B], a block
    [correction, slots] at the correction's position ``length`` (the count of committed taps).

    The round-start staging call's own adapter call (``KthRowsDrafter._stage_packed`` for one
    sequence): the prefix is the target's taps of positions 0..length-1 and the window is the block
    itself, so no slot reads an adapter tail.  Under D1 a window slot reads the prefix and the
    strictly earlier slots (the refresh's rule; the cached C1 call builds that window itself), so a
    slot's row does not depend on the tokens after it.  One call per distinct row.  A cached C1
    head that did not hold this sequence's prefix before the call has its key cleared after it, so
    the next staging call rebuilds from position 0 as it does without the lab (the rule of
    ``heads.decoder.e04_seed_admission_tail``).  Returns [R, B, H].
    """
    from jetspec.engines.jetspec_plus_plus.draft.drafter import ServingCtxKVC1Head
    head = drafter.adapter.head
    device = drafter.device
    prefix = sequence.exact_hidden(device=device, dtype=drafter.dtype)
    length, width = int(prefix.shape[1]), int(ids.shape[1])
    query = torch.arange(width, device=device)
    causal = getattr(head, 'resolved_config', {}).get('window_causal_over_tokens', False)
    window = query[:, None] > query[None, :] if causal else query[:, None] == query[None, :]
    allowed = torch.cat((torch.ones(width, length, dtype=torch.bool, device=device), window), dim=1)
    mask = prefix.new_zeros((1, 1) + tuple(allowed.shape)).masked_fill_(
        ~allowed, torch.finfo(prefix.dtype).min)
    cached = isinstance(head, ServingCtxKVC1Head)
    held = cached and getattr(head, '_c1_prefix_key', None) is sequence
    unique, inverse = torch.unique(ids, dim=0, return_inverse=True)
    rows = [head(token_embeddings=drafter.embed_tokens(row[None].to(device)), h_prev=prefix[:, -1:],
                 prefix_h=prefix, prefix_position_ids=torch.arange(length, device=device)[None],
                 query_position_ids=(length + query)[None], attention_mask=mask,
                 shared_prefix_projection=True, **({'prefix_key': sequence} if cached else {}))[0]
            for row in unique]
    if cached and not held:
        head._c1_prefix_key = None
    return torch.stack(rows)[inverse.to(device)]


def _lab_block(adapter_head, embed_tokens, sequence, *, length, tail_counts, ids, slots, pack, exact):
    """One lab call's adapter rows and the decoder's tail: ``adapter_block`` and the staged tail, or
    (E04_LAB_EXACT) ``exact(ids)`` over the exact prefix and no tail.  ``(None, None)`` when the
    staged call has no rows."""
    if exact is not None:
        return exact(ids), None
    block = adapter_block(adapter_head, embed_tokens, sequence, length=length, tail_counts=tail_counts,
                          ids=ids, slots=slots, pack=pack)
    return (None, None) if block is None else (block, sequence.staged.predicted_hidden[0])


@torch.inference_mode()
def shadow_ar_decode(adapter_head, embed_tokens, decoder, sequence, *, length, tail_counts, first,
                     lab, pack, count, exact=None):
    """E04_LAB_SHADOW_AR=count, before the J0 pass: returns the J0 input with slots 1..count the
    shadow's greedy decode, pinned.

    Step i = 1..count: the adapter over every row's block slots 0..i-1 (``adapter_block`` with i
    slots: the anchor at length + tail_counts[r], then the i-1 tokens decoded so far), the
    refresh-context decoder over those i slots, and the argmax of its row i-1 (the distribution
    for slot i) becomes slot i.  Rows advance together, one slot per step (count sequential
    adapter + decoder calls, each over all rows; a row never reads another).  Slots 1..count
    are pinned, so every pass keeps them and the passes draft slots count+1.. only.
    ``exact`` (E04_LAB_EXACT, default None): the exact round's adapter (``exact_adapter_rows``
    over every slot, of which step i reads slots 0..i-1) and no tail (``tail_counts`` zero).
    """
    if decoder is None:
        raise ValueError('E04_LAB_SHADOW_AR needs the anchor decoder (JETSPEC_ANCHOR=decoder)')
    rows = first.shape[0]
    count = min(int(count), first.shape[1] - 1)
    ids = first.clone()
    for step in range(1, count + 1):
        block, tail = _lab_block(adapter_head, embed_tokens, sequence, length=length,
                                 tail_counts=tail_counts, ids=ids, slots=step, pack=pack, exact=exact)
        if block is None:
            lab.shadow = dict(skipped=1)
            return first
        logits, _, _ = refresh_decoder_logits(
            decoder, sequence, length=length, tail_hidden=tail,
            tail_counts=tail_counts, block_hidden=block,
            slot_counts=torch.full((rows,), step, dtype=torch.long))
        # packed row-major: row r's slots 0..step-1, and slot step-1's row predicts slot step
        ids[:, step] = logits.reshape(rows, step, -1)[:, step - 1].argmax(dim=-1).to(ids.device)
    if count:
        lab.pins[:, 1:count + 1] = ids[:, 1:count + 1]
    lab.shadow = dict(ids=ids[:, 1:count + 1].tolist())
    return ids


@torch.inference_mode()
def slot1_pre_j0(adapter_head, embed_tokens, decoder, sequence, *, length, tail_counts, first, lab,
                 pack, exact=None):
    """E04_LAB_SWAP=slot1_pre_j0, before the J0 pass: returns the J0 input with slot 1 swapped.

    The round-start decoder rows cover the in-flight block (its own slots, and the new rows'
    anchors); none is a distribution for a new row's slot 1, which needs the adapter's state at
    the row's anchor.  So each row's anchor state is predicted first: the adapter over block
    slot 0 only (``adapter_block`` with one slot: the anchor at length + tail_counts[r], reading
    the prefix and the row's tail slots), then the refresh-context decoder over that one slot;
    its row 0 is the distribution for slot 1.  Where p(top-1) >= margin under that row (the slot1
    margin, ``swap_margin('slot1_pre_j0')``), the top-1 replaces the J0 input's slot 1 (the
    random-context fill; there is no draft yet) and is
    pinned: J0 conditions on it and every later pass's slot 1 keeps it.  A row whose top-1 is
    already the fill token is pinned too (recorded with old == new); a row whose slot 1 is
    already pinned (E04_LAB_SHADOW_AR) is left alone.
    ``exact`` (E04_LAB_EXACT, default None): the exact round's adapter; the anchor state is slot 0
    of ``exact_adapter_rows`` (the correction at its position over the exact prefix, reading no
    other slot) and the decoder reads no tail (``tail_counts`` zero).
    """
    if decoder is None:
        raise ValueError('E04_LAB_SWAP=slot1_pre_j0 needs the anchor decoder (JETSPEC_ANCHOR=decoder)')
    block, tail = _lab_block(adapter_head, embed_tokens, sequence, length=length,
                             tail_counts=tail_counts, ids=first, slots=1, pack=pack, exact=exact)
    if block is None:
        lab.slot1 = dict(skipped=1)
        return first
    rows = first.shape[0]
    logits, _, _ = refresh_decoder_logits(
        decoder, sequence, length=length, tail_hidden=tail,
        tail_counts=tail_counts, block_hidden=block,
        slot_counts=torch.ones(rows, dtype=torch.long))
    probabilities = torch.softmax(logits.float(), dim=-1)
    top, new = probabilities.max(dim=-1)
    free = (lab.pins[:, 1] < 0).to(top.device)
    chosen = ((top >= swap_margin('slot1_pre_j0')) & free).nonzero()[:, 0].to(first.device)
    new = new.to(first.device)
    swapped = first.clone()
    swapped[chosen, 1] = new[chosen]
    lab.slot1 = dict(p_top1=top.tolist(), top1=new.tolist(),
                     swap_new=_per_row(rows, chosen, new[chosen], first.device))
    lab.pin(0, chosen, torch.ones_like(chosen), first[chosen, 1], new[chosen],
            top=_top(probabilities, chosen))
    return swapped










@torch.inference_mode()
def pre_verify(adapter_head, embed_tokens, decoder, sequence, *, length, tail_counts, tokens, lab,
               pack, exact=None):
    """E04_LAB_SWAP=pre_verify, after the final pass and before the rows leave for verify: returns
    ``tokens`` [R, B] (the rows' final blocks [anchor, d1..]) with at most one token swapped per row
    per round (one round unless E04_LAB_PREVERIFY_ROUNDS).

    The adapter runs over each row's final block (``adapter_block``, slots 0..B-2) and the
    refresh-context decoder over those states: decoder row j-1 is the distribution for slot j.
    The swapped slot is the first unpinned j >= 1 with argmax(row j-1) != d_j and
    p(argmax) - p(d_j) >= margin under that row (``swap_margin('pre_verify')``); d_j becomes the
    argmax and is pinned.  No pass runs after it.
    E04_LAB_PREVERIFY_ROUNDS=n: rounds 2..n repeat this on the rows that swapped in the round before
    (batched, one full adapter call over each such row's edited block per round), and a row stops
    when no slot qualifies.  The adapter is causal within the block, so the decoder rows before a
    swapped slot read as before and the next swap lands after it.  ``lab.final`` holds round 1's
    F and swap_slot / _old / _new (today's fields); with n > 1 it adds per row ``swap_count`` and
    ``swap_slots`` (each pre_verify swap's slot, in round order).  Each swap's ``lab.swaps`` entry
    carries the decoder's top-5 at its slot in its round's read (``top_ids`` / ``top_p``).
    ``exact`` (E04_LAB_EXACT, default None): the exact round's adapter (``exact_adapter_rows``) and
    no tail (``tail_counts`` zero).
    E04_LAB_RECORD_TOPK=n: ``lab.final`` adds ``keys`` (``lab.keys``), ``draft_top_ids`` /
    ``draft_top_p`` (``lab.final_top``: each row's final pass, the pass the block ``tokens`` froze at;
    None when the round ran no pass) and ``reads``, one entry per read in round order: ``rows`` (the
    rows it read) and its decoder's top-n at slots 1..B-1 (``dec_top_ids`` / ``dec_top_p``), each read
    in one host transfer (round 1's with the final pass's).
    E04_LAB_SWAP_RULE_FILE: every round's swap is the rule's (``_pre_verify_read``): the first unpinned
    slot j whose x* differs from d_j with a gain of at least bias(j, k), k the row's draft k-hat over the
    final block (``lab.rule_top``'s top-1 probabilities, the refresh's rule); x* is written and pinned, and
    its ``lab.swaps`` entry carries the ``rule`` record.  ``lab.final`` keeps today's fields (its F counts
    the decoder argmax's agreements).
    """
    if decoder is None:
        raise ValueError('E04_LAB_SWAP=pre_verify needs the anchor decoder (JETSPEC_ANCHOR=decoder)')
    rows, width = tokens.shape
    ids = tokens.clamp_min(0)
    margin = swap_margin('pre_verify')
    topk = record_topk()
    rule = swap_rule()
    if rule is not None:
        # E04_LAB_SWAP_RULE_FILE: the rule, each row's final-pass drafter top-n and its draft k-hat over the block
        if lab.rule_top is None:
            raise ValueError('E04_LAB_SWAP_RULE_FILE: pre_verify reads each row\'s final-pass drafter top-n '
                             '(packed_refinement\'s lab_top), and none was kept')
        from jetspec.engines.jetspec_plus_plus.draft.live_rows import refresh_khat_anchor0; from jetspec.engines.jetspec_plus_plus.draft.live_rows import refresh_khat_threshold
        rule = (rule, lab.rule_top, refresh_khat_anchor0(lab.rule_top[1][..., 0], refresh_khat_threshold()))
    read = _pre_verify_read(adapter_head, embed_tokens, decoder, sequence, length=length,
                            tail_counts=tail_counts, ids=ids, pins=lab.pins, pack=pack, exact=exact,
                            margin=margin, topk=topk, rule=rule)
    if read is None:
        lab.final = dict(skipped=1)
        return tokens
    lab.reads += 1  # the exact round's record: s_reads
    argmax, hit, top, ranked, written, values = read
    drafts = ids[:, 1:]
    chosen = hit.any(dim=1).nonzero()[:, 0]
    first = hit.long().argmax(dim=1)[chosen]
    swapped = tokens.clone()
    slot = first + 1
    new = written[chosen, first]
    lab.final = dict(F=torch.cumprod((argmax == drafts).long(), dim=1).sum(dim=1).tolist(),
                     swap_slot=_per_row(rows, chosen, slot, ids.device),
                     swap_old=_per_row(rows, chosen, tokens[chosen, slot], ids.device),
                     swap_new=_per_row(rows, chosen, new, ids.device))
    if topk is not None:
        # E04_LAB_RECORD_TOPK: this read's decoder top-n and the final pass's drafter top-n, one transfer
        final = () if lab.final_top is None else lab.final_top
        home = _home([*ranked, *final])
        lab.final.update(keys=lab.keys, draft_top_ids=home[2].long().tolist() if final else None,
                         draft_top_p=home[3].tolist() if final else None,
                         reads=[dict(rows=list(range(rows)), dec_top_ids=home[0].long().tolist(),
                                     dec_top_p=home[1].tolist())])
    if len(chosen):
        swapped[chosen, slot] = new
        lab.pin('pre_verify', chosen, slot, tokens[chosen, slot], new, top=_top_at(top, chosen, first),
                **({} if values is None else _rule_pins(values, rule, chosen, first, new)))
    rounds = preverify_rounds()
    if rounds == 1:
        return swapped
    # E04_LAB_PREVERIFY_ROUNDS: rounds 2..n over the rows that swapped in the round before
    slots = [[] for _ in range(rows)]
    for row, value in zip(chosen.tolist(), slot.tolist()):
        slots[row].append(value)
    active = chosen
    for index in range(2, rounds + 1):
        if not len(active):
            break
        read_rule = None if rule is None else _rule_rows(rule, active)
        read = _pre_verify_read(adapter_head, embed_tokens, decoder, sequence, length=length,
                                tail_counts=tail_counts[active.to(tail_counts.device)],
                                ids=swapped[active].clamp_min(0), pins=lab.pins[active.to(lab.pins.device)],
                                pack=pack, exact=exact, margin=margin, topk=topk, rule=read_rule)
        if read is None:
            lab.final['skipped_round'] = index
            break
        lab.reads += 1
        argmax, hit, top, ranked, written, values = read
        if topk is not None:
            home = _home([active, *ranked])  # E04_LAB_RECORD_TOPK: the rows read and their decoder top-n
            lab.final['reads'].append(dict(rows=home[0].long().tolist(), dec_top_ids=home[1].long().tolist(),
                                           dec_top_p=home[2].tolist()))
        again = hit.any(dim=1).nonzero()[:, 0]
        first = hit.long().argmax(dim=1)[again]
        active, slot, new = active[again], first + 1, written[again, first]
        lab.pin('pre_verify', active, slot, swapped[active, slot], new, top=_top_at(top, again, first),
                **({} if values is None else _rule_pins(values, read_rule, again, first, new)))
        swapped[active, slot] = new
        for row, value in zip(active.tolist(), slot.tolist()):
            slots[row].append(value)
    lab.final.update(swap_count=[len(values) for values in slots], swap_slots=slots)
    return swapped


@torch.inference_mode()
def refresh_hook(decoder, sequence, *, length, tail_counts, ids, khat, block, lab, pass_index,
                 exact=False, draft_logits=None):
    """The lab at one adapter refresh: the cutoff, the first_wrong_refresh swap and the record.

    ``khat`` is the draft cutoff (``refresh_khat_anchor0``, after the optional cap), ``block`` the
    adapter's refreshed rows [R, B, H] (``refresh_cached`` / ``refresh_window`` output) for the
    block ``ids`` [R, B].  With the anchor decoder loaded, the refresh-context decoder runs over
    each row's refreshed slots and its first disagreement F is recorded at every refresh (the
    cutoff uses it under decoder_f).  Returns ``(khat_used, ids)``: khat_used <= khat on every
    row, and ids with the swapped token (unchanged unless first_wrong_refresh swapped a row; its
    margin is ``swap_margin('first_wrong_refresh')``).
    E04_LAB_SWAP_KHAT_MAX / _SLOT_MAX hold back a swap the margin passes on a row whose ``khat``
    is >= the bound / at a slot past the bound; with the swap on, each refresh records how many
    each held back (``gated_khat``, ``gated_slot``).  ``exact`` (E04_LAB_EXACT): the exact round's
    refresh, ``block`` from ``exact_adapter_rows`` and no tail (``tail_counts`` zero).
    Every refresh records ``block_ids``, the block ``ids`` it read (before its swap); a swap's
    ``lab.swaps`` entry carries the decoder's top-5 at its slot (``top_ids`` / ``top_p``).
    ``draft_logits`` [R, B, V] (the refresh's callers pass it; default None) are the logits of the pass
    whose block ``ids`` the refresh read.  E04_LAB_RECORD_TOPK=n: the refresh's record adds ``keys``
    (``lab.keys``) and both experts' top-n at every slot of that block, before its swap
    (``_refresh_topk``: the drafter's from ``draft_logits``, the decoder's from this refresh's read, null
    without one), in one host transfer.
    E04_LAB_SWAP_RULE_FILE: the first_wrong_refresh swap is the rule's (``_rule_refresh``, from this read and
    ``draft_logits``): on a row whose slot khat_used + 1 is free, the first slot F from there on that is
    unpinned and qualifies takes x*, pinned unless the rule's pin is 0; the gates act at F, and its
    ``lab.swaps`` entry adds the ``rule`` record.
    """
    cutoff, swaps = refresh_cutoff(), swap_modes()
    rows, width = ids.shape
    counts = khat.clamp(max=width - 2) + 1
    block_ids = ids.tolist()
    info = dict(pass_index=int(pass_index),
                extra=int(lab.recipe_passes is not None and int(pass_index) >= lab.recipe_passes),
                khat_draft=khat.tolist())
    khat_used = khat
    logits = f = None
    if (cutoff == 'decoder_f' or 'first_wrong_refresh' in swaps) and decoder is None:
        raise ValueError('E04_LAB_REFRESH_CUTOFF=decoder_f and E04_LAB_SWAP=first_wrong_refresh '
                         'need the anchor decoder (JETSPEC_ANCHOR=decoder)')
    if decoder is not None:
        logits, row_of, slot_of = refresh_decoder_logits(
            decoder, sequence, length=length,
            tail_hidden=None if exact else sequence.staged.predicted_hidden[0],
            tail_counts=tail_counts, block_hidden=block, slot_counts=counts)
        argmax = dense_rows(logits.argmax(dim=-1), row_of, slot_of, rows, width, -1)
        f = rule_f(argmax, ids, counts)
        if cutoff == 'decoder_f':
            khat_used = torch.minimum(khat, f.to(khat.device))
    info['F'] = None if f is None else f.tolist()
    if cutoff == 'oracle':
        if live_sequence(sequence):
            reference = reference_tokens().full_sequence(sequence)
            true_k = oracle_true_k(ids, reference, length + tail_counts.to(ids.device))
            info['true_k'] = true_k.tolist()
            khat_used = torch.minimum(khat, true_k.to(khat.device))
        else:
            info['oracle_skipped'] = 1  # a warm-up state has no reference: the draft cutoff
    info['khat_used'] = khat_used.tolist()
    chosen = torch.zeros(0, dtype=torch.long, device=ids.device)
    slot = old = new = chosen
    read_ids = ids  # the block this refresh read (E04_LAB_PIN_AGREED pins its agreed run)
    if 'first_wrong_refresh' in swaps:
        fill = after_swap_fill()
        if fill != 'keep':
            lab.filled = torch.zeros(ids.shape, dtype=torch.bool, device=ids.device)
        khat_max, slot_max = swap_khat_max(), swap_slot_max()
        gated_khat = gated_slot = 0
        target = khat_used.to(ids.device) + 1
        pinned = lab.pins.gather(1, target.clamp(max=width - 1)[:, None])[:, 0] >= 0
        eligible = (target <= width - 1) & ~pinned
        candidates = eligible.nonzero()[:, 0]
        if len(candidates):
            # decoder row khat_used is block slot khat_used's output: the distribution for slot
            # khat_used + 1; it was computed since khat_used <= min(khat, width - 2)
            packed = {(r, s): i for i, (r, s) in enumerate(zip(row_of.tolist(), slot_of.tolist()))}
            rule = swap_rule()
            if rule is None:
                where = target[candidates]
            else:
                # E04_LAB_SWAP_RULE_FILE: the first slot from khat_used + 1 on that qualifies under the rule
                where, best, apply, values, rule_khat = _rule_refresh(
                    rule, logits, row_of, slot_of, ids=ids, khat=khat, khat_used=khat_used, pins=lab.pins,
                    draft_logits=draft_logits, candidates=candidates)
                where = where.to(ids.device)
            index = torch.tensor([packed[(r, s)] for r, s in zip(
                candidates.tolist(), (where - 1).tolist())], dtype=torch.long)
            current = ids[candidates, where]
            if rule is None:
                best, apply = margin_swap(logits[index.to(logits.device)], current.to(logits.device),
                                          swap_margin('first_wrong_refresh'), exclude_current=True)
            apply = apply.to(ids.device)
            if khat_max is not None or slot_max is not None:
                # E04_LAB_SWAP_KHAT_MAX / _SLOT_MAX: count, then drop, the swaps each gate holds back
                keep = torch.ones_like(apply)
                if khat_max is not None:
                    below = khat.to(ids.device)[candidates] < khat_max
                    gated_khat = int((apply & ~below).sum())
                    keep &= below
                if slot_max is not None:
                    inside = where <= slot_max
                    gated_slot = int((apply & ~inside).sum())
                    keep &= inside
                apply = apply & keep
            chosen = candidates[apply]
            if len(chosen):
                slot, old, new = where[apply], current[apply], best.to(ids.device)[apply]
                ids = ids.clone()
                ids[chosen, slot] = new
                lab.pin(pass_index, chosen, slot, old, new,
                        top=_top(torch.softmax(logits[index.to(logits.device)].float(), dim=-1),
                                 apply.nonzero()[:, 0]),
                        **({} if rule is None else dict(pinned=bool(rule.pin), rule=_rule_records(
                            values[apply.to(values.device)], rule_khat[apply.to(rule_khat.device)], new,
                            rule.pin))))
                if fill != 'keep':
                    # E04_LAB_AFTER_SWAP_FILL: the next pass reads the filler past each swapped slot
                    ids = _after_swap_fill(fill, ids, chosen, slot, lab, sequence=sequence,
                                           pass_index=pass_index, draft_logits=draft_logits,
                                           decoder_argmax=argmax if fill == 'decoder' else None)
        info.update(gated_khat=gated_khat, gated_slot=gated_slot)
        if pin_agreed():
            # E04_LAB_PIN_AGREED (decoder_f): slots 1..khat_used hold the tokens J0 and the decoder agreed on
            info['agreed_pins'] = _pin_agreed(lab, read_ids, khat_used)
    info.update(swap_slot=_per_row(rows, chosen, slot, ids.device),
                swap_old=_per_row(rows, chosen, old, ids.device),
                swap_new=_per_row(rows, chosen, new, ids.device))
    info['block_ids'] = block_ids
    topk = record_topk()
    if topk is not None:
        # E04_LAB_RECORD_TOPK: both experts' top-n at every slot of the block this refresh read
        info.update(_refresh_topk(topk, lab.keys, draft_logits,
                                  None if logits is None else (logits, row_of, slot_of),
                                  rows=rows, width=width))
    lab.refreshes.append(info)
    return khat_used, ids


# --- tl10: the after-swap fill and the agreed-run pins ----------------------------------------------

def _fill_generator(*parts):
    """E04_LAB_AFTER_SWAP_FILL=noise: a CPU generator seeded from ``parts`` (the knob's seed, the request, the
    round, the exact flag, the pass and the row), so a read draws the same noise every time."""
    digest = hashlib.sha256(':'.join(str(int(part)) for part in parts).encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], 'little') & ((1 << 63) - 1))


def _after_swap_fill(kind, ids, rows, slots, lab, *, sequence, pass_index, draft_logits, decoder_argmax=None):
    """E04_LAB_AFTER_SWAP_FILL at one first_wrong_refresh refresh: on each swapped row ``rows[i]`` (swap slot
    ``slots[i]``), every unpinned slot after the swap, slots[i] + 1..W, takes the filler in the next pass's input
    ``ids`` [R, B]; the slots stay unpinned, so that pass's argmax replaces them.  ``mask``: the drafter's mask token;
    ``noise``: token ids uniform over the drafter's vocabulary (``draft_logits``' last dim) less the mask token,
    one draw per block slot from ``_fill_generator(E04_LAB_FILL_SEED, seq id, round, exact, pass, row)``;
    ``context``: the round's J0 random-context draws at those slots (``LabRound.fill_source``).  The filled slots
    go to ``lab.filled`` (the mask guard's exemption) and each swap's record gains ``fill`` (kind, slots).
    ``decoder`` (tl11): ``decoder_argmax`` [R, B] is the refresh read's dense argmax (block slot j - 1's row predicts
    slot j, -1 past the read); only the unpinned slots after the swap that the read covers are filled, with
    ``changed`` (the filled slots whose token differs from ``ids``') in the record; they are not pinned or written."""
    if kind == 'decoder':
        return _decoder_fill(ids, rows, slots, lab, decoder_argmax)
    if lab.fill_mask_token is None:
        raise ValueError('E04_LAB_AFTER_SWAP_FILL: the round gave the lab no fill source (LabRound.fill_source)')
    width = ids.shape[1]
    mask = lab.fill_mask_token
    if kind == 'noise':
        if draft_logits is None:
            raise ValueError('E04_LAB_AFTER_SWAP_FILL=noise draws over the drafter\'s vocabulary: the refresh needs '
                             'the pass\'s logits')
        vocab = int(draft_logits.shape[-1])
        seed, seq_id = fill_seed(), lab_seq_id(sequence)
        round_id = int(getattr(sequence, '_jetspec_candidate_round', -1))
    ids = ids.clone()
    records = []
    for row, slot in zip(rows.tolist(), slots.tolist()):
        free = (lab.pins[row, slot + 1:].to(ids.device) < 0).nonzero()[:, 0] + slot + 1
        if kind == 'mask':
            values = torch.full((len(free),), mask, dtype=ids.dtype, device=ids.device)
        elif kind == 'context':
            values = lab.fill_context[row].to(ids.device)[free]
        else:
            generator = _fill_generator(seed, seq_id, round_id, int(lab.exact), pass_index, row)
            inside = 0 <= mask < vocab
            draws = torch.randint(0, vocab - 1 if inside else vocab, (width,), generator=generator)
            if inside:
                draws = draws + (draws >= mask).long()  # the mask token is never drawn
            values = draws.to(device=ids.device, dtype=ids.dtype)[free]
        ids[row, free] = values
        lab.filled[row, free.to(lab.filled.device)] = True
        records.append(dict(kind=kind, slots=free.tolist()))
    for swap, record in zip(lab.swaps[-len(records):] if records else [], records):
        swap['fill'] = record
    return ids


def _decoder_fill(ids, rows, slots, lab, decoder_argmax):
    """E04_LAB_AFTER_SWAP_FILL=decoder (``_after_swap_fill``): the read's argmax at each covered unpinned slot."""
    if decoder_argmax is None:
        raise ValueError('E04_LAB_AFTER_SWAP_FILL=decoder writes the refresh\'s decoder argmax: the refresh read none')
    decoder_argmax = decoder_argmax.to(ids.device)
    ids = ids.clone()
    records = []
    for row, slot in zip(rows.tolist(), slots.tolist()):
        free = (lab.pins[row, slot + 1:].to(ids.device) < 0).nonzero()[:, 0] + slot + 1
        values = decoder_argmax[row, free - 1]
        free, values = free[values >= 0], values[values >= 0]  # the slots the decoder read covers
        changed = free[values != ids[row, free]]
        ids[row, free] = values.to(ids.dtype)
        lab.filled[row, free.to(lab.filled.device)] = True
        records.append(dict(kind='decoder', slots=free.tolist(), changed=changed.tolist()))
    for swap, record in zip(lab.swaps[-len(records):] if records else [], records):
        swap['fill'] = record
    return ids


def _pin_agreed(lab, ids, khat_used):
    """E04_LAB_PIN_AGREED at one first_wrong_refresh refresh under decoder_f: pin slots 1..khat_used of every row
    to the block ``ids`` the refresh read (khat_used <= F: J0's tokens the decoder agrees with), slots already
    pinned left to their pins.  A row with no disagreement in its window pins its whole agreed window the same way.
    The pins are kept apart (``lab.agreed``): they are not decoder-written slots.  Returns each row's newly
    pinned slots (the refresh record's ``agreed_pins``)."""
    device = lab.pins.device
    slots = torch.arange(ids.shape[1], device=device)[None, :]
    agree = (slots >= 1) & (slots <= khat_used.to(device)[:, None]) & (lab.pins < 0)
    lab.pins[agree] = ids.to(device)[agree]
    lab.agreed = agree if lab.agreed is None else lab.agreed | agree
    return [row.nonzero()[:, 0].tolist() for row in agree]


def mask_outside_fill(ids, mask_token, lab):
    """The mask guard of a pass whose input the after-swap fill wrote: whether ``ids`` holds ``mask_token`` at a
    slot the last refresh did not fill (a device flag); the exempted slots are cleared after the read."""
    hit = ((ids == int(mask_token)) & ~lab.filled.to(ids.device)).any()
    lab.filled = None
    return hit


# --- tl06: decoder-written slots, the k score there and the rows after the joint rule -------------



def written_terms(block, *, after_miss):
    """E04_LAB_KSCORE_WRITTEN: the k-score terms of the verified ``block``'s decoder-written slots
    (``mode``; ``mask`` [W + 1] bool over block slots 0..W; ``p`` [W + 1], the draft model's p of each
    written token, or None), or None for today's score: the knob unset, a round after a miss (its k
    score is not run), or a block with no written slot (drafted without the lab, or never pinned)."""
    mode = kscore_written()
    if mode is None or after_miss or block is None:
        return None
    mask = getattr(block, 'e04_lab_written', None)
    if mask is None or not bool(mask.any()):
        return None
    p = getattr(block, 'e04_lab_written_p', None)
    if mode != 'disagree' and p is None:
        raise ValueError('E04_LAB_KSCORE_WRITTEN=draft_p / both: the verified block carries no '
                         'written-token probability')
    return SimpleNamespace(mode=mode, mask=mask.reshape(-1), p=None if p is None else p.reshape(-1))






def joint_rows(block, prediction, k_probabilities, keys, requested, *, after_miss):
    """The lab after the joint row rule: E04_LAB_ROWS_MIN's floor and E04_LAB_SWAP_ROWS' rows appended
    to the rule's ``keys`` [m, 2] (``requested``: the count of pairs that passed tau), and the round's
    ``rows`` record (``LabRound.rows``).

    The score table is ``live_rows.e04_rowrule_joint``'s arithmetic on the host copy of the decoder
    offers: scored row k (0..W; k = 0 alone after a miss) offers 32 anchors a with post-exclusion
    probabilities p; P_T(a | k) = p^(1/T) renormalized in the pool and s = P(k) P_T(a | k) on the
    2^-32 grid, P(k) the 16-slot BLEND mass ``k_probabilities`` the rule read (P(0) = 1 after a miss).
    Floor (E04_LAB_ROWS_MIN=n): when ``requested`` < n, the table's pairs with s > 0, in the rule's
    order (-s, k, pool rank), are appended, skipping published ones, until min(n, JETSPEC_ROWS_CAP)
    rows are published.  Swap rows (E04_LAB_SWAP_ROWS=1, not after a miss): each decoder-written slot
    s of the verified ``block`` (``e04_lab_written``), in slot order, appends (s - 1, row s - 1's
    first offer) unless it is published, while fewer than JETSPEC_ROWS_CAP rows are published.
    Returns ``(keys, record)``, keys the rule's own tensor when nothing was added; None when there is
    no rule to extend (E04_ROWRULE unset, or no decoder offers: the session warm-up's synthetic
    round)."""
    if os.environ.get('E04_ROWRULE') != 'joint' or prediction is None:
        return None
    from jetspec.engines.jetspec_plus_plus.models.heads.fast import dataset_entry; from jetspec.engines.jetspec_plus_plus.models.heads.fast import read_thresholds
    from jetspec.engines.jetspec_plus_plus.draft.live_rows import E04_ROWRULE_POOL
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    rule = dataset_entry(read_thresholds(os.environ['JETSPEC_HEADS_DIR'])['e04_rowrule_joint'],
                         os.environ['JETSPEC_DATASET'])
    from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled as eval_enabled
    if eval_enabled():
        from jetspec.engines.jetspec_plus_plus.draft.pass_labels import resolve_tau
        tau = resolve_tau(rule['tau'], str(block.kpred_pass))
    else:
        tau = float(rule['tau'][str(block.kpred_pass)]['value'])
    temperature = float(rule['temperature']['value'])
    offsets = [0] if after_miss else list(range(draft_width() + 1))
    offer_ids, offer_p = prediction.offers.ids, prediction.offers.probabilities
    ids = torch.stack([torch.as_tensor(offer_ids[k], dtype=torch.long).reshape(-1).cpu() for k in offsets])
    probabilities = torch.stack([torch.as_tensor(offer_p[k]).double().reshape(-1).cpu() for k in offsets])
    if ids.shape[1] != E04_ROWRULE_POOL:
        raise ValueError('E04_LAB_ROWS_MIN / E04_LAB_SWAP_ROWS read the joint rule\'s 32 offers per row')
    mass = (torch.ones(1, dtype=torch.float64) if after_miss else
            torch.as_tensor(k_probabilities).double().reshape(-1).cpu()[1:])
    from jetspec.engines.jetspec_plus_plus.draft.method.rows import joint_scores
    score = joint_scores(mass, probabilities, temperature)
    table = {(k, a): s for k, row_ids, row_scores in zip(offsets, ids.tolist(), score.tolist())
             for a, s in zip(row_ids, row_scores)}
    published = [tuple(pair) for pair in keys.reshape(-1, 2).tolist()]
    present = set(published)
    capacity = int(os.environ.get('JETSPEC_ROWS_CAP', '128'))
    floor, floor_added = rows_min(), 0
    if floor is not None and int(requested) < floor:
        flat, flat_ids = score.reshape(-1).tolist(), ids.reshape(-1).tolist()
        for index in torch.argsort(score.reshape(-1), descending=True, stable=True).tolist():
            if len(published) >= min(floor, capacity) or not flat[index] > 0.0:
                break
            pair = (offsets[index // E04_ROWRULE_POOL], flat_ids[index])
            if pair not in present:
                published.append(pair)
                present.add(pair)
                floor_added += 1
    swap_added = []
    if swap_rows() and not after_miss:
        written = getattr(block, 'e04_lab_written', None)
        for slot in ([] if written is None else written.reshape(-1).nonzero()[:, 0].tolist()):
            if slot < 1:
                continue  # slot 0 holds the anchor: never written
            if len(published) >= capacity:
                break
            pair = (slot - 1, int(ids[slot - 1, 0]))
            if pair not in present:
                published.append(pair)
                present.add(pair)
                swap_added.append(pair)
    record = dict(tau=tau, temperature=temperature, rule_mass=mass.tolist(),
                  pair_scores=[table.get(pair) for pair in published], rows_min=floor,
                  floor_added=floor_added, swap_rows=[list(pair) for pair in swap_added])
    if len(published) == len(keys):
        return keys, record
    return keys.new_tensor(published, dtype=torch.long).reshape(-1, 2), record


# --- tl07: the seeded exact round -----------------------------------------------------------------




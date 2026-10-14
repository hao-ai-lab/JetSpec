"""F=1 batching of the live kth protocol; wire rows remain guess + 14 drafts."""
from __future__ import annotations

import os
import math
import json
import time
import functools
import weakref
import numpy as np
import torch
from collections import OrderedDict
from types import SimpleNamespace
from jetspec.engines.jetspec_plus_plus.draft.pass_labels import completed_label; from jetspec.engines.jetspec_plus_plus.draft.pass_labels import resolve_tau

from jetspec.engines.jetspec_plus_plus.draft.drafter import Block; from jetspec.engines.jetspec_plus_plus.draft.drafter import DraftSuperseded; from jetspec.engines.jetspec_plus_plus.draft.drafter import DraftForwardPass; from jetspec.engines.jetspec_plus_plus.draft.drafter import build_random_context_block_token_ids; from jetspec.engines.jetspec_plus_plus.draft.drafter import run_refinement_protocol
from jetspec.engines.jetspec_plus_plus.draft.rows import RowDraftHead; from jetspec.engines.jetspec_plus_plus.draft.rows import RowLayout; from jetspec.engines.jetspec_plus_plus.draft.rows import BucketRowDraftHead; from jetspec.engines.jetspec_plus_plus.draft.rows import FastBucketRowDraftHead; from jetspec.engines.jetspec_plus_plus.draft.rows import project_row_logits; from jetspec.engines.jetspec_plus_plus.draft.rows import device_tensor; from jetspec.engines.jetspec_plus_plus.draft.rows import rows_ctx_incr; from jetspec.engines.jetspec_plus_plus.draft.rows import rows_ctx_early; from jetspec.engines.jetspec_plus_plus.draft.rows import EarlyContextKV


# W324_MARKS=1 (default off): host timestamps at named points of the draft round, read
# and cleared by an offline trace tool to place the device gaps
HOST_MARKS = [] if os.environ.get('W324_MARKS') == '1' else None


def host_mark(label):
    if HOST_MARKS is not None:
        HOST_MARKS.append((label, time.perf_counter()))






class _ContextSlots(dict):
    """The graphed rows' per-sequence context K/V shelf (a dict its slot finalizers can
    reference weakly)."""


def batch_draft_graph():
    """JETSPEC_BATCH_DRAFT_GRAPH=1 (default off): the N>1 draft chain runs CUDA graphs captured
    per batch bucket (batch_draft_bucket), with inactive rows padded up to the bucket. N=1
    never reads it."""
    value = os.environ.get('JETSPEC_BATCH_DRAFT_GRAPH', '0')
    if value not in ('0', '1'):
        raise ValueError('JETSPEC_BATCH_DRAFT_GRAPH must be 0 or 1')
    if value == '1' and os.environ.get('JETSPEC_ADAPTER_REFRESH', 'off') != 'off':
        raise ValueError('JETSPEC_BATCH_DRAFT_GRAPH requires JETSPEC_ADAPTER_REFRESH=off')
    return value == '1'


def batch_draft_front():
    """JETSPEC_BATCH_DRAFT_GRAPH_FRONT=1 (default off, requires JETSPEC_BATCH_DRAFT_GRAPH=1): the
    front of the N>1 chain captured too -- the adapter's window query over per-sequence prefix
    K/V slots, the anchor decoder's window, and its prefix append. A stand-in until the N=1
    static front bodies land; it replaces JETSPEC_C1_MULTI_SLOT."""
    value = os.environ.get('JETSPEC_BATCH_DRAFT_GRAPH_FRONT', '0')
    if value not in ('0', '1'):
        raise ValueError('JETSPEC_BATCH_DRAFT_GRAPH_FRONT must be 0 or 1')
    if value == '1':
        if not batch_draft_graph():
            raise ValueError('JETSPEC_BATCH_DRAFT_GRAPH_FRONT requires JETSPEC_BATCH_DRAFT_GRAPH=1')
        if os.environ.get('JETSPEC_C1_MULTI_SLOT', '0') == '1':
            raise ValueError('JETSPEC_BATCH_DRAFT_GRAPH_FRONT replaces JETSPEC_C1_MULTI_SLOT; unset it')
    return value == '1'




def e04_lean():
    """JETSPEC_E04_LEAN=1 (default off): the e04 joint-rule round without its dead k
    selection and keyed rows, and the BLEND k score with one trip home (exact)."""
    value = os.environ.get('JETSPEC_E04_LEAN', '0')
    if value not in ('0', '1'):
        raise ValueError('JETSPEC_E04_LEAN must be 0 or 1')
    return value == '1'


def rows_assembly():
    """JETSPEC_ROWS_ASSEMBLY=1 (default off): the round's k score, row rule and keyed rows in
    one rule-agnostic device stage with one trip home (heads/assembly.py); exact against the
    host path, which stays the reference and the fallback."""
    value = os.environ.get('JETSPEC_ROWS_ASSEMBLY', '0')
    if value not in ('0', '1'):
        raise ValueError('JETSPEC_ROWS_ASSEMBLY must be 0 or 1')
    return value == '1'


def rows_static():
    """JETSPEC_ROWS_STATIC=1 (default off; graphed regime with JETSPEC_ROWS_ASSEMBLY=1): the
    round's rows run in the fixed row-cap bucket.  The bucket's context K/V is issued before the
    anchor decoder, the geometry is rewritten from the assembly stage's device keys, and the
    stage's one trip home is read after the passes are queued, so no host read sits between the
    decoder and the passes.  The published keys are the stage's; a round whose offers need the
    exact full-row recompute reruns on the host path."""
    value = os.environ.get('JETSPEC_ROWS_STATIC', '0')
    if value not in ('0', '1'):
        raise ValueError('JETSPEC_ROWS_STATIC must be 0 or 1')
    return value == '1'


def rows_static_chain():
    """Enable the U2 fixed-cap row path, including refresh-capable preissue."""
    return _flag01('JETSPEC_ROWS_STATIC_CHAIN')


def rows_defer_home():
    """Run a fixed-cap graphed bucket before reading the selected rows."""
    value = os.environ.get('JETSPEC_ROWS_DEFER_HOME', '1')
    if value not in ('0', '1'):
        raise ValueError('JETSPEC_ROWS_DEFER_HOME must be 0 or 1')
    return value == '1'




def rows_rule_cap():
    """JETSPEC_ROWS_RULE_CAP (default: JETSPEC_ROWS_CAP): the joint rule's row cap on a
    proposal round, below the wire capacity JETSPEC_ROWS_CAP that the admission round's 16
    guessed rows (which do not go through the rule) still need."""
    capacity = int(os.environ.get('JETSPEC_ROWS_CAP', '128'))
    cap = int(os.environ.get('JETSPEC_ROWS_RULE_CAP', str(capacity)))
    from jetspec.engines.jetspec_plus_plus.serving.presets.budget import budget_type
    cap = budget_type().row_cap(cap, capacity)
    from jetspec.engines.jetspec_plus_plus.draft.serve_policy import rows_min_capacity
    cap = rows_min_capacity(cap)
    if cap > capacity:
        raise ValueError('rows_min bucket exceeds JETSPEC_ROWS_CAP')
    return cap


class _StaticFallback(Exception):
    """JETSPEC_ROWS_STATIC: the stage's deferred home read asked for the host path."""


def _flag01(name):
    value = os.environ.get(name, '0')
    if value not in ('0', '1'):
        raise ValueError(f'{name} must be 0 or 1')
    return value == '1'


def miss_redraft():
    """JETSPEC_MISS_REDRAFT=1 (default off): on a round whose pre-drafted rows miss the true
    (k, correction), both ranks see the miss at the round's request; the draft rank drafts one
    fresh row on the critical path (exact context, anchored on the committed correction) and the
    verify rank verifies that block instead of a one-token AR step."""
    return _flag01('JETSPEC_MISS_REDRAFT')


def miss_redraft_force():
    """JETSPEC_MISS_REDRAFT_FORCE=1 (measurement only): every round after the first is served by
    the fallback, the serial floor of the fallback path."""
    return _flag01('JETSPEC_MISS_REDRAFT_FORCE')


def miss_redraft_anchor():
    """JETSPEC_MISS_REDRAFT_ANCHOR: ``none`` (default; the drafter predicts the token after the
    correction) or ``decoder`` (the anchor decoder's after-miss top-1 is pinned there)."""
    value = os.environ.get('JETSPEC_MISS_REDRAFT_ANCHOR', 'none')
    if value not in ('none', 'decoder'):
        raise ValueError('JETSPEC_MISS_REDRAFT_ANCHOR must be none or decoder')
    return value


def miss_redraft_passes():
    """JETSPEC_MISS_REDRAFT_PASSES: refinement passes of the fallback row (default 1 = J0)."""
    passes = int(os.environ.get('JETSPEC_MISS_REDRAFT_PASSES', '1'))
    if not 1 <= passes <= 4:
        raise ValueError('JETSPEC_MISS_REDRAFT_PASSES must be in 1..4')
    return passes


def draft_deadline():
    """JETSPEC_DRAFT_DEADLINE=1 (default off): a round-ahead draft publishes its last completed
    pass instead of running past the verify's running median (minus
    JETSPEC_DRAFT_DEADLINE_MARGIN_MS, default 1.0) from the round start the draft rank sees."""
    from jetspec.engines.jetspec_plus_plus.serving.presets.budget import budget_type
    return budget_type().deadline_enabled(os.environ)


from jetspec.engines.jetspec_plus_plus.serving.presets.budget import DeadlineClock; from jetspec.engines.jetspec_plus_plus.serving.presets.budget import DraftDeadlineMiss


def deadline_miss_blocks(fallback, *, device):
    """A keyed empty candidate; the next request commits zero drafts before redrafting."""
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    capacity = int(os.environ.get('JETSPEC_ROWS_CAP', '128'))
    wire = torch.full((capacity, draft_width() + 1), -1, dtype=torch.long, device=device)
    keys = torch.full((capacity, 2), -1, dtype=torch.long, device=device)
    blocks = PackedBlocks(wire, None, torch.empty((0, 4), dtype=torch.long, device=device),
                          [], row_keys=keys)
    blocks.fallback = fallback
    return blocks


def unique_decoder_hypotheses(values):
    """Return valid decoder prefix lengths once, preserving wire order."""
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    return list(dict.fromkeys(int(k) for k in values if 0 <= int(k) <= draft_width()))


def decoder_k_selection(k_hypotheses, k_probabilities, *, predictor, after_miss):
    """Return decoder wire k values and their unnormalized selection masses."""
    if after_miss and os.environ.get('JETSPEC_KSET_AFTER_MISS') == 'zero':
        return [0], torch.ones(1, dtype=torch.float64)
    probabilities = torch.as_tensor(k_probabilities, device='cpu').double()
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    ks = (list(range(draft_width() + 1)) if os.environ.get('JETSPEC_ROW_SELECT') == 'budget'
          else unique_decoder_hypotheses(k_hypotheses))
    shift = int(bool(predictor) and os.environ.get('JETSPEC_KPRED_FRAME') == 'wire')
    indices = torch.tensor([k + shift for k in ks], device=probabilities.device,
                           dtype=torch.long)
    return ks, probabilities.index_select(0, indices)






def flat_bucket_host_fast(*, host_fast, graphed):
    """Avoid per-length mask banks when retaining all 24 pow2 graph buckets."""
    return bool(host_fast) and not (
        graphed and os.environ.get('JETSPEC_G_ROW_BUCKETS') == 'pow2')


def cached_flat_bucket(buckets, key, make, *, graphs, limit=None):
    """Retain the graph's pow2 row and prefix shapes across repeated rounds."""
    limit = max(27, limit or 0) if graphs and os.environ.get('JETSPEC_G_ROW_BUCKETS') == 'pow2' else 2
    if key not in buckets:
        if len(buckets) >= limit:
            buckets.popitem(last=False)
        buckets[key] = make()
    buckets.move_to_end(key)
    return buckets[key]


def graphed_flat_warmup_keys(*, max_model_len, row_cap, anchor0, refresh=False):
    """Finite flat graph shapes admitted by this model length and row cap."""
    widths = []
    width = 1
    while width <= row_cap:
        widths.append(width)
        width *= 2
    return [('flat', 1, width, prefix) + (('refresh',) if refresh else ()) +
            (('anchor0',) if anchor0 else ())
            for prefix in range(256, max_model_len + 255, 256)
            for width in widths]


def graphed_admission_warmup_keys(max_model_len):
    """The initial/catchup proposal's finite N1 row and prefix graph shapes."""
    return [(1, width, prefix)
            for prefix in range(256, max_model_len + 255, 256)
            for width in (1, 2, 4, 8, 16)]


def adapter_refresh_mode():
    """JETSPEC_ADAPTER_REFRESH: ``off`` (default) or ``after_j0`` -- one adapter
    refresh of every drafted row's block between the first and second pass.
    ``after_j0_j1`` refreshes again between the second and third pass, from the
    pass-1 draft and logits."""
    mode = os.environ.get('JETSPEC_ADAPTER_REFRESH', 'off')
    if mode not in ('off', 'after_j0', 'after_j0_j1'):
        raise ValueError('JETSPEC_ADAPTER_REFRESH must be off, after_j0 or after_j0_j1')
    return mode


def refresh_pass_indices(mode):
    """The pass indices whose start runs a refresh from the previous pass."""
    return {'after_j0': (1,), 'after_j0_j1': (1, 2)}.get(mode, ())


def refresh_khat_threshold(again=False):
    from jetspec.engines.jetspec_plus_plus.draft.method.refresh import refresh_khat_threshold as threshold
    selected = {name: os.environ[name] for name in
                ('JETSPEC_REFRESH_KHAT_T', 'JETSPEC_REFRESH_KHAT_T2') if name in os.environ}
    return threshold(again, environ=selected)


def refresh_khat_cap(again=False):
    from jetspec.engines.jetspec_plus_plus.draft.method.refresh import refresh_khat_cap as cap
    selected = {name: os.environ[name] for name in
                ('JETSPEC_REFRESH_KHAT_CAP2',) if name in os.environ}
    from jetspec.engines.jetspec_plus_plus.serving.presets.budget import budget_type
    return budget_type().refresh_cap(cap(again, environ=selected))


def refresh_khat(p_top1, threshold):
    from jetspec.engines.jetspec_plus_plus.draft.method.refresh import refresh_cutoff
    return refresh_cutoff(p_top1[:, 1:], threshold, start=2, final=15)


def row_layout_anchor0():
    """JETSPEC_ROW_LAYOUT=anchor0: a keyed row (k, a) puts the last accepted
    token's predicted hidden into the context tail (k + 1 tail rows) and starts the
    block at the pinned anchor, DFlash's shape: input [a, 15 slots], 15 draft outputs.
    Unset or ``kth``: the trained kth layout [last accepted, a, 14 slots]."""
    value = os.environ.get('JETSPEC_ROW_LAYOUT', 'kth')
    if value not in ('kth', 'anchor0'):
        raise ValueError('JETSPEC_ROW_LAYOUT must be kth or anchor0')
    return value == 'anchor0'


def wide_rows():
    """Use every anchor0 block slot as a draft row."""
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    return draft_width() in (3, 7, 15, 31)


def refresh_khat_anchor0(p_drafts, threshold):
    from jetspec.engines.jetspec_plus_plus.draft.method.refresh import refresh_cutoff
    return refresh_cutoff(p_drafts, threshold, final=15)


def refresh_khat_anchor0_width31(p_drafts, threshold):
    from jetspec.engines.jetspec_plus_plus.draft.method.refresh import refresh_cutoff
    if p_drafts.shape[1] != 31:
        raise ValueError('width-31 refresh requires all 31 draft probabilities')
    return refresh_cutoff(p_drafts, threshold, final=31)


def refresh_khat_anchor0_short(p_drafts, threshold):
    from jetspec.engines.jetspec_plus_plus.draft.method.refresh import refresh_cutoff
    if p_drafts.shape[1] not in (3, 7):
        raise ValueError('short refresh requires 1, 3, or 7 draft probabilities')
    return refresh_cutoff(p_drafts, threshold)


def refresh_token_bucket(tokens):
    """JETSPEC_REFRESH_PACK=2: the padded token count of a packed refresh call (a
    multiple of 16: few Triton rotary compiles, and no cuBLAS shape cliff between)."""
    return -(-tokens // 16) * 16


def refresh_pack_mode():
    pack = os.environ.get('JETSPEC_REFRESH_PACK', '0')
    if pack not in ('0', '1', '2'):
        raise ValueError('JETSPEC_REFRESH_PACK must be 0, 1 or 2')
    if os.environ.get('JETSPEC_ROWS_FUSED_GRAPH', '0') == '1':
        if pack == '1':
            raise ValueError('JETSPEC_ROWS_FUSED_GRAPH requires cached packed refresh')
        if (os.environ.get('JETSPEC_REFRESH_LEAN', '0') != '1'
                or os.environ.get('JETSPEC_C1_NORM') != 'wide'
                or os.environ.get('JETSPEC_C1_KVCACHE', '0') != '1'):
            raise ValueError('JETSPEC_ROWS_FUSED_GRAPH refresh requires LEAN=1, C1_NORM=wide, C1_KVCACHE=1')
        return '2'
    return pack




def kpred_forward_off():
    """Whether decoder-BLEND serving omits the legacy standalone k predictor."""
    value = os.environ.get("JETSPEC_LEGACY_KPRED_FORWARD", "0")
    if value not in ("0", "1"):
        raise ValueError("JETSPEC_LEGACY_KPRED_FORWARD must be 0 or 1")
    if value == "1":
        if os.environ.get("JETSPEC_KSET") != "predictor":
            raise ValueError(
                "JETSPEC_LEGACY_KPRED_FORWARD=1 requires JETSPEC_KSET=predictor")
        return False
    if os.environ.get("E04_KSCORE") is None:
        if os.environ.get("JETSPEC_KSET") == "predictor":
            raise ValueError(
                "JETSPEC_KSET=predictor is legacy-only; set JETSPEC_LEGACY_KPRED_FORWARD=1")
        return False
    required = {
        "E04_KSCORE": "blend_dec",
        "E04_ROWRULE": "joint",
        "JETSPEC_KSET_AFTER_MISS": "zero",
    }
    mismatched = [f"{name}={expected}" for name, expected in required.items()
                  if os.environ.get(name) != expected]
    if mismatched:
        raise ValueError(
            "JETSPEC_LEGACY_KPRED_FORWARD=0 requires " + ", ".join(mismatched))
    return True


def decoder_blend_kset():
    """Whether the decoder-BLEND dynamic k policy is active, with or without its legacy head."""
    off = kpred_forward_off()
    return os.environ.get("JETSPEC_KSET") == "predictor" or off


def served_kpred_pass(passes_run):
    """Map refinement passes to the threshold label without running the head."""
    from jetspec.engines.jetspec_plus_plus.models.heads.kpred import pass_name
    from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled as eval_enabled
    lab_on = any(name.startswith('E04_LAB_') for name in os.environ)
    from jetspec.engines.jetspec_plus_plus.draft.pass_labels import eval_completed_label
    label = (eval_completed_label(passes_run) if eval_enabled() else
             pass_name(passes_run) if os.environ.get('JETSPEC_CHAIN_GRAPH', '0') == 'seg' or lab_on
             else completed_label(passes_run))
    import jetspec.engines.jetspec_plus_plus.draft.run_manifest as run_manifest
    run_manifest.observe("kpred_pass", (int(passes_run), label))
    return label


def load_kpredictor(drafter):
    """Load the legacy network only when its forward is part of the served path."""
    off = kpred_forward_off()
    enabled = os.environ.get("JETSPEC_KSET") == "predictor" and not off
    import jetspec.engines.jetspec_plus_plus.draft.run_manifest as run_manifest
    run_manifest.record_value("kpred_forward", "legacy" if enabled else "not_loaded")
    if not enabled:
        return None
    print("DEPRECATION JETSPEC_LEGACY_KPRED_FORWARD=1: the standalone k predictor is legacy-only",
          flush=True)
    from jetspec.engines.jetspec_plus_plus.models.heads.kpred import KPredictor
    return KPredictor.load(
        os.environ["JETSPEC_HEADS_DIR"], drafter.embed_tokens, device=drafter.device)


class KthRowsDrafter:
    def __init__(self, drafter, *, backend='flashinfer', optimized=False, graphs=False, branches=1, f_graphs=False):
        from jetspec.engines.jetspec_plus_plus.draft.rows import check_rows_fused_regime
        check_rows_fused_regime()
        self.optimized = optimized
        self.host_fast = os.environ.get("JETSPEC_HOST_FAST", "0") == "1"
        self._host_stage_cache = {}
        self.graphs = graphs
        from jetspec.engines.jetspec_plus_plus.draft.tau_lab import check_regime as e04_lab_check_regime
        e04_lab_check_regime(graphed=graphs)  # a no-op while every E04_LAB_* name is unset
        self.f_graphs = f_graphs
        self.backend = backend
        self.buckets = OrderedDict()
        if branches not in (1, 2, 3, 5):
            raise ValueError("branches must be 1, 2, 3, or 5")
        from jetspec.engines.jetspec_plus_plus.wire.protocol import anchor_mode
        self.branches = 1 if anchor_mode() == 'oracle' else branches
        self.drafter = drafter
        self.predictor = load_kpredictor(drafter)
        self.decoder = None
        if anchor_mode() == 'decoder':
            from jetspec.engines.jetspec_plus_plus.models.heads.decoder import AnchorDecoder
            self.decoder = AnchorDecoder.load(os.environ['JETSPEC_HEADS_DIR'], drafter)
        if self.predictor is not None or self.decoder is not None:
            import jetspec.engines.jetspec_plus_plus.draft.run_manifest as run_manifest
            run_manifest.record_value('heads_dir', os.path.realpath(os.environ['JETSPEC_HEADS_DIR']))
            run_manifest.record_value('heads_dataset', os.environ.get('JETSPEC_DATASET'))
        # Initial live admission has one KV group per hypothesis; FlashInfer's
        # split-K scratch exceeds the bench's two-MiB allowance at 512 rows.
        self.row_head = RowDraftHead(drafter.head, backend=backend,
            workspace_bytes_per_row=4 * 1024 * 1024)
        if os.environ.get('JETSPEC_FUSE','off')!='off':
            from jetspec.engines.jetspec_plus_plus.kernels.fuse import configure_rows
            configure_rows(self.row_head,os.environ['JETSPEC_FUSE'])

    @torch.inference_mode()
    def prewarm_flat_graph_buckets(self, max_model_len):
        """Capture every bounded pow2 flat bucket before measured decode."""
        if (not self.graphs or os.environ.get('JETSPEC_G_ROW_BUCKETS') != 'pow2'
                or self.decoder is None or self.branches != 1):
            return 0
        from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled as eval_enabled
        if eval_enabled() and os.environ.get('JETSPEC_CHAIN_GRAPH', '0') == 'seg':
            # Session warmup already exercises each exact prompt admission.
            # Decode and redraft use the prewarmed chain entries. Retaining the
            # unused flat/admission/fallback shelves can exhaust a single card
            # before the long-read chain keys are captured.
            return 0
        d = self.drafter
        anchor0, wide = row_layout_anchor0(), wide_rows()
        block_width = d.block_size if wide else 16
        refresh = adapter_refresh_mode() != 'off'
        keys = graphed_flat_warmup_keys(max_model_len=max_model_len,
                                        row_cap=rows_rule_cap(), anchor0=anchor0,
                                        refresh=refresh)
        admission_keys = graphed_admission_warmup_keys(max_model_len)
        prefix_buckets = range(256, max_model_len + 255, 256)
        if (os.environ.get('JETSPEC_CHAIN_GRAPH', '0') == 'seg'
                and os.environ.get('JETSPEC_CHAIN_WARM_SINGLE_KEY', '0') == '1'):
            # Steady decode and miss redraft use the chain's ceiling entry.
            # Flat graphs are needed only for synthetic warm-up and admission;
            # retaining every intervening prefix wastes private CUDA pools.
            read = json.loads(os.environ['JETSPEC_CHAIN_WARM_READ'])
            lengths = [32, *read['prompt_lengths'], *json.loads(
                os.environ.get('JETSPEC_WARMUP_PREFIX_LENGTHS', '[]'))]
            prefix_buckets = sorted({((int(length) + 255) // 256) * 256
                                     for length in lengths})
            keys = [key for key in keys if key[3] in prefix_buckets]
            admission_keys = [key for key in admission_keys if key[2] in prefix_buckets]
        self._graph_bucket_limit = max(27, len(keys) + len(admission_keys))
        width = d.head.fc.in_features
        for key in keys:
            _, _, count, prefix_bucket, *_ = key
            bucket = cached_flat_bucket(self.buckets, key, lambda: BucketPass(
                d, N=1, H=count, prefix_bucket=prefix_bucket, backend=self.backend,
                flat=True, graphs=True, refresh=refresh, anchor0=anchor0,
                wide=wide), graphs=True, limit=self._graph_bucket_limit)
            prefix = torch.zeros(prefix_bucket, width, device=d.device, dtype=d.dtype)
            tail = torch.zeros(1, block_width if wide else 15, width, device=d.device, dtype=d.dtype)
            bucket.head.update([prefix], tail, [int(anchor0)] * count, mark=True)
            bucket(torch.zeros(count, block_width, device=d.device, dtype=torch.long))
        for key in admission_keys:
            _, count, prefix_bucket = key
            bucket = cached_flat_bucket(self.buckets, key, lambda: BucketPass(
                d, N=1, H=count, prefix_bucket=prefix_bucket, backend=self.backend,
                host_fast=self.host_fast, graphs=True), graphs=True,
                limit=self._graph_bucket_limit)
            prefix = torch.zeros(prefix_bucket, width, device=d.device, dtype=d.dtype)
            tail = torch.zeros(1, 15, width, device=d.device, dtype=d.dtype)
            bucket.head.update([prefix], tail, [0] * count, mark=True,
                **({'prefix_key': self} if self.host_fast else {}))
            bucket(torch.zeros(count, 16, device=d.device, dtype=torch.long))
        fallback = self.__dict__.setdefault('_fallback_buckets', OrderedDict())
        fallback_keys = [(count, prefix_bucket, 'anchor0')
                         for prefix_bucket in prefix_buckets
                         for count in (1, 16)]
        for key in fallback_keys:
            count, prefix_bucket, _ = key
            bucket = cached_flat_bucket(fallback, key, lambda: BucketPass(
                d, N=1, H=count, prefix_bucket=prefix_bucket, backend=self.backend,
                host_fast=False, flat=True, graphs=True, anchor0=True, wide=wide), graphs=True)
            prefix = torch.zeros(prefix_bucket, width, device=d.device, dtype=d.dtype)
            tail = torch.zeros(1, block_width if wide else 15, width, device=d.device, dtype=d.dtype)
            bucket.head.update([prefix], tail, [0] * count, mark=True)
            bucket(torch.zeros(count, block_width, device=d.device, dtype=torch.long))
        if (rows_assembly() and os.environ.get('JETSPEC_ROW_STAGE_GRAPH', '0') == '1'
                and d.device.type == 'cuda'):
            from jetspec.engines.jetspec_plus_plus.models.heads.assembly import RowStage
            from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
            RowStage.prewarm_graph(d.device, rows=draft_width() + 1,
                                   width=E04_ROWRULE_POOL, cap=rows_rule_cap())
        torch.cuda.synchronize()
        print(f'W409_GRAPH_PREWARM flat={len(keys)} admission={len(admission_keys)} '
              f'fallback={len(fallback_keys)} cached={len(self.buckets)} '
              f'fallback_cached={len(fallback)}', flush=True)
        return len(keys) + len(admission_keys) + len(fallback_keys)

    def _attach_predictor_metrics(self, outputs, sequences, drafted_rows):
        if self.predictor is None:
            return outputs
        from jetspec.engines.jetspec_plus_plus.models.heads.kpred import k0_in_lac; from jetspec.engines.jetspec_plus_plus.models.heads.kpred import k15_in_lac
        kpred_calls, kpred_total_ms = self.predictor.consume_metrics()
        for blocks, sequence, drafted in zip(outputs, sequences, drafted_rows):
            pending = sequence.pending_block
            ready = pending is not None and pending.k_probabilities is not None
            wire_frame = os.environ.get('JETSPEC_KPRED_FRAME') == 'wire'
            dropped = int(ready and not wire_frame and k15_in_lac(
                pending, root=os.environ['JETSPEC_HEADS_DIR'],
                dataset=os.environ['JETSPEC_DATASET'],
                coverage=os.environ.get('JETSPEC_KPRED_COVERAGE', '0.99')))
            dropped0 = int(ready and wire_frame and k0_in_lac(
                pending, root=os.environ['JETSPEC_HEADS_DIR'],
                dataset=os.environ['JETSPEC_DATASET'],
                coverage=os.environ.get('JETSPEC_KPRED_COVERAGE', '0.99')))
            live = len(blocks.hypotheses)
            blocks.head_metrics = dict(
                decoder_ready=False, decoder_ms=0.0, decoder_total_ms=0.0,
                decoder_prefix_ms=0.0, decoder_select_ms=0.0,
                kpred_ms=float(pending.kpred_ms) if ready else 0.0,
                kpred_calls=kpred_calls, kpred_total_ms=kpred_total_ms,
                lac_fallbacks=0,
                kpred_pass=pending.kpred_pass if ready else None,
                rows_drafted=int(drafted), rows_published=live,
                rows_truncated=0, rows_requested=live,
                k_set_size=live, k15_dropped=dropped,
                kpred_frame_dropped0=dropped0,
                k_set=list(map(int, blocks.hypotheses)),
                anchor_set_sizes=[1 if k in blocks.hypotheses else 0 for k in range(15)])
        return outputs

    @torch.inference_mode()
    def stage(self, sequences, blocks):
        """One shared-prefix adapter call with a masked, padded sequence batch."""
        if len(sequences) > 1 and os.environ.get('JETSPEC_DEVICE_PUBLISH', '0') == '1':
            from copy import copy
            staged_blocks = []
            for block in blocks:
                staged = copy(block)
                object.__setattr__(staged, 'draft_ids', block.draft_ids.clone())
                staged_blocks.append(staged)
            blocks = staged_blocks
        if self.optimized:
            if len(sequences) > 1 and batch_draft_front():
                self._stage_batch_graph(sequences, blocks)
                return
            if len(sequences) > 1 and os.environ.get('JETSPEC_C1_KVCACHE', '0') == '1':
                if os.environ.get('JETSPEC_C1_MULTI_SLOT', '0') == '1':
                    self._stage_packed(sequences, blocks, multi_cached=True)
                    return
                # The reference C1 forward keeps each row's prefix/window mask
                # independent.  The last sequence still fills the single live
                # cache used by its following refresh, as in the serial path.
                if len(sequences) > 2:
                    self._stage_packed(sequences[:-1], blocks[:-1], batched_uncached=True)
                    self._stage_packed(sequences[-1:], blocks[-1:])
                else:
                    for sequence, block in zip(sequences, blocks):
                        self._stage_packed([sequence], [block])
                return
            if self.host_fast and len(sequences) == 1:
                return self._stage_host_fast(sequences, blocks)
            return self._stage_packed(sequences, blocks)
        d = self.drafter
        prefixes = [s.exact_hidden(device=d.device, dtype=d.dtype) for s in sequences]
        lengths = [p.shape[1] for p in prefixes]
        count, capacity, query = len(prefixes), max(lengths), 15
        prefix = prefixes[0].new_zeros(count, capacity, prefixes[0].shape[-1])
        previous = torch.cat([p[:, -1:] for p in prefixes])
        allowed = torch.zeros(count, 1, query, capacity + query, dtype=torch.bool, device=d.device)
        query_positions = torch.empty(count, query, dtype=torch.long, device=d.device)
        ids = []
        for row, (s, b, p, length) in enumerate(zip(sequences, blocks, prefixes, lengths)):
            if b.draft_ids.numel() != 14:
                raise ValueError('live kth staging requires 14 draft ids')
            prefix[row, :length] = p[0]
            allowed[row, 0, :, :length] = True
            allowed[row, 0, :, capacity:] = torch.eye(query, dtype=torch.bool, device=d.device)
            query_positions[row] = torch.arange(length, length + query, device=d.device)
            ids.append(torch.cat([s.committed_ids[:, -1:].to(d.device), b.draft_ids.reshape(1, -1).to(d.device)], 1))
        ids = torch.cat(ids)
        mask = prefix.new_zeros(allowed.shape).masked_fill(~allowed, torch.finfo(d.dtype).min)
        predicted = d.adapter.head(token_embeddings=d.embed_tokens(ids), h_prev=previous,
            prefix_h=prefix, prefix_position_ids=torch.arange(capacity, device=d.device)[None].expand(count,-1),
            query_position_ids=query_positions, attention_mask=mask, shared_prefix_projection=True)
        for row, (s, b, length) in enumerate(zip(sequences, blocks, lengths)):
            s.staged.stage(start_position=length, token_ids=ids[row:row+1], predicted_hidden=predicted[row:row+1])
            s.pending_block = b

    @torch.inference_mode()
    def propose(self, sequences, *, k_hypotheses, passes, initial=False, catchup=False, abort_if=None, oracle_predictions=None):
        """One T-module forward per pass, preserving preceding-token pinning.

        Initial proposals have no predicted tail and identical k inputs, so each
        initial hypothesis uses an independent KV group. Following rounds share
        each sequence's exact prefix and staged tail across all its hypotheses.
        """
        posterior = posterior_mode()
        dynamic_k = posterior or decoder_blend_kset()
        requested_ks = list(k_hypotheses)
        if self.drafter.correction_head is not None and (
                self.branches > 1 or not self.optimized or oracle_predictions is not None):
            raise ValueError('the correction head serves only optimized F=1 rows')
        if posterior:
            # Initial admission uses a uniform retained set across prefilled prompts.
            if self.branches not in (1, 5) or (len(sequences) != 1 and not initial):
                raise ValueError("posterior serving requires N=1 and F=1 or F=5")
            if self.branches == 5 and self.f_graphs:
                raise ValueError("posterior F5 requires F graphs off")
            if not self.optimized or (not self.graphs and self.decoder is None):
                raise ValueError("posterior serving requires fixed-capacity optimized graph rows")
            if initial and oracle_predictions is None:
                k_hypotheses = list(range(16))
            elif self.branches == 5 and not requested_ks:
                k_hypotheses = [0]  # Real unpublished block for empty-set staging.
        if (self.decoder is not None and initial and
                os.environ.get('JETSPEC_ROW_SELECT') == 'budget'):
            capacity = int(os.environ.get('JETSPEC_ROWS_CAP', '128'))
            if capacity < 1:
                raise ValueError('JETSPEC_ROWS_CAP must be positive')
            from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
            k_hypotheses = list(range(min(draft_width() + 1, capacity)))
            requested_ks = list(k_hypotheses)
        if (self.decoder is not None and initial and not catchup and oracle_predictions is None
                and row_layout_anchor0()):
            # The rows after prefill know the correction (the prefill's own token), so
            # under the anchor0 layout they are the exact DFlash round, not the kth bootstrap
            if self.branches != 1 or not self.optimized or posterior:
                raise ValueError('anchor0 initial rows require F=1, optimized, non-posterior rows')
            capacity = int(os.environ.get('JETSPEC_ROWS_CAP', '128'))
            if capacity < 1:
                raise ValueError('JETSPEC_ROWS_CAP must be positive')
            ks = list(dict.fromkeys(int(k) for k in k_hypotheses))[:capacity]
            from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled as eval_enabled
            if eval_enabled():
                ks = ks[:int(os.environ.get('JETSPEC_EVAL_ROWS', '16'))]
            # every prefilled sequence gets its own exact rows (admission batches the prompts)
            # JETSPEC_BATCH_DRAFT_GRAPH (armed by prewarm_batch_graphs): the rows replay the
            # captured N=1 bucket of the miss redraft's shelf instead of an eager bucket
            admission = ({'graphs': True, 'admission': True}
                         if self.__dict__.get('_batch_admission_graphs', False) else {})
            output = [self._exact_anchor0_rows(sequence, ks=ks, passes=passes, **admission)
                      for sequence in sequences]
            self._key_guessed_output(output)
            return self._attach_predictor_metrics(output, sequences, [len(ks)] * len(sequences))
        if self.decoder is not None and not initial and not catchup and oracle_predictions is None:
            return self._propose_flat_bucketed(
                sequences, k_hypotheses=k_hypotheses, passes=passes, abort_if=abort_if)
        k_hypotheses = proposal_hypotheses(
            k_hypotheses, predictor=decoder_blend_kset(),
            decoder=self.decoder is not None)
        if (self.host_fast and self.optimized and self.graphs and self.f_graphs and
                not initial and not catchup and self.branches > 1 and
                len(sequences) == 1 and len(k_hypotheses) == 1):
            return self._propose_host_branches(sequences[0], k_hypotheses, passes)
        if self.optimized and self.graphs and not initial and not catchup and self.branches == 1 and oracle_predictions is None:
            return self._propose_bucketed(
                sequences, k_hypotheses=k_hypotheses, passes=passes,
                abort_if=abort_if,
                output_hypotheses=requested_ks if dynamic_k else None)
        d = self.drafter
        ks = list(map(int, k_hypotheses))
        if not ks or len(set(ks)) != len(ks) or any(k < 0 or k > 15 for k in ks):
            raise ValueError('live kth hypotheses must be unique within 0..15')
        contexts, prefix_lengths, ksets, first, anchors = [], [], [], [], []
        for seq in sequences:
            prefix = seq.exact_hidden(device=d.device, dtype=d.dtype)
            ids = seq.committed_ids.to(d.device)
            random_ids = build_random_context_block_token_ids(ids, block_size=16,
                seed=d.seed, prompt_index=seq.prompt_index, mode='kth').repeat(len(ks),1)
            marker = d.head.adapted_position_embedding.weight
            if catchup:
                if ks != [0] or seq.staged.predicted_hidden is None:
                    raise ValueError('known-correction catchup requires one-k and its staged hidden')
                context = prefix[0].clone() + marker[0]
                context[-1:] = seq.staged.predicted_hidden[0,:1] + marker[1]
                contexts.append(context)
                prefix_lengths.append(context.shape[0])
                ksets.append([1])
                anchor = ids[:,-1:]
            elif initial:
                context = prefix[0, :-1] + marker[0]
                for _ in ks:
                    contexts.append(context)
                    prefix_lengths.append(context.shape[0])
                    ksets.append([1])
                anchor = ids[:, -2:-1].expand(len(ks),1)
            else:
                if seq.pending_block is None or seq.staged.predicted_hidden is None:
                    raise RuntimeError('live kth proposal requires a staged pending block')
                predicted = seq.staged.predicted_hidden[0]
                # k=15 uses exactly the 15 staged positions. The reference's
                # duplicated sixteenth state serves validation only, not context.
                context = torch.cat([prefix[0] + marker[0], predicted[:max(ks)] + marker[1]])
                contexts.append(context)
                prefix_lengths.append(prefix.shape[1])
                ksets.append([k+1 for k in ks])
                drafts = seq.pending_block.draft_ids.to(d.device)
                anchor = torch.stack([ids[0,-1] if k == 0 else drafts[min(k-1,13)] for k in ks])[:,None]
            random_ids[:, :1] = anchor
            first.append(random_ids)
            anchors.append(anchor)
        layout = RowLayout(prefix_lengths, ksets, F=1, device=d.device, vectorized=self.optimized)
        self.row_head.prepare(layout, torch.cat(contexts))
        first = torch.cat(first)
        anchors = torch.cat(anchors)

        def forward(*, fill_tokens, pass_index):
            del pass_index
            ids = first if fill_tokens is None else torch.cat([anchors, fill_tokens],1)
            if bool((ids == int(d.head.mask_token_id)).any()):
                raise ValueError('Jacobi input must be mask-free')
            hidden = self.row_head(d.embed_tokens(ids))
            logits = project_row_logits(d.lm_head, hidden)[:,1:]
            return DraftForwardPass(logits=logits, input_tokens=ids[:,1:], hidden=hidden[:,1:])

        if self.branches > 1 or oracle_predictions is not None:
            rejected = torch.tensor([
                -1 if initial or catchup or k >= 14 else int(seq.pending_block.draft_ids[k])
                for seq in sequences for k in ks], device=d.device)
            output = self._propose_branches(forward, contexts, prefix_lengths, ksets,
                first, anchors, rejected, ks, len(sequences), passes,
                bucketed=self.f_graphs and not initial and not catchup, oracle_predictions=oracle_predictions)
            if posterior and self.branches == 5:
                from jetspec.engines.jetspec_plus_plus.wire.protocol import PosteriorBranches
                return [PosteriorBranches(groups, output_hypotheses=requested_ks) for groups in output]
            return output

        if self.optimized:
            output = packed_refinement(forward, anchors=anchors, hypotheses=ks, sequences=len(sequences),
                passes=passes, convergence_limit=d.convergence_limit, abort_if=abort_if,
                output_hypotheses=requested_ks if dynamic_k else None,
                fixed_wire_capacity=dynamic_wire_capacity(
                    posterior=posterior, predictor=self.predictor), predictor=self.predictor,
                posterior=posterior, correction_head=d.correction_head)
            if self.decoder is not None:
                self._key_guessed_output(output)
            return self._attach_predictor_metrics(
                output, sequences, [len(requested_ks)] * len(output))

        results = run_refinement_protocol(forward, batch_size=len(sequences)*len(ks),
            depth=15, max_passes=passes, convergence_limit=d.convergence_limit)
        output = []
        for n in range(len(sequences)):
            blocks = []
            for j,k in enumerate(ks):
                row = n*len(ks)+j
                result = results[row]
                blocks.append(Block(hypothesis=k, mode='kth', draft_ids=result.final_tokens[1:].detach(),
                    logits=result.logits[1:].detach().clone(), passes_run=result.passes_run, converged=result.converged,
                    anchor=int(anchors[row,0].item()), correction_guess=int(result.final_tokens[0].item())))
            output.append(blocks)
        return output

    @torch.inference_mode()
    def _exact_anchor0_rows(self, sequence, *, ks, passes, round_ahead=None,
                            graphs=None, admission=False):
        """JETSPEC_ROW_LAYOUT=anchor0, a known correction (the miss fallback, and the
        initial rows after prefill): the exact DFlash round. Context = the target's own taps for
        every committed token before the correction (no adapter tail, no refresh); block =
        [correction, 15 slots] at the correction's position, J0 slots drawn from the context as
        in training; the correction is pinned in slot 0 and drafts are slots 1..15 (1..14 with
        14 drafts). One row per k in ``ks``, each keyed (k, correction).

        E04_LAB_EXACT=1 (tau_lab.py, default off): the lab hooks act in this round as in the
        pipelined one, over the exact prefix with no tail (``_exact_lab_input``): SHADOW_AR and
        slot1_pre_j0 before J0, the refresh before the passes ``tau_lab.refresh_schedule`` gives
        for this round's pass count (``_exact_refresh``; the bucket then carries refresh slots),
        pre_verify on the final block, and a W280_RECORD line (``_exact_lab_record``).

        E04_LAB_SEED_EXACT=1 (with E04_LAB_EXACT, under JETSPEC_MISS_REDRAFT_FORCE=1): ``round_ahead``
        (default None) is the draft rank's (row, hit) for a redraft it requested, the row
        ``select_live_block`` chose among the round-ahead rows and whether the verdict matched it.  On a
        hit the round starts from that row's final block (``tau_lab.round_ahead_seed``: the correction
        pinned in slot 0, none of the row's lab pins) instead of the J0 initialization (the context
        draws, SHADOW_AR and slot1_pre_j0), and runs E04_LAB_SEED_PASSES=m passes from it on
        ``tau_lab.refresh_schedule``'s schedule for m passes (m = 0: none, ``_exact_zero_pass``).  A
        miss and the admission round are unchanged.
        ``admission`` seeds the batched context slot from this bucket (``_stash_admission_context``)."""
        from jetspec.engines.jetspec_plus_plus.models.heads.lac import row_bucket
        import jetspec.engines.jetspec_plus_plus.draft.tau_lab as e04_lab
        d = self.drafter
        prefix = sequence.exact_hidden(device=d.device, dtype=d.dtype)
        ids = sequence.committed_ids.to(d.device)
        if ids.shape[1] < 2 or prefix.shape[1] != ids.shape[1] - 1:
            raise RuntimeError('an exact row requires committed taps for all but the correction')
        n = len(ks)
        correction = ids[0, -1:]
        first = build_random_context_block_token_ids(ids, block_size=d.block_size, seed=d.seed,
            prompt_index=sequence.prompt_index, mode='kth').clone()
        first[:, :1] = correction
        exact = prefix[0]
        prefix_bucket = ((exact.shape[0] + 255) // 256) * 256
        # graphs=True (JETSPEC_BATCH_DRAFT_GRAPH's N>1 miss redraft): a captured bucket on its
        # own shelf, whatever the regime's graphs
        batch_graph = graphs is not None and graphs and not self.graphs
        graphs = self.graphs if graphs is None else graphs
        rows = row_bucket(n, cap=int(os.environ.get('JETSPEC_ROWS_CAP', '128')), graphed=graphs)
        # E04_LAB_EXACT (tau_lab.py, default off): an environment read while unset, and nothing
        # below reads the lab state then
        lab_round, lab_swaps, lab_shadow = None, (), 0
        refresh_at, total_passes, lab_fixed_passes, refresh = (), passes, False, False
        from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled as eval_enabled
        eval_mode = eval_enabled()
        if eval_mode:
            config = os.environ['JETSPEC_CHAIN_FIXED_CONFIG']
            refresh_at = tuple(range(1, int(config[-1]) + 1))
            total_passes, lab_fixed_passes, refresh = int(config[1]) + 1, True, True
        seed_row = seed = None  # E04_LAB_SEED_EXACT: the round-ahead hit row and its final block
        if e04_lab.exact_lab():
            e04_lab.check_regime(graphed=self.graphs)
            lab_swaps, lab_shadow = e04_lab.swap_modes(), e04_lab.shadow_ar()
            if self.decoder is None and (lab_swaps or lab_shadow or e04_lab.refresh_cutoff() == 'decoder_f'):
                raise ValueError('E04_LAB_EXACT: E04_LAB_SWAP, E04_LAB_SHADOW_AR and '
                                 'E04_LAB_REFRESH_CUTOFF=decoder_f need the anchor decoder '
                                 '(JETSPEC_ANCHOR=decoder)')
            seed_row, seed = e04_lab.round_ahead_seed(round_ahead, ks=ks, correction=int(correction[0]),
                                                      width=d.block_size, device=d.device)
            # E04_LAB_SEED_PASSES: a seeded round's own pass count; every other round runs the recipe's
            round_passes = passes if seed is None else e04_lab.seed_passes()
            refresh_mode = adapter_refresh_mode()
            refresh_at, total_passes, lab_fixed_passes = e04_lab.refresh_schedule(
                refresh_pass_indices(refresh_mode), round_passes)
            refresh = refresh_mode != 'off' and total_passes is not None and total_passes > 1
            lab_round = e04_lab.LabRound(n, d.block_size, d.device, recipe_passes=round_passes, exact=True)
            lab_round.seed_round(seed_row, seed, round_ahead, forced=miss_redraft_force())
            if e04_lab.record_topk() is not None:
                lab_round.keys = [[int(k), int(correction[0])] for k in ks]  # E04_LAB_RECORD_TOPK: the rows' keys
            if e04_lab.after_swap_fill() != 'keep':
                # E04_LAB_AFTER_SWAP_FILL: the mask token and the J0 context draws the fill reads
                lab_round.fill_source(d.head.mask_token_id, first.expand(n, -1))
            if seed is not None and not round_passes:
                return self._exact_zero_pass(sequence, seed_row, seed, ids, lab=lab_round, k=int(ks[0]),
                                             label_passes=passes)
        buckets = self.__dict__.setdefault('_batch_fallback_buckets' if batch_graph
                                           else '_fallback_buckets', OrderedDict())
        key = (rows, prefix_bucket, 'anchor0') + (('refresh',) if refresh else ())
        if key not in buckets:
            # a19: eager one-row buckets (the miss fallback) and the wider admission buckets
            # evict only their own kind, so an admission no longer evicts the fallback's bucket
            # and the next miss does not rebuild it on the critical path (~6 ms)
            kind = [old for old in buckets if graphs or (old[0] == 1) == (rows == 1)]
            # the N>1 shelf keeps every prefix bucket prewarm_batch_graphs captured (at least 6),
            # so a read never recaptures an evicted bucket
            limit = (max(6, self.__dict__.get('_batch_fallback_bucket_limit', 6)) if graphs and batch_graph
                     else 6 if graphs and os.environ.get('JETSPEC_G_ROW_BUCKETS') == 'pow2' else 2)
            if len(kind) >= limit:
                buckets.pop(kind[0])
            buckets[key] = BucketPass(d, N=1, H=rows, prefix_bucket=prefix_bucket,
                backend=self.backend, host_fast=False, flat=True, graphs=graphs,
                anchor0=True, **({'wide': True} if wide_rows() else {}),
                **({'refresh': True} if refresh else {}))
        bucket = buckets[key]
        buckets.move_to_end(key)
        bucket.head.update([exact], prefix.new_zeros(1, d.block_size if bucket.wide else 15, prefix.shape[-1]),
                           [0] * rows, mark=True)
        # E04_LAB_SEED_EXACT: a seeded round's first pass reads the seed, not the context draws
        block0 = (first if seed is None else seed).expand(rows, -1).clone()
        refresh_state = {}
        eval_swap = None
        launches = refreshes = 0
        if eval_mode:
            from jetspec.engines.jetspec_plus_plus.draft.eager_refresh_swap import prepare_eager_swap
            # Admission has an exact prefix and no predicted tail. Reuse the same
            # decoder swap and graph-backed draft bucket as the serving round.
            saved_hidden = sequence.staged.predicted_hidden
            sequence.staged.predicted_hidden = prefix.new_zeros(1, d.block_size, prefix.shape[-1])
            try:
                eval_swap = prepare_eager_swap(self.decoder, sequence, bucket, refresh_at)
            finally:
                sequence.staged.predicted_hidden = saved_hidden

        def forward(*, fill_tokens, pass_index, finished=None):
            nonlocal launches, refreshes
            if fill_tokens is None:
                block = block0
            else:
                fill = torch.cat([fill_tokens, fill_tokens[:1].expand(rows - n, -1)]) if rows > n else fill_tokens
                block = fill if bucket.wide else torch.cat([fill, bucket.extra_tokens], 1)
            if eval_mode and pass_index in refresh_at:
                from jetspec.engines.jetspec_plus_plus.draft.tau_lab import exact_adapter_rows
                refreshed = exact_adapter_rows(d, sequence, block)
                active = torch.ones(rows, dtype=torch.bool, device=d.device)
                full = torch.full((rows,), d.block_size - 1, dtype=torch.long, device=d.device)
                before = block.clone() if getattr(eval_swap, 'states_only', False) else None
                inject = eval_swap.apply(length=eval_swap.length, ids=block, khat=full,
                                         slot_counts=full, tail_counts=torch.zeros_like(full),
                                         active=active, block=refreshed, finished=finished)
                if before is not None:
                    eval_swap.eval_effects[0].add_(block.ne(before).sum())
                if os.environ.get('JETSPEC_REFRESH_INJECT_INCLUSIVE', '0') == '1':
                    inject = inject - 1
                bucket.head.inject_refresh(refreshed, inject, reset=pass_index > 1,
                                           fixed_layout=True)
                refreshes += 1
            if lab_round is not None:
                # E04_LAB_EXACT: the hooks act on the n published rows; padding rows repeat row 0
                block = self._exact_lab_input(
                    bucket, sequence, exact.shape[0], block[:n], rows=rows, pass_index=pass_index,
                    lab=lab_round, refresh_at=refresh_at if refresh else (), state=refresh_state)
                if lab_round.filled is not None and bool(e04_lab.mask_outside_fill(
                        block[:n], d.head.mask_token_id, lab_round)):
                    # E04_LAB_AFTER_SWAP_FILL: a mask token outside the slots the refresh filled
                    raise ValueError('Jacobi input must be mask-free')
            logits = bucket(block)[:n]
            launches += 1
            if refresh and pass_index + 1 in refresh_at:
                refresh_state['logits'] = logits
            return DraftForwardPass(logits=logits,
                                    input_tokens=block[:n] if bucket.wide else block[:n, :15],
                                    hidden=bucket.hidden[:n])

        keys = torch.tensor([[int(k), int(correction[0])] for k in ks], dtype=torch.long, device=d.device)
        output = packed_refinement(
            forward, anchors=ids[:, -2:-1].expand(n, 1), hypotheses=list(map(int, ks)), sequences=1,
            passes=total_passes, convergence_limit=d.convergence_limit,
            output_hypotheses=list(map(int, ks)), output_indices=list(range(n)), row_keys=keys,
            pinned_tokens=correction.expand(n), fixed_wire_capacity=int(os.environ.get('JETSPEC_ROWS_CAP', '128')),
            predictor=self.predictor, posterior=posterior_mode(), correction_head=d.correction_head,
            **({'refresh_swap': eval_swap} if eval_mode else {}),
            # E04_LAB_EXACT: the pins, pre_verify and the pass budget as in the pipelined round
            **({'lab_pins': lab_round.pins, 'lab_touched': lab_round.touched,
                'lab_pass': lab_round.observe_pass}
               if lab_shadow or {'slot1_pre_j0', 'first_wrong_refresh'} & set(lab_swaps) else {}),
            **({'final_edit': lambda tokens: e04_lab.pre_verify(
                None, None, self.decoder, sequence, length=exact.shape[0],
                tail_counts=torch.zeros(n, dtype=torch.long, device=d.device), tokens=tokens,
                lab=lab_round, pack=None, exact=functools.partial(e04_lab.exact_adapter_rows, d, sequence))}
               if 'pre_verify' in lab_swaps else {}),
            # E04_LAB_RECORD_TOPK: each row's final-pass top-n for pre_verify's record
            # (E04_LAB_SWAP_RULE_FILE: and for pre_verify's rule, the drafter's distribution it reads)
            **({'lab_top': lab_round.observe_top}
               if 'pre_verify' in lab_swaps and (e04_lab.record_topk() is not None
                                                  or e04_lab.swap_rule() is not None) else {}),
            **({'fixed_passes': True} if lab_fixed_passes else {}),
            # E04_LAB_SEED_EXACT: a seeded block's label is capped at the recipe's count too
            **({'label_passes': passes} if total_passes != passes or seed is not None else {}),
            # the published blocks carry their rows' decoder-written slots, as in the pipelined round
            **({'lab_written': lab_round.written_pins if e04_lab.pin_agreed() else lab_round.pins,
                'lab_written_p': e04_lab.kscore_written() in ('draft_p', 'both')}
               if lab_round is not None else {}))
        if lab_round is not None:
            # E04_LAB_SEED_EXACT: the record's passes run in this round; a seeded block's passes run adds
            # the seed row's own (the threshold label it reads stays capped at the recipe's count)
            lab_round.seed['passes'] = output[0].metadata[:, 2].tolist()
            if e04_lab.tl10_on():
                lab_round.passes_run = output[0].metadata[:, 2].tolist()  # tl10: the passes each row ran
            if seed_row is not None:
                output[0].metadata[:, 2] += int(seed_row.passes_run)
            self._exact_lab_record(sequence, lab_round, output[0], keys)
        if admission:
            self._stash_admission_context(sequence, bucket, exact.shape[0])
        if eval_mode:
            audit = dict(
                seq_id=getattr(sequence, '_jetspec_seq_id', -1), round=0, kind='proposal',
                j_passes=launches - 1, refreshes=refreshes, rows_published=n,
                expected_j=total_passes - 1, expected_r=len(refresh_at),
                expected_rows=int(os.environ.get('JETSPEC_EVAL_ROWS', '16')),
                refresh_khat_skips=0)
            if getattr(eval_swap, 'states_only', False):
                audit.update(refresh_mode='states_only', swaps=int(eval_swap.eval_effects[0].item()),
                             pins=int(eval_swap.eval_effects[1].item()))
            print('EVAL_ROUND ' + json.dumps(audit), flush=True)
        return output[0]

    @torch.inference_mode()
    def _exact_lab_input(self, bucket, sequence, length, ids, *, rows, pass_index, lab, refresh_at,
                         state):
        """E04_LAB_EXACT: one pass input of the exact round after the lab hooks, from its ``ids``
        [n, B] (the published rows): SHADOW_AR and slot1_pre_j0 before J0, the refresh before a pass
        in ``refresh_at`` (``_exact_refresh``, from the previous pass's logits in ``state``); every
        adapter call is ``tau_lab.exact_adapter_rows`` (the exact prefix, no tail).  Padded to the
        bucket's ``rows`` with row 0, as the round pads its fill.  A seeded round (``lab.seeded``,
        E04_LAB_SEED_EXACT) has no J0 initialization: SHADOW_AR and slot1_pre_j0 do not run."""
        import jetspec.engines.jetspec_plus_plus.draft.tau_lab as tau_lab
        exact = functools.partial(tau_lab.exact_adapter_rows, self.drafter, sequence)
        zeros = torch.zeros(ids.shape[0], dtype=torch.long, device=ids.device)
        swaps, shadow = tau_lab.swap_modes(), tau_lab.shadow_ar()
        if shadow and pass_index == 0 and not lab.seeded:
            ids = tau_lab.shadow_ar_decode(None, None, self.decoder, sequence, length=length,
                                           tail_counts=zeros, first=ids, lab=lab, pack=None,
                                           count=shadow, exact=exact)
        if 'slot1_pre_j0' in swaps and pass_index == 0 and not lab.seeded:
            ids = tau_lab.slot1_pre_j0(None, None, self.decoder, sequence, length=length,
                                       tail_counts=zeros, first=ids, lab=lab, pack=None, exact=exact)
        if pass_index in refresh_at:
            _, ids = self._exact_refresh(bucket, sequence, length, ids, state.pop('logits'), rows=rows,
                                         lab=lab, pass_index=pass_index,
                                         **({'again': True} if pass_index > refresh_at[0] else {}))
        return torch.cat((ids, ids[:1].expand(rows - ids.shape[0], -1))) if rows > ids.shape[0] else ids

    @torch.inference_mode()
    def _exact_refresh(self, bucket, sequence, length, ids, logits, *, rows, lab, pass_index,
                       again=False):
        """E04_LAB_EXACT: the adapter refresh inside the exact anchor0 round.

        k-hat is ``_adapter_refresh``'s rule on the previous pass's top-1 probabilities.  The rows
        are ``tau_lab.exact_adapter_rows`` of ``ids`` [n, B] (the staging call over the exact
        committed prefix: no tail), ``tau_lab.refresh_hook`` runs with zero tail slots, and the
        eager inject exposes entries 0..khat_used of each row (the bucket's padding rows repeat
        row 0, as the pass input does).  JETSPEC_REFRESH_PACK / _INJECT_PACK / _FIXED / _REUSE0
        shape the pipelined call over the staged tail and do not apply.  Returns
        ``(khat_used, ids)``, ids carrying any first_wrong_refresh swap."""
        import jetspec.engines.jetspec_plus_plus.draft.tau_lab as tau_lab
        d = self.drafter
        p_top1 = torch.softmax(logits[:, 1:], -1, dtype=torch.float32).amax(-1)
        refresh_cutoff = (refresh_khat_anchor0 if d.block_size == 16 else
                          refresh_khat_anchor0_width31 if d.block_size == 32 else
                          refresh_khat_anchor0_short)
        khat = refresh_cutoff(p_top1, refresh_khat_threshold(again) if again else refresh_khat_threshold())
        khat_cap = refresh_khat_cap(again) if again else None
        if khat_cap is not None:
            khat = khat.clamp(max=khat_cap)
        block = tau_lab.exact_adapter_rows(d, sequence, ids)
        khat, ids = tau_lab.refresh_hook(
            getattr(self, 'decoder', None), sequence, length=length, tail_counts=torch.zeros_like(khat),
            ids=ids, khat=khat, block=block, lab=lab, pass_index=pass_index, exact=True, draft_logits=logits)
        pad = rows - ids.shape[0]
        bucket.head.inject_refresh(torch.cat((block, block[:1].expand(pad, -1, -1))) if pad else block,
                                   torch.cat((khat, khat[:1].expand(pad))) if pad else khat,
                                   **({'reset': True} if again else {}))
        return khat, ids




    @torch.inference_mode()
    def propose_fallback(self, sequence, *, true_k, passes, bucketed=None, anchor=None,
                         round_ahead=None, batch_graph=False):
        """JETSPEC_MISS_REDRAFT=1: one row from the committed state after a miss.

        The kth frame with an exact context: the last accepted token in slot 0 (its tap left
        out of the context, as for every served row), the committed correction pinned in slot 1
        from the first pass on, and the target's own taps for everything before (no adapter
        tail, no refresh). The row is keyed (true_k, correction), so the verifier's ordinary
        row match takes it as a hit.  ``round_ahead`` (E04_LAB_*, default None): the draft
        rank's round-ahead decision, read by the anchor0 exact round (E04_LAB_SEED_EXACT)."""
        d = self.drafter
        prefix = sequence.exact_hidden(device=d.device, dtype=d.dtype)
        ids = sequence.committed_ids.to(d.device)
        if ids.shape[1] < 2 or prefix.shape[1] != ids.shape[1] - 1:
            raise RuntimeError('fallback row requires committed taps for all but the correction')
        marker = d.head.adapted_position_embedding.weight
        context = prefix[0, :-1] + marker[0]
        first = build_random_context_block_token_ids(ids, block_size=16, seed=d.seed,
            prompt_index=sequence.prompt_index, mode='kth').clone()
        anchors = ids[:, -2:-1]
        pinned = ids[0, -1:]
        first[:, :1] = anchors
        first[:, 1] = pinned
        second = None
        if anchor is not None:
            # JETSPEC_MISS_REDRAFT_ANCHOR=decoder: the token after the correction pinned too
            second = torch.tensor([int(anchor)], dtype=torch.long, device=d.device)
            first[:, 2] = second
        if row_layout_anchor0():
            # Under the anchor0 layout the fallback is an exact DFlash round
            if bucketed is False:
                raise ValueError('the anchor0 fallback row is served from the flat bucket only')
            if anchor is not None:
                raise ValueError('JETSPEC_MISS_REDRAFT_ANCHOR pins a draft; the anchor0 fallback '
                                 'is the exact DFlash round and pins only the correction')
            return self._exact_anchor0_rows(sequence, ks=[int(true_k)], passes=passes,
                                            round_ahead=round_ahead,
                                            **({'graphs': True} if batch_graph else {}))
        if bucketed is None:
            bucketed = self.optimized and d.device.type == 'cuda'
        if bucketed:
            # The served flat bucket (a 256-row padded prefix, fixed kernel shapes): one k=0 row
            # over the exact prefix minus its last tap; the tail is masked out at k=0.
            from jetspec.engines.jetspec_plus_plus.models.heads.lac import row_bucket
            exact = prefix[0, :-1]
            prefix_bucket = ((exact.shape[0] + 255) // 256) * 256
            rows = row_bucket(1, cap=int(os.environ.get('JETSPEC_ROWS_CAP', '128')), graphed=self.graphs)
            buckets = self.__dict__.setdefault('_fallback_buckets', OrderedDict())
            key = (rows, prefix_bucket)
            if key not in buckets:
                if len(buckets) == 2:
                    buckets.popitem(last=False)
                buckets[key] = BucketPass(d, N=1, H=rows, prefix_bucket=prefix_bucket,
                    backend=self.backend, host_fast=False, flat=True, graphs=self.graphs)
            bucket = buckets[key]
            buckets.move_to_end(key)
            bucket.head.update([exact], prefix.new_zeros(1, 15, prefix.shape[-1]), [0] * rows, mark=True)
            wide = first.expand(rows, -1).clone()
            wide_anchors = anchors.expand(rows, -1)

            def forward(*, fill_tokens, pass_index):
                del pass_index
                block = wide if fill_tokens is None else torch.cat(
                    [wide_anchors, fill_tokens.expand(rows, -1)], 1)
                logits = bucket(block)[:1]
                return DraftForwardPass(logits=logits, input_tokens=block[:1, 1:], hidden=bucket.hidden[:1])
        else:
            layout = RowLayout([context.shape[0]], [[1]], F=1, device=d.device, vectorized=self.optimized)
            self.row_head.prepare(layout, context)

            def forward(*, fill_tokens, pass_index):
                del pass_index
                block = first if fill_tokens is None else torch.cat([anchors, fill_tokens], 1)
                hidden = self.row_head(d.embed_tokens(block))
                logits = project_row_logits(d.lm_head, hidden)[:, 1:]
                return DraftForwardPass(logits=logits, input_tokens=block[:, 1:], hidden=hidden[:, 1:])

        correction = int(sequence.committed_ids[0, -1].item())
        keys = torch.tensor([[int(true_k), correction]], dtype=torch.long, device=d.device)
        capacity = int(os.environ.get('JETSPEC_ROWS_CAP', '128'))
        output = packed_refinement(
            forward, anchors=anchors, hypotheses=[int(true_k)], sequences=1, passes=passes,
            convergence_limit=d.convergence_limit, output_hypotheses=[int(true_k)],
            output_indices=[0], row_keys=keys, pinned_tokens=pinned,
            fixed_wire_capacity=capacity, predictor=self.predictor, posterior=posterior_mode(),
            correction_head=d.correction_head,
            **({} if second is None else {'pinned_second': second}))
        return output[0]







    def _key_guessed_output(self, outputs):
        """Bootstrap decoder mode with the ordinary rank-1 guesses, statically padded."""
        capacity = int(os.environ.get('JETSPEC_ROWS_CAP', '128'))
        for blocks in outputs:
            live = len(blocks)
            if live > capacity:
                raise ValueError('initial decoder rows exceed JETSPEC_ROWS_CAP')
            if wide_rows() and blocks.block_wire.shape[-1] == 15:
                blocks.widen()
            wire = blocks.block_wire.new_full((capacity, blocks.block_wire.shape[-1]), -1)
            wire[:live] = blocks.block_wire
            keys = blocks.block_wire.new_full((capacity, 2), -1)
            if live:
                keys[:live, 0] = torch.tensor(blocks.hypotheses, device=keys.device)
                keys[:live, 1] = blocks.metadata[:live, 1].to(keys.device)
            blocks.wire = wire
            blocks.row_keys = keys
            blocks.head_metrics = dict(decoder_ready=False, rows_drafted=live,
                rows_published=live, rows_truncated=0, anchor_set_sizes=[1] * live,
                decoder_ms=0.0, decoder_total_ms=0.0, decoder_prefix_ms=0.0,
                decoder_select_ms=0.0, kpred_ms=0.0, kpred_calls=0,
                kpred_total_ms=0.0, lac_fallbacks=0, kpred_pass=None,
                k_set_size=live, k_set=list(map(int, blocks.hypotheses)),
                k15_dropped=0, kpred_frame_dropped0=0)

    @torch.inference_mode()
    def _propose_flat_bucketed(self, sequences, *, k_hypotheses, passes, abort_if=None,
                               prediction=None):
        from jetspec.engines.jetspec_plus_plus.models.heads.lac import keyed_rows; from jetspec.engines.jetspec_plus_plus.models.heads.lac import row_bucket
        if len(sequences) > 1:
            split = (os.environ.get('JETSPEC_DEVICE_SPLIT', '0') == '1'
                     and self.drafter.device.type == 'cuda')
            decoder_start = torch.cuda.Event(enable_timing=True) if split else None
            if split:
                decoder_start.record()
            predictions = self.decoder.predict_batch(
                sequences, [sequence.pending_block for sequence in sequences],
                dataset=os.environ['JETSPEC_DATASET'],
                coverage=os.environ.get('JETSPEC_ANCHOR_COVERAGE', '0.99'),
                grouped=os.environ.get('JETSPEC_ANCHOR_DREJ', '0') == '1',
                cap=int(os.environ.get('JETSPEC_ANCHOR_CAP', '16')),
                after_misses=[bool(getattr(sequence, '_jetspec_previous_selection_miss', False))
                              for sequence in sequences])
            decoder_end = torch.cuda.Event(enable_timing=True) if split else None
            if split:
                decoder_end.record()
            from jetspec.engines.jetspec_plus_plus.draft.tau_lab import lab_active as e04_lab_active
            # E04_LAB_*: the lab lives in the N=1 path, so a lab read drafts each request there
            if not self.graphs and rows_assembly() and row_layout_anchor0() and not e04_lab_active():
                batched = self._flat_rows_and_passes_batch(sequences, predictions,
                                                            passes=passes, abort_if=abort_if,
                                                            device_split=((decoder_start, decoder_end)
                                                                          if split else None))
                if batched is not None:
                    return batched
                # s_miss: one line per N>1 round that leaves the batched rows for the
                # per-sequence loop below (the count the batched-miss gate reads)
                self.batch_rows_fallbacks = getattr(self, 'batch_rows_fallbacks', 0) + 1
                print('BATCH_ROWS_FALLBACK ' + json.dumps(dict(
                    N=len(sequences), count=self.batch_rows_fallbacks,
                    after_miss=sum(bool(getattr(sequence, '_jetspec_previous_selection_miss', False))
                                   for sequence in sequences))), flush=True)
            if any(item.logits is None for item in predictions):
                from dataclasses import replace
                predictions = [replace(item, logits=self.decoder.lm_head(item.hidden))
                               if item.logits is None else item for item in predictions]
            return [self._propose_flat_bucketed([sequence], k_hypotheses=k_hypotheses,
                                                passes=passes, abort_if=abort_if,
                                                prediction=prediction)[0]
                    for sequence, prediction in zip(sequences, predictions)]
        if len(sequences) != 1 or self.branches != 1 or not self.optimized:
            raise ValueError('decoder rows require one sequence per call, F=1, and optimized rows')
        sequence = sequences[0]
        if sequence.pending_block is None or sequence.staged.predicted_hidden is None:
            raise RuntimeError('decoder proposal requires a staged pending block')
        # Flat keyed rows can repeat a k when the decoder publishes multiple
        # anchors for that prefix.  The next decoder proposal needs one entry
        # per prefix length, in the same first-seen order as the wire rows.
        after_miss = bool(getattr(sequence, '_jetspec_previous_selection_miss', False))
        profile = os.environ.get('W278_HEAD_PROFILE') == '1'
        ks = unique_decoder_hypotheses(k_hypotheses)
        early = None
        if rows_ctx_early() and not self.graphs and self.drafter.device.type == 'cuda':
            # The round's context K/V projected now, on a side stream beside the decoder
            early_prefix = sequence.exact_hidden(device=self.drafter.device, dtype=self.drafter.dtype)[0]
            early_bucket = ((early_prefix.shape[0] + 255) // 256) * 256
            stages = self.__dict__.setdefault('_early_contexts', OrderedDict())
            if early_bucket not in stages:
                if len(stages) == 2:
                    stages.popitem(last=False)
                stages[early_bucket] = EarlyContextKV(self.drafter.head, early_bucket,
                                                      **({'tail_rows': self.drafter.block_size} if wide_rows() else {}))
            early = stages[early_bucket]
            early.issue(early_prefix, sequence.staged.predicted_hidden[:, :self.drafter.block_size if wide_rows() else 15],
                        key=sequence)
        static = None
        if ((rows_static() or rows_defer_home()) and rows_assembly() and self.graphs and self.drafter.device.type == 'cuda'
                and not flat_bucket_host_fast(host_fast=self.host_fast, graphed=self.graphs)
                and (adapter_refresh_mode() == 'off' or rows_static_chain())):
            # JETSPEC_ROWS_STATIC: the fixed row-cap bucket's context K/V (and its
            # length-only rotary rows) issued now; the stage's keys set the geometry later
            static = self._static_preissue(sequence)
        prediction = prediction if prediction is not None else self.decoder.predict(
            sequence, sequence.pending_block,
            dataset=os.environ['JETSPEC_DATASET'],
            coverage=os.environ.get('JETSPEC_ANCHOR_COVERAGE', '0.99'),
            grouped=os.environ.get('JETSPEC_ANCHOR_DREJ', '0') == '1',
            cap=int(os.environ.get('JETSPEC_ANCHOR_CAP', '16')),
            after_miss=after_miss,
        )
        if getattr(self, 'deadline', None) is not None:
            self.deadline.checkpoint('before_rows')
        if static is not None:
            deferred = self._assemble_rows(sequence, prediction, after_miss=after_miss,
                                           defer=True, reuse_home=rows_defer_home())
            if deferred is not None:
                try:
                    return self._flat_rows_and_passes(
                        sequence, prediction, after_miss=after_miss, profile=profile, ks=ks,
                        early=early, passes=passes, abort_if=abort_if, static=(static, deferred))
                except _StaticFallback:
                    self.static_fallbacks = getattr(self, 'static_fallbacks', 0) + 1
        return self._flat_rows_and_passes(
            sequence, prediction, after_miss=after_miss, profile=profile, ks=ks,
            early=early, passes=passes, abort_if=abort_if, static=None)


    @torch.inference_mode()
    def _flat_rows_and_passes(self, sequence, prediction, *, after_miss, profile, ks, early, passes,
                              abort_if, static):
        from jetspec.engines.jetspec_plus_plus.models.heads.lac import keyed_rows; from jetspec.engines.jetspec_plus_plus.models.heads.lac import row_bucket
        # E04_LAB_* (tau_lab.py, default off): an environment read while every lab name is unset.
        # E04_LAB_KSCORE_WRITTEN: the k score's terms at the verified block's decoder-written slots
        import jetspec.engines.jetspec_plus_plus.draft.tau_lab as e04_lab
        lab_on = e04_lab.lab_active()
        written = e04_lab.written_terms(sequence.pending_block, after_miss=after_miss) if lab_on else None
        if static is not None:
            static, (stage_obj, start_home, finish) = static
            ks, k_probabilities, live_keys, requested, truncated, anchor_sizes = [0], None, None, 0, 0, None
            decoder_ms, decoder_ready = prediction.elapsed_ms, True
            kselect_ms = keyed_ms = 0.0
            stage = 'static'
        else:
            stage = self._assemble_rows(sequence, prediction, after_miss=after_miss,
                                        **({'written': written} if written is not None else {})
                                        ) if rows_assembly() else None
        if stage == 'static':
            pass
        elif stage is not None:
            # JETSPEC_ROWS_ASSEMBLY=1: the round's k score, row rule and keys in one device
            # pass and one trip home (heads/assembly.py); the host path below is its reference
            ks, k_probabilities, live_keys, requested, truncated, anchor_sizes, stage_keys = stage
            decoder_ms, decoder_ready = prediction.elapsed_ms, True
            kselect_ms = keyed_ms = 0.0
        else:
            if decoder_blend_kset():
                # One host copy serves k selection and the frame-drop reports below.
                probabilities = sequence.pending_block.k_probabilities
                k_probabilities = None if probabilities is None else probabilities.double().cpu()
            elif posterior_mode():
                k_probabilities = (sequence.pending_block.k_probabilities
                                   if sequence.pending_block.k_probabilities is not None
                                   else posterior_k_mass(sequence.pending_block.logits))
            else:
                from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
                width = draft_width() + 1
                k_probabilities = torch.ones(width, device=sequence.pending_block.draft_ids.device,
                                             dtype=torch.float64) / width
            if os.environ.get('E04_KSCORE'):
                ks, k_probabilities = e04_kscore_blend(
                    sequence.pending_block, None if prediction is None else prediction.logits,
                    ks, k_probabilities, after_miss=after_miss,
                    live=hasattr(sequence, '_jetspec_seq_id') and hasattr(sequence, '_jetspec_candidate_round'),
                    offers=None if prediction is None else prediction.offers,
                    **({'written': written} if written is not None else {}))
            live_round = hasattr(sequence, '_jetspec_seq_id') and hasattr(sequence, '_jetspec_candidate_round')
            # JETSPEC_E04_LEAN=1: under E04_ROWRULE=joint a live round's rows, k set and counts are
            # all replaced by e04_rowrule_joint, so the k selection and keyed rows below are dead work.
            joint_lean = (e04_lean() and os.environ.get('E04_ROWRULE') == 'joint'
                          and prediction is not None and live_round)
            if profile and k_probabilities is not None and k_probabilities.device.type == 'cuda':
                torch.cuda.synchronize(k_probabilities.device)
            kselect_started = time.perf_counter()
            off_passthrough = kpred_forward_off() and prediction is None and not live_round
            ks, selected_k_probabilities = ((ks, None) if joint_lean or off_passthrough else
                                            decoder_k_selection(
                ks, k_probabilities,
                predictor=decoder_blend_kset(),
                after_miss=after_miss))
            if profile and k_probabilities is not None and k_probabilities.device.type == 'cuda':
                torch.cuda.synchronize(k_probabilities.device)
            kselect_ms = (time.perf_counter() - kselect_started) * 1000.0
            keyed_ms = 0.0
            if prediction is None:
                fallback = int(sequence.pending_block.correction_guess)
                live_keys = torch.tensor([(k, fallback) for k in ks],
                                         device=sequence.pending_block.draft_ids.device,
                                         dtype=torch.long).reshape(-1, 2)
                requested = len(ks)
                truncated = 0
                if os.environ.get('JETSPEC_ROW_SELECT') == 'budget' and selected_k_probabilities is not None:
                    capacity = int(os.environ.get('JETSPEC_ROWS_CAP', '128'))
                    order = torch.argsort(selected_k_probabilities, descending=True,
                                          stable=True)[:capacity]
                    live_keys = live_keys.index_select(0, order.to(live_keys.device))
                    truncated = requested - len(live_keys)
                from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
                anchor_sizes = [1 if k in ks else 0 for k in range(draft_width() + 1)]
                decoder_ms = 0.0
                decoder_ready = False
            else:
                select_rows = keyed_rows
                if os.environ.get('JETSPEC_HEADS_FAST') == '1':
                    from jetspec.engines.jetspec_plus_plus.models.heads.fast import fast_keyed_rows
                    select_rows = fast_keyed_rows
                if profile and k_probabilities is not None and k_probabilities.device.type == 'cuda':
                    torch.cuda.synchronize(k_probabilities.device)
                keyed_started = time.perf_counter()
                if joint_lean:
                    # placeholders: e04_rowrule_joint below returns the round's keys and counts
                    live_keys = torch.empty((0, 2), dtype=torch.long, device='cpu')
                    requested, truncated = 0, 0
                else:
                    selected = select_rows(
                        torch.tensor(ks, dtype=torch.long, device='cpu'),
                        selected_k_probabilities, prediction.offers.ids,
                        prediction.offers.probabilities,
                        cap=int(os.environ.get('JETSPEC_ROWS_CAP', '128')))
                if profile and k_probabilities is not None and k_probabilities.device.type == 'cuda':
                    torch.cuda.synchronize(k_probabilities.device)
                keyed_ms = (time.perf_counter() - keyed_started) * 1000.0
                if not joint_lean:
                    live_keys = selected.keys
                    requested, truncated = selected.requested, selected.truncated
                anchor_sizes = prediction.offers.set_sizes
                decoder_ms = prediction.elapsed_ms
                decoder_ready = True
            if os.environ.get('E04_ROWRULE'):
                ks, live_keys, requested, truncated = e04_rowrule_joint(
                    sequence.pending_block, prediction, k_probabilities, ks, live_keys, requested, truncated,
                    after_miss=after_miss,
                    live=hasattr(sequence, '_jetspec_seq_id') and hasattr(sequence, '_jetspec_candidate_round'))
        if profile:
            print('W278_ROWS_PROFILE ' + json.dumps(dict(
                seq_id=getattr(sequence, '_jetspec_seq_id', None),
                round_id=getattr(sequence, '_jetspec_candidate_round', None),
                kset_selection_ms=kselect_ms, keyed_rows_ms=keyed_ms,
                fast=os.environ.get('JETSPEC_HEADS_FAST') == '1')), flush=True)
        # E04_LAB_ROWS_MIN / E04_LAB_SWAP_ROWS: pairs appended after the joint rule, and the rule's
        # P(k) and pair scores for the round's record (tau_lab.joint_rows; None: no rule to extend)
        lab_rows = (e04_lab.joint_rows(sequence.pending_block, prediction, k_probabilities, live_keys,
                                       requested, after_miss=after_miss)
                    if lab_on and stage != 'static' else None)
        lab_added = lab_rows is not None and lab_rows[0] is not live_keys
        if lab_added:
            live_keys = lab_rows[0]
            ks = sorted({int(k) for k in live_keys[:, 0].tolist()})
        capacity = int(os.environ.get('JETSPEC_ROWS_CAP', '128'))
        d = self.drafter
        refresh_mode = adapter_refresh_mode()
        refresh = refresh_mode != 'off' and passes is not None and passes > 1
        refresh_at = refresh_pass_indices(refresh_mode)
        # E04_LAB_* (tau_lab.py, default off): with every lab name unset nothing below reads the
        # lab state; E04_LAB_REFRESH_EVERY / E04_LAB_EXTRA_PASSES set the refresh schedule and
        # the pass count (total_passes) the round runs
        total_passes, lab_fixed_passes = passes, False
        if lab_on:
            refresh_at, total_passes, lab_fixed_passes = e04_lab.refresh_schedule(refresh_at, passes)
            refresh = refresh_mode != 'off' and total_passes is not None and total_passes > 1
        swap_on = os.environ.get('JETSPEC_REFRESH_SWAP', '0') == '1' and not self.graphs
        schedule = None
        executed_operations = []
        if swap_on:
            from jetspec.engines.jetspec_plus_plus.draft.eager_refresh_swap import eager_swap_schedule
            from jetspec.engines.jetspec_plus_plus.draft.eager_schedule import parse_schedule
            schedule = None if lab_on else parse_schedule(os.environ['JETSPEC_CHAIN_FIXED_CONFIG'])
            passes, refresh_at = eager_swap_schedule()
            refresh = True
            total_passes = passes
        late_rows = {}
        after_passes = None
        anchor0 = row_layout_anchor0()
        if stage == 'static':
            # JETSPEC_ROWS_STATIC: the stage's device keys, the fixed row-cap bucket
            # whose context K/V was issued before the decoder, and only the geometry here.
            # An empty round keeps the host path's private row [0, correction guess].
            drafted, bucket, bucket_built = static['drafted'], static['bucket'], static['built']
            prefix, use_host_fast = static['prefix'], False
            raw = stage_obj.keys[:drafted]
            private = raw.new_tensor([0, int(sequence.pending_block.correction_guess)])
            first_row = torch.where(stage_obj.requested > 0, raw[0], private)
            device_keys = torch.cat((first_row[None], raw[1:]), dim=0).clamp_min(0)
            active_rows = torch.arange(drafted, device=d.device) < stage_obj.requested.clamp(min=1, max=drafted)
            padded_ks = [0] * drafted  # placeholder: only its length is read by the passes
            geometry = device_keys[:, 0] + 1 if anchor0 else device_keys[:, 0]
            if rows_static_chain():
                tail = sequence.staged.predicted_hidden[:, :d.block_size if wide_rows() else 15]
                bucket.head.update_chain(static['prefix_padded'], tail, static['lengths'],
                                         geometry.contiguous())
            else:
                bucket.head._update_geometry(static['lengths'], geometry.contiguous())
            after_passes = start_home

            def late():
                done = finish()
                if done is None:
                    raise _StaticFallback()
                late_rows.update(done)
                keys = done['live_keys']
                return keys[:, 0].tolist(), list(range(len(keys))), keys
        else:
            late = None
            active_rows = None
            drafted = row_bucket(len(live_keys), cap=capacity, graphed=self.graphs)
            if swap_on:
                drafted = rows_rule_cap()
            padded_keys = live_keys.new_full((drafted, 2), -1)
            padded_keys[:len(live_keys)] = live_keys
            if not len(live_keys):
                # Keep one private row to advance the drafter after the verifier's
                # ordinary AR miss.  The public fixed-capacity packet stays all -1.
                padded_keys[0] = padded_keys.new_tensor(
                    [ks[0] if ks else 0, int(sequence.pending_block.correction_guess)])
            padded_ks = padded_keys[:, 0].clamp_min(0).tolist()
            # One pinned copy of the row keys serves the anchor gather and the pin.
            device_keys = (padded_keys.clamp_min(0) if padded_keys.device == d.device
                           else device_tensor(padded_keys.clamp_min(0).tolist(), d.device)) \
                if stage is None or lab_added or not len(live_keys) else stage_keys[:drafted].clamp_min(0)
            prefix = sequence.exact_hidden(device=d.device, dtype=d.dtype)[0]
            prefix_bucket = ((prefix.shape[0] + 255) // 256) * 256
            key = ('flat', 1, drafted, prefix_bucket) + (('refresh',) if refresh else ()) + \
                (('anchor0',) if anchor0 else ())
            use_host_fast = flat_bucket_host_fast(
                host_fast=self.host_fast, graphed=self.graphs) and not refresh
            # The graphed flat bucket takes anchor0 too (the same geometry body in its
            # geometry graph, the pass graph's slot-15 outputs rewritten in place per replay)
            if anchor0 and (use_host_fast or d.correction_head is not None):
                raise ValueError('JETSPEC_ROW_LAYOUT=anchor0 serves non-host-fast rows '
                                 'without a correction head only')
            bucket_built = key not in self.buckets
            bucket = cached_flat_bucket(self.buckets, key, lambda: BucketPass(
                    d, N=1, H=drafted, prefix_bucket=prefix_bucket, backend=self.backend,
                    host_fast=use_host_fast, flat=True, graphs=self.graphs,
                    **({'refresh': True} if refresh else {}),
                    **({'anchor0': True} if anchor0 else {}),
                    **({'wide': True} if wide_rows() else {})), graphs=self.graphs,
                    limit=getattr(self, '_graph_bucket_limit', None))
            tail = sequence.staged.predicted_hidden[:, :d.block_size if wide_rows() else 15]
            # anchor0: row (k, a) reads tail rows c, d1..dk and its block starts at the
            # anchor's position, so the geometry takes k + 1 where the kth layout takes k
            geometry_ks = [k + 1 for k in padded_ks] if anchor0 else padded_ks
            bucket.head.update([prefix], tail, geometry_ks, mark=True,
                **({'prefix_key': sequence} if use_host_fast or rows_ctx_incr() or early is not None else {}),
                **({'early': early} if early is not None and not use_host_fast else {}))
        # The same seeded indices gathered on the device: the host gather
        # copied the whole context down and the slots back up every round.
        sample = build_random_context_block_token_ids(
            sequence.committed_ids.to(d.device), block_size=d.block_size, seed=d.seed,
            prompt_index=sequence.prompt_index, mode='kth',
            device_gather=use_host_fast or d.device.type == 'cuda')
        first = sample[:, None, :].expand(-1, drafted, -1).reshape(drafted, d.block_size).clone()
        source = torch.cat((
            sequence.committed_ids[:, -1:].to(d.device),
            sequence.pending_block.draft_ids.reshape(1, -1).to(d.device),
            sequence.pending_block.draft_ids[-1:].reshape(1, 1).to(d.device)), dim=1)
        preceding = source.index_select(1, device_keys[:, 0]).reshape(-1, 1)
        first[:, :1].copy_(device_keys[:, 1:2] if anchor0 else preceding)
        refresh_ks = device_keys[:, 0] + 1 if anchor0 else device_keys[:, 0]
        # E04_LAB_*: the round's lab state, None while every lab name is unset
        lab_round, lab_swaps, lab_shadow = None, (), 0
        if lab_on:
            e04_lab.check_regime(graphed=self.graphs or stage == 'static')
            lab_round = e04_lab.LabRound(drafted, first.shape[1], d.device, recipe_passes=passes)
            lab_swaps, lab_shadow = e04_lab.swap_modes(), e04_lab.shadow_ar()
            lab_round.rows = None if lab_rows is None else lab_rows[1]
            lab_round.verified = e04_lab.verified_record(sequence.pending_block)
            if e04_lab.record_topk() is not None:
                lab_round.keys = padded_keys.tolist()  # E04_LAB_RECORD_TOPK: each drafted row's key (-1: padding)
            if e04_lab.after_swap_fill() != 'keep':
                # E04_LAB_AFTER_SWAP_FILL: the mask token and the J0 context draws the fill reads
                lab_round.fill_source(d.head.mask_token_id, first)

        mask_hits = []
        refresh_state = {}
        swap = None
        if swap_on:
            from jetspec.engines.jetspec_plus_plus.draft.eager_refresh_swap import prepare_eager_swap
            swap = prepare_eager_swap(self.decoder, sequence, bucket, refresh_at)
            active_rows = torch.arange(drafted, device=d.device) < max(1, len(live_keys))
            from jetspec.engines.jetspec_plus_plus.draft.serve_policy import section
            tail = section('tail_passes')
            tail_count = tail['count'] if tail['enabled'] else 0

        def swap_refresh(ids, pass_index, finished):
            from jetspec.engines.jetspec_plus_plus.draft.eager_refresh_swap import apply_eager_refresh
            result = apply_eager_refresh(
                swap, self._adapter_refresh, bucket, sequence, prefix.shape[0],
                refresh_ks, ids, refresh_state['logits'], active=active_rows,
                finished=finished, again=pass_index > refresh_at[0])
            if pass_index > refresh_at[0] and 'tokens' in result and refresh_state.get('tokens') is not None:
                result['tokens'] += refresh_state['tokens']
            refresh_state.update(result)
            executed_operations.append(f'R{pass_index}')
            return result

        def forward(*, fill_tokens, pass_index, finished=None):
            if anchor0 and bucket.wide:
                # JETSPEC_DRAFTS=15: fill_tokens is the whole block [pinned anchor, d1..d15]
                ids = first if fill_tokens is None else fill_tokens
            elif anchor0:
                # fill_tokens is the frame [pinned anchor, d1..d14]; slot 15 takes the
                # previous pass's draft 15
                ids = first if fill_tokens is None else torch.cat((fill_tokens, bucket.extra_tokens), dim=1)
            else:
                ids = first if fill_tokens is None else torch.cat((preceding, fill_tokens), dim=1)
            if lab_shadow and pass_index == 0:
                # E04_LAB_SHADOW_AR=n: the J0 input's slots 1..n are the shadow's greedy decode
                ids = e04_lab.shadow_ar_decode(
                    d.adapter.head, d.embed_tokens, self.decoder, sequence, length=prefix.shape[0],
                    tail_counts=refresh_ks, first=ids, lab=lab_round, pack=refresh_pack_mode(),
                    count=lab_shadow)
            if 'slot1_pre_j0' in lab_swaps and pass_index == 0:
                # E04_LAB_SWAP=slot1_pre_j0: the J0 input's slot 1 takes the decoder's top-1
                ids = e04_lab.slot1_pre_j0(
                    d.adapter.head, d.embed_tokens, self.decoder, sequence, length=prefix.shape[0],
                    tail_counts=refresh_ks, first=ids, lab=lab_round, pack=refresh_pack_mode())
            # The mask-free guard is a device flag read once after the passes,
            # so it no longer drains the device before every pass.
            if swap is None:
                mask_hits.append((ids == int(d.head.mask_token_id)).any())
            elif pass_index == 0 and swap.fill_mode == 'context':
                swap.fill_context.copy_(ids)
            if (refresh and pass_index in refresh_at
                    and (schedule is None or pass_index <= schedule.j)):
                # The refresh hook: at the start of pass j+1, from the pass-j draft and logits.
                # A later refresh (after_j0_j1) replaces the earlier entries and mask.
                again = pass_index > refresh_at[0]
                if swap is None:
                    result = self._adapter_refresh(
                        bucket, sequence, prefix.shape[0], refresh_ks, ids,
                        refresh_state.pop('logits'), **({'again': True} if again else {}),
                        **({'lab': lab_round, 'pass_index': pass_index} if lab_round is not None else {}))
                    # E04_LAB_SWAP=first_wrong_refresh: this pass's input carries the swapped token
                    ids = result.pop('lab_ids', ids)
                    if lab_round is not None and lab_round.filled is not None:
                        # E04_LAB_AFTER_SWAP_FILL: the guard on this pass's input, the filled slots exempt
                        mask_hits.append(e04_lab.mask_outside_fill(ids, d.head.mask_token_id, lab_round))
                    if lab_round is not None and 'skipped' in result:
                        lab_round.refreshes.append(dict(pass_index=pass_index, skipped=1))
                else:
                    result = swap_refresh(ids, pass_index, finished)
                if swap is None and again and 'tokens' in result and refresh_state.get('tokens') is not None:
                    result['tokens'] += refresh_state['tokens']
                if again and 'skipped' in result:
                    result = {}  # the earlier refresh stays in place
                refresh_state.update(result)
            if swap is not None:
                mask = (ids == int(d.head.mask_token_id)) & active_rows[:, None]
                if swap.fill_mode != 'keep':
                    mask &= ~swap.fill_filled
                mask_hits.append(mask.any())
                if swap.fill_mode != 'keep':
                    swap.fill_filled.zero_()
            logits = bucket(ids)
            if schedule is not None:
                executed_operations.append(f'J{pass_index}')
            if swap is not None and swap.rule is not None and (pass_index == 0 or pass_index + 1 in refresh_at):
                swap.retain_draft(logits)
            if refresh and pass_index + 1 in refresh_at:
                refresh_state['logits'] = logits
            return DraftForwardPass(logits=logits, input_tokens=(ids if bucket.wide else ids[:, :15])
                                    if anchor0 else ids[:, 1:],
                                    hidden=bucket.hidden)

        output = packed_refinement(
            forward, anchors=preceding, hypotheses=padded_ks, sequences=1,
            passes=total_passes, convergence_limit=d.convergence_limit, abort_if=abort_if,
            output_hypotheses=None if late is not None else live_keys[:, 0].tolist(),
            output_indices=None if late is not None else list(range(len(live_keys))),
            row_keys=None if late is not None else live_keys,
            pinned_tokens=device_keys[:, 1], fixed_wire_capacity=capacity,
            active_rows=active_rows,
            after_passes=after_passes,
            predictor=self.predictor, posterior=posterior_mode(),
            correction_head=d.correction_head, **({'late': late} if late is not None else {}),
            **({'refresh_swap': swap, 'fixed_passes': True, 'tail_passes': tail_count}
               if swap is not None else {}),
            **({'operation_schedule': schedule, 'refresh_only': swap_refresh}
               if schedule is not None else {}),
            **({'publish': self.publish_hook} if getattr(self, 'publish_hook', None) is not None
               and late is None else {}),
            **({'truncate_if': self.deadline.truncate_if(
                len(schedule.stages) + tail_count if schedule is not None else
                passes + (tail_count if swap is not None else 0))}
               if getattr(self, 'deadline', None) is not None and passes is not None else {}),
            **({'lab_pins': lab_round.pins, 'lab_touched': lab_round.touched,
                'lab_pass': lab_round.observe_pass}
               if lab_shadow or {'slot1_pre_j0', 'first_wrong_refresh'} & set(lab_swaps) else {}),
            **({'final_edit': lambda tokens: e04_lab.pre_verify(
                d.adapter.head, d.embed_tokens, self.decoder, sequence, length=prefix.shape[0],
                tail_counts=refresh_ks, tokens=tokens, lab=lab_round, pack=refresh_pack_mode())}
               if 'pre_verify' in lab_swaps else {}),
            # E04_LAB_RECORD_TOPK: each row's final-pass top-n for pre_verify's record
            # (E04_LAB_SWAP_RULE_FILE: and for pre_verify's rule, the drafter's distribution it reads)
            **({'lab_top': lab_round.observe_top}
               if 'pre_verify' in lab_swaps and (e04_lab.record_topk() is not None
                                                  or e04_lab.swap_rule() is not None) else {}),
            # E04_LAB_REFRESH_EVERY / E04_LAB_EXTRA_PASSES: the whole pass budget runs, and the
            # extra passes read the recipe's last pass label
            **({'fixed_passes': True} if lab_fixed_passes else {}),
            **({'label_passes': passes} if total_passes != passes else {}),
            # E04_LAB_*: every published block carries its row's decoder-written slots (and, under
            # E04_LAB_KSCORE_WRITTEN=draft_p / both, the draft model's p of each written token)
            **({'lab_written': lab_round.written_pins if e04_lab.pin_agreed() else lab_round.pins,
                'lab_written_p': e04_lab.kscore_written() in ('draft_p', 'both')}
               if lab_round is not None else {}))
        if late is not None:
            ks, k_probabilities, live_keys = late_rows['ks'], late_rows['mass'], late_rows['live_keys']
            requested, truncated = late_rows['requested'], late_rows['truncated']
            anchor_sizes = late_rows['anchor_sizes']
        if mask_hits and bool(torch.stack(mask_hits).any()):
            raise ValueError('Jacobi input must be mask-free')
        if lab_round is not None and e04_lab.tl10_on():
            lab_round.passes_run = output[0].metadata[:, 2].tolist()  # tl10: the passes each published row ran
        if anchor0 and not bucket.wide and os.environ.get('W323_D15_LOG') and len(live_keys) and \
                hasattr(sequence, '_jetspec_seq_id') and hasattr(sequence, '_jetspec_candidate_round'):
            # Measurement: each published row's final-pass draft 15, joined offline with
            # the verifier's round records on (seq_id, round, k, correction)
            with open(os.environ['W323_D15_LOG'], 'a') as handle:
                handle.write(json.dumps(dict(
                    seq_id=int(sequence._jetspec_seq_id),
                    round=int(sequence._jetspec_candidate_round),
                    keys=live_keys.tolist(),
                    d15=bucket.extra_tokens[:len(live_keys), 0].tolist())) + '\n')
        host_mark('flat_mask')
        kpred_calls, kpred_total_ms = ((0, 0.0) if self.predictor is None else
                                      self.predictor.consume_metrics())
        k15_dropped = 0
        if os.environ.get('JETSPEC_KSET') == 'predictor' and k_probabilities is not None:
            from jetspec.engines.jetspec_plus_plus.models.heads.kpred import k0_in_lac; from jetspec.engines.jetspec_plus_plus.models.heads.kpred import k15_in_lac
            wire_frame = os.environ.get('JETSPEC_KPRED_FRAME') == 'wire'
            host_block = SimpleNamespace(kpred_pass=sequence.pending_block.kpred_pass,
                                         k_probabilities=k_probabilities)
            k15_dropped = int(not wire_frame and k15_in_lac(
                host_block, root=os.environ['JETSPEC_HEADS_DIR'],
                dataset=os.environ['JETSPEC_DATASET'],
                coverage=os.environ.get('JETSPEC_KPRED_COVERAGE', '0.99')))
            kpred_frame_dropped0 = int(wire_frame and k0_in_lac(
                host_block, root=os.environ['JETSPEC_HEADS_DIR'],
                dataset=os.environ['JETSPEC_DATASET'],
                coverage=os.environ.get('JETSPEC_KPRED_COVERAGE', '0.99')))
        else:
            kpred_frame_dropped0 = 0
        output[0].head_metrics = dict(
            decoder_ready=decoder_ready, decoder_ms=decoder_ms,
            decoder_total_ms=0.0 if prediction is None else prediction.total_ms,
            decoder_prefix_ms=0.0 if prediction is None else prediction.prefix_ms,
            decoder_select_ms=0.0 if prediction is None else prediction.select_ms,
            kpred_ms=float(sequence.pending_block.kpred_ms),
            kpred_calls=kpred_calls, kpred_total_ms=kpred_total_ms,
            lac_fallbacks=0 if prediction is None else prediction.offers.lac_fallbacks,
            kpred_pass=sequence.pending_block.kpred_pass,
            rows_drafted=drafted, rows_published=len(live_keys),
            rows_truncated=truncated, rows_requested=requested,
            **(dict(row_select='budget',
                    row_select_exp=os.environ.get('JETSPEC_ROW_SELECT_EXP', '1,1'),
                    row_select_length=os.environ.get('JETSPEC_ROW_SELECT_LENGTH', '0'))
               if os.environ.get('JETSPEC_ROW_SELECT') == 'budget' else {}),
            k_set_size=len(ks), k15_dropped=k15_dropped,
            kpred_frame_dropped0=kpred_frame_dropped0,
            k_set=list(map(int, ks)), anchor_set_sizes=anchor_sizes,
            # How the round's bucket got its context K/V (built = a new bucket)
            ctx_update=(bucket.head.last_update_kind if isinstance(bucket.head, FastBucketRowDraftHead)
                        else bucket.head.last_ctx_update),
            ctx_rows=(bucket.head.last_refresh_context_rows if isinstance(bucket.head, FastBucketRowDraftHead)
                      else bucket.head.last_ctx_rows),
            bucket_built=int(bucket_built),
            # The published keys (W321_EVENTS=1) and the static-rows state
            **({'row_keys': live_keys.tolist()} if os.environ.get('W321_EVENTS') == '1' else {}),
            **({'rows_static': int(late is not None),
                'static_fallbacks': getattr(self, 'static_fallbacks', 0)} if rows_static() else {}),
            **({'row_layout': 'anchor0'} if anchor0 else {}))
        if schedule is not None:
            output[0].head_metrics['executed_operations'] = executed_operations
            deadline = getattr(self, 'deadline', None)
            if deadline is not None:
                deadline.eager_published_passes = sum(op.startswith('J') for op in executed_operations)
            if deadline is not None and getattr(deadline, 'decision', None) is not None:
                stopped = deadline.decision.get('stopped_before')
                if stopped is not None:
                    deadline.decision['cutoff_stage'] = (
                        ' '.join(schedule.stages[stopped]) if stopped < len(schedule.stages) else
                        f'J{passes + stopped - len(schedule.stages)}')
        if refresh:
            output[0].head_metrics.update(
                refresh_rounds=int('khat' in refresh_state),
                refresh_skipped=int(refresh_state.get('skipped', 0)),
                refresh_tokens=refresh_state.get('tokens'),
                refresh_khat_mean=(float(refresh_state['khat'][:max(1, len(live_keys))]
                                         .double().mean()) if 'khat' in refresh_state else None))
        record_root = os.environ.get('W280_RECORD')
        live_record = record_root and os.environ.get('JETSPEC_CHAIN_WIRE_RECORD', '0') != '1' and \
            hasattr(sequence, '_jetspec_seq_id') and \
            hasattr(sequence, '_jetspec_candidate_round')
        if live_record:
            from jetspec.engines.jetspec_plus_plus.models.heads.decoder import rejected_tokens
            from jetspec.engines.jetspec_plus_plus.models.heads.wirecal import append_proposal_record; from jetspec.engines.jetspec_plus_plus.models.heads.wirecal import proposal_record
            record = proposal_record(
                seq_id=sequence._jetspec_seq_id,
                candidate_round=sequence._jetspec_candidate_round,
                pass_label=sequence.pending_block.kpred_pass,
                predictor_probabilities=sequence.pending_block.k_probabilities,
                anchor_logits=None if prediction is None else prediction.logits,
                rejected=(None if prediction is None else rejected_tokens(
                    sequence.pending_block, after_miss=after_miss)),
                after_miss=after_miss, k_set=ks,
                rows_requested=requested, rows_published=len(live_keys),
                rows_truncated=truncated,
                fallback_anchor=int(sequence.pending_block.correction_guess),
                published_keys=live_keys)
            passes_run = int(sequence.pending_block.passes_run)
            if schedule is not None:
                record['executed_operations'] = executed_operations
            record.update(passes_run=passes_run, raw_pass_label=f'J{passes_run - 1}',
                          served_pass_label=sequence.pending_block.kpred_pass)
            if os.environ.get('E04_RECORD_PTOP1'):
                record = e04_record_p_top1(record, sequence.pending_block)
            if os.environ.get('E04_RECORD_QSHADOW'):
                record = e04_record_q_shadow(
                    record, None if prediction is None else prediction.logits,
                    None if prediction is None else rejected_tokens(sequence.pending_block, after_miss=after_miss))
            if lab_round is not None:
                # E04_LAB_*: per refresh khat_draft / F / true_k / khat_used over the drafted
                # rows (row i is published_keys[i]; rows past rows_published are padding) and
                # every swap as (pass, row, slot, old, new)
                record['e04_lab'] = lab_round.record()
            append_proposal_record(record_root, record)
        host_mark('flat_ret')
        return output

    @torch.inference_mode()
    def _flat_rows_and_passes_batch(self, sequences, predictions, *, passes, abort_if,
                                    device_split=None):
        """The eager joint-rule arm: private row keys, one N-request head and J loop."""
        from jetspec.engines.jetspec_plus_plus.models.heads.lac import row_bucket
        d = self.drafter
        count = len(sequences)
        capacity = int(os.environ.get('JETSPEC_ROWS_CAP', '128'))
        publish_wire = (getattr(self, '_device_publish_wire', None) if count > 1 and
                        os.environ.get('JETSPEC_DEVICE_PUBLISH', '0') == '1' else None)
        refresh_mode = adapter_refresh_mode()
        refresh = refresh_mode != 'off' and passes is not None and passes > 1
        split_events = {'decoder': [device_split]} if device_split is not None else None

        def split_start():
            if split_events is None:
                return None
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event

        def split_end(label, started):
            if started is not None:
                ended = torch.cuda.Event(enable_timing=True)
                ended.record()
                split_events.setdefault(label, []).append((started, ended))

        assembly_start = split_start()
        # JETSPEC_BATCH_DRAFT_GRAPH: the stage's host parse waits until the passes are queued
        deferred = batch_draft_graph() and os.environ.get('B02_KEY_DUMP') != '1'
        stages, selected_keys = self._assemble_rows_batch(sequences, predictions, defer=deferred)
        if stages is None:
            return None  # preserve the host full-row fallback
        finish_stages = None
        if deferred:
            stage_device, finish_stages = stages, selected_keys
            stages, selected_keys = None, stage_device.keys
        if publish_wire is not None and (not deferred or self.predictor is not None):
            raise ValueError('JETSPEC_DEVICE_PUBLISH requires deferred rows without k predictor')
        if os.environ.get('B02_KEY_DUMP') == '1':
            for sequence, prediction, stage in zip(sequences, predictions, stages):
                reference = self._assemble_rows(
                    sequence, prediction, after_miss=stage['after_miss'])
                if reference is None:
                    raise RuntimeError('batched key dump requires exact decoder offers')
                actual, expected = stage['keys'], reference[2]
                print('B02_KEY_DUMP ' + json.dumps(dict(
                    seq_id=getattr(sequence, '_jetspec_seq_id', None),
                    round_id=getattr(sequence, '_jetspec_candidate_round', None),
                    batched=actual.tolist(), independent=expected.tolist())), flush=True)
                if not torch.equal(actual, expected):
                    raise AssertionError('batched row keys differ from independent selection')
        # JETSPEC_BATCH_DRAFT_GRAPH=1: one captured bucket per (batch bucket, stage row
        # capacity, prefix bucket); the padded requests' rows are inactive from the start
        graphed = batch_draft_graph()
        padded = batch_draft_bucket(count) if graphed else count
        H = (int(selected_keys.shape[1]) if graphed else
             max(row_bucket(len(stage['keys']), cap=capacity, graphed=False) for stage in stages))
        prefixes = [sequence.exact_hidden(device=d.device, dtype=d.dtype)[0]
                    for sequence in sequences]
        prefix_bucket = ((max(prefix.shape[0] for prefix in prefixes) + 255) // 256) * 256
        key = ('flat', padded, H, prefix_bucket) + (('refresh',) if refresh else ()) + ('anchor0',)
        if graphed:
            # a separate shelf: the eager and N=1 buckets keep their own two-entry cache
            buckets = self.__dict__.setdefault('_batch_graph_buckets', OrderedDict())
            built = key not in buckets
            if built:
                if len(buckets) >= self.__dict__.get('_batch_graph_bucket_limit', 6):
                    buckets.popitem(last=False)
                buckets[key] = BucketPass(
                    d, N=padded, H=H, prefix_bucket=prefix_bucket, backend=self.backend,
                    host_fast=False, flat=True, graphs=True, anchor0=True, wide=wide_rows())
            buckets.move_to_end(key)
            bucket = buckets[key]
        else:
            built = key not in self.buckets
            bucket = cached_flat_bucket(self.buckets, key, lambda: BucketPass(
                d, N=count, H=H, prefix_bucket=prefix_bucket, backend=self.backend,
                host_fast=False, flat=True, graphs=False, refresh=refresh,
                anchor0=True, wide=wide_rows()), graphs=False)
        # JETSPEC_BATCH_ROWS_LIVE: the passes run over the round's live rows (rows_live.py)
        from jetspec.engines.jetspec_plus_plus.draft.rows_live import live_row_bucket; from jetspec.engines.jetspec_plus_plus.draft.rows_live import live_row_order; from jetspec.engines.jetspec_plus_plus.draft.rows_live import rows_live; from jetspec.engines.jetspec_plus_plus.draft.rows_live import rows_live_quantum
        compact = (graphed and H == capacity and not refresh and rows_live()
                   and os.environ.get('JETSPEC_FUSE', 'off') == 'off')
        rows_pass = bucket
        if compact:
            live_passes = self._live_row_passes(bucket, padded, H)
        # JETSPEC_BATCH_ROWS_ASSEMBLY_GRAPH: the captured assembly built the padded row inputs
        # (assembly_batch.row_inputs, the values of the batched gathers below)
        captured = getattr(stage_device, 'row_inputs', None) if stages is None else None
        B = d.block_size
        tails = torch.cat([sequence.staged.predicted_hidden[:, :B if wide_rows() else 15]
                           for sequence in sequences], dim=0)
        if captured is not None:
            keys_device, live_counts, geometry, active_rows, preceding, first = captured
            if padded > count:
                extra = padded - count
                tails = torch.cat((tails, tails.new_zeros(extra, *tails.shape[1:])))
                dummy = prefixes[0].new_zeros(1, prefixes[0].shape[-1])
                prefixes = prefixes + [dummy] * extra
        else:
            # the per-request row inputs as batched gathers (the same values as the loop they
            # replace: a clone per request, the private row of an empty round, index_select)
            keys_device = selected_keys[:, :H].clone()
            if stages is None:
                # the deferred stage: the same private row and live count from the device counts
                requested = stage_device.requested
                guesses = device_tensor([int(sequence.pending_block.correction_guess)
                                         for sequence in sequences], d.device)
                private = torch.stack((torch.zeros_like(guesses), guesses), dim=1)
                keys_device[:, 0] = torch.where((requested == 0)[:, None], private, keys_device[:, 0])
                live_counts = requested.clamp(max=stage_device.cap).clamp(min=1)
            else:
                live_counts = []
                for n, (sequence, stage) in enumerate(zip(sequences, stages)):
                    if not len(stage['keys']):
                        keys_device[n, 0] = keys_device.new_tensor(
                            [0, int(sequence.pending_block.correction_guess)])
                    live_counts.append(max(1, len(stage['keys'])))
            keys_device = keys_device.clamp_min(0)
            samples = batched_kth_random_context(sequences, block_size=B, seed=d.seed, device=d.device)
            source = torch.cat((
                torch.cat([sequence.committed_ids[:, -1:].to(d.device) for sequence in sequences]),
                torch.stack([sequence.pending_block.draft_ids.reshape(-1).to(d.device)
                             for sequence in sequences])), dim=1)
            source = torch.cat((source, source[:, -1:]), dim=1)
            if padded > count:
                extra = padded - count
                keys_device = torch.cat((keys_device, keys_device.new_zeros(extra, H, 2)))
                live_counts = (torch.cat((live_counts, live_counts.new_zeros(extra)))
                               if isinstance(live_counts, torch.Tensor) else live_counts + [0] * extra)
                tails = torch.cat((tails, tails.new_zeros(extra, *tails.shape[1:])))
                samples = torch.cat((samples, samples.new_zeros(extra, B)))
                source = torch.cat((source, source.new_zeros(extra, source.shape[1])))
                dummy = prefixes[0].new_zeros(1, prefixes[0].shape[-1])
                prefixes = prefixes + [dummy] * extra
            geometry = (keys_device[:, :, 0] + 1).reshape(-1)
            active_rows = (torch.arange(H, device=d.device)[None, :]
                           < (live_counts if isinstance(live_counts, torch.Tensor)
                              else device_tensor(live_counts, d.device))[:, None]).reshape(-1)
        if graphed:
            self._batch_graph_context(bucket, sequences, prefixes[:count], tails, geometry)
        else:
            bucket.head.update(prefixes, tails, geometry, mark=True,
                               prefix_key=tuple(sequences), early=getattr(self, '_early_batch', None))
        if captured is None:
            preceding = source.gather(1, keys_device[:, :, 0]).reshape(-1, 1)
            first = samples[:, None, :].expand(-1, H, -1).clone()
            first[:, :, 0] = keys_device[:, :, 1]
            first = first.reshape(padded * H, B)
        refresh_ks = list((keys_device[:, :, 0] + 1).unbind(0))
        device_keys = keys_device.reshape(-1, 2)
        pass_rows, row_starts = padded * H, [n * H for n in range(count)]
        if compact:
            # the round's drafted row counts, from the stage's copy home (queued behind the
            # context K/V, so the device keeps working while the host reads them)
            if stages is None:
                requested_home = stage_device.home_read()[0][:count].tolist()
                counts = [max(1, min(int(value), stage_device.cap)) for value in requested_home]
            else:
                counts = list(live_counts[:count])
            counts += [0] * (padded - count)
            live_total = sum(counts)
            pass_rows = live_row_bucket(live_total, padded * H, rows_live_quantum(padded * H))
            rows_pass = live_passes.get(pass_rows)
            if rows_pass is None:  # a bucket built before a quantum change: build it now
                rows_pass = live_passes[pass_rows] = self._live_row_pass(bucket, pass_rows)
            row_starts = [sum(counts[:n]) for n in range(count)]
            if os.environ.get('JETSPEC_BATCH_ROWS_LIVE_CHECK', '0') == '1':
                padded_inputs = (preceding, first, device_keys, active_rows)
            row_order, (preceding, first, device_keys, active_rows) = bucket.live_shelf.write_order(
                live_row_order(counts, H), pass_rows, bucket.head,
                (preceding, first, device_keys, active_rows))
        split_end('row_assembly', assembly_start)
        refresh_states = [{} for _ in sequences]
        refresh_at = refresh_pass_indices(refresh_mode)
        mask_hits = []
        parsed = {}
        last_pass = (passes if passes is not None else d.convergence_limit) - 1

        def parse_stages():
            if finish_stages is not None and 'stages' not in parsed:
                parsed['stages'] = finish_stages()

        def forward(*, fill_tokens, pass_index):
            if rows_pass.wide:
                ids = first if fill_tokens is None else fill_tokens
            else:
                ids = first if fill_tokens is None else torch.cat((fill_tokens, rows_pass.extra_tokens), dim=1)
            mask_hits.append((ids == int(d.head.mask_token_id)).any())
            if refresh and pass_index in refresh_at:
                refresh_start = split_start()
                again = pass_index > refresh_at[0]
                previous = torch.cat([state.pop('logits') for state in refresh_states])
                result = self._adapter_refresh_batch(
                    bucket, sequences, prefixes, torch.cat(refresh_ks), ids, previous, again=again)
                for n, state in enumerate(refresh_states):
                    state['khat'] = result['khat'][n * H:(n + 1) * H]
                    if 'tokens' in result:
                        state['tokens'] = result['tokens']
                split_end('refresh', refresh_start)
            jacobi_start = split_start()
            logits = rows_pass(ids)
            split_end('jacobi', jacobi_start)
            if pass_index >= last_pass:
                parse_stages()  # the stage's host checks run while the last pass computes
            if refresh and pass_index + 1 in refresh_at:
                for n, state in enumerate(refresh_states):
                    state['logits'] = logits[n * H:(n + 1) * H]
            return DraftForwardPass(logits=logits, input_tokens=ids if rows_pass.wide else ids[:, :15],
                                    hidden=rows_pass.hidden)

        combined = packed_refinement(
            forward, anchors=preceding, hypotheses=list(range(pass_rows)), sequences=1,
            passes=passes, convergence_limit=d.convergence_limit, abort_if=abort_if,
            output_indices=list(range(pass_rows)), pinned_tokens=device_keys[:, 1],
            fast_top1=graphed,
            active_rows=active_rows, predictor=self.predictor, posterior=posterior_mode(),
            correction_head=d.correction_head,
            device_metadata=publish_wire is not None,
            **({'truncate_if': self.deadline.truncate_if(passes)}
               if getattr(self, 'deadline', None) is not None and passes is not None else {}))
        if finish_stages is not None:
            parse_stages()
            stages = parsed['stages']
            if stages is None:
                return None  # the host full-row fallback, as the undeferred stage returns
        if publish_wire is not None:
            publish_wire.check_word.copy_(torch.stack(mask_hits).any().long())
        elif mask_hits and bool(torch.stack(mask_hits).any()):
            raise ValueError('Jacobi input must be mask-free')
        all_rows = combined[0]
        if compact and os.environ.get('JETSPEC_BATCH_ROWS_LIVE_CHECK', '0') == '1':
            self._check_live_rows(bucket, padded_inputs, all_rows, row_order, live_total, passes)
        kpred_calls, kpred_total_ms = ((0, 0.0) if self.predictor is None else
                                      self.predictor.consume_metrics())
        record_root = os.environ.get('W280_RECORD')
        recorded = {}
        batched_offers = {}
        if record_root and os.environ.get('JETSPEC_CHAIN_WIRE_RECORD', '0') != '1':
            from jetspec.engines.jetspec_plus_plus.models.heads.decoder import rejected_tokens
            from jetspec.engines.jetspec_plus_plus.models.heads.wirecal import anchor_evidence_batch; from jetspec.engines.jetspec_plus_plus.models.heads.wirecal import append_proposal_record; from jetspec.engines.jetspec_plus_plus.models.heads.wirecal import proposal_record
            for n, (sequence, stage) in enumerate(zip(sequences, stages)):
                if (hasattr(sequence, '_jetspec_seq_id')
                        and hasattr(sequence, '_jetspec_candidate_round')):
                    recorded[n] = rejected_tokens(sequence.pending_block,
                                                  after_miss=stage['after_miss'])
            if len(recorded) > 1:
                indices = list(recorded)
                offers = anchor_evidence_batch(
                    [predictions[n].logits for n in indices],
                    [recorded[n] for n in indices], 32)
                batched_offers = dict(zip(indices, offers))
        output = []
        fields = ('predictor_hidden', 'predictor_inputs', 'predictor_outputs',
                  'predictor_p_top1', 'posterior_probabilities')
        packed = None
        if graphed and H == capacity:
            # JETSPEC_BATCH_DRAFT_GRAPH: one copy of every request's rows (the padded rows of
            # the fixed bucket are already -1; an empty round's private row is hidden), and the
            # per-request tensors below are views of it (the same values as the clones)
            kept = pass_rows if compact else count * H
            wires = all_rows.block_wire[:kept].clone()
            if publish_wire is not None:
                blocks_view, keys_view = publish_wire.buffers[2], publish_wire.buffers[5]
                if (blocks_view.shape[1:] != (H, wires.shape[-1]) or
                        keys_view.shape[1:] != stage_device.keys.shape[1:] or
                        padded > blocks_view.shape[0]):
                    raise ValueError('JETSPEC_DEVICE_PUBLISH packet shape differs from row stage')
            if compact:
                # the compact rows back at their bucket rows; every row the pass left out and
                # every inactive compact row is -1, as the fixed-cap pass leaves a dead row
                candidates = (blocks_view[:padded].view(padded * H, -1)
                              if publish_wire is not None else
                              wires.new_full((padded * H, wires.shape[-1]), -1))
                if publish_wire is not None:
                    candidates.fill_(-1)
                candidates.index_copy_(0, row_order, wires)
                candidates = candidates[:count * H].view(count, H, -1)
            else:
                candidates = (blocks_view[:count] if publish_wire is not None else
                              wires.view(count, H, -1).clone())
                if publish_wire is not None:
                    candidates.copy_(wires.view(count, H, -1))
            empty = [n for n, stage in enumerate(stages) if not len(stage['keys'])]
            if empty:
                candidates[empty, 0] = -1
            packed = dict(wires=wires, candidates=candidates,
                          **{name: getattr(all_rows, name)[:kept].clone() for name in fields
                             if getattr(all_rows, name) is not None})
            if publish_wire is not None:
                keys_view[:count].copy_(stage_device.keys[:count])
                metadata_view = publish_wire.metadata_view[:padded]
                if compact:
                    metadata_view.fill_(-1)
                    metadata_view.view(padded * H, 4).index_copy_(
                        0, row_order, all_rows.metadata[:pass_rows])
                else:
                    metadata_view[:count].copy_(all_rows.metadata[:count * H].view(count, H, 4))
        elif publish_wire is not None:
            raise ValueError('JETSPEC_DEVICE_PUBLISH requires fixed graphed row capacity')
        for n, (sequence, prediction, stage) in enumerate(zip(sequences, predictions, stages)):
            row0 = row_starts[n]  # the request's first row in the pass (n * H without JETSPEC_BATCH_ROWS_LIVE)
            sl = slice(row0, row0 + H)
            live = len(stage['keys'])
            if packed is not None:
                wire = packed['wires'][sl][:live]
                candidate = packed['candidates'][n]
            else:
                wire = all_rows.block_wire[sl][:live].clone()
                candidate = wire.new_full((capacity, wire.shape[-1]), -1)
                candidate[:live] = wire
            if publish_wire is not None:
                row_keys = publish_wire.buffers[5][n]
            else:
                row_keys = stage['keys'].new_full((capacity, 2), -1)
                row_keys[:live] = stage['keys']


            metadata = all_rows.metadata[sl][:live]
            result = PackedBlocks(wire, None if all_rows.logits is None else all_rows.logits[sl][:live],
                                  metadata if publish_wire is not None else metadata.clone(),
                                  stage['keys'][:, 0].tolist(),
                                  candidate_wire=candidate, row_keys=row_keys,
                                  predictor=self.predictor, **kwargs(slice(row0, row0 + live)))
            fallback = None
            if publish_wire is None or not live:
                fallback = PackedBlocks(packed['wires'][row0:row0 + 1] if packed is not None
                                        else all_rows.block_wire[row0:row0 + 1].clone(),
                                        None if all_rows.logits is None else all_rows.logits[row0:row0 + 1],
                                        (all_rows.metadata[row0:row0 + 1] if publish_wire is not None
                                         else all_rows.metadata[row0:row0 + 1].clone()),
                                        [int(stage['keys'][0, 0]) if live else 0], predictor=self.predictor,
                                        **kwargs(slice(row0, row0 + 1)))
            result.posterior = True
            if fallback is not None:
                if os.environ.get('JETSPEC_HEADS_FAST') == '1':
                    result._fallback_blocks = fallback
                else:
                    result.fallback = fallback[0]
            metrics = dict(
                decoder_ready=True, decoder_ms=prediction.elapsed_ms,
                decoder_total_ms=prediction.total_ms, decoder_prefix_ms=prediction.prefix_ms,
                decoder_select_ms=prediction.select_ms,
                kpred_ms=float(sequence.pending_block.kpred_ms), kpred_calls=kpred_calls,
                kpred_total_ms=kpred_total_ms, lac_fallbacks=prediction.offers.lac_fallbacks,
                kpred_pass=sequence.pending_block.kpred_pass, rows_drafted=H,
                rows_published=live, rows_truncated=stage['truncated'],
                rows_requested=stage['requested'], k_set_size=len(stage['ks']),
                k_set=list(map(int, stage['ks'])), anchor_set_sizes=stage['sizes'],
                k15_dropped=0, kpred_frame_dropped0=0,
                ctx_update=bucket.head.last_ctx_update, ctx_rows=bucket.head.last_ctx_rows,
                bucket_built=int(built), row_layout='anchor0')
            if refresh:
                state = refresh_states[n]
                metrics.update(refresh_rounds=int('khat' in state),
                               refresh_skipped=int(state.get('skipped', 0)),
                               refresh_tokens=state.get('tokens'),
                               refresh_khat_mean=(float(state['khat'][:max(1, live)].double().mean())
                                                  if 'khat' in state else None))
            result.head_metrics = metrics
            if n in recorded:
                rejected = recorded[n]
                record = proposal_record(
                    seq_id=sequence._jetspec_seq_id,
                    candidate_round=sequence._jetspec_candidate_round,
                    pass_label=sequence.pending_block.kpred_pass,
                    predictor_probabilities=sequence.pending_block.k_probabilities,
                    anchor_logits=prediction.logits, rejected=rejected,
                    after_miss=stage['after_miss'], k_set=stage['ks'],
                    rows_requested=stage['requested'], rows_published=live,
                    rows_truncated=stage['truncated'],
                    fallback_anchor=int(sequence.pending_block.correction_guess),
                    published_keys=stage['keys'], batched_offers=batched_offers.get(n))
                passes_run = int(sequence.pending_block.passes_run)
                record.update(passes_run=passes_run, raw_pass_label=f'J{passes_run - 1}',
                              served_pass_label=sequence.pending_block.kpred_pass)
                if os.environ.get('E04_RECORD_PTOP1'):
                    record = e04_record_p_top1(record, sequence.pending_block)
                if os.environ.get('E04_RECORD_QSHADOW'):
                    record = e04_record_q_shadow(record, prediction.logits, rejected)
                append_proposal_record(record_root, record)
            output.append(result)
        host_mark('flat_ret')
        if publish_wire is not None:
            publish_wire._device_publish_ready = True
        if split_events is not None:
            split_events['adapter'] = self.__dict__.pop('_w434_adapter_events', [])
            split_events['prefetch'] = self.__dict__.pop('_w434_prefetch_events', [])
            torch.cuda.synchronize(d.device)
            values = {name + '_ms': round(sum(start.elapsed_time(end) for start, end in events), 3)
                      for name, events in split_events.items()}
            if compact:
                values.update(rows_live=live_total, rows_pass=pass_rows)
            early = getattr(self, '_early_batch', None)
            if early is not None and early.start_event is not None:
                values.update(context_early_ms=round(early.start_event.elapsed_time(early.end_event), 3),
                              context_start_to_decoder_ms=round(
                                  early.start_event.elapsed_time(device_split[0]), 3),
                              decoder_to_context_end_ms=round(
                                  device_split[0].elapsed_time(early.end_event), 3))
            self._w434_split_rounds = getattr(self, '_w434_split_rounds', 0) + 1
            print('W434_DEVICE_SPLIT ' + json.dumps(dict(C=count,
                  round=self._w434_split_rounds, **values)), flush=True)
        return output

    def _check_live_rows(self, bucket, padded_inputs, compact_rows, row_order, live_total, passes):
        """JETSPEC_BATCH_ROWS_LIVE_CHECK=1 (diagnostic, slow): the round's fixed row-cap passes
        on the same inputs after the compact ones, and every live row's published tokens,
        metadata and top-1 probabilities compared; a running W470_ROWS_CHECK line (the round
        serves the compact rows either way)."""
        d = self.drafter
        preceding, first, device_keys, active_rows = padded_inputs

        def forward(*, fill_tokens, pass_index):
            ids = first if fill_tokens is None else (
                fill_tokens if bucket.wide else torch.cat((fill_tokens, bucket.extra_tokens), dim=1))
            return DraftForwardPass(logits=bucket(ids), input_tokens=ids if bucket.wide else ids[:, :15],
                                    hidden=bucket.hidden)

        rows = int(first.shape[0])
        full = packed_refinement(
            forward, anchors=preceding, hypotheses=list(range(rows)), sequences=1, passes=passes,
            convergence_limit=d.convergence_limit, output_indices=list(range(rows)),
            pinned_tokens=device_keys[:, 1], fast_top1=True, active_rows=active_rows,
            predictor=self.predictor, posterior=posterior_mode(), correction_head=d.correction_head)[0]
        live = row_order[:live_total]
        tokens = int((full.block_wire.index_select(0, live) != compact_rows.block_wire[:live_total]
                      ).any(1).sum())
        meta = int((full.metadata.index_select(0, live.cpu()) != compact_rows.metadata[:live_total]
                    ).any(1).sum())
        top1 = 0.0
        if full.predictor_p_top1 is not None:
            top1 = float((full.predictor_p_top1.index_select(0, live).float()
                          - compact_rows.predictor_p_top1[:live_total].float()).abs().max())
        tally = self.__dict__.setdefault('_rows_check', dict(rounds=0, rows=0, token_rows=0,
                                                             metadata_rows=0, max_top1_diff=0.0))
        tally['rounds'] += 1
        tally['rows'] += int(live_total)
        tally['token_rows'] += tokens
        tally['metadata_rows'] += meta
        tally['max_top1_diff'] = max(tally['max_top1_diff'], top1)
        if tokens or meta or top1 or tally['rounds'] % 100 == 1:
            print('W470_ROWS_CHECK ' + json.dumps(dict(C=int(bucket.head.N), **tally)), flush=True)









    @torch.inference_mode()
    def _adapter_refresh_batch(self, bucket, sequences, prefixes, ks, ids, logits, *, again=False):
        """Refresh C private row windows in one masked, uncached C1 adapter call."""
        from jetspec.engines.jetspec_plus_plus.draft.drafter import ServingCtxKVC1Head
        d = self.drafter
        count, H = len(sequences), bucket.head.H
        head = d.adapter.head
        if not isinstance(head, ServingCtxKVC1Head):
            raise ValueError('batched refresh requires the serving C1 adapter')
        if any(sequence.staged.start_position != prefix.shape[0] or
               sequence.staged.token_ids is None for sequence, prefix in zip(sequences, prefixes)):
            raise RuntimeError('batched refresh requires every request staged at its exact prefix')
        tails = torch.cat([sequence.staged.token_ids.to(d.device) for sequence in sequences])
        T = tails.shape[1]
        B = getattr(d, 'block_size', ids.shape[1])
        width = T + H * B
        block_ids = ids.reshape(count, H * B)
        embedded = d.embed_tokens(torch.cat((tails, block_ids), dim=1))
        lengths = device_tensor([prefix.shape[0] for prefix in prefixes], d.device)
        padded = torch.nn.utils.rnn.pad_sequence(prefixes, batch_first=True)
        P = padded.shape[1]
        positions = torch.arange(P, device=d.device)
        tail_positions = torch.arange(T, device=d.device)
        block_slots = torch.arange(B, device=d.device)
        row_ks = ks.reshape(count, H)
        query_positions = torch.cat((
            lengths[:, None] + tail_positions,
            (lengths[:, None, None] + row_ks[:, :, None] + block_slots).reshape(count, H * B)),
            dim=1)
        query = torch.arange(width, device=d.device)
        key = query
        tail_spans = torch.cat((tail_positions.expand(count, -1),
                                row_ks.repeat_interleave(B, dim=1)), dim=1)
        tail_allowed = (key[None, None, :] < T) & (key[None, None, :] < tail_spans[:, :, None])
        block_allowed = ((query[None, :, None] >= T) & (key[None, None, :] >= T)
                         & ((query[None, :, None] - T) // B == (key[None, None, :] - T) // B)
                         & ((key[None, None, :] - T) % B < (query[None, :, None] - T) % B))
        prefix_allowed = (positions[None] < lengths[:, None])[:, None, :].expand(-1, width, -1)
        allowed = torch.cat((prefix_allowed, tail_allowed | block_allowed), dim=2)
        mask = padded.new_zeros((count, 1, width, P + width)).masked_fill_(
            ~allowed[:, None], torch.finfo(d.dtype).min)
        multi_cached = (count > 1 and os.environ.get('JETSPEC_C1_KVCACHE', '0') == '1'
                        and os.environ.get('JETSPEC_C1_MULTI_SLOT', '0') == '1')
        refreshed = head(token_embeddings=embedded, prefix_h=padded,
                         prefix_position_ids=positions[None].expand(count, -1),
                         query_position_ids=query_positions, attention_mask=mask,
                         shared_prefix_projection=True,
                         **({'prefix_keys': sequences, 'prefix_lengths': [p.shape[0] for p in prefixes]}
                            if multi_cached else {'batched_uncached': True}))
        blocks = refreshed[:, T:].reshape(count * H, B, -1)
        p_top1 = torch.softmax(logits[:, 1:], -1, dtype=torch.float32).amax(-1)
        cutoff = (refresh_khat_anchor0 if B == 16 else
                  refresh_khat_anchor0_width31 if B == 32 else refresh_khat_anchor0_short)
        khat = cutoff(p_top1, refresh_khat_threshold(again))
        cap = refresh_khat_cap(again)
        if cap is not None:
            khat = khat.clamp(max=cap)
        bucket.head.inject_refresh(blocks, khat, reset=again)
        return dict(khat=khat)

    @torch.inference_mode()
    def _assemble_rows(self, sequence, prediction, *, after_miss, defer=False, reuse_home=False,
                       written=None):
        """JETSPEC_ROWS_ASSEMBLY=1 under E04_ROWRULE=joint: the BLEND k score and the joint
        row rule as one device stage (heads/assembly.py) over the decoder's device offers, and one
        trip home carrying the keys, the requested count, the offer summary, the 16-slot mass and
        the checks the host path raises on.  None (the host path runs) when the round is not a
        live joint-rule round with device offers, or when an offer row needs the exact full-row
        recompute (its tie reaches the top-k slack).  ``written`` (E04_LAB_KSCORE_WRITTEN, default
        None) goes to the normal round's k score (e04_kscore_blend_device)."""
        from jetspec.engines.jetspec_plus_plus.models.heads.assembly import DeviceOffers; from jetspec.engines.jetspec_plus_plus.models.heads.assembly import RowStage; from jetspec.engines.jetspec_plus_plus.models.heads.assembly import joint_rule
        from jetspec.engines.jetspec_plus_plus.models.heads.decoder import DeviceAnchorOffers
        from jetspec.engines.jetspec_plus_plus.models.heads.fast import read_thresholds
        live = hasattr(sequence, '_jetspec_seq_id') and hasattr(sequence, '_jetspec_candidate_round')
        if (os.environ.get('E04_ROWRULE') != 'joint' or os.environ.get('E04_KSCORE') not in ('blend', 'blend_dec')
                or prediction is None or not live or not isinstance(prediction.offers, DeviceAnchorOffers)
                or not decoder_blend_kset() or os.environ.get('JETSPEC_KPRED_FRAME') != 'wire'
                or os.environ.get('JETSPEC_DECODER_FRAME') != 'wire'
                or os.environ.get('JETSPEC_KSET_AFTER_MISS') != 'zero'
                or os.environ.get('E04_RECORD_PTOP1') != '1' or os.environ.get('E04_RECORD_QSHADOW') != '1'
                or os.environ.get('JETSPEC_ANCHOR_CAP') != str(E04_ROWRULE_POOL)):
            return None
        block = sequence.pending_block
        if rows_static_chain() and defer:
            from jetspec.engines.jetspec_plus_plus.wire.chain.contract import RoundParams; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import TokenBuffers; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import rng_words
            ids, p64, pack = prediction.offers.device
            device = p64.device
            lanes = ids.shape[0]
            tokens = getattr(sequence, '_jetspec_chain_tokens', None)
            if tokens is None:
                tokens = TokenBuffers(sequence.committed_ids.reshape(-1).to(device),
                                      block.draft_ids.to(device))
            dataset = os.environ['JETSPEC_DATASET']
            from jetspec.engines.jetspec_plus_plus.models.heads.fast import dataset_entry; from jetspec.engines.jetspec_plus_plus.models.heads.fast import read_thresholds
            thresholds = read_thresholds(os.environ['JETSPEC_HEADS_DIR'])
            values = dataset_entry(thresholds['e04_rowrule_joint'], dataset)
            tau = device_tensor([resolve_tau(values['tau'], str(block.kpred_pass))],
                                device, torch.float64)
            params = RoundParams(sequence.committed_ids.numel(), 0, 0, 0, 0, int(block.correction_guess),
                                 int(after_miss), 0,
                                 (0,) * rng_words(lanes), 0).pack()
            rp = params.pin_memory().to(device, non_blocking=True) if device.type == 'cuda' else params
            cap = rows_rule_cap()
            R = 1 << (cap - 1).bit_length()
            words = rows_chain_home_words(R, lanes)
            out = dict(keys=torch.empty(R, 2, dtype=torch.long, device=device),
                       active=torch.empty(R, dtype=torch.bool, device=device),
                       requested=torch.empty(1, dtype=torch.long, device=device),
                       row_home=torch.empty(words, dtype=torch.long, device=device),
                       err=torch.empty(1, dtype=torch.long, device=device))
            previous_mass = block.k_probabilities
            rows_chain_body(
                rp, tokens, ids, p64, pack, prediction.logits, block.e04_p_top1,
                R, thresholds=thresholds, dataset=dataset,
                tau_table=tau, temperature=float(values['temperature']['value']),
                out=out,
                previous_mass16=previous_mass.to(device).double().reshape(-1)
                if previous_mass is not None else None)
            stage = SimpleNamespace(keys=out['keys'], requested=out['requested'][0])
            with torch.inference_mode(False):
                host = torch.empty(words, dtype=torch.long, pin_memory=device.type == 'cuda')
            pending = None



            return stage, start_home, finish
        if after_miss and block.k_probabilities is None:
            return None
        ids, p64, pack = prediction.offers.device
        device = p64.device
        dataset = os.environ['JETSPEC_DATASET']
        thresholds = read_thresholds(os.environ['JETSPEC_HEADS_DIR'])
        from jetspec.engines.jetspec_plus_plus.models.heads.fast import dataset_entry
        rule_values = dataset_entry(thresholds['e04_rowrule_joint'], dataset)
        rule = joint_rule(resolve_tau(rule_values['tau'], str(block.kpred_pass)),
                          rule_values['temperature']['value'])
        pool = E04_ROWRULE_POOL
        sizes = pack[0]
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        rows = [0] if after_miss else list(range(draft_width() + 1))
        if after_miss:
            # e04_kscore_blend returns the set and mass unchanged after a miss; the rule scores
            # k = 0 alone with P(0) = 1
            mass16 = block.k_probabilities.to(device).double().reshape(-1)
            mass = torch.ones(1, dtype=torch.float64, device=device)
            checks = torch.zeros(1, dtype=torch.long, device=device)
        else:
            mass16, checks = e04_kscore_blend_device(block, prediction.logits, p64, sizes, dataset=dataset,
                                                     thresholds=thresholds,
                                                     **({'written': written} if written is not None else {}))
            mass = mass16[1:]
        index = torch.tensor(rows, dtype=torch.long, device=device)
        offers = DeviceOffers(ids=ids.index_select(0, index)[:, :pool],
                              probabilities=p64.index_select(0, index)[:, :pool],
                              valid=torch.arange(pool, device=device)[None, :] < sizes.index_select(0, index)[:, None],
                              ks=index)
        score, passing = rule(mass, offers)
        cap = rows_rule_cap()
        stage = RowStage(score, passing, offers, cap=cap, capacity=1 << (cap - 1).bit_length(),
                         graph=False if os.environ.get('TORCHDYNAMO_DISABLE') == '1' else None)
        score_bad = (~torch.isfinite(score)).any().long().reshape(1)
        if defer:
            # The checks travel as one device-computed flag tensor in the same async trip.
            # Only the selected keys and offer summary need host parsing after the passes.
            device_checks = reuse_home and rows_device_checks()
            flags = torch.stack((
                (~pack[3].bool()).any(), pack[4].bool().any(),
                checks[0].bool() if not after_miss else torch.zeros((), dtype=torch.bool, device=device),
                checks[1].bool() if not after_miss else torch.zeros((), dtype=torch.bool, device=device),
                ((~torch.isfinite(mass16)).any() if not after_miss else
                 torch.zeros((), dtype=torch.bool, device=device)),
                (pack[0].index_select(0, index) < pool).any(),
                score_bad[0].bool(),
            )).long() if device_checks else None
            host = None
            if reuse_home:
                size = 1 + 2 * stage.capacity + sum(x.numel() for x in
                                                     ((pack, mass16, flags) if device_checks else
                                                      (pack, mass16, checks, score_bad)))
                cache = self.__dict__.setdefault('_row_home_buffers', {})
                key = (str(device), size)
                host = cache.get(key)
                if host is None:
                    host = torch.empty(size, dtype=torch.long, pin_memory=device.type == 'cuda')
                    cache[key] = host
            pending = None


            return stage, start_home, finish
        done = KthRowsDrafter._finish_rows(prediction, stage.home(pack, mass16, checks, score_bad), rows=rows,
                                 pool=pool, cap=cap, after_miss=after_miss)
        return None if done is None else done + (stage.keys,)



    @torch.inference_mode()
    def _adapter_refresh(self, bucket, sequence, length, ks, ids, logits, again=False,
                         defer_inject=False, lab=None, pass_index=None):
        """JETSPEC_ADAPTER_REFRESH=after_j0: re-predict every row's block from
        [staged tail up to its k, pass-0 block] and inject entries 0..k-hat.

        ``lab`` (E04_LAB_*, tau_lab.LabRound; None by default): after the refreshed rows exist,
        ``tau_lab.refresh_hook`` lowers k-hat (E04_LAB_REFRESH_CUTOFF) and may swap one draft per
        row (E04_LAB_SWAP=first_wrong_refresh, returned as ``lab_ids``); the rows are computed
        with the draft k-hat's counts, and the lowered k-hat is the one injected.  The hook also
        reads ``logits`` (E04_LAB_RECORD_TOPK: the drafter's top-n of the pass whose block it read)."""
        d = self.drafter
        staged = sequence.staged
        if staged.start_position != length or staged.token_ids is None:
            return dict(skipped=1)
        # the pass-0 top-1 probabilities of slots 2..15 (slot 1 never enters k-hat) in
        # one softmax call: the same kernel on the same rows, so the same values as
        # top1_probabilities(logits, chunk_rows=16)[:, 1:]
        p_top1 = torch.softmax(logits[:, 1:], -1, dtype=torch.float32).amax(-1)
        if row_layout_anchor0():
            # The caller passes k + 1 as each row's tail count; the frame's
            # logits[:, 1:] are drafts 1..14 in block slots 1..14
            refresh_cutoff = (refresh_khat_anchor0 if d.block_size == 16 else
                              refresh_khat_anchor0_width31 if d.block_size == 32 else
                              refresh_khat_anchor0_short)
            khat = refresh_cutoff(
                p_top1, refresh_khat_threshold(again) if again else refresh_khat_threshold())
        else:
            khat = refresh_khat(torch.cat((p_top1[:, :1], p_top1), dim=1),
                                refresh_khat_threshold(again) if again else refresh_khat_threshold())
        khat_cap = refresh_khat_cap(again) if again else None
        if khat_cap is not None:
            khat = khat.clamp(max=khat_cap)
        embedded = d.embed_tokens(ids)  # queued before the host reads the counts
        # JETSPEC_REFRESH_PACK=1: slot i reads entries j < i, so no slot reads an entry
        # past min(k-hat, 14); the window stops there (one host read of the counts).
        pack = refresh_pack_mode()
        fixed_refresh = os.environ.get('JETSPEC_REFRESH_FIXED', '0') == '1'
        if fixed_refresh and (pack != '2' or os.environ.get('JETSPEC_REFRESH_REUSE0', '0') == '1'):
            raise ValueError('JETSPEC_REFRESH_FIXED requires PACK=2 and REUSE0=0')
        counts = khat.clamp(max=d.block_size - 2) + 1
        reuse = None
        if pack == '2' and os.environ.get('JETSPEC_REFRESH_REUSE0', '0') == '1':
            # JETSPEC_REFRESH_REUSE0=1: a row whose block slot 0 holds the staged tail's
            # token k (k < 15) is the staged window slot k: same token, same position,
            # same context (prefix and tail slots < k).  Its entry 0 is the staged row
            # and its later slots read that slot's cached keys; read with the counts.
            tail_ids = staged.token_ids.to(d.device).reshape(-1)
            reuse = ((ks < tail_ids.shape[0])
                     & (tail_ids[ks.clamp(max=tail_ids.shape[0] - 1)] == ids[:, 0])).long()
            counts = counts - reuse
            row_slots, reuse_list = torch.stack((counts, reuse)).tolist()
        elif fixed_refresh:
            row_slots = None
        else:
            row_slots = counts.tolist() if pack != '0' else None
        if pack == '2':
            # the staged tail's keys/values read from the cache, the token count padded
            # to a multiple of 16 (the Triton rotary kernel compiles per count)
            fixed_tokens = refresh_token_bucket(bucket.head.R * (d.block_size - 1)) if fixed_refresh else None
            block = d.adapter.head.refresh_cached(
                block_embeddings=embedded, length=length, tail_counts=ks,
                prefix_key=sequence, row_slots=row_slots, slot_counts=counts,
                pad_to=fixed_tokens if fixed_refresh else refresh_token_bucket(sum(row_slots)),
                **({'fixed_tokens': fixed_tokens} if fixed_refresh else {}),
                **({} if reuse is None else dict(first_slots=reuse)))
            if block is not None and reuse is not None and any(reuse_list):
                # The staged window holds 15 rows, or 16 with JETSPEC_DRAFTS=15
                staged_rows = staged.predicted_hidden[0, ks.clamp(max=staged.token_ids.numel() - 1)]
                block[:, 0] = torch.where(reuse.bool()[:, None], staged_rows.to(block.dtype),
                                          block[:, 0])
            if block is not None:
                entries = None
                if not fixed_refresh and os.environ.get('JETSPEC_REFRESH_INJECT_PACK', '0') == '1':
                    # entries 0..min(k-hat,14) of each row: every entry the mask exposes
                    visible = khat.clamp(max=d.block_size - 2) + 1
                    total = sum(row_slots) + (0 if reuse is None else sum(reuse_list))
                    row = torch.repeat_interleave(
                        torch.arange(visible.shape[0], device=visible.device), visible,
                        output_size=total)
                    entries = (row * d.block_size + torch.arange(total, device=visible.device)
                               - (torch.cumsum(visible, 0) - visible)[row])
                lab_ids = None
                if lab is not None:
                    # E04_LAB_*: the entries above stay the draft k-hat's (a superset of what
                    # the lowered k-hat exposes); the mask exposes entries 0..khat_used
                    from jetspec.engines.jetspec_plus_plus.draft.tau_lab import refresh_hook
                    khat, lab_ids = refresh_hook(
                        getattr(self, 'decoder', None), sequence, length=length, tail_counts=ks,
                        ids=ids, khat=khat, block=block, lab=lab, pass_index=pass_index,
                        draft_logits=logits)
                if not defer_inject:
                    bucket.head.inject_refresh(block, khat,
                                               **({} if entries is None else dict(entries=entries)),
                                               **({'fixed_layout': True} if fixed_refresh else {}),
                                               **({'reset': True} if again else {}))
                return dict(khat=khat, tokens=fixed_tokens if fixed_refresh else sum(row_slots),
                            **({'hidden': block} if defer_inject else {}),
                            **({'lab_ids': lab_ids} if lab_ids is not None else {}))
            if fixed_refresh:
                raise RuntimeError('fixed refresh cache is unavailable for this prefix')
            if os.environ.get('JETSPEC_ROWS_FUSED_GRAPH', '0') == '1':
                raise RuntimeError('fused graph adapter refresh has no cached prefix or capacity')
            row_slots = None  # no room after the cached tail: the full window
        rows = d.adapter.head.refresh_window(
            tail_embeddings=d.embed_tokens(staged.token_ids.to(d.device))[0],
            block_embeddings=embedded, length=length, tail_counts=ks,
            prefix_key=sequence, **({} if row_slots is None else dict(row_slots=row_slots)))
        if rows is None:
            return dict(skipped=1)
        lab_ids = None
        if lab is not None:
            from jetspec.engines.jetspec_plus_plus.draft.tau_lab import refresh_hook
            khat, lab_ids = refresh_hook(
                getattr(self, 'decoder', None), sequence, length=length, tail_counts=ks,
                ids=ids, khat=khat, block=rows[1], lab=lab, pass_index=pass_index, draft_logits=logits)
        if not defer_inject:
            bucket.head.inject_refresh(rows[1], khat, **({'reset': True} if again else {}))
        return dict(khat=khat, **({'hidden': rows[1]} if defer_inject else {}),
                    **({'lab_ids': lab_ids} if lab_ids is not None else {}))


    def stage_admission_batch(self, sequence, blocks):
        """Stage one prefill prefix against its published row blocks in one adapter call."""
        if os.environ.get('JETSPEC_CHAIN_ADMIT_GRAPH', '0') == '1' and self._admission_fast_ok(blocks):
            return self._stage_admission_fast(sequence, blocks)
        return self._stage_packed([sequence] * len(blocks), blocks,
                                  batched_uncached=True, admission=True)

    def _admission_fast_ok(self, blocks):
        from jetspec.engines.jetspec_plus_plus.draft.drafter import ServingCtxKVC1Head
        return (len(blocks) > 1 and isinstance(self.drafter.adapter.head, ServingCtxKVC1Head)
                and os.environ.get('JETSPEC_DEVICE_SPLIT', '0') != '1')

    def _admission_forward(self, exact, ids):
        """_stage_packed(admission=True)'s adapter call on one shared prefix, as a pure function.

        The same operations on the same shapes: ``exact`` is the [1, P, H] exact prefix and
        ``ids`` the [rows, width + 1] (correction, drafts) block per row. Every row has
        length P, so the length tensor is P everywhere and h_prev is the last prefix row.
        """
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        d = self.drafter
        count = ids.shape[0]
        prefix = exact.expand(count, -1, -1)
        capacity = prefix.shape[1]
        positions = torch.arange(capacity, device=d.device)
        length_tensor = torch.full((count,), capacity, dtype=torch.long, device=d.device)
        width = draft_width()
        query = torch.arange(width + 1, device=d.device)
        allowed_prefix = (positions[None] < length_tensor[:, None])[:, None, :].expand(-1, width + 1, -1)
        if d.adapter.head.resolved_config.get('window_causal_over_tokens', False):
            allowed_query = (query[:, None] > query[None, :]).expand(count, -1, -1)
        else:
            allowed_query = (query[:, None] == query[None, :]).expand(count, -1, -1)
        allowed = torch.cat((allowed_prefix, allowed_query), dim=-1)[:, None]
        mask = prefix.new_zeros(allowed.shape).masked_fill_(~allowed, torch.finfo(d.dtype).min)
        previous = prefix[torch.arange(count, device=d.device), length_tensor-1][:, None]
        return d.adapter.head(token_embeddings=d.embed_tokens(ids), h_prev=previous,
            prefix_h=prefix, prefix_position_ids=positions[None].expand(count, -1),
            query_position_ids=length_tensor[:, None]+query, attention_mask=mask,
            shared_prefix_projection=True, batched_uncached=True, prefix_key=None)

    @torch.inference_mode()
    def prewarm_admission_graph(self, length, count):
        """JETSPEC_CHAIN_ADMIT_GRAPH=1: capture the admission adapter call for one prompt length.

        Startup work, outside every measured round. The replay is checked byte for byte
        against the eager call on random inputs before the graph is kept.
        """
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        d = self.drafter
        key = (int(length), int(count))
        graphs = self.__dict__.setdefault('_admission_graphs', {})
        if d.device.type != 'cuda' or key in graphs:
            return False
        torch.cuda.synchronize(d.device)
        started = time.perf_counter()
        allocated_before = torch.cuda.memory_allocated(d.device)
        reserved_before = torch.cuda.memory_reserved(d.device)
        head = d.adapter.head
        exact = torch.zeros(1, key[0], head.hspace_dim, device=d.device, dtype=d.dtype)
        ids = torch.zeros(key[1], draft_width() + 1, dtype=torch.long, device=d.device)
        pool = self.__dict__.setdefault('_admission_graph_pool', torch.cuda.graph_pool_handle())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._admission_forward(exact, ids)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            output = self._admission_forward(exact, ids)
        generator = torch.Generator(device=d.device)
        generator.manual_seed(key[0] * 131 + key[1])
        exact.copy_(torch.randn(exact.shape, generator=generator, device=d.device, dtype=torch.float32))
        ids.random_(0, d.embed_tokens.num_embeddings, generator=generator)
        expected = self._admission_forward(exact, ids).clone()
        graph.replay()
        if not torch.equal(output, expected):
            raise RuntimeError(f'chain admission graph replay byte parity failed for {key}')
        graphs[key] = SimpleNamespace(graph=graph, exact=exact, ids=ids, output=output)
        torch.cuda.synchronize(d.device)
        print(f'CHAIN_ADMIT_CAPTURE prefix={key[0]} rows={key[1]} '
              f'ms={(time.perf_counter() - started) * 1000:.3f} '
              f'allocated_bytes={torch.cuda.memory_allocated(d.device) - allocated_before} '
              f'reserved_bytes={torch.cuda.memory_reserved(d.device) - reserved_before}',
              flush=True)
        return True

    @torch.inference_mode()
    def _stage_admission_fast(self, sequence, blocks):
        """stage_admission_batch with the prewarmed graph of this prompt length, else eager."""
        from jetspec.engines.jetspec_plus_plus.draft.drafter import ServingCtxKVC1Head
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        d = self.drafter
        if any(b.draft_ids.numel() != draft_width() for b in blocks):
            raise ValueError('live kth staging requires 14 draft ids')
        exact = sequence.exact_hidden(device=d.device, dtype=d.dtype)
        count = len(blocks)
        ids = torch.cat((sequence.committed_ids[:, -1:].to(d.device).expand(count, -1),
                         torch.stack([b.draft_ids.to(d.device) for b in blocks])), dim=1)
        entry = self.__dict__.get('_admission_graphs', {}).get((exact.shape[1], count))
        if entry is None:
            predicted = self._admission_forward(exact, ids)
        else:
            entry.exact.copy_(exact)
            entry.ids.copy_(ids)
            entry.graph.replay()
            predicted = entry.output.clone()
        counts = self.__dict__.setdefault('_admission_graph_counts', {'graph': 0, 'fallback': 0})
        counts['graph' if entry is not None else 'fallback'] += 1
        print(f'CHAIN_ADMIT_GRAPH prefix={exact.shape[1]} rows={count} '
              f'graphed={entry is not None} graph_total={counts["graph"]} '
              f'fallback_total={counts["fallback"]}', flush=True)
        if os.environ.get('JETSPEC_CHAIN_ADMIT_GRAPH_CHECK', '0') == '1':
            # Check every admission, including an unseen-length eager fallback.
            same = torch.equal(predicted, self._admission_forward(exact, ids))
            mismatches = self.__dict__.get('_admission_graph_mismatches', 0) + int(not same)
            self._admission_graph_mismatches = mismatches
            print(f'CHAIN_ADMIT_GRAPH_CHECK prefix={exact.shape[1]} rows={count} '
                  f'graphed={entry is not None} equal={same} mismatches={mismatches}',
                  flush=True)
            if not same:
                raise RuntimeError('chain admission graph differs from the eager admission call')
        sequence.staged.stage(start_position=exact.shape[1], token_ids=ids[-1:],
                              predicted_hidden=predicted[-1:])
        sequence.pending_block = blocks[-1]
        if isinstance(d.adapter.head, ServingCtxKVC1Head):
            d.adapter.head.prime_admission_prefix(sequence, exact)
        return predicted


    @torch.inference_mode()
    def _stage_host_fast(self, sequences, blocks):
        from jetspec.engines.jetspec_plus_plus.draft.drafter import ServingCtxKVC1Head
        d = self.drafter
        seq, block = sequences[0], blocks[0]
        # The window is [c, d1..d_width] (16 rows with JETSPEC_DRAFTS=15), as _stage_packed
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        width = draft_width()
        if block.draft_ids.numel() != width:
            raise ValueError('live kth staging requires JETSPEC_DRAFTS (14) draft ids')
        prefix = seq.exact_hidden(device=d.device, dtype=d.dtype)
        length = prefix.shape[1]
        query = width + 1
        if (length, query) not in self._host_stage_cache:
            positions = torch.arange(length + query, device=d.device)
            allowed = torch.cat((torch.ones(query, length, dtype=torch.bool, device=d.device),
                                 torch.eye(query, dtype=torch.bool, device=d.device)), dim=1)[None, None]
            mask = prefix.new_zeros(allowed.shape).masked_fill_(~allowed, torch.finfo(d.dtype).min)
            self._host_stage_cache[(length, query)] = (positions, mask)
        positions, mask = self._host_stage_cache[(length, query)]
        ids = torch.cat((seq.committed_ids[:, -1:].to(d.device), block.draft_ids.reshape(1, -1).to(d.device)), dim=1)
        host_mark('adapter_call')
        predicted = d.adapter.head(token_embeddings=d.embed_tokens(ids), h_prev=prefix[:, -1:],
            prefix_h=prefix, prefix_position_ids=positions[:length][None],
            query_position_ids=positions[length:][None], attention_mask=mask, shared_prefix_projection=True,
            **({"prefix_key": seq} if isinstance(d.adapter.head, ServingCtxKVC1Head) else {}))
        host_mark('adapter_ret')
        seq.staged.stage(start_position=length, token_ids=ids, predicted_hidden=predicted)
        seq.pending_block = block

    @torch.inference_mode()
    def _propose_bucketed(self, sequences, *, k_hypotheses, passes, abort_if=None,
                          output_hypotheses=None):
        d = self.drafter
        ks = list(map(int,k_hypotheses))
        posterior = posterior_mode()
        wire_capacity = dynamic_wire_capacity(
            posterior=posterior, predictor=self.predictor)
        if (not ks and not posterior) or len(set(ks)) != len(ks) or any(k<0 or k>(14 if posterior else 15) for k in ks):
            raise ValueError('live kth hypotheses must be unique within 0..15')
        N,H=len(sequences),len(ks)
        Nb,Hb=1<<(N-1).bit_length(),wire_capacity or 1<<(H-1).bit_length()
        padded_states=list(sequences)+[sequences[-1]]*(Nb-N)
        padded_ks = bucket_hypotheses(
            ks + [k for k in range(16) if k not in ks][:Hb-H],
            capacity=wire_capacity)
        prefixes=[s.exact_hidden(device=d.device,dtype=d.dtype)[0] for s in padded_states]
        P=((max(p.shape[0] for p in prefixes)+255)//256)*256
        key=(Nb,Hb,P)
        if key not in self.buckets:
            if len(self.buckets)==2:
                self.buckets.popitem(last=False)
            self.buckets[key]=BucketPass(d,N=Nb,H=Hb,prefix_bucket=P,backend=self.backend,host_fast=self.host_fast and Nb==1)
        bucket=self.buckets[key]
        self.buckets.move_to_end(key)
        if any(s.pending_block is None or s.staged.predicted_hidden is None for s in padded_states):
            raise RuntimeError('live kth proposal requires a staged pending block')
        tails=torch.cat([s.staged.predicted_hidden[:,:15] for s in padded_states])
        bucket.head.update(prefixes,tails,padded_ks,mark=True,
            **({'prefix_key':padded_states[0]} if self.host_fast and Nb==1 else {}))
        # Sequence objects are ingested once; all hypotheses use tensor gathers.
        samples=torch.cat([build_random_context_block_token_ids(s.committed_ids.to(d.device),
            block_size=16,seed=d.seed,prompt_index=s.prompt_index,mode='kth',device_gather=self.host_fast and Nb==1) for s in padded_states])
        committed=torch.cat([s.committed_ids[:,-1:].to(d.device) for s in padded_states])
        drafts=torch.stack([s.pending_block.draft_ids.to(d.device) for s in padded_states])
        anchor_source=torch.cat((committed,drafts,drafts[:,-1:]),dim=1)
        anchors=anchor_source[:,padded_ks].reshape(-1,1)
        first=samples[:,None,:].expand(-1,Hb,-1).reshape(Nb*Hb,16).clone()
        first[:,:1].copy_(anchors)
        def forward(*,fill_tokens,pass_index):
            del pass_index
            ids=first if fill_tokens is None else torch.cat((anchors,fill_tokens),dim=1)
            if bool((ids==int(d.head.mask_token_id)).any()):
                raise ValueError('Jacobi input must be mask-free')
            logits = bucket(ids)
            return DraftForwardPass(logits=logits,input_tokens=ids[:,1:],hidden=bucket.hidden)
        output = packed_refinement(forward,anchors=anchors,hypotheses=padded_ks,sequences=Nb,
            passes=passes,convergence_limit=d.convergence_limit,output_sequences=N,
            output_hypotheses=ks if output_hypotheses is None else output_hypotheses,
            abort_if=abort_if,
            fixed_wire_capacity=wire_capacity,predictor=self.predictor,
            posterior=posterior, correction_head=d.correction_head)
        return self._attach_predictor_metrics(output, sequences, [Hb] * N)


    @torch.inference_mode()
    def _propose_host_branches(self, seq, hypotheses, passes):
        d = self.drafter
        k = int(hypotheses[0])
        if not 0 <= k <= 15:
            raise ValueError('live kth hypotheses must be within 0..15')
        if seq.pending_block is None or seq.staged.predicted_hidden is None:
            raise RuntimeError('live kth proposal requires a staged pending block')
        prefix = seq.exact_hidden(device=d.device, dtype=d.dtype)[0]
        tails = seq.staged.predicted_hidden[:, :15]
        free, _ = self._host_branch_bucket(seq, prefix, tails, k, 1)
        drafts = seq.pending_block.draft_ids.to(d.device).reshape(1, -1)
        anchor_source = torch.cat((seq.committed_ids[:, -1:].to(d.device), drafts, drafts[:, -1:]), dim=1)
        anchors = anchor_source[:, k:k+1]
        first = build_random_context_block_token_ids(seq.committed_ids.to(d.device),
            block_size=16, seed=d.seed, prompt_index=seq.prompt_index, mode='kth', device_gather=True)
        first[:, :1].copy_(anchors)
        rejected = drafts[:, k] if k < 14 else drafts.new_full((1,), -1)
        def forward(*, fill_tokens, pass_index):
            del pass_index
            ids = first if fill_tokens is None else torch.cat((anchors, fill_tokens), dim=1)
            if bool((ids == int(d.head.mask_token_id)).any()):
                raise ValueError('Jacobi input must be mask-free')
            return DraftForwardPass(logits=free(ids), input_tokens=ids[:, 1:])
        return self._propose_branches(forward, None, None, None, first, anchors,
            rejected, [k], 1, passes, bucket_factory=lambda:
                self._host_branch_bucket(seq, prefix, tails, k, self.branches))





class BucketPass:
    """One captured embedding/head/projection pass; changing data keeps pointers."""
    def __init__(self,drafter,*,N,H,prefix_bucket,backend,F=1,row_capacity=None,
                 host_fast=False,flat=False,graphs=True,refresh=False,anchor0=False,wide=False,
                 pool=None,live_of=None,live_shelf=None):
        self.drafter=drafter
        self.graphs=bool(graphs)
        self.pool=pool
        # JETSPEC_ROW_LAYOUT=anchor0: the pass reads all 16 slot outputs
        self.anchor0=bool(anchor0)
        if self.anchor0 and (host_fast or not flat):
            raise ValueError('anchor0 rows are flat, non-host-fast rows only')
        # JETSPEC_DRAFTS=15: a 16-row tail (c, d1..d15) and all 16 slots published
        self.wide=bool(wide)
        self.B=getattr(drafter, 'block_size', 16) if self.wide else 16
        if self.wide and not self.anchor0:
            raise ValueError('15 drafts require the anchor0 layout')
        head_type=FastBucketRowDraftHead if host_fast else BucketRowDraftHead
        if live_of is not None:
            # JETSPEC_BATCH_ROWS_LIVE: N rows of the flat bucket head ``live_of`` (rows_live.py)
            from jetspec.engines.jetspec_plus_plus.draft.rows_live import CompactRowHead
            if host_fast or refresh or not flat or os.environ.get('JETSPEC_FUSE','off')!='off':
                raise ValueError('a live-rows pass is a flat pass without refresh or JETSPEC_FUSE')
            self.head=CompactRowHead(live_of,rows=N,shelf=live_shelf)
        else:
            self.head=head_type(drafter.head,N=N,H=H,prefix_bucket=prefix_bucket,
                backend=backend,F=F,row_capacity=row_capacity,flat=flat,graphs=self.graphs,
                **({'refresh': True} if refresh else {}),
                **({'tail_rows': self.B, 'block_size': self.B} if self.wide else {}))
        if os.environ.get('JETSPEC_FUSE','off')!='off':
            from jetspec.engines.jetspec_plus_plus.kernels.fuse import configure_rows
            configure_rows(self.head,os.environ['JETSPEC_FUSE'])
        self.ids=torch.empty(N*self.head.R,self.B,dtype=torch.long,device=drafter.device)
        self.graph=None
        if os.environ.get('JETSPEC_FUSE','off') in ('compile','all'):
            from jetspec.engines.jetspec_plus_plus.kernels.fuse import compile_pass; from jetspec.engines.jetspec_plus_plus.kernels.fuse import uses_compile
            if uses_compile(os.environ['JETSPEC_FUSE']):
                self.step=compile_pass(self.step)

    def step(self, *, hidden_only=False):
        d=self.drafter
        if self.wide:
            self.hidden=self.head(d.embed_tokens(self.ids))
            if hidden_only:
                return self.hidden
            return project_row_logits(d.lm_head,self.hidden)
        if self.anchor0:
            # Slots 0..14 (anchor, d1..d14) are the kth frame's 15 outputs; slot 15
            # (draft 15) feeds the next pass and the would-be 16th wire token
            hidden=self.head(d.embed_tokens(self.ids))
            self.hidden=hidden[:,:15]
            logits=project_row_logits(d.lm_head,hidden)
            self.extra_logits=logits[:,15]
            self.extra_tokens=self.extra_logits.argmax(dim=-1,keepdim=True)
            return logits[:,:15]
        self.hidden=self.head(d.embed_tokens(self.ids))[:,1:]
        return project_row_logits(d.lm_head,self.hidden)

    @torch.inference_mode()
    def __call__(self,ids):
        self.ids.copy_(ids)
        if self.ids.device.type!='cuda' or not self.graphs:
            return self.step()
        if self.graph is None:
            import time
            started=time.perf_counter()
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3): self.step()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, pool=self.pool):
                self.logits=self.step()
            print(f'W240_CAPTURE N={self.head.N} H={self.head.H} F={self.head.F} rows={self.head.layout.r} prefix={self.head.P} ms={(time.perf_counter()-started)*1000:.3f}',flush=True)
        self.graph.replay()
        return self.logits


def passes_chain_segments(rp, tokens, keys, active, staged_hidden, bucket, J,
                          refresh_at, T_ref, *, outputs=None, refresh_swap=None,
                          standalone_refresh_at=(), exact_prefix=False):
    """Fixed-shape anchor0 Jacobi bodies for separate graph capture per pass.

    ``tokens`` is the per-sequence device TokenBuffers contract. The bodies read
    device RoundParams and row keys only; all host choices are captured in their
    shapes and the static ``refresh_at`` schedule.

    ``refresh_swap`` (JETSPEC_REFRESH_SWAP, default None: no change): a
    ``chain.refresh_swap.RefreshSwap``; J0 clears its pins, the refresh swaps and pins
    one draft per row and injects entries 0..khat_used, and every later pass keeps
    the pinned tokens.
    """
    from jetspec.engines.jetspec_plus_plus.wire.chain.contract import Offset; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import SegmentId; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import rng_words
    from jetspec.engines.jetspec_plus_plus.kernels.chain.glue2 import j0_prep; from jetspec.engines.jetspec_plus_plus.kernels.chain.glue2 import mask_token_guard
    from jetspec.engines.jetspec_plus_plus.kernels.chain.refine_k3 import allocate_partials; from jetspec.engines.jetspec_plus_plus.kernels.chain.refine_k3 import chain_pass; from jetspec.engines.jetspec_plus_plus.kernels.chain.refine_k3 import reference_chain_pass

    rows, lanes = bucket.ids.shape
    if not (bucket.anchor0 and bucket.wide and type(J) is int and J >= 0):
        raise ValueError('chain passes require wide anchor0 rows and nonnegative J')
    if tokens.pending_draft_ids.shape != (rng_words(lanes),):
        raise ValueError('chain pending draft width does not match the bucket')
    if any(index < 1 or (index > J and index not in standalone_refresh_at)
           for index in refresh_at):
        raise ValueError('refresh indices must be pass indices after J0')
    if refresh_at and T_ref < rows * rng_words(lanes):
        raise ValueError('fixed refresh token capacity is too small')
    if refresh_swap is not None:
        refresh_swap.configure_pins(refresh_at, rows=rows, lanes=lanes)
    owns_outputs = outputs is None
    if outputs is None:
        outputs = {
            'final_tokens': torch.empty_like(bucket.ids),
            'final_p_top1': torch.empty(rows, lanes, device=keys.device, dtype=torch.float32),
            'passes_run': torch.empty(rows, device=keys.device, dtype=torch.long),
            'finished': torch.empty(rows, device=keys.device, dtype=torch.bool),
            'meta': torch.empty(rows, 4, device=keys.device, dtype=torch.long),
            'err': torch.empty(1, device=keys.device, dtype=torch.long),
        }
    if 'final_p_top1' not in outputs:
        outputs['final_p_top1'] = torch.empty(rows, lanes, device=keys.device,
                                             dtype=torch.float32)
    if (outputs['final_p_top1'].shape != (rows, lanes)
            or outputs['final_p_top1'].dtype != torch.float32
            or outputs['final_p_top1'].device != keys.device):
        raise ValueError('chain final_p_top1 must be float32[R, lanes] on the row device')
    bucket.chain_outputs = outputs
    from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled as eval_enabled; from jetspec.engines.jetspec_plus_plus.draft.eval_profile import states_only
    eval_mode = eval_enabled()
    if eval_mode:
        outputs['eval_audit'] = torch.zeros(4, device=keys.device, dtype=torch.int64)
        if states_only():
            outputs['eval_refresh_effects'] = (refresh_swap.eval_effects
                if refresh_swap is not None else
                torch.zeros(2, device=keys.device, dtype=torch.int64))
    from jetspec.engines.jetspec_plus_plus.draft.eval_profile import capture_candidates; from jetspec.engines.jetspec_plus_plus.draft.eval_profile import draft_candidates; from jetspec.engines.jetspec_plus_plus.draft.eval_profile import draft_token_values
    capture_top = capture_candidates()
    if capture_top:
        outputs['final_draft_top16_ids'] = torch.empty(rows, lanes, 16, device=keys.device, dtype=torch.long)
        outputs['final_draft_top16_probabilities'] = torch.empty(rows, lanes, 16, device=keys.device)
        outputs['final_draft_token_probabilities'] = torch.empty(rows, lanes, device=keys.device)
        outputs['final_draft_pinned'] = torch.empty(rows, lanes, device=keys.device, dtype=torch.bool)
        captured_pass = [None, None]
    p_top1_prev = torch.empty(rows, rng_words(lanes), device=keys.device, dtype=torch.float32)
    k3 = os.environ.get('JETSPEC_REFINE_K3', '0') == '1'
    fused_head = os.environ.get('JETSPEC_LMHEAD_FUSED', '0') == '1'
    head_kn = fused_head and os.environ.get('JETSPEC_J0_LMHEAD_KN', '0') == '1'
    if fused_head and (not k3 or keys.device.type != 'cuda'):
        raise ValueError('JETSPEC_LMHEAD_FUSED requires CUDA JETSPEC_REFINE_K3=1')
    if head_kn:
        from jetspec.engines.jetspec_plus_plus.draft.projections import prepare_j0_lmhead_kn
        weight_kn = prepare_j0_lmhead_kn(bucket.drafter.lm_head)
        partials = allocate_partials(rows, lanes, bucket.drafter.lm_head.weight.shape[0], keys.device)
        bucket.chain_lmhead_partials = partials
    elif fused_head:
        from jetspec.engines.jetspec_plus_plus.kernels.chain.lmhead_fused import allocate_partials as allocate_head_partials
        partials = allocate_head_partials(rows, lanes,
                                          bucket.drafter.lm_head.weight.shape[0], keys.device)
        bucket.chain_lmhead_partials = partials
    else:
        partials = (allocate_partials(rows, lanes, bucket.drafter.lm_head.weight.shape[0],
                                      keys.device) if k3 and keys.device.type == 'cuda' else None)

    if refresh_swap is not None and refresh_swap.fill_mode != 'keep':
        refresh_swap.fill_mask_token = int(bucket.drafter.head.mask_token_id)
    refresh_stream = None
    if (refresh_at and os.environ.get('JETSPEC_REFRESH_STREAMS', '0') == '1'
            and staged_hidden.dtype == torch.bfloat16):
        if (not staged_hidden.is_cuda or refresh_swap is None
                or os.environ.get('JETSPEC_REFRESH_INJECT_LEAN2', '0') != '1'):
            raise ValueError('JETSPEC_REFRESH_STREAMS requires CUDA BF16 swap and lean2 injection')
        from jetspec.engines.jetspec_plus_plus.draft.chain.draft_streams import DraftStreams
        refresh_stream = DraftStreams(staged_hidden.device)
    refresh_projected = [None]

    def refresh(pass_index):
        if T_ref < rows * rng_words(lanes):
            raise ValueError('B1 refresh needs fixed token capacity')
        d = bucket.drafter
        head = d.adapter.head
        window = getattr(head, '_c1_window', None)
        if window != lanes:
            raise ValueError(f'chain refresh window must equal live lanes: {window} != {lanes}')
        again = pass_index != refresh_at[0] if refresh_at else False
        chained = torch.cumprod(p_top1_prev.double(), dim=1)
        prep_lean = (os.environ.get('JETSPEC_REFRESH_PREP_LEAN', '0') == '1'
                     and bucket.ids.is_cuda)
        if eval_mode:
            khat = torch.where(active, rng_words(lanes), 0).to(torch.int64)
            counts = torch.where(active, lanes - 1, 0).to(torch.int64)
            prep_lean = False
            outputs['eval_audit'][1].add_(1)
            outputs['eval_audit'][3].add_((active & (khat != rng_words(lanes))).sum())
        elif prep_lean:
            from jetspec.engines.jetspec_plus_plus.kernels.refresh_prep import bookkeeping
            khat, counts, tails, fixed_start = bookkeeping(
                chained, active, keys, rp[Offset.L], window,
                refresh_khat_threshold(again), refresh_khat_cap(again) if again else None)
        else:
            below = chained < refresh_khat_threshold(again)
            first = below.to(torch.int8).argmax(dim=1) + 1
            khat = torch.where(below.any(dim=1), first,
                               torch.full_like(first, rng_words(lanes)))
            cap = refresh_khat_cap(again) if again else None
            if cap is not None:
                khat = khat.clamp(max=cap)
            khat = torch.where(active, khat, torch.zeros_like(khat))
            counts = torch.where(active, khat.clamp(max=lanes - 2) + 1, torch.zeros_like(khat))
        embedded = d.embed_tokens(bucket.ids)
        if exact_prefix:
            tails = torch.zeros_like(keys[:, 0])
        elif not prep_lean:
            tails = keys[:, 0] + 1
        key_width = None
        if os.environ.get('JETSPEC_CHAIN_REFRESH_KEY_BUCKET', '0') == '1':
            capacity = head._c1_kv[0][0].shape[2]
            upper = bucket.head.P + window + T_ref
            key_width = min(capacity, ((upper + 255) // 256) * 256)
        if prep_lean:
            args = (embedded, rp[Offset.L], tails, counts, T_ref, fixed_start)
        else:
            args = (embedded, rp[Offset.L], tails, counts, T_ref,
                    rp[Offset.L] + window)
        block = (head._refresh_cached_fixed(*args) if key_width is None else
                 head._refresh_cached_fixed(*args, key_width=key_width))
        if refresh_stream is not None:
            from jetspec.engines.jetspec_plus_plus.draft.refresh_inject import project

            # Keep the pre-swap allocation alive until the auxiliary work joins.
            def project_before_swap(cutoff=khat):
                refresh_projected[0] = project(bucket.head, block, cutoff)

            refresh_stream.start(project_before_swap)
        if refresh_swap is not None:
            # The decoder's first disagreement lowers k-hat and swaps that draft
            before = bucket.ids.clone() if getattr(refresh_swap, 'states_only', False) else None
            khat = refresh_swap.apply(length=rp[Offset.L], ids=bucket.ids, khat=khat,
                                      slot_counts=counts,
                                      tail_counts=tails,
                                      active=active, block=block,
                                      finished=outputs['finished'])
            if before is not None:
                refresh_swap.eval_effects[0].add_(bucket.ids.ne(before).sum())
            if os.environ.get('JETSPEC_REFRESH_INJECT_INCLUSIVE', '0') == '1':
                khat = khat - 1  # swap.apply returns a count, inclusive inject takes a cutoff
        if refresh_stream is None:
            bucket.head.inject_refresh(block, khat, reset=again,
                                       fixed_layout=True)
        else:
            from jetspec.engines.jetspec_plus_plus.draft.refresh_inject import commit
            refresh_stream.finish()
            commit(bucket.head, refresh_projected[0], khat, again)

    def pass_body(index, *, do_refresh=False):
        if eval_mode:
            if index == 0:
                outputs['eval_audit'].zero_()
                outputs['eval_audit'][2].copy_(active.sum())
            else:
                outputs['eval_audit'][0].add_(1)
        if index == 0:
            if os.environ.get('JETSPEC_CHAIN_MR_GLUE_LEAN', '0') == '1' and bucket.ids.is_cuda:
                j0_prep(tokens.committed_ids, rp, keys, active, bucket.ids,
                        outputs['finished'], outputs['passes_run'], outputs['err'],
                        owns_err=owns_outputs)
            else:
                draw_idx = rp[Offset.RNG_IDX:Offset.RNG_IDX + rng_words(lanes)]
                drawn = (tokens.committed_ids[draw_idx]
                         if os.environ.get('JETSPEC_CHAIN_GATHER_VEC', '0') == '1'
                         else tokens.committed_ids.index_select(0, draw_idx))
                first = torch.cat((keys[:, 1:2].clamp_min(0),
                                   drawn.expand(rows, -1)), dim=1)
                bucket.ids.copy_(first)
                outputs['finished'].copy_(~active)
                outputs['passes_run'].zero_()
                if owns_outputs:
                    outputs['err'].zero_()
            if refresh_swap is not None:
                refresh_swap.reset()
                if refresh_swap.fill_mode == 'context':
                    refresh_swap.fill_context.copy_(bucket.ids)
        if do_refresh:
            refresh(index)
        mask_token = int(bucket.drafter.head.mask_token_id)
        if refresh_swap is not None and refresh_swap.fill_mode != 'keep':
            outputs['err'].bitwise_or_(
                (bucket.ids.eq(mask_token) & active[:, None] & ~refresh_swap.fill_filled)
                .any().to(torch.long).reshape(1) * (1 << 7))
            refresh_swap.fill_filled.zero_()
        elif ((os.environ.get('JETSPEC_CHAIN_SEGA_GLUE', '0') == '2'
             and index in (0, 1))
                or (os.environ.get('JETSPEC_CHAIN_MR_GLUE_LEAN', '0') == '1'
                    and index == 2)):
            mask_token_guard(bucket.ids, active, outputs['err'], mask_token)
        else:
            outputs['err'].bitwise_or_(
                (bucket.ids.eq(mask_token) & active[:, None])
                .any().to(torch.long).reshape(1) * (1 << 7))
        logits = bucket.step(hidden_only=True) if fused_head else bucket.step()
        if capture_top:
            distribution = (project_row_logits(bucket.drafter.lm_head, logits)
                            if fused_head else logits)
            captured_pass[:] = [distribution, outputs['finished'].clone()]
            previous = (outputs['final_draft_top16_ids'], outputs['final_draft_top16_probabilities'])
            ranked = draft_candidates(distribution, previous if index else None,
                                      outputs['finished'])
            ranked[0].masked_fill_(~active[:, None, None], -1)
            ranked[1].masked_fill_(~active[:, None, None], 0)
            for target, value in zip(previous, ranked):
                target.copy_(value)
        if (refresh_swap is not None and refresh_swap.rule is not None
                and (index == 0 or index + 1 in refresh_at)):
            refresh_swap.retain_draft(logits, bucket.drafter.lm_head.weight if fused_head else None)
        # After the refresh every pass's output keeps the round's pinned swaps
        pins = refresh_swap.pins() if refresh_swap is not None and index >= 1 else None
        if head_kn:
            logits = torch.mm(logits.reshape(-1, logits.shape[-1]), weight_kn).reshape(rows, lanes, -1)
            chain_pass(logits, bucket.ids, keys, active, outputs, p_top1_prev,
                       tokens, rp, partials, index,
                       save_previous=index == 0 or index + 1 in refresh_at, pins=pins)
            return
        if fused_head:
            from jetspec.engines.jetspec_plus_plus.kernels.chain.lmhead_fused import chain_pass as fused_chain_pass
            save_previous = index == 0 or index + 1 in refresh_at
            fused_chain_pass(logits, bucket.drafter.lm_head.weight, bucket.ids,
                             keys, active, outputs, p_top1_prev, tokens, rp,
                             partials, index, save_previous=save_previous, pins=pins)
            return
        if k3:
            save_previous = index == 0 or index + 1 in refresh_at
            if logits.device.type == 'cuda':
                chain_pass(logits, bucket.ids, keys, active, outputs, p_top1_prev,
                           tokens, rp, partials, index, save_previous=save_previous,
                           pins=pins)
            else:
                reference_chain_pass(logits, bucket.ids, keys, active, outputs,
                                     p_top1_prev, tokens, rp, index,
                                     save_previous=save_previous, pins=pins)
            return
        pass_top1 = top1_probabilities(logits, chunk_rows=16)
        from jetspec.engines.jetspec_plus_plus.draft.method.passes import jacobi_tokens; from jetspec.engines.jetspec_plus_plus.draft.method.passes import jacobi_converged
        current = jacobi_tokens(logits, keys[:, 1].clamp_min(0), pins=pins)
        fixed = jacobi_converged(current, bucket.ids)
        if index == 0:
            outputs['final_tokens'].copy_(current)
            outputs['final_p_top1'].copy_(pass_top1)
        else:
            outputs['final_tokens'].copy_(torch.where(
                outputs['finished'][:, None], outputs['final_tokens'], current))
            outputs['final_p_top1'].copy_(torch.where(
                outputs['finished'][:, None], outputs['final_p_top1'], pass_top1))
        outputs['passes_run'].masked_fill_(~outputs['finished'], index + 1)
        outputs['finished'].logical_or_(fixed)
        bucket.ids.copy_(current)
        if index == 0 or index + 1 in refresh_at:
            p_top1_prev.copy_(pass_top1[:, 1:])
        outputs['final_tokens'].masked_fill_(~active[:, None], -1)
        outputs['final_p_top1'].masked_fill_(~active[:, None], 0)
        source = torch.cat((tokens.committed_ids.index_select(
            0, rp[Offset.L].reshape(1)), tokens.pending_draft_ids,
            tokens.pending_draft_ids[-1:]))
        preceding = source.index_select(0, keys[:, 0].clamp(0, lanes))
        outputs['meta'].copy_(torch.stack((preceding, outputs['final_tokens'][:, 0],
                                            outputs['passes_run'],
                                            outputs['finished'].long()), dim=1))

    run_pass = pass_body
    if capture_top:
        run_pass = capture_pass
    bodies = {SegmentId.A.value: lambda: run_pass(0)}
    if J >= 1:
        bodies[SegmentId.B1R.value] = lambda: run_pass(1, do_refresh=True)
        bodies[SegmentId.B1N.value] = lambda: run_pass(1)
    if J >= 2:
        bodies[SegmentId.B2.value] = lambda: run_pass(2, do_refresh=2 in refresh_at)
    if J >= 3:
        bodies[SegmentId.C.value] = lambda: run_pass(3, do_refresh=3 in refresh_at)
    for index in range(4, J + 1):
        bodies[f'J{index}'] = lambda index=index: run_pass(index, do_refresh=index in refresh_at)
    for index in standalone_refresh_at:
        def refresh_only(index=index):
            refresh(index)
            outputs['final_tokens'].copy_(bucket.ids)
            outputs['final_tokens'].masked_fill_(~active[:, None], -1)
            outputs['meta'][:, 1].copy_(outputs['final_tokens'][:, 0])
            outputs['meta'][:, 3].copy_(outputs['finished'].long())
        bodies[f'R{index}'] = refresh_only
    if J and refresh_at:
        from jetspec.engines.jetspec_plus_plus.draft.chain.graph import tail_segments
        tails = tail_segments()
        if tails:
            # Tail passes preserve the recipe's threshold label and row convergence.
            saved_passes = torch.empty_like(outputs['passes_run'])
            outputs['tail_saved_passes'] = saved_passes


            bodies.update({name: tail_body for name in tails})
    return bodies


class PackedBlocks:
    """A sequence-owned wire and logits; materialize only the selected Block."""
    # E04_LAB_EXTRA_PASSES (default None): a block's threshold label reads min(passes run, this)
    lab_label_passes = None
    # E04_LAB_* (default None): per published row, the decoder-written slots [rows, width] bool and
    # (E04_LAB_KSCORE_WRITTEN=draft_p / both) the draft model's p of each written token; a block
    # carries its row's as e04_lab_written / e04_lab_written_p for the next round (tau_lab.py)
    lab_written = None
    lab_written_p = None

    def __init__(self, wire, logits, metadata, hypotheses, *, candidate_wire=None,
                 predictor=None, predictor_hidden=None, predictor_logits=None,
                 predictor_inputs=None, predictor_outputs=None, predictor_p_top1=None,
                 posterior_probabilities=None, row_keys=None):
        self.block_wire = wire
        self.wire = wire if candidate_wire is None else candidate_wire
        self.logits, self.metadata = logits, metadata
        self._host_metadata = None
        self.hypotheses = hypotheses
        self.row_keys = row_keys
        self.predictor = predictor
        self.predictor_hidden = predictor_hidden
        self.predictor_logits = predictor_logits
        self.predictor_p_top1 = predictor_p_top1
        self.predictor_inputs = predictor_inputs
        self.predictor_outputs = predictor_outputs
        self.posterior_probabilities = posterior_probabilities
        self._blocks = {}
        self._fallback = None
        self._fallback_blocks = None

    @property
    def fallback(self):
        if self._fallback is None and self._fallback_blocks is not None:
            self._fallback = self._fallback_blocks[0]
            self._fallback_blocks = None
        if self._fallback is None:
            raise AttributeError('fallback')
        return self._fallback

    @fallback.setter
    def fallback(self, value):
        self._fallback = value
        self._fallback_blocks = None


    def __len__(self):
        return len(self.hypotheses)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if not -len(self) <= index < len(self):
            raise IndexError(index)
        index %= len(self)
        if index in self._blocks:
            return self._blocks[index]
        metadata = self.metadata if self._host_metadata is None else self._host_metadata
        anchor, guess, passes, converged = metadata[index].tolist()
        label_passes = passes if self.lab_label_passes is None else min(passes, self.lab_label_passes)
        prediction = None if self.predictor is None else self.predictor.predict_p_top1(
            self.predictor_hidden[index][:15], self.predictor_p_top1[index][:15],
            self.predictor_inputs[index][:15], self.predictor_outputs[index][:15],
            passes_run=label_passes)
        if prediction is not None and self.predictor_hidden.shape[1] == 16:
            # JETSPEC_DRAFTS=15: the head reads its 15 trained slots [anchor, d1..d14];
            # its all-accepted mass (wire k = 14) splits by draft 15's top-1 probability
            import dataclasses
            mass = prediction.probabilities
            p15 = self.predictor_p_top1[index][15].to(mass.device, mass.dtype)
            prediction = dataclasses.replace(prediction, probabilities=torch.cat(
                (mass[:15], (mass[15] * (1 - p15)).reshape(1), (mass[15] * p15).reshape(1))))
        probabilities = (prediction.probabilities if prediction is not None else
                         None if self.posterior_probabilities is None else
                         self.posterior_probabilities[index])
        pass_label = (prediction.pass_label if prediction is not None else
                      served_kpred_pass(label_passes) if kpred_forward_off() else None)
        block = Block(hypothesis=self.hypotheses[index], mode='kth', draft_ids=self.block_wire[index, 1:],
            logits=None if self.logits is None else self.logits[index],
            passes_run=passes, converged=bool(converged),
            anchor=anchor, correction_guess=guess,
            k_probabilities=probabilities,
            kpred_pass=pass_label,
            kpred_ms=0.0 if prediction is None else prediction.elapsed_ms)
        if os.environ.get('E04_RECORD_PTOP1'):
            e04_attach_p_top1(block, self.predictor_p_top1, index)
        if self.lab_written is not None:
            object.__setattr__(block, 'e04_lab_written', self.lab_written[index])
            if self.lab_written_p is not None:
                object.__setattr__(block, 'e04_lab_written_p', self.lab_written_p[index])
        self._blocks[index] = block
        return block


@torch.inference_mode()
def packed_refinement(forward, *, anchors, hypotheses, sequences, passes, convergence_limit,
                      output_sequences=None, output_hypotheses=None, abort_if=None, fixed_wire_capacity=None,
                      predictor=None, output_indices=None, row_keys=None, pinned_tokens=None,
                      posterior=False, correction_head=None, late=None, truncate_if=None,
                      pinned_second=None, publish=None, active_rows=None, after_passes=None,
                      fast_top1=False, fixed_passes=False, device_metadata=False,
                      refresh_swap=None, tail_passes=0, lab_pins=None, lab_touched=None, final_edit=None,
                      label_passes=None, lab_written=None, lab_written_p=False, lab_pass=None, lab_top=None,
                      operation_schedule=None, refresh_only=None):
    """Same per-row Jacobi fixed-point decisions, with one metadata transfer.

    ``lab_pins`` (E04_LAB_SWAP, default None): a [rows, width] token tensor, -1 where nothing is
    pinned, read after every pass (the forward may fill it mid-round); a pinned slot's token
    replaces that pass's argmax, as ``pinned_tokens`` does for slot 0.  ``lab_touched`` (with
    ``lab_pins``): a bool per row the forward sets when it swapped that row's input; the row is
    reopened for that pass (its output published even if it had converged) and the flags cleared.
    ``final_edit`` (E04_LAB_SWAP=pre_verify, default None): called once on the final tokens after
    the passes, before anything is published; its return value replaces them.  ``label_passes``
    (E04_LAB_EXTRA_PASSES, default None): the published blocks read their threshold label at
    min(passes run, label_passes), the recipe's pass count (``PackedBlocks.lab_label_passes``).
    ``lab_written`` (E04_LAB_*, default None): the round's lab pins [rows, width] (-1: none), or a
    callable returning them (E04_LAB_PIN_AGREED: ``LabRound.written_pins``), read once after ``final_edit``: its pinned slots are the decoder-written slots, and each published
    row's mask goes to ``PackedBlocks.lab_written`` (each Block's ``e04_lab_written``) for the
    next round.  ``lab_written_p`` (with it; E04_LAB_KSCORE_WRITTEN=draft_p / both): every row's
    logits at the pass its tokens freeze at are kept, and ``PackedBlocks.lab_written_p`` holds the
    draft model's fp32 softmax probability of the written token at each written slot (NaN
    elsewhere).  ``lab_pass`` (E04_LAB_*, default None): called after every pass with the pass
    index and that pass's argmax [rows, width] before the lab pins replace it.  ``lab_top``
    (E04_LAB_RECORD_TOPK, default None): called after every pass with its logits and the rows whose
    tokens stay frozen (None at the first pass, which every row takes), the mask the final tokens
    read, so the callee can keep each row's values at the pass its tokens freeze at.

    ``late`` (JETSPEC_ROWS_STATIC): a callable read after the passes are queued that
    returns (output_hypotheses, output_indices, row_keys) in place of the arguments.

    ``publish`` (JETSPEC_TAIL_PUBLISH): called with the one sequence's published
    device rows (the fixed-capacity wire and its padded keys, the tensors the returned
    PackedBlocks carry) as soon as they are queued, before the metadata's trip home and
    the host bookkeeping, so the candidate leaves before that work.

    With a correction head the pass feedback and the fixed-point test stay on
    the bare argmax; the published tokens are the head's chained read of each
    row's final-pass logits, run once after the passes. The k predictor sees
    the published (wire) tokens unless JETSPEC_CORRECTION_HEAD_PREDICTOR=bare.
    """
    limit = convergence_limit if passes is None else passes
    if limit < 1:
        raise ValueError('passes must be positive')
    count = sequences * len(hypotheses)
    if active_rows is not None:
        if active_rows.shape != (count,) or active_rows.dtype != torch.bool:
            raise ValueError('active_rows must be one bool per fixed-cap row')
        finished = ~active_rows.clone()
    else:
        finished = torch.zeros(count, dtype=torch.bool, device=anchors.device)
    passes_run = torch.zeros(count, dtype=torch.long, device=anchors.device)
    record_top1 = os.environ.get('E04_RECORD_PTOP1') == '1'
    from jetspec.engines.jetspec_plus_plus.draft.eval_profile import capture_candidates; from jetspec.engines.jetspec_plus_plus.draft.eval_profile import draft_candidates; from jetspec.engines.jetspec_plus_plus.draft.eval_profile import draft_token_values
    capture_top = capture_candidates()
    final_draft_top16 = None
    final_draft_values = None
    compact_logits = predictor is not None or posterior or record_top1
    loop_device = (os.environ.get('JETSPEC_PASS_LOOP_DEVICE', '0') == '1'
                   and fixed_passes and correction_head is None and not fast_top1
                   and lab_pins is None and lab_touched is None and lab_pass is None
                   and lab_top is None and not lab_written_p and not capture_top)
    loop_state = []
    if loop_device:
        from jetspec.engines.jetspec_plus_plus.kernels.rows_loop import extension
        pass_loop = extension(cuda=anchors.is_cuda)
    current = final_logits = final_tokens = final_hidden = final_inputs = final_p_top1 = None
    lab_logits = None  # lab_written_p: each row's logits at its final pass
    if pinned_tokens is not None:
        pinned_tokens = torch.as_tensor(pinned_tokens, device=anchors.device, dtype=torch.long)
    # On CUDA the all-converged flag is read one pass late through a pinned
    # host copy, after the next pass is already queued: the host never waits
    # for the device between passes, every published value is frozen by
    # ``finished``, and only an all-converged round spends one extra pass.
    # Elsewhere the flag is read directly.
    lazy = anchors.device.type == 'cuda'
    pending = None
    head_logits = None
    # JETSPEC_TAIL_PUBLISH: each pass's top-1 probabilities (a softmax over the vocabulary,
    # read only by the host bookkeeping) are folded in before the next pass overwrites its
    # logits, or after the publish for the last pass, so the published rows do not wait on it
    early = (publish is not None and output_sequences in (None, 1) and sequences == 1
             and fixed_wire_capacity is not None and row_keys is not None and late is None
             and correction_head is None)
    defer_top1 = early and compact_logits and not loop_device
    pending_top1 = None

    def fold_top1():
        nonlocal final_p_top1, pending_top1
        if pending_top1 is None:
            return
        logits_, snapshot = pending_top1
        pending_top1 = None
        value = top1_probabilities(logits_, chunk_rows=16, fast=fast_top1)
        final_p_top1 = value.clone() if snapshot is None else torch.where(
            snapshot[:, None], final_p_top1, value)

    stages = None if operation_schedule is None else operation_schedule.stages
    stage_count = limit if stages is None else len(stages)
    for stage_index in range(stage_count + tail_passes):
        stage = (None if operation_schedule is None or stage_index >= stage_count else
                 stages[stage_index])
        index = ((stage_index if stages is None else limit + stage_index - stage_count)
                 if stage is None else
                 int(stage[-1][1:]) if stage[-1].startswith('J') else limit)
        fold_top1()
        if abort_if is not None and abort_if():
            raise DraftSuperseded('newer committed verifier state is ready')
        # JETSPEC_DRAFT_DEADLINE=1: publish the last completed pass (never before pass 0).
        if truncate_if is not None and truncate_if(stage_index):
            break
        if stage is not None and stage[-1].startswith('R'):
            refresh_only(final_tokens, int(stage[-1][1:]), finished)
            current = final_tokens
            continue
        result = (forward(fill_tokens=current, pass_index=index, finished=finished)
                  if refresh_swap is not None else
                  forward(fill_tokens=current, pass_index=index))
        if lab_touched is not None:
            finished &= ~lab_touched
            lab_touched.zero_()
        logits = result.logits
        if capture_top:
            final_draft_top16 = draft_candidates(logits, final_draft_top16, finished)
        if loop_device:
            pin_mask = pin_values = None
            if refresh_swap is not None and index >= 1:
                pins = refresh_swap.pins()
                pin_mask = (pins[0] if pins[0].ndim == 2 else
                            torch.arange(logits.shape[1], device=logits.device)[None] == pins[0][:, None])
                pin_values = pins[1] if pins[0].ndim == 2 else pins[1][:, None]
            loop_state = pass_loop.pass_step(
                logits, result.input_tokens, result.hidden, finished, passes_run, loop_state,
                pinned_tokens, pinned_second, pin_mask, pin_values,
                index, limit, compact_logits, predictor is not None)
            current, final_tokens, saved_logits, saved_prob, saved_hidden, saved_inputs = loop_state
            final_logits = None if compact_logits else saved_logits
            final_p_top1 = saved_prob if compact_logits else None
            if predictor is not None:
                final_hidden, final_inputs = saved_hidden, saved_inputs
            continue
        from jetspec.engines.jetspec_plus_plus.draft.method.passes import jacobi_tokens; from jetspec.engines.jetspec_plus_plus.draft.method.passes import jacobi_converged; from jetspec.engines.jetspec_plus_plus.draft.method.passes import apply_pins
        current = jacobi_tokens(logits, pinned_tokens, pinned_second)
        if lab_pass is not None:
            lab_pass(index, current)
        if lab_pins is not None:
            current = apply_pins(current, (lab_pins >= 0, lab_pins))
        if refresh_swap is not None and index >= 1:
            pins = refresh_swap.pins()
            current = apply_pins(current, pins)
        if capture_top:
            pinned = torch.zeros_like(current, dtype=torch.bool)
            if pinned_tokens is not None:
                pinned[:, 0] = True
            if pinned_second is not None:
                pinned[:, 1] = True
            if lab_pins is not None:
                pinned.logical_or_(lab_pins >= 0)
            if refresh_swap is not None and index >= 1:
                mask = (pins[0] if pins[0].ndim == 2 else
                        torch.arange(current.shape[1], device=current.device)[None] == pins[0][:, None])
                pinned.logical_or_(mask)
            final_draft_values = draft_token_values(logits, current, pinned,
                                                    final_draft_values, finished)
        fixed = jacobi_converged(current, result.input_tokens)
        if correction_head is not None:
            head_logits = (logits.clone() if head_logits is None else
                           torch.where(finished[:, None, None], head_logits, logits))
        # Sixteen rows per chunk: the same float softmax in one launch per op.
        if defer_top1:
            pending_top1 = (logits, None if final_tokens is None else finished.clone())
            p_top1 = None
        else:
            p_top1 = top1_probabilities(logits, chunk_rows=16, fast=fast_top1) if compact_logits else None
        if final_tokens is None:
            final_logits = None if compact_logits else logits.clone()
            final_tokens = current.clone()
            final_p_top1 = None if p_top1 is None else p_top1.clone()
            if predictor is not None:
                if result.hidden is None:
                    raise RuntimeError('predictor serving requires final-pass hidden states')
                final_hidden, final_inputs = result.hidden.clone(), result.input_tokens.clone()
            if lab_written_p:
                lab_logits = logits.clone()
            if lab_top is not None:
                lab_top(logits, None)
        else:
            if not compact_logits:
                final_logits = torch.where(finished[:, None, None], final_logits, logits)
            elif not defer_top1:
                final_p_top1 = torch.where(finished[:, None], final_p_top1, p_top1)
            final_tokens = torch.where(finished[:, None], final_tokens, current)
            if predictor is not None:
                if result.hidden is None:
                    raise RuntimeError('predictor serving requires final-pass hidden states')
                final_hidden = torch.where(finished[:, None, None], final_hidden, result.hidden)
                final_inputs = torch.where(finished[:, None], final_inputs, result.input_tokens)
            if lab_written_p:
                lab_logits = torch.where(finished[:, None, None], lab_logits, logits)
            if lab_top is not None:
                lab_top(logits, finished)
        if index < limit:
            passes_run.masked_fill_(~finished, index+1)
        finished |= fixed
        if pending is not None and not fixed_passes:
            # The previous pass had every row converged: this pass changed no
            # published value (``finished`` froze them all), so stop here.
            event, flag = pending
            event.synchronize()
            if bool(flag[0]):
                break
        if stage_index + 1 >= stage_count + tail_passes:
            break
        if fixed_passes:
            continue
        if not lazy:
            if bool(finished.all()):
                break
            continue
        flag = torch.zeros(1, dtype=torch.bool, pin_memory=True)
        flag.copy_(finished.all().view(1), non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        pending = (event, flag)
    host_mark('pr_loop')
    if final_edit is not None:
        final_tokens = final_edit(final_tokens)
    lab_masks = None
    if callable(lab_written):
        lab_written = lab_written()  # E04_LAB_PIN_AGREED: the lab pins less the agreed run's (written_pins)
    if lab_written is not None:
        # E04_LAB_*: the decoder-written slots (the lab pins once final_edit has run) and, under
        # lab_written_p, the draft model's probability of each written token at its row's final pass
        written = lab_written >= 0
        written_p = None
        if lab_written_p:
            written_p = torch.full(written.shape, float('nan'), dtype=torch.float64, device=written.device)
            at = written.nonzero(as_tuple=True)
            if at[0].numel():
                written_p[at] = lab_logits[at].float().softmax(-1).gather(
                    1, lab_written[at][:, None])[:, 0].double()
        lab_masks = (written, written_p)
    if active_rows is not None:
        # The fixed-cap replay computes dummy rows, but they cannot become a wire row.
        final_tokens.masked_fill_(~active_rows[:, None], -1)
    if after_passes is not None:
        after_passes()
    published = None
    if early:
        # the published rows exactly as the prepared loop below builds them for n = 0
        early_indices = torch.tensor(
            [hypotheses.index(k) for k in (hypotheses if output_hypotheses is None
                                           else output_hypotheses)] if output_indices is None
            else list(map(int, output_indices)), dtype=torch.long,
            pin_memory=anchors.device.type == 'cuda')
        # a pinned, non-blocking upload: a pageable one would hold the host until the pass ends
        early_indices = early_indices.to(anchors.device, non_blocking=True)
        early_wire = final_tokens[:len(hypotheses)].index_select(0, early_indices).clone()
        early_candidate = early_wire.new_full((fixed_wire_capacity, early_wire.shape[-1]), -1)
        early_candidate[:early_wire.shape[0]] = early_wire
        early_keys = row_keys.new_full((fixed_wire_capacity, 2), -1)
        early_keys[:len(row_keys)] = row_keys
        publish(early_candidate, early_keys)
        host_mark('pr_publish')
        published = (early_wire, early_candidate, early_keys)
        if anchors.device.type == 'cuda' and not loop_device:
            # the metadata's trip home queued ahead of the deferred softmax (metadata_device's
            # own stack below), so the host bookkeeping does not wait on it
            early_metadata = torch.stack((anchors[:, 0], final_tokens[:, 0], passes_run,
                                          finished.long()), dim=1)
            early_host = torch.empty(early_metadata.shape, dtype=early_metadata.dtype,
                                     pin_memory=True)
            early_host.copy_(early_metadata, non_blocking=True)
            early_event = torch.cuda.Event()
            early_event.record()
            published += ((early_host, early_event),)
    fold_top1()
    if late is not None:
        output_hypotheses, output_indices, row_keys = late()
    predictor_tokens = final_tokens
    if correction_head is not None:
        final_tokens = correction_head.chain(head_logits, anchors[:, 0], pinned=pinned_tokens)
        if os.environ.get('JETSPEC_CORRECTION_HEAD_PREDICTOR', 'wire') != 'bare':
            predictor_tokens = final_tokens
    metadata_device = torch.stack((anchors[:, 0], final_tokens[:, 0], passes_run, finished.long()), dim=1)
    # JETSPEC_TAIL_PREISSUE=1: the published rows' device copies are queued before the
    # metadata's trip home, so their launches overlap the last pass instead of following the
    # sync (the same ops on the same values; only the host read moves)
    preissue = os.environ.get('JETSPEC_TAIL_PREISSUE', '0') == '1'
    if published is not None and len(published) == 4:
        published[3][1].synchronize()
        metadata = published[3][0]
    else:
        metadata = None if preissue or loop_device else (metadata_device if device_metadata else metadata_device.cpu())
    host_mark('pr_meta')
    width = len(hypotheses)
    output_sequences = sequences if output_sequences is None else output_sequences
    output_hypotheses = hypotheses if output_hypotheses is None else output_hypotheses
    index_list = ([hypotheses.index(k) for k in output_hypotheses]
                  if output_indices is None else list(map(int, output_indices)))
    if loop_device:
        host_indices = torch.tensor(index_list, dtype=torch.long, pin_memory=anchors.is_cuda)
        indices = host_indices.to(anchors.device, non_blocking=True)
    else:
        indices = torch.tensor(index_list, dtype=torch.long, device=anchors.device)
    # JETSPEC_TAIL_LEAN=1: the metadata row selection reads the host list, not a
    # second device trip home (the same indices)
    if not loop_device:
        host_indices = (torch.tensor(index_list, dtype=torch.long)
                        if os.environ.get('JETSPEC_TAIL_LEAN', '0') == '1' or preissue else None)
    if row_keys is not None and (sequences != 1 or len(row_keys) != len(output_hypotheses)):
        raise ValueError('flat row keys currently require N=1 and one key per published row')
    prepared = []
    for n in range(output_sequences):
        span = slice(n*width, (n+1)*width)
        wire = (final_tokens[span].index_select(0, indices).clone() if published is None
                else published[0])
        logits = (None if compact_logits else
                  final_logits[span, 1:].index_select(0, indices).clone())
        candidate_wire = None
        if published is not None:
            candidate_wire = published[1]
        elif fixed_wire_capacity is not None:
            candidate_wire = wire.new_full((fixed_wire_capacity, wire.shape[-1]), -1)
            if row_keys is not None:
                candidate_wire[:wire.shape[0]] = wire
            elif output_hypotheses:
                candidate_wire[output_hypotheses] = wire
        predictor_kwargs = {}
        if final_p_top1 is not None:
            predictor_kwargs['predictor_p_top1'] = final_p_top1[span].index_select(0, indices).clone()
        if predictor is not None:
            predictor_kwargs.update(
                predictor=predictor,
                predictor_hidden=final_hidden[span].index_select(0, indices).clone(),
                predictor_inputs=final_inputs[span].index_select(0, indices).clone(),
                predictor_outputs=predictor_tokens[span].index_select(0, indices).clone(),
            )
        if posterior:
            predictor_kwargs["posterior_probabilities"] = posterior_k_mass_from_top1(
                final_p_top1[span, 1:].index_select(0, indices)).clone()
        padded_keys = None
        if published is not None:
            padded_keys = published[2]
        elif row_keys is not None:
            padded_keys = row_keys.new_full((fixed_wire_capacity, 2), -1)
            padded_keys[:len(row_keys)] = row_keys
        fallback = None
        if fixed_wire_capacity is not None:
            fallback_predictor = {}
            if final_p_top1 is not None:
                fallback_predictor['predictor_p_top1'] = final_p_top1[span][:1].clone()
            if predictor is not None:
                fallback_predictor.update(
                    predictor=predictor,
                    predictor_hidden=final_hidden[span][:1].clone(),
                    predictor_inputs=final_inputs[span][:1].clone(),
                    predictor_outputs=predictor_tokens[span][:1].clone(),
                )
            if posterior:
                fallback_predictor["posterior_probabilities"] = posterior_k_mass_from_top1(
                    final_p_top1[span, 1:][:1]).clone()
            fallback = (final_tokens[span][:1].clone(),
                        None if compact_logits else final_logits[span, 1:][:1].clone(),
                        fallback_predictor)
        prepared.append((span, wire, logits, candidate_wire, predictor_kwargs, padded_keys,
                         fallback))
    if (publish is not None and published is None and output_sequences == 1
            and prepared[0][3] is not None and prepared[0][5] is not None):
        publish(prepared[0][3], prepared[0][5])
        host_mark('pr_publish')
    if metadata is None:
        if loop_device and not device_metadata and anchors.is_cuda:
            metadata = torch.empty(metadata_device.shape, dtype=metadata_device.dtype, pin_memory=True)
            metadata.copy_(metadata_device, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record()
            ready.synchronize()
        else:
            metadata = metadata_device if device_metadata else metadata_device.cpu()
    output = []
    for span, wire, logits, candidate_wire, predictor_kwargs, padded_keys, fallback in prepared:
        info = metadata[span].index_select(
            0, indices if device_metadata else
            (indices.cpu() if host_indices is None else host_indices)).clone()
        result = PackedBlocks(wire, logits, info, output_hypotheses,
                              candidate_wire=candidate_wire, **predictor_kwargs)
        if capture_top:
            result.eval_draft_top16 = tuple(value[span].index_select(0, indices).clone()
                                            for value in final_draft_top16)
            result.eval_draft_values = tuple(value[span].index_select(0, indices).clone()
                                             for value in final_draft_values)
        if label_passes is not None:
            result.lab_label_passes = int(label_passes)
        if lab_masks is not None:
            rows_at = indices.to(lab_masks[0].device)
            result.lab_written = lab_masks[0][span].index_select(0, rows_at).clone()
            if lab_masks[1] is not None:
                result.lab_written_p = lab_masks[1][span].index_select(0, rows_at).clone()
        if padded_keys is not None:
            result.row_keys = padded_keys
        if fallback is not None:
            result.posterior = True
            fallback_blocks = PackedBlocks(
                fallback[0], fallback[1], metadata[span][:1].clone(), hypotheses[:1],
                **fallback[2],
            )
            if label_passes is not None:
                fallback_blocks.lab_label_passes = int(label_passes)
            if os.environ.get('JETSPEC_HEADS_FAST') == '1':
                result._fallback_blocks = fallback_blocks
            else:
                result.fallback = fallback_blocks[0]
        output.append(result)
    host_mark('pr_ret')
    return output


def e04_attach_p_top1(block, p_top1, index):
    """e04 opt-in (E04_RECORD_PTOP1=1): carry the block's 15 per-slot drafter top-1 probabilities.

    Block is a frozen dataclass, so the row is attached out of band; nothing reads it except
    e04_record_p_top1 below, which writes it into that round's proposal record. The tensor already
    exists (packed_refinement freezes final_p_top1 per row at the pass where the row converged), so
    this adds one small clone per materialized block and no compute.
    """
    if os.environ['E04_RECORD_PTOP1'] != '1':
        raise ValueError('E04_RECORD_PTOP1 must be unset or 1')
    if p_top1 is None:
        raise ValueError('E04_RECORD_PTOP1 requires predictor serving (per-slot top-1 probabilities)')
    object.__setattr__(block, 'e04_p_top1', torch.as_tensor(p_top1[index]).detach().clone())






E04_KSCORE_EPS = 1e-6
E04_KSCORE_GRID = 2.0 ** 32
E04_KSCORE_DEC = ('argmax', 'top', 'pos')
E04_KSCORE_POSITION_SCALE = 14.0


def e04_kscore_blend(block, logits, ks, k_probabilities, *, after_miss, live=True, offers=None, written=None):
    """e04 opt-in (E04_KSCORE=blend): the unified reader's k score replaces the head's for the flat decoder set.

    On a normal round, slot i = 1..14 blends the drafter's top-1 probability p_top1[i] (the block's
    E04_RECORD_PTOP1 row) with the anchor decoder's fp32 softmax probability q[i-1] of draft d_i at decoder
    row i-1 (rejected_tokens in the wire frame, gathered as e04_record_q_shadow records it):
    x_i = sigmoid(a logit p_top1[i] + b logit q[i-1] + c), each probability clipped to [EPS, 1 - EPS];
    P(k) = prod_(i<=k) x_i (1 - x_(k+1)) for k < 14 and P(14) = prod_(i<=14) x_i over wire k 0..14, in
    float64 on the logits' device, then rounded to the 2^-32 grid so that an offline rebuild through other
    log/exp kernels reaches the same set. K is the LAC set of that score at the pass label's q-hat
    (thresholds k_predictor/<label>/<dataset>/<JETSPEC_KPRED_COVERAGE>), capped at JETSPEC_KSET_CAP, in
    ascending k; the coefficients are thresholds e04_kscore_blend/<dataset>/<a|b|c>/value of the same bundle.
    The returned 16-slot mass carries P(k) at 1 + k, so decoder_k_selection ranks the keyed rows by
    P(k) x p_anchor. After a miss the set and the mass are returned unchanged (K = {0} follows). The head's
    own mass stays on the block, so the round's record still carries the FR scores. A live round (the served
    sequence carries its seq id and candidate round) without decoder logits raises; the session warm-up's
    synthetic, unpublished round has no decoder tail and passes its set and mass through unchanged.

    E04_KSCORE=blend_dec (patch 0009) serves the decoder-feature BLEND score: slot i adds, in this order,
    + g 1{argmax of decoder row i-1 == d_i} + t logit top[i-1] + p i/14, where the argmax is the row's argmax
    id over the same fp32 logits (e04_record_q_shadow's shadow_argmax) and top[i-1] is the row's top
    post-exclusion probability, the largest of the round's decoder offers at row i-1 (the pool the wire
    serves, recorded as offers.top_probabilities[i-1][0]); the six coefficients are thresholds
    e04_kscore_blend/<dataset>/<a|b|c|argmax|top|pos>/value. A live normal round without the offers raises.

    ``written`` (E04_LAB_KSCORE_WRITTEN, tau_lab.written_terms; default None: the terms above): at the
    block's decoder-written slots i, the draft model's probability of d_i replaces p_top1[i] (draft_p,
    both) and the argmax indicator is 0 (disagree, both); every other term is unchanged.
    """
    if os.environ['E04_KSCORE'] not in ('blend', 'blend_dec'):
        raise ValueError('E04_KSCORE must be unset, blend or blend_dec')
    decoder_features = os.environ['E04_KSCORE'] == 'blend_dec'
    if (not decoder_blend_kset() or os.environ.get('JETSPEC_KPRED_FRAME') != 'wire'
            or os.environ.get('JETSPEC_DECODER_FRAME') != 'wire'
            or os.environ.get('JETSPEC_KSET_AFTER_MISS') != 'zero'):
        raise ValueError('E04_KSCORE=blend requires the predictor k set, the wire k and decoder frames and '
                         'JETSPEC_KSET_AFTER_MISS=zero')
    if os.environ.get('E04_RECORD_PTOP1') != '1' or os.environ.get('E04_RECORD_QSHADOW') != '1':
        raise ValueError('E04_KSCORE=blend requires E04_RECORD_PTOP1=1 and E04_RECORD_QSHADOW=1')
    if after_miss:
        return ks, k_probabilities
    if logits is None:
        if live:
            raise ValueError('E04_KSCORE=blend requires the decoder logits on every live normal proposal round')
        return ks, k_probabilities
    from jetspec.engines.jetspec_plus_plus.models.heads.decoder import rejected_tokens
    from jetspec.engines.jetspec_plus_plus.models.heads.fast import read_thresholds; from jetspec.engines.jetspec_plus_plus.models.heads.fast import dataset_entry
    from jetspec.engines.jetspec_plus_plus.models.heads.lac import lac_candidates
    dataset = os.environ['JETSPEC_DATASET']
    thresholds = read_thresholds(os.environ['JETSPEC_HEADS_DIR'])
    joint_kset = (os.environ.get('E04_ROWRULE') == 'joint' and kpred_forward_off()
                  and not any(name.startswith('E04_LAB_') for name in os.environ))
    q_hat = (None if joint_kset else
             dataset_entry(thresholds['k_predictor'][str(block.kpred_pass)], dataset)[
                 str(os.environ.get('JETSPEC_KPRED_COVERAGE', '0.99'))]['value'])
    coefficients = dataset_entry(thresholds['e04_kscore_blend'], dataset)
    a, b, c = (float(coefficients[name]['value']) for name in ('a', 'b', 'c'))
    if decoder_features:
        g, t, p = (float(coefficients[name]['value']) for name in E04_KSCORE_DEC)
        if offers is None:
            raise ValueError('E04_KSCORE=blend_dec requires the decoder offers on every live normal proposal round')
    lean = e04_lean()
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    n = draft_width()  # 14, or 15 drafts (rows 0..15, slots 1..15)
    logits = torch.as_tensor(logits).float()
    rejected = torch.as_tensor(rejected_tokens(block, after_miss=False), device=logits.device,
                               dtype=torch.long).reshape(-1)
    p_top1 = torch.as_tensor(block.e04_p_top1).to(device=logits.device).double().reshape(-1)
    if logits.ndim != 2 or logits.shape[0] != n + 1 or rejected.shape != (n + 1,) or p_top1.shape != (n + 1,):
        raise ValueError('E04_KSCORE=blend requires [15,V] decoder logits, 15 rejected ids and 15 top-1 probabilities')
    if not lean and bool((rejected[:n] < 0).any()):
        # JETSPEC_E04_LEAN=1 reads this check in the score's one trip home below
        raise ValueError('E04_KSCORE=blend requires a draft at every decoder row 0..13')
    q = logits.softmax(-1).gather(1, rejected.clamp_min(0)[:, None])[:, 0].double()
    drafter = p_top1[1:n + 1].clamp(E04_KSCORE_EPS, 1.0 - E04_KSCORE_EPS)
    if written is not None:
        drafter = written_drafter(written, drafter, n)  # E04_LAB_KSCORE_WRITTEN=draft_p / both
    decoder = q[:n].clamp(E04_KSCORE_EPS, 1.0 - E04_KSCORE_EPS)
    z = a * (drafter.log() - (-drafter).log1p()) + b * (decoder.log() - (-decoder).log1p()) + c
    if decoder_features:
        if any(torch.as_tensor(offers.probabilities[k]).numel() < 1 for k in range(n)):
            raise ValueError('E04_KSCORE=blend_dec requires an anchor offer at every decoder row 0..13')
        argmax_eq = (logits.argmax(-1)[:n] == rejected[:n]).double()
        if written is not None:
            argmax_eq = written_argmax(written, argmax_eq, n)  # E04_LAB_KSCORE_WRITTEN=disagree / both
        if lean:
            # the same maxima taken where the offers live (the host), then one copy
            top = torch.stack([torch.as_tensor(offers.probabilities[k]).double().reshape(-1).max()
                               for k in range(n)]).to(device=logits.device)
        else:
            top = torch.stack([torch.as_tensor(offers.probabilities[k]).to(device=logits.device).double()
                               .reshape(-1).max() for k in range(n)])
        top = top.clamp(E04_KSCORE_EPS, 1.0 - E04_KSCORE_EPS)
        position = torch.arange(1, n + 1, device=logits.device, dtype=torch.float64) / E04_KSCORE_POSITION_SCALE
        z = z + g * argmax_eq + t * (top.log() - (-top).log1p()) + p * position
    from jetspec.engines.jetspec_plus_plus.draft.method.rows import blend_mass
    mass = blend_mass(z)
    if lean:
        # one device-to-host copy carries the score and the draft check; the LAC set is
        # then taken on the host copy (the same stable ranking of the same float64 values)
        home = torch.cat((mass, rejected[:n].double())).cpu()
        mass = home[:n + 1]
        if bool((home[n + 1:] < 0).any()):
            raise ValueError('E04_KSCORE=blend requires a draft at every decoder row 0..13')
    if not bool(torch.isfinite(mass).all()):
        raise ValueError('E04_KSCORE=blend produced a non-finite k score')
    if joint_kset:
        ids = torch.arange(n + 1, device=mass.device)
    else:
        ids, _ = lac_candidates(mass, q_hat=q_hat, cap=int(os.environ.get('JETSPEC_KSET_CAP', '16')))
    return sorted(int(k) for k in ids.tolist()), torch.cat((mass.new_zeros(1), mass))



E04_ROWRULE_POOL = 32




def rows_chain_home_words(R, lanes):
    """Packed requested, keys, offer summary, BLEND mass and error word."""
    return 1 + 2 * R + 5 * lanes + (lanes + 1) + 1


def rows_chain_body(rp, tokens, offers_ids, offers_p64, offers_pack, dec_logits, p_top1_prev,
                    R, *, thresholds, dataset, tau_table, temperature,
                    out, previous_mass16=None):
    """U2's fixed-shape, capture-safe row body; all round choices come from ``rp``.

    ``out`` is the shared contract's preallocated keys/active/requested/row_home/err.
    The host may read row_home only after the publish event. No tensor value is
    inspected in Python here.
    """
    from jetspec.engines.jetspec_plus_plus.wire.chain.contract import Offset
    from jetspec.engines.jetspec_plus_plus.models.heads.assembly import DeviceOffers; from jetspec.engines.jetspec_plus_plus.models.heads.assembly import chain_joint_rows
    if os.environ.get('JETSPEC_ROWS_K3', '0') == '1' and dec_logits.is_cuda:
        from jetspec.engines.jetspec_plus_plus.kernels.rows_k3 import rows_k3_body
        return rows_k3_body(
            rp, tokens, offers_ids, offers_p64, offers_pack, dec_logits,
            p_top1_prev, R, thresholds=thresholds, dataset=dataset,
            tau_table=tau_table, temperature=temperature, out=out,
            previous_mass16=previous_mass16)
    lanes = offers_ids.shape[0]
    if R not in (4, 8, 16) or offers_pack.shape != (5, lanes):
        raise ValueError('chain rows require R in 4,8,16 and aligned offers')
    if out['row_home'].numel() != rows_chain_home_words(R, lanes):
        raise ValueError('chain row home buffer has the wrong size')
    block = SimpleNamespace(draft_ids=tokens.pending_draft_ids, e04_p_top1=p_top1_prev)
    mass16, checks = e04_kscore_blend_device(
        block, dec_logits, offers_p64, offers_pack[0], dataset=dataset, thresholds=thresholds)
    after_miss = rp[Offset.AFTER_MISS].bool()
    mass_home = (torch.where(after_miss, previous_mass16, mass16)
                 if previous_mass16 is not None else mass16)
    offers = DeviceOffers(
        ids=offers_ids[:, :E04_ROWRULE_POOL],
        probabilities=offers_p64[:, :E04_ROWRULE_POOL],
        valid=torch.arange(E04_ROWRULE_POOL, device=offers_ids.device)[None, :] <
              offers_pack[0, :, None],
        ks=torch.arange(lanes, device=offers_ids.device))
    stage, active, score = chain_joint_rows(
        mass16[1:], offers, after_miss=after_miss, tau_table=tau_table,
        tau_idx=rp[Offset.TAU_IDX], temperature=temperature, R=R)
    private = torch.stack((rp[Offset.CORRECTION].new_zeros(()), rp[Offset.CORRECTION]))
    first = torch.where(stage.requested > 0, stage.keys[0], private)
    keys = torch.cat((first[None], stage.keys[1:]), dim=0)
    scored = ~after_miss | (torch.arange(lanes, device=offers_ids.device) == 0)
    flags = torch.stack((
        (~offers_pack[3].bool()).any(),
        offers_pack[4].bool().any(),
        checks[0].bool() & ~after_miss,
        checks[1].bool() & ~after_miss,
        (~torch.isfinite(mass16)).any() & ~after_miss,
        ((offers_pack[0] < E04_ROWRULE_POOL) & scored).any(),
        ((~torch.isfinite(score)).any(1) & scored).any(),
    )).long()
    err = (flags * (1 << torch.arange(flags.numel(), device=flags.device))).sum().reshape(1)
    requested = stage.requested.reshape(1)
    home = torch.cat((requested, keys.reshape(-1), offers_pack.reshape(-1),
                      mass_home.view(torch.int64), err))
    for name, value in (('keys', keys), ('active', active), ('requested', requested),
                        ('row_home', home), ('err', err)):
        out[name].copy_(value)
    return out['keys'], out['active'], out['requested'], out['row_home'], out['err']




def e04_rowrule_joint(block, prediction, k_probabilities, ks, live_keys, requested, truncated, *, after_miss,
                      live=True):
    """e04 opt-in (E04_ROWRULE=joint): publish every (k, anchor) pair whose joint score reaches tau.

    The rows of a proposal round are no longer the LAC k set x top-m anchors. Every pair of a wire k and an
    anchor of that k's pool is scored jointly: P(k) is the BLEND k score of E04_KSCORE=blend (the quantized
    16-slot mass e04_kscore_blend returned, P(k) at 1 + k) on a normal round, and P(0) = 1 with only k = 0
    after a miss; the pool is the decoder's 32 offers of row k (anchor q-hat 1.0 and JETSPEC_ANCHOR_CAP=32,
    so the post-exclusion softmax top-32 with d_rej removed, exactly the ids and probabilities the recorder
    writes as offers.top_ids / top_probabilities); P_T(a | k) = p^(1/T) renormalized inside the pool; the
    score s = P(k) P_T(a | k) is rounded to the 2^-32 grid (E04_KSCORE_GRID). Every pair with s >= tau
    (the pass label's) and s > 0 is published, ranked by (-s, k, pool rank) as lac.keyed_rows ranks, first
    JETSPEC_ROWS_CAP. tau per pass label and T are thresholds e04_rowrule_joint/<dataset>/tau/<label>/value
    and e04_rowrule_joint/<dataset>/temperature/value of the served bundle. The returned k set is the served
    one (the distinct k of the published keys, ascending), so the record's k_set and the round's k_hit read
    the served k set. The session warm-up's synthetic, unpublished round has no decoder tail and passes its
    rows through unchanged; a live round without decoder offers raises, as does a missing BLEND score.
    """
    if os.environ['E04_ROWRULE'] != 'joint':
        raise ValueError('E04_ROWRULE must be unset or joint')
    if os.environ.get('E04_KSCORE') not in ('blend', 'blend_dec'):
        raise ValueError('E04_ROWRULE=joint requires the BLEND k score (E04_KSCORE=blend or blend_dec)')
    if os.environ.get('JETSPEC_ANCHOR_CAP') != str(E04_ROWRULE_POOL):
        raise ValueError('E04_ROWRULE=joint requires JETSPEC_ANCHOR_CAP=32 (the anchor pool)')
    if prediction is None:
        if live:
            raise ValueError('E04_ROWRULE=joint requires the decoder offers on every live proposal round')
        return ks, live_keys, requested, truncated
    from jetspec.engines.jetspec_plus_plus.models.heads.fast import read_thresholds; from jetspec.engines.jetspec_plus_plus.models.heads.fast import dataset_entry
    dataset = os.environ['JETSPEC_DATASET']
    rule = dataset_entry(read_thresholds(os.environ['JETSPEC_HEADS_DIR'])['e04_rowrule_joint'], dataset)
    tau = resolve_tau(rule['tau'], str(block.kpred_pass))
    temperature = float(rule['temperature']['value'])
    if not 0.0 <= tau <= 1.0 or not (0.0 < temperature < float('inf')):
        raise ValueError('E04_ROWRULE=joint requires tau in [0, 1] and a finite positive temperature')
    cap = rows_rule_cap()
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    n = draft_width()
    offsets = [0] if after_miss else list(range(n + 1))
    rows = [torch.as_tensor(prediction.offers.probabilities[k]).double().reshape(-1) for k in offsets]
    tokens = [torch.as_tensor(prediction.offers.ids[k], dtype=torch.long).reshape(-1) for k in offsets]
    if any(row.numel() != E04_ROWRULE_POOL for row in rows) or any(ids.numel() != E04_ROWRULE_POOL for ids in tokens):
        raise ValueError('E04_ROWRULE=joint requires 32 anchor offers at every scored decoder row')
    probabilities = torch.stack(rows)
    device = probabilities.device
    tokens = torch.stack(tokens).to(device)
    if after_miss:
        mass = probabilities.new_ones(1)
    else:
        mass = torch.as_tensor(k_probabilities).to(device=device, dtype=torch.float64).reshape(-1)
        if mass.shape != (n + 2,):
            raise ValueError('E04_ROWRULE=joint requires the 16-slot BLEND mass on a normal round')
        mass = mass[1:]
    from jetspec.engines.jetspec_plus_plus.draft.method.rows import joint_scores
    score = joint_scores(mass, probabilities, temperature).reshape(-1)
    if not bool(torch.isfinite(score).all()):
        raise ValueError('E04_ROWRULE=joint produced a non-finite pair score')
    passing = (score >= tau) & (score > 0.0)
    from jetspec.engines.jetspec_plus_plus.models.heads.assembly import floor_passing
    from jetspec.engines.jetspec_plus_plus.draft.serve_policy import rows_min_count
    floor = rows_min_count()
    if floor:
        passing = floor_passing(score, passing, torch.ones_like(passing), floor)
    order = torch.argsort(score, descending=True, stable=True)
    order = order[passing.index_select(0, order)]
    requested = int(order.numel())
    order = order[:cap]
    k_of = torch.tensor(offsets, device=device, dtype=torch.long).repeat_interleave(E04_ROWRULE_POOL)
    keys = torch.stack((k_of.index_select(0, order), tokens.reshape(-1).index_select(0, order)), dim=1)
    return sorted({int(k) for k in keys[:, 0].tolist()}), keys, requested, max(0, requested - cap)


def top1_probabilities(logits, chunk_rows=4, fast=False):
    """Compute top-1 probabilities without materializing a full-row clone."""
    if logits.ndim != 3 or chunk_rows < 1:
        raise ValueError("top-1 probability input must be [rows, slots, vocab]")
    if fast:
        # JETSPEC_BATCH_DRAFT_GRAPH's N>1 rows: one reduction kernel instead of a float
        # copy, a softmax and an amax per chunk (not bitwise: the fp32 sum's order differs)
        fast = top1_probabilities_fast(logits)
        if fast is not None:
            return fast
    from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled as eval_enabled
    if eval_enabled():
        # Keep Torch's exact per-vocabulary reduction while limiting both eager
        # scratch and each private graph pool to one candidate row's FP32 copy.
        chunk_rows = 1
    values = []
    for start in range(0, logits.shape[0], chunk_rows):
        chunk = logits[start:start + chunk_rows]
        values.append(chunk.float().softmax(-1).amax(-1))
    return torch.cat(values) if values else logits.new_empty(logits.shape[:2], dtype=torch.float32)








class PosteriorHitCounter:
    """Logical k coverage, independently of correction-pin survival."""
    def __init__(self):
        self.rounds = self.hits = self.logical_rows = 0
        self.seen = set()



def posterior_mode():
    return os.environ.get("JETSPEC_ROWS", "0") == "1" and os.environ.get("JETSPEC_KSET") in (
        "posterior", "posterior-topm")






def resolve_kset_count(configured):
    """Fixed served-chain frequency ranking: k=0..15, descending probability.

    Source: the measured served-chain k frequencies of the current recipe, equal
    dataset weights.
    This contiguous ranking keeps wire index == k and the existing miss fallback.
    """
    if os.environ.get('JETSPEC_ROWS','0')!='1':
        return configured
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    drafts = draft_width()
    # The served k set has at most 16 hypotheses even when verification uses 32 slots.
    width = 16 if drafts == 14 else min(16, drafts + 1)
    if kpred_forward_off():
        return width
    mode=os.environ.get('JETSPEC_KSET')
    if mode is None:
        return configured
    if mode=='one-k':
        return 1
    if mode=='all-k':
        return width
    if mode in ('posterior', 'posterior-topm'):
        posterior_parameters()
        return 16
    if mode == 'predictor':
        return width
    if mode=='top-m':
        m=int(os.environ.get('JETSPEC_KSET_M','7'))
        if not 1<=m<=16:
            raise ValueError('JETSPEC_KSET_M must be in 1..16')
        return m
    raise ValueError('JETSPEC_KSET must be one-k, top-m, all-k, posterior, posterior-topm, or predictor')


def candidate_wire_numel(capacity, hypotheses, branches=1):
    """Flags-off length is byte-compatible with the original candidate packet."""
    from jetspec.engines.jetspec_plus_plus.wire.protocol import anchor_mode
    if anchor_mode() == 'decoder':
        if int(os.environ.get('JETSPEC_TREE_B', '0')) != 0:
            raise ValueError('decoder anchor is not supported on the tree path')
        rows = int(os.environ.get('JETSPEC_ROWS_CAP', '128'))
        if rows < 1:
            raise ValueError('JETSPEC_ROWS_CAP must be positive')
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        return capacity * (6 + rows * (draft_width() + 3))
    tree = int(os.environ.get('JETSPEC_TREE_B', '0')) != 0
    return capacity * (6 + hypotheses * branches * (15 + (196 if tree else 0)))


def candidate_wire_views(wire, capacity, hypotheses, branches=1):
    """Aligned int64 packet; timing occupies a float64 bit view."""
    from jetspec.engines.jetspec_plus_plus.wire.protocol import anchor_mode
    if anchor_mode() == 'decoder':
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        rows, width = int(os.environ.get('JETSPEC_ROWS_CAP', '128')), draft_width() + 1
        sizes=(capacity,capacity*2,capacity*rows*width,capacity,capacity*2,capacity*rows*2)
        available,keys,blocks,timing,telemetry,row_keys=wire.split(sizes)
        return (available,keys.view(capacity,2),blocks.view(capacity,rows,width),
                timing.view(torch.float64),telemetry.view(capacity,2),row_keys.view(capacity,rows,2))
    sizes=(capacity,capacity*2,capacity*hypotheses*branches*15,capacity,capacity*2)
    tree = int(os.environ.get('JETSPEC_TREE_B', '0')) != 0
    if tree:
        sizes += (capacity*hypotheses*branches*98,)*2
    parts = wire.split(sizes)
    available,keys,blocks,timing,telemetry=parts[:5]
    result = (available,keys.view(capacity,2),blocks.view((capacity,hypotheses,15) if branches == 1 else (capacity,hypotheses,branches,15)),
            timing.view(torch.float64),telemetry.view(capacity,2))
    if tree:
        shape = (capacity,hypotheses) + (() if branches == 1 else (branches,)) + (14,7)
        result += (parts[5].view(shape),parts[6].view(torch.float64).view(shape))
    return result


class CandidateWire:
    """Persistent pinned control staging and one NCCL payload per sequence group."""
    def __init__(self,capacity,hypotheses,*,device,branches=1,publish=False):
        from jetspec.engines.jetspec_plus_plus.wire.protocol import anchor_mode
        self.capacity,self.hypotheses,self.branches=capacity,hypotheses,branches
        self.keyed=anchor_mode() == 'decoder'
        self.publish=publish
        packet_words=candidate_wire_numel(capacity,hypotheses,branches)
        if publish and (not self.keyed or branches != 1):
            raise ValueError('device publication requires flat keyed rows')
        self.rows=int(os.environ.get('JETSPEC_ROWS_CAP', '128')) if publish else 0
        if publish:
            self.wire_all=torch.empty(packet_words+capacity*self.rows*4+1,
                                      dtype=torch.int64,device=device)
            self.wire,self.check_word=self.wire_all[:packet_words],self.wire_all[-1]
            self.metadata_view=self.wire_all[packet_words:-1].view(capacity,self.rows,4)
        else:
            self.wire=self.wire_all=torch.empty(packet_words,dtype=torch.int64,device=device)
            self.check_word=self.metadata_view=None
        self.buffers=candidate_wire_views(self.wire,capacity,hypotheses,branches)
        self.control_cpu=torch.empty(capacity,6,dtype=torch.int64,pin_memory=torch.device(device).type=='cuda')
        self.control=torch.empty_like(self.control_cpu,device=device)
        rows=int(os.environ.get('JETSPEC_ROWS_CAP','128')) if self.keyed else hypotheses
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        width=draft_width()+1 if self.keyed else 15
        self.empty=torch.full((rows,width) if branches == 1 else (rows,branches,width),-1,dtype=torch.int64,device=device)
        self.empty_keys=torch.full((rows,2),-1,dtype=torch.int64,device=device) if self.keyed else None

    def _pack_control(self,sets,*,times,coalesced,gaps):
        if len(sets)!=self.capacity:
            raise ValueError('candidate rows must match wire capacity')
        data=[(0,-1,-1,0,0,0) if s is None else
              (1,int(s.seq_id),int(s.round_id),0,0,int(gaps.get((s.seq_id,s.round_id),0))) for s in sets]
        self.control_cpu.copy_(torch.tensor(data,dtype=torch.int64))
        if self.capacity:
            self.control_cpu[0,4]=int(coalesced)
        # Preserve the old float32 telemetry rounding, then bit-pack aligned f64.
        ms=torch.tensor([0.0 if s is None else times.get((s.seq_id,s.round_id),0.0) for s in sets],dtype=torch.float32).double()
        self.control_cpu[:,3].copy_(ms.view(torch.int64))
        self.control.copy_(self.control_cpu,non_blocking=True)
        available,keys,blocks,timing,telemetry=self.buffers[:5]
        available.copy_(self.control[:,0]);keys.copy_(self.control[:,1:3])
        timing.copy_(self.control[:,3].contiguous().view(torch.float64))
        telemetry.copy_(self.control[:,4:6])



    def pack(self,sets,*,times,coalesced,gaps):
        self._pack_control(sets,times=times,coalesced=coalesced,gaps=gaps)
        blocks=self.buffers[2]
        block_rows = [
            self.empty if s is None else s.blocks.to(
                device=self.wire.device, dtype=torch.int64)
            for s in sets
        ]
        if self.capacity > 1:
            torch.stack(block_rows, out=blocks)
        else:
            blocks.copy_(torch.stack(block_rows))
        if self.keyed:
            row_keys=self.buffers[5]
            key_rows = [
                self.empty_keys if s is None else s.row_keys.to(
                    device=self.wire.device, dtype=torch.int64)
                for s in sets
            ]
            if self.capacity > 1:
                torch.stack(key_rows, out=row_keys)
            else:
                row_keys.copy_(torch.stack(key_rows))
        if len(self.buffers) == 7:
            from jetspec.engines.jetspec_plus_plus.wire.protocol import pack_candidate_topk
            ids,logps=pack_candidate_topk(sets,hypotheses=self.hypotheses,
                branches=self.branches,device=self.wire.device)
            self.buffers[5].copy_(ids)
            self.buffers[6].copy_(logps.double())
        return self.wire




def cache_packed_metadata_batch(blocks):
    """Read the prior round's packed row metadata with one device-to-host transfer."""
    packed = [block for block in blocks if isinstance(block, PackedBlocks)]
    if not packed:
        return
    lengths = [len(block.metadata) for block in packed]
    if any(block._host_metadata is not None for block in packed):
        # A direct packet already brought those rows home beside the candidate wire.
        # Admission can mix CPU metadata into the same batch.
        rows = [(torch.as_tensor(block._host_metadata) if block._host_metadata is not None
                 else block.metadata.cpu()) for block in packed]
        metadata = torch.cat(rows).numpy()
    else:
        metadata = torch.cat([block.metadata for block in packed]).cpu().numpy()
    offset = 0
    for block, length in zip(packed, lengths):
        block._host_metadata = metadata[offset:offset + length]
        offset += length


def select_live_blocks_batch(blocks, true_ks, corrections):
    """Select the first exact row, then first matching k, then row zero, across a batch."""
    if (not blocks or not all(isinstance(block, PackedBlocks) and
            block._host_metadata is not None and len(block) == len(blocks[0]) and len(block)
            for block in blocks)):
        return [select_live_block(block, true_k=k, correction=c, report_miss=True)
                for block, k, c in zip(blocks, true_ks, corrections)]
    metadata = np.stack([block._host_metadata for block in blocks])
    hypotheses = np.asarray([block.hypotheses for block in blocks])
    matching_k = hypotheses == np.asarray(true_ks)[:, None]
    exact = matching_k & (metadata[:, :, 1] == np.asarray(corrections)[:, None])
    matched = exact.any(axis=1)
    first_k = matching_k.argmax(axis=1)
    first_exact = exact.argmax(axis=1)
    indices = np.where(matched, first_exact, first_k)
    return [(block[int(index)], not bool(hit))
            for block, index, hit in zip(blocks, indices, matched)]


def select_live_block(blocks, *, true_k, correction, report_miss=False):
    """Mirror verifier survivor selection when staging the next adapter tail."""
    matched = False
    if isinstance(blocks, PackedBlocks):
        metadata = blocks.metadata if blocks._host_metadata is None else blocks._host_metadata
        matching_k = [i for i, k in enumerate(blocks.hypotheses) if int(k) == int(true_k)]
        exact = [i for i in matching_k if int(metadata[i, 1]) == int(correction)]
        matched = bool(exact)
        index = (exact or matching_k or ([0] if len(blocks) else []))
        if index:
            selected = blocks[index[0]]
            return (selected, not matched) if report_miss else selected
        if hasattr(blocks, 'fallback'):
            return (blocks.fallback, True) if report_miss else blocks.fallback
    for hypothesis in blocks:
        branches = hypothesis if isinstance(hypothesis, (list,tuple)) else [hypothesis]
        if branches[0].hypothesis == true_k:
            exact = next((b for b in branches if b.correction_guess == correction), None)
            selected = branches[0] if exact is None else exact
            return (selected, exact is None) if report_miss else selected
    if not len(blocks) and hasattr(blocks, "fallback"):
        return (blocks.fallback, True) if report_miss else blocks.fallback
    first = blocks[0]
    selected = first[0] if isinstance(first,(list,tuple)) else first
    return (selected, True) if report_miss else selected


def live_hypotheses(hypotheses, *, block=None):
    if os.environ.get('JETSPEC_ROWS','0') != '1':
        return list(range(hypotheses))
    if kpred_forward_off():
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        return list(range(draft_width() + 1))
    mode = os.environ.get('JETSPEC_KSET')
    if mode is None or mode == 'all-k':
        return list(range(hypotheses))
    if mode == 'one-k':
        return [0]
    if mode in ('posterior', 'posterior-topm'):
        mode, q, maximum = posterior_parameters()
        probabilities = None if block is None else getattr(block, 'k_probabilities', None)
        if probabilities is not None:
            return select_posterior_mass(probabilities, mode=mode, q=q, maximum=maximum)
        return list(range(15)) if block is None else select_posterior_hypotheses(
            block.logits, mode=mode, q=q, maximum=maximum)
    if mode == 'predictor':
        if kpred_forward_off():
            return list(range(15))
        if block is None:
            return list(range(15))
        return selected_hypotheses(
            block,
            root=os.environ['JETSPEC_HEADS_DIR'],
            dataset=os.environ['JETSPEC_DATASET'],
            coverage=os.environ.get('JETSPEC_KPRED_COVERAGE', '0.99'),
            cap=int(os.environ.get('JETSPEC_KSET_CAP', '16')),
        )
    if mode != 'top-m':
        raise ValueError('JETSPEC_KSET must be one-k, top-m, all-k, posterior, posterior-topm, or predictor')
    order = list(map(int,os.environ.get('JETSPEC_KSET_ORDER', ','.join(map(str,range(hypotheses)))) .split(',')))
    m = int(os.environ.get('JETSPEC_KSET_M','7'))
    if not 1 <= m <= len(order) or len(set(order)) != len(order) or any(k<0 or k>=hypotheses for k in order):
        raise ValueError('invalid frozen k-frequency ordering')
    return sorted(order[:m])

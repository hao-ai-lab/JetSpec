"""F=1 batching of the live kth protocol; wire rows remain guess + 14 drafts."""
from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value

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
from jetspec.engines.jetspec_plus_plus.jetspec.pass_labels import completed_label
from jetspec.engines.jetspec_plus_plus.jetspec.pass_labels import resolve_tau

from jetspec.engines.jetspec_plus_plus.jetspec.drafter import Block
from jetspec.engines.jetspec_plus_plus.jetspec.drafter import DraftSuperseded
from jetspec.engines.jetspec_plus_plus.jetspec.drafter import DraftForwardPass
from jetspec.engines.jetspec_plus_plus.jetspec.drafter import build_random_context_block_token_ids
from jetspec.engines.jetspec_plus_plus.jetspec.drafter import run_refinement_protocol
from jetspec.engines.jetspec_plus_plus.jetspec.rows import RowDraftHead
from jetspec.engines.jetspec_plus_plus.jetspec.rows import RowLayout
from jetspec.engines.jetspec_plus_plus.jetspec.rows import BucketRowDraftHead
from jetspec.engines.jetspec_plus_plus.jetspec.rows import FastBucketRowDraftHead
from jetspec.engines.jetspec_plus_plus.jetspec.rows import project_row_logits
from jetspec.engines.jetspec_plus_plus.jetspec.rows import device_tensor
from jetspec.engines.jetspec_plus_plus.jetspec.rows import rows_ctx_incr
from jetspec.engines.jetspec_plus_plus.jetspec.rows import rows_ctx_early
from jetspec.engines.jetspec_plus_plus.jetspec.rows import EarlyContextKV


# serving W324_MARKS=1 (default off): host timestamps at named points of the draft round, read
# and cleared by the trace tool (tools/a100/trace_w240.py) to place the device gaps
HOST_MARKS = None


def host_mark(label, *, run_config=None):
    global HOST_MARKS
    if config_values(run_config).get('engine.engine_settings.w324_marks') == '1':
        if HOST_MARKS is None:
            HOST_MARKS = []
        HOST_MARKS.append((label, time.perf_counter()))




def batched_kth_random_context(sequences, *, block_size, seed, device):
    """Keep each request's CPU RNG draw and upload all gather indices at once."""
    indices, offset, contexts = [], 0, []
    for sequence in sequences:
        context = sequence.committed_ids.reshape(-1).to(device)
        length = context.numel()
        if length < 2:
            raise ValueError('kth mode requires a token before the correction')
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(seed) ^ int(sequence.prompt_index))
        draws = torch.randint(length, (block_size - 1,), generator=generator)
        indices.extend([offset + length - 2, *(offset + draws).tolist()])
        contexts.append(context)
        offset += length
    flat = torch.cat(contexts)
    return flat.index_select(0, device_tensor(indices, device)).reshape(len(sequences), block_size)


class _ContextSlots(dict):
    """The graphed rows' per-sequence context K/V shelf (a dict its slot finalizers can
    reference weakly)."""


def batch_draft_graph(*, run_config=None):
    """JETSPEC_BATCH_DRAFT_GRAPH=1 (default off): the N>1 draft chain runs CUDA graphs captured
    per batch bucket (batch_draft_bucket), with inactive rows padded up to the bucket. N=1
    never reads it."""
    value = config_values(run_config).get('engine.engine_settings.batch_draft_graph', '0')
    if value not in ('0', '1'):
        raise ValueError('JETSPEC_BATCH_DRAFT_GRAPH must be 0 or 1')
    if value == '1' and config_values(run_config).get('method.method_settings.adapter_refresh', 'off') != 'off':
        raise ValueError('JETSPEC_BATCH_DRAFT_GRAPH requires JETSPEC_ADAPTER_REFRESH=off')
    return value == '1'


def batch_draft_front(*, run_config=None):
    """JETSPEC_BATCH_DRAFT_GRAPH_FRONT=1 (default off, requires JETSPEC_BATCH_DRAFT_GRAPH=1): the
    front of the N>1 chain captured too -- the adapter's window query over per-sequence prefix
    K/V slots, the anchor decoder's window, and its prefix append. A stand-in until the N=1
    static front bodies land; it replaces JETSPEC_C1_MULTI_SLOT."""
    value = config_values(run_config).get('engine.engine_settings.batch_draft_graph_front', '0')
    if value not in ('0', '1'):
        raise ValueError('JETSPEC_BATCH_DRAFT_GRAPH_FRONT must be 0 or 1')
    if value == '1':
        if not batch_draft_graph(run_config=run_config):
            raise ValueError('JETSPEC_BATCH_DRAFT_GRAPH_FRONT requires JETSPEC_BATCH_DRAFT_GRAPH=1')
        if config_values(run_config).get('engine.engine_settings.c1_multi_slot', '0') == '1':
            raise ValueError('JETSPEC_BATCH_DRAFT_GRAPH_FRONT replaces JETSPEC_C1_MULTI_SLOT; unset it')
    return value == '1'


def batch_draft_bucket(count):
    """The captured batch size for ``count`` live requests: 2, 4, 8, 16, 32, then powers of two."""
    if count < 1:
        raise ValueError('a batch bucket needs at least one request')
    return max(2, 1 << (int(count) - 1).bit_length())


def e04_lean():
    """JETSPEC_E04_LEAN=1 (default off): the e04 joint-rule round without its dead k
    selection and keyed rows, and the BLEND k score with one trip home (exact)."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_value
    value = kernel_value('e04_lean', '0')
    if value not in ('0', '1'):
        raise ValueError('JETSPEC_E04_LEAN must be 0 or 1')
    return value == '1'


def rows_assembly(*, run_config=None):
    """JETSPEC_ROWS_ASSEMBLY=1 (default off): the round's k score, row rule and keyed rows in
    one rule-agnostic device stage with one trip home (heads/assembly.py); exact against the
    host path, which stays the reference and the fallback."""
    value = config_values(run_config).get('engine.engine_settings.rows_assembly', '0')
    if value not in ('0', '1'):
        raise ValueError('JETSPEC_ROWS_ASSEMBLY must be 0 or 1')
    return value == '1'


def rows_static(*, run_config=None):
    """JETSPEC_ROWS_STATIC=1 (default off; graphed regime with JETSPEC_ROWS_ASSEMBLY=1): the
    round's rows run in the fixed row-cap bucket.  The bucket's context K/V is issued before the
    anchor decoder, the geometry is rewritten from the assembly stage's device keys, and the
    stage's one trip home is read after the passes are queued, so no host read sits between the
    decoder and the passes.  The published keys are the stage's; a round whose offers need the
    exact full-row recompute reruns on the host path."""
    value = config_values(run_config).get('engine.engine_settings.rows_static', '0')
    if value not in ('0', '1'):
        raise ValueError('JETSPEC_ROWS_STATIC must be 0 or 1')
    return value == '1'


def rows_static_chain(*, run_config=None):
    """Enable the U2 fixed-cap row path, including refresh-capable preissue."""
    return config_flag01('engine.engine_settings.rows_static_chain', run_config)


def rows_defer_home(*, run_config=None):
    """Run a fixed-cap graphed bucket before reading the selected rows."""
    value = config_values(run_config).get('engine.engine_settings.rows_defer_home', '1')
    if value not in ('0', '1'):
        raise ValueError('JETSPEC_ROWS_DEFER_HOME must be 0 or 1')
    return value == '1'


def rows_device_checks(*, run_config=None):
    """Keep device flag checks separable from fixed-cap replay for measurement."""
    value = config_values(run_config).get('engine.engine_settings.rows_device_checks', '1')
    if value not in ('0', '1'):
        raise ValueError('JETSPEC_ROWS_DEVICE_CHECKS must be 0 or 1')
    return value == '1'


def rows_rule_cap(run_config=None):
    """JETSPEC_ROWS_RULE_CAP (serving, default: JETSPEC_ROWS_CAP): the joint rule's row cap on a
    proposal round, below the wire capacity JETSPEC_ROWS_CAP that the admission round's 16
    guessed rows (which do not go through the rule) still need."""
    capacity = int(config_values(run_config).get('engine.engine_settings.rows_cap', '128'))
    cap = int(config_values(run_config).get('method.method_settings.rows_rule_cap', str(capacity)))
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    cap = current_engine().budget.row_cap(cap, capacity)
    from jetspec.engines.jetspec_plus_plus.draft.serve_policy import rows_min_capacity
    cap = rows_min_capacity(cap, run_config=run_config)
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


def miss_redraft(*, run_config=None):
    """JETSPEC_MISS_REDRAFT=1 (default off): on a round whose pre-drafted rows miss the true
    (k, correction), both ranks see the miss at the round's request; the draft rank drafts one
    fresh row on the critical path (exact context, anchored on the committed correction) and the
    verify rank verifies that block instead of a one-token AR step."""
    return config_flag01('method.method_settings.miss_redraft', run_config)


def miss_redraft_force(*, run_config=None):
    """JETSPEC_MISS_REDRAFT_FORCE=1 (measurement only): every round after the first is served by
    the fallback, the serial floor of the fallback path."""
    return config_flag01('method.method_settings.miss_redraft_force', run_config)


def miss_redraft_anchor(*, run_config=None):
    """JETSPEC_MISS_REDRAFT_ANCHOR: ``none`` (default; the drafter predicts the token after the
    correction) or ``decoder`` (the anchor decoder's after-miss top-1 is pinned there)."""
    value = config_values(run_config).get('method.method_settings.miss_redraft_anchor', 'none')
    if value not in ('none', 'decoder'):
        raise ValueError('JETSPEC_MISS_REDRAFT_ANCHOR must be none or decoder')
    return value


def miss_redraft_passes(*, run_config=None):
    """JETSPEC_MISS_REDRAFT_PASSES: refinement passes of the fallback row (default 1 = J0)."""
    passes = int(config_values(run_config).get('method.method_settings.miss_redraft_passes', '1'))
    if not 1 <= passes <= 4:
        raise ValueError('JETSPEC_MISS_REDRAFT_PASSES must be in 1..4')
    return passes


def draft_deadline():
    """JETSPEC_DRAFT_DEADLINE=1 (default off): a round-ahead draft publishes its last completed
    pass instead of running past the verify's running median (minus
    JETSPEC_DRAFT_DEADLINE_MARGIN_MS, default 1.0) from the round start the draft rank sees."""
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    return current_engine().deadline


from jetspec.engines.jetspec_plus_plus.cutoffs import DeadlineClock, DraftDeadlineMiss
from jetspec.engines.jetspec_plus_plus.jetspec.config_values import config_values
from jetspec.engines.jetspec_plus_plus.jetspec.config_values import config_flag01




def unique_decoder_hypotheses(values, *, run_config=None):
    """Return valid decoder prefix lengths once, preserving wire order."""
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    return list(dict.fromkeys(int(k) for k in values if 0 <= int(k) <= draft_width(run_config=run_config)))


def decoder_k_selection(k_hypotheses, k_probabilities, *, predictor, after_miss, run_config=None):
    """Return decoder wire k values and their unnormalized selection masses."""
    if after_miss and config_values(run_config).get('method.method_settings.kset_after_miss') == 'zero':
        return [0], torch.ones(1, dtype=torch.float64)
    probabilities = torch.as_tensor(k_probabilities, device='cpu').double()
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    ks = (list(range(draft_width(run_config=run_config) + 1)) if config_values(run_config).get('method.method_settings.row_select') == 'budget'
          else unique_decoder_hypotheses(k_hypotheses, run_config=run_config))
    shift = int(bool(predictor) and config_values(run_config).get('method.method_settings.kpred_frame') == 'wire')
    indices = torch.tensor([k + shift for k in ks], device=probabilities.device,
                           dtype=torch.long)
    return ks, probabilities.index_select(0, indices)


def proposal_hypotheses(requested, *, predictor, decoder):
    """Keep one private row when a non-keyed dynamic set publishes nothing."""
    requested = list(requested)
    return [0] if predictor and not decoder and not requested else requested


def dynamic_wire_capacity(*, posterior, predictor):
    """Keep dynamic k sets stackable beside still-pending batch members."""
    return 16 if posterior or predictor is not None else None


def flat_bucket_host_fast(*, host_fast, graphed, run_config=None):
    """Avoid per-length mask banks when retaining all 24 pow2 graph buckets."""
    return bool(host_fast) and not (
        graphed and config_values(run_config).get('engine.engine_settings.g_row_buckets') == 'pow2')


def cached_flat_bucket(buckets, key, make, *, graphs, limit=None, run_config=None):
    """Retain the graph's pow2 row and prefix shapes across repeated rounds."""
    limit = max(27, limit or 0) if graphs and config_values(run_config).get('engine.engine_settings.g_row_buckets') == 'pow2' else 2
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


def adapter_refresh_mode(*, run_config=None):
    """JETSPEC_ADAPTER_REFRESH: ``off`` (default) or ``after_j0`` -- one adapter
    refresh of every drafted row's block between the first and second pass.
    ``after_j0_j1`` refreshes again between the second and third pass, from the
    pass-1 draft and logits."""
    mode = config_values(run_config).get('method.method_settings.adapter_refresh', 'off')
    if mode not in ('off', 'after_j0', 'after_j0_j1'):
        raise ValueError('JETSPEC_ADAPTER_REFRESH must be off, after_j0 or after_j0_j1')
    return mode


def refresh_pass_indices(mode):
    """The pass indices whose start runs a refresh from the previous pass."""
    return {'after_j0': (1,), 'after_j0_j1': (1, 2)}.get(mode, ())


def refresh_khat_threshold(again=False, *, run_config=None):
    from jetspec.engines.jetspec_plus_plus.draft.method.refresh import refresh_khat_threshold as threshold
    return threshold(again, threshold=config_values(run_config).get('engine.engine_settings.refresh_khat_t', '0.1'),
                     second_threshold=config_values(run_config).get('engine.engine_settings.refresh_khat_t2'))


def refresh_khat_cap(again=False, *, run_config=None):
    from jetspec.engines.jetspec_plus_plus.draft.method.refresh import refresh_khat_cap as cap
    from jetspec.engines.jetspec_plus_plus.draft.serve_policy import method_settings
    return (cap(again, second_cap=config_values(run_config).get('engine.engine_settings.refresh_khat_cap2'))
            if method_settings(run_config=run_config)['refresh_cap'] else None)


def refresh_khat(p_top1, threshold):
    from jetspec.engines.jetspec_plus_plus.draft.method.refresh import refresh_cutoff
    return refresh_cutoff(p_top1[:, 1:], threshold, start=2, final=15)


def row_layout_anchor0(*, run_config=None):
    """JETSPEC_ROW_LAYOUT=anchor0 (serving): a keyed row (k, a) puts the last accepted
    token's predicted hidden into the context tail (k + 1 tail rows) and starts the
    block at the pinned anchor, DFlash's shape: input [a, 15 slots], 15 draft outputs.
    Unset or ``kth``: the trained kth layout [last accepted, a, 14 slots]."""
    value = config_values(run_config).get('engine.engine_settings.row_layout', 'kth')
    if value not in ('kth', 'anchor0'):
        raise ValueError('JETSPEC_ROW_LAYOUT must be kth or anchor0')
    return value == 'anchor0'


def wide_rows(run_config=None):
    """Use every anchor0 block slot as a draft row."""
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    return draft_width(run_config=run_config) in (3, 7, 15, 31)


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


def refresh_pack_mode(*, run_config=None):
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value
    pack = kernel_value('refresh_pack', '0')
    if pack not in ('0', '1', '2'):
        raise ValueError('JETSPEC_REFRESH_PACK must be 0, 1 or 2')
    if kernel_enabled('rows_fused_graph', '1'):
        if pack == '1':
            raise ValueError('JETSPEC_ROWS_FUSED_GRAPH requires cached packed refresh')
        if (not kernel_enabled('refresh_lean', '1')
                or not kernel_enabled('c1_norm', 'wide')
                or config_values(run_config).get('engine.engine_settings.c1_kvcache', '0') != '1'):
            raise ValueError('JETSPEC_ROWS_FUSED_GRAPH refresh requires LEAN=1, C1_NORM=wide, C1_KVCACHE=1')
        return '2'
    return pack


def bucket_hypotheses(hypotheses, *, capacity):
    """Use stable k ordering when a dynamic set has a fixed graph capacity."""
    return list(range(capacity)) if capacity is not None else list(hypotheses)


def kpred_forward_off(*, run_config=None):
    """Whether decoder-BLEND serving omits the legacy standalone k predictor."""
    value = config_values(run_config).get('engine.engine_settings.legacy_kpred_forward', "0")
    if value not in ("0", "1"):
        raise ValueError("JETSPEC_LEGACY_KPRED_FORWARD must be 0 or 1")
    if value == "1":
        if config_values(run_config).get('method.method_settings.kset') != "predictor":
            raise ValueError(
                "JETSPEC_LEGACY_KPRED_FORWARD=1 requires JETSPEC_KSET=predictor")
        return False
    if config_values(run_config).get('method.method_settings.kscore') is None:
        if config_values(run_config).get('method.method_settings.kset') == "predictor":
            raise ValueError(
                "JETSPEC_KSET=predictor is legacy-only; set JETSPEC_LEGACY_KPRED_FORWARD=1")
        return False
    required = {
        "method.method_settings.kscore": "blend_dec",
        "method.method_settings.rowrule": "joint",
        "method.method_settings.kset_after_miss": "zero",
    }
    mismatched = [f"{name}={expected}" for name, expected in required.items()
                  if config_values(run_config).get(name) != expected]
    if mismatched:
        raise ValueError(
            "JETSPEC_LEGACY_KPRED_FORWARD=0 requires " + ", ".join(mismatched))
    return True


def decoder_blend_kset(*, run_config=None):
    """Whether the decoder-BLEND dynamic k policy is active, with or without its legacy head."""
    off = kpred_forward_off(run_config=run_config)
    return config_values(run_config).get('method.method_settings.kset') == "predictor" or off


def served_kpred_pass(passes_run, run_config=None):
    """Map refinement passes to the threshold label without running the head."""
    from jetspec.engines.jetspec_plus_plus.jetspec.heads.kpred import pass_name
    from jetspec.engines.jetspec_plus_plus.jetspec.profile import enabled as profile_enabled
    lab_on = any(name.startswith('E04_LAB_') for name in os.environ)
    from jetspec.engines.jetspec_plus_plus.jetspec.pass_labels import eval_completed_label
    label = (eval_completed_label(passes_run, run_config=run_config) if profile_enabled() else
             pass_name(passes_run, run_config=run_config) if config_values(run_config).get('engine.engine_settings.chain_graph', '0') == 'seg' or lab_on
             else completed_label(passes_run))
    import jetspec.engines.jetspec_plus_plus.jetspec.run_manifest as run_manifest
    run_manifest.observe("kpred_pass", (int(passes_run), label), run_config=run_config)
    return label


def load_kpredictor(drafter, *, run_config=None):
    """Load the legacy network only when its forward is part of the served path."""
    from jetspec.engines.jetspec_plus_plus.draft.explore import method_value
    selected = method_value(run_config, 'legacy_kpred')
    if selected is None:
        off = kpred_forward_off(run_config=run_config)
        enabled = config_values(run_config).get('method.method_settings.kset') == "predictor" and not off
    else:
        enabled = selected
    import jetspec.engines.jetspec_plus_plus.jetspec.run_manifest as run_manifest
    run_manifest.record_value("kpred_forward", "legacy" if enabled else "not_loaded", run_config=run_config)
    if not enabled:
        return None
    print("DEPRECATION JETSPEC_LEGACY_KPRED_FORWARD=1: the standalone k predictor is legacy-only",
          flush=True)
    from jetspec.engines.jetspec_plus_plus.draft.explore import implementation
    KPredictor = implementation("explore.legacy_kpred", "legacy_kpredictor")
    return KPredictor.load(
        config_values(run_config)['model.model_settings.heads_dir'], drafter.embed_tokens, device=drafter.device, run_config=run_config)


def require_segmented_rows(run_config):
    """Refuse retired eager rows before a launch allocates models or starts ranks."""
    from jetspec.engines.jetspec_plus_plus.draft.config import setting
    if (setting(run_config, 'engine.engine_settings.rows', '0') == '1'
            and setting(run_config, 'engine.engine_settings.chain_graph', '0') != 'seg'):
        raise ValueError('retired eager row path: JetSpec rows require chain_graph=seg; '
                         'its results are sealed records pinned to their commits')


class KthRowsDrafter:
    def __init__(self, drafter, *, backend='flashinfer', optimized=False, graphs=False, branches=1, f_graphs=False, run_config=None, target_model_path=None):
        self.run_config = run_config
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        from jetspec.engines.jetspec_plus_plus.jetspec.rows import check_rows_fused_regime
        check_rows_fused_regime(run_config=run_config)
        self.optimized = optimized
        self.host_fast = config_values(self.run_config).get('engine.engine_settings.host_fast', "0") == "1"
        self._host_stage_cache = {}
        self.graphs = graphs
        self.f_graphs = f_graphs
        self.backend = backend
        self.buckets = OrderedDict()
        if branches not in (1, 2, 3, 5):
            raise ValueError("branches must be 1, 2, 3, or 5")
        from jetspec.engines.jetspec_plus_plus.wire.protocol import anchor_mode
        self.branches = 1 if anchor_mode(run_config=self.run_config) == 'oracle' else branches
        self.drafter = drafter
        self.predictor = None
        import jetspec.engines.jetspec_plus_plus.jetspec.run_manifest as run_manifest
        run_manifest.record_value('kpred_forward', 'not_loaded', run_config=run_config)
        self.decoder = None
        if anchor_mode(run_config=self.run_config) == 'decoder':
            from jetspec.engines.jetspec_plus_plus.models.decoder_head import AnchorDecoder
            self.decoder = AnchorDecoder.load(config_values(self.run_config)['model.model_settings.heads_dir'], drafter, run_config=run_config,
                                              target_model_path=target_model_path)
        if self.predictor is not None or self.decoder is not None:
            import jetspec.engines.jetspec_plus_plus.jetspec.run_manifest as run_manifest
            run_manifest.record_value('heads_dir', os.path.realpath(config_values(self.run_config)['model.model_settings.heads_dir']), run_config=run_config)
            run_manifest.record_value('heads_dataset', config_values(self.run_config).get('benchmark.benchmark_settings.dataset'), run_config=run_config)
        # Initial live admission has one KV group per hypothesis; FlashInfer's
        # split-K scratch exceeds the bench's two-MiB allowance at 512 rows.
        self.row_head = RowDraftHead(drafter.head, backend=backend,
            workspace_bytes_per_row=4 * 1024 * 1024, run_config=self.run_config)


    def _attach_predictor_metrics(self, outputs, sequences, drafted_rows):
        if self.predictor is None:
            return outputs
        from jetspec.engines.jetspec_plus_plus.jetspec.heads.kpred import k0_in_lac
        from jetspec.engines.jetspec_plus_plus.jetspec.heads.kpred import k15_in_lac
        kpred_calls, kpred_total_ms = self.predictor.consume_metrics()
        for blocks, sequence, drafted in zip(outputs, sequences, drafted_rows):
            pending = sequence.pending_block
            ready = pending is not None and pending.k_probabilities is not None
            wire_frame = config_values(self.run_config).get('method.method_settings.kpred_frame') == 'wire'
            dropped = int(ready and not wire_frame and k15_in_lac(
                pending, root=config_values(self.run_config)['model.model_settings.heads_dir'],
                dataset=config_values(self.run_config)['benchmark.benchmark_settings.dataset'],
                coverage=config_values(self.run_config).get('method.method_settings.kpred_coverage', '0.99'), run_config=getattr(self, 'run_config', None)))
            dropped0 = int(ready and wire_frame and k0_in_lac(
                pending, root=config_values(self.run_config)['model.model_settings.heads_dir'],
                dataset=config_values(self.run_config)['benchmark.benchmark_settings.dataset'],
                coverage=config_values(self.run_config).get('method.method_settings.kpred_coverage', '0.99'), run_config=getattr(self, 'run_config', None)))
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
    def stage(self, sequences, blocks, run_config=None):
        """One shared-prefix adapter call with a masked, padded sequence batch."""
        if len(sequences) > 1 and config_values(self.run_config).get('engine.engine_settings.device_publish', '0') == '1':
            from copy import copy
            staged_blocks = []
            for block in blocks:
                staged = copy(block)
                object.__setattr__(staged, 'draft_ids', block.draft_ids.clone())
                staged_blocks.append(staged)
            blocks = staged_blocks
        if self.optimized:
            if len(sequences) > 1 and batch_draft_front(run_config=self.run_config):
                self._stage_batch_graph(sequences, blocks, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
                return
            if len(sequences) > 1 and config_values(self.run_config).get('engine.engine_settings.c1_kvcache', '0') == '1':
                if config_values(self.run_config).get('engine.engine_settings.c1_multi_slot', '0') == '1':
                    self._stage_packed(sequences, blocks, multi_cached=True, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
                    return
                # The reference C1 forward keeps each row's prefix/window mask
                # independent.  The last sequence still fills the single live
                # cache used by its following refresh, as in the serial path.
                if len(sequences) > 2:
                    self._stage_packed(sequences[:-1], blocks[:-1], batched_uncached=True, run_config=(run_config if run_config is not None else getattr(self, 'run_config', None)))
                    self._stage_packed(sequences[-1:], blocks[-1:], run_config=(run_config if run_config is not None else getattr(self, 'run_config', None)))
                else:
                    for sequence, block in zip(sequences, blocks):
                        self._stage_packed([sequence], [block], run_config=(run_config if run_config is not None else getattr(self, 'run_config', None)))
                return
            if self.host_fast and len(sequences) == 1:
                return self._stage_host_fast(sequences, blocks, run_config=(run_config if run_config is not None else getattr(self, 'run_config', None)))
            return self._stage_packed(sequences, blocks, run_config=(run_config if run_config is not None else getattr(self, 'run_config', None)))
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
    def propose(self, sequences, *, k_hypotheses, passes, initial=False, catchup=False, abort_if=None, oracle_predictions=None, run_config=None):
        """Draft round-zero admission rows from the committed prefill context."""
        if not initial or catchup:
            raise ValueError('KthRowsDrafter serves admission only (initial=True)')
        dynamic_k = decoder_blend_kset(run_config=self.run_config)
        requested_ks = list(k_hypotheses)
        if self.drafter.correction_head is not None and (
                self.branches > 1 or not self.optimized or oracle_predictions is not None):
            raise ValueError('the correction head serves only optimized F=1 rows')
        if (self.decoder is not None and initial and
                config_values(self.run_config).get('method.method_settings.row_select') == 'budget'):
            capacity = int(config_values(self.run_config).get('engine.engine_settings.rows_cap', '128'))
            if capacity < 1:
                raise ValueError('JETSPEC_ROWS_CAP must be positive')
            from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
            k_hypotheses = list(range(min(draft_width(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)) + 1, capacity)))
            requested_ks = list(k_hypotheses)
        if (self.decoder is not None and initial and not catchup and oracle_predictions is None
                and row_layout_anchor0(run_config=self.run_config)):
            # serving: the rows after prefill know the correction (the prefill's own token), so
            # under the anchor0 layout they are the exact DFlash round, not the kth bootstrap
            if self.branches != 1 or not self.optimized:
                raise ValueError('anchor0 initial rows require F=1, optimized, non-posterior rows')
            capacity = int(config_values(self.run_config).get('engine.engine_settings.rows_cap', '128'))
            if capacity < 1:
                raise ValueError('JETSPEC_ROWS_CAP must be positive')
            ks = list(dict.fromkeys(int(k) for k in k_hypotheses))[:capacity]
            from jetspec.engines.jetspec_plus_plus.jetspec.profile import enabled as profile_enabled
            if profile_enabled():
                ks = ks[:int(config_values(self.run_config).get('method.method_settings.eval_rows', '16'))]
            # every prefilled sequence gets its own exact rows (admission batches the prompts)
            # JETSPEC_BATCH_DRAFT_GRAPH (armed by prewarm_batch_graphs): the rows replay the
            # captured N=1 bucket of the miss redraft's shelf instead of an eager bucket
            admission = ({'graphs': True, 'admission': True}
                         if self.__dict__.get('_batch_admission_graphs', False) else {})
            output = [self._exact_anchor0_rows(sequence, ks=ks, passes=passes, **admission, run_config=(run_config if run_config is not None else getattr(self, 'run_config', None)))
                      for sequence in sequences]
            self._key_guessed_output(output, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
            return self._attach_predictor_metrics(output, sequences, [len(ks)] * len(sequences))
        k_hypotheses = proposal_hypotheses(
            k_hypotheses, predictor=decoder_blend_kset(run_config=self.run_config),
            decoder=self.decoder is not None)
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
            context = prefix[0, :-1] + marker[0]
            for _ in ks:
                contexts.append(context)
                prefix_lengths.append(context.shape[0])
                ksets.append([1])
            anchor = ids[:, -2:-1].expand(len(ks),1)
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
            return output

        if self.optimized:
            output = admission_refinement(forward, anchors=anchors, hypotheses=ks, sequences=len(sequences),
                passes=passes, convergence_limit=d.convergence_limit,
                output_hypotheses=requested_ks if dynamic_k else None,
                correction_head=d.correction_head, run_config=self.run_config)
            if self.decoder is not None:
                self._key_guessed_output(output, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
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
                            graphs=None, admission=False, run_config=None):
        """serving JETSPEC_ROW_LAYOUT=anchor0, a known correction (the miss fallback, and the
        initial rows after prefill): the exact DFlash round. Context = the target's own taps for
        every committed token before the correction (no adapter tail, no refresh); block =
        [correction, 15 slots] at the correction's position, J0 slots drawn from the context as
        in training; the correction is pinned in slot 0 and drafts are slots 1..15 (1..14 with
        14 drafts). One row per k in ``ks``, each keyed (k, correction).

        ``admission`` seeds the batched context slot from this bucket (``_stash_admission_context``)."""
        from jetspec.engines.jetspec_plus_plus.models.lac import row_bucket
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
        rows = row_bucket(n, cap=int(config_values(self.run_config).get('engine.engine_settings.rows_cap', '128')), graphed=graphs, run_config=getattr(self, 'run_config', None))
        refresh_at, total_passes, lab_fixed_passes, refresh = (), passes, False, False
        from jetspec.engines.jetspec_plus_plus.jetspec.profile import enabled as profile_enabled
        eval_mode = profile_enabled()
        if eval_mode:
            config = config_values(self.run_config)['method.method_settings.chain_fixed_config']
            refresh_at = tuple(range(1, int(config[-1]) + 1))
            total_passes, lab_fixed_passes, refresh = int(config[1]) + 1, True, True
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
                     else 6 if graphs and config_values(self.run_config).get('engine.engine_settings.g_row_buckets') == 'pow2' else 2)
            if len(kind) >= limit:
                buckets.pop(kind[0])
            buckets[key] = BucketPass(d, N=1, H=rows, prefix_bucket=prefix_bucket,
                backend=self.backend, host_fast=False, flat=True, graphs=graphs,
                anchor0=True, **({'wide': True} if wide_rows(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)) else {}),
                **({'refresh': True} if refresh else {}), run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
        bucket = buckets[key]
        buckets.move_to_end(key)
        bucket.head.update([exact], prefix.new_zeros(1, d.block_size if bucket.wide else 15, prefix.shape[-1]),
                           [0] * rows, mark=True)
        block0 = first.expand(rows, -1).clone()
        eval_swap = None
        launches = refreshes = 0
        if eval_mode:
            from jetspec.engines.jetspec_plus_plus.draft.admission_swap import prepare_eager_swap
            # Admission has an exact prefix and no predicted tail. Reuse the same
            # decoder swap and graph-backed draft bucket as the serving round.
            saved_hidden = sequence.staged.predicted_hidden
            sequence.staged.predicted_hidden = prefix.new_zeros(1, d.block_size, prefix.shape[-1])
            try:
                eval_swap = prepare_eager_swap(self.decoder, sequence, bucket, refresh_at, run_config=self.run_config)
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
                refreshed = exact_adapter_rows(d, sequence, block, run_config=run_config)
                active = torch.ones(rows, dtype=torch.bool, device=d.device)
                full = torch.full((rows,), d.block_size - 1, dtype=torch.long, device=d.device)
                before = block.clone() if getattr(eval_swap, 'states_only', False) else None
                inject = eval_swap.apply(length=eval_swap.length, ids=block, khat=full,
                                         slot_counts=full, tail_counts=torch.zeros_like(full),
                                         active=active, block=refreshed, finished=finished)
                if before is not None:
                    eval_swap.eval_effects[0].add_(block.ne(before).sum())
                if config_values(self.run_config).get('engine.engine_settings.refresh_inject_inclusive', '0') == '1':
                    inject = inject - 1
                bucket.head.inject_refresh(refreshed, inject, reset=pass_index > 1,
                                           fixed_layout=True, run_config=run_config)
                refreshes += 1
            logits = bucket(block)[:n]
            launches += 1
            return DraftForwardPass(logits=logits,
                                    input_tokens=block[:n] if bucket.wide else block[:n, :15],
                                    hidden=bucket.hidden[:n])

        keys = torch.tensor([[int(k), int(correction[0])] for k in ks], dtype=torch.long, device=d.device)
        output = admission_refinement(
            forward, anchors=ids[:, -2:-1].expand(n, 1), hypotheses=list(map(int, ks)), sequences=1,
            passes=total_passes, convergence_limit=d.convergence_limit,
            output_hypotheses=list(map(int, ks)), output_indices=list(range(n)), row_keys=keys,
            pinned_tokens=correction.expand(n), fixed_wire_capacity=int(config_values(self.run_config).get('engine.engine_settings.rows_cap', '128')),
            correction_head=d.correction_head,
            **({'refresh_swap': eval_swap} if eval_mode else {}),
            **({'fixed_passes': True} if lab_fixed_passes else {}),
            **({'label_passes': passes} if total_passes != passes else {}),
            run_config=self.run_config)
        if admission:
            self._stash_admission_context(sequence, bucket, exact.shape[0])
        if eval_mode:
            audit = dict(
                seq_id=getattr(sequence, '_jetspec_seq_id', -1), round=0, kind='proposal',
                j_passes=launches - 1, refreshes=refreshes, rows_published=n,
                expected_j=total_passes - 1, expected_r=len(refresh_at),
                expected_rows=int(config_values(self.run_config).get('method.method_settings.eval_rows', '16')),
                refresh_khat_skips=0)
            if getattr(eval_swap, 'states_only', False):
                audit.update(refresh_mode='states_only', swaps=int(eval_swap.eval_effects[0].item()),
                             pins=int(eval_swap.eval_effects[1].item()))
            print('EVAL_ROUND ' + json.dumps(audit), flush=True)
        return output[0]





    @torch.inference_mode()
    def _stash_admission_context(self, sequence, bucket, length):
        """JETSPEC_BATCH_DRAFT_GRAPH: the admission rows' bucket holds the context K/V of the
        request's ``length`` prompt rows (its taps plus marker 0 at their own positions), the
        rows ``_batch_graph_context`` would project eagerly into a new request's context slot in
        its first batched round. Copy them into that slot now (one copy per layer), so the
        round only appends. The slot may evict only a request outside the last batched round.
        Not bitwise to the eager slot fill (other GEMM row counts). False: nothing stashed."""
        store = self.__dict__.get('_ctx_store')
        head = bucket.head
        span = getattr(head, 'context_span', None)
        if (store is None or head.N != 1 or span is None or length > store['capacity']
                or length > head.P or len(store['kv']) != len(head.kv)):
            return False
        try:
            slot = self._ctx_slot(store, sequence, set(store.get('active', ())) | {id(sequence)})
        except ValueError:
            return False
        for cache, kv in zip(head.kv, store['kv']):
            kv[slot, :length].copy_(cache[span[0]:span[0] + length])
        store['lengths'][slot] = length
        store['generation'] = store.get('generation', 0) + 1
        return True






    def _key_guessed_output(self, outputs, *, run_config=None):
        """Bootstrap decoder mode with the ordinary rank-1 guesses, statically padded."""
        capacity = int(config_values(self.run_config).get('engine.engine_settings.rows_cap', '128'))
        for blocks in outputs:
            live = len(blocks)
            if live > capacity:
                raise ValueError('initial decoder rows exceed JETSPEC_ROWS_CAP')
            if wide_rows(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)) and blocks.block_wire.shape[-1] == 15:
                blocks.widen(run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
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








    def _ctx_slot(self, store, sequence, active):
        key = id(sequence)
        entry = store['slots'].get(key)
        if entry is not None and entry[1]() is sequence:
            return entry[0]
        if not store['free']:
            for old_key, (old_slot, _, finalizer) in list(store['slots'].items()):
                if old_key not in active:
                    finalizer.detach()
                    store['slots'].pop(old_key)
                    store['lengths'][old_slot] = 0
                    store['free'].append(old_slot)
                    break
            else:
                raise ValueError('batched rows context cache has no inactive sequence slots')
        slot = store['free'].pop()
        store['lengths'][slot] = 0
        shelf = weakref.ref(store)

        def drop():
            owner = shelf()
            if owner is not None and owner['slots'].get(key, (None,))[0] == slot:
                owner['slots'].pop(key, None)
                owner['lengths'][slot] = 0
                owner['free'].append(slot)
        store['slots'][key] = (slot, weakref.ref(sequence), weakref.finalize(sequence, drop))
        return slot











    @torch.inference_mode()
    def _stage_packed(self, sequences, blocks, *, batched_uncached=False, multi_cached=False,
                      admission=False, run_config=None):
        from jetspec.engines.jetspec_plus_plus.jetspec.drafter import ServingCtxKVC1Head
        d = self.drafter
        if admission:
            exact = sequences[0].exact_hidden(device=d.device, dtype=d.dtype)
            lengths = [exact.shape[1]] * len(sequences)
            prefix = exact.expand(len(sequences), -1, -1)
        else:
            prefixes = [s.exact_hidden(device=d.device, dtype=d.dtype)[0] for s in sequences]
            lengths = [p.shape[0] for p in prefixes]
            prefix = torch.nn.utils.rnn.pad_sequence(prefixes, batch_first=True)
        count, capacity = prefix.shape[:2]
        positions = torch.arange(capacity, device=d.device)
        length_tensor = device_tensor(lengths, d.device)
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        width = draft_width(run_config=run_config if run_config is not None else getattr(self, 'run_config', None))  # serving: the window is [c, d1..d_width]
        query = torch.arange(width + 1, device=d.device)
        allowed_prefix = (positions[None] < length_tensor[:, None])[:, None, :].expand(-1, width + 1, -1)
        if (batched_uncached or multi_cached) and isinstance(d.adapter.head, ServingCtxKVC1Head) and \
                d.adapter.head.resolved_config.get('window_causal_over_tokens', False):
            # The cached D1 path constructs this strictly earlier window itself.
            # The batched reference call reads the supplied query mask directly.
            allowed_query = (query[:, None] > query[None, :]).expand(count, -1, -1)
        else:
            allowed_query = (query[:, None] == query[None, :]).expand(count, -1, -1)
        allowed = torch.cat((allowed_prefix, allowed_query), dim=-1)[:, None]
        mask = prefix.new_zeros(allowed.shape).masked_fill_(~allowed, torch.finfo(d.dtype).min)
        if any(b.draft_ids.numel() != width for b in blocks):
            raise ValueError('live kth staging requires 14 draft ids')
        ids = torch.cat((torch.cat([s.committed_ids[:, -1:].to(d.device) for s in sequences]),
                         torch.stack([b.draft_ids.to(d.device) for b in blocks])), dim=1)
        previous = prefix[torch.arange(count, device=d.device), length_tensor-1][:, None]
        split = (count > 1 and config_values(self.run_config).get('engine.engine_settings.device_split', '0') == '1'
                 and d.device.type == 'cuda')
        adapter_start = torch.cuda.Event(enable_timing=True) if split else None
        if split:
            adapter_start.record()
        predicted = d.adapter.head(token_embeddings=d.embed_tokens(ids), h_prev=previous,
            prefix_h=prefix, prefix_position_ids=positions[None].expand(count, -1),
            query_position_ids=length_tensor[:, None]+query, attention_mask=mask,
            shared_prefix_projection=True,
            **({'batched_uncached': True} if batched_uncached else {}),
            **({'prefix_keys': sequences, 'prefix_lengths': lengths} if multi_cached else {}),
            **({"prefix_key": sequences[0] if len(sequences) == 1 else None, "run_config": run_config if run_config is not None else getattr(self, 'run_config', None)} if isinstance(d.adapter.head, ServingCtxKVC1Head) else {}))
        if split:
            adapter_end = torch.cuda.Event(enable_timing=True)
            adapter_end.record()
            self.__dict__.setdefault('_w434_adapter_events', []).append((adapter_start, adapter_end))
        # These are per-sequence protocol state transitions; tensor work is batched.
        if admission:
            sequences[-1].staged.stage(start_position=lengths[-1], token_ids=ids[-1:],
                                       predicted_hidden=predicted[-1:])
            sequences[-1].pending_block = blocks[-1]
            if isinstance(d.adapter.head, ServingCtxKVC1Head):
                d.adapter.head.prime_admission_prefix(sequences[-1], prefix[:1], run_config=(run_config if run_config is not None else getattr(self, 'run_config', None)))
            return predicted
        for row, (s, b, length) in enumerate(zip(sequences, blocks, lengths)):
            s.staged.stage(start_position=length, token_ids=ids[row:row+1], predicted_hidden=predicted[row:row+1])
            s.pending_block = b

    def stage_admission_batch(self, sequence, blocks, run_config=None):
        """Stage one prefill prefix against its published row blocks in one adapter call."""
        if config_values(self.run_config).get('engine.engine_settings.chain_admit_graph', '0') == '1' and self._admission_fast_ok(blocks):
            return self._stage_admission_fast(sequence, blocks, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
        return self._stage_packed([sequence] * len(blocks), blocks,
                                  batched_uncached=True, admission=True, run_config=(run_config if run_config is not None else getattr(self, 'run_config', None)))

    def _admission_fast_ok(self, blocks):
        from jetspec.engines.jetspec_plus_plus.jetspec.drafter import ServingCtxKVC1Head
        return (len(blocks) > 1 and isinstance(self.drafter.adapter.head, ServingCtxKVC1Head)
                and config_values(self.run_config).get('engine.engine_settings.device_split', '0') != '1')

    def _admission_forward(self, exact, ids, run_config=None):
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
        width = draft_width(run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
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
            shared_prefix_projection=True, batched_uncached=True, prefix_key=None, run_config=(run_config if run_config is not None else getattr(self, 'run_config', None)))

    @torch.inference_mode()
    def prewarm_admission_graph(self, length, count, run_config=None):
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
        ids = torch.zeros(key[1], draft_width(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)) + 1, dtype=torch.long, device=d.device)
        pool = self.__dict__.setdefault('_admission_graph_pool', torch.cuda.graph_pool_handle())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._admission_forward(exact, ids, run_config=(run_config if run_config is not None else getattr(self, 'run_config', None)))
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            output = self._admission_forward(exact, ids, run_config=(run_config if run_config is not None else getattr(self, 'run_config', None)))
        generator = torch.Generator(device=d.device)
        generator.manual_seed(key[0] * 131 + key[1])
        exact.copy_(torch.randn(exact.shape, generator=generator, device=d.device, dtype=torch.float32))
        ids.random_(0, d.embed_tokens.num_embeddings, generator=generator)
        expected = self._admission_forward(exact, ids, run_config=(run_config if run_config is not None else getattr(self, 'run_config', None))).clone()
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
    def _stage_admission_fast(self, sequence, blocks, run_config=None):
        """stage_admission_batch with the prewarmed graph of this prompt length, else eager."""
        from jetspec.engines.jetspec_plus_plus.jetspec.drafter import ServingCtxKVC1Head
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        d = self.drafter
        if any(b.draft_ids.numel() != draft_width(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)) for b in blocks):
            raise ValueError('live kth staging requires 14 draft ids')
        exact = sequence.exact_hidden(device=d.device, dtype=d.dtype)
        count = len(blocks)
        ids = torch.cat((sequence.committed_ids[:, -1:].to(d.device).expand(count, -1),
                         torch.stack([b.draft_ids.to(d.device) for b in blocks])), dim=1)
        entry = self.__dict__.get('_admission_graphs', {}).get((exact.shape[1], count))
        if entry is None:
            predicted = self._admission_forward(exact, ids, run_config=(run_config if run_config is not None else getattr(self, 'run_config', None)))
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
        if config_values(self.run_config).get('engine.engine_settings.chain_admit_graph_check', '0') == '1':
            # Check every admission, including an unseen-length eager fallback.
            same = torch.equal(predicted, self._admission_forward(exact, ids, run_config=(run_config if run_config is not None else getattr(self, 'run_config', None))))
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
            d.adapter.head.prime_admission_prefix(sequence, exact, run_config=(run_config if run_config is not None else getattr(self, 'run_config', None)))
        return predicted

    @torch.inference_mode()
    def _stage_batch_graph(self, sequences, blocks, *, run_config=None):
        """JETSPEC_BATCH_DRAFT_GRAPH_FRONT=1, N>1: the adapter's window query for all N requests
        in one captured call over per-sequence prefix K/V slots (padded to the batch bucket)."""
        from jetspec.engines.jetspec_plus_plus.jetspec.drafter import ServingCtxKVC1Head
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        d = self.drafter
        head = d.adapter.head
        if not isinstance(head, ServingCtxKVC1Head):
            raise ValueError('JETSPEC_BATCH_DRAFT_GRAPH requires the serving C1 adapter')
        width = draft_width(run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
        if any(b.draft_ids.numel() != width for b in blocks):
            raise ValueError('live kth staging requires JETSPEC_DRAFTS draft ids')
        prefixes = [s.exact_hidden(device=d.device, dtype=d.dtype)[0] for s in sequences]
        ids = torch.cat((torch.cat([s.committed_ids[:, -1:].to(d.device) for s in sequences]),
                         torch.stack([b.draft_ids.to(d.device) for b in blocks])), dim=1)
        split = config_values(self.run_config).get('engine.engine_settings.device_split', '0') == '1' and d.device.type == 'cuda'
        adapter_start = torch.cuda.Event(enable_timing=True) if split else None
        if split:
            adapter_start.record()
        predicted = head.forward_batch_graph(sequences, prefixes, d.embed_tokens(ids),
                                             bucket=batch_draft_bucket(len(sequences)), run_config=self.run_config)
        if split:
            adapter_end = torch.cuda.Event(enable_timing=True)
            adapter_end.record()
            self.__dict__.setdefault('_w434_adapter_events', []).append((adapter_start, adapter_end))
        for row, (s, b, prefix) in enumerate(zip(sequences, blocks, prefixes)):
            s.staged.stage(start_position=prefix.shape[0], token_ids=ids[row:row+1],
                           predicted_hidden=predicted[row:row+1])
            s.pending_block = b

    @torch.inference_mode()
    def _stage_host_fast(self, sequences, blocks, run_config=None):
        from jetspec.engines.jetspec_plus_plus.jetspec.drafter import ServingCtxKVC1Head
        d = self.drafter
        seq, block = sequences[0], blocks[0]
        # serving: the window is [c, d1..d_width] (16 rows with JETSPEC_DRAFTS=15), as _stage_packed
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        width = draft_width(run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
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
        host_mark('adapter_call', run_config=self.run_config)
        predicted = d.adapter.head(token_embeddings=d.embed_tokens(ids), h_prev=prefix[:, -1:],
            prefix_h=prefix, prefix_position_ids=positions[:length][None],
            query_position_ids=positions[length:][None], attention_mask=mask, shared_prefix_projection=True,
            **({"prefix_key": seq} if isinstance(d.adapter.head, ServingCtxKVC1Head) else {}))
        host_mark('adapter_ret', run_config=self.run_config)
        seq.staged.stage(start_position=length, token_ids=ids, predicted_hidden=predicted)
        seq.pending_block = block




    def _propose_branches(self, forward, contexts, prefix_lengths, ksets,
                          first, anchors, rejected, ks, count, passes, *, bucketed=False, bucket_factory=None, oracle_predictions=None):
        """One free pass, then F independently pinned rows per hypothesis."""
        d, F = self.drafter, self.branches
        first_pass = forward(fill_tokens=None, pass_index=0)
        if oracle_predictions is None:
            pins = top_anchor_candidates(first_pass.logits[:,0], rejected, F).reshape(-1)
        else:
            if F != 1 or oracle_predictions.shape != (count, 15) or any(k >= 15 for k in ks):
                raise ValueError('oracle pins require F=1 and verifier positions k=0..14')
            pins = oracle_predictions[:, ks].reshape(-1).to(first_pass.logits.device)
            if bool((pins < 0).any()):
                raise ValueError('oracle hypothesis lacks a verifier prediction')
        tokens = first_pass.logits.argmax(-1).repeat_interleave(F,0)
        tokens[:,0] = pins
        logits = first_pass.logits.repeat_interleave(F,0)
        limit = d.convergence_limit if passes is None else passes
        if limit < 1:
            raise ValueError('passes must be positive')
        preceding = anchors.repeat_interleave(F,0)
        if limit > 1:
            layout = RowLayout(prefix_lengths, ksets, F=F, device=d.device)
            self.row_head.prepare(layout, torch.cat(contexts))
        converged = torch.zeros(tokens.shape[0],dtype=torch.bool,device=d.device)
        for index in range(1,limit):
            ids = torch.cat([preceding,tokens],1)
            if bool((ids == int(d.head.mask_token_id)).any()):
                raise ValueError('Jacobi input must be mask-free')
            hidden = self.row_head(d.embed_tokens(ids))
            logits = project_row_logits(d.lm_head,hidden)[:,1:]
            updated = logits.argmax(-1)
            updated[:,0] = pins
            converged = (updated == tokens).all(1)
            tokens = updated
        return [[[
            Block(hypothesis=k, mode='kth', draft_ids=tokens[(n*len(ks)+j)*F+f,1:].detach().clone(),
                logits=logits[(n*len(ks)+j)*F+f,1:].detach().clone(), passes_run=limit,
                converged=bool(converged[(n*len(ks)+j)*F+f]),
                anchor=int(anchors[n*len(ks)+j,0]),
                correction_guess=int(pins[(n*len(ks)+j)*F+f]))
            for f in range(F)] for j,k in enumerate(ks)] for n in range(count)]




class BucketPass:
    """One captured embedding/head/projection pass; changing data keeps pointers."""
    def __init__(self,drafter,*,N,H,prefix_bucket,backend,F=1,row_capacity=None,
                 host_fast=False,flat=False,graphs=True,refresh=False,anchor0=False,wide=False,
                 pool=None,live_of=None,live_shelf=None, run_config=None):
        self.run_config = run_config
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value
        self.drafter=drafter
        self.graphs=bool(graphs)
        self.pool=pool
        # serving JETSPEC_ROW_LAYOUT=anchor0: the pass reads all 16 slot outputs
        self.anchor0=bool(anchor0)
        if self.anchor0 and (host_fast or not flat):
            raise ValueError('anchor0 rows are flat, non-host-fast rows only')
        # serving JETSPEC_DRAFTS=15: a 16-row tail (c, d1..d15) and all 16 slots published
        self.wide=bool(wide)
        self.B=getattr(drafter, 'block_size', 16) if self.wide else 16
        if self.wide and not self.anchor0:
            raise ValueError('15 drafts require the anchor0 layout')
        head_type=FastBucketRowDraftHead if host_fast else BucketRowDraftHead
        if live_of is not None:
            # JETSPEC_BATCH_ROWS_LIVE: N rows of the flat bucket head ``live_of`` (rows_live.py)
            from jetspec.engines.jetspec_plus_plus.draft.rows_live import CompactRowHead
            if ((host_fast) or (refresh) or (not flat)):
                raise ValueError('a live-rows pass is a flat pass without refresh')
            self.head=CompactRowHead(live_of,rows=N,shelf=live_shelf)
        else:
            self.head=head_type(drafter.head,N=N,H=H,prefix_bucket=prefix_bucket,
                backend=backend,F=F,row_capacity=row_capacity,flat=flat,graphs=self.graphs,
                run_config=run_config,
                **({'refresh': True} if refresh else {}),
                **({'tail_rows': self.B, 'block_size': self.B} if self.wide else {}))
        self.ids=torch.empty(N*self.head.R,self.B,dtype=torch.long,device=drafter.device)
        self.graph=None

    def step(self, *, hidden_only=False):
        d=self.drafter
        if self.wide:
            self.hidden=self.head(d.embed_tokens(self.ids))
            if hidden_only:
                return self.hidden
            return project_row_logits(d.lm_head,self.hidden)
        if self.anchor0:
            # serving: slots 0..14 (anchor, d1..d14) are the kth frame's 15 outputs; slot 15
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
                          standalone_refresh_at=(), exact_prefix=False, run_config=None):
    """Fixed-shape anchor0 Jacobi bodies for separate graph capture per pass.

    ``tokens`` is the per-sequence device TokenBuffers contract. The bodies read
    device RoundParams and row keys only; all host choices are captured in their
    shapes and the static ``refresh_at`` schedule.

    ``refresh_swap`` (serving JETSPEC_REFRESH_SWAP, default None: no change): a
    ``chain.refresh_swap.RefreshSwap``; J0 clears its pins, the refresh swaps and pins
    one draft per row and injects entries 0..khat_used, and every later pass keeps
    the pinned tokens.
    """
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    from jetspec.engines.jetspec_plus_plus.wire.contract import Offset
    from jetspec.engines.jetspec_plus_plus.wire.contract import SegmentId
    from jetspec.engines.jetspec_plus_plus.wire.contract import rng_words
    from jetspec.engines.jetspec_plus_plus.draft.glue import j0_prep
    from jetspec.engines.jetspec_plus_plus.draft.glue import mask_token_guard
    from jetspec.engines.jetspec_plus_plus.kernels.drafter_pass import allocate_partials
    from jetspec.engines.jetspec_plus_plus.kernels.drafter_pass import chain_pass
    from jetspec.engines.jetspec_plus_plus.kernels.drafter_pass import reference_chain_pass

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
    from jetspec.engines.jetspec_plus_plus.jetspec.profile import enabled as profile_enabled
    from jetspec.engines.jetspec_plus_plus.jetspec.profile import states_only
    eval_mode = profile_enabled()
    if eval_mode:
        outputs['eval_audit'] = torch.zeros(4, device=keys.device, dtype=torch.int64)
        if states_only(run_config=run_config):
            outputs['eval_refresh_effects'] = (refresh_swap.eval_effects
                if refresh_swap is not None else
                torch.zeros(2, device=keys.device, dtype=torch.int64))
    from jetspec.engines.jetspec_plus_plus.jetspec.profile import capture_candidates
    from jetspec.engines.jetspec_plus_plus.jetspec.profile import draft_candidates
    from jetspec.engines.jetspec_plus_plus.jetspec.profile import draft_token_values
    capture_top = capture_candidates(run_config=run_config)
    if capture_top:
        outputs['final_draft_top16_ids'] = torch.empty(rows, lanes, 16, device=keys.device, dtype=torch.long)
        outputs['final_draft_top16_probabilities'] = torch.empty(rows, lanes, 16, device=keys.device)
        outputs['final_draft_token_probabilities'] = torch.empty(rows, lanes, device=keys.device)
        outputs['final_draft_pinned'] = torch.empty(rows, lanes, device=keys.device, dtype=torch.bool)
        captured_pass = [None, None]
    p_top1_prev = torch.empty(rows, rng_words(lanes), device=keys.device, dtype=torch.float32)
    k3 = kernel_enabled('refine_k3', '1')
    fused_head = kernel_enabled('lmhead_fused', '1')
    head_kn = fused_head and kernel_enabled('j0_lmhead_kn', '1')
    if fused_head and (not k3 or keys.device.type != 'cuda'):
        raise ValueError('JETSPEC_LMHEAD_FUSED requires CUDA JETSPEC_REFINE_K3=1')
    if head_kn:
        from jetspec.engines.jetspec_plus_plus.kernels.projections import prepare_j0_lmhead_kn
        weight_kn = prepare_j0_lmhead_kn(bucket.drafter.lm_head)
        partials = allocate_partials(rows, lanes, bucket.drafter.lm_head.weight.shape[0], keys.device)
        bucket.chain_lmhead_partials = partials
    elif fused_head:
        from jetspec.engines.jetspec_plus_plus.kernels.drafter_lmhead import allocate_partials as allocate_head_partials
        partials = allocate_head_partials(rows, lanes,
                                          bucket.drafter.lm_head.weight.shape[0], keys.device)
        bucket.chain_lmhead_partials = partials
    else:
        partials = (allocate_partials(rows, lanes, bucket.drafter.lm_head.weight.shape[0],
                                      keys.device) if k3 and keys.device.type == 'cuda' else None)

    if refresh_swap is not None and refresh_swap.fill_mode != 'keep':
        refresh_swap.fill_mask_token = int(bucket.drafter.head.mask_token_id)
    refresh_stream = None
    from jetspec.engines.jetspec_plus_plus.profiles import transport_enabled
    if (refresh_at and transport_enabled('refresh_streams')
            and staged_hidden.dtype == torch.bfloat16):
        if (not staged_hidden.is_cuda or refresh_swap is None
                or not kernel_enabled('refresh_inject_lean2', '1')):
            raise ValueError('JETSPEC_REFRESH_STREAMS requires CUDA BF16 swap and lean2 injection')
        from jetspec.engines.jetspec_plus_plus.jetspec.chain.draft_streams import DraftStreams
        refresh_stream = DraftStreams(staged_hidden.device)
    refresh_projected = [None]

    def refresh(pass_index):
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        if T_ref < rows * rng_words(lanes):
            raise ValueError('B1 refresh needs fixed token capacity')
        d = bucket.drafter
        head = d.adapter.head
        window = getattr(head, '_c1_window', None)
        if window != lanes:
            raise ValueError(f'chain refresh window must equal live lanes: {window} != {lanes}')
        again = pass_index != refresh_at[0] if refresh_at else False
        chained = torch.cumprod(p_top1_prev.double(), dim=1)
        prep_lean = (kernel_enabled('refresh_prep_lean', '1')
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
                refresh_khat_threshold(again, run_config=run_config), refresh_khat_cap(again, run_config=run_config) if again else None)
        else:
            from jetspec.engines.jetspec_plus_plus.draft.method.refresh import refresh_bookkeeping
            khat, counts, tails, fixed_start = refresh_bookkeeping(
                chained, active, keys, rp[Offset.L], window,
                refresh_khat_threshold(again, run_config=run_config), refresh_khat_cap(again, run_config=run_config) if again else None)
        embedded = d.embed_tokens(bucket.ids)
        if exact_prefix:
            tails = torch.zeros_like(keys[:, 0])
        elif not prep_lean:
            tails = keys[:, 0] + 1
        key_width = None
        if config_values(run_config).get('engine.engine_settings.chain_refresh_key_bucket', '0') == '1':
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
            from jetspec.engines.jetspec_plus_plus.jetspec.refresh_inject import project

            # Keep the pre-swap allocation alive until the auxiliary work joins.
            def project_before_swap(cutoff=khat):
                refresh_projected[0] = project(bucket.head, block, cutoff, run_config=run_config)

            refresh_stream.start(project_before_swap)
        if refresh_swap is not None:
            # serving: the decoder's first disagreement lowers k-hat and swaps that draft
            before = bucket.ids.clone() if getattr(refresh_swap, 'states_only', False) else None
            khat = refresh_swap.apply(length=rp[Offset.L], ids=bucket.ids, khat=khat,
                                      slot_counts=counts,
                                      tail_counts=tails,
                                      active=active, block=block,
                                      finished=outputs['finished'])
            if before is not None:
                refresh_swap.eval_effects[0].add_(bucket.ids.ne(before).sum())
            if config_values(run_config).get('engine.engine_settings.refresh_inject_inclusive', '0') == '1':
                khat = khat - 1  # swap.apply returns a count, inclusive inject takes a cutoff
        if refresh_stream is None:
            bucket.head.inject_refresh(block, khat, reset=again,
                                       fixed_layout=True, run_config=run_config)
        else:
            from jetspec.engines.jetspec_plus_plus.jetspec.refresh_inject import commit
            refresh_stream.finish()
            commit(bucket.head, refresh_projected[0], khat, again)

    def pass_body(index, *, do_refresh=False):
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        if eval_mode:
            if index == 0:
                outputs['eval_audit'].zero_()
                outputs['eval_audit'][2].copy_(active.sum())
            else:
                outputs['eval_audit'][0].add_(1)
        if index == 0:
            if kernel_enabled('chain_mr_glue_lean', '1') and bucket.ids.is_cuda:
                j0_prep(tokens.committed_ids, rp, keys, active, bucket.ids,
                        outputs['finished'], outputs['passes_run'], outputs['err'],
                        owns_err=owns_outputs)
            else:
                draw_idx = rp[Offset.RNG_IDX:Offset.RNG_IDX + rng_words(lanes)]
                from jetspec.engines.jetspec_plus_plus.draft.method.passes import j0_fill
                first = j0_fill(tokens.committed_ids, block_size=lanes, indices=draw_idx,
                                pinned_first=keys[:, 1:2],
                                gather_vec=config_values(run_config).get('engine.engine_settings.chain_gather_vec', '0') == '1')
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
        elif ((kernel_enabled('chain_sega_glue', '2')
             and index in (0, 1))
                or (kernel_enabled('chain_mr_glue_lean', '1')
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
        # serving: after the refresh every pass's output keeps the round's pinned swaps
        pins = refresh_swap.pins() if refresh_swap is not None and index >= 1 else None
        if head_kn:
            logits = torch.mm(logits.reshape(-1, logits.shape[-1]), weight_kn).reshape(rows, lanes, -1)
            chain_pass(logits, bucket.ids, keys, active, outputs, p_top1_prev,
                       tokens, rp, partials, index,
                       save_previous=index == 0 or index + 1 in refresh_at, pins=pins)
            return
        if fused_head:
            from jetspec.engines.jetspec_plus_plus.kernels.drafter_lmhead import chain_pass as fused_chain_pass
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
        reference_chain_pass(logits, bucket.ids, keys, active, outputs,
                             p_top1_prev, tokens, rp, index,
                             save_previous=index == 0 or index + 1 in refresh_at,
                             pins=pins, p_top1=top1_probabilities(logits, chunk_rows=16))

    run_pass = pass_body
    if capture_top:
        def capture_pass(index, *, do_refresh=False):
            pass_body(index, do_refresh=do_refresh)
            pins = refresh_swap.pins() if refresh_swap is not None and index >= 1 else None
            pinned = torch.zeros_like(bucket.ids, dtype=torch.bool)
            if pins is not None:
                pinned.copy_(pins[0] if pins[0].ndim == 2 else
                             torch.arange(lanes, device=keys.device)[None] == pins[0][:, None])
            pinned[:, 0] = True  # The known correction is forced at the anchor slot.
            previous = (outputs['final_draft_token_probabilities'], outputs['final_draft_pinned'])
            values = draft_token_values(captured_pass[0], outputs['final_tokens'], pinned,
                                        previous if index else None, captured_pass[1])
            for target, value in zip(previous, values):
                target.copy_(value)
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
        from jetspec.engines.jetspec_plus_plus.draft.graph import tail_segments
        tails = tail_segments(run_config=run_config)
        if tails:
            # Tail passes preserve the recipe's threshold label and row convergence.
            saved_passes = torch.empty_like(outputs['passes_run'])
            outputs['tail_saved_passes'] = saved_passes

            def tail_body():
                saved_passes.copy_(outputs['passes_run'])
                run_pass(1)
                outputs['passes_run'].copy_(saved_passes)
                outputs['meta'][:, 2].copy_(saved_passes)

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
                 posterior_probabilities=None, row_keys=None, run_config=None):
        self.run_config = run_config
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

    def widen(self, *, run_config=None):
        """serving JETSPEC_DRAFTS=15: bootstrap rows come from the kth path (14 drafts); repeat
        the last draft as draft 15 (any token id verifies losslessly) and its top-1 slot."""
        import dataclasses
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        extra = draft_width(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)) + 1 - self.block_wire.shape[1]
        if extra < 0:
            raise ValueError('bootstrap block exceeds draft width')
        self.block_wire = torch.cat((self.block_wire, self.block_wire[:, -1:].expand(-1, extra)), dim=1)
        if self.predictor_p_top1 is not None:
            self.predictor_p_top1 = torch.cat((self.predictor_p_top1,
                self.predictor_p_top1[:, -1:].expand(-1, extra)), dim=1)
        self._blocks.clear()
        if self._fallback_blocks is not None:
            self._fallback_blocks.widen(run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
        elif self._fallback is not None:
            old = self._fallback
            block = dataclasses.replace(old, draft_ids=torch.cat((old.draft_ids,
                old.draft_ids[-1:].expand(extra))))
            if hasattr(old, 'e04_p_top1'):
                object.__setattr__(block, 'e04_p_top1', torch.cat((old.e04_p_top1,
                    old.e04_p_top1[-1:].expand(extra))))
            self._fallback = block

    def __len__(self):
        return len(self.hypotheses)

    def __getitem__(self, index, run_config=None):
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
            # serving JETSPEC_DRAFTS=15: the head reads its 15 trained slots [anchor, d1..d14];
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
                      served_kpred_pass(label_passes, run_config=self.run_config) if kpred_forward_off(run_config=self.run_config) else None)
        block = Block(hypothesis=self.hypotheses[index], mode='kth', draft_ids=self.block_wire[index, 1:],
            logits=None if self.logits is None else self.logits[index],
            passes_run=passes, converged=bool(converged),
            anchor=anchor, correction_guess=guess,
            k_probabilities=probabilities,
            kpred_pass=pass_label,
            kpred_ms=0.0 if prediction is None else prediction.elapsed_ms)
        if config_values(self.run_config).get('engine.engine_settings.record_ptop1'):
            e04_attach_p_top1(block, self.predictor_p_top1, index, run_config=self.run_config)
        if self.lab_written is not None:
            object.__setattr__(block, 'e04_lab_written', self.lab_written[index])
            if self.lab_written_p is not None:
                object.__setattr__(block, 'e04_lab_written_p', self.lab_written_p[index])
        self._blocks[index] = block
        return block




def e04_attach_p_top1(block, p_top1, index, *, run_config=None):
    """e04 opt-in (E04_RECORD_PTOP1=1): carry the block's 15 per-slot drafter top-1 probabilities.

    Block is a frozen dataclass, so the row is attached out of band; nothing reads it except
    e04_record_p_top1 below, which writes it into that round's proposal record. The tensor already
    exists (packed_refinement freezes final_p_top1 per row at the pass where the row converged), so
    this adds one small clone per materialized block and no compute.
    """
    if config_values(run_config)['engine.engine_settings.record_ptop1'] != '1':
        raise ValueError('E04_RECORD_PTOP1 must be unset or 1')
    if p_top1 is None:
        raise ValueError('E04_RECORD_PTOP1 requires predictor serving (per-slot top-1 probabilities)')
    object.__setattr__(block, 'e04_p_top1', torch.as_tensor(p_top1[index]).detach().clone())


def e04_record_p_top1(record, block, *, run_config=None):
    """e04 opt-in (E04_RECORD_PTOP1=1): add the 15 per-slot top-1 probabilities to a proposal record.

    They are the inputs the FR head and the product posterior both consume, so both k scores of a
    recorded round are recomputable offline from the round itself.
    """
    if config_values(run_config)['engine.engine_settings.record_ptop1'] != '1':
        raise ValueError('E04_RECORD_PTOP1 must be unset or 1')
    values = torch.as_tensor(block.e04_p_top1).double().reshape(-1)
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    if values.shape != (draft_width(run_config=run_config) + 1,) or not bool(torch.isfinite(values).all()):
        raise ValueError('E04_RECORD_PTOP1 requires 15 finite top-1 probabilities')
    return dict(record, p_top1=values.tolist())


def e04_record_q_shadow(record, logits, rejected, *, run_config=None):
    """e04 opt-in (E04_RECORD_QSHADOW=1): the anchor decoder's shadow k chain on a proposal record.

    Row k of the decoder logits (the output that predicts the slot after k accepted drafts) is the row
    whose rejected draft the record already carries as offers.d_rej[k]: in the wire frame d_(k+1), -1 at
    k = 14, and -1 at k = 0 after a miss under JETSPEC_KSET_AFTER_MISS=zero. q_shadow[k] is the fp32
    softmax probability of that draft at row k before the exclusion (None where d_rej is -1), and
    shadow_argmax[k] the row's argmax id. Both come from the logits the round already produced (no
    decoder compute): softmax, gather and argmax run on the logits' device, and one transfer moves
    q, argmax and the rejected ids to host after the record's own transfers, so no sync is added
    before them. The transferred ids must equal the recorded d_rej. A round whose decoder did not run
    (offers null) records both as None.
    """
    if config_values(run_config)['engine.engine_settings.record_qshadow'] != '1':
        raise ValueError('E04_RECORD_QSHADOW must be unset or 1')
    offers = record.get('offers')
    if logits is None or rejected is None or offers is None:
        if logits is not None or rejected is not None or offers is not None:
            raise ValueError('E04_RECORD_QSHADOW: decoder logits, rejected ids and recorded offers disagree')
        return dict(record, q_shadow=None, shadow_argmax=None)
    logits = torch.as_tensor(logits).float()
    rejected = torch.as_tensor(rejected, device=logits.device, dtype=torch.long).reshape(-1)
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    if logits.ndim != 2 or logits.shape[0] != draft_width(run_config=run_config) + 1 or rejected.shape != (draft_width(run_config=run_config) + 1,):
        raise ValueError('E04_RECORD_QSHADOW requires [15,V] decoder logits and 15 rejected ids')
    probabilities = logits.softmax(-1).gather(1, rejected.clamp_min(0)[:, None])[:, 0]
    host = torch.stack((probabilities.double(), logits.argmax(-1).double(), rejected.double())).tolist()
    ids = [int(value) for value in host[2]]
    if ids != [int(value) for value in offers['d_rej']]:
        raise ValueError('E04_RECORD_QSHADOW: rejected ids differ from the recorded offers.d_rej')
    q_shadow = [float(value) if token >= 0 else None for value, token in zip(host[0], ids)]
    if not all(value is None or 0.0 <= value <= 1.0 for value in q_shadow):
        raise ValueError('E04_RECORD_QSHADOW requires finite probabilities')
    return dict(record, q_shadow=q_shadow, shadow_argmax=[int(value) for value in host[1]],
                slot_index=list(range(1, draft_width(run_config=run_config) + 1)))


E04_KSCORE_EPS = 1e-6
E04_KSCORE_GRID = 2.0 ** 32
E04_KSCORE_DEC = ('argmax', 'top', 'pos')
E04_KSCORE_POSITION_SCALE = 14.0


def e04_kscore_blend(block, logits, ks, k_probabilities, *, after_miss, live=True, offers=None, written=None, run_config=None):
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
    if config_values(run_config)['method.method_settings.kscore'] not in ('blend', 'blend_dec'):
        raise ValueError('E04_KSCORE must be unset, blend or blend_dec')
    decoder_features = config_values(run_config)['method.method_settings.kscore'] == 'blend_dec'
    if (not decoder_blend_kset(run_config=run_config) or config_values(run_config).get('method.method_settings.kpred_frame') != 'wire'
            or config_values(run_config).get('engine.engine_settings.decoder_frame') != 'wire'
            or config_values(run_config).get('method.method_settings.kset_after_miss') != 'zero'):
        raise ValueError('E04_KSCORE=blend requires the predictor k set, the wire k and decoder frames and '
                         'JETSPEC_KSET_AFTER_MISS=zero')
    if config_values(run_config).get('engine.engine_settings.record_ptop1') != '1' or config_values(run_config).get('engine.engine_settings.record_qshadow') != '1':
        raise ValueError('E04_KSCORE=blend requires E04_RECORD_PTOP1=1 and E04_RECORD_QSHADOW=1')
    if after_miss:
        return ks, k_probabilities
    if logits is None:
        if live:
            raise ValueError('E04_KSCORE=blend requires the decoder logits on every live normal proposal round')
        return ks, k_probabilities
    from jetspec.engines.jetspec_plus_plus.models.decoder_head import rejected_tokens
    from jetspec.engines.jetspec_plus_plus.models.fast import read_thresholds
    from jetspec.engines.jetspec_plus_plus.models.fast import dataset_entry
    from jetspec.engines.jetspec_plus_plus.models.lac import lac_candidates
    dataset = config_values(run_config)['benchmark.benchmark_settings.dataset']
    thresholds = read_thresholds(run_config)
    joint_kset = (config_values(run_config).get('method.method_settings.rowrule') == 'joint' and kpred_forward_off(run_config=run_config)
                  and not any(name.startswith('E04_LAB_') for name in os.environ))
    q_hat = (None if joint_kset else
             dataset_entry(thresholds['k_predictor'][str(block.kpred_pass)], dataset)[
                 str(config_values(run_config).get('method.method_settings.kpred_coverage', '0.99'))]['value'])
    coefficients = dataset_entry(thresholds['e04_kscore_blend'], dataset)
    a, b, c = (float(coefficients[name]['value']) for name in ('a', 'b', 'c'))
    if decoder_features:
        g, t, p = (float(coefficients[name]['value']) for name in E04_KSCORE_DEC)
        if offers is None:
            raise ValueError('E04_KSCORE=blend_dec requires the decoder offers on every live normal proposal round')
    lean = e04_lean()
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    n = draft_width(run_config=run_config)  # serving: 14, or 15 drafts (rows 0..15, slots 1..15)
    logits = torch.as_tensor(logits).float()
    rejected = torch.as_tensor(rejected_tokens(block, after_miss=False, run_config=run_config), device=logits.device,
                               dtype=torch.long).reshape(-1)
    p_top1 = torch.as_tensor(block.e04_p_top1).to(device=logits.device).double().reshape(-1)
    if logits.ndim != 2 or logits.shape[0] != n + 1 or rejected.shape != (n + 1,) or p_top1.shape != (n + 1,):
        raise ValueError('E04_KSCORE=blend requires [15,V] decoder logits, 15 rejected ids and 15 top-1 probabilities')
    if not lean and bool((rejected[:n] < 0).any()):
        # JETSPEC_E04_LEAN=1 reads this check in the score's one trip home below
        raise ValueError('E04_KSCORE=blend requires a draft at every decoder row 0..13')
    q = logits.softmax(-1).gather(1, rejected.clamp_min(0)[:, None])[:, 0].double()
    drafter = p_top1[1:n + 1].clamp(E04_KSCORE_EPS, 1.0 - E04_KSCORE_EPS)
    decoder = q[:n].clamp(E04_KSCORE_EPS, 1.0 - E04_KSCORE_EPS)
    argmax_eq = top = None
    if decoder_features:
        if any(torch.as_tensor(offers.probabilities[k]).numel() < 1 for k in range(n)):
            raise ValueError('E04_KSCORE=blend_dec requires an anchor offer at every decoder row 0..13')
        argmax_eq = (logits.argmax(-1)[:n] == rejected[:n]).double()
        if lean:
            # the same maxima taken where the offers live (the host), then one copy
            top = torch.stack([torch.as_tensor(offers.probabilities[k]).double().reshape(-1).max()
                               for k in range(n)]).to(device=logits.device)
        else:
            top = torch.stack([torch.as_tensor(offers.probabilities[k]).to(device=logits.device).double()
                               .reshape(-1).max() for k in range(n)])
        top = top.clamp(E04_KSCORE_EPS, 1.0 - E04_KSCORE_EPS)
    from jetspec.engines.jetspec_plus_plus.draft.method.cache_rows import blend_features, blend_mass
    values = (a, b, c, g, t, p) if decoder_features else (a, b, c)
    z = blend_features(drafter, decoder, values, argmax_eq=argmax_eq, top=top)
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
        ids, _ = lac_candidates(mass, q_hat=q_hat, cap=int(config_values(run_config).get('method.method_settings.kset_cap', '16')))
    return sorted(int(k) for k in ids.tolist()), torch.cat((mass.new_zeros(1), mass))



E04_ROWRULE_POOL = 32


def e04_kscore_blend_device(block, logits, p64, sizes, *, dataset, thresholds, written=None, run_config=None):
    """serving: e04_kscore_blend's 16-slot mass on a live normal round, left on the device.

    The same device arithmetic in the same order; ``top`` (blend_dec) is each row's largest
    offer, taken from the device offers (``p64[k, :sizes[k]]``, the values the host offers
    copy).  No LAC set (the joint rule replaces it) and no host read: the checks the host
    path raises on come back as ``checks`` = [a draft missing at rows 0..13, an offer
    missing at rows 0..13] for the caller's one trip home.  ``written`` (E04_LAB_KSCORE_WRITTEN,
    default None) changes the decoder-written slots' terms as in e04_kscore_blend."""
    from jetspec.engines.jetspec_plus_plus.models.decoder_head import rejected_tokens
    decoder_features = config_values(run_config)['method.method_settings.kscore'] == 'blend_dec'
    from jetspec.engines.jetspec_plus_plus.models.fast import dataset_entry
    coefficients = dataset_entry(thresholds['e04_kscore_blend'], dataset)
    a, b, c = (float(coefficients[name]['value']) for name in ('a', 'b', 'c'))
    if decoder_features:
        g, t, p = (float(coefficients[name]['value']) for name in E04_KSCORE_DEC)
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    n = draft_width(run_config=run_config)  # serving: 14, or 15 drafts (rows 0..15, slots 1..15)
    logits = torch.as_tensor(logits).float()
    rejected = torch.as_tensor(rejected_tokens(block, after_miss=False, run_config=run_config), device=logits.device,
                               dtype=torch.long).reshape(-1)
    p_top1 = torch.as_tensor(block.e04_p_top1).to(device=logits.device).double().reshape(-1)
    if logits.ndim != 2 or logits.shape[0] != n + 1 or rejected.shape != (n + 1,) or p_top1.shape != (n + 1,):
        raise ValueError('E04_KSCORE=blend requires [15,V] decoder logits, 15 rejected ids and 15 top-1 probabilities')
    missing_draft = (rejected[:n] < 0).any()
    q = logits.softmax(-1).gather(1, rejected.clamp_min(0)[:, None])[:, 0].double()
    drafter = p_top1[1:n + 1].clamp(E04_KSCORE_EPS, 1.0 - E04_KSCORE_EPS)
    decoder = q[:n].clamp(E04_KSCORE_EPS, 1.0 - E04_KSCORE_EPS)
    argmax_eq = top = None
    missing_offer = torch.zeros((), dtype=torch.bool, device=logits.device)
    if decoder_features:
        missing_offer = (sizes[:n] < 1).any()
        argmax_eq = (logits.argmax(-1)[:n] == rejected[:n]).double()
        width = p64.shape[1]
        valid = torch.arange(width, device=p64.device)[None, :] < sizes[:n, None]
        top = torch.where(valid, p64[:n], float('-inf')).amax(-1).to(device=logits.device)
        top = top.clamp(E04_KSCORE_EPS, 1.0 - E04_KSCORE_EPS)
    from jetspec.engines.jetspec_plus_plus.draft.method.cache_rows import blend_features, blend_mass
    values = (a, b, c, g, t, p) if decoder_features else (a, b, c)
    z = blend_features(drafter, decoder, values, argmax_eq=argmax_eq, top=top)
    mass = blend_mass(z)
    return torch.cat((mass.new_zeros(1), mass)), torch.stack((missing_draft, missing_offer)).long()


def rows_chain_home_words(R, lanes):
    """Packed requested, keys, offer summary, BLEND mass and error word."""
    return 1 + 2 * R + 5 * lanes + (lanes + 1) + 1


def rows_chain_body(rp, tokens, offers_ids, offers_p64, offers_pack, dec_logits, p_top1_prev,
                    R, *, thresholds, dataset, tau_table, temperature,
                    out, previous_mass16=None, run_config=None):
    """Resolve engine configuration and call the method's fixed-shape cache-row rule."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    from jetspec.engines.jetspec_plus_plus.wire.contract import Offset
    if kernel_enabled('rows_k3', '1') and dec_logits.is_cuda:
        from jetspec.engines.jetspec_plus_plus.kernels.cache_rows_k3 import rows_k3_body
        from jetspec.engines.jetspec_plus_plus.draft.serve_policy import rows_min_count
        from jetspec.engines.jetspec_plus_plus.jetspec.profile import enabled as profile_enabled
        return rows_k3_body(
            rp, tokens, offers_ids, offers_p64, offers_pack, dec_logits,
            p_top1_prev, R, thresholds=thresholds, dataset=dataset,
            tau_table=tau_table, temperature=temperature, out=out,
            previous_mass16=previous_mass16,
            k_score=config_values(run_config)['method.method_settings.kscore'],
            decoder_frame=config_values(run_config).get('engine.engine_settings.decoder_frame'),
            floor=rows_min_count(run_config=run_config), fill_zero=profile_enabled())
    lanes = offers_ids.shape[0]
    if R not in (4, 8, 16) or offers_pack.shape != (5, lanes):
        raise ValueError('chain rows require R in 4,8,16 and aligned offers')
    if out['row_home'].numel() != rows_chain_home_words(R, lanes):
        raise ValueError('chain row home buffer has the wrong size')
    from jetspec.engines.jetspec_plus_plus.models.decoder_head import rejected_tokens
    from jetspec.engines.jetspec_plus_plus.models.fast import dataset_entry
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    block = SimpleNamespace(draft_ids=tokens.pending_draft_ids, e04_p_top1=p_top1_prev)
    rejected = rejected_tokens(block, after_miss=False, run_config=run_config)
    n = draft_width(run_config=run_config)
    if (dec_logits.ndim != 2 or dec_logits.shape[0] != n + 1
            or rejected.shape != (n + 1,) or p_top1_prev.reshape(-1).shape != (n + 1,)):
        raise ValueError('E04_KSCORE=blend requires [15,V] decoder logits, 15 rejected ids and 15 top-1 probabilities')
    coeff = dataset_entry(thresholds['e04_kscore_blend'], dataset)
    names = ('a', 'b', 'c') + (E04_KSCORE_DEC if config_values(run_config)['method.method_settings.kscore'] == 'blend_dec' else ())
    from jetspec.engines.jetspec_plus_plus.draft.method.cache_rows import cache_rows_chain
    from jetspec.engines.jetspec_plus_plus.draft.serve_policy import rows_min_count
    from jetspec.engines.jetspec_plus_plus.draft.serve_policy import method_settings
    return cache_rows_chain(
        tokens.pending_draft_ids, offers_ids, offers_p64, offers_pack, dec_logits,
        p_top1_prev, R, coefficients=tuple(float(coeff[name]['value']) for name in names),
        miss=rp[Offset.AFTER_MISS], tau_idx=rp[Offset.TAU_IDX], correction=rp[Offset.CORRECTION],
        tau_table=tau_table, temperature=temperature, out=out, previous_mass16=previous_mass16,
        floor=rows_min_count(run_config=run_config), fill_zero=method_settings(run_config=run_config)['fill_zero'], control=rp)


def e04_kscore_blend_batch_device(blocks, logits, p64, sizes, *, dataset, thresholds,
                                  after_misses, rejected=None, compact_q=None,
                                  compact_argmax_eq=None, run_config=None):
    """Score independent decoder rounds together, preserving the scalar BLEND arithmetic."""
    from jetspec.engines.jetspec_plus_plus.models.decoder_head import rejected_tokens
    from jetspec.engines.jetspec_plus_plus.models.fast import dataset_entry
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    n = draft_width(run_config=run_config)
    count = len(blocks)
    if (p64.shape[:2] != (count, n + 1)
            or (logits is not None and logits.shape[:2] != (count, n + 1))
            or (logits is None and (compact_q is None or compact_q.shape != (count, n + 1)
                                    or compact_argmax_eq is None
                                    or compact_argmax_eq.shape != (count, n + 1)))):
        raise ValueError('batched BLEND inputs must align with decoder rows')
    coefficients = dataset_entry(thresholds['e04_kscore_blend'], dataset)
    a, b, c = (float(coefficients[name]['value']) for name in ('a', 'b', 'c'))
    decoder_features = config_values(run_config)['method.method_settings.kscore'] == 'blend_dec'
    if decoder_features:
        g, t, p = (float(coefficients[name]['value']) for name in E04_KSCORE_DEC)
    if rejected is None:
        rejected = torch.stack([rejected_tokens(block, after_miss=False, run_config=run_config) for block in blocks])
    p_top1 = torch.stack([block.e04_p_top1.double().reshape(-1) for block in blocks])
    device = p64.device
    if rejected.device != device:
        rejected = (rejected.pin_memory() if rejected.device.type == 'cpu'
                    else rejected).to(device, non_blocking=True)
    if p_top1.device != device:
        p_top1 = (p_top1.pin_memory() if p_top1.device.type == 'cpu'
                  else p_top1).to(device, non_blocking=True)
    rejected = rejected.long()
    if rejected.shape != (count, n + 1) or p_top1.shape != (count, n + 1):
        raise ValueError('batched BLEND requires one rejected id and top-1 value per decoder row')
    missing_draft = (rejected[:, :n] < 0).any(1)
    q = (compact_q if logits is None else
         logits.float().softmax(-1).gather(2, rejected.clamp_min(0)[..., None])[..., 0]).double()
    drafter = p_top1[:, 1:n + 1].clamp(E04_KSCORE_EPS, 1.0 - E04_KSCORE_EPS)
    decoder = q[:, :n].clamp(E04_KSCORE_EPS, 1.0 - E04_KSCORE_EPS)
    argmax_eq = top = None
    missing_offer = torch.zeros(count, dtype=torch.bool, device=device)
    if decoder_features:
        missing_offer = (sizes[:, :n] < 1).any(1)
        argmax_eq = (compact_argmax_eq[:, :n] if logits is None else
                     logits.argmax(-1)[:, :n] == rejected[:, :n]).double()
        valid = torch.arange(p64.shape[-1], device=p64.device)[None, None, :] < sizes[:, :n, None]
        top = torch.where(valid, p64[:, :n], float('-inf')).amax(-1)
        top = top.clamp(E04_KSCORE_EPS, 1.0 - E04_KSCORE_EPS)
    from jetspec.engines.jetspec_plus_plus.draft.method.cache_rows import blend_features, blend_mass
    values = (a, b, c, g, t, p) if decoder_features else (a, b, c)
    z = blend_features(drafter, decoder, values, argmax_eq=argmax_eq, top=top)
    mass = blend_mass(z)
    mass16 = torch.cat((mass.new_zeros(count, 1), mass), dim=1)
    checks = torch.stack((missing_draft, missing_offer), dim=1).long()
    miss = device_tensor(after_misses, device, torch.bool)
    scoring_mass = torch.where(miss[:, None], torch.nn.functional.one_hot(
        torch.zeros(count, dtype=torch.long, device=device), n + 1).double(), mass)
    return mass16, scoring_mass, checks


def e04_rowrule_joint(block, prediction, k_probabilities, ks, live_keys, requested, truncated, *, after_miss,
                      live=True, run_config=None):
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
    if config_values(run_config)['method.method_settings.rowrule'] != 'joint':
        raise ValueError('E04_ROWRULE must be unset or joint')
    if config_values(run_config).get('method.method_settings.kscore') not in ('blend', 'blend_dec'):
        raise ValueError('E04_ROWRULE=joint requires the BLEND k score (E04_KSCORE=blend or blend_dec)')
    if config_values(run_config).get('method.method_settings.anchor_cap') != str(E04_ROWRULE_POOL):
        raise ValueError('E04_ROWRULE=joint requires JETSPEC_ANCHOR_CAP=32 (the anchor pool)')
    if prediction is None:
        if live:
            raise ValueError('E04_ROWRULE=joint requires the decoder offers on every live proposal round')
        return ks, live_keys, requested, truncated
    from jetspec.engines.jetspec_plus_plus.models.fast import read_thresholds
    from jetspec.engines.jetspec_plus_plus.models.fast import dataset_entry
    dataset = config_values(run_config)['benchmark.benchmark_settings.dataset']
    rule = dataset_entry(read_thresholds(run_config)['e04_rowrule_joint'], dataset)
    tau = resolve_tau(rule['tau'], str(block.kpred_pass))
    temperature = float(rule['temperature']['value'])
    if not 0.0 <= tau <= 1.0 or not (0.0 < temperature < float('inf')):
        raise ValueError('E04_ROWRULE=joint requires tau in [0, 1] and a finite positive temperature')
    cap = rows_rule_cap(run_config=run_config)
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    n = draft_width(run_config=run_config)
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
    from jetspec.engines.jetspec_plus_plus.draft.method.cache_rows import DeviceOffers, joint_rule, select_rows
    from jetspec.engines.jetspec_plus_plus.draft.serve_policy import rows_min_count
    offers = DeviceOffers(tokens, probabilities, torch.ones_like(tokens, dtype=torch.bool),
                          torch.tensor(offsets, device=device, dtype=torch.long))
    score, passing = joint_rule(tau, temperature, floor=rows_min_count(run_config=run_config))(mass, offers)
    if not bool(torch.isfinite(score).all()):
        raise ValueError('E04_ROWRULE=joint produced a non-finite pair score')
    count, keys = select_rows(score, passing, offers, cap, cap)
    requested = int(count)
    keys = keys[:min(requested, cap)]
    return sorted({int(k) for k in keys[:, 0].tolist()}), keys, requested, max(0, requested - cap)


def top1_probabilities(logits, chunk_rows=4, fast=False):
    """Compute top-1 probabilities without materializing a full-row clone."""
    if logits.ndim != 3 or chunk_rows < 1:
        raise ValueError("top-1 probability input must be [rows, slots, vocab]")
    if fast:
        # JETSPEC_BATCH_DRAFT_GRAPH's N>1 rows: one reduction kernel instead of a float
        # copy, a softmax and an amax per chunk (not bitwise: the fp32 sum's order differs)
        from jetspec.engines.jetspec_plus_plus.jetspec.rows import top1_probabilities_fast
        fast = top1_probabilities_fast(logits)
        if fast is not None:
            return fast
    from jetspec.engines.jetspec_plus_plus.jetspec.profile import enabled as profile_enabled
    if profile_enabled():
        # Keep Torch's exact per-vocabulary reduction while limiting both eager
        # scratch and each private graph pool to one candidate row's FP32 copy.
        chunk_rows = 1
    values = []
    for start in range(0, logits.shape[0], chunk_rows):
        chunk = logits[start:start + chunk_rows]
        values.append(chunk.float().softmax(-1).amax(-1))
    return torch.cat(values) if values else logits.new_empty(logits.shape[:2], dtype=torch.float32)


















def candidate_wire_numel(capacity, hypotheses, branches=1, *, run_config=None):
    """Flags-off length is byte-compatible with the original serving packet."""
    from jetspec.engines.jetspec_plus_plus.wire.protocol import anchor_mode
    if int(config_values(run_config).get('method.method_settings.tree_b', '0')) != 0:
        raise ValueError('retired tree wire requires a sealed historical read')
    if anchor_mode(run_config=run_config) == 'decoder':
        rows = int(config_values(run_config).get('engine.engine_settings.rows_cap', '128'))
        if rows < 1:
            raise ValueError('JETSPEC_ROWS_CAP must be positive')
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        return capacity * (6 + rows * (draft_width(run_config=run_config) + 3))
    return capacity * (6 + hypotheses * branches * 15)


def candidate_wire_views(wire, capacity, hypotheses, branches=1, *, run_config=None):
    """Aligned int64 packet; timing occupies a float64 bit view."""
    from jetspec.engines.jetspec_plus_plus.wire.protocol import anchor_mode
    if int(config_values(run_config).get('method.method_settings.tree_b', '0')) != 0:
        raise ValueError('retired tree wire requires a sealed historical read')
    if anchor_mode(run_config=run_config) == 'decoder':
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        rows, width = int(config_values(run_config).get('engine.engine_settings.rows_cap', '128')), draft_width(run_config=run_config) + 1
        sizes=(capacity,capacity*2,capacity*rows*width,capacity,capacity*2,capacity*rows*2)
        available,keys,blocks,timing,telemetry,row_keys=wire.split(sizes)
        return (available,keys.view(capacity,2),blocks.view(capacity,rows,width),
                timing.view(torch.float64),telemetry.view(capacity,2),row_keys.view(capacity,rows,2))
    sizes=(capacity,capacity*2,capacity*hypotheses*branches*15,capacity,capacity*2)
    parts = wire.split(sizes)
    available,keys,blocks,timing,telemetry=parts[:5]
    result = (available,keys.view(capacity,2),blocks.view((capacity,hypotheses,15) if branches == 1 else (capacity,hypotheses,branches,15)),
            timing.view(torch.float64),telemetry.view(capacity,2))
    return result


class CandidateWire:
    """Persistent pinned control staging and one NCCL payload per sequence group."""
    def __init__(self,capacity,hypotheses,*,device,branches=1,publish=False, run_config=None):
        self.run_config = run_config
        from jetspec.engines.jetspec_plus_plus.wire.protocol import anchor_mode
        self.capacity,self.hypotheses,self.branches=capacity,hypotheses,branches
        self.keyed=anchor_mode(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)) == 'decoder'
        self.publish=publish
        packet_words=candidate_wire_numel(capacity,hypotheses,branches, run_config=self.run_config)
        if publish and (not self.keyed or branches != 1):
            raise ValueError('device publication requires flat keyed rows')
        self.rows=int(config_values(self.run_config).get('engine.engine_settings.rows_cap', '128')) if publish else 0
        if publish:
            self.wire_all=torch.empty(packet_words+capacity*self.rows*4+1,
                                      dtype=torch.int64,device=device)
            self.wire,self.check_word=self.wire_all[:packet_words],self.wire_all[-1]
            self.metadata_view=self.wire_all[packet_words:-1].view(capacity,self.rows,4)
        else:
            self.wire=self.wire_all=torch.empty(packet_words,dtype=torch.int64,device=device)
            self.check_word=self.metadata_view=None
        self.buffers=candidate_wire_views(self.wire,capacity,hypotheses,branches, run_config=self.run_config)
        self.control_cpu=torch.empty(capacity,6,dtype=torch.int64,pin_memory=torch.device(device).type=='cuda')
        self.control=torch.empty_like(self.control_cpu,device=device)
        rows=int(config_values(self.run_config).get('engine.engine_settings.rows_cap','128')) if self.keyed else hypotheses
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        width=draft_width(run_config=run_config if run_config is not None else getattr(self, 'run_config', None))+1 if self.keyed else 15
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

    def pack_published(self,sets,*,times,coalesced,gaps):
        """Finish a keyed packet whose device rows and keys were written by the draft pass."""
        if not self.publish:
            raise ValueError('device publication requires flat keyed rows')
        self._pack_control(sets,times=times,coalesced=coalesced,gaps=gaps)
        blocks,row_keys=self.buffers[2],self.buffers[5]
        end=len(sets)
        while end and sets[end-1] is None:
            end-=1
        if end<len(sets):
            blocks[end:].fill_(-1)
            row_keys[end:].fill_(-1)
        for index,candidate in enumerate(sets[:end]):
            if candidate is None:
                blocks[index].fill_(-1)
                row_keys[index].fill_(-1)
        return self.wire

    def published_host(self,host, *, run_config=None):
        """Validate the direct packet and its adjacent check word after one host copy."""
        if not self.publish or host.numel()!=self.wire_all.numel():
            raise ValueError('device publication host packet has the wrong shape')
        if int(host[-1]):
            raise ValueError('Jacobi input must be mask-free')
        packet=host[:self.wire.numel()]
        self.host_metadata=host[self.wire.numel():-1].view(self.capacity,self.rows,4)
        from jetspec.engines.jetspec_plus_plus.wire.protocol import validate_candidate_row_keys_batch
        keys=candidate_wire_views(packet,self.capacity,self.hypotheses,self.branches, run_config=self.run_config)[5]
        validate_candidate_row_keys_batch(keys, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
        return packet

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
        return self.wire


def top_anchor_candidates(scores, rejected_ids, branches):
    """Exclude the rejected draft only where the hypothesis has one."""
    scores = scores.clone()
    valid = rejected_ids >= 0
    scores[valid,rejected_ids[valid]] = float('-inf')
    return scores.topk(branches,dim=-1).indices


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




def resolve_admission_count(configured, *, run_config=None):
    """The segmented chain admits its fixed width; legacy KSET modes are retired."""
    if config_values(run_config).get('engine.engine_settings.rows', '0') != '1':
        return configured
    if kpred_forward_off(run_config=run_config):
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        drafts = draft_width(run_config=run_config)
        return 16 if drafts == 14 else min(16, drafts + 1)
    if config_values(run_config).get('method.method_settings.kset') is not None:
        raise ValueError('retired KSET modes; use the segmented chain or sealed records')
    return configured


def admission_hypotheses(hypotheses, *, block=None, run_config=None):
    """Admission uses the same fixed hypotheses as the segmented chain."""
    if config_values(run_config).get('engine.engine_settings.rows', '0') == '1':
        if kpred_forward_off(run_config=run_config):
            from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
            return list(range(draft_width(run_config=run_config) + 1))
        if config_values(run_config).get('method.method_settings.kset') is not None:
            raise ValueError('retired KSET modes; use the segmented chain or sealed records')
    return list(range(hypotheses))


@torch.inference_mode()
def admission_refinement(forward, *, anchors, hypotheses, sequences, passes, convergence_limit, output_hypotheses=None, fixed_wire_capacity=None, output_indices=None, row_keys=None, pinned_tokens=None, correction_head=None, fixed_passes=False, refresh_swap=None, lab_pins=None, lab_touched=None, final_edit=None, label_passes=None, lab_written=None, lab_written_p=False, lab_pass=None, lab_top=None, run_config=None):
    """Round-zero refinement with the exact metadata, swap staging and tau lab hooks."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    limit = convergence_limit if passes is None else passes
    if limit < 1:
        raise ValueError('passes must be positive')
    count = sequences * len(hypotheses)
    finished = torch.zeros(count, dtype=torch.bool, device=anchors.device)
    passes_run = torch.zeros(count, dtype=torch.long, device=anchors.device)
    record_top1 = config_values(run_config).get('engine.engine_settings.record_ptop1') == '1'
    from jetspec.engines.jetspec_plus_plus.jetspec.profile import capture_candidates
    from jetspec.engines.jetspec_plus_plus.jetspec.profile import draft_candidates
    from jetspec.engines.jetspec_plus_plus.jetspec.profile import draft_token_values
    capture_top = capture_candidates(run_config=run_config)
    final_draft_top16 = None
    final_draft_values = None
    compact_logits = record_top1
    current = final_logits = final_tokens = final_hidden = final_inputs = final_p_top1 = None
    lab_logits = None
    if pinned_tokens is not None:
        pinned_tokens = torch.as_tensor(pinned_tokens, device=anchors.device, dtype=torch.long)
    lazy = anchors.device.type == 'cuda'
    pending = None
    head_logits = None
    for stage_index in range(limit + 0):
        index = stage_index
        result = forward(fill_tokens=current, pass_index=index, finished=finished) if refresh_swap is not None else forward(fill_tokens=current, pass_index=index)
        if lab_touched is not None:
            finished &= ~lab_touched
            lab_touched.zero_()
        logits = result.logits
        if capture_top:
            final_draft_top16 = draft_candidates(logits, final_draft_top16, finished)
        from jetspec.engines.jetspec_plus_plus.draft.method.passes import jacobi_tokens, jacobi_converged, apply_pins
        current = jacobi_tokens(logits, pinned_tokens, None)
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
            if lab_pins is not None:
                pinned.logical_or_(lab_pins >= 0)
            if refresh_swap is not None and index >= 1:
                mask = pins[0] if pins[0].ndim == 2 else torch.arange(current.shape[1], device=current.device)[None] == pins[0][:, None]
                pinned.logical_or_(mask)
            final_draft_values = draft_token_values(logits, current, pinned, final_draft_values, finished)
        fixed = jacobi_converged(current, result.input_tokens)
        if correction_head is not None:
            head_logits = logits.clone() if head_logits is None else torch.where(finished[:, None, None], head_logits, logits)
        p_top1 = top1_probabilities(logits, chunk_rows=16, fast=False) if compact_logits else None
        if final_tokens is None:
            final_logits = None if compact_logits else logits.clone()
            final_tokens = current.clone()
            final_p_top1 = None if p_top1 is None else p_top1.clone()
            if lab_written_p:
                lab_logits = logits.clone()
            if lab_top is not None:
                lab_top(logits, None)
        else:
            if not compact_logits:
                final_logits = torch.where(finished[:, None, None], final_logits, logits)
            else:
                final_p_top1 = torch.where(finished[:, None], final_p_top1, p_top1)
            final_tokens = torch.where(finished[:, None], final_tokens, current)
            if lab_written_p:
                lab_logits = torch.where(finished[:, None, None], lab_logits, logits)
            if lab_top is not None:
                lab_top(logits, finished)
        if index < limit:
            passes_run.masked_fill_(~finished, index + 1)
        finished |= fixed
        if pending is not None and (not fixed_passes):
            event, flag = pending
            event.synchronize()
            if bool(flag[0]):
                break
        if stage_index + 1 >= limit + 0:
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
    host_mark('pr_loop', run_config=run_config)
    if final_edit is not None:
        final_tokens = final_edit(final_tokens)
    lab_masks = None
    if callable(lab_written):
        lab_written = lab_written()
    if lab_written is not None:
        written = lab_written >= 0
        written_p = None
        if lab_written_p:
            written_p = torch.full(written.shape, float('nan'), dtype=torch.float64, device=written.device)
            at = written.nonzero(as_tuple=True)
            if at[0].numel():
                written_p[at] = lab_logits[at].float().softmax(-1).gather(1, lab_written[at][:, None])[:, 0].double()
        lab_masks = (written, written_p)
    predictor_tokens = final_tokens
    if correction_head is not None:
        final_tokens = correction_head.chain(head_logits, anchors[:, 0], pinned=pinned_tokens)
        if config_values(run_config).get('engine.engine_settings.correction_head_predictor', 'wire') != 'bare':
            predictor_tokens = final_tokens
    metadata_device = torch.stack((anchors[:, 0], final_tokens[:, 0], passes_run, finished.long()), dim=1)
    preissue = config_values(run_config).get('engine.engine_settings.tail_preissue', '0') == '1'
    metadata = None if preissue else metadata_device.cpu()
    host_mark('pr_meta', run_config=run_config)
    width = len(hypotheses)
    output_hypotheses = hypotheses if output_hypotheses is None else output_hypotheses
    index_list = [hypotheses.index(k) for k in output_hypotheses] if output_indices is None else list(map(int, output_indices))
    indices = torch.tensor(index_list, dtype=torch.long, device=anchors.device)
    host_indices = torch.tensor(index_list, dtype=torch.long) if kernel_enabled('tail_lean', '1') or preissue else None
    if row_keys is not None and (sequences != 1 or len(row_keys) != len(output_hypotheses)):
        raise ValueError('flat row keys currently require N=1 and one key per published row')
    prepared = []
    for n in range(sequences):
        span = slice(n * width, (n + 1) * width)
        wire = final_tokens[span].index_select(0, indices).clone()
        logits = None if compact_logits else final_logits[span, 1:].index_select(0, indices).clone()
        candidate_wire = None
        if fixed_wire_capacity is not None:
            candidate_wire = wire.new_full((fixed_wire_capacity, wire.shape[-1]), -1)
            if row_keys is not None:
                candidate_wire[:wire.shape[0]] = wire
            elif output_hypotheses:
                candidate_wire[output_hypotheses] = wire
        predictor_kwargs = {}
        if final_p_top1 is not None:
            predictor_kwargs['predictor_p_top1'] = final_p_top1[span].index_select(0, indices).clone()
        padded_keys = None
        if row_keys is not None:
            padded_keys = row_keys.new_full((fixed_wire_capacity, 2), -1)
            padded_keys[:len(row_keys)] = row_keys
        fallback = None
        if fixed_wire_capacity is not None:
            fallback_predictor = {}
            if final_p_top1 is not None:
                fallback_predictor['predictor_p_top1'] = final_p_top1[span][:1].clone()
            fallback = (final_tokens[span][:1].clone(), None if compact_logits else final_logits[span, 1:][:1].clone(), fallback_predictor)
        prepared.append((span, wire, logits, candidate_wire, predictor_kwargs, padded_keys, fallback))
    if metadata is None:
        metadata = metadata_device.cpu()
    output = []
    for span, wire, logits, candidate_wire, predictor_kwargs, padded_keys, fallback in prepared:
        info = metadata[span].index_select(0, indices.cpu() if host_indices is None else host_indices).clone()
        result = PackedBlocks(wire, logits, info, output_hypotheses, candidate_wire=candidate_wire, **predictor_kwargs, run_config=run_config)
        if capture_top:
            result.eval_draft_top16 = tuple((value[span].index_select(0, indices).clone() for value in final_draft_top16))
            result.eval_draft_values = tuple((value[span].index_select(0, indices).clone() for value in final_draft_values))
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
            fallback_blocks = PackedBlocks(fallback[0], fallback[1], metadata[span][:1].clone(), hypotheses[:1], **fallback[2], run_config=run_config)
            if label_passes is not None:
                fallback_blocks.lab_label_passes = int(label_passes)
            if config_values(run_config).get('engine.engine_settings.heads_fast') == '1':
                result._fallback_blocks = fallback_blocks
            else:
                result.fallback = fallback_blocks[0]
        output.append(result)
    host_mark('pr_ret', run_config=run_config)
    return output


@torch.inference_mode()
def exact_adapter_rows(drafter, sequence, ids, run_config=None):
    """E04_LAB_EXACT: the adapter's rows for every slot of each row of ``ids`` [R, B], a block
    [correction, slots] at the correction's position ``length`` (the count of committed taps).

    The round-start staging call's own adapter call (``KthRowsDrafter._stage_packed`` for one
    sequence): the prefix is the target's taps of positions 0..length-1 and the window is the block
    itself, so no slot reads an adapter tail.  Under D1 a window slot reads the prefix and the
    strictly earlier slots (the refresh's rule; the cached C1 call builds that window itself), so a
    slot's row does not depend on the tokens after it.  One call per distinct row.  A cached C1
    head that did not hold this sequence's prefix before the call has its key cleared after it, so
    the next staging call rebuilds from position 0 as it does without the lab (the rule of
    the historical admission seed).  Returns [R, B, H].
    """
    from jetspec.engines.jetspec_plus_plus.jetspec.drafter import ServingCtxKVC1Head
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
                 shared_prefix_projection=True, **({'prefix_key': sequence, 'run_config': run_config} if cached else {}))[0]
            for row in unique]
    if cached and not held:
        head._c1_prefix_key = None
    return torch.stack(rows)[inverse.to(device)]

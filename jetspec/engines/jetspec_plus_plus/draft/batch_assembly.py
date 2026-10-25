"""Deferred C>1 row assembly for the captured batch draft path."""

import torch

from jetspec.engines.jetspec_plus_plus.draft.assembly_batch import BatchRowAssembly, rows_assembly_graph
from jetspec.engines.jetspec_plus_plus.jetspec.live_rows import (
    E04_KSCORE_DEC, E04_ROWRULE_POOL, batch_draft_bucket, batch_draft_graph,
    decoder_blend_kset, rows_rule_cap,
)
from jetspec.engines.jetspec_plus_plus.jetspec.pass_labels import resolve_tau
from jetspec.engines.jetspec_plus_plus.models.decoder_head import DeviceAnchorOffers
from jetspec.engines.jetspec_plus_plus.models.fast import dataset_entry, read_thresholds
from jetspec.engines.jetspec_plus_plus.jetspec.config_values import config_values
from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width

__all__ = ['BatchRowAssemblyError', 'assemble_rows_batch', 'batch_row_assembly', 'finish_rows']


class BatchRowAssemblyError(RuntimeError):
    """The batch cannot proceed without a retired per-request fallback."""


def assemble_rows_batch(drafter, sequences, predictions, defer=False, eager=False, run_config=None):
    """Load and run the captured assembly, returning it and a deferred stage parser.

    Only the deferred captured branch is supported. On CPU BatchRowAssembly runs
    the same body eagerly; CUDA captures it once per assembly shape.
    """
    config = run_config if run_config is not None else drafter.run_config
    values = config_values(config)
    if (values.get('method.method_settings.rowrule') != 'joint'
            or values.get('method.method_settings.kscore') not in ('blend', 'blend_dec')
            or not decoder_blend_kset(run_config=config)
            or values.get('method.method_settings.kpred_frame') != 'wire'
            or values.get('engine.engine_settings.decoder_frame') != 'wire'
            or values.get('method.method_settings.kset_after_miss') != 'zero'
            or values.get('engine.engine_settings.record_ptop1') != '1'
            or values.get('engine.engine_settings.record_qshadow') != '1'
            or values.get('method.method_settings.anchor_cap') != str(E04_ROWRULE_POOL)):
        raise BatchRowAssemblyError('ineligible batched row assembly configuration')
    if not (defer and not eager and batch_draft_graph(run_config=config)
            and rows_assembly_graph(run_config=config)):
        raise BatchRowAssemblyError('batched row assembly requires the deferred captured path')
    if not sequences or len(sequences) != len(predictions):
        raise BatchRowAssemblyError('batched row assembly requires one prediction per request')
    blocks = [sequence.pending_block for sequence in sequences]
    misses = [bool(getattr(sequence, '_jetspec_previous_selection_miss', False))
              for sequence in sequences]
    for sequence, block, prediction in zip(sequences, blocks, predictions):
        if block is None or sequence.staged.predicted_hidden is None:
            raise BatchRowAssemblyError('decoder proposal requires a staged pending block')
        if (prediction is None or not isinstance(prediction.offers, DeviceAnchorOffers)
                or not hasattr(sequence, '_jetspec_seq_id')
                or not hasattr(sequence, '_jetspec_candidate_round')):
            raise BatchRowAssemblyError('batched row assembly is missing device offers or request identity')
    mass_missing = [miss and block.k_probabilities is None for block, miss in zip(blocks, misses)]
    cap = rows_rule_cap(run_config=config)
    assembly = batch_row_assembly(drafter, sequences, predictions, blocks, misses,
                                 cap=cap, run_config=config)
    read = assembly.run()

    def finish():
        requested, keys, extra = read()
        pack_home, mass_home, checks_home, bad_home = extra
        stages = []
        for index, (prediction, miss) in enumerate(zip(predictions, misses)):
            rows = [0] if miss else list(range(draft_width(run_config=config) + 1))
            want = int(requested[index])
            done = finish_rows(
                prediction, (want, keys[index, :min(want, cap)],
                             [pack_home[index], mass_home[index], checks_home[index],
                              bad_home[index:index + 1]]),
                rows=rows, pool=E04_ROWRULE_POOL, cap=cap, after_miss=miss,
                batched=len(sequences) > 1)
            ks, mass_one, live_keys, want, truncated, offer_sizes = done
            stages.append(dict(ks=ks, mass=None if mass_missing[index] else mass_one,
                               keys=live_keys, requested=want, truncated=truncated,
                               sizes=offer_sizes, after_miss=miss))
        return stages

    return assembly, finish


def batch_row_assembly(drafter, sequences, predictions, blocks, misses, *, cap, run_config=None):
    """Load the round into the assembly cached for its shape and calibration."""
    config = run_config if run_config is not None else drafter.run_config
    d = drafter.drafter
    dataset = config_values(config)['benchmark.benchmark_settings.dataset']
    thresholds = read_thresholds(config)
    rule_values = dataset_entry(thresholds['e04_rowrule_joint'], dataset)
    tau_values = [resolve_tau(rule_values['tau'], str(block.kpred_pass)) for block in blocks]
    temperature = float(rule_values['temperature']['value'])
    if any(not 0.0 <= value <= 1.0 for value in tau_values) or not 0.0 < temperature < float('inf'):
        raise ValueError('the joint row rule requires tau in [0, 1] and a finite positive temperature')
    coefficients = dataset_entry(thresholds['e04_kscore_blend'], dataset)
    decoder_features = config_values(config)['method.method_settings.kscore'] == 'blend_dec'
    names = ('a', 'b', 'c') + (E04_KSCORE_DEC if decoder_features else ())
    values = tuple(float(coefficients[name]['value']) for name in names)
    ids, _, pack = predictions[0].offers.device
    count, capacity = len(sequences), 1 << (cap - 1).bit_length()
    drafts = draft_width(run_config=config)
    key = (count, drafts, int(ids.shape[-1]), pack.dtype, cap, int(d.block_size), values,
           temperature, str(ids.device))
    assemblies = drafter.__dict__.setdefault('_batch_row_assemblies', {})
    assembly = assemblies.get(key)
    if assembly is None:
        assembly = assemblies[key] = BatchRowAssembly(
            count=count, padded=batch_draft_bucket(count), drafts=drafts,
            offer_width=int(ids.shape[-1]), pack_dtype=pack.dtype, pool=E04_ROWRULE_POOL,
            H=capacity, cap=cap, capacity=capacity, block_size=d.block_size,
            coefficients=values, temperature=temperature, decoder_features=decoder_features,
            device=ids.device, graphs=True)
    assembly.load(predictions, blocks, misses, tau_values, sequences, seed=d.seed, run_config=config)
    return assembly


def finish_rows(prediction, homed, *, rows, pool, cap, after_miss, device_checks=False,
                batched=False):
    """Validate a packed host copy and return its selected rows and offer summary.

    This was a static method: it retains its prediction-first signature.
    A required full-row recompute now raises BatchRowAssemblyError.
    """
    requested, keys, extra = homed
    summary, mass_home = extra[:2]
    if device_checks:
        flags = extra[2]
        if int(flags[0]):
            raise ValueError('fast LAC probabilities and q values must be finite')
        if int(flags[1]):
            raise BatchRowAssemblyError('anchor offers require the retired full-row recompute')
        if int(flags[2]):
            raise ValueError('E04_KSCORE=blend requires a draft at every decoder row 0..13')
        if int(flags[3]):
            raise ValueError('E04_KSCORE=blend_dec requires an anchor offer at every decoder row 0..13')
        if int(flags[4]):
            raise ValueError('E04_KSCORE=blend produced a non-finite k score')
        if int(flags[5]):
            raise ValueError('E04_ROWRULE=joint requires 32 anchor offers at every scored decoder row')
        if int(flags[6]):
            raise ValueError('E04_ROWRULE=joint produced a non-finite pair score')
        summary = summary.tolist()
    else:
        check_home, score_home = extra[2:]
        if batched:
            if not bool(summary[3].all()):
                raise ValueError('fast LAC probabilities and q values must be finite')
            if bool(summary[4].any()):
                raise BatchRowAssemblyError('anchor offers require the retired full-row recompute')
            missing_pool = bool((summary[0, rows] < pool).any())
            summary = summary.tolist()
        else:
            summary = summary.tolist()
            if not all(summary[3]):
                raise ValueError('fast LAC probabilities and q values must be finite')
            if any(summary[4]):
                raise BatchRowAssemblyError('anchor offers require the retired full-row recompute')
            missing_pool = any(summary[0][k] < pool for k in rows)
        prediction.offers.set_summary(summary[0], summary[1], sum(summary[2]))
        if not after_miss:
            if int(check_home[0]):
                raise ValueError('E04_KSCORE=blend requires a draft at every decoder row 0..13')
            if int(check_home[1]):
                raise ValueError('E04_KSCORE=blend_dec requires an anchor offer at every decoder row 0..13')
            if not bool(torch.isfinite(mass_home).all()):
                raise ValueError('E04_KSCORE=blend produced a non-finite k score')
        if missing_pool:
            raise ValueError('E04_ROWRULE=joint requires 32 anchor offers at every scored decoder row')
        if int(score_home[0]):
            raise ValueError('E04_ROWRULE=joint produced a non-finite pair score')
    if device_checks:
        prediction.offers.set_summary(summary[0], summary[1], sum(summary[2]))
    live_keys = keys.clone()
    ks = (torch.unique(live_keys[:, 0], sorted=True).tolist() if batched else
          sorted({int(k) for k in live_keys[:, 0].tolist()}))
    return ks, mass_home, live_keys, requested, max(0, requested - cap), prediction.offers.set_sizes

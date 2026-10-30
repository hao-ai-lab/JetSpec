"""Packed Jacobi passes for the batched draft path.

Ported from live_rows.packed_refinement. Rows freeze at
their first fixed point, including their confidence and predictor inputs.
"""
import torch

from jetspec.engines.jetspec_plus_plus.draft.method.passes import jacobi_converged, jacobi_tokens
from jetspec.engines.jetspec_plus_plus.runtime.config_values import config_values
from jetspec.engines.jetspec_plus_plus.draft.drafter import DraftSuperseded
from jetspec.engines.jetspec_plus_plus.draft.live_rows import PackedBlocks, top1_probabilities

__all__ = ['packed_refinement', 'UnsupportedBatchRefinement']


class UnsupportedBatchRefinement(ValueError):
    """An option requires a refinement path outside the batched recipe."""


@torch.inference_mode()
def packed_refinement(forward, *, anchors, hypotheses, sequences, passes, convergence_limit,
                      output_sequences=None, output_hypotheses=None, abort_if=None, fixed_wire_capacity=None,
                      predictor=None, output_indices=None, row_keys=None, pinned_tokens=None,
                      posterior=False, correction_head=None, late=None, truncate_if=None,
                      pinned_second=None, publish=None, active_rows=None, after_passes=None,
                      fast_top1=False, fixed_passes=False, device_metadata=False,
                      refresh_swap=None, tail_passes=0, lab_pins=None, lab_touched=None, final_edit=None,
                      label_passes=None, lab_written=None, lab_written_p=False, lab_pass=None, lab_top=None,
                      operation_schedule=None, refresh_only=None, run_config=None):
    """Run packed rows, then return one ``PackedBlocks`` per output sequence.

    ``passes`` counts forwards (J0 is one); None uses ``convergence_limit``.
    ``active_rows`` covers all sequences * hypotheses, including padded rows.
    The historical parameters remain accepted, but excluded recipe options
    raise ``UnsupportedBatchRefinement`` rather than selecting a retired path.
    CPU fast_top1 uses the exact torch reduction; the CUDA fast reduction has
    the historical float32 summation-order tolerance.
    """
    excluded = dict(late=late, truncate_if=truncate_if, publish=publish,
                    refresh_swap=refresh_swap, lab_pins=lab_pins, lab_touched=lab_touched,
                    final_edit=final_edit, label_passes=label_passes, lab_written=lab_written,
                    lab_pass=lab_pass, lab_top=lab_top, operation_schedule=operation_schedule,
                    refresh_only=refresh_only)
    unsupported = [name for name, value in excluded.items() if value is not None]
    unsupported += [name for name, value in dict(posterior=posterior, tail_passes=tail_passes,
                                                lab_written_p=lab_written_p).items() if value]
    settings = config_values(run_config)
    if settings.get('engine.engine_settings.pass_loop_device', '0') == '1':
        unsupported.append('pass_loop_device')
    if unsupported:
        raise UnsupportedBatchRefinement('unsupported batch refinement options: ' + ', '.join(unsupported))

    limit = convergence_limit if passes is None else passes
    if limit < 1:
        raise ValueError('passes must be positive')
    count, width = sequences * len(hypotheses), len(hypotheses)
    if active_rows is not None:
        if active_rows.shape != (count,) or active_rows.dtype != torch.bool:
            raise ValueError('active_rows must be one bool per fixed-cap row')
        finished = ~active_rows.clone()
    else:
        finished = torch.zeros(count, dtype=torch.bool, device=anchors.device)
    passes_run = torch.zeros(count, dtype=torch.long, device=anchors.device)
    compact_logits = predictor is not None or settings.get('engine.engine_settings.record_ptop1') == '1'
    if pinned_tokens is not None:
        pinned_tokens = torch.as_tensor(pinned_tokens, device=anchors.device, dtype=torch.long)
    current = final_tokens = final_logits = final_p_top1 = None
    final_hidden = final_inputs = head_logits = None
    pending = None
    for index in range(limit):
        if abort_if is not None and abort_if():
            raise DraftSuperseded('newer committed verifier state is ready')
        result = forward(fill_tokens=current, pass_index=index)
        logits = result.logits
        current = jacobi_tokens(logits, pinned_tokens, pinned_second)
        fixed = jacobi_converged(current, result.input_tokens)
        if correction_head is not None:
            head_logits = (logits.clone() if head_logits is None else
                           torch.where(finished[:, None, None], head_logits, logits))
        p_top1 = top1_probabilities(logits, chunk_rows=16, fast=fast_top1) if compact_logits else None
        if predictor is not None and result.hidden is None:
            raise RuntimeError('predictor serving requires final-pass hidden states')
        if final_tokens is None:
            final_tokens = current.clone()
            final_logits = None if compact_logits else logits.clone()
            final_p_top1 = None if p_top1 is None else p_top1.clone()
            if predictor is not None:
                final_hidden, final_inputs = result.hidden.clone(), result.input_tokens.clone()
        else:
            final_tokens = torch.where(finished[:, None], final_tokens, current)
            if compact_logits:
                final_p_top1 = torch.where(finished[:, None], final_p_top1, p_top1)
            else:
                final_logits = torch.where(finished[:, None, None], final_logits, logits)
            if predictor is not None:
                final_hidden = torch.where(finished[:, None, None], final_hidden, result.hidden)
                final_inputs = torch.where(finished[:, None], final_inputs, result.input_tokens)
        passes_run.masked_fill_(~finished, index + 1)
        finished |= fixed
        if pending is not None and not fixed_passes:
            event, flag = pending
            event.synchronize()
            if bool(flag[0]):
                break
        if index + 1 == limit:
            break
        if fixed_passes:
            continue
        if anchors.device.type != 'cuda':
            if bool(finished.all()):
                break
        else:
            # Read all-converged one pass late, as in OLD, without a device
            # synchronization between forward launches. Finished rows stay frozen.
            flag = torch.zeros(1, dtype=torch.bool, pin_memory=True)
            flag.copy_(finished.all().view(1), non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            pending = event, flag

    if active_rows is not None:
        final_tokens.masked_fill_(~active_rows[:, None], -1)
    if after_passes is not None:
        after_passes()
    predictor_tokens = final_tokens
    if correction_head is not None:
        final_tokens = correction_head.chain(head_logits, anchors[:, 0], pinned=pinned_tokens)
        if settings.get('engine.engine_settings.correction_head_predictor', 'wire') != 'bare':
            predictor_tokens = final_tokens
        if active_rows is not None:
            final_tokens.masked_fill_(~active_rows[:, None], -1)
    metadata_device = torch.stack((anchors[:, 0], final_tokens[:, 0], passes_run, finished.long()), dim=1)
    metadata = metadata_device if device_metadata else metadata_device.cpu()
    output_sequences = sequences if output_sequences is None else output_sequences
    output_hypotheses = hypotheses if output_hypotheses is None else output_hypotheses
    index_list = ([hypotheses.index(k) for k in output_hypotheses]
                  if output_indices is None else list(map(int, output_indices)))
    indices = torch.tensor(index_list, dtype=torch.long, device=anchors.device)
    host_indices = indices if device_metadata else torch.tensor(index_list, dtype=torch.long)
    if row_keys is not None and (sequences != 1 or len(row_keys) != len(output_hypotheses)):
        raise ValueError('flat row keys currently require N=1 and one key per published row')
    output = []
    for n in range(output_sequences):
        span = slice(n * width, (n + 1) * width)
        wire = final_tokens[span].index_select(0, indices).clone()
        logits = None if compact_logits else final_logits[span, 1:].index_select(0, indices).clone()
        candidate_wire = padded_keys = None
        if fixed_wire_capacity is not None:
            candidate_wire = wire.new_full((fixed_wire_capacity, wire.shape[-1]), -1)
            if row_keys is not None:
                candidate_wire[:wire.shape[0]] = wire
            elif output_hypotheses:
                candidate_wire[output_hypotheses] = wire
        if row_keys is not None:
            padded_keys = row_keys.new_full((fixed_wire_capacity, 2), -1)
            padded_keys[:len(row_keys)] = row_keys
        predictor_kwargs = {}
        if final_p_top1 is not None:
            predictor_kwargs['predictor_p_top1'] = final_p_top1[span].index_select(0, indices).clone()
        if predictor is not None:
            predictor_kwargs.update(
                predictor=predictor,
                predictor_hidden=final_hidden[span].index_select(0, indices).clone(),
                predictor_inputs=final_inputs[span].index_select(0, indices).clone(),
                predictor_outputs=predictor_tokens[span].index_select(0, indices).clone())
        info = metadata[span].index_select(0, host_indices).clone()
        output.append(PackedBlocks(wire, logits, info, output_hypotheses,
                                   candidate_wire=candidate_wire, row_keys=padded_keys,
                                   **predictor_kwargs, run_config=run_config))
    return output

"""Device-side, offer-identical selection helpers for the optional heads path."""
from __future__ import annotations

from collections.abc import Mapping
import torch

from jetspec.engines.jetspec_plus_plus.models.lac import KeyedRows
from jetspec.engines.jetspec_plus_plus.models.lac import keep_device
from jetspec.engines.jetspec_plus_plus.models.lac import lac_candidates
from jetspec.engines.jetspec_plus_plus.models.lac import row_rank_parameters


def read_thresholds(model):
    """Return the calibrated thresholds carried by the selected model entry."""
    if hasattr(model, 'model'):
        model = model.model
    if model is None or 'thresholds' not in model:
        raise ValueError('model entry requires inline thresholds')
    def plain(value):
        if isinstance(value, Mapping):
            return {key: plain(child) for key, child in value.items()}
        if isinstance(value, tuple):
            return [plain(child) for child in value]
        return value

    thresholds = model['thresholds']
    return thresholds if isinstance(thresholds, dict) else plain(thresholds)


from jetspec.engines.jetspec_plus_plus.draft.method.calibration import dataset_entry


def fast_lac_candidates(probabilities, q_hats, *, cap, rejected=None, also=None):
    """Select exact stable LAC offers with one host read for sizes and checks.

    ``also`` is an optional long vector carried home in that same read; its
    host list is returned fourth.
    """
    probabilities = keep_device(probabilities).double()
    q_hats = torch.as_tensor(q_hats, device=probabilities.device,
                             dtype=probabilities.dtype).reshape(-1)
    if probabilities.ndim != 2 or q_hats.shape != probabilities.shape[:1]:
        raise ValueError('fast LAC expects [rows,vocab] probabilities and one q per row')
    checks = [torch.isfinite(probabilities).all(),
              ((q_hats >= 0) & (q_hats <= 1)).all()]
    cap = int(cap)
    if cap < 1:
        raise ValueError('fast LAC cap must be positive')
    keep = probabilities >= (1.0 - q_hats[:, None])
    if rejected is not None:
        rejected = torch.as_tensor(rejected, device=probabilities.device,
                                   dtype=torch.long).reshape(-1)
        if rejected.shape != probabilities.shape[:1]:
            raise ValueError('fast LAC rejected ids must have one value per row')
        checks.append((rejected < probabilities.shape[1]).all())
        if probabilities.shape[1]:
            index = rejected.clamp(0, probabilities.shape[1] - 1)[:, None]
            keep.scatter_(1, index, keep.gather(1, index) & (rejected < 0)[:, None])
    masked = probabilities.masked_fill(~keep, float('-inf'))
    width = min(cap, probabilities.shape[1])
    # One stable descending sort over every row is the reference order itself
    # (value descending, id ascending among ties), so a tie across the cap
    # never needs the per-row host fallback and its stream drains.
    top_ids = torch.argsort(masked, dim=-1, descending=True, stable=True)[:, :width]
    top_values = masked.gather(1, top_ids)
    counts = keep.sum(-1)
    crosses = torch.zeros_like(counts, dtype=torch.bool)
    rows = probabilities.shape[0]
    # One host read carries the checks, sizes and the candidate ids; the values
    # follow in a second copy.  Row slices are then host views, so keyed-row
    # assembly no longer copies each offer to the host on its own.
    also = counts.new_empty(0) if also is None else also.reshape(-1).long()
    metadata = torch.cat((torch.stack(checks).long(), counts, crosses.long(),
                          counts.new_empty(0) if rejected is None else rejected,
                          top_ids.reshape(-1), also,
                          top_values.reshape(-1).contiguous().view(torch.int64))).tolist()
    if not all(metadata[:2]):
        raise ValueError('fast LAC probabilities and q values must be finite')
    if rejected is not None and not metadata[2]:
        raise ValueError('fast LAC rejected id is outside the probability vector')
    offset = len(checks)
    host_counts = metadata[offset:offset + rows]
    host_crosses = metadata[offset + rows:offset + 2 * rows]
    offset += 2 * rows
    host_rejected = metadata[offset:offset + (0 if rejected is None else rows)]
    offset += len(host_rejected)
    host_ids = torch.tensor(metadata[offset:offset + rows * width], dtype=torch.long,
                            device='cpu').reshape(rows, width)
    offset += rows * width
    host_also = metadata[offset:offset + also.numel()]
    offset += also.numel()
    host_values = torch.tensor(metadata[offset:], dtype=torch.int64,
                               device='cpu').view(torch.float64).reshape(rows, width)
    ids, selected = [], []
    for row, (count, fallback) in enumerate(zip(host_counts, host_crosses)):
        if fallback:
            excluded = (() if rejected is None or host_rejected[row] < 0
                        else {host_rejected[row]})
            row_ids, row_probabilities = lac_candidates(
                probabilities[row], q_hat=float(q_hats[row]), cap=cap,
                excluded=excluded)
        else:
            size = min(int(count), cap)
            row_ids, row_probabilities = host_ids[row, :size], host_values[row, :size]
        ids.append(row_ids)
        selected.append(row_probabilities)
    if also.numel():
        return ids, selected, sum(host_crosses), host_also
    return ids, selected, sum(host_crosses)

def fast_keyed_rows(k_ids, k_probabilities, anchor_ids, anchor_probabilities, *, cap, run_config=None,
                    rank_parameters=None):
    """Vectorize ragged joint score construction while retaining stable order."""
    k_ids = torch.as_tensor(k_ids, dtype=torch.long, device='cpu')
    k_probabilities = torch.as_tensor(
        k_probabilities, dtype=torch.float64, device=k_ids.device)
    if k_ids.ndim != 1 or k_probabilities.shape != k_ids.shape:
        raise ValueError('k ids and probabilities must be equal one-dimensional vectors')
    id_rows = [torch.as_tensor(anchor_ids[int(k)], dtype=torch.long,
                               device=k_ids.device) for k in k_ids]
    probability_rows = [torch.as_tensor(anchor_probabilities[int(k)],
                                        dtype=torch.float64, device=k_ids.device)
                        for k in k_ids]
    if any(ids.ndim != 1 or probabilities.shape != ids.shape
           for ids, probabilities in zip(id_rows, probability_rows)):
        raise ValueError('each anchor id/probability offer must be one-dimensional')
    sizes = torch.tensor([ids.numel() for ids in id_rows], device=k_ids.device,
                         dtype=torch.long)
    requested = int(sizes.sum().item())
    if requested:
        repeated_k = torch.repeat_interleave(k_ids, sizes)
        repeated_pk = torch.repeat_interleave(k_probabilities, sizes)
        anchors = torch.cat(id_rows)
        a, b, length_weight = (row_rank_parameters(run_config=run_config)
                              if rank_parameters is None else rank_parameters)
        anchor_scores = torch.cat(probability_rows)
        repeated_pk = repeated_pk if a == 1 else repeated_pk.pow(a)
        anchor_scores = anchor_scores if b == 1 else anchor_scores.pow(b)
        if length_weight:
            repeated_pk = repeated_pk * (repeated_k + 1)
        scores = repeated_pk * anchor_scores
        keys = torch.stack((repeated_k, anchors), dim=1)
        order = torch.argsort(scores, descending=True, stable=True)[:int(cap)]
        keys, scores = keys.index_select(0, order), scores.index_select(0, order)
    else:
        keys = torch.empty((0, 2), dtype=torch.long, device=k_ids.device)
        scores = torch.empty((0,), dtype=torch.float64, device=k_ids.device)
    return KeyedRows(keys=keys, joint_probabilities=scores, requested=requested,
                     truncated=max(0, requested - int(cap)))

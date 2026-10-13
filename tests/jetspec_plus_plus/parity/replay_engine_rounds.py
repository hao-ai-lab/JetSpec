"""Tier 2b: replay recorded decoder windows, BLEND mass and row selection.

This does not cover J/R passes, swaps or pins; tier 2a (compare_rounds.py)
covers those through free-running fp32 T0 method identity. Admissions lack
decoder inputs and are explicitly skipped; redrafts check their exact key.
Decoder inputs are raw BF16 bytes indexed by JSONL, not torch pickles. Exact
field equality is required. Capture stores BF16 even for fp32 reads, so this
check also exposes capture quantization or decoder-kernel differences.

Ported from the JetSpec++ reference implementation.
Mirrors its decoder-window capture records and causal mask.
"""
import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from jetspec.methods.jetspec_plus_plus.artifacts import dataset_entry, read_thresholds
from jetspec.methods.jetspec_plus_plus.method.kmass import blend_mass
from jetspec.methods.jetspec_plus_plus.method.rows import DeviceOffers, joint_rule, select_rows
from jetspec.methods.jetspec_plus_plus.method.decoder.window import window_mask
from jetspec.methods.jetspec_plus_plus.models.decoder import AnchorDecoder, _WindowCache
from jetspec.methods.jetspec_plus_plus.round import resolve_tau


def jsonl(path, schema):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if any(row.get('schema') != schema for row in rows):
        raise ValueError(f'unknown sidecar schema; expected {schema}')
    return rows


def read_tensor(stream, offset, shape):
    if (type(offset) is not int or offset < 0 or offset % 2 or not isinstance(shape, list)
            or not shape or any(type(n) is not int or n < 0 for n in shape)):
        raise ValueError('invalid BF16 tensor offset or shape')
    stream.seek(offset)
    count = math.prod(shape)
    content = bytearray(stream.read(2 * count))
    if len(content) != 2 * count:
        raise ValueError('truncated decoder-input binary')
    return torch.frombuffer(content, dtype=torch.bfloat16).reshape(shape) if count else torch.empty(shape, dtype=torch.bfloat16)


@torch.inference_mode()
def decoder_logits(decoder, window, taps, length):
    """True deepest-tap prefix followed by the captured stale+staged causal window."""
    parameter = next(decoder.parameters())
    # _prefix reads only tap 4; the four unrecorded taps are never consumed here.
    sequence = SimpleNamespace(taps=taps[:, None, :].expand(-1, 5, -1))
    prefix = decoder._prefix(sequence, length)
    try:
        hidden = window.to(parameter)[None]
        positions = torch.arange(length, length + window.shape[0], device=parameter.device)[None]
    # Reference chain-window staging: positions length+lane, all true prefix
        # keys followed by the causal window. No refresh-row geometry is used.
        mask = window_mask([length + 4], length, parameter.dtype, size=window.shape[0]).to(parameter.device)
        _, logits = decoder(hidden, mask=mask, positions=positions,
                            prefix=_WindowCache(prefix.kv, append=False))
        return logits[0, 4:].float()
    finally:
        decoder.drop_sequence(sequence)


def replay_features(record, logits, thresholds, dataset):
    """Recompute decoder evidence, then use the frozen reference's quantized row rule."""
    drafts = torch.tensor(record['draft_token_ids'], device=logits.device, dtype=torch.long)
    rejected = torch.cat((drafts[1:], drafts.new_full((1,), -1)))
    if record['after_miss']:
        rejected[0] = -1
    valid = rejected >= 0
    q = logits.softmax(-1).gather(1, rejected.clamp_min(0)[:, None])[:, 0]
    filtered = logits.clone()
    filtered[valid, rejected[valid]] = float('-inf')
    probabilities = filtered.softmax(-1)
    # GraphedSelect's rank excludes rejected IDs even when their probability is zero.
    probabilities[valid, rejected[valid]] = -1
    ids = probabilities.argsort(descending=True, stable=True)[:, :32]
    probabilities = probabilities.gather(1, ids).double()
    argmax = logits.argmax(-1)
    coefficients = dataset_entry(thresholds['e04_kscore_blend'], dataset)
    rule = dataset_entry(thresholds['e04_rowrule_joint'], dataset)
    p = torch.tensor(record['p_top1'][1:], device=logits.device, dtype=torch.float64).clamp(1e-6, 1-1e-6)
    decoder = torch.where(valid, q, 0.0)[:-1].double().clamp(1e-6, 1-1e-6)
    a, b, c = (float(coefficients[name]['value']) for name in ('a', 'b', 'c'))
    z = a * (p.log() - (-p).log1p()) + b * (decoder.log() - (-decoder).log1p()) + c
    if 'argmax' in coefficients:
        g, t, pos = (float(coefficients[name]['value']) for name in ('argmax', 'top', 'pos'))
        eq = (argmax[:-1] == rejected[:-1]).double()
        top = probabilities[:-1].amax(-1).clamp(1e-6, 1-1e-6)
        z = z + g * eq + t * (top.log() - (-top).log1p()) + pos * torch.arange(
            1, len(p) + 1, device=logits.device, dtype=torch.float64) / 14.0
    mass = blend_mass(z)
    prediction = torch.cat((mass.new_zeros(1), mass))
    if record['after_miss']:
        mass = torch.zeros_like(mass)
        mass[0] = 1.0
    offers = DeviceOffers(ids, probabilities, torch.ones_like(ids, dtype=torch.bool),
                          torch.arange(len(ids), device=logits.device))
    # Match wire_records.replay_record's narrow rule (no additional policy floor).
    score, passing = joint_rule(resolve_tau(rule['tau'], record['pass_label']),
                                rule['temperature']['value'], floor=0)(mass, offers)
    cap = record['row_cap']
    requested, keys = select_rows(score, passing, offers, cap, cap)
    keys = keys[:min(int(requested), cap)].cpu().tolist() or [[0, record['fallback_anchor']]]
    q_values = q.cpu().tolist()
    return dict(q_shadow=[v if token >= 0 else None for v, token in zip(q_values, rejected.cpu().tolist())],
                shadow_argmax=argmax.cpu().tolist(),
                offers=dict(d_rej=rejected.cpu().tolist(), top_ids=ids.cpu().tolist(),
                            top_probabilities=probabilities.cpu().tolist(),
                            pre_exclusion_logsumexp=logits.logsumexp(-1).cpu().tolist()),
                p_pred=prediction.cpu().tolist(), p_wire=mass.cpu().tolist(), published_keys=keys,
                anchor_hit=[record['k_true'], record['anchor_true']] in keys,
                outcome_miss=[record['k_true'], record['anchor_true']] not in keys)


def replay(candidates, index, data, *, decoder, thresholds, dataset):
    records = jsonl(candidates, 'jetspec.eval_round_candidates/1')
    indices = jsonl(index, 'jetspec.eval_decoder_inputs/1')
    proposals = {}
    skipped = {'admission': 0}
    fields = {}
    def compare(record, actual):
        for key, value in actual.items():
            if key not in record or record[key] is None:
                continue
            field = fields.setdefault(key, dict(rounds_compared=0, rounds_equal=0, first_difference=None))
            field['rounds_compared'] += 1
            if value == record[key]:
                field['rounds_equal'] += 1
            elif field['first_difference'] is None:
                field['first_difference'] = dict(seq_id=record['seq_id'], candidate_round=record['candidate_round'],
                                                  candidate=value, reference=record[key])
    for record in records:
        kind = record['record_kind']
        if kind == 'admission':
            skipped['admission'] += 1
        elif kind == 'redraft':
            compare(record, {'published_keys': [[record['k_true'], record['anchor_true']]]})
        elif kind == 'proposal':
            for name in ('draft_token_ids', 'p_top1', 'offers', 'q_shadow', 'shadow_argmax',
                         'published_keys', 'pass_label', 'k_true', 'anchor_true', 'row_cap'):
                if name not in record or record[name] is None:
                    raise ValueError(f'proposal lacks required {name}')
            key = record['seq_id'], record['candidate_round']
            if key in proposals:
                raise ValueError('duplicate proposal round')
            proposals[key] = record
        else:
            raise ValueError(f'unknown record kind {kind}')
    indexed = [(row['seq_id'], row['candidate_round']) for row in indices]
    if len(set(indexed)) != len(indexed):
        raise ValueError('duplicate decoder-input round')
    if proposals.keys() - set(indexed):
        raise ValueError('missing decoder inputs for a recorded proposal')
    prefixes = {}
    with Path(data).open('rb') as stream:
        for row in indices:
            if row['dtype'] != 'bfloat16' or row['taps_layer'] != 4:
                raise ValueError('decoder inputs must contain raw BF16 layer-4 taps')
            window = read_tensor(stream, row['window_offset'], row['window_shape'])
            taps = read_tensor(stream, row['taps_offset'], row['taps_shape'])
            seq = row['seq_id']
            previous = prefixes.get(seq, taps[:0])
            if (row['taps_from'] != len(previous) or row['taps_to'] != len(previous) + len(taps)
                    or taps.ndim != 2 or window.shape != (20, decoder.hidden_size)
                    or taps.shape[1] != decoder.hidden_size):
                raise ValueError('decoder inputs have invalid window geometry or noncontiguous taps')
            prefixes[seq] = torch.cat((previous, taps))
            if not 0 <= row['dec_len'] <= len(prefixes[seq]):
                raise ValueError('decoder length exceeds recorded prefix taps')
            key = seq, row['candidate_round']
            if key in proposals:
                record = proposals[key]
                if len(record['p_top1']) != 16 or len(record['draft_token_ids']) != 16:
                    raise ValueError('replay requires B16 candidates')
                logits = decoder_logits(decoder, window, prefixes[seq], row['dec_len'])
                compare(record, replay_features(record, logits, thresholds, dataset))
    if not fields or not proposals:
        raise ValueError('replay requires at least one captured proposal')
    return dict(schema='jetspec.decoder_row_replay/1', fields=fields, skipped=skipped,
                proposals_compared=len(proposals),
                equal=all(f['rounds_equal'] == f['rounds_compared'] for f in fields.values()),
                limitation='decoder/BLEND/row selection only; J/R passes, swaps and pins are covered by tier 2a')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--decoder-index', type=Path)
    parser.add_argument('--decoder-data', type=Path)
    parser.add_argument('--target', required=True, help='local target directory matching the capture')
    parser.add_argument('--heads-bundle', type=Path, required=True)
    parser.add_argument('--dataset', default='gsm8k')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--dtype', choices=['fp32', 'bf16'], default='bf16')
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args(argv)
    from transformers import Qwen3ForCausalLM
    target, info = Qwen3ForCausalLM.from_pretrained(args.target, local_files_only=True,
        torch_dtype=torch.float32 if args.dtype == 'fp32' else torch.bfloat16,
        attn_implementation='sdpa', output_loading_info=True)
    if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'target checkpoint does not match its model: {info}')
    target = target.to(args.device).eval()
    decoder = AnchorDecoder.from_target(target)
    base = str(args.candidates).removesuffix('.candidates.jsonl')
    result = replay(args.candidates, args.decoder_index or base + '.decoder_inputs.jsonl',
                    args.decoder_data or base + '.decoder_inputs.bin', decoder=decoder,
                    thresholds=read_thresholds(args.heads_bundle), dataset=args.dataset)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + '\n')
    return int(not result['equal'])


if __name__ == '__main__':
    raise SystemExit(main())

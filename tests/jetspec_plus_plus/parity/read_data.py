"""Shared ordered-prompt and emitted-round handling for torch and engine reads."""
import hashlib
import json


def token_sha256(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode('ascii')).hexdigest()


def read_identity(raw):
    if 'manifest' in raw:
        identity = dict(raw['manifest'])
        if 'prompts_sha256' in identity:
            identity['prompts_file_sha256'] = identity.pop('prompts_sha256')
        return identity
    manifest = raw.get('run_manifest', {})
    values, env = manifest.get('values', {}), manifest.get('env', {})
    args = values.get('args', {})
    # The reference run_manifest hashes selected input arrays; its prompts file pin
    # hashes bytes, matching the torch reader's original prompts_sha256 field.
    files = manifest.get('files', [])
    prompt_file = next((row['sha256'] for row in files if row['role'] == 'prompts'), None)
    identity = dict(prompt_inputs_sha256=values.get('prompts_sha256'),
                    prompts_file_sha256=prompt_file,
                    max_new_tokens=raw.get('max_new'), temperature=args.get('temp'),
                    dtype=raw.get('eval', {}).get('dtype', env.get('JETSPEC_TARGET_DTYPE')))
    return {key: value for key, value in identity.items() if value is not None}


def paired_outputs(candidate, reference):
    a, b = candidate['outputs'], reference['outputs']
    if not a or len(a) != len(b):
        raise ValueError('reads need the same nonempty ordered prompt population')
    am, bm = read_identity(candidate), read_identity(reference)
    keys = ['max_new_tokens', 'temperature', 'dtype']
    if 'prompt_inputs_sha256' in am and 'prompt_inputs_sha256' in bm:
        keys.insert(0, 'prompt_inputs_sha256')
    else:
        keys[:0] = ['prompts_file_sha256', 'selected_prompts_sha256']
    for key in keys:
        if key in am and key in bm and am[key] != bm[key]:
            raise ValueError(f'reads do not have matched {key}')
    pairs = []
    for i, (one, other) in enumerate(zip(a, b)):
        if one.get('refused') or other.get('refused'):
            raise ValueError('refused prompts cannot be silently dropped from paired reads')
        if ('prompt_index' in one and 'prompt_index' in other and
                one['prompt_index'] != other['prompt_index']):
            raise ValueError('reads have different ordered prompt indices')
        for row in (one, other):
            if not isinstance(row['token_ids'], list) or any(type(t) is not int or t < 0 for t in row['token_ids']):
                raise ValueError('token_ids must be nonnegative integer arrays')
        pairs.append((one.get('prompt_index', i), one, other))
    return pairs


def compare_tokens(candidate, reference):
    prompts = []
    for index, one, other in paired_outputs(candidate, reference):
        a, b = one['token_ids'], other['token_ids']
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if first is None and len(a) != len(b):
            first = min(len(a), len(b))
        prompts.append(dict(prompt_index=index, candidate_sha256=token_sha256(a),
                            reference_sha256=token_sha256(b), equal=first is None,
                            first_differing_index=first,
                            candidate_token=None if first is None or first >= len(a) else a[first],
                            reference_token=None if first is None or first >= len(b) else b[first]))
    diverged = sum(not row['equal'] for row in prompts)
    return dict(equal=diverged == 0, prompts=prompts, diverged_prompts=diverged,
                diverged_share=diverged / len(prompts),
                alignment='ordered outputs; prompt indices and selected bank hashes checked when both recorded')


def emitted_rounds(raw):
    """Apply the reference segment reducer's terminal credit convention, without clocks."""
    outputs = raw['outputs']
    grouped = [[] for _ in outputs]
    records = raw.get('metrics', {}).get('jetspec_round_records')
    if records is not None:
        indices = [record['prompt_index'] for record in records]
        if (any(type(i) is not int or i < 0 or i >= len(outputs) for i in indices)
                or indices != sorted(indices)):
            raise ValueError('round records are not aligned to ordered prompt positions')
        for record in records:
            grouped[record['prompt_index']].append(record)
    else:
        for i, output in enumerate(outputs):
            if 'accept_lengths' not in output:
                raise ValueError('read lacks per-round accepted lengths')
            labels = output.get('pass_labels', [])
            passes = output.get('passes_run', [])
            for r, accepted in enumerate(output['accept_lengths']):
                row = dict(accepted_len=accepted)
                if r < len(labels):
                    row['pass_label'] = labels[r]
                elif r < len(passes):
                    row['pass_label'] = 'J0d0' if passes[r] == 1 else f'J{passes[r]-1}'
                grouped[i].append(row)
    result = []
    for output, records in zip(outputs, grouped):
        remaining = max(0, len(output['token_ids']) - 1)
        rounds = []
        for record in records:
            accepted = record['accepted_len']
            if type(accepted) is not int or accepted < 1:
                raise ValueError('accepted_len must be a positive integer')
            if remaining:
                length = min(accepted, remaining)
                rounds.append(dict(record, accepted_len=length))
                remaining -= length
        if remaining:
            raise ValueError('round records do not cover the generated output')
        result.append(rounds)
    return result


def paper_credit(raw):
    from jetspec.methods.jetspec_plus_plus.reader import reduce_read
    rounds = emitted_rounds(raw)
    outputs = [dict(row, accept_lengths=[r['accepted_len'] for r in records])
               for row, records in zip(raw['outputs'], rounds)]
    eos = raw.get('manifest', {}).get('eos_token_ids',
          raw.get('eval', {}).get('eos_token_ids', [151643, 151645]))
    cap = raw.get('manifest', {}).get('max_new_tokens', raw.get('max_new', 2048))
    return reduce_read({'outputs': outputs}, eos_token_ids=eos, max_new_tokens=cap)

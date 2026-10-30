"""Reduce a matched eval/eager pair without transferring device artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from statistics import mean, median
import sys
import tarfile

ROOT = (Path(__file__).resolve().parent.parent.parent.parent / '.')
sys.path[:0] = [str(ROOT)]
from jetspec.engines.jetspec_plus_plus.analysis import paper_metrics
from jetspec.engines.jetspec_plus_plus.launch_env import environment_name


def quantile(values, q):
    ordered = sorted(values)
    at = (len(ordered) - 1) * q
    lo = int(at)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (at - lo)


def stats(values):
    values = list(values)
    if not values:
        raise ValueError('round anatomy requires timing samples')
    return dict(count=len(values), mean=mean(values), median=median(values),
                p90=quantile(values, .9))


def aligned_round_ms(raw):
    """Walk serial N=1 prompts, skipping each prompt's admission/prefill step."""
    metrics = raw['metrics']
    records = metrics['jetspec_round_records']
    steps = metrics['target_step_times']
    n = len(raw['outputs'])
    indices = [int(row['prompt_index']) for row in records]
    if (not records or len(steps) != len(records) + n or indices != sorted(indices)
            or any(i < 0 or i >= n for i in indices)):
        raise ValueError('round steps are not aligned to serial prompt records')
    cursor = pos = 0
    values = []
    for prompt in range(n):
        cursor += 1
        while pos < len(records) and indices[pos] == prompt:
            values.append(steps[cursor] * 1000)
            cursor += 1
            pos += 1
    if cursor != len(steps) or pos != len(records):
        raise ValueError('round steps were not exhausted')
    return values


def anatomy(raw, *, log=None):
    metrics = raw['metrics']
    rounds = aligned_round_ms(raw)
    verify = metrics['jetspec_verify_round_ms']
    draft = metrics['jetspec_draft_round_ms']
    if len(verify) != len(rounds):
        raise ValueError('verify timings are not aligned to verifier records')
    result = dict(round_ms=stats(rounds), verify_ms=stats(verify), draft_ms=stats(draft),
                inner_round_ms_mean=1000 * metrics['decode_total_time'] / len(rounds),
                draft_work_ms_per_verified_round=sum(draft) / len(rounds),
                draft_source='candidate wire draft_ms',
                source='existing host counters; target_step_times excluding every prompt prefill',
                population='all measured physical rounds, before EOS/rule-A credit reduction',
                limitation='draft publications include unused proposals and redrafts; draft and verify overlap, so their durations do not add to round wall time')
    if raw.get('run_manifest', {}).get('env', {}).get(environment_name('engine', 'engine_settings.chain_graph')) == 'seg':
        result.update(draft_ms=None, draft_work_ms_per_verified_round=None,
                      draft_source='unmeasured: graphed candidate wire timer is zero-filled')
        records, active = [], None
        log = log or ''
        phased = 'SERVE_READ_PHASE_START ' in log
        for line in log.splitlines():
            if line.startswith('SERVE_READ_PHASE_START '):
                active = line.split(' ', 1)[1]
            elif line.startswith('SERVE_READ_PHASE_END '):
                active = None
            if phased and active != 'measured':
                continue
            if 'EVAL_DRAFT_TIMING ' in line:
                row, _ = json.JSONDecoder().raw_decode(line.split('EVAL_DRAFT_TIMING ', 1)[1])
                if row['seq_id'] >= 0:
                    records.append(row)
        if records:
            total = [r['total_ms'] for r in records]
            redraft = [r['redraft_ms'] for r in records if r['redraft_ms'] is not None]
            proposal = [r['total_ms'] - (r['redraft_ms'] or 0) for r in records]
            if any(v < 0 for v in proposal + redraft):
                raise ValueError('draft timing spans are negative')
            result.update(draft_ms=stats(proposal), redraft_ms=stats(redraft) if redraft else None,
                          draft_graph_call_ms=stats(total),
                          draft_work_ms_per_verified_round=sum(total) / len(rounds),
                          draft_source='eval host spans from graph.run entry through synchronized publication, split at exact-redraft publication; admission excluded')
    return result


def paired_tau_delta(candidate, reference, *, replicates=20000, seed=9504):
    """Resample shared prompt indices, then subtract the two ratios of sums."""
    et, er = candidate['prompt_tokens'], candidate['prompt_rounds']
    rt, rr = reference['prompt_tokens'], reference['prompt_rounds']
    n = len(et)
    if (n == 0 or any(len(v) != n for v in (er, rt, rr))
            or any(v <= 0 for v in er + rr) or any(v < 0 for v in et + rt)
            or replicates < 2):
        raise ValueError('paired bootstrap requires complete positive-round prompt clusters')
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        a = b = c = d = 0
        for _ in range(n):
            i = rng.randrange(n)
            a += et[i]
            b += er[i]
            c += rt[i]
            d += rr[i]
        samples.append(a / b - c / d)
    ci = [quantile(samples, .025), quantile(samples, .975)]
    return dict(eval_tau=sum(et) / sum(er), reference_tau=sum(rt) / sum(rr),
                delta_tau=sum(et) / sum(er) - sum(rt) / sum(rr), ci95=ci,
                includes_zero=ci[0] <= 0 <= ci[1], seed=seed, replicates=replicates,
                prompts=n, method='paired prompt-cluster bootstrap; difference of ratios of sums; linearly interpolated percentiles',
                limitation='conditional on these two reads; excludes between-run and node variance')


def _identity(raw):
    manifest = raw['run_manifest']
    values, env = manifest['values'], manifest['env']
    args = values['args']
    target_files = tuple(sorted((row['role'], row['sha256']) for row in manifest['files']
                                if row['role'].startswith('target/')))
    if not target_files:
        raise ValueError('matched target file identity is missing')
    return (values['prompts_sha256'], len(raw['outputs']), raw['max_new'],
            args['temp'], args['seed'], args['size'],
            env.get(environment_name('model', 'model_settings.target_dtype'), 'bf16'), target_files)


def reduce_paper(raw):
    paper_metrics.EOS = raw.get('eval', {}).get('eos_token_ids', [151643, 151645])
    return paper_metrics.reduce_read(raw, 'jetspecpp', loop_cut=True)


def compare_reads(candidate, reference, *, replicates=20000, seed=9504, log=None):
    if _identity(candidate) != _identity(reference):
        raise ValueError('reads do not have matched ordered prompts, count, cap, sampling, model and dtype')
    ec, rc = reduce_paper(candidate), reduce_paper(reference)
    ea, ra = anatomy(candidate, log=log), anatomy(reference)
    ew, rw = candidate['wall_seconds'], reference['wall_seconds']
    return dict(bootstrap=paired_tau_delta(ec, rc, replicates=replicates, seed=seed),
                eval_metrics=ec, reference_metrics=rc, eval_anatomy=ea, reference_anatomy=ra,
                payoff=dict(eval_timed_read_wall_seconds=ew, reference_timed_read_wall_seconds=rw,
                            timed_read_wall_ratio_eager_over_eval=rw / ew,
                            eval_round_ms=ea['round_ms']['mean'], reference_round_ms=ra['round_ms']['mean'],
                            round_ratio_eager_over_eval=ra['round_ms']['mean'] / ea['round_ms']['mean'],
                            wall_population='timed read only; model loading, graph capture and queue wait excluded',
                            placement=dict(eval_rank_device_map=candidate['run_manifest']['env'].get(
                                environment_name('hardware', 'hardware_settings.rank_device_map')), reference_rank_device_map=reference['run_manifest']['env'].get(
                                environment_name('hardware', 'hardware_settings.rank_device_map')))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate', type=Path)
    p.add_argument('--reference', type=Path)
    p.add_argument('--reference-member', help='read.json member of a local reference tar archive')
    p.add_argument('--output-json', type=Path, required=True)
    p.add_argument('--log', type=Path, help='raw eval log with synchronized draft spans')
    p.add_argument('--replicates', type=int, default=20000)
    p.add_argument('--seed', type=int, default=9504)
    a = p.parse_args()
    candidate = json.loads(a.candidate.read_text())
    if a.reference:
        if a.reference_member:
            with tarfile.open(a.reference) as archive:
                reference = json.load(archive.extractfile(a.reference_member))
        else:
            reference = json.loads(a.reference.read_text())
        result = compare_reads(candidate, reference, replicates=a.replicates, seed=a.seed,
                               log=a.log.read_text() if a.log else None)
    else:
        result = dict(paper_metrics=reduce_paper(candidate), anatomy=anatomy(
            candidate, log=a.log.read_text() if a.log else None))
    result['candidate'] = str(a.candidate)
    result['reference'] = str(a.reference) if a.reference else None
    result['reference_member'] = a.reference_member
    a.output_json.write_text(json.dumps(result, indent=2) + '\n')
    print('EVAL_ANALYSIS ' + str(a.output_json))


if __name__ == '__main__':
    main()

"""Paper rule-A credit and timing, using the bank's native per-request reducer."""
import random
from statistics import mean
from jetspec.engines.jetspec_plus_plus.read import subset_recompute as sr

EOS = [248044, 248046]


def reduce_read(raw, method, loop_cut=True):
    outputs, metrics = raw['outputs'], raw['metrics']
    cuts = {}
    for i, row in enumerate(outputs):
        ids = row['token_ids'][:2048]
        if loop_cut and len(ids) == 2048 and not any(t in EOS for t in ids):
            last = ids[-64:]
            occurrences, j = [], 0
            while j + 64 <= len(ids):
                if ids[j:j+64] == last:
                    occurrences.append(j)
                    j += 64
                else:
                    j += 1
            if len(occurrences) >= 3:
                cuts[i] = occurrences[0]
    full = sr.segment(method, outputs, metrics, EOS, 2048)
    cut = sr.segment(method, outputs, metrics, EOS, 2048, cut=cuts)
    stops, pre, dec, cred, rounds, allr, credited_time = cut
    # Scale outer step attribution to the measured inner prefill/decode clocks,
    # exactly as subset_recompute.view and the paper baseline bank do.
    P, D = metrics['prefill_total_time'], metrics['decode_total_time']
    pre_scale = P / sum(full[1])
    dec_scale = D / sum(full[2])
    times = [pre_scale*p + dec_scale*(c if i in cuts else d)
             for i, (p, d, c) in enumerate(zip(pre, dec, credited_time))]
    tokens = [max(0, stop-1) for stop in stops]
    assert tokens == cred, 'CREDIT_NOT_EQUAL_STOP_MINUS_PREFILL'
    total, R, seconds = sum(tokens), sum(rounds), sum(times)
    return dict(tokens=total, rounds=R, tau=total/R if R else 0,
                served_seconds=seconds, tok_s=total/seconds,
                cut_prompts=sorted(cuts), cut_onsets=cuts,
                prompt_tokens=tokens, prompt_rounds=rounds, prompt_seconds=times)


def pooled(parts):
    tokens = sum(p['tokens'] for p in parts)
    rounds = sum(p['rounds'] for p in parts)
    seconds = sum(p['served_seconds'] for p in parts)
    return dict(tokens=tokens, rounds=rounds, served_seconds=seconds,
                tau=tokens/rounds, tok_s=tokens/seconds)


def clustered_bootstrap(clusters, replicates=5000, seed=0):
    rng = random.Random(seed)
    def ratio(rows):
        ot = sum(a[0] for a, b in rows)
        os = sum(a[1] for a, b in rows)
        bt = sum(b[0] for a, b in rows)
        bs = sum(b[1] for a, b in rows)
        return (ot/os)/(bt/bs)
    point = {b:ratio(rows) for b, rows in clusters.items()}
    samples = []
    for _ in range(replicates):
        values = []
        for rows in clusters.values():
            sampled = [rows[rng.randrange(len(rows))] for _ in rows]
            values.append(ratio(sampled))
        samples.append(mean(values))
    samples.sort()
    return dict(mean_ratio=mean(point.values()), benchmark_ratios=point,
                ci95=[samples[int(.025*(replicates-1))], samples[int(.975*(replicates-1))]],
                replicates=replicates, seed=seed,
                method='paired prompt-cluster bootstrap, stratified by benchmark; MT-Bench turns share the original turn-1 prompt cluster; equal-weight arithmetic mean over seven benchmarks',
                limitation='conditional on banked runs; does not estimate node, kernel-cache, or between-run variance')

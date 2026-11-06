"""Paper rule-A credit and timing, using the bank's native per-request reducer."""
import random
from statistics import mean
from jetspec.engines.jetspec_plus_plus.launch.records import reduce_paper_read

EOS = [248044, 248046]




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


def reduce_read(raw, method, loop_cut=True):
    return reduce_paper_read(raw, method, loop_cut, eos_token_ids=EOS)

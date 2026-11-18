"""Per-round DSpark draft timing from a read's blockhead_round_records (CUDA events on rank 0).

head_ms: context extension + query graph (with TP collectives, includes any wait for the followers);
markov_ms: the Markov correction plus the copy-out and the host sync that ends propose().

usage: python draft_timing.py read.json [read.json ...]
"""
import json
import statistics as st
import sys

for path in sys.argv[1:]:
    metrics = json.load(open(path))['metrics']
    rows = [r for r in metrics.get('blockhead_round_records', []) if r.get('round', 0) > 1]
    if not rows:
        print(f'{path}: no blockhead round records')
        continue
    head = [r['head_ms'] for r in rows]
    markov = [r['markov_ms'] for r in rows]
    print(f'{path}\n  rounds {len(rows)}  head_ms mean {st.mean(head):.3f} p50 {st.median(head):.3f}'
          f'  markov_ms mean {st.mean(markov):.3f} p50 {st.median(markov):.3f}')

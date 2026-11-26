"""Plot one-chip vs two-chip JetSpec++ serving from foresight_pareto.py outputs.

    python -m jetspec.engines.jetspec_plus_plus_tpu.plot_two_chip out/fsp_exact.json out/fsp_exact_b4.json \
        --baseline out/dflash_v5e.json --output assets/frontier_two_chip_v5e.png --data assets/frontier_two_chip_v5e.json

--baseline: one-chip runs in pareto.py's format (e.g. DFlash on the same requests), drawn as extra lines.

Each B is one point per engine; for two chips the best row count at that B (by total tok/s) is on the line and
the other row counts are drawn as small hollow markers.
"""
import argparse
import json

INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
ONE, TWO, BASE = '#2a78d6', '#1baf7a', '#eb6834'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--data', help='also write the plotted points as JSON')
    parser.add_argument('--baseline', action='append', default=[], help="one-chip pareto.py JSON, e.g. DFlash")
    parser.add_argument('--chip', default='TPU v5e')
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = [json.load(open(path)) for path in args.runs]
    meta = {k: v for k, v in runs[0].items() if k != 'points'}
    pts = [p for r in runs for p in r['points'] if 'error' not in p]
    one = {p['slots']: p for p in pts if p['engine'] == 'one-chip'}
    two_all = [p for p in pts if p['engine'] == 'two-chip']
    two = {}
    for p in two_all:
        if p['slots'] not in two or p['total_tok_s'] > two[p['slots']]['total_tok_s']:
            two[p['slots']] = p
    bs = sorted(set(one) & set(two))

    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    others = [p for p in two_all if p is not two.get(p['slots'])]
    ax.scatter([p['per_user_tok_s_mean'] for p in others], [p['total_tok_s'] for p in others], s=26,
               facecolor='none', edgecolor=TWO, linewidth=1, alpha=0.55, zorder=2,
               label='JetSpec++ two chips, other row counts')
    baselines = [json.load(open(path)) for path in args.baseline]
    for base in baselines:
        bp = sorted((p for p in base['points'] if 'error' not in p), key=lambda p: p['slots'])
        name = {'dflash': 'DFlash', 'ar': 'AR'}.get(base.get('method'), base.get('method', 'baseline'))
        xs, ys = [p['per_user_tok_s_mean'] for p in bp], [p['total_tok_s'] for p in bp]
        ax.plot(xs, ys, color=BASE, linewidth=2, zorder=3)
        ax.scatter(xs, ys, s=60, marker='s', color=BASE, edgecolor=SURFACE, linewidth=2, zorder=4,
                   label=f'{name}, one chip')
    for series, color, marker, label in ((one, ONE, 'o', 'JetSpec++, one chip'),
                                         (two, TWO, 'D', 'JetSpec++, two chips (verify | draft ahead), best rows')):
        xs = [series[b]['per_user_tok_s_mean'] for b in bs]
        ys = [series[b]['total_tok_s'] for b in bs]
        ax.plot(xs, ys, color=color, linewidth=2, zorder=3)
        ax.scatter(xs, ys, s=64, marker=marker, color=color, edgecolor=SURFACE, linewidth=2, zorder=4, label=label)
    for b in bs:
        q = two[b]
        ax.annotate(f'B={b}', (q['per_user_tok_s_mean'], q['total_tok_s']), textcoords='offset points',
                    xytext=(-8, 8), ha='right', fontsize=8.5, color=SECONDARY)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Per-user throughput (tok/s, mean over requests)', color=SECONDARY)
    ax.set_ylabel('Total throughput (tok/s)', color=SECONDARY)
    ax.set_title(f'Qwen3-8B on {args.chip}: JetSpec++ {meta.get("schedule", "")} on one chip vs two chips{", and DFlash" if baselines else ""}\n'
                 f'{meta.get("requests")} GSM8K requests, ≤{meta.get("max_tokens")} tokens, greedy, continuous batching, '
                 f'int8 target', color=INK, fontsize=10.5, loc='left')
    ax.grid(True, which='major', color=GRID, linewidth=0.8)
    ax.grid(True, which='minor', color=GRID, linewidth=0.4, alpha=0.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, which='both')
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f'{v:,.0f}')
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    minor = FuncFormatter(lambda v, _: f'{v:,.0f}' if str(int(v))[0] in '2356' else '')
    ax.xaxis.set_minor_formatter(minor)
    ax.yaxis.set_minor_formatter(minor)
    legend = ax.legend(frameon=False, loc='lower left', fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    fig.savefig(args.output, facecolor=SURFACE)
    rows = []
    print(f'{"B":>3s} {"one per-user":>12s} {"one total":>10s} {"two per-user":>12s} {"two total":>10s} {"rows":>5s}')
    for b in bs:
        p, q = one[b], two[b]
        rows.append(dict(slots=b, one_chip=p, two_chip=q))
        print(f'{b:3d} {p["per_user_tok_s_mean"]:12.1f} {p["total_tok_s"]:10.1f} {q["per_user_tok_s_mean"]:12.1f} '
              f'{q["total_tok_s"]:10.1f} {q.get("rows", 0):5d}')
    if args.data:
        json.dump(dict(meta, chip=args.chip, best=rows, two_chip_all=two_all,
                       baselines=[dict(method=b.get('method'), requests=b.get('requests'), max_tokens=b.get('max_tokens'),
                                       points=b['points']) for b in baselines]), open(args.data, 'w'), indent=1)


if __name__ == '__main__':
    main()

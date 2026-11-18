"""Plot the serving frontier from pareto.py outputs: per-user tok/s (x) vs total tok/s (y).

    python -m jetspec.engines.jetspec_plus_plus_tpu.plot_pareto out/ar.json out/dflash.json \
        out/jetspec.json --output out/pareto.png
"""
import argparse
import json

INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
# Fixed categorical order (validated palette), so a run keeps its color whichever runs are plotted.
STYLE = {'AR': ('#2a78d6', 'o'), 'DFlash': ('#eb6834', 's'), 'J1-R1': ('#1baf7a', 'D'),
         'J1-R0': ('#eda100', '^'), 'J0-R0': ('#e87ba4', 'v')}


def style_key(run):
    return {'ar': 'AR', 'dflash': 'DFlash'}.get(run['method'], run['schedule'])


def frontier(points):
    """Points not dominated on (per-user, total), sorted by per-user speed."""
    keep = [p for p in points if not any(q is not p and q[0] >= p[0] and q[1] >= p[1]
                                         and (q[0] > p[0] or q[1] > p[1]) for q in points)]
    return sorted(keep)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    runs = [json.load(open(path)) for path in args.runs]
    runs.sort(key=lambda r: list(STYLE).index(style_key(r)))
    table = []
    every = []
    for run in runs:
        color, marker = STYLE.get(style_key(run), ('#6250d6', 'P'))
        pts = [p for p in run['points'] if 'error' not in p]
        xs = [p['per_user_tok_s_mean'] for p in pts]
        ys = [p['total_tok_s'] for p in pts]
        every += [(x, y) for x, y in zip(xs, ys)]
        ax.plot(xs, ys, color=color, linewidth=2, zorder=2)
        ax.scatter(xs, ys, s=64, marker=marker, color=color, edgecolor=SURFACE, linewidth=2, zorder=3,
                   label=run['label'])
        for p, x, y in zip(pts, xs, ys):
            if run['method'] in ('ar', 'dflash'):      # B labels on two series keep the plot legible
                ax.annotate(f'B={p["slots"]}', (x, y), textcoords='offset points', xytext=(7, -3),
                            fontsize=7.5, color=SECONDARY)
            table.append((run['label'], p['slots'], p['per_user_tok_s_mean'], p['total_tok_s'], p.get('tau')))
    front = frontier(every)
    ax.plot([x for x, _ in front], [y for _, y in front], color=INK, linewidth=1, linestyle=(0, (4, 3)),
            zorder=4, label='Pareto frontier (all runs)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Per-user throughput (tok/s, mean over requests)', color=SECONDARY)
    ax.set_ylabel('Total throughput (tok/s per chip)', color=SECONDARY)
    first = runs[0]
    ax.set_title(f'Qwen3-8B serving frontier on one TPU v6e chip — {first["dataset"]}, {first["requests"]} '
                 f'requests, ≤{first["max_tokens"]} tokens, greedy', color=INK, fontsize=11, loc='left')
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
    legend = ax.legend(frameon=False, loc='lower center', fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    fig.savefig(args.output, facecolor=SURFACE)
    print(f'{"method":32s} {"B":>3s} {"per-user":>9s} {"total":>9s} {"tau":>6s}')
    for label, b, x, y, tau in table:
        print(f'{label:32s} {b:3d} {x:9.1f} {y:9.1f} {tau if tau is None else round(tau, 2)!s:>6s}')


if __name__ == '__main__':
    main()

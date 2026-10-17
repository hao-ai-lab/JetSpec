"""Draw an exhaustively attributed two-rank round directly from report.json."""

# Locate the component from either its category or its stable forwarding path.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_TOOL_ROOT = next((p for p in _ToolPath(__file__).resolve().parents
                   if (p / "artifacts/manifest.json").is_file() or (p / "artifacts/manifest.json").is_file()), _ToolPath.cwd())
_tool_sys.path[:0] = [str(_TOOL_ROOT), str(_TOOL_ROOT / 'engine'), str(_ToolPath(__file__).resolve().parent)]
if __name__ == "__main__" and any(a in ("-h", "--help") for a in _tool_sys.argv[1:]):
    from jetspec.engines.jetspec_plus_plus.serving.entrypoints import cpu_help as _cpu_help
    _cpu_help(__file__)

import argparse
from html import escape
import json
from pathlib import Path


# One colour per served component; a component keeps its colour wherever it runs
# (the adapter refresh is the adapter, the J1..J3 passes are the drafter).
FAMILIES = [
    ('verify forward', '#3f6fb5', lambda c: c == 'verify forward'),
    ('accept + KV commit', '#8fb8de',
     lambda c: c in ('accept decision', 'KV commit', 'verify accept commit')),
    ('adapter', '#3f9a55', lambda c: c.startswith('adapter')),
    ('anchor decoder', '#6b46a1', lambda c: 'decoder' in c),
    ('drafter pass', '#e0973a',
     lambda c: c.startswith('draft pass') or c in ('block head', 'markov sampling')),
    ('context update', '#1f9e9e', lambda c: c.startswith('context')),
    ('rows', '#c084d2', lambda c: c in ('row selection', 'row assembly', 'block selection')),
    ('k predictor', '#d14fa0', lambda c: c == 'k predictor'),
    ('wait', '#d9d9d9', lambda c: c.startswith('wait for')),
]
HOST, UNATTRIBUTED = '#ddd3c0', 'url(#hatch)'
SHORT = {'verify forward': 'verify', 'anchor decoder': 'decoder', 'adapter refresh': 'refresh',
         'context early': 'ctx', 'context update': 'ctx', 'context refresh inject': 'ctx',
         'row assembly': 'rows', 'row selection': 'sel', 'block head': 'block head',
         'markov sampling': 'markov'}
FONT = 'font-family="sans-serif"'


def _family(row):
    if row['category'] == 'unattributed':
        return 'unattributed', UNATTRIBUTED
    for name, fill, match in FAMILIES:
        if match(row['component']):
            return name, fill
    return 'host bookkeeping', HOST


def _labels(segments):
    """Short in-bar labels: drafter passes read J0, J1, ... in lane order."""
    passes, out = 0, []
    for row in segments:
        name, _ = _family(row)
        if row['component'] in ('block head', 'markov sampling'):
            out.append(SHORT[row['component']])
        elif name == 'drafter pass' and row['category'] == 'named':
            out.append(f'J{passes}')
            passes += 1
        elif name == 'drafter pass':
            out.append(None)
        elif name in ('host bookkeeping', 'unattributed', 'wait'):
            out.append(None)
        else:
            out.append(SHORT.get(row['component'], row['component']))
    return out


def draw(report_path, output_path, *, title=None):
    report_path, output_path = Path(report_path), Path(output_path)
    result = json.loads(report_path.read_text(encoding='utf-8'))
    representative = result['representative']
    lanes = [side for side in ('verify', 'draft') if side in representative]
    maximum = max(representative[side]['window_ms'] for side in lanes)
    # 860 px of viewBox scaled to the column (width 100 %): no horizontal scroll.
    width, left, right, bar, lane_gap = 860, 58, 64, 30, 78
    scale = (width - left - right) / maximum if maximum else 1.
    body, used, top = [], {}, 40
    for index, side in enumerate(lanes):
        y = top + index * lane_gap
        body.append(f'<text x="{left-8}" y="{y+20}" text-anchor="end" {FONT} '
                    f'font-size="13">{side}</text>')
        segments = representative[side]['segments']
        below = []
        for row, label in zip(segments, _labels(segments)):
            x, w = left + row['start_ms'] * scale, max(.8, row['ms'] * scale)
            family, fill = _family(row)
            used[family] = fill
            body.append(f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="{bar}" fill="{fill}" '
                        f'stroke="#fff" stroke-width=".6"><title>{escape(row["component"])} · '
                        f'{row["ms"]:.3f} ms ({escape(row["category"])})</title></rect>')
            if label is None:
                continue
            text = f'{label} {row["ms"]:.1f}'
            if w >= len(text) * 5.6 + 6:
                body.append(f'<text x="{x+w/2:.2f}" y="{y+19}" text-anchor="middle" {FONT} '
                            f'font-size="10" fill="#fff">{escape(text)}</text>')
            elif w >= len(label) * 5.6 + 4:
                body.append(f'<text x="{x+w/2:.2f}" y="{y+19}" text-anchor="middle" {FONT} '
                            f'font-size="10" fill="#fff">{escape(label)}</text>')
            elif row['ms'] >= .5:
                below.append((x + w / 2, text))
        # labels that do not fit in their bar sit under the lane, staggered, never clipped
        ends = []
        for cx, text in below:
            half = len(text) * 2.6 + 3
            cx = min(max(cx, left + half), width - half)
            level = next((i for i, end in enumerate(ends) if cx - half > end), len(ends))
            if level == len(ends):
                ends.append(0.)
            ends[level] = cx + half
            ty = y + bar + 11 + level * 11
            body.append(f'<line x1="{cx:.2f}" y1="{y+bar}" x2="{cx:.2f}" y2="{ty-8}" '
                        f'stroke="#999" stroke-width=".6"/>')
            body.append(f'<text x="{cx:.2f}" y="{ty}" text-anchor="middle" {FONT} '
                        f'font-size="9" fill="#333">{escape(text)}</text>')
        lane = representative[side]
        body.append(f'<text x="{left+lane["window_ms"]*scale+5:.2f}" y="{y+20}" {FONT} '
                    f'font-size="11">{lane["window_ms"]:.1f} ms</text>')
    y = top + len(lanes) * lane_gap + 4
    totals = {row['component']: row['median_ms'] for row in result.get('span_totals', [])
              if row['side'] == 'verify'}
    if 'verify_accept_commit' in totals:
        forward = totals.get('verify_forward', 0.)
        body.append(f'<text x="{left}" y="{y}" {FONT} font-size="11">'
                    f'verify {totals["verify_accept_commit"]:.2f} · forward {forward:.2f} ms '
                    f'(median over rounds)</text>')
        y += 16
    caption = 'Host-clock spans; CUDA event pairs annotate device components. Every lane pixel is attributed.'
    if any(row['component'] == 'k predictor' for side in lanes
           for row in representative[side]['segments']):
        caption += ' k predictor still in the served path on this sha.'
    body.append(f'<text x="{left}" y="{y}" {FONT} font-size="10" fill="#555">{escape(caption)}</text>')
    y += 20
    x = left
    order = [name for name, _, _ in FAMILIES] + ['host bookkeeping', 'unattributed']
    for family in sorted(used, key=order.index):
        item = 22 + len(family) * 6
        if x + item > width - 8:
            x, y = left, y + 18
        body.append(f'<rect x="{x}" y="{y-10}" width="13" height="11" fill="{used[family]}" stroke="#777"/>')
        body.append(f'<text x="{x+17}" y="{y}" {FONT} font-size="10">{escape(family)}</text>')
        x += item + 12
    height = y + 12
    head = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="xMinYMin meet">',
        '<defs><pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(45)"><rect width="7" height="7" fill="#fff"/>'
        '<line x1="0" y1="0" x2="0" y2="7" stroke="#555" stroke-width="2"/></pattern></defs>',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="{left}" y="20" {FONT} font-size="14" font-weight="bold">'
        f'{escape(title or report_path.parent.name)}</text>',
    ]
    output_path.write_text('\n'.join(head + body + ['</svg>']) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report'); parser.add_argument('--output'); parser.add_argument('--title')
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path(args.report).with_name('round.svg')
    draw(args.report, output, title=args.title)
    print(output)


if __name__ == '__main__':
    main()

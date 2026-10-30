"""Check every JetSpec++ output token against plain AR greedy generation."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.jetspec_plus_plus._common import inputs, prepare, timed, read_result, write_result


def main(argv=None, *, llm_factory=None, drafter_factory=None):
    args, rows = inputs(argv, description=__doc__, samples=4, max_tokens=256,
                        output='bench/outputs/jetspec_plus_plus_verify.json')
    from jetspec.methods.jetspec_plus_plus.reader import read_ar, read_passes

    llm, rows, config, drafter = prepare(args, rows, llm_factory=llm_factory,
                                        drafter_factory=drafter_factory)
    ar, ar_seconds = timed(llm, lambda: read_ar(llm, rows, tokens=args.max_tokens))
    outputs, seconds = timed(llm, lambda: read_passes(
        llm, rows, drafter_factory=lambda **kwargs: drafter, config=config, tokens=args.max_tokens))
    equal = len(outputs) == len(ar) and all(
        actual['token_ids'] == expected['token_ids'] for actual, expected in zip(outputs, ar))
    result = read_result(args, config, llm, outputs, seconds)
    result.update(equal=equal, ar_outputs=ar, ar_wall_seconds=ar_seconds)
    print(f"{'PASS' if equal else 'FAIL'} prompts={len(rows)}")
    write_result(args.output, result)
    print(f'JSON: {args.output}')
    return 0 if equal else 1


if __name__ == '__main__':
    raise SystemExit(main())

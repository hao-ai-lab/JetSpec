"""Measure JetSpec++ acceptance length with the paper's torch read loop."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.jetspec_plus_plus._common import inputs, prepare, timed, read_result, write_result


def main(argv=None, *, llm_factory=None, drafter_factory=None):
    args, rows = inputs(argv, description=__doc__, samples=64, max_tokens=2048,
                        output='bench/outputs/jetspec_plus_plus_tau.json')
    from jetspec.methods.jetspec_plus_plus.reader import read_passes

    llm, rows, config, drafter = prepare(args, rows, llm_factory=llm_factory,
                                        drafter_factory=drafter_factory)
    outputs, seconds = timed(llm, lambda: read_passes(
        llm, rows, drafter_factory=lambda **kwargs: drafter, config=config, tokens=args.max_tokens))
    result = read_result(args, config, llm, outputs, seconds)
    summary = result['summary']
    tau = summary['tau'] if summary['tau'] is not None else 0.0
    print(f"prompts={summary['n_prompts']} tokens={summary['tokens']} rounds={summary['rounds']} "
          f"tau={tau:.3f} wall={seconds:.3f}s {summary['tokens_per_second']:.1f} tok/s")
    write_result(args.output, result)
    print(f'JSON: {args.output}')
    return result


if __name__ == '__main__':
    main()

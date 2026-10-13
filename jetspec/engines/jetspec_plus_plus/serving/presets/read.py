"""All engine entries use the existing manifest-pinned recipe read."""
import sys


def run_read(name, argv=None):
    import jetspec.engines.jetspec_plus_plus.serving.read as serve_read
    arguments = list(sys.argv[1:] if argv is None else argv)
    if '--engine' not in arguments:
        arguments += ['--engine', name]
    return serve_read._main(serve_read._parser().parse_args(arguments))

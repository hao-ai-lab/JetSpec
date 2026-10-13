"""The reference engine entry."""
from jetspec.engines.jetspec_plus_plus.serving.presets.read import run_read


def run(argv=None):
    return run_read('reference', argv)

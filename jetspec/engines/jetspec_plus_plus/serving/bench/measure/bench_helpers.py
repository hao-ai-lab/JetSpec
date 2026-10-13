"""Category entry point for the immutable upstream tool."""
from pathlib import Path
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "artifacts/manifest.json").is_file())
sys.path[:0] = [str(ROOT), str(ROOT / "serve")]
from jetspec.engines.jetspec_plus_plus.serving.tools.lib.entrypoints import forward
forward(ROOT / 'serving/bench/bench_helpers.py', globals())

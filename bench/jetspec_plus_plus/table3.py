"""Run the paper's Table 3 arms on locally provisioned artifacts."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jetspec.engines.jetspec_plus_plus.table3.command import jobs, main, parser


if __name__ == '__main__':
    raise SystemExit(main())

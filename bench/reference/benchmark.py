import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1] / "jetspec/reference/benchmark.py"), run_name="__main__")

import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1] / "jetspec/profiling/collect_tree_diagnostics.py"), run_name="__main__")

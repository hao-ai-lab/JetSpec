import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1] / "jetspec/profiling/compare_conditioned_draft_logits.py"), run_name="__main__")

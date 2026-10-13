import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1] / "jetspec/profiling/compare_engine_with_vllm_integration.py"), run_name="__main__")

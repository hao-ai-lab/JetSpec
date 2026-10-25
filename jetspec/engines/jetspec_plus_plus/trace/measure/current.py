"""Install roundtrace on the current portable serving launcher.

The launcher imports this module only when ``JETSPEC_ROUNDTRACE`` is set.  The
hook preserves the serving recorder: rank-local files and a fixed CUDA-event pool
allocated after each model runner is constructed, before measured serving.
"""

# Locate the component from either its category or its stable forwarding path.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_TOOL_ROOT = next((p for p in _ToolPath(__file__).resolve().parents
                   if (p / "artifacts/manifest.json").is_file() or (p / "artifacts/manifest.json").is_file()), _ToolPath.cwd())
_tool_sys.path[:0] = [str(_TOOL_ROOT), str(_TOOL_ROOT / 'engine'), str(_ToolPath(__file__).resolve().parent)]
if __name__ == "__main__" and any(a in ("-h", "--help") for a in _tool_sys.argv[1:]):
    from jetspec.engines.jetspec_plus_plus.serving.entrypoints import cpu_help as _cpu_help
    _cpu_help(__file__)

import functools
import os

from .spans import DEFAULT_CUDA_EVENT_PAIRS


_installed = None




def install(run_config=None):
    """Attach declared spans and prepare each rank's fixed event pool once."""
    global _installed
    if _installed is not None:
        return _installed

    from . import attach as attachment

    handle = attachment.attach(run_config=run_config)
    if handle is None:
        return None

    from jetspec.engines.jetspec_plus_plus.serving.model_runner import ModelRunner

    original_init = ModelRunner.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        rank = kwargs.get("rank", args[1] if len(args) > 1 else 0)
        handle.recorder.rank = rank
        handle.recorder.path = handle.recorder.path.with_name(f"rounds-{rank}.jsonl")
        config = kwargs.get("config", args[0] if args else None)
        handle.recorder.graph_replays = not config.enforce_eager
        if handle.recorder.graph_replays:
            attachment.enable_graph_replays(handle)
        result = original_init(self, *args, **kwargs)
        prepared = handle.recorder.prepare_cuda_events(
            int(setting(self, 'engine.engine_settings.roundtrace_event_pairs', str(DEFAULT_CUDA_EVENT_PAIRS)))
        )
        if prepared:
            print(f"ROUNDTRACE_EVENT_POOL rank={rank} pairs={prepared}", flush=True)
        return result

    ModelRunner.__init__ = init
    handle.originals.append((ModelRunner, "__init__", original_init, init))
    _installed = handle
    return handle

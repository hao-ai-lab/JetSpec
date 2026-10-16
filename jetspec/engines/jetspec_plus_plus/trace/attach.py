"""Install only the declared call-boundary wrappers; no engine source changes."""

# Locate the component from either its category or its stable forwarding path.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_TOOL_ROOT = next((p for p in _ToolPath(__file__).resolve().parents
                   if (p / "artifacts/manifest.json").is_file() or (p / "artifacts/manifest.json").is_file()), _ToolPath.cwd())
_tool_sys.path[:0] = [str(_TOOL_ROOT), str(_TOOL_ROOT / 'engine'), str(_ToolPath(__file__).resolve().parent)]
if __name__ == "__main__" and any(a in ("-h", "--help") for a in _tool_sys.argv[1:]):
    from jetspec.engines.jetspec_plus_plus.serving.entrypoints import cpu_help as _cpu_help
    _cpu_help(__file__)

import atexit
import functools
import importlib
import os
import warnings
from .spans import Recorder

_active = None


class Attachment:
    def __init__(self, recorder):
        self.recorder = recorder
        self.originals = []

    def detach(self):
        global _active
        for owner, name, original, wrapper in reversed(self.originals):
            if getattr(owner, name) is wrapper:
                setattr(owner, name, original)
        self.originals.clear()
        self.recorder.close()
        if _active is self:
            _active = None


def _wrapper(original, recorder, label, side, kind):
    if kind == 'chain_driver':
        @functools.wraps(original)
        def driver(owner, *args, **kwargs):
            with recorder.round(side, sequences=[(owner.seq_id, owner.next_round)]):
                result = original(owner, *args, **kwargs)
            recorder.local.last_round['command'] = result
            return result
        return driver
    if kind == 'chain_metrics':
        @functools.wraps(original)
        def chain(owner, key, *args, **kwargs):
            if recorder.closed or getattr(recorder.local, 'round', None) is None:
                return original(owner, key, *args, **kwargs)
            with recorder.span(label, side) as row:
                result = original(owner, key, *args, **kwargs)
                # These events and the publish synchronization already belong
                # to the chain. Do not add events, waits, or inferred offsets.
                row['segments_ms'] = dict(owner.last_ms.get(key, {}))
                row['device_span_ms'] = owner.last_span_ms.get(key)
                row['chain_key'] = str(key)
                return result
        return chain
    if kind in ('round', 'round_publish'):
        @functools.wraps(original)
        def rounded(*args, **kwargs):
            metadata = {'await_publish': kind == 'round_publish'}
            if side == 'verify':
                seqs = args[1] if len(args) > 1 else kwargs.get('seqs', [])
                metadata['sequences'] = [(s.seq_id, getattr(s, 'jetspec_round', None)) for s in seqs]
            with recorder.round(side, **metadata):
                return original(*args, **kwargs)
        return rounded
    if kind == 'publish':
        @functools.wraps(original)
        def published(*args, **kwargs):
            previous = getattr(recorder.local, 'round', None)
            latest = getattr(recorder.local, 'last_round', None)
            extend = previous is None and latest is not None and latest['side'] == 'draft' and latest.get('await_publish', False)
            if previous is None and not extend:
                return original(*args, **kwargs)
            if extend:
                recorder.local.round = latest['id']
            row = recorder.begin(label, side)
            try:
                return original(*args, **kwargs)
            finally:
                recorder.end(row)
                if extend:
                    latest['stop'] = row['stop']
                    recorder.local.round = previous
                    recorder.local.last_round = None
        return published
    if kind == 'active_round':
        @functools.wraps(original)
        def active_round(*args, **kwargs):
            active_side = getattr(recorder.local, 'round_side', None)
            if recorder.closed or active_side is None:
                return original(*args, **kwargs)
            row = recorder.begin(label, active_side)
            try:
                return original(*args, **kwargs)
            finally:
                recorder.end(row)
        return active_round
    if kind == 'engine_close':
        @functools.wraps(original)
        def engine_close(*args, **kwargs):
            # LLMEngine.exit(hard=True) ends with os._exit, so the recorder must
            # flush explicitly.  Stop the worker ranks first: waiting on CUDA
            # events while the async drafter is still live can deadlock teardown.
            call_args = list(args)
            call_kwargs = dict(kwargs)
            hard = call_kwargs.get('hard', call_args[1] if len(call_args) > 1 else True)
            if hard:
                if len(call_args) > 1:
                    call_args[1] = False
                else:
                    call_kwargs['hard'] = False
            result = original(*call_args, **call_kwargs)
            recorder.close()
            if hard:
                os._exit(0)
            return result
        return engine_close
    if kind == 'close':
        @functools.wraps(original)
        def close(*args, **kwargs):
            recorder.close()
            return original(*args, **kwargs)
        return close
    if kind == 'passes':
        @functools.wraps(original)
        def protocol(forward, *args, **kwargs):
            if getattr(recorder, 'scoped', False) and getattr(recorder.local, 'round', None) is None:
                return original(forward, *args, **kwargs)
            @functools.wraps(forward)
            def measured(*a, **kw):
                index = kw.get('pass_index', 0)
                row = recorder.begin('draft_pass' if index == 0 else f'refine_{index}', side)
                try:
                    return forward(*a, **kw)
                finally:
                    recorder.end(row)
            row = recorder.begin(label, side)
            try:
                return original(measured, *args, **kwargs)
            finally:
                recorder.end(row)
        return protocol
    if kind == 'sync_query':
        @functools.wraps(original)
        def query(*args, **kwargs):
            name = args[1] if len(args) > 1 else kwargs.get('name')
            if recorder.closed or name != 'query' or getattr(recorder.local, 'round_side', None) != 'draft':
                return original(*args, **kwargs)
            with recorder.span(label, side):
                return original(*args, **kwargs)
        return query
    if kind in ('both', 'side_stream', 'replay'):
        @functools.wraps(original)
        def dual(*args, **kwargs):
            if recorder.closed:
                return original(*args, **kwargs)
            if (not recorder._stack()
                    and getattr(recorder.local, 'round', None) is None
                    and getattr(recorder, 'scoped', False)):
                return original(*args, **kwargs)
            stream = getattr(args[0], 'stream', None) if kind == 'side_stream' else None
            context = (recorder.span(label, side)
                       if kind == 'replay' and recorder.graph_replays
                       else recorder.dual_span(label, side, stream=stream))
            with context:
                return original(*args, **kwargs)
        return dual
    @functools.wraps(original)
    def timed(*args, **kwargs):
        if recorder.closed:
            return original(*args, **kwargs)
        if not recorder._stack() and getattr(recorder.local, 'round', None) is None and label not in ('engine_step', 'scheduler_finished') and getattr(recorder, 'scoped', False) and kind not in ('background', 'background_key'):
            return original(*args, **kwargs)
        if kind == 'cuda':
            with recorder.cuda_span(label, side):
                return original(*args, **kwargs)
        row = recorder.begin(label, side)
        if kind == 'background_key':
            row['key'] = args[2] if len(args) > 2 else kwargs.get('key')
        elif kind == 'message_wait':
            row['key'] = getattr(args[0], 'key', None)
        try:
            return original(*args, **kwargs)
        finally:
            recorder.end(row)
    return timed


def attach(*, recorder=None, table=None, run_config=None):
    """Attach once per process. Unset env returns before any engine import."""
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    global _active
    if recorder is None:
        directory = setting(run_config, 'engine.engine_settings.roundtrace')
        if not directory:
            return None
        if _active is not None:
            return _active
        # Each rank has a distinct process; filenames use PID when RANK is absent.
        recorder = Recorder(directory, rank=os.environ.get('RANK', os.getpid()), run_config=run_config)
    if table is None:
        from .spans_table import SPANS
        table = SPANS
        recorder.scoped = True
    handle = Attachment(recorder)
    for target, label, side, kind in table:
        try:
            module, path = target.split(':', 1)
            owner = importlib.import_module(module)
            names = path.split('.')
            for name in names[:-1]:
                owner = getattr(owner, name)
            original = getattr(owner, names[-1])
        except (ImportError, AttributeError) as error:
            warnings.warn(f'roundtrace missing target {target}: {error}', RuntimeWarning)
            continue
        wrapper = _wrapper(original, recorder, label, side, kind)
        setattr(owner, names[-1], wrapper)
        handle.originals.append((owner, names[-1], original, wrapper))
    if recorder.graph_replays:
        enable_graph_replays(handle)
    atexit.register(recorder.close)
    _active = handle
    return handle


def enable_graph_replays(handle):
    recorder = handle.recorder
    recorder.graph_replays = True
    if not recorder.device_enabled:
        return
    import torch
    owner = torch.cuda.CUDAGraph
    if any(cls is owner and name == 'replay' for cls, name, *_ in handle.originals):
        return
    original = owner.replay

    @functools.wraps(original)
    def replay(graph, *args, **kwargs):
        with recorder.graph_replay():
            return original(graph, *args, **kwargs)

    owner.replay = replay
    handle.originals.append((owner, 'replay', original, replay))

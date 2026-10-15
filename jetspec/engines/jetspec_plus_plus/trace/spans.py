"""In-memory host timestamps and nonblocking CUDA duration collection."""

# Locate the component from either its category or its stable forwarding path.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_TOOL_ROOT = next((p for p in _ToolPath(__file__).resolve().parents
                   if (p / "artifacts/manifest.json").is_file() or (p / "artifacts/manifest.json").is_file()), _ToolPath.cwd())
_tool_sys.path[:0] = [str(_TOOL_ROOT), str(_TOOL_ROOT / 'engine'), str(_ToolPath(__file__).resolve().parent)]
if __name__ == "__main__" and any(a in ("-h", "--help") for a in _tool_sys.argv[1:]):
    from jetspec.engines.jetspec_plus_plus.serving.entrypoints import cpu_help as _cpu_help
    _cpu_help(__file__)

from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time

DEFAULT_CUDA_EVENT_PAIRS = 32768  # covers the 17k draft spans in a 32x256 JetSpec++ read


class Recorder:
    def __init__(self, directory, rank, *, clock=time.perf_counter, event_factory=None,
                 device_enabled=None, cpu_clock=time.process_time, run_config=None):
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        self.run_config = run_config
        self.path = Path(directory) / f'rounds-{rank}.jsonl'
        self.rank = rank
        self.clock = clock
        self.cpu_clock = cpu_clock
        self.event_factory = event_factory
        self.device_enabled = (setting(self, 'engine.engine_settings.roundtrace_device', '0') == '1'
                               if device_enabled is None else device_enabled)
        self.graph_replays = False
        self.event_pool = []
        self.available_events = []
        self.records = []
        self.pending = []
        self.local = threading.local()
        self.serial = 0
        self.round_serial = 0
        self.written = False
        self.closed = False
        self.kernel_profile = setting(self, 'engine.engine_settings.w446_kernel_profile') == '1'
        self.kernel_profile_calls = 0
        self.kernel_profiler = None

    def _stack(self):
        if not hasattr(self.local, 'stack'):
            self.local.stack = []
        return self.local.stack

    def begin(self, label, side, kind='host'):
        if side not in ('verify', 'draft'):
            raise ValueError('side must be verify or draft')
        if self.kernel_profile and side == 'verify' and label == 'verify_forward':
            self.kernel_profile_calls += 1
            if self.kernel_profile_calls == 40:
                import torch
                self.kernel_profiler = torch.profiler.profile(activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA])
                self.kernel_profiler.__enter__()
                print(f'KERNEL_PROFILE_START rank={self.rank}', flush=True)
        stack = self._stack()
        row = dict(type='span', id=self.serial, label=label, side=side, kind=kind,
                   rank=self.rank, thread=threading.get_ident(),
                   round=getattr(self.local, 'round', None),
                   parent=stack[-1]['id'] if stack else None, start=self.clock())
        if self.kernel_profile and side == 'verify' and label == 'verify_forward':
            row['_profile_number'] = self.kernel_profile_calls
        self.serial += 1
        stack.append(row)
        if self.kernel_profiler is not None:
            import torch
            scope = torch.profiler.record_function(f'serving/{side}/{label}')
            scope.__enter__()
            row['_profile_scope'] = scope
        return row

    def end(self, row, *, append=True):
        row['stop'] = self.clock()
        active = self._stack().pop()
        if active is not row:
            raise RuntimeError('span nesting mismatch')
        scope = row.pop('_profile_scope', None)
        if scope is not None:
            scope.__exit__(None, None, None)
        if (self.kernel_profiler is not None and row['side'] == 'verify'
                and row['label'] == 'verify_forward'
                and row.pop('_profile_number', None) == 71):
            self.kernel_profiler.__exit__(None, None, None)
            output = self.path.with_name(f'kernels-{self.rank}.json')
            output.parent.mkdir(parents=True, exist_ok=True)
            self.kernel_profiler.export_chrome_trace(str(output))
            if not output.is_file():
                raise RuntimeError(f'kernel profile was not exported: {output}')
            print(f'KERNEL_PROFILE_COMPLETE {output}', flush=True)
            self.kernel_profiler = None
            self.kernel_profile = False
        row.pop('_profile_number', None)
        if append:
            self.records.append(row)

    @contextmanager
    def span(self, label, side):
        row = self.begin(label, side)
        try:
            yield row
        finally:
            self.end(row)

    @contextmanager
    def round(self, side, **metadata):
        previous = getattr(self.local, 'round', None)
        previous_side = getattr(self.local, 'round_side', None)
        if previous is not None and previous_side == side:
            yield previous
            return
        ident = self.round_serial
        self.round_serial += 1
        self.local.round = ident
        self.local.round_side = side
        start = self.clock()
        cpu_start = self.cpu_clock()
        try:
            yield ident
        finally:
            stop = self.clock()
            row = dict(type='round', id=ident, side=side,
                       rank=self.rank, start=start, stop=stop,
                       cpu_ms=(self.cpu_clock() - cpu_start) * 1000, **metadata)
            self.records.append(row)
            self.local.last_round = row
            self.local.round = previous
            self.local.round_side = previous_side

    @contextmanager
    def dual_span(self, label, side, *, stream=None):
        """Record unchanged host bounds plus one preallocated CUDA event pair.

        CUDA elapsed time is a duration on one stream, not a host timestamp.  Event
        pairs are recycled only after nonblocking completion queries succeed.
        """
        if not self.device_enabled:
            with self.span(label, side) as row:
                yield row
            return
        a, b = self._acquire_event_pair()
        if stream is None:
            import torch
            stream = torch.cuda.current_stream()
        device = getattr(stream, 'device', 'cuda')
        device_index = getattr(device, 'index', None)
        device_name = f'cuda:{device_index}' if isinstance(device_index, int) else str(device)
        stream_id = getattr(stream, 'cuda_stream', 'unknown')
        row = self.begin(label, side)
        row.update(cuda_stream=f'{device_name}:{stream_id}', cuda_status='PENDING')
        a.record(stream)
        try:
            yield row
        finally:
            b.record(stream)
            self.end(row)
            self.pending.append((row, a, b))

    cuda_span = dual_span



    def _acquire_event_pair(self):
        if not self.event_pool:
            raise RuntimeError('CUDA event pool not preallocated')
        if not self.available_events:
            self.resolve(wait=False)
        if not self.available_events:
            raise RuntimeError('CUDA event pool exhausted with every pair still in flight')
        return self.available_events.pop()

    def resolve(self, *, wait=False):
        ready, waiting = [], []
        for row, a, b in self.pending:
            if wait:
                a.synchronize()
                b.synchronize()
            if wait or (a.query() and b.query()):
                row.update(cuda_ms=a.elapsed_time(b),
                           cuda_status='RESOLVED_AT_TEARDOWN' if wait else 'RESOLVED')
                ready.append(row)
                self.available_events.append((a, b))
            else:
                waiting.append((row, a, b))
        self.pending = waiting
        return len(ready)

    def flush(self):
        if not self.records:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Span and round IDs are process-local serials, so a file holds exactly one
        # recorder's run: the first write creates it and fails on an existing file.
        with self.path.open('a' if self.written else 'x') as stream:
            stream.writelines(json.dumps(row, separators=(',', ':')) + '\n'
                              for row in self.records)
        self.written = True
        self.records.clear()

    def close(self):
        if self.closed:
            return
        # Never add a teardown synchronization to the served path.  A CUDA
        # stream may be parked behind work owned by another rank; waiting here
        # can deadlock process shutdown.  Completed event pairs retain exact
        # device durations, while the report exposes the remainder explicitly.
        self.resolve(wait=False)
        for row, a, b in self.pending:
            row['cuda_status'] = 'NOT_READY'
        self.pending.clear()
        self.flush()
        self.closed = True

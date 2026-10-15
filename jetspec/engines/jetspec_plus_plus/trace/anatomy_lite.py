"""Round observation with delayed, query-only CUDA event collection.

Rows: seq, round, start, intake start/end, verify start/end, outcome publish
start/end, round end (CLOCK_MONOTONIC ns), replay-to-logits GPU milliseconds.
Version 2 appends named monotonic target-host stamps. Compression happens after
generation, outside the timed serving loop. No host stamp waits for device work.
"""
import base64
import json
from time import monotonic_ns
import zlib


def enabled(run_config=None, *, engine=None):
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    engine = current_engine() if engine is None else engine
    return (engine.record == 'anatomy'
            or setting(run_config, 'engine.engine_settings.anatomy_lite', '0') != '0')


def target_host_stamps(jetspec, run_config=None, *, engine=None):
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    return jetspec and ((engine if engine is not None else current_engine()).kind == 'reference'
                       or setting(run_config, 'engine.engine_settings.device_host_stamps', '0') == '1')


def pack(rows):
    raw = json.dumps(rows, separators=(',', ':')).encode()
    schema = 'jetspec.anatomy_lite/2' if rows and len(rows[0]) == 12 else 'jetspec.anatomy_lite/1'
    return dict(schema=schema, clock='CLOCK_MONOTONIC',
                rounds=len(rows), rows_zlib=base64.b64encode(zlib.compress(raw)).decode())




class _DeviceVerifyTimes(list):
    """Observe the existing native metric without wrapping the verifier.

    perf_counter and monotonic_ns share CLOCK_MONOTONIC. The append clock is
    the end boundary; recover its start from the unchanged elapsed duration.
    The append-site clock lag shifts both boundaries equally, preserving the
    native interval and exposing entry/tail work in the surrounding call.
    """
    def __init__(self, values, timing):
        super().__init__(values)
        self.timing = timing

    def __deepcopy__(self, memo):
        # Request snapshots contain float durations, never the live CUDA observer.
        copied = list(self)
        memo[id(self)] = copied
        return copied

    def append(self, duration):
        timing = self.timing
        if timing.active is not None:
            ended = timing.clock()
            timing.active[11].update(device_verify_start=ended-round(duration*1e9),
                                     device_verify_end=ended)
        super().append(duration)


class RoundTiming:
    def __init__(self, event_factory=None, clock=monotonic_ns, stream=None, host_stamps=True):
        if event_factory is None:
            import torch
            event_factory = lambda: torch.cuda.Event(enable_timing=True)
            if stream is None:
                stream = torch.cuda.current_stream()
        # Target replay and logits use this rank's serving stream throughout.
        self.stream = stream
        self.pool = [(event_factory(), event_factory()) for _ in range(4)]
        self.clock = clock
        self.pending = []
        self.rows = []
        self.active = None
        self.events = None
        self.previous = None
        self.host_stamps = host_stamps

    def observe_device_verify(self, metrics):
        if self.host_stamps:
            metrics['target_verify_times'] = _DeviceVerifyTimes(metrics['target_verify_times'], self)

    def drain(self):
        remaining = []
        for row, events in self.pending:
            if events[1].query():
                row[10] = round(events[0].elapsed_time(events[1]), 6)
                self.pool.append(events)
            else:
                remaining.append((row, events))
        self.pending = remaining

    def begin(self, seq_id, round_id, start=None):
        self.drain()
        self.active = [seq_id, round_id, self.clock() if start is None else start,
                       0, 0, 0, 0, 0, 0, 0, None]
        if self.host_stamps:
            self.active.append({})
        if self.host_stamps and self.previous is not None and self.previous[0] == seq_id:
            self.previous[11]['next_start'] = self.active[2]
        self.rows.append(self.active)

    def mark(self, index):
        self.active[index] = self.clock()

    def stamp(self, name):
        if self.host_stamps and self.active is not None:
            self.active[11][name] = self.clock()

    def close_step(self):
        if self.host_stamps and self.previous is not None:
            self.previous[11]['step_return'] = self.clock()


    def gpu_begin(self):
        self.stamp('replay_start')
        if self.active is not None and self.pool:
            self.events = self.pool.pop()
            self.events[0].record(self.stream)

    def gpu_end(self):
        self.stamp('logits_end')
        if self.active is not None and self.events is not None:
            self.events[1].record(self.stream)

    def finish(self):
        self.mark(9)
        self.previous = self.active
        if self.events is not None:
            self.pending.append((self.active, self.events))
        self.active = self.events = None


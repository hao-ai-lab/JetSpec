"""Diagnostic per-kernel device time of sampled served chain rounds (JETSPEC_CHAIN_KERNEL_PROFILE).

A sampled round runs its segment replays under ``torch.profiler`` with one
``CHAIN_SEG_<name>`` annotation per replay. After the round's publish sync the
trace is exported and every kernel launched by a graph replay inside an
annotation is written as ``[name, device_us]``, one JSON line per round. Only
sampled rounds pay the profiler and export cost; the default is off.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import tempfile

import torch

PREFIX = "CHAIN_SEG_"


def segment_kernels(events, names):
    """Map each annotated segment to the kernels its graph launches ran, in order."""
    spans = {}
    for event in events:
        if (event.get("ph") == "X" and event.get("cat") == "user_annotation"
                and str(event.get("name", "")).startswith(PREFIX)):
            spans.setdefault(event["name"][len(PREFIX):], event)
    by_correlation = defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            by_correlation[event.get("args", {}).get("correlation")].append(event)
    result = {}
    for name in names:
        span = spans.get(name)
        if span is None:
            continue
        launches = [event for event in events
                    if event.get("ph") == "X" and event.get("cat") in ("cuda_runtime", "cuda_driver")
                    and event.get("tid") == span.get("tid")
                    and span["ts"] <= event["ts"] <= span["ts"] + span["dur"]
                    and "GraphLaunch" in event.get("name", "")]
        kernels = sorted((kernel for launch in launches
                          for kernel in by_correlation.get(launch.get("args", {}).get("correlation"), ())),
                         key=lambda kernel: kernel["ts"])
        result[name] = [[kernel["name"][:160], round(float(kernel.get("dur", 0.0)), 3)]
                        for kernel in kernels]
    return result


class ChainKernelProfile:
    def __init__(self, path=None, every=None):
        path = os.environ.get("JETSPEC_CHAIN_KERNEL_PROFILE") if path is None else path
        self.path = Path(path) if path else None
        self.every = int(os.environ.get("JETSPEC_CHAIN_KERNEL_PROFILE_EVERY", "20")
                         if every is None else every)
        if self.every < 1:
            raise ValueError("JETSPEC_CHAIN_KERNEL_PROFILE_EVERY must be positive")
        self.calls = 0
        self.active = None

    @property
    def enabled(self):
        return self.path is not None

    def begin(self):
        """Start the profiler if this round is sampled; returns whether it did."""
        if self.path is None:
            return False
        self.calls += 1
        if (self.calls - 1) % self.every:
            return False
        self.active = torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
        self.active.__enter__()
        return True

    def segment(self, name):
        if self.active is None:
            return nullcontext()
        return torch.profiler.record_function(PREFIX + name)

    def finish(self, names, **meta):
        """Stop after the round's sync and append one line; no-op when not sampled."""
        if self.active is None:
            return
        profile, self.active = self.active, None
        profile.__exit__(None, None, None)
        with tempfile.TemporaryDirectory(prefix="chain-kprofile-") as directory:
            trace = Path(directory) / "trace.json"
            profile.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text())["traceEvents"]
        line = dict(meta, segments=segment_kernels(events, names))
        with self.path.open("a") as sink:
            sink.write(json.dumps(line) + "\n")

"""Sampled verify-rank CPU/CUDA trace, from graph replay through packet sync."""
from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import tempfile

import torch

PREFIX = "VERIFY_SEG_"
DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
LAUNCH_CATS = {"cuda_runtime", "cuda_driver"}
SOURCE_SITES = {
    "replay": "jetspec.engines.jetspec_plus_plus.verify.forward:run_jetspec_mixed_device:graph.replay",
    "logits": "jetspec.engines.jetspec_plus_plus.verify.forward:run_jetspec_mixed_device:compute_logits",
    "accept": "jetspec.engines.jetspec_plus_plus.verify.forward:_verify_jetspec_device:greedy_accept_device",
    "stage_ids": "jetspec.engines.jetspec_plus_plus.verify.forward:_verify_jetspec_device:stage_committed_ids",
    "stage_taps_first": "jetspec.engines.jetspec_plus_plus.verify.forward:_verify_jetspec_device:stage_committed_taps:first",
    "stage_taps_second": "jetspec.engines.jetspec_plus_plus.verify.forward:_verify_jetspec_device:stage_committed_taps:second",
    "packet_head": "jetspec.engines.jetspec_plus_plus.verify.forward:_verify_jetspec_device:host_words.to",
    "outcome_writes": "jetspec.engines.jetspec_plus_plus.verify.forward:_verify_jetspec_device:outcome.copy_",
    "recovery_id": "jetspec.engines.jetspec_plus_plus.verify.forward:_verify_jetspec_device:stage_recovery_id",
    "hash": "jetspec.engines.jetspec_plus_plus.verify.forward:_verify_jetspec_device:hash16_committed",
    "eos": "jetspec.engines.jetspec_plus_plus.verify.forward:_verify_jetspec_device:eos_and_limit",
    "packet_mirror": "jetspec.engines.jetspec_plus_plus.verify.forward:_verify_jetspec_device:host_words.copy_",
    "presync_pack": "jetspec.engines.jetspec_plus_plus.verify.multirow:SpeculatorAsync.pack_device_outcome_presync",
    "publish_head": "jetspec.engines.jetspec_plus_plus.verify.multirow:SpeculatorAsync.publish_device_outcome",
}


def _active_range(spans, timestamp):
    matches = [(name, span) for name, span in spans
               if span["ts"] <= timestamp <= span["ts"] + span["dur"]]
    return min(matches, key=lambda pair: pair[1]["dur"])[0] if matches else None


def summarize_trace(events, meta=None):
    """Account for device events correlated to launches in named host ranges.

    Chrome trace timestamps share a clock. Graph child kernels use the graph
    launch's correlation ID, as in the served chain profiler.
    """
    spans = [(str(e["name"])[len(PREFIX):], e) for e in events
             if e.get("ph") == "X" and e.get("cat") == "user_annotation"
             and str(e.get("name", "")).startswith(PREFIX)]
    replay = next((s for name, s in spans if name == "replay"), None)
    sync = next((s for name, s in reversed(spans) if name == "sync"), None)
    if replay is None or sync is None:
        raise ValueError("verify trace lacks replay or sync range")
    start, end = replay["ts"], sync["ts"] + sync["dur"]
    trace_end = max(span["ts"] + span["dur"] for _, span in spans)
    # The mirror event is recorded before tap staging and pre-sync packing.
    # Those later kernels may overlap the host's sync but are after its event
    # on the verify stream, hence outside the measured device interval v.
    mirror = next((s for name, s in spans if name == "packet_mirror"), None)
    v_launch_end = mirror["ts"] + mirror["dur"] if mirror is not None else end
    launches = {}
    for e in events:
        if (e.get("ph") == "X" and e.get("cat") in LAUNCH_CATS
                and start <= e.get("ts", -1) <= trace_end):
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                launches[corr] = e
    kernels = []
    for e in events:
        if e.get("ph") != "X" or e.get("cat") not in DEVICE_CATS:
            continue
        launch = launches.get(e.get("args", {}).get("correlation"))
        if launch is None:
            continue
        name = _active_range(spans, launch["ts"])
        kernels.append({"name": str(e.get("name", ""))[:160],
                        "range": name, "launch_site": launch.get("name"),
                        "source_site": ("jetspec.engines.jetspec_plus_plus.verify.multirow:_queue_jetspec_request:dist.isend"
                                        if name and name.startswith("send_")
                                        else SOURCE_SITES.get(name)),
                        "launch_us": launch["ts"], "start_us": e["ts"],
                        "device_us": float(e.get("dur", 0)),
                        "inside_v": launch["ts"] <= v_launch_end})
    kernels.sort(key=lambda k: k["start_us"])
    ranges = defaultdict(lambda: {"host_us": 0.0, "device_us": 0.0, "kernels": 0})
    for name, span in spans:
        ranges[name]["host_us"] += float(span["dur"])
    for kernel in kernels:
        row = ranges[kernel["range"] or "unattributed"]
        row["device_us"] += kernel["device_us"]
        row["kernels"] += 1
    v_kernels = [kernel for kernel in kernels if kernel["inside_v"]]
    post_v_ranges = defaultdict(lambda: {"device_us": 0.0, "kernels": 0})
    for kernel in kernels:
        if not kernel["inside_v"]:
            post_row = post_v_ranges[kernel["range"] or "unattributed"]
            post_row["device_us"] += kernel["device_us"]
            post_row["kernels"] += 1
    v_ranges = defaultdict(lambda: {"device_us": 0.0, "kernels": 0})
    for kernel in v_kernels:
        range_row = v_ranges[kernel["range"] or "unattributed"]
        range_row["device_us"] += kernel["device_us"]
        range_row["kernels"] += 1
    gaps = []
    busy = 0.0
    device_end = None
    for kernel in v_kernels:
        k_start = kernel["start_us"]
        k_end = k_start + kernel["device_us"]
        if device_end is not None and k_start > device_end:
            midpoint = (device_end + k_start) / 2
            gaps.append({"start_us": device_end, "duration_us": k_start - device_end,
                         "host_range": _active_range(spans, midpoint)})
        busy += max(0, k_end - max(k_start, device_end if device_end is not None else k_start))
        device_end = max(device_end if device_end is not None else k_end, k_end)
    first = v_kernels[0]["start_us"] if v_kernels else None
    row = dict(meta or {}, ranges=dict(ranges), v_ranges=dict(v_ranges),
               post_v_ranges=dict(post_v_ranges),
               kernels=kernels, idle_gaps=gaps,
               v_host_us=float(end - start),
               post_mirror_kernels=len(kernels) - len(v_kernels),
               device_window_us=(device_end - first) if first is not None else 0.0,
               device_busy_us=busy,
               device_idle_us=sum(g["duration_us"] for g in gaps),
               start_lag_us=(first - start) if first is not None else None,
               tail_lag_us=(end - device_end) if device_end is not None else None)
    return row


class VerifyKernelProfile:
    def __init__(self, path, every=20):
        self.path = Path(path)
        self.every = int(every)
        if self.every < 1:
            raise ValueError("JETSPEC_VERIFY_KERNEL_PROFILE_EVERY must be positive")
        self.calls = 0
        self.active = None
        self.marker = None

    @classmethod
    def from_env(cls, *, run_config=None):
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        path = setting(run_config, 'engine.engine_settings.verify_kernel_profile')
        if not path:
            return None
        return cls(path, setting(run_config, 'engine.engine_settings.verify_kernel_profile_every', '20'))

    def begin(self):
        self.calls += 1
        if self.calls % self.every:
            return False
        self.active = torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
        self.active.__enter__()
        return True

    def step(self, name):
        if self.active is None:
            return
        if self.marker is not None:
            self.marker.__exit__(None, None, None)
        self.marker = torch.profiler.record_function(PREFIX + name)
        self.marker.__enter__()

    def finish(self, **meta):
        if self.active is None:
            return
        if self.marker is not None:
            self.marker.__exit__(None, None, None)
            self.marker = None
        profile, self.active = self.active, None
        profile.__exit__(None, None, None)
        with tempfile.TemporaryDirectory(prefix="verify-kprofile-") as directory:
            trace = Path(directory) / "trace.json"
            profile.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text())["traceEvents"]
        row = summarize_trace(events, meta)
        with self.path.open("a") as sink:
            sink.write(json.dumps(row) + "\n")

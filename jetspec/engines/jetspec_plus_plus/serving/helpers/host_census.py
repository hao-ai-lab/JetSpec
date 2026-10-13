"""Opt-in host stamps for the N=1 device-verify round (JETSPEC_VERIFY_HOST_CENSUS=1).

Each ``mark(name)`` records a wall stamp; the segment ``name`` is the host time since
the previous stamp of the same step.  ``flush`` runs at the end of every engine step
and appends one row per device-verify decode round to
``metrics["verify_host_census"]``, aligned 1:1 with ``jetspec_round_records``:

- one key per segment (ms; a name marked twice in a round is summed);
- ``loop``: from the previous step's end to this step's start (outside the round);
- ``gap``: from the previous round's packet sync to this round's graph replay call,
  the verify stream's idle window when the previous step was a decode round;
- ``cw_store`` / ``cw_decode`` / ``cw_wake`` / ``cw_consume``: the candidate wait split by
  the receiver thread's stamps (``note_candidate``): payload fetched from the store after
  the wait began, thread-side decode, main-thread wake-up after the payload was ready,
  and the mailbox bookkeeping after it (the last packet taken before ``cand_wait``);
  ``cw_arrival`` is the signed fetch time relative to the wait's start (negative: early)
  and ``cw_decode_all`` the whole thread-side decode.

Off (the default), ``mark`` returns at once and ``flush`` does nothing.
"""
from __future__ import annotations

import os
from time import perf_counter

_enabled = os.environ.get("JETSPEC_VERIFY_HOST_CENSUS", "0") == "1"
_stamps: list[tuple[str, float]] = []
_notes: list[tuple[float, float, float]] = []
_state = {"last_end": None, "last_sync": None}


def enabled() -> bool:
    return _enabled




def mark(name: str) -> None:
    if _enabled:
        _stamps.append((name, perf_counter()))


def note_candidate(message) -> None:
    """Keep the receiver thread's stamps of the packet the main thread just took."""
    stamps = getattr(message, "census_stamps", None) if _enabled else None
    if stamps is not None:
        _notes.append((stamps[0], stamps[1], perf_counter()))


def flush(metrics: dict) -> None:
    """Close one engine step; keep a row only for a device-verify decode round."""
    if not _enabled:
        return
    ended = perf_counter()
    stamps = list(_stamps)
    _stamps.clear()
    notes = list(_notes)
    _notes.clear()
    last_end, last_sync = _state["last_end"], _state["last_sync"]
    _state["last_end"] = ended
    names = [name for name, _ in stamps]
    if "sync" not in names or not stamps or stamps[0][0] != "step_start":
        _state["last_sync"] = None
        return
    row = {}
    for (_, before), (name, at) in zip(stamps, stamps[1:]):
        row[name] = row.get(name, 0.0) + (at - before) * 1000
    row["step_end"] = (ended - stamps[-1][1]) * 1000
    if last_end is not None:
        row["loop"] = (stamps[0][1] - last_end) * 1000
    times = dict(stamps)
    if last_sync is not None and "plan" in times:
        row["gap"] = (times["plan"] - last_sync) * 1000
    _state["last_sync"] = times["sync"]
    taken = [note for note in notes if "cand_wait" in times and note[2] <= times["cand_wait"]]
    if taken and "request" in times:
        got, decoded, took = taken[-1]
        began = times["request"]
        ready = max(began, decoded)
        row["cw_store"] = max(0.0, got - began) * 1000
        row["cw_decode"] = max(0.0, decoded - max(began, got)) * 1000
        row["cw_wake"] = max(0.0, took - ready) * 1000
        row["cw_consume"] = (times["cand_wait"] - max(ready, took)) * 1000
        row["cw_arrival"] = (got - began) * 1000       # signed: < 0 means the payload was early
        row["cw_decode_all"] = (decoded - got) * 1000
    records = metrics.get("jetspec_round_records") or []
    if records:
        row["seq_id"] = int(records[-1]["seq_id"])
        row["round"] = int(records[-1]["round"])
    metrics.setdefault("verify_host_census", []).append(row)

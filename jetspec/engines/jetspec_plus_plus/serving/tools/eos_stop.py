"""Recompute EOS-stopped serving metrics from saved read files, without a model."""

# Resolve the repository independently of the tool's category or campaign depth.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_TOOL_ROOT = next((p for p in _ToolPath(__file__).resolve().parents
                   if (p / "artifacts/manifest.json").is_file() or (p / "serve/README.md").is_file()), _ToolPath.cwd())
_tool_sys.path[:0] = [str(_TOOL_ROOT), str(_TOOL_ROOT / "jetspec.engines.jetspec_plus_plus.serving.tools"), str(_ToolPath(__file__).resolve().parent),
                      str(_TOOL_ROOT / "predictor/reader")]
if __name__ == "__main__" and any(a in ("-h", "--help") for a in _tool_sys.argv[1:]):
    from jetspec.engines.jetspec_plus_plus.serving.tools.lib.entrypoints import cpu_help as _cpu_help
    _cpu_help(__file__)


import argparse
import csv
import json
from pathlib import Path


ROUND_METHODS = {"blockhead", "dflash", "jetspecpp", "jetspec_sync", "spec_sync"}
FIELDS = ("read", "method", "tau_to_cap", "tau_eos_stopped", "tps_to_cap",
          "tps_eos_stopped", "decode_tokens_eos_stopped", "rounds_eos_stopped",
          "hit_rate_eos_stopped", "tau_hit_eos_stopped", "tau_miss_eos_stopped",
          "cutoff_rate_eos_stopped", "error")


def round_is_miss(record, population):
    """A miss can emit a correction without taking the redraft fallback path."""
    return bool(population.get("fallback") or record.get("decision") == "miss")


def _need(method, obj, fields, source):
    missing = [name for name in fields if name not in obj or obj[name] is None]
    if missing:
        raise ValueError(f"{method}: missing {source}: {', '.join(missing)}")


def stop_lengths(outputs, eos_token_ids, *, tokens):
    """Count tokens through the first target EOS, subject to the served cap."""
    if not eos_token_ids:
        raise ValueError("missing target_eos_token_ids (use --eos-token-ids for old campaigns)")
    if not isinstance(tokens, int) or tokens < 1:
        raise ValueError("missing positive token budget")
    eos = set(map(int, eos_token_ids))
    stops = []
    for index, row in enumerate(outputs):
        _need("outputs", row, ("token_ids",), f"outputs[{index}]")
        ids = row["token_ids"][:tokens]
        first = next((i for i, token in enumerate(ids) if token in eos), None)
        stops.append(len(ids) if first is None else first + 1)
    return stops




def compute_eos_stopped(method, outputs, metrics, *, tokens, eos_token_ids,
                        populations=(), batch=1, prompt_indices=None):
    """Count decode tokens and complete rounds up to EOS for one saved read.

    The first output token is produced by prefill. It is in the EOS search but
    excluded from decode tokens and round timing, matching the served receipt.
    """
    if method not in ROUND_METHODS | {"ar"}:
        raise ValueError(f"{method}: unsupported EOS-stopped method")
    stops = stop_lengths(outputs, eos_token_ids, tokens=tokens)
    native_indices = (list(range(len(outputs))) if prompt_indices is None
                      else list(map(int, prompt_indices)))
    if len(native_indices) != len(outputs) or len(set(native_indices)) != len(outputs):
        raise ValueError(f"{method}: prompt indices must identify each output once")
    positions = {index: position for position, index in enumerate(native_indices)}
    kept = []  # (record, credited tokens, complete round milliseconds)
    if method == "ar":
        if batch != 1:
            # At batch > 1 one target step decodes a token for every live request: no per-token time.
            for stop in stops:
                kept.extend((None, 1, None) for _ in range(max(0, stop - 1)))
        else:
            _need(method, metrics, ("target_step_times",), "metrics")
            steps = metrics["target_step_times"]
            if len(steps) != sum(len(row["token_ids"]) for row in outputs):
                raise ValueError(f"{method}: target_step_times cannot be aligned to outputs")
            offset = 0
            for row, stop in zip(outputs, stops):
                length = len(row["token_ids"])
                kept.extend((None, 1, float(step) * 1000)
                            for step in steps[offset + 1:offset + stop])
                offset += length
    else:
        source = ("jetspec_round_records" if method == "jetspecpp" else
                  "blockhead_round_records")
        if method == "spec_sync" and not metrics.get(source):
            records = _spec_sync_records(outputs, metrics, batch=batch)
        else:
            _need(method, metrics, (source,), "metrics")
            records = metrics[source]
        if not records and any(stop > 1 for stop in stops):
            raise ValueError(f"{method}: empty {source} for nonempty decode")
        for record in records:
            _need(method, record, ("prompt_index",), source)
            index = int(record["prompt_index"])
            if index not in positions:
                raise ValueError(f"{method}: invalid prompt_index {index}")
        step_ms = None
        if method == "jetspecpp" and batch == 1:
            _need(method, metrics, ("target_step_times",), "metrics")
            if len(metrics["target_step_times"]) != len(records) + len(outputs):
                raise ValueError(f"{method}: target_step_times cannot be aligned to {source}")
            indices = [positions[int(record["prompt_index"])] for record in records]
            if indices != sorted(indices):
                raise ValueError(f"{method}: interleaved {source} cannot align step times")
            step_ms = []
            cursor = 0
            for index in native_indices:
                cursor += 1  # prefill/admission step
                for record in records:
                    if int(record["prompt_index"]) == index:
                        step_ms.append(float(metrics["target_step_times"][cursor]) * 1000)
                        cursor += 1
        progress = [0] * len(outputs)
        for pos, record in enumerate(records):
            index = positions[int(record["prompt_index"])]
            if method == "jetspecpp":
                _need(method, record, ("accepted_len", "seq_id", "candidate_round"), source)
                before = progress[index]
                accepted = int(record["accepted_len"])
                progress[index] += accepted
                # At batch > 1 one target step serves many requests: no per-round time.
                duration = step_ms[pos] if step_ms is not None else None
            else:
                _need(method, record, ("completion_before", "accepted", "round_ms"), source)
                before = int(record["completion_before"])
                accepted = int(record["accepted"])
                duration = float(record["round_ms"])
            left = stops[index] - 1 - before
            if left <= 0:
                continue
            if accepted < 1 or (duration is not None and duration <= 0):
                raise ValueError(f"{method}: invalid accepted tokens or round_ms")
            credited = min(accepted, left)
            if credited:
                kept.append((record, credited, duration))
    expected = sum(max(0, stop - 1) for stop in stops)
    count = sum(tokens_kept for _, tokens_kept, _ in kept)
    if count != expected:
        raise ValueError(f"{method}: round records credit {count} decode tokens; expected {expected}")
    rounds = len(kept)
    timed = all(duration is not None for _, _, duration in kept)
    elapsed_ms = sum(duration for _, _, duration in kept) if timed else None
    result = {"decode_tokens_eos_stopped": count, "rounds_eos_stopped": rounds,
              "round_ms_eos_stopped": elapsed_ms,
              "tau_eos_stopped": count / rounds if rounds else None,
              "tps_eos_stopped": count * 1000 / elapsed_ms if elapsed_ms else None}
    if not timed:
        result["timing_eos_stopped"] = ("NOT_MEASURED: batch > 1 target steps are shared "
                                        "across requests")
    if method == "jetspecpp":
        if not populations:
            raise ValueError(f"{method}: missing populations for hit/cutoff metrics")
        population_index = {(row["seq_id"], row["candidate_round"]): row
                            for row in populations}
        if len(population_index) != len(populations):
            raise ValueError(f"{method}: duplicate population round key")
        hit_tokens = miss_tokens = hits = misses = cutoffs = 0
        for record, credited, _ in kept:
            key = (record["seq_id"], record["candidate_round"])
            row = population_index.get(key)
            if row is None or row.get("join_status") not in ("admission", "joined"):
                raise ValueError(f"{method}: missing joined/admission population for {key}")
            if not round_is_miss(record, row):
                hits += 1
                hit_tokens += credited
            else:
                misses += 1
                miss_tokens += credited
            deadline = (row or {}).get("deadline") or {}
            cutoffs += bool(deadline.get("truncated") or deadline.get("deadline_miss"))
        result.update(hit_rate_eos_stopped=hits / rounds if rounds else None,
                      tau_hit_eos_stopped=hit_tokens / hits if hits else None,
                      tau_miss_eos_stopped=miss_tokens / misses if misses else None,
                      cutoff_rate_eos_stopped=cutoffs / rounds if rounds else None)
    return result








def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="campaign JSON files or read directories")
    parser.add_argument("--eos-token-ids", help="comma-separated target EOS IDs for old campaigns")
    parser.add_argument("--csv", type=Path, help="write one row per read")
    args = parser.parse_args(argv)
    eos = [int(item) for item in args.eos_token_ids.split(",")] if args.eos_token_ids else None
    rows = [row for path in args.paths for row in rows_from_path(path, eos_token_ids=eos)]
    for row in rows:
        print(" ".join(f"{key}={row.get(key)}" for key in FIELDS))
    if args.csv:
        with args.csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({key: row.get(key) for key in FIELDS} for row in rows)
    return 1 if any(row["error"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())

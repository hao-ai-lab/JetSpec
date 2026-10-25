"""Build and collect cheap per-read serving anatomy.

Host measurements come from the existing serve_read counters. Device measurements
are optional and come only from an explicitly requested roundtrace report.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
from statistics import mean, median


CSV_FIELDS = ("method", "regime", "rounds", "hit_rate", "tau", "tau_hit",
              "tau_miss", "round_ms_mean",
              "round_ms_median", "round_ms_p90", "verify_ms_median",
              "verify_ms_p90", "tps", "hit_rounds", "miss_rounds",
              "acceptance_misses", "deadline_misses", "cutoff_rate",
              "published_depth", "cutoff_stages")


def percentile(values, q):
    values = sorted(float(v) for v in values)
    if not values:
        return None
    at = (len(values) - 1) * q
    low = int(at)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (at - low)


def summary(values):
    values = [float(v) for v in values if v is not None]
    return {"mean": mean(values), "median": median(values), "p90": percentile(values, .9)} if values else {
        "mean": None, "median": None, "p90": None}


def _cutoff_stage(deadline, refresh):
    if deadline.get("cutoff_stage"):
        return deadline["cutoff_stage"]
    index = deadline.get("stopped_before")
    if index == 1:
        return "refresh-1" if refresh in ("after_j0", "after_j0_j1") else "J1"
    if index == 2:
        return "refresh-2" if refresh == "after_j0_j1" else "J2"
    return f"J{index}" if index is not None else "unclassified"


def _running_stage(records, seq_id, candidate_round, published_ns):
    """Locate the draft host span running when the preceding verify ended."""
    if published_ns is None or candidate_round is None:
        return "unclassified"
    preceding = [r for r in records if r.get("type") == "round" and r.get("side") == "verify"
                 and [seq_id, candidate_round - 1] in r.get("sequences", [])]
    if len(preceding) != 1:
        return "unclassified"
    cutoff = preceding[0]["stop"]
    published = published_ns / 1e9
    if published <= cutoff:
        return "idle"
    draft = [r for r in records if r.get("type") == "round" and r.get("side") == "draft"
             and r["start"] <= published <= r["stop"]]
    if len(draft) != 1:
        return "unclassified"
    owner = draft[0]
    active = [r for r in records if r.get("type") == "span" and r.get("side") == "draft"
              and r.get("kind", "host") != "cuda" and r.get("rank") == owner["rank"]
              and r.get("round") == owner["id"] and r["start"] <= cutoff < r["stop"]]
    for row in sorted(active, key=lambda r: r["stop"] - r["start"]):
        label = row["label"]
        if label in ("adapter", "adapter_stage"):
            return "adapter"
        if label in ("anchor_decoder", "anchor_prefix", "anchor_window"):
            return "decoder"
        if label in ("row_assembly", "row_selection", "proposal"):
            return "rows"
        if label == "adapter_refresh":
            earlier = sum(r.get("label") == "adapter_refresh" and r.get("round") == owner["id"]
                          and r.get("rank") == owner["rank"] and r.get("stop", 0) <= cutoff
                          for r in records)
            return "refresh-2" if earlier else "refresh-1"
        if label == "draft_pass":
            return "J0"
        if label.startswith("refine_"):
            return "J" + label.split("_", 1)[1]
    return "unclassified"


def _stage(values, rounds, round_mean):
    times = summary(values)
    return {"mean_ms": times["mean"], "median_ms": times["median"],
            "p90_ms": times["p90"], "share_round":
            times["mean"] / round_mean if times["mean"] is not None and round_mean else None,
            "present_rounds": len(values), "rounds": rounds}


def _trace_samples(records):
    """Collect per-draft-round device durations, preserving each stage's samples."""
    names = {"adapter": "adapter", "anchor_prefix": "decoder",
             "row_assembly": "rows", "block_head": "block_head",
             "verify_forward": "target_verify",
             "draft_pass": "J0", "refine_1": "J1",
             "refine_2": "J2", "refine_3": "J3"}
    by_id = {(r.get("rank"), r.get("id")): r for r in records if r.get("type") == "span"}

    def stage(row):
        if row.get("label") != "bucket_pass":
            return names.get(row.get("label"))
        parent = row.get("parent")
        while parent is not None:
            ancestor = by_id.get((row.get("rank"), parent))
            if ancestor is None:
                break
            if ancestor.get("label") in ("draft_pass", "refine_1", "refine_2", "refine_3"):
                return names[ancestor["label"]]
            parent = ancestor.get("parent")
        return None

    by_round = defaultdict(lambda: defaultdict(float))
    refresh_device = defaultdict(list)
    refresh_host = defaultdict(list)
    for row in records:
        if row.get("type") != "span" or row.get("side") not in ("draft", "verify") or row.get("round") is None:
            continue
        key = (row["side"], row.get("rank"), row["round"])
        name = stage(row)
        if row.get("cuda_ms") is not None and name is not None:
            by_round[key][name] += float(row["cuda_ms"])
        elif row.get("label") == "adapter_refresh" and row.get("cuda_ms") is not None:
            refresh_device[key].append((row["start"], float(row["cuda_ms"])))
        elif row.get("label") == "adapter_refresh" and row.get("kind", "host") != "cuda":
            refresh_host[key].append((row["start"], (row["stop"] - row["start"]) * 1000))
    for key in refresh_device.keys() | refresh_host.keys():
        calls = refresh_device[key] or refresh_host[key]
        for index, (_, duration) in enumerate(sorted(calls)[:2], 1):
            by_round[key][f"refresh-{index}"] = duration
    values = defaultdict(list)
    for stages in by_round.values():
        for name, value in stages.items():
            values[name].append(value)
    return dict(values)


def build_anatomy(method, regime, measured, populations, *, refresh="off",
                  passes=None, rows_cap=None, trace_report=None, trace_records=None,
                  eos_stopped=None):
    """Return one fixed-schema record without touching model or serving state."""
    metrics = measured["metrics"]
    block = metrics.get("blockhead_round_records") or []
    jetspec = metrics.get("jetspec_round_records") or []
    steps = metrics.get("target_step_times") or []
    verify_seconds = metrics.get("target_verify_times") or []
    if block:
        round_values = [r["round_ms"] for r in block]
        verify_values = [r["verify_ms"] for r in block]
        accepted = [r["accepted"] for r in block]
    elif jetspec:
        # target_step_times has one admission/prefill entry before the measured
        # verifier records in the one-sequence serve_read path.
        step_aligned = len(steps) == len(jetspec) + 1
        round_values = ([1000 * v for v in steps[1:]] if step_aligned
                        else list(metrics.get("jetspec_verify_round_ms") or []))
        verify_values = list(metrics.get("jetspec_verify_round_ms") or [])
        accepted = [r["accepted_len"] for r in jetspec]
    else:
        # AR and stock methods expose per-step host wall but no keyed round log.
        # The first step is prefill; the rest are decode rounds.
        round_values = [1000 * v for v in steps[1:]]
        verify_values = [1000 * v for v in verify_seconds] or round_values
        accepted = []
    if not round_values:
        raise ValueError(f"no host round durations for {method}")
    rounds = len(round_values)
    round_stats = summary(round_values)
    result = {"schema": "jetspec.serve_anatomy/1", "method": method,
              "regime": regime, "rounds": rounds,
              "tau": mean(accepted) if accepted else measured["decode_tokens"] / rounds,
              "hit_rate": None, "tau_hit": None, "tau_miss": None,
              "round_ms": round_stats, "verify_ms": summary(verify_values),
              "tps": float(measured["decode_tok_s"]), "stages": {},
              "hit_rounds": None, "miss_rounds": None,
              "admission_rounds": None,
              "fallback_causes": {}, "cutoff_rate": None,
              "cutoff_stages": {}, "published_depth": {}, "by_depth": {},
              "verify_end_stages": {},
              "cutoff_stage_basis": None,
              "hit_round_ms": summary([]), "miss_round_ms": summary([]),
              "device_stages": {}, "host_trace_stages": {},
              "source": "existing_host_counters"}
    result["metric_population"] = {"tau": "to-cap", "tps": "to-cap"}
    result.update({key: None for key in (
        "decode_tokens_eos_stopped", "rounds_eos_stopped", "round_ms_eos_stopped",
        "tau_eos_stopped", "tps_eos_stopped", "hit_rate_eos_stopped",
        "tau_hit_eos_stopped", "tau_miss_eos_stopped", "cutoff_rate_eos_stopped")})
    if eos_stopped is not None:
        result.update(eos_stopped)
    result["config"] = {"passes": passes, "refresh": refresh, "rows_cap": rows_cap}
    result["round_source"] = ("blockhead_round_records" if block else
                              "target_step_times" if not jetspec or step_aligned else
                              "verify_only")
    if method in ("blockhead", "dflash", "jetspec_sync", "spec_sync"):
        result["tau_hit"] = result["tau"]
    if block:
        for name, key in (("block_head", "head_ms"), ("markov", "markov_ms"),
                          ("handoff", "residual_ms")):
            result["stages"][name] = _stage(
                [r[key] for r in block if r.get(key) is not None], rounds, round_stats["mean"])
        if any("domino_ms" in r for r in block):
            result["stages"]["domino"] = _stage(
                [r["domino_ms"] for r in block if r.get("domino_ms") is not None],
                rounds, round_stats["mean"])
    elif method == "ar":
        result["stages"]["target_decode"] = _stage(round_values, rounds, round_stats["mean"])
    else:
        result["stages"]["target_verify"] = _stage(verify_values, rounds, round_stats["mean"])
        if len(round_values) == len(verify_values):
            result["stages"]["other_host"] = _stage(
                [max(0., total - verify) for total, verify in zip(round_values, verify_values)],
                rounds, round_stats["mean"])
    if method == "jetspecpp":
        joined = [r for r in populations if r.get("join_status") == "joined"]
        if not joined:
            raise ValueError("JetSpec++ anatomy requires joined populations")
        record_index = {(r.get("seq_id"), r.get("candidate_round")): i
                        for i, r in enumerate(jetspec)}
        if len(record_index) != len(jetspec):
            raise ValueError("duplicate JetSpec++ round key")
        joined_keys = {(r["seq_id"], r["candidate_round"]) for r in joined}
        admission_indices = [i for key, i in record_index.items() if key not in joined_keys]
        by_depth = defaultdict(list)
        cutoffs, depths, causes, verify_end = Counter(), Counter(), Counter(), Counter()
        hit_ms = [round_values[i] for i in admission_indices if i < len(round_values)]
        miss_ms = []
        hit_lengths = [jetspec[i]["accepted_len"] for i in admission_indices]
        miss_lengths = []
        for row in joined:
            key = (row.get("seq_id"), row.get("candidate_round"))
            if key not in record_index:
                raise ValueError(f"unmatched JetSpec++ anatomy round: {key}")
            index = record_index[key]
            deadline = row.get("deadline") or {}
            depth = deadline.get("published_pass")
            pass_label = re.match(r"J[0-3]", str(row.get("pass")))
            depth = (f"J{int(depth) - 1}" if depth is not None and int(depth) > 0 else
                     pass_label.group() if row.get("deadline") is None and pass_label else "NONE")
            depth = depth or "NONE"
            depths[depth] += 1
            by_depth[depth].append(row)
            if deadline.get("truncated") or deadline.get("deadline_miss"):
                stage = (_cutoff_stage(deadline, refresh) if deadline.get("truncated") else
                         _running_stage(trace_records or [], row["seq_id"],
                                        row["candidate_round"], deadline.get("published_ns")))
                cutoffs[stage] += 1
            if trace_records is not None and deadline.get("published_ns") is not None:
                verify_end[_running_stage(trace_records, row["seq_id"],
                                          row["candidate_round"],
                                          deadline["published_ns"])] += 1
            (miss_lengths if row.get("fallback") else hit_lengths).append(
                jetspec[index]["accepted_len"])
            if index < len(round_values):
                (miss_ms if row.get("fallback") else hit_ms).append(round_values[index])
            if row.get("fallback"):
                causes["deadline" if deadline.get("deadline_miss") or
                       (row.get("deadline") is not None and depth == "NONE")
                       else "acceptance"] += 1
        result.update(hit_rounds=len(hit_lengths),
                      hit_rate=len(hit_lengths) / rounds,
                      tau_hit=mean(hit_lengths) if hit_lengths else None,
                      tau_miss=mean(miss_lengths) if miss_lengths else None,
                      miss_rounds=sum(bool(r.get("fallback")) for r in joined),
                      admission_rounds=len(admission_indices),
                      fallback_causes=dict(causes),
                      cutoff_rate=sum(cutoffs.values()) / rounds,
                      cutoff_stages=dict(cutoffs),
                      cutoff_stage_basis="deadline decision before next stage; deadline misses use active trace span",
                      verify_end_stages=dict(verify_end), published_depth=dict(depths),
                      by_depth={depth: {"rounds": len(rows),
                                        "hit_rate": sum(not r.get("fallback") for r in rows) / len(rows),
                                        "tau": mean(r["accepted_length"] for r in rows),
                                        "hit_tau": (mean(r["accepted_length"] for r in rows
                                                         if not r.get("fallback"))
                                                    if any(not r.get("fallback") for r in rows) else None),
                                        "miss_tau": (mean(r["accepted_length"] for r in rows
                                                          if r.get("fallback"))
                                                     if any(r.get("fallback") for r in rows) else None)}
                                for depth, rows in by_depth.items()},
                      hit_round_ms=summary(hit_ms), miss_round_ms=summary(miss_ms))
    stage_samples = _trace_samples(trace_records or []) if trace_records is not None else {}
    refresh_device = any(r.get("label") == "adapter_refresh" and r.get("cuda_ms") is not None
                         for r in (trace_records or []))
    result["device_stages"] = {
        name: {"median_ms": median(values), "p90_ms": percentile(values, .9),
               "clock": "cuda"}
        for name, values in stage_samples.items()
        if not name.startswith("refresh-") or refresh_device}
    result["host_trace_stages"] = {
        name: {"median_ms": median(values), "p90_ms": percentile(values, .9),
               "clock": "host"}
        for name, values in stage_samples.items()
        if name.startswith("refresh-") and not refresh_device}
    if method == "jetspecpp":
        hit_residuals = [round_values[i] - verify_values[i]
                         for r in joined if not r.get("fallback")
                         for i in [record_index[(r["seq_id"], r["candidate_round"])]]
                         if step_aligned and i < len(round_values) and i < len(verify_values)]
        ordinary_host = median(hit_residuals) if hit_residuals else None
        fallback = [max(0., round_values[i] - verify_values[i] - ordinary_host)
                    for r in joined if r.get("fallback") and ordinary_host is not None
                    for i in [record_index[(r["seq_id"], r["candidate_round"])]]
                    if i < len(round_values) and i < len(verify_values)]
        joined_indices = [record_index[(r["seq_id"], r["candidate_round"])] for r in joined]
        result["model_samples"] = {"verify_ms": [verify_values[i] for i in joined_indices
                                                 if i < len(verify_values)],
                                   "fallback_ms": fallback,
                                   "round_ms": [round_values[i] for i in joined_indices
                                                if i < len(round_values)],
                                   "stage_ms": stage_samples}
    return result


def load_anatomy(path):
    """Load an anatomy.json file or the unique anatomy in a read directory."""
    path = Path(path)
    if path.is_dir():
        matches = list(path.rglob("anatomy.json")) or list(path.glob("*.anatomy.json"))
        if len(matches) != 1:
            raise ValueError(f"expected one anatomy.json in {path}, found {len(matches)}")
        path = matches[0]
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "jetspec.serve_anatomy/1":
        raise ValueError(f"unsupported anatomy schema in {path}")
    return data


def headline(data):
    value = lambda n: "NOT_MEASURED" if n is None else f"{n:.4f}"
    return (f"ANATOMY method={data['method']} regime={data['regime']} rounds={data['rounds']} "
            f"hit_rate={value(data['hit_rate'])} tau={value(data['tau'])} "
            f"tau_hit={value(data['tau_hit'])} tau_miss={value(data['tau_miss'])} "
            f"cutoff_rate={value(data['cutoff_rate'])} "
            f"median_round_ms={value(data['round_ms']['median'])} TPS={value(data['tps'])} "
            f"tau_eos={value(data.get('tau_eos_stopped'))} "
            f"TPS_eos={value(data.get('tps_eos_stopped'))}")


def csv_row(data):
    values = {"method": data["method"], "regime": data["regime"],
              "rounds": data["rounds"], "hit_rate": data["hit_rate"],
              "tau": data["tau"], "tau_hit": data["tau_hit"],
              "tau_miss": data["tau_miss"],
              "tps": data["tps"], "hit_rounds": data["hit_rounds"],
              "miss_rounds": data["miss_rounds"], "cutoff_rate": data["cutoff_rate"],
              "acceptance_misses": data["fallback_causes"].get("acceptance"),
              "deadline_misses": data["fallback_causes"].get("deadline"),
              "published_depth": json.dumps(data["published_depth"], sort_keys=True),
              "cutoff_stages": json.dumps(data["cutoff_stages"], sort_keys=True)}
    for name in ("round_ms", "verify_ms"):
        for key in ("mean", "median", "p90"):
            values[f"{name}_{key}"] = data[name][key]
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="anatomy.json files or read directories")
    parser.add_argument("--csv", type=Path, help="write fixed-column collection CSV")
    args = parser.parse_args(argv)
    paths = []
    for path in map(Path, args.paths):
        found = sorted(path.rglob("anatomy.json")) if path.is_dir() else [path]
        if not found:
            raise ValueError(f"no anatomy.json in {path}")
        paths.extend(found)
    data = [load_anatomy(path) for path in paths]
    for row in data:
        print(headline(row))
        print(json.dumps({key: row[key] for key in
                          ("rounds", "hit_rate", "tau", "tau_hit", "tau_miss",
                           "hit_rounds", "miss_rounds", "fallback_causes", "cutoff_stages",
                           "published_depth", "by_depth", "verify_end_stages",
                           "round_ms", "hit_round_ms",
                           "miss_round_ms", "verify_ms", "stages", "device_stages",
                           "host_trace_stages")},
                         sort_keys=True))
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(csv_row(row) for row in data)
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_row(row) for row in data)


if __name__ == "__main__":
    main()

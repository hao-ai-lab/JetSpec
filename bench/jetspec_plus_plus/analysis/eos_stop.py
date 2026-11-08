"""Recompute EOS-stopped serving metrics from saved read files, without a model."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT)]
from jetspec.engines.jetspec_plus_plus.launch.records import (
    ROUND_METHODS, round_is_miss, stop_lengths, compute_eos_stopped,
)


FIELDS = ("read", "method", "tau_to_cap", "tau_eos_stopped", "tps_to_cap",
          "tps_eos_stopped", "decode_tokens_eos_stopped", "rounds_eos_stopped",
          "hit_rate_eos_stopped", "tau_hit_eos_stopped", "tau_miss_eos_stopped",
          "cutoff_rate_eos_stopped", "error")












def _sidecar(read_path, suffix):
    path = Path(str(read_path) + suffix)
    return json.loads(path.read_text()) if path.exists() else None


def _read_path(campaign_path, output):
    path = Path(output)
    return path if path.is_file() else campaign_path.parent / path.name


def rows_from_path(path, *, eos_token_ids=None):
    path = Path(path)
    campaigns = sorted(path.glob("*.campaign.json")) if path.is_dir() else [path]
    sources = [(campaign_path, json.loads(campaign_path.read_text()))
               for campaign_path in campaigns]
    if path.is_dir() and not campaigns:
        reads = []
        for read_path in sorted(path.glob("*.json")):
            if read_path.name.endswith((".manifest.json", ".populations.json")):
                continue
            raw = json.loads(read_path.read_text())
            if "outputs" not in raw or "metrics" not in raw:
                continue
            population_file = _sidecar(read_path, ".populations.json") or {}
            method = population_file.get("method")
            if method is None:
                manifest = _sidecar(read_path, ".manifest.json") or raw.get("run_manifest", {})
                method = manifest.get("values", {}).get("method_mode")
                # The runner records the engine's mode name; map it back to the read method.
                method = {"spec": "spec_sync", "spec_async": "ssd"}.get(method, method)
            if method is None:
                raise ValueError(f"{read_path}: missing method in populations or run manifest")
            reads.append((read_path, method, raw["max_new"]))
        if not reads:
            raise ValueError(f"no saved reads in {path}")
        sources = [(path / "read-dir", {"resolved_condition": {
            "method": method, "tokens": tokens}, "reads": [{"output": str(read_path)}]})
            for read_path, method, tokens in reads]
    for campaign_path, campaign in sources:
        condition = campaign["resolved_condition"]
        method = condition["method"]
        eos = eos_token_ids or condition.get("eos_token_ids")
        for read in campaign["reads"]:
            read_path = _read_path(campaign_path, read["output"])
            row = {key: None for key in FIELDS}
            row.update(read=str(read_path), method=method)
            try:
                raw = json.loads(read_path.read_text())
                measured = raw["metrics"]["serve_read"]["measured"]
                if read.get("measured", {}).get("tau_to_cap") is not None:
                    row["tau_to_cap"] = read["measured"]["tau_to_cap"]
                elif method == "ar":
                    row["tau_to_cap"] = 1
                elif read.get("summary") and isinstance(read["summary"].get("tau"), dict):
                    row["tau_to_cap"] = read["summary"]["tau"]["pooled_kept"]
                else:
                    source = ("jetspec_round_records" if method == "jetspecpp" else
                              "blockhead_round_records")
                    records = measured["metrics"].get(source) or []
                    if records:
                        accepted = "accepted_len" if method == "jetspecpp" else "accepted"
                        row["tau_to_cap"] = sum(float(record[accepted]) for record in records) / len(records)
                    elif method == "spec_sync":
                        lengths = measured["metrics"].get("accepted_suffix_lens_with_recovery") or []
                        if lengths:
                            row["tau_to_cap"] = sum(lengths) / len(lengths)
                row["tps_to_cap"] = measured["decode_tok_s"]
                population_file = _sidecar(read_path, ".populations.json")
                populations = population_file["populations"] if population_file else []
                result = compute_eos_stopped(
                    method, raw["outputs"], measured["metrics"],
                    tokens=int(condition["tokens"]), eos_token_ids=eos,
                    populations=populations, batch=int(raw.get("batch", 1)))
                row.update(result)
                if row["tau_to_cap"] is None:
                    raise ValueError(f"{method}: missing to-cap tau or saved round lengths")
            except (KeyError, OSError, ValueError) as error:
                row["error"] = str(error)
            yield row


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

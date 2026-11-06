"""Acceptance per decode round from the existing serving counters."""
from statistics import mean


def tau_to_cap(method, measured):
    metrics = measured["metrics"]
    block = metrics.get("blockhead_round_records") or []
    jetspec = metrics.get("jetspec_round_records") or []
    steps = metrics.get("target_step_times") or []
    if block:
        rounds = len(block)
        accepted = [r["accepted"] for r in block]
    elif jetspec:
        rounds = (len(steps) - 1 if len(steps) == len(jetspec) + 1
                  else len(metrics.get("jetspec_verify_round_ms") or []))
        accepted = [r["accepted_len"] for r in jetspec]
    else:
        rounds = len(steps[1:])
        accepted = []
    if not rounds:
        raise ValueError(f"no host round durations for {method}")
    return mean(accepted) if accepted else measured["decode_tokens"] / rounds

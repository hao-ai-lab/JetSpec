"""Explicit offline profile over the serving chain; serving defaults stay intact."""
from __future__ import annotations

import json
import math
import os

from jetspec.engines.jetspec_plus_plus.draft.method.schedule import FIXED_CONFIGS
from jetspec.engines.jetspec_plus_plus.engine import current_engine
from jetspec.engines.jetspec_plus_plus.launch_env import environment_name
from jetspec.engines.jetspec_plus_plus.runtime.config_values import config_values


def enabled(environ=None):
    from jetspec.engines.jetspec_plus_plus.runtime.config_values import ConfigValues
    return current_engine(None if isinstance(environ, ConfigValues) else environ).execution in ("pipelined", "critical-path", "on")


def critical_path(environ=None):
    return current_engine(environ).execution == "critical-path"


def kv_cache_blocks(config, available_blocks):
    """Bound N=1 eval's paged KV to two complete contexts plus spare pages."""
    if not enabled() or not config.jetspec:
        return available_blocks
    if config.max_num_seqs != 1:
        raise ValueError('eval KV reservation requires N=1')
    if config.max_model_len <= 0 or config.kvcache_block_size <= 0:
        raise ValueError('eval KV reservation requires positive context and page sizes')
    # The scheduler needs at most one full context, including its lookahead.
    # Keep another context and two pages for cold admission/prefix-cache reuse.
    required = 2 * math.ceil(config.max_model_len / config.kvcache_block_size) + 2
    if available_blocks < required:
        raise ValueError(f'eval KV reservation needs {required} pages, has {available_blocks}')
    return required


def release_finished_sequences(tables, active):
    """Drop every request id >= 0 not in ``active`` from the per-request tables (warm ids are negative)."""
    keep = {int(seq_id) for seq_id in active}
    released = 0
    for table in tables:
        for seq_id in [key for key in table if int(key) >= 0 and int(key) not in keep]:
            del table[seq_id]
            released += 1
    return released


def draft_delay_seconds(environ=None, *, run_config=None):
    """An eval-only host delay for the timing-independence gate."""
    env = environ
    if not enabled(env):
        return 0.0
    delay = float(config_values((run_config if environ is None else environ)).get('engine.engine_settings.eval_draft_delay_ms', '0'))
    if not math.isfinite(delay) or delay < 0:
        raise ValueError('eval draft delay must be finite nonnegative milliseconds')
    return delay / 1000


def states_only(environ=None, *, run_config=None):
    selected = config_values((run_config if environ is None else environ)).get('method.method_settings.eval_refresh_mode', 'full')
    if selected not in ('full', 'states_only'):
        raise ValueError('JETSPEC_EVAL_REFRESH_MODE must be full or states_only')
    if selected == 'states_only' and not current_engine(environ).deterministic:
        raise ValueError('method.method_settings.eval_refresh_mode=states_only requires engine.deterministic')
    return selected == 'states_only'


def capture_candidates(environ=None, *, run_config=None):
    value = config_values((run_config if environ is None else environ)).get('engine.engine_settings.eval_capture_round_candidates', '0')
    if value not in ('0', '1', 'decoder'):
        raise ValueError('JETSPEC_EVAL_CAPTURE_ROUND_CANDIDATES must be 0, 1 or decoder')
    if value != '0' and not current_engine(environ).deterministic:
        raise ValueError(f'engine.engine_settings.eval_capture_round_candidates={value} requires engine.deterministic')
    return current_engine(environ).record == 'candidates'


def capture_decoder_inputs(environ=None, *, run_config=None):
    """Eval-only: also record each proposal's decoder window and the prefix taps it attends to."""
    env = environ
    return capture_candidates(env, run_config=run_config) and config_values((run_config if environ is None else environ)).get('engine.engine_settings.eval_capture_round_candidates', '0') == 'decoder'


def draft_candidates(logits, previous=None, frozen=None):
    """Raw draft softmax top-16, preserving each published row's final pass."""
    import torch
    if logits.shape[-1] < 16:
        raise ValueError('eval capture requires draft vocabulary of at least 16')
    ranked = logits.float().softmax(-1).topk(16, -1)
    result = ranked.indices, ranked.values
    if previous is not None:
        result = tuple(torch.where(frozen[:, None, None], old, new)
                       for old, new in zip(previous, result))
    return result


def draft_token_values(logits, tokens, pinned, previous=None, frozen=None):
    """Raw probability of the published token, including forced/pinned tokens."""
    import torch
    probabilities = logits.float().softmax(-1).gather(
        -1, tokens.clamp_min(0)[..., None]).squeeze(-1)
    result = probabilities.masked_fill(tokens < 0, 0), pinned & (tokens >= 0)
    if previous is not None:
        result = tuple(torch.where(frozen[:, None], old, new)
                       for old, new in zip(previous, result))
    return result


def assert_uncut(environ=None, *, run_config=None):
    env = config_values(run_config if environ is None else environ)
    states_only(environ, run_config=run_config)
    capture = capture_candidates(environ, run_config=run_config)
    capture_decoder_inputs(environ, run_config=run_config)
    if not enabled(environ):
        return
    draft_delay_seconds(environ, run_config=run_config)
    if capture:
        for name in ('engine.engine_settings.chain_wire_record', 'engine.engine_settings.record_ptop1', 'engine.engine_settings.record_qshadow'):
            if env.get(name) != '1':
                raise ValueError(f'eval capture requires {environment_name(*name.split(".", 1))}=1')
        if not config_values((run_config if environ is None else environ)).get('engine.engine_settings.w280_record'):
            raise ValueError('eval capture requires W280_RECORD destination')
    required = {
        "engine.engine_settings.draft_deadline": "0",
        "engine.engine_settings.chain_graph": "seg",
        "engine.engine_settings.chain_miss_redraft": "1",
        "engine.engine_settings.chain_miss_redraft_max_ms": "inf",
        "engine.engine_settings.refresh_fixed": "1",
        "method.method_settings.rows_rule_cap": env.get('method.method_settings.eval_rows', "16"),
    }
    for name, value in required.items():
        if env.get(name) != value:
            raise ValueError(f"eval requires {environment_name(*name.split('.', 1))}={value}")
    for name in ("engine.engine_settings.chain_j_switch_l", "engine.engine_settings.chain_tau_table",
                 "engine.engine_settings.refresh_khat_cap2", "method.method_settings.tree_b"):
        if name in env:
            raise ValueError(f"eval refuses cutoff {environment_name(*name.split('.', 1))}")
    if config_values((run_config if environ is None else environ)).get('method.method_settings.chain_fixed_config') not in FIXED_CONFIGS:
        raise ValueError("eval requires fixed J0..3 with R0..J")
    if config_values((run_config if environ is None else environ)).get('method.method_settings.drafts') not in ("15", "31"):
        raise ValueError("eval requires budget 16 or 32")
    if config_values((run_config if environ is None else environ)).get('method.method_settings.eval_rows', "16") not in ("4", "8", "16"):
        raise ValueError("eval rows must be 4, 8 or 16")
    from jetspec.engines.jetspec_plus_plus.draft.chain.refresh_swap import check_regime
    check_regime(environ=environ, run_config=run_config)


def round_audit(buffers, *, seq_id, round_id, j, r, rows, redraft=False, run_config=None):
    """Read graph-owned counters after its existing publish synchronization."""
    audit = buffers["eval_redraft_audit" if redraft else "eval_audit"].tolist()
    config = config_values(run_config).get('method.method_settings.chain_fixed_config')
    if config in FIXED_CONFIGS:
        j, r = int(config[1]), int(config[-1])
    result = dict(seq_id=seq_id, round=round_id, kind="redraft" if redraft else "proposal",
                j_passes=int(audit[0]), refreshes=int(audit[1]),
                rows_published=int(audit[2]), expected_j=j, expected_r=r,
                refresh_khat_skips=int(audit[3]),
                expected_rows=1 if redraft else rows)
    effects = buffers.get('eval_redraft_refresh_effects' if redraft else 'eval_refresh_effects')
    if effects is not None:
        result.update(refresh_mode='states_only', swaps=int(effects[0].item()),
                      pins=int(effects[1].item()))
    return result


CUT_COUNTERS = ("deadline_misses", "j_fallbacks", "valve_hits", "ar_steps", "late",
                "refresh_khat_skips", "short_j_rounds", "short_r_rounds", "short_row_rounds")


def read_counters(log, metrics):
    """Fail closed when launch evidence is missing; do not infer zero from settings."""
    counters = dict.fromkeys(CUT_COUNTERS, 0)
    records = []
    active = None
    phased = "SERVE_READ_PHASE_START " in log
    for line in log.splitlines():
        if line.startswith("SERVE_READ_PHASE_START "):
            active = line.split(" ", 1)[1]
        elif line.startswith("SERVE_READ_PHASE_END "):
            active = None
        if phased and active != "measured":
            continue
        if "EVAL_ROUND " in line:
            row, _ = json.JSONDecoder().raw_decode(line.split("EVAL_ROUND ", 1)[1])
            if row["seq_id"] >= 0:
                records.append(row)
        if "CHAIN_FALLBACK " in line:
            counters["j_fallbacks"] += 1
        if "W320_DEADLINE " in line:
            row, _ = json.JSONDecoder().raw_decode(line.split("W320_DEADLINE ", 1)[1])
            counters["deadline_misses"] += int(row.get("deadline_miss", 0))
            counters["j_fallbacks"] += int(row.get("truncated", 0))
    if not records:
        raise ValueError("eval read has no EVAL_ROUND launch evidence")
    for row in records:
        counters["refresh_khat_skips"] += row['refresh_khat_skips']
        counters["short_j_rounds"] += int(row["j_passes"] != row["expected_j"])
        counters["short_r_rounds"] += int(row["refreshes"] != row["expected_r"])
        counters["short_row_rounds"] += int(row["rows_published"] != row["expected_rows"])
        if row.get('refresh_mode') == 'states_only':
            for name in ('swaps', 'pins'):
                counters[name] = counters.get(name, 0) + row[name]
                if row[name]:
                    raise ValueError(f'eval states_only has nonzero {name}: {row[name]}')
    counters["valve_hits"] = int(metrics.get("jetspec_chain_redraft_missing", 0))
    decisions = metrics.get("jetspec_round_records", [])
    if not decisions:
        raise ValueError("eval read has no verifier decisions")
    if 'jetspec_late_count' not in metrics:
        raise ValueError('eval read has no late decision counter')
    counters['late'] = int(metrics['jetspec_late_count'])
    for row in decisions:
        if row.get('decision') not in ('hit', 'miss'):
            raise ValueError(f"eval verifier decision must be hit or miss: {row.get('decision')}")
        if row.get('candidate_kind') != 'speculative':
            raise ValueError('eval verifier candidate_kind must be speculative')
        if row.get('previous_round_ar') is not False:
            raise ValueError('eval verifier previous_round_ar must be false')
    audited = {(row['seq_id'], row['round']) for row in records}
    missing = [(row.get('seq_id'), row.get('candidate_round')) for row in decisions
               if (row.get('seq_id'), row.get('candidate_round')) not in audited]
    if missing:
        raise ValueError(f'eval read has unaudited verifier candidates: {missing[:8]}')
    if 'jetspec_ar_steps' not in metrics:
        raise ValueError('eval read has no AR execution counter')
    # A successful miss redraft retains its original "miss" decision kind;
    # verifier.py counts the actual one-token target forwards separately.
    counters["ar_steps"] = int(metrics['jetspec_ar_steps'])
    proposals = [row for row in records if row["kind"] == "proposal"]
    if not proposals:
        raise ValueError("eval read has no proposal publications")
    counters["rows_published_mean"] = sum(row["rows_published"] for row in proposals) / len(proposals)
    counters["audited_rounds"] = len(records)
    counters["verified_rounds"] = len(decisions)
    counters["redraft_rounds"] = len(records) - len(proposals)
    bad = {name: counters[name] for name in CUT_COUNTERS if counters[name]}
    if bad:
        raise ValueError(f"eval cut counters are nonzero: {bad}")
    return counters

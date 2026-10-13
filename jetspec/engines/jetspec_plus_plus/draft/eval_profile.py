"""Explicit offline profile over the serving chain; serving defaults stay intact."""
from __future__ import annotations

import json
import math
import os

FIXED_CONFIGS = tuple(f'J{j}-R{r}' for j in range(4) for r in range(j + 1))


def enabled(environ=None):
    env = os.environ if environ is None else environ
    return env.get("JETSPEC_EVAL_MODE", "off") != "off"


def critical_path(environ=None):
    env = os.environ if environ is None else environ
    return env.get("JETSPEC_EVAL_MODE", "off") == "critical-path"


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




def states_only(environ=None):
    env = os.environ if environ is None else environ
    selected = env.get('JETSPEC_EVAL_REFRESH_MODE', 'full')
    if selected not in ('full', 'states_only'):
        raise ValueError('JETSPEC_EVAL_REFRESH_MODE must be full or states_only')
    if selected == 'states_only' and not enabled(env):
        raise ValueError('JETSPEC_EVAL_REFRESH_MODE=states_only requires JETSPEC_EVAL_MODE')
    return selected == 'states_only'


def capture_candidates(environ=None):
    env = os.environ if environ is None else environ
    value = env.get('JETSPEC_EVAL_CAPTURE_ROUND_CANDIDATES', '0')
    if value not in ('0', '1', 'decoder'):
        raise ValueError('JETSPEC_EVAL_CAPTURE_ROUND_CANDIDATES must be 0, 1 or decoder')
    if value != '0' and not enabled(env):
        raise ValueError(f'JETSPEC_EVAL_CAPTURE_ROUND_CANDIDATES={value} requires JETSPEC_EVAL_MODE')
    return value != '0'


def capture_decoder_inputs(environ=None):
    """Eval-only: also record each proposal's decoder window and the prefix taps it attends to."""
    env = os.environ if environ is None else environ
    return capture_candidates(env) and env.get('JETSPEC_EVAL_CAPTURE_ROUND_CANDIDATES', '0') == 'decoder'


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


def assert_uncut(environ=None):
    env = os.environ if environ is None else environ
    states_only(env)
    capture = capture_candidates(env)
    capture_decoder_inputs(env)
    if not enabled(env):
        return
    draft_delay_seconds(env)
    if capture:
        for name in ('JETSPEC_CHAIN_WIRE_RECORD', 'E04_RECORD_PTOP1', 'E04_RECORD_QSHADOW'):
            if env.get(name) != '1':
                raise ValueError(f'eval capture requires {name}=1')
        if not env.get('W280_RECORD'):
            raise ValueError('eval capture requires W280_RECORD destination')
    required = {
        "JETSPEC_DRAFT_DEADLINE": "0",
        "JETSPEC_CHAIN_GRAPH": "seg",
        "JETSPEC_CHAIN_MISS_REDRAFT": "1",
        "JETSPEC_CHAIN_MISS_REDRAFT_MAX_MS": "inf",
        "JETSPEC_REFRESH_FIXED": "1",
        "JETSPEC_ROWS_RULE_CAP": env.get("JETSPEC_EVAL_ROWS", "16"),
    }
    for name, value in required.items():
        if env.get(name) != value:
            raise ValueError(f"eval requires {name}={value}")
    for name in ("JETSPEC_CHAIN_J_SWITCH_L", "JETSPEC_CHAIN_TAU_TABLE",
                 "JETSPEC_REFRESH_KHAT_CAP2", "JETSPEC_TREE_B"):
        if name in env:
            raise ValueError(f"eval refuses cutoff {name}")
    if env.get("JETSPEC_CHAIN_FIXED_CONFIG") not in FIXED_CONFIGS:
        raise ValueError("eval requires fixed J0..3 with R0..J")
    if env.get("JETSPEC_DRAFTS") not in ("15", "31"):
        raise ValueError("eval requires budget 16 or 32")
    if env.get("JETSPEC_EVAL_ROWS", "16") not in ("4", "8", "16"):
        raise ValueError("eval rows must be 4, 8 or 16")
    from jetspec.engines.jetspec_plus_plus.draft.chain.refresh_swap import check_regime
    check_regime(environ=env)




CUT_COUNTERS = ("deadline_misses", "j_fallbacks", "valve_hits", "ar_steps", "late",
                "refresh_khat_skips", "short_j_rounds", "short_r_rounds", "short_row_rounds")



"""Pure split-conformal set and flat keyed-row selection."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from jetspec.engines.jetspec_plus_plus.models.config import setting


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.lac import KeyedRows


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.offers import keep_device


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.offers import lac_candidates


def row_rank_parameters(*, run_config=None):
    """Budget-only score knobs; coverage mode ignores these settings."""
    if setting(run_config, 'method', 'method_settings.row_select') != 'budget':
        return 1.0, 1.0, False
    exponents = setting(run_config, 'method', 'method_settings.row_select_exp', '1,1')
    values = tuple(map(float, exponents.split(',')))
    if len(values) != 2 or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('JETSPEC_ROW_SELECT_EXP requires two finite nonnegative exponents')
    return *values, setting(run_config, 'method', 'method_settings.row_select_length') == '1'


def keyed_rows(k_ids, k_probabilities, anchor_ids, anchor_probabilities, *, cap, run_config=None,
               rank_parameters=None):
    from jetspec.engines.jetspec_plus_plus.draft.method.decoder.lac import keyed_rows as method_rows
    return method_rows(k_ids, k_probabilities, anchor_ids, anchor_probabilities, cap=cap,
                       rank_parameters=(row_rank_parameters(run_config=run_config)
                                        if rank_parameters is None else rank_parameters))


def row_bucket(live_count, *, cap, graphed, run_config=None):
    from jetspec.engines.jetspec_plus_plus.draft.method.decoder.lac import row_bucket as method_bucket
    return method_bucket(live_count, cap=cap, graphed=graphed,
                         fixed_graph=graphed and setting(run_config, 'engine', 'engine_settings.g_row_buckets') != 'pow2',
                         budget=setting(run_config, 'method', 'method_settings.row_select') == 'budget')

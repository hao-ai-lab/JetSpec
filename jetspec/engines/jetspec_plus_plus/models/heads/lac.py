"""Pure split-conformal set and flat keyed-row selection."""
from __future__ import annotations

import os
import math
from dataclasses import dataclass

import torch


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.lac import KeyedRows


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.lac import keep_device


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.lac import lac_candidates


def row_rank_parameters():
    """Budget-only score knobs; coverage mode ignores these settings."""
    if os.environ.get('JETSPEC_ROW_SELECT') != 'budget':
        return 1.0, 1.0, False
    values = tuple(map(float, os.environ.get('JETSPEC_ROW_SELECT_EXP', '1,1').split(',')))
    if len(values) != 2 or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('JETSPEC_ROW_SELECT_EXP requires two finite nonnegative exponents')
    return *values, os.environ.get('JETSPEC_ROW_SELECT_LENGTH') == '1'


def keyed_rows(k_ids, k_probabilities, anchor_ids, anchor_probabilities, *, cap):
    return method_rows(k_ids, k_probabilities, anchor_ids, anchor_probabilities, cap=cap,
                       rank_parameters=row_rank_parameters())


def row_bucket(live_count, *, cap, graphed):
    from jetspec.engines.jetspec_plus_plus.draft.method.decoder.lac import row_bucket as method_bucket
    return method_bucket(live_count, cap=cap, graphed=graphed,
                         fixed_graph=graphed and os.environ.get('JETSPEC_G_ROW_BUCKETS') != 'pow2',
                         budget=os.environ.get('JETSPEC_ROW_SELECT') == 'budget')

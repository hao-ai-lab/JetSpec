
from __future__ import annotations
from jetspec.engines.jetspec_plus_plus.draft.config import setting
from jetspec.engines.jetspec_plus_plus.draft.method.geometry import Geometry, MIN_WINDOW_ROWS
from jetspec.engines.jetspec_plus_plus.draft.block_geometry import window_rows

def staged_context(run_config=None):
    from jetspec.engines.jetspec_plus_plus.draft.drafter import OneRoundAdaptedContext
    return OneRoundAdaptedContext(max_adapted=window_rows(run_config=run_config))

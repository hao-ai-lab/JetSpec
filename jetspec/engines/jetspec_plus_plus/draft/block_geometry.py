"""Block geometry: the served block width and the sizes derived from it.

JETSPEC_DRAFTS (drafts verified per round) is the one width input. A served block holds the
anchor plus its drafts, so the block width is drafts + 1. This module reads the value without the
layout checks of ``protocol.draft_width`` and imports nothing heavy, so env validation can use it.
"""
from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.draft.config import setting
from jetspec.engines.jetspec_plus_plus.draft.method.geometry import Geometry, MIN_WINDOW_ROWS


def drafts(environ=None, *, run_config=None):
    """JETSPEC_DRAFTS as an int, or None when it is unset or not a canonical integer."""
    value = setting(run_config if run_config is not None else environ, 'method.method_settings.drafts')
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    # A non-canonical spelling ('031', ' 31') never matched the old string compares.
    return parsed if str(parsed) == value else None


def block_width(environ=None, *, run_config=None):
    """Rows in one served block (anchor + drafts), or None when JETSPEC_DRAFTS is unset."""
    count = drafts(environ, run_config=run_config)
    return None if count is None else Geometry.from_drafts(count).block


def window_rows(environ=None, *, run_config=None):
    """Rows the C1 window, dirty lanes and live key bucket reserve for one block."""
    return Geometry(block_width(environ, run_config=run_config) or 0).window_rows


def staged_context(run_config=None):
    from jetspec.engines.jetspec_plus_plus.jetspec.drafter import OneRoundAdaptedContext
    return OneRoundAdaptedContext(max_adapted=window_rows(run_config=run_config))

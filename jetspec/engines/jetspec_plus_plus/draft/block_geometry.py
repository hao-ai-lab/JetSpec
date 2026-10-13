"""Block geometry: the served block width and the sizes derived from it.

JETSPEC_DRAFTS (drafts verified per round) is the one width input. A served block holds the
anchor plus its drafts, so the block width is drafts + 1. This module reads the value without the
layout checks of ``protocol.draft_width`` and imports nothing heavy, so env validation can use it.
"""
from __future__ import annotations

import os

# The C1 window, dirty lanes and live key bucket were sized for B16 before B4 and B8 served.
# Narrower blocks keep that storage; a wider block takes its own width.
MIN_WINDOW_ROWS = 16


def drafts(environ=None):
    """JETSPEC_DRAFTS as an int, or None when it is unset or not a canonical integer."""
    value = (os.environ if environ is None else environ).get('JETSPEC_DRAFTS')
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    # A non-canonical spelling ('031', ' 31') never matched the old string compares.
    return parsed if str(parsed) == value else None


def block_width(environ=None):
    """Rows in one served block (anchor + drafts), or None when JETSPEC_DRAFTS is unset."""
    count = drafts(environ)
    return None if count is None else count + 1


def window_rows(environ=None):
    """Rows the C1 window, dirty lanes and live key bucket reserve for one block."""
    return max(MIN_WINDOW_ROWS, block_width(environ) or 0)

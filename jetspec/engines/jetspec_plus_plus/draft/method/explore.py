"""Construction seams for optional exploration methods."""
from .ops import reference_op

KEYS = ("explore.swap_rule", "explore.after_swap_fill", "explore.legacy_kpred")

@reference_op("refresh_swap_state")
def refresh_swap_state():
    return None

@reference_op("legacy_kpredictor")
def legacy_kpredictor():
    return None

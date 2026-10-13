"""Finite-sample calibration and threshold lookup, with no model dependency."""
import math
from jetspec.engines.jetspec_plus_plus.draft.method.ops import reference_op




def dataset_entry(entries, dataset):
    """Resolve a per-dataset threshold, falling back to its pooled entry."""
    if dataset in entries and 'pooled' in entries:
        raise ValueError(f'ambiguous pooled and per-dataset thresholds for {dataset}')
    return entries[dataset] if dataset in entries else entries['pooled']

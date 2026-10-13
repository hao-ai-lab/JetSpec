"""Completed-depth labels and the joint rule's explicit-label/pooled tau lookup."""
from jetspec.engines.jetspec_plus_plus.draft.method.passes import normalized_label; from jetspec.engines.jetspec_plus_plus.draft.method.passes import completed_label; from jetspec.engines.jetspec_plus_plus.draft.method.passes import resolve_tau
import os


def eval_completed_label(passes_run, environ=None):
    """Label control for the eval J3/J2 comparison; physical depth is unchanged."""
    label = completed_label(passes_run)
    from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled
    env = os.environ if environ is None else environ
    return env.get('JETSPEC_EVAL_J3_LABEL', 'J3') if enabled(env) and label == 'J3' else label


def chain_labels():
    from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled
    return ('J0d0', 'J1', 'J2', eval_completed_label(4)) if enabled() else ('J0d0', 'J1', 'J2')


def chain_tau_values(table, *, bundle=None, recipe=None):
    from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled
    if enabled():
        return [resolve_tau(table, label, bundle=bundle, recipe=recipe)
                for label in chain_labels()]
    return [float(table[label]['value']) for label in chain_labels()]



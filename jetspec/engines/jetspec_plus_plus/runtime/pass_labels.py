"""Completed-depth labels and the joint rule's explicit-label/pooled tau lookup."""
from jetspec.engines.jetspec_plus_plus.draft.method.schedule import PASS_LABELS, normalized_label, completed_label, resolve_tau
import os
from jetspec.engines.jetspec_plus_plus.runtime.config_values import config_values


def eval_completed_label(passes_run, environ=None, *, run_config=None):
    """Label control for the eval J3/J2 comparison; physical depth is unchanged."""
    label = completed_label(passes_run)
    from jetspec.engines.jetspec_plus_plus.runtime.profile import enabled
    env = environ
    return config_values((run_config if environ is None else environ)).get('method.method_settings.eval_j3_label', 'J3') if enabled(env) and label == 'J3' else label


def chain_labels(*, run_config=None):
    from jetspec.engines.jetspec_plus_plus.runtime.profile import enabled
    return PASS_LABELS[:3] + (eval_completed_label(4, run_config=run_config),) if enabled() else PASS_LABELS[:3]


def chain_tau_values(table, *, bundle=None, recipe=None, run_config=None):
    from jetspec.engines.jetspec_plus_plus.runtime.profile import enabled
    if enabled():
        return [resolve_tau(table, label, bundle=bundle, recipe=recipe)
                for label in chain_labels(run_config=run_config)]
    return [float(table[label]['value']) for label in chain_labels(run_config=run_config)]


def validate_eager_labels(environ, schedule, *, recipe=None, run_config=None):
    """Refuse unsupported labels before loading models or starting a read."""
    import json
    from pathlib import Path

    if any(name.startswith('E04_LAB_') for name in environ):
        return
    from jetspec.engines.jetspec_plus_plus.draft.serve_policy import section
    tail = section('tail_passes', environ=environ, run_config=run_config)
    deepest = schedule.j + (tail['count'] if tail['enabled'] else 0)
    if config_values((run_config if environ is None else environ)).get('method.method_settings.kset') == 'predictor' and deepest >= 3:
        raise ValueError(f'JETSPEC_KSET=predictor refuses J{deepest}: no trained group; recipe={recipe}')
    root = config_values((run_config if environ is None else environ)).get('model.model_settings.heads_dir')
    if root is None or config_values((run_config if environ is None else environ)).get('method.method_settings.rowrule') != 'joint':
        return
    path = Path(root) / 'thresholds.json'
    if not path.is_file():
        return  # Artifact loading reports missing files; dry resolves need no artifacts.
    entries = json.loads(path.read_text())['e04_rowrule_joint']
    dataset = config_values((run_config if environ is None else environ)).get('benchmark.benchmark_settings.dataset')
    if dataset in entries and 'pooled' in entries:
        raise ValueError(f'ambiguous pooled and per-dataset thresholds for {dataset}')
    rule = entries[dataset] if dataset in entries else entries['pooled']
    for passes in range(1, deepest + 2):
        resolve_tau(rule['tau'], completed_label(passes), bundle=path, recipe=recipe)

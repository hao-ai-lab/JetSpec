"""Import-selected exploration methods; ordinary serving imports no variant."""
from importlib import import_module
from jetspec.engines.jetspec_plus_plus.draft.method import explore
from jetspec.engines.jetspec_plus_plus.draft.method.ops import register_impl, use_engine

METHODS = {}
_MODULES = {"explore.swap_rule": "swap_rule", "explore.after_swap_fill": "after_swap_fill",
            "explore.legacy_kpred": "legacy_kpred"}

def register(key, **overrides):
    if key not in explore.KEYS:
        raise ValueError(f"unknown exploration method: {key}")
    for op, implementation in overrides.items():
        register_impl(op, implementation, engine=key)
    METHODS[key] = tuple(overrides)


def implementation(key, op):
    load(key)
    if op not in METHODS[key]:
        raise ValueError(f"{key} does not implement {op}")
    with use_engine(key):
        return getattr(explore, op)()

def method_value(run_config, name):
    method = getattr(run_config, "method", {})
    if isinstance(run_config, dict):
        method = run_config.get("method", {})
    return method.get("explore", {}).get(name)

def refresh_type(run_config):
    from jetspec.engines.jetspec_plus_plus.draft.serve_policy import section
    # Existing compositions retain their effective policy. New compositions can
    # select the same method directly by its declared method key.
    for name in ("swap_rule", "after_swap_fill"):
        selected = method_value(run_config, name)
        if selected is None:
            selected = section(name, run_config=run_config)["enabled"]
        if selected:
            return implementation("explore." + name, "refresh_swap_state")
    return None

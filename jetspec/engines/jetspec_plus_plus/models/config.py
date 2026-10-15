"""Read optional model settings from the caller's immutable rank snapshot."""


def setting(run_config, category, path, default=None):
    if run_config is None:
        return default
    value = getattr(run_config, category)
    for key in path.split('.'):
        if key not in value:
            return default
        value = value[key]
    return default if value is None else value

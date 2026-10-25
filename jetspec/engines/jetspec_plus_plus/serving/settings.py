"""Read launch settings from a received record or its owning rank."""
from jetspec.engines.jetspec_plus_plus.run_config import RunConfig


def config_for(owner):
    if isinstance(owner, RunConfig) or owner is None:
        return owner
    try:
        return object.__getattribute__(owner, 'run_config')
    except AttributeError:
        return None


def setting(owner, path, default=None):
    config = config_for(owner)
    if config is None:
        return default
    if path not in config.provided_values:
        return default
    value = config.provided_values[path]
    return default if value is None else value


def required_setting(owner, path):
    value = setting(owner, path)
    if value is None:
        from jetspec.engines.jetspec_plus_plus.legacy_env import environment_name
        category, field = path.split('.', 1)
        raise KeyError(environment_name(category, field))
    return value


def run_rank(runner, config, *args):
    """Install per-process trace hooks from the record before constructing a rank."""
    record = getattr(config, 'run_config', None)
    if record is None:
        raise ValueError('rank missing RunConfig launch_overlay')
    if setting(record, 'engine.engine_settings.roundtrace'):
        from jetspec.engines.jetspec_plus_plus.trace.current import install
        install(record)
    return runner(config, *args)

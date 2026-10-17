"""Training registers its consumer contract without a serving dependency."""


TRAINING_CONTAINERS = {
    'method': ('objective', 'optimization'),
    'model': ('training_model',),
    'engine': ('trainer',),
    'hardware': ('layout',),
}
_training_schemas = {}


def register_training_schemas(schemas):
    global _training_schemas
    _training_schemas = dict(schemas)


def training_schemas():
    return dict(_training_schemas)


def _training_key_paths(value, prefix):
    if isinstance(value, dict) and value:
        return [leaf for key, child in value.items()
                for leaf in _training_key_paths(child, prefix + '.' + str(key))]
    return [prefix]


def require_training_schemas(config):
    if _training_schemas:
        return
    keys = []
    for category, containers in TRAINING_CONTAINERS.items():
        for container in containers:
            if container in config[category]:
                keys.extend(_training_key_paths(config[category][container], f'{category}.{container}'))
    if keys:
        raise ValueError('training keys require missing package jetspec_train: ' + ', '.join(sorted(keys)))

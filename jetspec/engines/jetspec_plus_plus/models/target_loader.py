"""Spec-directed target loading; retained comparison loaders keep their protocol."""
import os
from glob import glob

from safetensors import safe_open
from tqdm import tqdm

from jetspec.engines.jetspec_plus_plus.models.loader import default_weight_loader, load_model as load_comparison_model


def target_weight_name(spec, packed_modules_mapping, source_name):
    """Return the parameter and optional packed shard, or ignore non-text data."""
    if source_name.startswith(spec.weight_prefix):
        name = 'model.' + source_name.removeprefix(spec.weight_prefix)
    elif source_name == 'lm_head.weight':
        name = source_name
    elif spec.weight_prefix != 'model.':
        return None
    else:
        name = source_name
    for source, (target, shard) in packed_modules_mapping.items():
        if source in name:
            return name.replace(source, target), shard
    return name, None


def load_target_model(model, path):
    files = glob(os.path.join(path, '*.safetensors'))
    strict = model.spec.weight_prefix != 'model.'
    if strict:
        files.sort()
        if not files:
            raise FileNotFoundError(f'no safetensors shards in {path}')
    loaded = set()
    for file in tqdm(files, desc='Loading model files'):
        with safe_open(file, 'pt', 'cpu') as weights:
            for source_name in weights.keys():
                mapped = target_weight_name(model.spec, model.packed_modules_mapping, source_name)
                if mapped is None:
                    continue
                name, shard = mapped
                parameter = model.get_parameter(name)
                loader = getattr(parameter, 'weight_loader', default_weight_loader)
                args = () if shard is None else (shard,)
                loader(parameter, weights.get_tensor(source_name), *args)
                loaded.add(name)
    if strict:
        missing = set(dict(model.named_parameters())) - loaded
        if missing:
            raise RuntimeError(f'hybrid text checkpoint is missing parameters: {sorted(missing)}')


def load_model(model, path, target_path=None, target_hidden_size=None):
    if hasattr(model, 'spec'):
        load_target_model(model, path)
    else:
        load_comparison_model(model, path, target_path=target_path,
                              target_hidden_size=target_hidden_size)

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
    # verify_cut_pack27b2: pack once at load, before warm-up and the KV cache are sized.
    from jetspec.engines.jetspec_plus_plus.models.target_pack import pack_model_projections
    pack_model_projections(model)


def load_model(model, path, target_path=None, target_hidden_size=None, is_draft=False):
    if hasattr(model, 'spec'):
        load_target_model(model, path)
        if is_draft:
            # The target precisions shape the target rank only: a spec model on the draft rank keeps BF16
            # target-role projections, as it does under a bf16 target, and is never rounded or packed here.
            from jetspec.engines.jetspec_plus_plus.models.target_linear import ProjectionPrecision, TargetLinear
            for child in model.modules():
                if isinstance(child, TargetLinear) and child.precision_role == 'target':
                    child.precision = ProjectionPrecision(role='target')
            return
        from jetspec.engines.jetspec_plus_plus.models.target_w4a8 import round_target_w4a8_weights
        round_target_w4a8_weights(model)
    else:
        load_comparison_model(model, path, target_path=target_path,
                              target_hidden_size=target_hidden_size)

"""Closed method and model contracts for five-category launches."""
import hashlib
import json
import math
from pathlib import PurePosixPath
import re

from .config import Schema, validate_fields


def _field(*parts):
    """Build a leaf descriptor from data keys, rather than a Python target name."""
    return '.'.join(parts)


def _positive_int(value):
    return type(value) is int and value > 0


def _nonnegative_int(value):
    return type(value) is int and value >= 0


def _concurrency_settings(value, schema, category):
    """Validate every same-category partial row, including unselected buckets."""
    if not isinstance(value, dict) or not value:
        return False
    fields = {key: rule for key, rule in schema.fields.items() if key != 'concurrency_settings'}
    for key, row in value.items():
        if (not isinstance(key, str) or not re.fullmatch(r'[1-9][0-9]*', key)
                or not isinstance(row, dict) or not row):
            return False
        validate_fields(row, fields, (), f'{category}.concurrency_settings.{key}',
                        atomic=schema.atomic)
    return True


def method_concurrency_settings(value):
    return _concurrency_settings(value, METHOD_SCHEMA, 'method')


def _probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def _is_sha256(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _nullable_sha256(value):
    return value is None or _is_sha256(value)


def _safe_file(value):
    return (isinstance(value, str) and bool(value) and not PurePosixPath(value).is_absolute()
            and '..' not in PurePosixPath(value).parts)


def _file_identifier(value, namespace):
    return (_identifier_record(value)
            and value['namespace'] == namespace and _safe_file(value['value']))


def _record(value, fields, required=()):
    """Validate closed provenance records without interpreting their text."""
    return (isinstance(value, dict) and set(required) <= value.keys() <= fields.keys()
            and all((type(child) is rule if isinstance(rule, type) else rule(child))
                    for key, child in value.items() for rule in (fields[key],)))


def _strings(value):
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _identity_metadata(value):
    return _record(value, {
        'namespace': str, 'resolves': lambda v: v is None or type(v) is bool,
        'resolved_by': str, 'reason': str, 'detail': str, 'measured_note': str,
        'cross_checked_against_upload': bool, 'cross_checked_by': str,
        'supersedes': _identifier_record,
        'dereferenced_at_upload': lambda v: _record(v, {
            'resolves': bool, 'resolved_by': str, 'reason': str}, ('resolves', 'resolved_by')),
    }, ('namespace',))


def _identifier_record(value):
    return (isinstance(value, dict) and 'value' in value
            and _identity_metadata({key: child for key, child in value.items() if key != 'value'})
            and (value['value'] is None or isinstance(value['value'], str)
                 or _record(value['value'], {'files': _positive_int, 'bytes': _positive_int},
                            ('files', 'bytes'))))


def _loaded_head(value):
    return _record(value, {
        'path': str, 'paths': str, 'parameters': _positive_int, 'tensors': _positive_int,
        'dtype': str, 'sha256': _identifier_record, 'architecture_cross_check': str,
        'files': _positive_int, 'distinct_parameter_counts': _positive_int,
        'seed0_J0d0_sha256': _identifier_record, 'note': str, 'shape_assertion': str,
    })


def _artifact_provenance(value):
    return _record(value, {
        'order': _nonnegative_int, 'has_files': bool,
        'repository': _identity_metadata, 'revision': _identity_metadata,
        'attributes': lambda v: _record(v, {
            'environment_variables': _strings, 'source_path': _identifier_record,
            'description': str, 'recipe': lambda v: v is None or isinstance(v, str), 'compute_capability': str,
            'versions': lambda v: _record(v, {'torch': str, 'triton': str}, ('torch', 'triton')),
            'banked_sha256': _identifier_record, 'fit_receipt': _safe_file,
            'serving_layout': str, 'checkpoint_origin': _identifier_record,
            'not_self_contained': _identifier_record,
            'resolved_file_inventory': _identifier_record,
            'unresolved_file_inventory': _identifier_record,
            'omitted_files': lambda x: _record(x, {'names': _strings, 'reason': str,
                'bytes': lambda y: isinstance(y, dict) and all(_safe_file(k) and _positive_int(v)
                    for k, v in y.items())}, ('names', 'reason')),
            'post_fetch_assertion': lambda x: _record(x, {
                'rule': str, 'measured_by': str, 'decoder': _loaded_head,
                'kpred': _loaded_head, 'thresholds': _loaded_head, 'three_properties': str},
                ('rule', 'measured_by', 'decoder', 'kpred', 'thresholds', 'three_properties')),
        }),
    }, ('order', 'has_files', 'attributes'))


def _file_provenance(value):
    return _record(value, {
        'sha256': _identity_metadata, 'uploaded_bytes': _nonnegative_int, 'bytes': _nonnegative_int,
        'bytes_measured_by': str, 'legacy': bool, 'legacy_reason': str,
        'serving_note': str, 'source_path': _identifier_record,
    }, ('sha256',))


def _manifest_header(value):
    return _record(value, {
        'schema_version': lambda v: type(v) is int and v == 1,
        'repository': _identifier_record, 'revision': _identifier_record,
        'runtime_builds': lambda v: isinstance(v, list) and all(_record(x, {
            'name': str, 'version': str, 'wheel_url': _identifier_record,
            'sha256': _identifier_record}, ('name', 'version', 'wheel_url', 'sha256')) for x in v),
    }, ('schema_version', 'repository', 'revision'))


def _receipts(value):
    return isinstance(value, list) and all(_record(item, {
        'path': _safe_file, 'schema': str, 'bundle': str,
        'thresholds_sha256': _is_sha256, 'decoder_manifest_sha256': _is_sha256,
        'sources': lambda v: isinstance(v, list) and all(_record(x, {
            'label': str, 'sha256': _is_sha256, 'repository': str, 'revision': str,
            'sections': lambda y: isinstance(y, str) or _strings(y)},
            ('label', 'sha256', 'sections')) for x in v),
    }, ('path', 'schema', 'bundle', 'thresholds_sha256', 'decoder_manifest_sha256', 'sources'))
        for item in value)


def _pins(value):
    if not isinstance(value, list):
        return False
    for item in value:
        if not _record(item, {'name': str, 'category': lambda v: v in ('checkpoints', 'prompts'),
                             'record': dict}, ('name', 'category', 'record')):
            return False
        checkpoint = {
            'repository': str, 'revision': str, 'directory': str,
            'files': lambda x: isinstance(x, dict) and all(_safe_file(k) and _record(y, {
                'sha256': _is_sha256}, ('sha256',)) for k, y in x.items()),
        }
        prompt = {
            'sha256': _is_sha256,
            'manifest_entry': lambda x: _record(x, {
                'bucket_path': _safe_file, 'bytes': lambda y: isinstance(y, str) and y.isdigit(),
                'sha256': _is_sha256, 'source_host': str, 'source_path': str, 'staged_utc': str},
                ('bucket_path', 'bytes', 'sha256', 'source_host', 'source_path', 'staged_utc')),
        }
        fields = checkpoint if item['category'] == 'checkpoints' else prompt
        if not _record(item['record'], fields, fields):
            return False
        if item['category'] == 'prompts' and item['record']['sha256'] != item['record']['manifest_entry']['sha256']:
            return False
    return True


def _artifacts(value):
    """Artifact inventories have a closed record schema, including each file."""
    if not isinstance(value, list):
        return False
    names = set()
    for item in value:
        required = {'name', 'kind', 'repository', 'revision', 'files'}
        optional = {'destination_root': lambda v: _file_identifier(v, 'fetch_destination_relative_path'),
                    'training_config_sha256': _nullable_sha256,
                    'registration_commit': lambda v: v is None or isinstance(v, str)
                        and re.fullmatch(r'[0-9a-f]{40}', v) is not None,
                    'provenance': _artifact_provenance}
        if (not isinstance(item, dict) or not required <= item.keys() <= required | optional.keys()
                or any(not rule(item[key]) for key, rule in optional.items() if key in item)):
            return False
        if not isinstance(item['name'], str) or not item['name']:
            return False
        if any(not (item['kind'] == 'generated' and item['files'] == [] and item[key] is None)
               and (not isinstance(item[key], str) or not item[key]) for key in ('repository', 'revision')):
            return False
        if item['name'] in names or item['kind'] not in ('published', 'generated', 'local'):
            return False
        names.add(item['name'])
        if not isinstance(item['files'], list):
            return False
        destinations = set()
        for file in item['files']:
            required_file = {'remote', 'destination', 'sha256'}
            if (not isinstance(file, dict) or not required_file <= file.keys() <= required_file | {'provenance'}
                    or 'provenance' in file and not _file_provenance(file['provenance'])):
                return False
            if not (_file_identifier(file['remote'], 'huggingface_repo_relative_path')
                    or item['kind'] == 'generated' and file['remote'] is None):
                return False
            if not _file_identifier(file['destination'], 'artifact_relative_path') or not _is_sha256(file['sha256']):
                return False
            destination = file['destination']['value']
            if destination in destinations:
                return False
            destinations.add(destination)
    return True


THRESHOLD_SECTIONS = ('anchor_predictor', 'anchor_predictor_v1', 'e04_kscore_blend',
                      'e04_rowrule_joint', 'eng04_ablation_refit', 'eng04_v3_legacy_refit',
                      'k_predictor', 'numeric_dtype', 'q_hat_definition', 'row_sums',
                      'schema_version', 'w314_provenance')


def _thresholds(value):
    return (isinstance(value, dict) and set(value) in (
                set(THRESHOLD_SECTIONS), set(THRESHOLD_SECTIONS) - {'eng04_v3_legacy_refit'})
            and value['schema_version'] == 1 and value['numeric_dtype'] == 'float64'
            and isinstance(value['e04_rowrule_joint'], dict)
            and isinstance(value['e04_kscore_blend'], dict))


def _decoder_slice(value):
    return (isinstance(value, dict) and set(value) == {'source', 'target_files'}
            and value['source'] == 'target_slice' and isinstance(value['target_files'], dict)
            and bool(value['target_files'])
            and all(_safe_file(name) and _is_sha256(digest) for name, digest in value['target_files'].items()))


def _release_slice(value):
    return (isinstance(value, dict) and set(value) == {'source', 'target_files', 'tensors'}
            and _decoder_slice({key: value[key] for key in ('source', 'target_files')})
            and 'model.safetensors.index.json' in value['target_files']
            and isinstance(value['tensors'], dict)
            and set(value['tensors']) == {'lm_head.weight', 'model.embed_tokens.weight'}
            and all(isinstance(name, str) and bool(name) for name in value['tensors'].values()))


def _check_geometry(config):
    method, model = config['method'], config['model']
    schedule = method.get('schedule')
    if schedule and not (0 <= schedule['r'] <= schedule['j']
                         and (method.get('round_protocol') == 'standalone-chain' or schedule['j'] <= 3)):
        raise ValueError('schedule requires 0 <= R <= J <= 3')
    fit = model.get('thresholds_fit')
    if fit:
        if config['engine'].get('kind') != 'trainer':
            if method.get('block_size') not in fit['block_sizes']:
                raise ValueError('threshold fit mismatch for method block size')
            for key in ('row_rule', 'k_score'):
                if method.get(key) != fit[key]:
                    raise ValueError(f'threshold fit mismatch for method.{key}')
        encoded = (json.dumps(model['thresholds'], sort_keys=True, indent=2) + '\n').encode()
        if hashlib.sha256(encoded).hexdigest() != model['thresholds_sha256']:
            raise ValueError('inline thresholds do not match the pinned file sha256')
        encoded_slice = (json.dumps(model['decoder_slice'], sort_keys=True, indent=2) + '\n').encode()
        if hashlib.sha256(encoded_slice).hexdigest() != model['decoder_slice_sha256']:
            raise ValueError('inline decoder slice does not match its pinned file sha256')
    return True


METHOD_SCHEMA = Schema({
    'round_protocol': ('served', 'standalone-chain'),
    _field('context', 'mode'): ('fresh', 'one_round'),
    _field('context', 'fill'): ('adapter', 'copy_previous'),
    _field('tap_knockout', 'layers'): lambda v: isinstance(v, list)
        and all(_nonnegative_int(x) for x in v) and len(set(v)) == len(v),
    _field('tap_knockout', 'value'): ('mean', 'zero'),
    _field('refresh', 'threshold'): lambda v: _probability(v) and 0 < v < 1,
    _field('refresh', 'second_threshold'): lambda v: _probability(v) and 0 < v < 1,
    _field('refresh', 'second_cap'): lambda v: v is None or (_nonnegative_int(v) and v <= 15),
    _field('noise', 'source'): ('context', 'mask'),
    _field('noise', 'seed'): _nonnegative_int,
    _field('noise', 'init_window'): _positive_int,
    'converge_until_fixed': bool, 'convergence_limit': _positive_int,
    'family': ('jetspecpp', 'dspark', 'dflash', 'dflash2', 'domino', 'ar', 'jetspec-sync', 'eagle3', 'fanout-tree'),
    'block_size': _positive_int, _field('schedule', 'j'): _nonnegative_int, _field('schedule', 'r'): _nonnegative_int,
    'label': ('J0d0', 'J1', 'J2', 'J3'), _field('cache_rows', 'cap'): _positive_int,
    _field('cache_rows', 'floor'): _nonnegative_int, _field('cache_rows', 'fill_zero'): bool,
    'row_rule': str, 'k_score': str, _field('anchor', 'cap'): _positive_int, _field('anchor', 'coverage'): _probability,
    _field('refresh', 'mode'): str, _field('refresh', 'cap'): lambda v: v is None or _positive_int(v),
    _field('refresh', 'inject_width'): _nonnegative_int, _field('refresh', 'swap'): bool,
    'agreed_pins': bool, 'miss_redraft': bool, 'chain_miss_redraft': bool,
    'row_floor': _nonnegative_int, 'refresh_cap': bool,
    'projected_khat': ('predicted', 'width-2'),
    'explore.swap_rule': lambda value: value is None or type(value) is bool,
    'explore.after_swap_fill': lambda value: value is None or type(value) is bool,
    'explore.legacy_kpred': lambda value: value is None or type(value) is bool,
}, required=('family', 'block_size'), checks=(_check_geometry,))

MODEL_SCHEMA = Schema({
    _field('target', 'repository'): str, _field('target', 'revision'): lambda v: v is None or isinstance(v, str),
    _field('target', 'family'): ('dense', 'hybrid'), _field('target', 'dtype'): ('bf16', 'fp32'),
    _field('target', 'taps'): lambda v: isinstance(v, list) and all(_nonnegative_int(x) for x in v),
    'draft_config_sha256': _is_sha256, 'artifacts': _artifacts,
    'artifact_manifest': _manifest_header, 'receipts': _receipts, 'pins': _pins,
    'decoder_slice': _decoder_slice, 'decoder_slice_sha256': _is_sha256,
    _field('drafter', 'artifact'): str, _field('drafter', 'config_sha256'): _is_sha256,
    _field('drafter', 'published'): bool,
    _field('adapter', 'artifact'): str, _field('adapter', 'config_sha256'): _is_sha256,
    _field('adapter', 'published'): bool,
    _field('target_release', 'artifact'): str, _field('target_release', 'file'): _safe_file,
    _field('target_release', 'slice'): _release_slice,
    'thresholds': _thresholds, 'thresholds_sha256': _is_sha256,
    _field('thresholds_fit', 'block_sizes'): lambda v: isinstance(v, list) and bool(v) and all(_positive_int(x) for x in v),
    _field('thresholds_fit', 'row_rule'): str,
    _field('thresholds_fit', 'k_score'): str,
}, atomic=('artifacts', 'thresholds', 'decoder_slice', 'target_release.slice', 'artifact_manifest', 'receipts', 'pins'))


# Closed compatibility leaves; legacy env spellings belong to the replay tool.
_HISTORICAL_FIELDS = {
    'method': (
        _field('method_settings', 'decoder_tail_seed'),
        _field('method_settings', 'joint_offers'),
        _field('method_settings', 'kpred_posterior'),
        _field('method_settings', 'kpred_renorm'),
        _field('method_settings', 'kscore'),
        _field('method_settings', 'rowrule'),
        _field('method_settings', 'adapter_refresh'),
        _field('method_settings', 'anchor'),
        _field('method_settings', 'anchor_cap'),
        _field('method_settings', 'anchor_coverage'),
        _field('method_settings', 'anchor_drej'),
        _field('method_settings', 'anchor_select'),
        _field('method_settings', 'anchor_select_graph'),
        _field('method_settings', 'anchor_select_home'),
        _field('method_settings', 'block_size_read'),
        _field('method_settings', 'chain_fixed_config'),
        _field('method_settings', 'drafts'),
        _field('method_settings', 'eval_j3_label'),
        _field('method_settings', 'eval_refresh_mode'),
        _field('method_settings', 'eval_rows'),
        _field('method_settings', 'f'),
        _field('method_settings', 'kpred_coverage'),
        _field('method_settings', 'kpred_frame'),
        _field('method_settings', 'kset'),
        _field('method_settings', 'kset_after_miss'),
        _field('method_settings', 'kset_cap'),
        _field('method_settings', 'kset_m'),
        _field('method_settings', 'kset_max'),
        _field('method_settings', 'kset_order'),
        _field('method_settings', 'kset_q'),
        _field('method_settings', 'miss_redraft'),
        _field('method_settings', 'miss_redraft_anchor'),
        _field('method_settings', 'miss_redraft_force'),
        _field('method_settings', 'miss_redraft_passes'),
        _field('method_settings', 'rows_rule_cap'),
        _field('method_settings', 'row_select'),
        _field('method_settings', 'row_select_exp'),
        _field('method_settings', 'row_select_length'),
        _field('method_settings', 'sync_passes'),
        _field('method_settings', 'tree_algo'),
        _field('method_settings', 'tree_b'),
        _field('method_settings', 'tree_spine'),
    ),
    'model': (
        _field('model_settings', 'adapter_config'),
        _field('model_settings', 'adapter_head'),
        _field('model_settings', 'blockhead'),
        _field('model_settings', 'checkpoint_dir'),
        _field('model_settings', 'drafter_correction_head'),
        _field('model_settings', 'draft_dtype'),
        _field('model_settings', 'eagle_draft_path'),
        _field('model_settings', 'heads_dir'),
        _field('model_settings', 'sync_drafter'),
        _field('model_settings', 'target_dtype'),
        _field('model_settings', 'target_release_weights'),
        _field('model_settings', 'eagle3_8b'),
        _field('model_settings', 'eagle3_qwen_32b'),
        _field('model_settings', 'eagle3_specforge_70b'),
    ),
}

# Execution qualifications travel with the layer whose facts they depend on.
# The engine lowers these closed defaults to its existing runtime fields.
MODEL_EXECUTION_FIELDS = frozenset((
    'adapter_gemm_tune', 'adapter_query_layout_fused', 'attention_backend', 'attn_sm100', 'attn_sm80',
    'c1_append_fold', 'c1_attn_len', 'c1_gateup_packed', 'c1_gluon', 'c1_kv_split_lean',
    'c1_qk_fused', 'c1_qk_warps', 'c1_span_kv_fused', 'candidate_decode_lean',
    'chain_admit_host_lean', 'chain_post_lean', 'chain_refresh_key_bucket',
    'ctx_kv_append', 'ctx_kv_direct', 'ctx_kv_fused',
    'decoder_attn_len', 'decoder_gqa_view', 'decoder_lmhead_kn', 'decoder_prefix_merge',
    'decoder_qkv_fused', 'decoder_select_radix', 'decoder_single_gateup_packed',
    'decoder_slot_write', 'draft_streams', 'hybrid_c1_fused', 'hybrid_c1_gluon', 'hybrid_c1_lean',
    'hybrid_decoder_refresh_state_lean', 'hybrid_fast_snapshot', 'hybrid_fused_norm',
    'hybrid_pair', 'hybrid_verify_flashinfer', 'hybrid_verify_triton', 'j0_lmhead_kn',
    'jpass_attn_len', 'k3_hazard_split', 'refresh_attn_lean', 'refresh_attn_tune',
    'refresh_gateup_packed', 'refresh_inject_inclusive', 'refresh_inject_lean2',
    'refresh_inject_mask_fused', 'refresh_kv_pack', 'refresh_prep_lean', 'refresh_qk_fused',
    'refresh_streams', 'refresh_swap_glue', 'verify_outcome_fused', 'verify_plan_lean', 'verify_plan_reuse',
    'verify_post_lean', 'verify_select_host', 'verify_select_lean',
))
METHOD_EXECUTION_DEFAULTS = {'refresh_swap': '1', 'refresh_fixed': '1',
                             'chain_miss_redraft': '1', 'refresh_swap_lean': '1'}


def _execution_defaults(value, fields, *, model=False):
    groups = {'defaults', 'legacy', 'sm80', 'sm100'}
    if not isinstance(value, dict) or not value or set(value) - (groups | {'capabilities', 'release_recipes'}):
        return False
    for group, settings in value.items():
        if group == 'capabilities':
            if (not model or not isinstance(settings, dict) or set(settings) != {'stateful'}
                    or type(settings['stateful']) is not bool):
                return False
        elif group == 'release_recipes':
            if (not model or not isinstance(settings, list) or not settings
                    or not all(_record(row, {
                        'architecture': lambda v: v in ('sm80', 'sm100'),
                        'block_size': _positive_int,
                        'schedule': lambda v: _record(v, {'j': _nonnegative_int, 'r': _nonnegative_int},
                                                    ('j', 'r')),
                        'recipe': _safe_file,
                    }, ('architecture', 'block_size', 'schedule', 'recipe')) for row in settings)):
                return False
        elif (not isinstance(settings, dict) or not settings or set(settings) - fields
              or any(item is not None and not isinstance(item, str) for item in settings.values())):
            return False
    return True


def _model_execution(value):
    return _execution_defaults(value, MODEL_EXECUTION_FIELDS, model=True)


def _method_execution(value):
    return _execution_defaults(value, set(METHOD_EXECUTION_DEFAULTS) | {
        'chain_refresh_key_bucket', 'refresh_attn_lean', 'refresh_inject_inclusive',
        'refresh_inject_lean2', 'refresh_kv_pack', 'refresh_prep_lean', 'refresh_swap_glue'})


METHOD_SCHEMA = Schema(
    {**METHOD_SCHEMA.fields, **dict.fromkeys(_HISTORICAL_FIELDS['method'],
        lambda value: value is None or isinstance(value, str)), 'method_execution': _method_execution},
    METHOD_SCHEMA.required, METHOD_SCHEMA.checks, METHOD_SCHEMA.atomic + ('method_execution',))

MODEL_SCHEMA = Schema(
    {**MODEL_SCHEMA.fields, **dict.fromkeys(_HISTORICAL_FIELDS['model'],
        lambda value: value is None or isinstance(value, str)), 'model_execution': _model_execution},
    MODEL_SCHEMA.required, MODEL_SCHEMA.checks, MODEL_SCHEMA.atomic + ('model_execution',))


RANK_CONFIG_FIELDS = {
    'method': {
        'explore.swap_rule': None,
        'explore.after_swap_fill': None,
        'explore.legacy_kpred': None,
        _field('method_settings', 'anchor_select'): None,
        _field('method_settings', 'anchor_select_graph'): '0',
        _field('method_settings', 'anchor_select_home'): None,
        _field('method_settings', 'kscore'): None,
        _field('method_settings', 'kset_after_miss'): None,
        _field('method_settings', 'row_select'): None,
        _field('method_settings', 'row_select_exp'): '1,1',
        _field('method_settings', 'row_select_length'): None,
        _field('method_settings', 'rowrule'): None,
    },
    'model': {
        'thresholds': None,
        'thresholds_sha256': None,
    },
}

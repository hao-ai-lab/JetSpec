"""Ported from the JetSpec++ reference implementation.

Public API: read_thresholds(root) -> dict; dataset_entry(entries, dataset);
HeadsBundle(thresholds, decoder_manifest); load_heads_bundle(root) -> HeadsBundle.
Only local JSON metadata is read. Decoder manifests use target_slice and carry
safe relative target-file SHA256 pins; the decoder loader verifies those files.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

__all__ = ['HeadsBundle', 'read_thresholds', 'dataset_entry', 'load_heads_bundle']


@dataclass(frozen=True)
class HeadsBundle:
    thresholds: dict
    decoder_manifest: dict


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'non-finite JSON number: {value}')


def _read_json(root, name):
    # A Hugging Face snapshot links into blobs/, so check the name, not the resolved path.
    if Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError(f'bundle metadata escapes its root: {name}')
    path = (Path(root) / name).resolve(strict=True)
    value = json.loads(path.read_text(), object_pairs_hook=_unique_object,
                       parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f'{name} must contain a JSON object')
    return value


def _value(entry):
    value = entry.get('value') if isinstance(entry, dict) else entry
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('threshold values must be finite numbers')
    return float(value)


def read_thresholds(root):
    """Read the bundle's calibration JSON afresh and validate consumed sections."""
    # Preserve serialized reference keys; expose no environment or serving I/O hooks.
    thresholds = _read_json(root, 'thresholds.json')
    # Legacy bundle keys are file-format identifiers, not configurable method names.
    for key in ('e04_kscore_blend', 'e04_rowrule_joint'):
        if not isinstance(thresholds.get(key), dict) or not thresholds[key]:
            raise ValueError('missing BLEND or joint calibration table')
    names = {'a', 'b', 'c', 'argmax', 'top', 'pos'}
    for coefficients in thresholds['e04_kscore_blend'].values():
        if not isinstance(coefficients, dict) or set(coefficients) != names:
            raise ValueError('BLEND requires exactly a, b, c, argmax, top and pos')
        for entry in coefficients.values():
            if not isinstance(entry, dict):
                raise ValueError('BLEND coefficient entries require a value field')
            _value(entry)
    for rule in thresholds['e04_rowrule_joint'].values():
        if not isinstance(rule, dict) or set(rule) != {'tau', 'temperature'}:
            raise ValueError('joint calibration requires exactly tau and temperature')
        if not isinstance(rule['temperature'], dict) or _value(rule['temperature']) <= 0:
            raise ValueError('joint temperature must be positive')
        table = rule['tau']
        if not isinstance(table, dict) or not table:
            raise ValueError('joint calibration requires a nonempty tau table')
        for label, entry in table.items():
            if re.fullmatch(r'pooled|J0d0|J[1-9][0-9]*', label) is None:
                raise ValueError(f'invalid tau label: {label}')
            if not 0 <= _value(entry) <= 1:
                raise ValueError('joint tau must be in [0, 1]')
    return thresholds


def dataset_entry(entries, dataset):
    """Resolve a per-dataset threshold, falling back to its pooled entry."""
    if dataset in entries and 'pooled' in entries:
        raise ValueError(f'ambiguous pooled and per-dataset thresholds for {dataset}')
    return entries[dataset] if dataset in entries else entries['pooled']


def load_heads_bundle(root):
    """Require valid local thresholds and a pinned target-slice decoder manifest."""
    thresholds = read_thresholds(root)
    manifest = _read_json(root, 'decoder/manifest.json')
    # Require the target-slice path; legacy checkpoint fallbacks use pickle files.
    if (set(manifest) - {'source', 'target_files', 'fallback_sha256'}
            or manifest.get('source') != 'target_slice'):
        raise ValueError('decoder manifest requires target_slice with no unexpected keys')
    pins = manifest.get('target_files')
    if not isinstance(pins, dict) or 'model.safetensors.index.json' not in pins:
        raise ValueError('decoder manifest requires target-file pins including the safetensors index')
    for name, digest in pins.items():
        path = Path(name)
        if (not name or path.is_absolute() or '..' in path.parts or '://' in name
                or path.suffix not in ('.json', '.safetensors')):
            raise ValueError(f'unsafe target-file pin: {name}')
        if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            raise ValueError(f'invalid SHA256 target-file pin: {name}')
    if not any(Path(name).suffix == '.safetensors' for name in pins):
        raise ValueError('decoder manifest requires a safetensors shard pin')
    if 'fallback_sha256' in manifest:
        digest = manifest['fallback_sha256']
        if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            raise ValueError('invalid fallback SHA256 pin')
    return HeadsBundle(thresholds, manifest)

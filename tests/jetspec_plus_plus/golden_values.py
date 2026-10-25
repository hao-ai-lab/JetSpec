"""Load frozen CPU reference values; repeated labels follow the case's call order."""
from collections import defaultdict
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).parent / 'fixtures/golden'
_case = None
_counts = defaultdict(int)
_documents = {}
_tensors = {}


def begin_case(nodeid):
    global _case
    _case = nodeid.split('::', 1)
    _case[0] = Path(_case[0]).stem
    _counts.clear()


def golden(label):
    file, case = _case
    if file not in _documents:
        _documents[file] = json.loads((ROOT / f'{file}.json').read_text())
        tensor_file = ROOT / f'{file}.safetensors'
        _tensors[file] = load_file(tensor_file) if tensor_file.exists() else {}
    index = _counts[label]
    _counts[label] += 1
    value = _documents[file][case][label][index]

    def decode(value):
        if isinstance(value, list):
            return [decode(item) for item in value]
        if isinstance(value, dict):
            if '__tensor__' in value:
                return _tensors[file][value['__tensor__']].clone()
            if '__tuple__' in value:
                return tuple(decode(item) for item in value['__tuple__'])
            if '__mapping__' in value:
                return {decode(k): decode(v) for k, v in value['__mapping__']}
            return {key: decode(item) for key, item in value.items()}
        return value

    return decode(value)

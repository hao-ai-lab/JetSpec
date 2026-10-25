"""Sequence-wise torch oracles preserve each sequence's reduction geometry.

C precedes every existing row/slot dimension. Models and weights are shared;
rows, pins and output buffers belong to one sequence. This is a correctness
reference, so it deliberately retains the single-sequence GEMM shapes.
"""
from dataclasses import fields, is_dataclass
from functools import wraps
from inspect import signature

import torch


def sequence_slice(value, index, count):
    if isinstance(value, torch.Tensor) and value.ndim:
        if value.shape[0] != count:
            raise ValueError('per-sequence tensors must have the same leading C')
        return value[index]
    if is_dataclass(value):
        return type(value)(**{f.name: sequence_slice(getattr(value, f.name), index, count)
                              for f in fields(value)})
    if isinstance(value, tuple):
        return tuple(sequence_slice(item, index, count) for item in value)
    if isinstance(value, dict):
        return {key: sequence_slice(item, index, count) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) != count:
            raise ValueError('per-sequence lists must have length C')
        return value[index]
    return value


def sequence_stack(values):
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(values)
    if is_dataclass(first):
        return type(first)(**{f.name: sequence_stack([getattr(v, f.name) for v in values])
                              for f in fields(first)})
    if isinstance(first, tuple):
        return tuple(sequence_stack(list(items)) for items in zip(*values))
    return tuple(values)


def sequencewise(tensor, ndim, *, shared=()):
    """Lift an oracle when its named tensor has an additional leading C axis.

    Tensor arguments (including nested pins and out buffers) and lists carry C; scalar
    arguments broadcast. ``shared`` names weights/tables/models without C.
    Legacy ranks call the original function without binding or tensor work.
    """
    def decorate(fn):
        parameters = signature(fn)
        position = tuple(parameters.parameters).index(tensor)

        @wraps(fn)
        def batched(*args, **kwargs):
            value = args[position] if len(args) > position else kwargs[tensor]
            if not isinstance(value, torch.Tensor) or value.ndim != ndim:
                return fn(*args, **kwargs)
            count = value.shape[0]
            if count < 1:
                raise ValueError('C must be positive')
            bound = parameters.bind(*args, **kwargs)
            outputs = []
            for index in range(count):
                call = parameters.bind(*args, **kwargs)
                call.arguments.update({name: item if name in shared else
                                       sequence_slice(item, index, count)
                                       for name, item in bound.arguments.items()})
                outputs.append(fn(*call.args, **call.kwargs))
            return sequence_stack(outputs)
        return batched
    return decorate


def prefix_lengths(prefix, lengths):
    """Validate right-padded [C, L] storage; only [:lengths[c]] is data."""
    if prefix.ndim != 2 or lengths.shape != (prefix.shape[0],) or not prefix.shape[0]:
        raise ValueError('prefix and prefix_lengths must have shapes [C, L] and [C]')
    if lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError('prefix_lengths must be an integer tensor')
    if bool(((lengths < 1) | (lengths > prefix.shape[1])).any()):
        raise ValueError('prefix_lengths must lie in 1..L')
    return lengths.tolist()

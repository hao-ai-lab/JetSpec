"""Target state inputs accompanying the existing attention forward context.

The attention Context is upstream-pinned. Keep target state metadata in a
context-local record, so graph setup can replace attention inputs while the
enclosing verification scope remains active.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace

import torch

from jetspec.engines.jetspec_plus_plus.models import context as attention_context


@dataclass(frozen=True)
class ForwardContext:
    verify: bool = False
    slot_ids: torch.Tensor | None = None
    prefill_lengths: torch.Tensor | None = None

    @property
    def attention(self):
        return attention_context.get_context()


_forward_context = ContextVar('target_forward_context', default=ForwardContext())


def get_forward_context():
    return _forward_context.get()


def set_verification(verify):
    _forward_context.set(replace(get_forward_context(), verify=bool(verify)))


@contextmanager
def forward_context(*, verify=False, slot_ids=None, prefill_lengths=None):
    token = _forward_context.set(ForwardContext(verify, slot_ids, prefill_lengths))
    try:
        yield get_forward_context()
    finally:
        _forward_context.reset(token)

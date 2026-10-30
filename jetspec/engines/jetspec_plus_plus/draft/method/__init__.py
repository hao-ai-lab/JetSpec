"""The JetSpec method, independent of execution engines."""
from .schedule import Schedule, parse_schedule

__all__ = ["Schedule", "parse_schedule", "Candidate", "MethodRound", "MethodState"]


def __getattr__(name):
    if name in ('Candidate', 'MethodRound', 'MethodState'):
        from . import round
        return getattr(round, name)
    raise AttributeError(name)

from __future__ import annotations

from dataclasses import dataclass
import torch
from jetspec.engines.jetspec_plus_plus.serving.sequence import Sequence
from abc import ABC, abstractmethod


@dataclass
class SpeculateResult:
    speculations: torch.Tensor
    logits_q: torch.Tensor
    cache_hits: torch.Tensor | None = None
    jetspec_verify_mask: torch.Tensor | list[int] | None = None
    jetspec_decisions: list[str] | None = None
    blockhead_anatomy: list[dict] | None = None


@dataclass
class VerifyResult:
    new_suffixes: list[list[int]]
    recovery_tokens: list[int]
    eagle_acts: torch.Tensor | None = None  # Is this a tensor?
    device_outcome: torch.Tensor | None = None


class SpeculatorBase(ABC):
    def __init__(self, lookahead: int, device: torch.device):
        self.lookahead = lookahead
        self.device = device

    @abstractmethod
    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        pass

    @abstractmethod
    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        pass


class VerifierBase(ABC):
    def __init__(self, lookahead: int, device: torch.device):
        self.lookahead = lookahead
        self.device = device

    @abstractmethod
    def prefill(self, seqs: list[Sequence], eagle: bool = False) -> VerifyResult:
        pass

    @abstractmethod
    def verify(self, seqs: list[Sequence], speculate_result: SpeculateResult, eagle: bool = False) -> VerifyResult:
        pass

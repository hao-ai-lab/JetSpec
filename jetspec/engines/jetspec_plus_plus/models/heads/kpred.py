"""Served five-member p-correction k predictor."""
from __future__ import annotations

import importlib.util
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

import jetspec.engines.jetspec_plus_plus.draft.run_manifest as run_manifest
from jetspec.engines.jetspec_plus_plus.models.heads.lac import keep_device


def pass_name(passes_run):
    from jetspec.engines.jetspec_plus_plus.draft.method.passes import completed_label
    passes_run = int(passes_run)
    from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled as eval_enabled
    if eval_enabled():
        from jetspec.engines.jetspec_plus_plus.draft.pass_labels import eval_completed_label
        return eval_completed_label(passes_run)
    label = completed_label(passes_run, max_depth=2)
    if os.environ.get("E04_LAB_PASS_LABEL") is not None:
        # E04_LAB_PASS_LABEL=J2 (tau lab, default unset): every block past J0 reads J2's thresholds
        label = lab_pass_name(passes_run, label)
    return label


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.kpred import assemble_fr_features


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.kpred import probabilities_from_logits




@dataclass(frozen=True)
class PredictorMember:
    head: torch.nn.Module
    scales: dict[str, float]


@dataclass(frozen=True)
class KPrediction:
    probabilities: torch.Tensor
    pass_label: str
    elapsed_ms: float


class KPredictor:
    def __init__(self, members, embedding, *, device):
        self.members = members
        self.embedding = embedding
        self.device = torch.device(device)
        if set(members) != {"J0d0", "J1", "J2"} or any(len(group) != 5 for group in members.values()):
            raise ValueError("k predictor requires five members for J0d0, J1, and J2")
        self._metric_calls = 0
        self._metric_total_ms = 0.0
        self._recorded_inputs = 0
        self._batched = {}
        self._prediction_graphs = {}
        self._graphs_warmed = False
        if os.environ.get('JETSPEC_HEADS_FAST') == '1':
            from torch.func import stack_module_state
            for label, group in members.items():
                heads = [member.head for member in group]
                signatures = [tuple((name, tuple(value.shape))
                                    for name, value in head.state_dict().items())
                              for head in heads]
                if all(signature == signatures[0] for signature in signatures[1:]):
                    params, buffers = stack_module_state(heads)
                    self._batched[label] = (heads[0], params, buffers)
            run_manifest.record_value('kpred_batched', {
                'batched': sorted(self._batched),
                'unbatched': sorted(set(members) - set(self._batched)),
            })














def _diagnostic_qhat(block, *, root, dataset, coverage):
    """Missing retired predictor groups are NA, not a joint-row selection error."""
    if os.environ.get('JETSPEC_HEADS_FAST') == '1':
        from jetspec.engines.jetspec_plus_plus.models.heads.fast import read_thresholds
        thresholds = read_thresholds(root)
    else:
        run_manifest.record_file_once("heads/thresholds", Path(root) / "thresholds.json")
        thresholds = json.loads((Path(root) / "thresholds.json").read_text())
    from jetspec.engines.jetspec_plus_plus.models.heads.fast import dataset_entry
    label = str(block.kpred_pass)
    groups = thresholds.get("k_predictor", {})
    if label not in groups:
        run_manifest.record_value('legacy_k_lac_diagnostic/' + label,
                                  'not_applicable_missing_group')
        return None
    return dataset_entry(groups[label], dataset)[str(coverage)]["value"]


def k15_in_lac(block, *, root, dataset, coverage):
    """Report whether the unmasked conformal set contained unsupported k=15."""
    q_hat = _diagnostic_qhat(block, root=root, dataset=dataset, coverage=coverage)
    if q_hat is None:
        return False
    return bool(block.k_probabilities[15] >= 1.0 - float(q_hat))


def k0_in_lac(block, *, root, dataset, coverage):
    """Report whether wire-frame serving excluded predictor-frame k=0."""
    q_hat = _diagnostic_qhat(block, root=root, dataset=dataset, coverage=coverage)
    if q_hat is None:
        return False
    return bool(block.k_probabilities[0] >= 1.0 - float(q_hat))

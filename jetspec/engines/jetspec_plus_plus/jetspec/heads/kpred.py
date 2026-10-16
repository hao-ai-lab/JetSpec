"""Served pass labels and conformal selection over k probabilities."""
from __future__ import annotations

from dataclasses import dataclass

import torch

import jetspec.engines.jetspec_plus_plus.jetspec.run_manifest as run_manifest
from jetspec.engines.jetspec_plus_plus.models.lac import keep_device


def pass_name(passes_run, run_config=None):
    from jetspec.engines.jetspec_plus_plus.draft.method.schedule import completed_label
    passes_run = int(passes_run)
    from jetspec.engines.jetspec_plus_plus.jetspec.profile import enabled as profile_enabled
    if profile_enabled():
        from jetspec.engines.jetspec_plus_plus.jetspec.pass_labels import eval_completed_label
        return eval_completed_label(passes_run, run_config=run_config)
    label = completed_label(passes_run, max_depth=2)
    return label


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.kpred import probabilities_from_logits
from jetspec.engines.jetspec_plus_plus.jetspec.config_values import config_values


@dataclass(frozen=True)
class KPrediction:
    probabilities: torch.Tensor
    pass_label: str
    elapsed_ms: float


def selected_hypotheses(block, *, root, dataset, coverage, cap, run_config=None):
    """Apply the pinned LAC threshold in the configured served-k frame."""
    if config_values(run_config).get('method.method_settings.row_select') == 'budget':
        return list(range(15))
    label = str(block.kpred_pass)
    coverage = str(coverage)
    if not dataset or coverage not in ("0.95", "0.99"):
        raise ValueError("predictor selection requires a known dataset and coverage 0.95 or 0.99")
    if config_values(run_config).get('engine.engine_settings.heads_fast') == '1':
        from jetspec.engines.jetspec_plus_plus.models.fast import read_thresholds
        thresholds = read_thresholds(run_config)
    else:
        from jetspec.engines.jetspec_plus_plus.models.fast import read_thresholds
        thresholds = read_thresholds(run_config)
    from jetspec.engines.jetspec_plus_plus.models.fast import dataset_entry
    q_hat = dataset_entry(thresholds["k_predictor"][label], dataset)[coverage]["value"]
    from jetspec.engines.jetspec_plus_plus.models.lac import lac_candidates
    wire_frame = config_values(run_config).get('method.method_settings.kpred_frame') == "wire"
    probabilities = block.k_probabilities
    if config_values(run_config).get('method.method_settings.kpred_renorm'):
        # e04 opt-in: LAC on p_pred[1+k] / (1 - p_pred[0]) (policy_lib "renorm"); keyed rows keep raw mass.
        if config_values(run_config)['method.method_settings.kpred_renorm'] != "1" or not wire_frame:
            raise ValueError("E04_KPRED_RENORM=1 requires JETSPEC_KPRED_FRAME=wire")
        probabilities = torch.as_tensor(probabilities).double()
        mass = 1.0 - probabilities[0]
        if not bool(mass > 0):
            raise ValueError("E04_KPRED_RENORM requires p_pred[0] < 1")
        probabilities = probabilities / mass
    ids, _ = lac_candidates(
        probabilities, q_hat=q_hat, cap=cap,
        excluded={0} if wire_frame else {15})
    values = [int(k) - 1 if wire_frame else int(k) for k in ids.tolist()]
    return sorted(values)


def _diagnostic_qhat(block, *, root, dataset, coverage, run_config=None):
    """Missing retired predictor groups are NA, not a joint-row selection error."""
    if config_values(run_config).get('engine.engine_settings.heads_fast') == '1':
        from jetspec.engines.jetspec_plus_plus.models.fast import read_thresholds
        thresholds = read_thresholds(run_config)
    else:
        from jetspec.engines.jetspec_plus_plus.models.fast import read_thresholds
        thresholds = read_thresholds(run_config)
    from jetspec.engines.jetspec_plus_plus.models.fast import dataset_entry
    label = str(block.kpred_pass)
    groups = thresholds.get("k_predictor", {})
    if label not in groups:
        run_manifest.record_value('legacy_k_lac_diagnostic/' + label,
                                  'not_applicable_missing_group', run_config=run_config)
        return None
    return dataset_entry(groups[label], dataset)[str(coverage)]["value"]


def k15_in_lac(block, *, root, dataset, coverage, run_config=None):
    """Report whether the unmasked conformal set contained unsupported k=15."""
    q_hat = _diagnostic_qhat(block, root=root, dataset=dataset, coverage=coverage, run_config=run_config)
    if q_hat is None:
        return False
    return bool(block.k_probabilities[15] >= 1.0 - float(q_hat))


def k0_in_lac(block, *, root, dataset, coverage, run_config=None):
    """Report whether wire-frame serving excluded predictor-frame k=0."""
    q_hat = _diagnostic_qhat(block, root=root, dataset=dataset, coverage=coverage, run_config=run_config)
    if q_hat is None:
        return False
    return bool(block.k_probabilities[0] >= 1.0 - float(q_hat))

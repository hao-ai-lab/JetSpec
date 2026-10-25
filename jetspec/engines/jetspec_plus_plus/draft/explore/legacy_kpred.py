"""Five-member serving predictor, selected by method.explore.legacy_kpred."""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
import torch
from jetspec.engines.jetspec_plus_plus.draft.method.ops import reference_op
from jetspec.engines.jetspec_plus_plus.models.lac import keep_device
from jetspec.engines.jetspec_plus_plus.jetspec import run_manifest
from jetspec.engines.jetspec_plus_plus.jetspec.config_values import config_values
from jetspec.engines.jetspec_plus_plus.jetspec.heads.kpred import (
    pass_name, KPrediction, selected_hypotheses, k15_in_lac, k0_in_lac, _diagnostic_qhat)
from jetspec.engines.jetspec_plus_plus.draft.method.decoder.kpred import probabilities_from_logits
from jetspec.engines.jetspec_plus_plus.draft.explore import register



def _reference_head_class():
    from .hazard_head import Head
    return Head


@dataclass(frozen=True)
class PredictorMember:
    head: torch.nn.Module
    scales: dict[str, float]


class KPredictor:
    def __init__(self, members, embedding, *, device, run_config=None):
        self.run_config = run_config
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
        if config_values(self.run_config).get('engine.engine_settings.heads_fast') == '1':
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
            }, run_config=run_config)


    def _member_scales(self, label, device):
        cache = self.__dict__.setdefault('_member_scale_columns', {})
        key = (label, device)
        if key not in cache:
            cache[key] = {
                name: torch.tensor([float(member.scales[name]) for member in self.members[label]],
                                   dtype=torch.float32, device=device)[:, None, None]
                for name in ('s', 'p_top1', 'stable', 'emb')}
        return cache[key]

    def _predict_batched(self, label, hidden, p_top1, stable, output_tokens):
        from torch.func import functional_call, vmap
        base, params, buffers = self._batched[label]
        embedded = self.embedding(output_tokens).float()
        # The member scales as one [members, 1, 1] float32 column each: the
        # same float32 divisors as the per-member Python floats, one launch
        # per feature group instead of one per member.
        scales = self._member_scales(label, hidden.device)
        features = torch.cat((
            hidden.float()[None] / scales['s'],
            (p_top1[None, :, None] / scales['p_top1']),
            (stable.float()[None, :, None] / scales['stable']),
            embedded[None] / scales['emb'],
        ), dim=-1)
        floor = torch.logit(p_top1.double().clamp(1e-6, 1 - 1e-6)).float()

        def call(one_params, one_buffers, one_features):
            return functional_call(
                base, (one_params, one_buffers), (one_features, floor))[0]

        return vmap(call)(params, buffers, features)

    def _predict_batched_graph(self, label, hidden, p_top1, stable, output_tokens):
        key = (label, hidden.dtype, tuple(hidden.shape))
        values = (hidden, p_top1, stable, output_tokens)
        entry = self._prediction_graphs.get(key)
        if entry is None:
            if self._graphs_warmed:
                raise RuntimeError(f'predictor graph missing after warm-up: {key}')
            inputs = tuple(value.clone() for value in values)

            def step():
                return probabilities_from_logits(
                    self._predict_batched(label, *inputs)).mean(0)

            current = torch.cuda.current_stream(hidden.device)
            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    step()
            current.wait_stream(stream)
            expected = step()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = step()
            graph.replay()
            if not torch.equal(expected, result):
                raise RuntimeError('k predictor graph replay changed probabilities')
            self._prediction_graphs[key] = (inputs, graph, result)
        else:
            inputs, graph, result = entry
            for destination, value in zip(inputs, values):
                destination.copy_(value)
            graph.replay()
        return result.clone()

    @classmethod
    def load(cls, root, embedding, *, device, run_config=None):
        root = Path(root)
        if config_values(run_config).get('engine.engine_settings.heads_fast') == '1':
            from jetspec.engines.jetspec_plus_plus.models.fast import read_thresholds
            read_thresholds(run_config)
        Head = _reference_head_class()
        groups = {}
        for label in ("J0d0", "J1", "J2"):
            members = []
            for seed in range(5):
                path = root / "kpred/arms/FR" / label / f"seed{seed}/ckpt.pt"
                saved = torch.load(path, map_location="cpu", weights_only=False)
                run_manifest.record_file(f"heads/kpred/{label}/seed{seed}", path, run_config=run_config)
                if saved["pas"] != label or int(saved["seed"]) != seed:
                    raise ValueError(f"predictor checkpoint identity mismatch: {path}")
                cfg = saved["cfg"]
                head = Head(
                    int(saved["d_base"]) + int(saved["d_app"]),
                    width=int(cfg["width"]),
                    n_out=int(cfg.get("n_out", 1)),
                    residual=cfg.get("residual"),
                    xattn=bool(cfg.get("xattn", False)),
                    xattn_seed=1_000_003 + seed,
                ).to(device=device, dtype=torch.float32)
                head.load_state_dict(saved["state_dict"], strict=True)
                head.eval().requires_grad_(False)
                members.append(PredictorMember(head=head, scales=dict(saved["scales"])))
            groups[label] = members
        predictor = cls(groups, embedding, device=device, run_config=run_config)
        if (predictor.device.type == 'cuda' and config_values(run_config).get('engine.engine_settings.heads_fast') == '1'
                and config_values(run_config).get('engine.engine_settings.rows_graphs') == '1'):
            hidden_size = head.l1.in_features - embedding.weight.shape[1] - 2
            predictor.warm_graphs(embedding.weight.new_zeros(15, hidden_size))
        return predictor

    @torch.inference_mode()
    def warm_graphs(self, hidden):
        """Warm all pass labels, including early convergence, before serving."""
        p_top1 = torch.full((15,), 0.5, device=hidden.device, dtype=torch.float32)
        stable = torch.zeros(15, device=hidden.device, dtype=torch.bool)
        output_tokens = torch.zeros(15, device=hidden.device, dtype=torch.long)
        for label in self._batched:
            self._predict_batched_graph(label, hidden, p_top1, stable, output_tokens)
        self._graphs_warmed = True






def _legacy_kpredictor():
    return KPredictor

METHOD_KEYS = ("explore.legacy_kpred",)
register(METHOD_KEYS[0], legacy_kpredictor=_legacy_kpredictor)

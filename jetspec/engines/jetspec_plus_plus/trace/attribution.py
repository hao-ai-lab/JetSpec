"""Opt-in first-divergence capture for JetSpec greedy verification."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch


class FirstDivergenceDumper:
    """Compare committed greedy ids to a reference and bank one raw logit row."""

    def __init__(
        self,
        reference: Path,
        output_dir: Path,
        targets: set[tuple[int, int]] | None = None,
    ):
        payload = json.loads(reference.read_text())
        self.expected = {
            int(row.get("index", index)): list(map(int, row["token_ids"]))
            for index, row in enumerate(payload["outputs"])
        }
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dumped: set[int] = set()
        self.targets = targets or set()
        self.captured_targets: set[tuple[int, int]] = set()

    @classmethod
    def from_env(cls, *, run_config=None) -> "FirstDivergenceDumper | None":
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        reference = setting(run_config, 'engine.engine_settings.logit_reference')
        output_dir = setting(run_config, 'engine.engine_settings.logit_dump_dir')
        if not reference and not output_dir:
            return None
        if not reference or not output_dir:
            raise ValueError(
                "JETSPEC_LOGIT_REFERENCE and JETSPEC_LOGIT_DUMP_DIR "
                "must be set together"
            )
        raw_targets = setting(run_config, 'engine.engine_settings.logit_targets', '')
        targets = {
            tuple(map(int, item.split(":", 1)))
            for item in raw_targets.split(",")
            if item
        }
        return cls(Path(reference), Path(output_dir), targets)

    def consider(
        self,
        *,
        prompt_index: int,
        seq_id: int,
        position: int,
        round_id: int | str,
        decision: str,
        actual: int,
        logits: torch.Tensor,
        source: str,
    ) -> bool:
        prompt_index = int(prompt_index)
        position = int(position)
        actual = int(actual)
        expected_row = self.expected.get(prompt_index)
        if expected_row is None or not 0 <= position < len(expected_row):
            return False
        expected = int(expected_row[position])
        target_key = (prompt_index, position)
        is_target = (
            target_key in self.targets
            and target_key not in self.captured_targets
        )
        is_divergence = actual != expected and prompt_index not in self.dumped
        if not is_target and not is_divergence:
            return False

        logits_cpu = logits.detach().to("cpu")
        if logits_cpu.ndim != 1:
            raise ValueError(
                f"expected one logit row, got shape={tuple(logits_cpu.shape)}"
            )
        record = {
            "prompt_index": prompt_index,
            "seq_id": int(seq_id),
            "position": position,
            "round": round_id,
            "decision": decision,
            "expected": expected,
            "actual": actual,
            "source": source,
            "kind": "divergence" if is_divergence else "target",
            "logits": logits_cpu,
        }
        prefix = "engine_stream" if is_divergence else "engine_target"
        path = self.output_dir / (
            f"{prefix}_{prompt_index:03d}_pos_{position:04d}.pt"
        )
        torch.save(record, path)
        metadata = {key: value for key, value in record.items() if key != "logits"}
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        if is_target:
            self.captured_targets.add(target_key)
        if is_divergence:
            self.dumped.add(prompt_index)
        print(
            "W217_ENGINE_LOGIT "
            f"kind={record['kind']} "
            f"stream={prompt_index} position={position} round={round_id} "
            f"decision={decision} actual={actual} expected={expected} "
            f"source={source} dump={path}",
            flush=True,
        )
        return True


def prompt_index(seq) -> int:
    value = getattr(seq, "jetspec_prompt_index", None)
    return int(seq.seq_id if value is None else value)

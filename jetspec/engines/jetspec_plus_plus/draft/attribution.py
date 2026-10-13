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
    def from_env(cls) -> "FirstDivergenceDumper | None":
        reference = os.environ.get("JETSPEC_LOGIT_REFERENCE")
        output_dir = os.environ.get("JETSPEC_LOGIT_DUMP_DIR")
        if not reference and not output_dir:
            return None
        if not reference or not output_dir:
            raise ValueError(
                "JETSPEC_LOGIT_REFERENCE and JETSPEC_LOGIT_DUMP_DIR "
                "must be set together"
            )
        raw_targets = os.environ.get("JETSPEC_LOGIT_TARGETS", "")
        targets = {
            tuple(map(int, item.split(":", 1)))
            for item in raw_targets.split(",")
            if item
        }
        return cls(Path(reference), Path(output_dir), targets)



def prompt_index(seq) -> int:
    value = getattr(seq, "jetspec_prompt_index", None)
    return int(seq.seq_id if value is None else value)

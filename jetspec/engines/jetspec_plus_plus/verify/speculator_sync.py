import torch

from jetspec.engines.jetspec_plus_plus.serving.sequence import Sequence
from jetspec.engines.jetspec_plus_plus.serving.model_runner import ModelRunner
from jetspec.engines.jetspec_plus_plus.serving.helpers.speculate_types import SpeculateResult; from jetspec.engines.jetspec_plus_plus.serving.helpers.speculate_types import VerifyResult; from jetspec.engines.jetspec_plus_plus.serving.helpers.speculate_types import SpeculatorBase


class SpeculatorSync(SpeculatorBase):

    def __init__(self, lookahead: int, device: torch.device, draft_model_runner: ModelRunner):
        super().__init__(lookahead, device)
        self.draft_model_runner = draft_model_runner

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        assert not verify_result.eagle_acts, "Eagle is not currently supported for synchronous speculation"
        print('[spec_prefill] target prefill', flush=True)
        self.draft_model_runner.call("run", seqs, True)

        if len(seqs) > 0:
            print(
                f"[PREFILL] seq0 prompt_len={seqs[0].num_prompt_tokens} recovery={seqs[0].recovery_token_id}", flush=True)

        return SpeculateResult([], [])


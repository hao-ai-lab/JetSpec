from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import torch
from time import perf_counter
from transformers import AutoTokenizer

from jetspec.engines.jetspec_plus_plus.serving.model_runner import ModelRunner
from jetspec.engines.jetspec_plus_plus.serving.sequence import Sequence
from jetspec.engines.jetspec_plus_plus.serving.scheduler import Scheduler
from jetspec.engines.jetspec_plus_plus.serving.speculate_types import SpeculatorBase
from jetspec.engines.jetspec_plus_plus.serving.speculate_types import VerifierBase
from jetspec.engines.jetspec_plus_plus.serving.speculate_types import VerifyResult
from jetspec.engines.jetspec_plus_plus.serving.misc import decode_tokens


class InferenceStep(ABC):

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    @abstractmethod
    def decode(self, seqs: list[Sequence]) -> int:
        pass

    @abstractmethod
    def prefill(self, seqs: list[Sequence]) -> int:
        pass


class AutoRegressiveStep(InferenceStep):

    def __init__(self, scheduler: Scheduler, model_runner: ModelRunner, tokenizer: AutoTokenizer, verbose: bool = False):
        super().__init__(scheduler)
        self.model_runner = model_runner
        self.tokenizer = tokenizer
        self.verbose = verbose

    def step(self, seqs: list[Sequence], is_prefill: bool) -> int:
        if self.verbose:
            print(f'[auto_regressive_step] is_prefill={is_prefill}', flush=True)

        timing = getattr(self.model_runner, '_anatomy_lite', None)
        if timing is not None and not is_prefill:
            timing.mark(5)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        if timing is not None and not is_prefill:
            timing.mark(6)

        if self.verbose:
            decoded_tokens = decode_tokens(token_ids, self.tokenizer)
            print(f"[auto_regressive_step] generated tokens: {decoded_tokens}", flush=True)

        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        return len(seqs) if not is_prefill else sum(len(seq) for seq in seqs)

    def prefill(self, seqs: list[Sequence]) -> int:
        return self.step(seqs, is_prefill=True)

    def decode(self, seqs: list[Sequence]) -> int:
        return self.step(seqs, is_prefill=False)


class SpecDecodeStep(InferenceStep):

    def __init__(
        self,
        scheduler: Scheduler,
        speculator: SpeculatorBase,
        verifier: VerifierBase,
        eagle: bool,
        tokenizer: AutoTokenizer,
        async_spec: bool,
        verbose: bool = False,
    ):
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        super().__init__(scheduler)
        self.speculator = speculator
        self.verifier = verifier
        self.run_config = getattr(verifier, 'run_config', None)
        speculator._anatomy_lite = getattr(verifier.target_model_runner, '_anatomy_lite', None)
        if (speculator._anatomy_lite is not None
                and setting(self, 'engine.engine_settings.device_host_stamps', '0') == '1'):
            speculator._anatomy_lite.observe_device_verify(verifier.metrics)
        verifier.target_model_runner._reference_tap_prepare_hook = (
            speculator.prepare_reference_taps
            if getattr(speculator, "_reference_tap_prepare", False) else None)
        kernel_profile = getattr(verifier, "_verify_kernel_profile", None)
        if kernel_profile is not None:
            speculator._verify_kernel_profile = kernel_profile
        if getattr(speculator, "_outcome_pack_presync", False):
            verifier._jetspec_presync_pack_hook = speculator.pack_device_outcome_presync
        if getattr(speculator, "_verify_record_late", False):
            verifier._jetspec_late_record_hook = speculator.finish_jetspec_late_record
        if getattr(speculator, "_verify_publish_first", False):
            if not getattr(speculator, "_verify_record_late", False):
                raise ValueError("publish-first requires late record")
            def finish_record_and_metrics(seq):
                speculator.finish_jetspec_late_record(seq)
                verifier.flush_jetspec_deferred_metrics()
            verifier._jetspec_late_record_hook = finish_record_and_metrics
            verifier._jetspec_publish_first = True
        self.eagle = eagle
        self.tokenizer = tokenizer
        self.async_spec = async_spec
        self.verbose = verbose
        if (getattr(speculator, "jetspec", False)
                and getattr(verifier.target_model_runner, "jetspec_paged_flashinfer", False)):
            speculator._verify_prestage_runner = verifier.target_model_runner

    def prefill(self, seqs: list[Sequence]) -> int:
        # When doing async speculation and not Eagle, we can do draft and target prefills in parallel.
        if (
            not self.eagle
            and self.async_spec
            and not getattr(self.speculator, "jetspec", False)
        ):
            empty_verify_result = VerifyResult([], [], None)
            self.speculator.prefill(seqs, empty_verify_result)
            verify_result = self.verifier.prefill(seqs, eagle=False)
        else:
            verify_result = self.verifier.prefill(seqs, eagle=self.eagle)
            self.speculator.prefill(seqs, verify_result)

        for seq in seqs:
            assert seq.recovery_token_id is not None
            seq.num_cached_tokens = seq.num_prompt_tokens
            seq.num_draft_cached_tokens = seq.num_prompt_tokens

        return sum(len(seq) for seq in seqs)

    def decode(self, seqs: list[Sequence]) -> int:
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        _prof = setting(self, 'engine.engine_settings.profile', '0') == "1"
        timing = getattr(getattr(self.verifier, 'target_model_runner', None), '_anatomy_lite', None)
        from jetspec.engines.jetspec_plus_plus.trace.host_census import mark as _census_mark
        if _prof:
            torch.cuda.synchronize()
            _t0 = perf_counter()

        # Save lightweight state instead of expensive clone_spec deep copy.
        # speculate() modifies: token_ids (append+extend), num_tokens, last_token, num_draft_cached_tokens
        # verify() modifies: num_cached_tokens (line 77 of verifier.py)
        # postprocess_speculate() needs the ORIGINAL state to apply new suffixes.
        saved = [(len(seq.token_ids), seq.num_tokens, seq.last_token, seq.num_draft_cached_tokens, seq.num_cached_tokens) for seq in seqs]

        eagle_sentinel = True if self.eagle else None
        in_verify_result = VerifyResult(
            new_suffixes=[],
            recovery_tokens=[],
            eagle_acts=eagle_sentinel,
        )
        #### STEP 1: SPECULATE ####
        _census_mark("save", run_config=getattr(self, 'run_config', None))
        if timing is not None:
            timing.stamp('speculate_start')
        speculate_result = self.speculator.speculate(seqs, in_verify_result)
        if timing is not None:
            timing.stamp('speculate_end')

        if _prof:
            torch.cuda.synchronize()
            _t1 = perf_counter()

        if self.verbose:
            speculations = speculate_result.speculations
            print(f"[SpecDecodeStep] speculations: {speculations}", flush=True)
            speculations_list = speculations.tolist()

            for i, speculation in enumerate(speculations_list):
                decoded_tokens = decode_tokens(speculation, self.tokenizer)
                print(f"[SpecDecodeStep] speculation {i}: {decoded_tokens}", flush=True)

        #### STEP 2: VERIFY ####
        if timing is not None:
            timing.mark(5)
        out_verify_result = self.verifier.verify(seqs, speculate_result, eagle=self.eagle)
        if timing is not None:
            timing.mark(6)
        device_outcome = getattr(out_verify_result, "device_outcome", None)
        if device_outcome is not None:
            kernel_profile = getattr(self.verifier, "_verify_kernel_profile", None)
            if kernel_profile is not None and kernel_profile.active is not None:
                kernel_profile.step("publish_head")
            if timing is not None:
                timing.mark(7)
            self.speculator.publish_device_outcome(seqs, device_outcome)
            if timing is not None:
                timing.mark(8)
            if getattr(self.verifier, "_jetspec_publish_first", False):
                self.verifier.finish_jetspec_postpublish(seqs[0])
            if kernel_profile is not None and kernel_profile.active is not None:
                kernel_profile.finish(**kernel_profile.round_meta)

        if _prof:
            torch.cuda.synchronize()
            _t2 = perf_counter()

        if self.verbose:
            recovery_tokens = out_verify_result.recovery_tokens
            new_suffixes = out_verify_result.new_suffixes
            for i, new_suffix in enumerate(new_suffixes):
                decoded_tokens = decode_tokens(new_suffix + [recovery_tokens[i]], self.tokenizer)
                print(f"[SpecDecodeStep] verification {i}: {decoded_tokens}", flush=True)

        # Restore original seq state before postprocess (undo speculate + verify modifications)
        if timing is not None:
            timing.stamp('restore_start')
        for seq, (orig_len, orig_nt, orig_lt, orig_ndc, orig_nct) in zip(seqs, saved):
            del seq.token_ids[orig_len:]
            seq.num_tokens = orig_nt
            seq.last_token = orig_lt
            seq.num_draft_cached_tokens = orig_ndc
            seq.num_cached_tokens = orig_nct
        _census_mark("restore", run_config=getattr(self, 'run_config', None))
        if timing is not None:
            timing.stamp('restore_end')

        #### STEP 3: POSTPROCESS ####
        self.scheduler.postprocess_speculate(
            seqs,
            out_verify_result.new_suffixes,
            out_verify_result.recovery_tokens,
            eagle_acts=out_verify_result.eagle_acts if self.eagle else None,
        )
        _census_mark("postprocess", run_config=getattr(self, 'run_config', None))
        if timing is not None:
            timing.stamp('postprocess_end')

        if _prof:
            torch.cuda.synchronize()
            _t3 = perf_counter()
            cache_hits = speculate_result.cache_hits
            hits_str = f"hits={cache_hits.sum().item()}/{len(cache_hits)}" if cache_hits is not None else ""
            toks = sum(len(s) for s in out_verify_result.new_suffixes)
            print(f"[PROFILE target] handshake={(_t1-_t0)*1000:.2f}ms verify={(_t2-_t1)*1000:.2f}ms postprocess={(_t3-_t2)*1000:.2f}ms total={(_t3-_t0)*1000:.2f}ms {hits_str} toks={toks}", flush=True)

        return sum(len(s) for s in out_verify_result.new_suffixes)

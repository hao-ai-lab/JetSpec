from jetspec.engines.jetspec_plus_plus.draft.tree_serve import tree_budget; from jetspec.engines.jetspec_plus_plus.draft.tree_serve import build_tree; from jetspec.engines.jetspec_plus_plus.draft.tree_serve import tree_api; from jetspec.engines.jetspec_plus_plus.draft.tree_serve import spine_indices; from jetspec.engines.jetspec_plus_plus.draft.tree_serve import chain_accept_depth
import os
import warnings
import torch
from time import perf_counter
from transformers import AutoTokenizer

from jetspec.engines.jetspec_plus_plus.serving.sequence import Sequence
from jetspec.engines.jetspec_plus_plus.serving.model_runner import ModelRunner
from jetspec.engines.jetspec_plus_plus.serving.utils.verify import verify
from jetspec.engines.jetspec_plus_plus.verify.helpers.greedy_accept import greedy_accept; from jetspec.engines.jetspec_plus_plus.verify.helpers.greedy_accept import greedy_accept_device; from jetspec.engines.jetspec_plus_plus.verify.helpers.greedy_accept import point_mass_accept_device
from jetspec.engines.jetspec_plus_plus.verify.helpers.device_verify import stage_committed_ids; from jetspec.engines.jetspec_plus_plus.verify.helpers.device_verify import finish_committed_ids; from jetspec.engines.jetspec_plus_plus.verify.helpers.device_verify import stage_recovery_id
from jetspec.engines.jetspec_plus_plus.verify.helpers.outcome_packet import hash16_committed; from jetspec.engines.jetspec_plus_plus.verify.helpers.outcome_packet import make_hash16_weights
from jetspec.engines.jetspec_plus_plus.serving.helpers.speculate_types import SpeculateResult; from jetspec.engines.jetspec_plus_plus.serving.helpers.speculate_types import VerifyResult; from jetspec.engines.jetspec_plus_plus.serving.helpers.speculate_types import VerifierBase
from jetspec.engines.jetspec_plus_plus.serving.utils.async_helpers.nccl_pack import append_committed_taps; from jetspec.engines.jetspec_plus_plus.serving.utils.async_helpers.nccl_pack import stage_committed_taps; from jetspec.engines.jetspec_plus_plus.serving.utils.async_helpers.nccl_pack import finish_staged_taps
from jetspec.engines.jetspec_plus_plus.draft.attribution import FirstDivergenceDumper; from jetspec.engines.jetspec_plus_plus.draft.attribution import prompt_index
from jetspec.engines.jetspec_plus_plus.wire.protocol import anchor_mode
from jetspec.engines.jetspec_plus_plus.wire.chain.contract import OUTCOME_WORDS; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import OutcomeOffset
from jetspec.engines.jetspec_plus_plus.draft.env import require_verify_device_batch




def _point_mass_logits_q(logits_q, speculations, temperatures_target, temperatures_draft):
    """Give greedy draft rows the logits of their actual proposed tokens."""
    rows = (temperatures_target > 0) & (temperatures_draft == 0)
    if not rows.any():
        return logits_q
    aligned = logits_q.clone()
    one_hot = torch.zeros_like(aligned[rows])
    one_hot.scatter_(2, speculations[rows, 1:].unsqueeze(-1), 1.0)
    aligned[rows] = one_hot
    return aligned


class Verifier(VerifierBase):
    _round_env = {
        "_jetspec_verify_device": "JETSPEC_VERIFY_DEVICE",
        "_jetspec_verify_reference": "JETSPEC_VERIFY_REFERENCE",
        "_jetspec_verify_sync_audit": "JETSPEC_VERIFY_SYNC_AUDIT",
        "_jetspec_chain_attn3": "JETSPEC_CHAIN_ATTN3",
        "_jetspec_verify_post_lean": "JETSPEC_VERIFY_POST_LEAN",
        "_jetspec_publish_first": "JETSPEC_VERIFY_PUBLISH_FIRST",
        "_jetspec_verify_outcome_fused": "JETSPEC_VERIFY_OUTCOME_FUSED",
        "_jetspec_verify_outcome_fused_check": "JETSPEC_VERIFY_OUTCOME_FUSED_CHECK",
    }

    def __getattr__(self, name):
        env_name = type(self)._round_env.get(name)
        if env_name is None:
            raise AttributeError(name)
        value = os.environ.get(env_name, "0") == "1"
        setattr(self, name, value)
        return value

    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        target_model_runner: ModelRunner,
        sampler_x: float | None = None,
        async_fan_out: int | None = None,
        jit_speculate: bool = False,
        tokenizer: AutoTokenizer = None,
        metrics: dict = None,
    ):
        super().__init__(lookahead, device)
        self.target_model_runner = target_model_runner
        self.sampler_x = sampler_x
        self.async_fan_out = async_fan_out
        self.jit_speculate = jit_speculate
        self.tokenizer = tokenizer
        self.metrics = metrics
        self.lossless_dumper = FirstDivergenceDumper.from_env()
        self.hybrid_verify_audit = os.environ.get("JETSPEC_HYBRID_VERIFY_AUDIT", "0") == "1"
        # The EAGLE target prefill returns its conditioning in place of logits, so the fork skips
        # the dumper there; JETSPEC_LOGIT_DUMP_EAGLE=1 asks the target for both (registered in env.py).
        self.dump_eagle_prefill = os.environ.get("JETSPEC_LOGIT_DUMP_EAGLE", "0") == "1"
        self._jetspec_offsets: dict[int, int] = {}
        self.oracle_anchor = getattr(getattr(self.target_model_runner, "config", None), "jetspec", False) and anchor_mode() == "oracle"
        self.perseq = os.environ.get("JETSPEC_PERSEQ", "0") == "1"
        self._hash16_weights = None
        self._jetspec_deferred_metrics = []
        if os.environ.get("JETSPEC_VERIFY_KERNEL_PROFILE"):
            from jetspec.engines.jetspec_plus_plus.draft.verify_kprofile import VerifyKernelProfile
            self._verify_kernel_profile = VerifyKernelProfile.from_env()
            self.target_model_runner._verify_kernel_profile = self._verify_kernel_profile

    def prefill(self, seqs: list[Sequence], eagle: bool = False) -> VerifyResult:
        if self.hybrid_verify_audit and any(seq.temperature > 0 for seq in seqs):
            raise ValueError("JETSPEC_HYBRID_VERIFY_AUDIT only supports T=0 argmax auditing")
        from jetspec.engines.jetspec_plus_plus.draft.hybrid_pair import eager_hybrid_pair
        if (self._jetspec_verify_device or eager_hybrid_pair(self.target_model_runner)) and self._hash16_weights is None:
            self._hash16_weights = make_hash16_weights(self.device)
        if (self.lossless_dumper is not None or self.hybrid_verify_audit) and (not eagle or self.dump_eagle_prefill):
            result = self.target_model_runner.call("run", seqs, True, True, True)
            if eagle:
                token_ids, eagle_acts, prefill_logits = result
            else:
                token_ids, prefill_logits = result
        else:
            result = self.target_model_runner.call("run", seqs, True)
            prefill_logits = None
            if eagle:
                token_ids, eagle_acts = result
            else:
                token_ids = result

        offset = 0
        for index, (seq, token_id) in enumerate(zip(seqs, token_ids)):
            seq.recovery_token_id = token_id
            if self.hybrid_verify_audit:
                top = prefill_logits[index].float().topk(2)
                greedy = int(prefill_logits[index].argmax().item())
                self.metrics.setdefault("jetspec_hybrid_prefill_verify", []).append({
                    "prompt_index": prompt_index(seq), "emitted": int(token_id),
                    "verify_argmax": greedy,
                    "verify_top2_ids": top.indices.tolist(),
                    "verify_top2_logits": top.values.tolist(),
                })
                if int(token_id) != greedy:
                    raise AssertionError("hybrid prefill token differs from target verify argmax")
            if self.lossless_dumper is not None and prefill_logits is not None:
                self.lossless_dumper.consider(
                    prompt_index=prompt_index(seq),
                    seq_id=seq.seq_id,
                    position=0,
                    round_id="prefill",
                    decision="prefill",
                    actual=token_id,
                    logits=prefill_logits[index],
                    source="prefill",
                )
            if eagle:
                seq_len = seq.num_prompt_tokens
                # this doesn't move acts onto cpu does it? 
                seq.last_target_hidden_state = eagle_acts[offset + seq_len - 1].clone()
                offset += seq_len

        return VerifyResult(
            [], # no accepted tokens for prefill, just recovery tokens (first sampled token).
            [seq.recovery_token_id for seq in seqs],
            eagle_acts if eagle else None,
        )

    def verify(self, seqs: list[Sequence], speculate_result: SpeculateResult, eagle: bool = False) -> VerifyResult:
        """Verify speculative tokens using the target model."""
        if speculate_result.jetspec_verify_mask is not None:
            return self._verify_jetspec(seqs, speculate_result)
        if (os.environ.get("SSD_VERIFY_DEVICE", "0") == "1"
                and speculate_result.blockhead_anatomy is not None):
            require_verify_device_batch(len(seqs))
            if seqs[0].draft_temperature not in (None, 0):
                raise ValueError("SSD_VERIFY_DEVICE requires a greedy draft (draft_temperature None or 0)")
            if not seqs[0].temperature >= 0:
                raise ValueError("SSD_VERIFY_DEVICE requires target temperature >= 0")
            if os.environ.get("SSD_VERIFY_SYNC_AUDIT", "0") == "1":
                torch.cuda.set_sync_debug_mode("warn")
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter("always")
                    result = self._verify_jetspec_device(seqs[0], speculate_result)
                self.metrics.setdefault("ssd_device_sync_warnings", []).append(
                    sum("synchroniz" in str(warning.message).lower()
                        for warning in seen))
                return result
            return self._verify_jetspec_device(seqs[0], speculate_result)
        if getattr(self.target_model_runner, "hybrid_target", False):
            raise RuntimeError("hybrid speculative verify requires SSD_VERIFY_DEVICE=1")
        _prof = os.environ.get("SSD_PROFILE", "0") == "1"
        batch_size = len(seqs)

        if _prof:
            torch.cuda.synchronize()
            _vt0 = perf_counter()

        _pt = os.environ.get("SSD_PROFILE_TARGET", "0") == "1"
        _tv0 = perf_counter()
        result = self.target_model_runner.call("run", seqs, False, False, True)

        if _prof:
            torch.cuda.synchronize()
            _vt1 = perf_counter()

        if _pt:
            torch.cuda.synchronize()
            _vt_call = perf_counter()
            print(f"[PROFILE verifier] target_call={(_vt_call-_tv0)*1000:.2f}ms eagle={eagle} bs={batch_size}", flush=True)

        if eagle:
            logits_p_flat, eagle_acts_flat = result
        elif self.target_model_runner.config.jetspec_taps:
            logits_p_flat, jetspec_taps_flat = result
        else:
            logits_p_flat = result

        for s in seqs:
            s.num_cached_tokens += self.lookahead + 1

        logits_p = logits_p_flat.view(
            batch_size, self.lookahead + 1, -1)  # [b, k+1, v]

        # Build per-seq temps for target verify and draft q respectively.
        temps_target = [seq.temperature for seq in seqs]
        temps_draft = [
            seq.draft_temperature if seq.draft_temperature is not None else seq.temperature
            for seq in seqs
        ]
        temperatures_target = torch.tensor(temps_target, dtype=torch.float32, device=self.device)
        temperatures_draft = torch.tensor(temps_draft, dtype=torch.float32, device=self.device)

        new_suffixes, recovery_tokens = verify(
            logits_p=logits_p,
            logits_q=_point_mass_logits_q(
                speculate_result.logits_q, speculate_result.speculations,
                temperatures_target, temperatures_draft),
            speculations=speculate_result.speculations,
            temperatures_target=temperatures_target,
            temperatures_draft=temperatures_draft,
            cache_hits=speculate_result.cache_hits,
            sampler_x=self.sampler_x,
            async_fan_out=self.async_fan_out,
            jit_speculate=self.jit_speculate,
        )

        if self.target_model_runner.config.jetspec_taps:
            jetspec_taps = jetspec_taps_flat.view(
                batch_size, self.lookahead + 1, 5, -1
            )
            for seq, candidates, new_suffix in zip(
                seqs, jetspec_taps, new_suffixes
            ):
                append_committed_taps(seq, candidates, len(new_suffix))

        self.metrics["target_verify_times"].append(perf_counter() - _tv0)

        if _prof:
            torch.cuda.synchronize()
            _vt2 = perf_counter()
            print(f"[PROFILE verify] target_fwd={(_vt1-_vt0)*1000:.2f}ms verify_compute={(_vt2-_vt1)*1000:.2f}ms", flush=True)


        # # Debug: print recovery tokens detokenized
        if getattr(self.target_model_runner.config, "verbose", False) and recovery_tokens is not None and len(recovery_tokens) > 0:
            recovery_texts = []
            for token in recovery_tokens:
                try:
                    text = self.tokenizer.decode([token], skip_special_tokens=False)
                    recovery_texts.append(text)
                except Exception:
                    recovery_texts.append(f"<token_id:{token}>")
            print(f"[verify] recovery tokens: {recovery_texts}", flush=True)

        self.metrics["accepted_suffix_lens_with_recovery"].extend(
            [len(s) for s in new_suffixes])

        # For async mode, also track accepted suffix lengths only for cache hits
        if speculate_result.cache_hits is not None:
            _ch_cpu = speculate_result.cache_hits.cpu()
            self.metrics["cache_hits"].append(_ch_cpu.float().mean().item())
            hit_count = int(_ch_cpu.sum().item())
            self.metrics["cache_hit_count"] += hit_count
            self.metrics["cache_miss_count"] += _ch_cpu.numel() - hit_count
            for i, suffix_len in enumerate([len(s) for s in new_suffixes]):
                if _ch_cpu[i] == 1:
                    self.metrics["accepted_suffix_lens_on_hit"].append(suffix_len)
                else:
                    self.metrics["accepted_suffix_lens_on_miss"].append(suffix_len)

        # Print mean length of new suffixes for monitoring
        if getattr(self.target_model_runner.config, "verbose", False) and new_suffixes:
            mean_suffix_len = sum([len(suffix) for suffix in new_suffixes]) / len(new_suffixes)
            print(f"[verify] mean new suffix length: {mean_suffix_len:.2f}", flush=True)

        eagle_acts = None
        if eagle:
            eagle_acts = eagle_acts_flat.view(batch_size, self.lookahead + 1, -1)
        
        return VerifyResult(
            new_suffixes=new_suffixes,
            recovery_tokens=recovery_tokens,
            eagle_acts=eagle_acts,
        )

    def _verify_jetspec(
        self,
        seqs: list[Sequence],
        speculate_result: SpeculateResult,
    ) -> VerifyResult:
        """Run full block verification for hits and a true one-token AR for misses."""
        require_verify_device_batch(len(seqs))
        if any(getattr(seq, "draft_temperature", None) not in (None, 0) for seq in seqs):
            raise ValueError("JetSpec verification requires a greedy draft")
        reference = getattr(self, "_jetspec_verify_reference", False)
        if reference:
            if (len(seqs) != 1 or seqs[0].temperature != 0 or self.lookahead != 15
                    or self._jetspec_verify_device or tree_budget() or self.oracle_anchor
                    or getattr(self.target_model_runner.config, "enforce_eager", False)
                    or not self.target_model_runner.config.jetspec_taps):
                raise ValueError("reference verify requires graphed N=1 T0 b16 with taps")
            reference_start = seqs[0].num_cached_tokens
        from jetspec.engines.jetspec_plus_plus.draft.hybrid_pair import eager_hybrid_pair
        if eager_hybrid_pair(getattr(self, 'target_model_runner', None)):
            if len(seqs) != 1 or not seqs[0].temperature >= 0 or tree_budget() or self.oracle_anchor:
                raise ValueError('eager hybrid pair requires N=1 target temperature >= 0, no tree budget or oracle anchor')
            # Reuse the pair protocol's accept/state/outcome work exactly once.
            # The runner executes its target forward directly under enforce_eager.
            return self._verify_jetspec_device(seqs[0], speculate_result)
        if (self._jetspec_verify_device and len(seqs) == 1
                and seqs[0].temperature >= 0 and not tree_budget()
                and not self.oracle_anchor and self.lossless_dumper is None):
            if self._jetspec_verify_sync_audit:
                torch.cuda.set_sync_debug_mode("warn")
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter("always")
                    result = self._verify_jetspec_device(seqs[0], speculate_result)
                self.metrics.setdefault("jetspec_device_sync_warnings", []).append(
                    sum("synchroniz" in str(warning.message).lower()
                        for warning in seen))
                return result
            return self._verify_jetspec_device(seqs[0], speculate_result)
        if not reference and getattr(getattr(self, "target_model_runner", None), "hybrid_target", False):
            raise ValueError("hybrid target requires device verification to commit its state")
        started = perf_counter()
        timing = getattr(self.target_model_runner, '_anatomy_lite', None)
        if timing is not None:
            timing.stamp('native_verify_start')
        mask = speculate_result.jetspec_verify_mask
        verify_mask = (torch.as_tensor(mask, dtype=torch.bool) if isinstance(mask, list)
                       else mask.to(torch.bool).cpu())
        full_indices = [i for i, value in enumerate(verify_mask.tolist()) if value]
        ar_indices = [i for i, value in enumerate(verify_mask.tolist()) if not value]
        from jetspec.engines.jetspec_plus_plus.serving.presets.budget import budget_type
        if ar_indices and budget_type().uncut:
            raise RuntimeError("uncut verify cannot execute an AR fallback")
        if reference and ar_indices:
            raise RuntimeError("reference verify refuses an AR step without a full tap block")
        new_suffixes = [None] * len(seqs)
        recovery_tokens = [None] * len(seqs)
        records = self.metrics["jetspec_round_records"][-len(seqs):]
        by_seq = {int(record["seq_id"]): record for record in records}
        independent_chain = not tree_budget() and os.environ.get('W264_INDEPENDENT_RECORDS', '0') == '1'
        if independent_chain:
            for index, seq in enumerate(seqs):
                record = by_seq[seq.seq_id]
                width = self.lookahead + 1 if index in full_indices else 1
                record['chain_prefix_ids'] = list(seq.token_ids[:seq.num_cached_tokens])
                record['chain_query_ids'] = list(seq.token_ids[seq.num_cached_tokens:seq.num_cached_tokens+width])
                if len(record['chain_query_ids']) != width or record['chain_query_ids'][0] != seq.recovery_token_id:
                    raise RuntimeError('CHAIN_INPUT_LINKAGE')

        mixed_logits = mixed_taps = None
        full_rows = len(full_indices) * (self.lookahead + 1)
        if self.perseq and full_indices and ar_indices:
            # Keep each sequence contiguous; scatter accepted results back below.
            ordered = [seqs[i] for i in full_indices + ar_indices]
            result = self.target_model_runner.call("run_jetspec_mixed", ordered)
            if self.target_model_runner.config.jetspec_taps:
                mixed_logits, mixed_taps = result
            else:
                mixed_logits = result

        if full_indices and tree_budget():
            full_seqs = [seqs[i] for i in full_indices]
            full_suffixes, full_recoveries = self._verify_jetspec_tree_hits(full_seqs, by_seq)
            for index, suffix, recovery in zip(full_indices, full_suffixes, full_recoveries):
                new_suffixes[index] = suffix
                recovery_tokens[index] = recovery
        elif full_indices:
            full_seqs = [seqs[i] for i in full_indices]
            if mixed_logits is None:
                result = self.target_model_runner.call(
                    "run", full_seqs, False, False, True
                )
            elif mixed_taps is None:
                result = mixed_logits[:full_rows]
            else:
                result = mixed_logits[:full_rows], mixed_taps[:full_rows]
            if self.target_model_runner.config.jetspec_taps:
                logits_flat, taps_flat = result
            else:
                logits_flat = result
                taps_flat = None
            for seq in full_seqs:
                seq.num_cached_tokens += self.lookahead + 1
            logits_p = logits_flat.view(
                len(full_indices), self.lookahead + 1, -1
            )
            full_predictions = None
            if not reference and all(
                seq.temperature == 0 and (seq.draft_temperature is None or seq.draft_temperature == 0)
                for seq in full_seqs
            ):
                proposal_rows = [seq.token_ids[-(self.lookahead + 1):] for seq in full_seqs]
                decisions = greedy_accept(logits_p, None, proposal_rows=proposal_rows,
                                          return_predictions=self.oracle_anchor)
                full_suffixes, full_recoveries = decisions[:2]
                if self.oracle_anchor:
                    full_predictions = decisions[2]
            else:
                temperatures_target = torch.tensor(
                    [seq.temperature for seq in full_seqs],
                    dtype=torch.float32,
                    device=self.device,
                )
                temperatures_draft = torch.tensor(
                    [
                        seq.draft_temperature
                        if seq.draft_temperature is not None
                        else 0
                        for seq in full_seqs
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
                if reference and full_indices == [0]:
                    full_suffixes, full_recoveries = verify(
                        logits_p=logits_p,
                        logits_q=_point_mass_logits_q(
                            speculate_result.logits_q, speculate_result.speculations,
                            temperatures_target, temperatures_draft),
                        speculations=speculate_result.speculations,
                        temperatures_target=temperatures_target,
                        temperatures_draft=temperatures_draft,
                        cache_hits=speculate_result.cache_hits,
                        sampler_x=self.sampler_x,
                        async_fan_out=self.async_fan_out,
                        jit_speculate=self.jit_speculate,
                    )
                else:
                    full_suffixes, full_recoveries = verify(
                        logits_p=logits_p,
                        logits_q=_point_mass_logits_q(
                            speculate_result.logits_q[full_indices],
                            speculate_result.speculations[full_indices],
                            temperatures_target, temperatures_draft),
                        speculations=speculate_result.speculations[full_indices],
                        temperatures_target=temperatures_target,
                        temperatures_draft=temperatures_draft,
                        cache_hits=(speculate_result.cache_hits[full_indices] if reference
                                    else torch.ones(len(full_indices), dtype=torch.int64,
                                                    device=self.device)),
                        sampler_x=self.sampler_x if reference else None,
                        async_fan_out=self.async_fan_out if reference else None,
                        jit_speculate=self.jit_speculate if reference else True,
                    )
            if self.oracle_anchor and full_predictions is None:
                full_predictions = logits_p.argmax(-1).tolist()
            if reference and getattr(self.target_model_runner, "hybrid_target", False):
                # The reference accept has synchronized its host count. Commit
                # that GDN snapshot before the existing hybrid envelope is sent.
                self.target_model_runner.commit_hybrid_state(torch.tensor(
                    len(full_suffixes[0]), dtype=torch.int64, device=self.device))
            if taps_flat is not None:
                taps = taps_flat.view(
                    len(full_indices), self.lookahead + 1, 5, -1
                )
                for seq, candidates, suffix in zip(
                    full_seqs, taps, full_suffixes
                ):
                    append_committed_taps(seq, candidates, len(suffix))
            for local_index, (index, seq, suffix, recovery) in enumerate(
                zip(full_indices, full_seqs, full_suffixes, full_recoveries)
            ):
                new_suffixes[index] = suffix
                recovery_tokens[index] = recovery
                if reference and getattr(self, "hybrid_verify_audit", False):
                    record = by_seq[seq.seq_id]
                    predictions = logits_p[local_index].argmax(-1).tolist()
                    emitted = [int(token) for token in suffix[1:]] + [int(recovery)]
                    record["verify_argmax_ids"] = predictions
                    record["verify_emitted_ids"] = emitted
                    if emitted != predictions[:len(emitted)]:
                        raise AssertionError("reference verify emitted a non-argmax token")
                    self.metrics["jetspec_hybrid_verify_rows"] = (
                        self.metrics.get("jetspec_hybrid_verify_rows", 0) + 1)
                    self.metrics["jetspec_hybrid_verify_tokens"] = (
                        self.metrics.get("jetspec_hybrid_verify_tokens", 0) + len(emitted))
                if independent_chain:
                    self._save_chain_logits(by_seq[seq.seq_id], logits_p[local_index, :len(suffix)])
                if self.oracle_anchor:
                    by_seq[seq.seq_id]['greedy_predictions'] = full_predictions[local_index]
                if self.lossless_dumper is not None:
                    record = by_seq[seq.seq_id]
                    generated_offset = self._jetspec_offsets.get(seq.seq_id, 0)
                    for suffix_index in range(1, len(suffix)):
                        self.lossless_dumper.consider(
                            prompt_index=prompt_index(seq),
                            seq_id=seq.seq_id,
                            position=generated_offset + suffix_index,
                            round_id=record["round"],
                            decision=record["decision"],
                            actual=suffix[suffix_index],
                            logits=logits_p[local_index, suffix_index - 1],
                            source="block_verify_draft",
                        )
                    self.lossless_dumper.consider(
                        prompt_index=prompt_index(seq),
                        seq_id=seq.seq_id,
                        position=generated_offset + len(suffix),
                        round_id=record["round"],
                        decision=record["decision"],
                        actual=recovery,
                        logits=logits_p[local_index, len(suffix) - 1],
                        source="block_verify_recovery",
                    )

        if ar_indices:
            ar_seqs = [seqs[i] for i in ar_indices]
            if tree_budget():
                ar_started = perf_counter()
                ar_start_event = torch.cuda.Event(enable_timing=True)
                ar_end_event = torch.cuda.Event(enable_timing=True)
                ar_start_event.record()
            if mixed_logits is not None:
                logits = mixed_logits[full_rows:]
                temperatures = self.target_model_runner.prepare_sample(ar_seqs)
                ar_recoveries = self.target_model_runner.sampler(
                    logits, temperatures
                ).tolist()
                if mixed_taps is not None:
                    for index, seq in enumerate(ar_seqs):
                        append_committed_taps(
                            seq, mixed_taps[full_rows + index:full_rows + index + 1], 1
                        )
                ar_logits = logits if self.lossless_dumper is not None or independent_chain else None
            elif self.lossless_dumper is not None or independent_chain:
                ar_recoveries, ar_logits = self.target_model_runner.call(
                    "run", ar_seqs, False, True, True
                )
            else:
                ar_recoveries = self.target_model_runner.call(
                    "run", ar_seqs, False
                )
                ar_logits = None
            if tree_budget():
                ar_end_event.record(); ar_end_event.synchronize()
                for seq in ar_seqs:
                    by_seq[seq.seq_id].update(tree_hit=False, tree_budget=tree_budget(),
                        tree_nodes=0, tree_topk_present=False, tree_graph_replay=False,
                        tree_build_ms=0., realized_queries=1,
                        verify_forward_ms=ar_start_event.elapsed_time(ar_end_event),
                        verify_forward_wall_ms=(perf_counter()-ar_started)*1000)
            for local_index, (index, seq, recovery) in enumerate(
                zip(ar_indices, ar_seqs, ar_recoveries)
            ):
                new_suffixes[index] = [int(seq.recovery_token_id)]
                recovery_tokens[index] = int(recovery)
                if independent_chain:
                    self._save_chain_logits(by_seq[seq.seq_id], ar_logits[local_index:local_index+1])
                if self.oracle_anchor:
                    # An AR miss has only its one real target prediction.
                    by_seq[seq.seq_id]['greedy_predictions'] = [int(recovery)] + [-1] * 14
                if ar_logits is not None and self.lossless_dumper is not None:
                    record = by_seq[seq.seq_id]
                    generated_offset = self._jetspec_offsets.get(seq.seq_id, 0)
                    self.lossless_dumper.consider(
                        prompt_index=prompt_index(seq),
                        seq_id=seq.seq_id,
                        position=generated_offset + 1,
                        round_id=record["round"],
                        decision=record["decision"],
                        actual=recovery,
                        logits=ar_logits[local_index],
                        source="ar_recovery",
                    )

        assert all(suffix is not None for suffix in new_suffixes)
        assert all(token is not None for token in recovery_tokens)
        for seq, suffix in zip(seqs, new_suffixes):
            self._jetspec_offsets[seq.seq_id] = (
                self._jetspec_offsets.get(seq.seq_id, 0) + len(suffix)
            )
        if timing is not None:
            timing.stamp('native_verify_end')
        self.metrics["target_verify_times"].append(perf_counter() - started)
        self.metrics["accepted_suffix_lens_with_recovery"].extend(
            [len(suffix) for suffix in new_suffixes]
        )
        self.metrics["jetspec_verify_round_ms"].append(
            (perf_counter() - started) * 1000
        )
        self.metrics["jetspec_ar_steps"] += len(ar_indices)
        self.metrics["jetspec_full_verify_steps"] += len(full_indices)
        for seq, suffix, recovery in zip(
            seqs, new_suffixes, recovery_tokens
        ):
            record = by_seq[seq.seq_id]
            record["accepted_len"] = len(suffix)
            record["accepted_ids"] = [int(token) for token in suffix[1:]]
            record["next_correction"] = int(recovery)
            if self.oracle_anchor:
                record['verify_complete_ns'] = int(perf_counter() * 1e9)
        if reference:
            from jetspec.engines.jetspec_plus_plus.verify.helpers.outcome_packet import build_reference_outcome
            if timing is not None:
                timing.stamp('envelope_start')
            outcome = build_reference_outcome(
                seqs[0], new_suffixes[0], recovery_tokens[0], taps[0],
                start=reference_start,
                max_model_len=self.target_model_runner.config.max_model_len,
                eos_id=getattr(self.tokenizer, "eos_token_id", None),
                record=records[-1])
            if timing is not None:
                timing.stamp('envelope_end')
            return VerifyResult(new_suffixes, recovery_tokens, device_outcome=outcome)
        return VerifyResult(
            new_suffixes=new_suffixes,
            recovery_tokens=recovery_tokens,
            eagle_acts=None,
        )

    def _verify_jetspec_device(self, seq: Sequence, speculate_result: SpeculateResult) -> VerifyResult:
        """N=1 verify with one small host outcome read after device commit."""
        from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled as eval_enabled
        # Recovery is staged for the next suffix, not accepted in this one.
        # Eval must consume a pending EOS through its full speculative round.
        recovery_eos_terminal = not eval_enabled()
        temperature = getattr(seq, "temperature", 0)
        if temperature > 0 and getattr(self, "hybrid_verify_audit", False):
            raise ValueError("JETSPEC_HYBRID_VERIFY_AUDIT only supports T=0 argmax auditing")
        from jetspec.engines.jetspec_plus_plus.serving.helpers.host_census import mark as _census_mark
        _census_mark("verify_entry")
        started = perf_counter()
        blockhead = getattr(speculate_result, "blockhead_anatomy", None) is not None
        if blockhead:
            full = True
        else:
            mask = speculate_result.jetspec_verify_mask
            if isinstance(mask, torch.Tensor) and mask.is_cuda:
                raise RuntimeError("device verify needs the host-known N=1 query width")
            full = bool(mask[0])
        width = self.lookahead + 1 if full else 1
        proposal = speculate_result.speculations[0, :width]
        kernel_profile = getattr(self, "_verify_kernel_profile", None)
        profiled = not blockhead and kernel_profile is not None and kernel_profile.begin()
        if profiled:
            kernel_profile.step("runner_stage")
        logits, taps, site_ms = self.target_model_runner.call(
            "run_jetspec_mixed_device", seq, proposal, full)
        verify_top2 = logits.float().topk(2, dim=-1) if getattr(self, "hybrid_verify_audit", False) else None
        verify_argmax = logits.argmax(dim=-1) if getattr(self, "hybrid_verify_audit", False) else None
        launched = perf_counter()
        if profiled:
            kernel_profile.step("late_record")
        late_record = getattr(self, "_jetspec_late_record_hook", None)
        if late_record is not None:
            late_record(seq)
        for site, duration in site_ms.items():
            self.metrics.setdefault(f"jetspec_device_{site}_host_ms", []).append(duration)
        _census_mark("logits")
        if profiled:
            kernel_profile.step("accept")
        if temperature == 0:
            counts, recoveries = greedy_accept_device(logits[None], proposal[None])
        else:
            temperatures = torch.tensor([temperature], dtype=torch.float32, device=logits.device)
            counts, recoveries = point_mass_accept_device(
                logits[None], proposal[None], temperatures,
                sampler_x=self.sampler_x, async_fan_out=self.async_fan_out)
        accepted = perf_counter()
        _census_mark("accept")
        count, recovery = counts[0], recoveries[0]
        suffix_len = count + 1
        if getattr(self.target_model_runner, "hybrid_target", False):
            self.target_model_runner.commit_hybrid_state(suffix_len)
        outcome_fused = getattr(self, "_jetspec_verify_outcome_fused", False)
        if outcome_fused:
            if blockhead or self.device.type != "cuda" or not self._jetspec_verify_device:
                raise RuntimeError("JETSPEC_VERIFY_OUTCOME_FUSED requires N=1 CUDA JetSpec device verify, not blockhead")
        else:
            if profiled:
                kernel_profile.step("stage_ids")
            stage_committed_ids(seq, proposal,
                capacity=self.target_model_runner.config.max_model_len + self.lookahead + 2)
            if taps is not None:
                if profiled:
                    kernel_profile.step("stage_taps_first")
                stage_committed_taps(seq, taps,
                    capacity=self.target_model_runner.config.max_model_len + self.lookahead + 1)

        if blockhead:
            # Same replay, greedy decision and device commit as V. The local
            # block head consumes accepted target taps directly; it has no wire.
            if taps is None:
                raise RuntimeError("block-head device verify requires target hidden states")
            stage_committed_taps(seq, taps,
                capacity=self.target_model_runner.config.max_model_len + self.lookahead + 1)
            host_pair = torch.empty(2, dtype=torch.int64, device="cpu",
                                    pin_memory=self.device.type == "cuda")
            host_pair.copy_(torch.stack((count, recovery)),
                            non_blocking=self.device.type == "cuda")
            if self.device.type == "cuda":
                mirror_event = torch.cuda.Event(enable_timing=False)
                mirror_event.record(torch.cuda.current_stream(self.device))
                mirror_event.synchronize()
            committed, recovery_id = int(host_pair[0]) + 1, int(host_pair[1])
            suffix = seq.token_ids[seq.num_cached_tokens:seq.num_cached_tokens + committed]
            if len(suffix) != committed:
                raise RuntimeError("device verify host proposal mirror is incomplete")
            finish_staged_taps(seq, committed)
            finish_committed_ids(seq, committed)
            seq.num_cached_tokens += width
            self.metrics["target_verify_times"].append(perf_counter() - started)
            self.metrics["accepted_suffix_lens_with_recovery"].append(committed)
            self.metrics.setdefault("ssd_verify_round_ms", []).append(
                (perf_counter() - started) * 1000)
            self.metrics.setdefault("ssd_device_launch_host_ms", []).append(
                (launched - started) * 1000)
            self.metrics.setdefault("ssd_device_accept_host_ms", []).append(
                (accepted - launched) * 1000)
            self.metrics.setdefault("ssd_device_commit_host_ms", []).append(
                (perf_counter() - accepted) * 1000)
            return VerifyResult([suffix], [recovery_id])

        fused_check = outcome_fused and getattr(self, "_jetspec_verify_outcome_fused_check", False)
        if outcome_fused:
            from jetspec.engines.jetspec_plus_plus.verify.helpers.outcome_fused import launch_outcome_fused; from jetspec.engines.jetspec_plus_plus.verify.helpers.outcome_fused import prepare_outcome_storage
            row_k, row_a = getattr(seq, "_jetspec_verified_row_key", (-1, -1))
            storage, tap_storage, tap_base = prepare_outcome_storage(
                seq, proposal, taps,
                token_capacity=self.target_model_runner.config.max_model_len + self.lookahead + 2,
                tap_capacity=self.target_model_runner.config.max_model_len + self.lookahead + 1)
            host_words = torch.empty(OUTCOME_WORDS, dtype=torch.int64,
                                     device="cpu", pin_memory=True)
            outcome = torch.empty(OUTCOME_WORDS, dtype=torch.int64, device=self.device)
            if profiled:
                kernel_profile.step("outcome_fused")
            launch_outcome_fused(
                proposal, count, recovery, storage, taps, tap_storage,
                self._hash16_weights, outcome, start=seq.num_cached_tokens,
                tap_base=tap_base, seq_id=seq.seq_id,
                round_id=seq.jetspec_round + 1, row_k=row_k, row_a=row_a,
                prompt_tokens=seq.num_prompt_tokens,
                max_new_tokens=seq.max_new_tokens,
                max_model_len=self.target_model_runner.config.max_model_len,
                eos_id=getattr(self.tokenizer, "eos_token_id", None),
                ignore_eos=seq.ignore_eos,
                recovery_eos_terminal=recovery_eos_terminal)
            seq._jetspec_device_outcome = outcome
            _census_mark("outcome_head")
            _census_mark("recovery_id")
            if fused_check:
                from types import SimpleNamespace
                scratch = SimpleNamespace(
                    num_cached_tokens=seq.num_cached_tokens, token_ids=seq.token_ids,
                    _jetspec_token_storage=torch.empty_like(storage),
                    _jetspec_device_valid_length=seq.num_cached_tokens,
                    jetspec_taps=seq.jetspec_taps)
                scratch._jetspec_token_storage[:seq.num_cached_tokens].copy_(
                    storage[:seq.num_cached_tokens])
                stage_committed_ids(scratch, proposal, capacity=storage.numel())
                scratch_length = seq.num_cached_tokens + suffix_len
                stage_recovery_id(scratch, scratch_length, recovery)
                scratch_words = torch.zeros(OUTCOME_WORDS, dtype=torch.int64,
                                            device="cpu", pin_memory=True)
                scratch_words[OutcomeOffset.CMD] = 1
                scratch_words[OutcomeOffset.SEQ_ID] = seq.seq_id
                scratch_words[OutcomeOffset.ROUND] = seq.jetspec_round + 1
                scratch_words[OutcomeOffset.K] = row_k
                scratch_words[OutcomeOffset.A] = row_a
                scratch_outcome = scratch_words.to(self.device, non_blocking=True)
                scratch_outcome[OutcomeOffset.TRUE_K].copy_(count)
                scratch_outcome[OutcomeOffset.RECOVERY].copy_(recovery)
                scratch_outcome[OutcomeOffset.TAP_COUNT].copy_(suffix_len)
                scratch_outcome[OutcomeOffset.L].copy_(scratch_length)
                scratch_outcome[OutcomeOffset.HASH16].copy_(
                    hash16_committed(scratch._jetspec_token_storage,
                                     scratch_length + 1, self._hash16_weights))
                eos_id = getattr(self.tokenizer, "eos_token_id", None)
                eos = torch.zeros((), dtype=torch.bool, device=self.device)
                if eos_id is not None and not seq.ignore_eos:
                    within = torch.arange(width, device=self.device) < suffix_len
                    eos = (proposal.eq(eos_id) & within).any()
                    if recovery_eos_terminal:
                        eos = eos | recovery.eq(eos_id)
                prior = seq.num_cached_tokens - seq.num_prompt_tokens
                limit = ((prior + suffix_len >= seq.max_new_tokens)
                         | (seq.num_cached_tokens + suffix_len >= self.target_model_runner.config.max_model_len))
                scratch_outcome[OutcomeOffset.EOS_FLAG].copy_((eos | limit).to(torch.int64))
                if taps is not None:
                    scratch._jetspec_tap_storage = torch.empty_like(tap_storage)
                    stage_committed_taps(scratch, taps, capacity=tap_storage.shape[0])
        else:
            if profiled:
                kernel_profile.step("packet_head")
            host_words = torch.zeros(OUTCOME_WORDS, dtype=torch.int64,
                                     device="cpu", pin_memory=self.device.type == "cuda")
            host_words[OutcomeOffset.CMD] = 1
            host_words[OutcomeOffset.SEQ_ID] = seq.seq_id
            host_words[OutcomeOffset.ROUND] = seq.jetspec_round + 1
            row_k, row_a = getattr(seq, "_jetspec_verified_row_key", (-1, -1))
            host_words[OutcomeOffset.K] = row_k
            host_words[OutcomeOffset.A] = row_a
            outcome = host_words.to(self.device, non_blocking=True)
            if profiled:
                kernel_profile.step("outcome_writes")
            outcome[OutcomeOffset.TRUE_K].copy_(count)
            outcome[OutcomeOffset.RECOVERY].copy_(recovery)
            outcome[OutcomeOffset.TAP_COUNT].copy_(suffix_len)
            committed_length = seq.num_cached_tokens + suffix_len
            _census_mark("outcome_head")
            if profiled:
                kernel_profile.step("recovery_id")
            stage_recovery_id(seq, committed_length, recovery)
            _census_mark("recovery_id")
            if profiled:
                kernel_profile.step("hash")
            outcome[OutcomeOffset.L].copy_(committed_length)
            outcome[OutcomeOffset.HASH16].copy_(
                hash16_committed(seq._jetspec_token_storage, committed_length + 1,
                                 self._hash16_weights))
            if profiled:
                kernel_profile.step("eos")
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            eos = torch.zeros((), dtype=torch.bool, device=self.device)
            if eos_id is not None and not seq.ignore_eos:
                within = torch.arange(width, device=self.device) < suffix_len
                eos = (proposal.eq(eos_id) & within).any()
                if recovery_eos_terminal:
                    eos = eos | recovery.eq(eos_id)
            prior = seq.num_cached_tokens - seq.num_prompt_tokens
            limit = ((prior + suffix_len >= seq.max_new_tokens)
                     | (seq.num_cached_tokens + suffix_len >= self.target_model_runner.config.max_model_len))
            outcome[OutcomeOffset.EOS_FLAG].copy_((eos | limit).to(torch.int64))
            seq._jetspec_device_outcome = outcome

        # Post the packet D2H as soon as it is complete, then enqueue tap
        # staging before the one wait needed by host sequence metadata.
        if profiled:
            kernel_profile.step("packet_mirror")
        host_words.copy_(outcome, non_blocking=outcome.is_cuda)
        if outcome.is_cuda:
            mirror_event = getattr(seq, "_jetspec_packet_mirror_event", None)
            if mirror_event is None:
                mirror_event = torch.cuda.Event(enable_timing=False)
                seq._jetspec_packet_mirror_event = mirror_event
            mirror_event.record(torch.cuda.current_stream(self.device))
        if taps is not None and not outcome_fused:
            if profiled:
                kernel_profile.step("stage_taps_second")
            stage_committed_taps(seq, taps,
                capacity=self.target_model_runner.config.max_model_len + self.lookahead + 1)
        presync_pack = getattr(self, "_jetspec_presync_pack_hook", None)
        if presync_pack is not None:
            if profiled:
                kernel_profile.step("presync_pack")
            presync_pack(seq)
        _census_mark("pre_sync")
        if profiled:
            kernel_profile.step("sync")
        if outcome.is_cuda:
            mirror_event.synchronize()
        if fused_check:
            mismatch = not torch.equal(outcome, scratch_outcome)
            mismatch |= not torch.equal(
                storage[:seq.num_cached_tokens + width],
                scratch._jetspec_token_storage[:seq.num_cached_tokens + width])
            mismatch |= not torch.equal(
                storage.index_select(0, scratch_length.reshape(1)),
                scratch._jetspec_token_storage.index_select(0, scratch_length.reshape(1)))
            if taps is not None:
                mismatch |= not torch.equal(
                    tap_storage[tap_base:tap_base + width],
                    scratch._jetspec_tap_storage[tap_base:tap_base + width])
            self._outcome_fused_check_rounds = getattr(self, "_outcome_fused_check_rounds", 0) + 1
            self._outcome_fused_check_mismatches = getattr(self, "_outcome_fused_check_mismatches", 0) + int(mismatch)
            print(f"OUTCOME_FUSED_CHECK rounds={self._outcome_fused_check_rounds} "
                  f"mismatches={self._outcome_fused_check_mismatches}", flush=True)
        if profiled:
            kernel_profile.step("verify_post")
            kernel_profile.round_meta = dict(seq_id=seq.seq_id,
                                             round=seq.jetspec_round + 1,
                                             width=width, full=full)
        _census_mark("sync")
        chain_attn3 = getattr(self, "_jetspec_chain_attn3", None)
        if chain_attn3 is None:
            chain_attn3 = os.environ.get("JETSPEC_CHAIN_ATTN3", "0") == "1"
        if chain_attn3:
            seq._jetspec_packet_sync_at = perf_counter()
        words = host_words.tolist() if getattr(self, "_jetspec_verify_post_lean", False) else host_words
        committed = int(words[OutcomeOffset.TAP_COUNT])
        recovery_id = int(words[OutcomeOffset.RECOVERY])
        packet_l = int(words[OutcomeOffset.L])
        if packet_l != seq.num_cached_tokens + committed:
            raise RuntimeError("verify packet length differs from accepted prefix")
        seq._jetspec_last_packet_l = packet_l
        seq._jetspec_last_packet_eos = int(words[OutcomeOffset.EOS_FLAG])
        suffix = seq.token_ids[seq.num_cached_tokens:seq.num_cached_tokens + committed]
        if len(suffix) != committed:
            raise RuntimeError("device verify host proposal mirror is incomplete")
        if verify_top2 is not None:
            top_ids = verify_top2.indices.tolist()
            top_logits = verify_top2.values.tolist()
            argmax_ids = verify_argmax.tolist()
            record = self.metrics["jetspec_round_records"][-1]
            record["verify_top2_ids"] = top_ids
            record["verify_top2_logits"] = top_logits
            record["verify_argmax_ids"] = argmax_ids
            record["verify_emitted_ids"] = [int(token) for token in suffix[1:]] + [recovery_id]
            for position, token in enumerate(record["verify_emitted_ids"]):
                if token != argmax_ids[position]:
                    raise AssertionError(
                        f"hybrid verify argmax mismatch seq={seq.seq_id} "
                        f"round={seq.jetspec_round + 1} row={position} "
                        f"emitted={token} verify={argmax_ids[position]}")
            self.metrics["jetspec_hybrid_verify_rows"] = (
                self.metrics.get("jetspec_hybrid_verify_rows", 0) + 1)
            self.metrics["jetspec_hybrid_verify_tokens"] = (
                self.metrics.get("jetspec_hybrid_verify_tokens", 0) + len(record["verify_emitted_ids"]))
        if getattr(self, "_jetspec_publish_first", False):
            seq._jetspec_device_tap_count = committed
            seq._jetspec_last_recovery_id = recovery_id
            self._jetspec_pending_postpublish = (
                seq, suffix, committed, recovery_id, width, full, taps is not None,
                started, launched, accepted, perf_counter(),
                self.metrics["jetspec_round_records"][-1])
            _census_mark("verify_post")
            return VerifyResult([suffix], [recovery_id], device_outcome=outcome)
        if taps is not None:
            finish_staged_taps(seq, committed)
        finish_committed_ids(seq, committed)
        seq._jetspec_device_tap_count = committed
        seq.num_cached_tokens += width
        self._jetspec_offsets[seq.seq_id] = self._jetspec_offsets.get(seq.seq_id, 0) + committed
        self.metrics["target_verify_times"].append(perf_counter() - started)
        self.metrics["accepted_suffix_lens_with_recovery"].append(committed)
        self.metrics["jetspec_verify_round_ms"].append((perf_counter() - started) * 1000)
        self.metrics["jetspec_ar_steps"] += int(not full)
        self.metrics["jetspec_full_verify_steps"] += int(full)
        record = self.metrics["jetspec_round_records"][-1]
        record["accepted_len"] = committed
        record["accepted_ids"] = [int(token) for token in suffix[1:]]
        record["next_correction"] = recovery_id
        self.metrics.setdefault("jetspec_device_launch_host_ms", []).append(
            (launched - started) * 1000)
        self.metrics.setdefault("jetspec_device_accept_host_ms", []).append(
            (accepted - launched) * 1000)
        self.metrics.setdefault("jetspec_device_commit_host_ms", []).append(
            (perf_counter() - accepted) * 1000)
        _census_mark("verify_post")
        return VerifyResult([suffix], [recovery_id], device_outcome=outcome)






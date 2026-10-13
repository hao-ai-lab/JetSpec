import os
import time
import jetspec.engines.jetspec_plus_plus.serving.paths as ssd  # noqa: F401 — sets TORCH_CUDA_ARCH_LIST before flashinfer import

from jetspec.engines.jetspec_plus_plus.serving.config import Config
from jetspec.engines.jetspec_plus_plus.serving.sampling_params import SamplingParams
from jetspec.engines.jetspec_plus_plus.serving.utils.misc import infer_model_family
from jetspec.engines.jetspec_plus_plus.serving.sequence import Sequence
from jetspec.engines.jetspec_plus_plus.serving.scheduler import Scheduler
from jetspec.engines.jetspec_plus_plus.serving.model_runner import ModelRunner
from jetspec.engines.jetspec_plus_plus.models.qwen3 import JETSPEC_TARGET_LAYER_IDS
from jetspec.engines.jetspec_plus_plus.draft.draft_runner import DraftRunner
from jetspec.engines.jetspec_plus_plus.verify.speculator_async import SpeculatorAsync
from jetspec.engines.jetspec_plus_plus.verify.speculator_sync import SpeculatorSync
from jetspec.engines.jetspec_plus_plus.verify.speculator_blockhead import SpeculatorBlockhead
from jetspec.engines.jetspec_plus_plus.verify.step import InferenceStep; from jetspec.engines.jetspec_plus_plus.verify.step import AutoRegressiveStep; from jetspec.engines.jetspec_plus_plus.verify.step import SpecDecodeStep
from jetspec.engines.jetspec_plus_plus.verify.verifier import Verifier
import jetspec.engines.jetspec_plus_plus.serving.helpers.host_census as host_census
from jetspec.engines.jetspec_plus_plus.draft.blockhead_serve import blockhead_config_overrides; from jetspec.engines.jetspec_plus_plus.draft.blockhead_serve import complete_metric_rounds
from jetspec.engines.jetspec_plus_plus.wire.protocol import summarize_candidate_kinds
from jetspec.engines.jetspec_plus_plus.draft.sync_drafter import sync_drafter_config_overrides; from jetspec.engines.jetspec_plus_plus.draft.sync_drafter import sync_drafter_dir

import atexit
import contextlib
from dataclasses import fields
from time import perf_counter
from time import monotonic_ns
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp



@contextlib.contextmanager
def draft_mps_client_environment():
    """Expose the opt-in draft-only MPS client knobs to the draft child's spawn only."""
    draft_mps = {
        "CUDA_MPS_CLIENT_PRIORITY": os.environ.get("JETSPEC_MPS_DRAFT_PRIORITY"),
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": os.environ.get(
            "JETSPEC_MPS_DRAFT_ACTIVE_THREAD_PERCENTAGE"),
    }
    saved = {key: os.environ.get(key) for key in draft_mps}
    try:
        for key, value in draft_mps.items():
            if value is not None:
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


METRICS = {
    "cache_hits": [],
    "cache_hit_count": 0,
    "cache_miss_count": 0,
    "accepted_suffix_lens_with_recovery": [],
    "accepted_suffix_lens_on_hit": [],  # Only for cache hits in async mode
    "accepted_suffix_lens_on_miss": [],  # Only for cache misses in async mode
    "prefill_total_time": 0,
    "decode_total_time": 0,
    "prefill_total_tokens": 0,
    "decode_total_tokens": 0,
    "target_step_times": [],
    "target_verify_times": [],
    "anatomy_lite": [],
    "jetspec_tap_payload_bytes": [],
    "jetspec_tap_send_ms": [],
    "jetspec_hit_count": 0,
    "jetspec_miss_count": 0,
    "jetspec_late_count": 0,
    "jetspec_late_fraction": 0.0,
    "jetspec_stale_count": 0,
    "jetspec_coalesced_count": 0,
    "jetspec_gap_histogram": [],
    "jetspec_round_records": [],
    "blockhead_round_records": [],
    "jetspec_kind_summary": {},
    "jetspec_verify_round_ms": [],
    "jetspec_draft_round_ms": [],
    "jetspec_ar_steps": 0,
    "jetspec_full_verify_steps": 0,
}


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.blockhead_checkpoint_dir = os.environ.get("JETSPEC_BLOCKHEAD")
        config_kwargs = blockhead_config_overrides(
            self.blockhead_checkpoint_dir, config_kwargs
        )
        self.sync_drafter_checkpoint_dir = sync_drafter_dir()
        config_kwargs = sync_drafter_config_overrides(
            self.sync_drafter_checkpoint_dir, config_kwargs
        )
        config = Config(model, **config_kwargs)
        self.config = config
        self.split_draft = os.environ.get("JETSPEC_DRAFT_SPLIT", "") == "verify_gpu"
        if os.environ.get("JETSPEC_DRAFT_SPLIT", "") not in ("", "verify_gpu"):
            raise ValueError("JETSPEC_DRAFT_SPLIT must be verify_gpu or unset")
        if self.split_draft and (not config.jetspec or not config.draft_async or config.num_gpus != 3):
            raise ValueError("split draft requires async JetSpec with three logical ranks")
        Sequence.block_size = config.kvcache_block_size 

        assert config.kvcache_block_size >= (
            2 * config.speculate_k + 2), "ERROR: support for block size < 2*k+2 is not implemented"
        assert config.num_gpus > 1 or not config.draft_async, "ERROR: draft_async requires at least 2 gpus"
            
        # Check that target and draft are from the same family
        if config.speculate:
            target_family = infer_model_family(config.model)
            draft_family = infer_model_family(config.draft)
            assert target_family == draft_family, f"ERROR: target model family and draft model family must match"

        self.ps = []
        self.events = []

        ctx = mp.get_context("spawn")
        # One shared-memory name per run, inherited by every spawned rank.
        os.environ.setdefault("SSD_SHM_NAME", f"ssd_{os.getpid()}")
        self.num_tp_gpus = (1 if self.split_draft else
                            config.num_gpus if not self.config.draft_async else config.num_gpus - 1)

        if config.speculate and config.draft_async:
            self.draft_ps = None

        for i in range(1, self.num_tp_gpus):
            if self.config.verbose:
                print(f'creating ModelRunner process {i}', flush=True)
            event = ctx.Event()
            # can't pass kwargs through ctx.Process args
            process = ctx.Process(target=ModelRunner, args=(
                config, i, event, False, self.num_tp_gpus))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        if self.config.verbose:
            print(
                f'config.speculate = {config.speculate} and config.draft_async = {config.draft_async} about to create draft runner', flush=True)

        if config.speculate and config.draft_async:
            init_q = ctx.Queue()
            draft_rank = config.num_gpus - 1
            self.draft_ps = []
            for rank in ((draft_rank, 1) if self.split_draft else (draft_rank,)):
                process = ctx.Process(target=DraftRunner, args=(config, rank, init_q))
                if self.split_draft and rank == draft_rank:
                    # spawn imports CUDA modules before DraftRunner.__init__.
                    # Give the dedicated child its own device before spawn.
                    keys = ("CUDA_VISIBLE_DEVICES", "CUDA_MPS_PIPE_DIRECTORY",
                            "CUDA_MPS_LOG_DIRECTORY")
                    saved = {key: os.environ.get(key) for key in keys}
                    try:
                        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SSD_SPLIT_DEDICATED_UUID"]
                        os.environ.pop("CUDA_MPS_PIPE_DIRECTORY", None)
                        os.environ.pop("CUDA_MPS_LOG_DIRECTORY", None)
                        process.start()
                    finally:
                        for key, value in saved.items():
                            if value is None:
                                os.environ.pop(key, None)
                            else:
                                os.environ[key] = value
                else:
                    with draft_mps_client_environment():
                        process.start()
                self.draft_ps.append(process)
            print(
                f'Draft runner created on rank {draft_rank} (async)!', flush=True)

        # modelRunner(0) will wait on all 5 processes, so other 4 need to have launched by now
        self.model_runner = ModelRunner(
            config, 0, self.events, is_draft=False, num_tp_gpus=self.num_tp_gpus)
        if (os.environ.get("JETSPEC_ANATOMY_LITE", "0") != "0"
                and config.max_num_seqs == 1):
            from jetspec.engines.jetspec_plus_plus.draft.anatomy_lite import RoundTiming; from jetspec.engines.jetspec_plus_plus.draft.anatomy_lite import target_host_stamps
            self.model_runner._anatomy_lite = RoundTiming(
                host_stamps=target_host_stamps(config.jetspec, os.environ))
        else:
            self.model_runner._anatomy_lite = None

        # do this after so we can launch model runner above so that the q is actually populated
        if config.speculate and config.draft_async:
            try:
                num_blocks = min(init_q.get(timeout=180) for _ in self.draft_ps)
            except Exception as e:
                raise RuntimeError(
                    "ERROR: Timed out waiting for draft kv cache size") from e

            init_q.close()
            self.draft_cfg = DraftRunner.create_draft_config(config)
            self.draft_cfg.num_kvcache_blocks = num_blocks  # set for block manager to knwo

            self.prev_allocated_blocks = None
            self.prev_blocks_per_fork = None

        if config.speculate and not config.draft_async:
            # keep it colocated on rank 0, process/dist agnostic in this case
            if self.blockhead_checkpoint_dir or self.sync_drafter_checkpoint_dir:
                self.draft_runner = None
                self.draft_cfg = DraftRunner.create_draft_config(config)
                self.draft_cfg.num_kvcache_blocks = config.num_kvcache_blocks
                print("W273_BLOCKHEAD_DRAFT_RUNNER=bypassed", flush=True)
            else:
                self.draft_runner = DraftRunner(config)
                self.draft_cfg = self.draft_runner.draft_cfg
                print(f'Draft runner created on rank 0 (no async)', flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config, draft_cfg=self.draft_cfg if config.speculate else None)
        assert config.max_model_len == self.scheduler.max_model_len

        print(f"[LLMEngine] finished llm_engine init", flush=True)

        self._exiting = False
        self._completed_jetspec_taps = {}
        atexit.register(lambda: self.exit(hard=True))

    def exit(self, hard: bool = True):
        print(f"[LLMEngine] Exiting (hard={hard})", flush=True)
        if getattr(self, "_exiting", False):
            return
        self._exiting = True
        # 1) If async, tell draft to quit before tearing down anything
        try:
            if self.config.speculate and self.config.draft_async:
                # Use local method (no SHM) to send cmd=2
                self.model_runner.send_draft_exit_signal()
        except Exception:
            pass
        # 2) Tell all target ranks (including rank 0 self) to exit (non-blocking cleanup, no os._exit inside)
        try:
            self.model_runner.call("exit",
                                   True if not self.config.draft_async else True)
        except Exception:
            pass
        # 3) Wait briefly for TP workers; terminate if still around
        try:
            if self.model_runner.world_size > 1:
                for p in self.ps:
                    p.join(timeout=3)
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=2)
        except Exception:
            pass
        # 4) Draft process: after sending cmd=2, give it a moment, then terminate if needed
        try:
            if self.config.speculate and self.config.draft_async and self.draft_ps is not None:
                for process in self.draft_ps:
                    process.join(timeout=3)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=2)
        except Exception:
            pass
        # 5) Force-exit current process if requested. Multiprocessing owns its
        # named semaphores; other engines may be starting on this host.
        if hard:
            os._exit(0)

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if (self.blockhead_checkpoint_dir or self.sync_drafter_checkpoint_dir or self.config.jetspec) and (
            sampling_params.draft_temperature not in (None, 0)
        ):
            raise ValueError("Block-head, sync drafter, and JetSpec modes require a greedy draft")
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        if (self.blockhead_checkpoint_dir or self.sync_drafter_checkpoint_dir or self.config.jetspec) and (
            seq.temperature > 0 and seq.draft_temperature is None
        ):
            seq.draft_temperature = 0
        if seq.temperature > 0 and seq.draft_temperature != 0 and self.config.use_eagle:
            raise ValueError("EAGLE at temperature > 0 requires a greedy draft (draft_temperature 0)")
        self.scheduler.add(seq)


    def step(self, step: InferenceStep):
        t = perf_counter()
        timing = getattr(self.model_runner, '_anatomy_lite', None)
        start_ns = monotonic_ns() if timing is not None else None
        host_census.mark("step_start")
        seqs, is_prefill = self.scheduler.schedule()
        host_census.mark("schedule")
        if timing is not None and not is_prefill:
            if len(seqs) != 1:
                raise ValueError("anatomy-lite requires sequential N=1 reads")
            seq = seqs[0]
            round_id = getattr(seq, '_anatomy_lite_round', 0)
            seq._anatomy_lite_round = round_id + 1
            timing.begin(seq.seq_id, round_id, start_ns)
            timing.stamp('schedule_end')
        ttl_tokens = step.prefill(seqs) if is_prefill else step.decode(seqs)
        if timing is not None and not is_prefill:
            timing.finish()

        time_taken = perf_counter() - t

        if is_prefill:
            METRICS["prefill_total_time"] += time_taken
            METRICS["prefill_total_tokens"] += ttl_tokens
        else:
            METRICS["decode_total_time"] += time_taken
            METRICS["decode_total_tokens"] += ttl_tokens

        outputs = [(seq.seq_id, seq.completion_token_ids)
                   for seq in seqs if seq.is_finished]
        if self.config.jetspec_taps and (self.config.jetspec_taps_dump
                or os.environ.get('JETSPEC_COMPLETED_TAPS_DUMP_ONLY', '0') != '1'):
            for seq in seqs:
                if seq.is_finished and seq.jetspec_taps is not None:
                    self._completed_jetspec_taps[seq.seq_id] = {
                        "token_ids": list(seq.token_ids),
                        "taps": seq.jetspec_taps.detach().cpu(),
                    }

        host_census.flush(METRICS)
        if timing is not None and not is_prefill:
            timing.close_step()
        return outputs

    def is_finished(self):
        return self.scheduler.is_finished()

    def log_metrics(self):
        avg_prefill_throughput = METRICS["prefill_total_tokens"] / \
            METRICS["prefill_total_time"]
        avg_decode_throughput = METRICS["decode_total_tokens"] / \
            METRICS["decode_total_time"]
        print(
            f"Final Prefill Throughput: {int(avg_prefill_throughput)}tok/s", flush=True)
        print(
            f"Final Decode Throughput: {int(avg_decode_throughput)}tok/s", flush=True)

        if self.config.speculate:
            ttl_accepted_with_recovery = sum(METRICS['accepted_suffix_lens_with_recovery'])
            ttl_num_spec_steps = len(METRICS['accepted_suffix_lens_with_recovery'])
            avg_tokens_per_step = ttl_accepted_with_recovery / ttl_num_spec_steps
            print(
                f"[metrics] Avg Tokens per step (incl recovery): {avg_tokens_per_step:.2f}", flush=True)

            total_accepted = ttl_accepted_with_recovery - ttl_num_spec_steps
            avg_acceptance_rate = (total_accepted / ttl_num_spec_steps) / self.config.speculate_k
            print(
                f"[metrics] Avg Fraction of Speculated Tokens Accepted: {avg_acceptance_rate:.2f}", flush=True)
            print(
                f"[metrics] Avg target time per full step (ms): {sum(METRICS['target_step_times']) * 1000 / len(METRICS['target_step_times']):.2f}", flush=True)
            if METRICS['target_verify_times']:
                print(
                    f"[metrics] Avg target verify time (ms): {sum(METRICS['target_verify_times']) * 1000 / len(METRICS['target_verify_times']):.2f}", flush=True)
            if self.config.draft_async and not self.config.jetspec:
                print(
                    f"[metrics] Avg Cache Hits: {sum(METRICS['cache_hits']) / len(METRICS['cache_hits']):.2f}", flush=True)
                # Log separate metrics for cache hits
                if METRICS['accepted_suffix_lens_on_hit']:
                    avg_suffix_len_on_hit = sum(
                        METRICS['accepted_suffix_lens_on_hit']) / len(METRICS['accepted_suffix_lens_on_hit'])
                    print(
                        f"[metrics] Avg Tokens per step on Cache Hit: {avg_suffix_len_on_hit:.2f}", flush=True)
                    
                    # Calculate empirical frequencies of accepted_suffix_lens_on_hit - 1
                    adjusted_lens = [length - 1 for length in METRICS['accepted_suffix_lens_on_hit']]
                    total_count = len(adjusted_lens)
                    freq_counts = {}
                    for length in adjusted_lens:
                        freq_counts[length] = freq_counts.get(length, 0) + 1
                    
                    # Print normalized empirical probabilities for range [0, K]
                    print(f"[metrics] Empirical frequencies of accepted_suffix_lens_on_hit - 1:", flush=True)
                    for k in range(self.config.speculate_k + 1):
                        prob = freq_counts.get(k, 0) / total_count
                        print(f"  {k}: {prob:.3f}", flush=True)
                if METRICS['accepted_suffix_lens_on_miss']:
                    avg_suffix_len_on_miss = sum(
                        METRICS['accepted_suffix_lens_on_miss']) / len(METRICS['accepted_suffix_lens_on_miss'])
                    print(
                        f"[metrics] Avg Tokens per step on Cache Miss: {avg_suffix_len_on_miss:.2f}", flush=True)
                else:
                    print(
                        f"[metrics] Avg Tokens per step on Cache Hit: N/A (no cache hits)", flush=True)
            if self.config.jetspec:
                rounds = (
                    METRICS["jetspec_hit_count"]
                    + METRICS["jetspec_miss_count"]
                    + METRICS["jetspec_late_count"]
                )
                print(
                    "[jetspec] "
                    f"hit={METRICS['jetspec_hit_count']} "
                    f"miss={METRICS['jetspec_miss_count']} "
                    f"late={METRICS['jetspec_late_count']} rounds={rounds} "
                    f"late_fraction={METRICS['jetspec_late_fraction']:.6f} "
                    f"coalesced={METRICS['jetspec_coalesced_count']}",
                    flush=True,
                )
                if os.environ.get("JETSPEC_HYBRID_VERIFY_AUDIT", "0") == "1":
                    print("HYBRID_VERIFY_LOGIT_AUDIT "
                          f"prefill={len(METRICS.get('jetspec_hybrid_prefill_verify', []))} "
                          f"rounds={METRICS.get('jetspec_hybrid_verify_rows', 0)} "
                          f"tokens={METRICS.get('jetspec_hybrid_verify_tokens', 0)} "
                          "violations=0", flush=True)
                kind_summary = summarize_candidate_kinds(
                    METRICS["jetspec_round_records"]
                )
                METRICS["jetspec_kind_summary"] = kind_summary
                print(
                    "[jetspec] "
                    f"served_tau={kind_summary['served_tau']:.6f}",
                    flush=True,
                )
                for row in kind_summary["by_kind"]:
                    print(
                        "W220_R3_KIND "
                        f"kind={row['candidate_kind']} "
                        f"count={row['count']} "
                        f"share_all={row['share_all_rounds']:.6f} "
                        f"hit_rate={row['hit_rate']:.6f} "
                        f"tau={row['tau']:.6f} "
                        f"gap_histogram={row['gap_histogram']}",
                        flush=True,
                    )
                for row in kind_summary["by_previous_ar_and_k"]:
                    print(
                        "W220_R3_SPLIT "
                        f"kind={row['candidate_kind']} "
                        f"previous_ar={int(row['previous_round_ar'])} "
                        f"k={row['k']} count={row['count']} "
                        f"share_all={row['share_all_rounds']:.6f} "
                        f"hit_rate={row['hit_rate']:.6f} "
                        f"tau={row['tau']:.6f} "
                        f"gap_histogram={row['gap_histogram']}",
                        flush=True,
                    )
                ready = [
                    record
                    for record in METRICS["jetspec_round_records"]
                    if record["decision"] != "late"
                ]
                ready_hit_rate = (
                    METRICS["jetspec_hit_count"] / len(ready) if ready else 0.0
                )
                ready_tau = (
                    sum(record.get("accepted_len", 0) for record in ready)
                    / len(ready)
                    if ready else 0.0
                )
                print(
                    "[jetspec] "
                    f"ready_rounds={len(ready)} "
                    f"ready_hit_rate={ready_hit_rate:.6f} "
                    f"ready_tau={ready_tau:.6f} "
                    f"gap_histogram={METRICS['jetspec_gap_histogram']}",
                    flush=True,
                )

    def create_inference_step(self, config: Config) -> InferenceStep:
        if config.speculate:
            if self.blockhead_checkpoint_dir:
                if self.num_tp_gpus > 1 and not hasattr(self.model_runner.model, "jetspec_full_vocab"):
                    # The rank-0 sync-slot head reads the full vocab rows the TP ranks shard.
                    self.model_runner.call("gather_full_vocab_weights")
                speculator = SpeculatorBlockhead(
                    lookahead=config.speculate_k,
                    device=config.device,
                    checkpoint_dir=self.blockhead_checkpoint_dir,
                    target=self.model_runner.model,
                    metrics=METRICS,
                    max_context=config.max_model_len,
                    max_num_seqs=config.max_num_seqs,
                )
            elif self.sync_drafter_checkpoint_dir:
                from jetspec.engines.jetspec_plus_plus.verify.speculator_sync_drafter import SpeculatorSyncDrafter
                speculator = SpeculatorSyncDrafter(
                    lookahead=config.speculate_k,
                    device=config.device,
                    checkpoint_dir=self.sync_drafter_checkpoint_dir,
                    metrics=METRICS,
                    max_context=config.max_model_len,
                    max_num_seqs=config.max_num_seqs,
                )
            elif config.draft_async:
                speculator_kwargs = dict(
                    lookahead=config.speculate_k,
                    device=config.device,
                    async_fan_out=config.async_fan_out,
                    max_blocks=config.max_blocks,
                    vocab_size=config.draft_hf_config.vocab_size,
                    draft_dtype=config.draft_hf_config.torch_dtype,
                    kvcache_block_size=config.kvcache_block_size,
                    max_model_len=config.max_model_len,
                    max_num_seqs=config.max_num_seqs,
                    tokenizer=self.tokenizer,
                    verbose=config.verbose,
                    jetspec_taps=config.jetspec_taps,
                    jetspec_tap_hidden_size=config.jetspec_tap_hidden_size,
                    jetspec=config.jetspec,
                    jetspec_sync=config.jetspec_sync,
                    jetspec_wait_ms=config.jetspec_wait_ms,
                    jetspec_hyp=config.jetspec_hyp,
                    jetspec_miss=config.jetspec_miss,
                    metrics=METRICS,
                    target_model_runner=self.model_runner,
                )
                speculator = SpeculatorAsync(
                    **speculator_kwargs,
                    async_pg=self.model_runner.async_pg,
                    jetspec_request_pg=getattr(self.model_runner, "jetspec_request_pg", None),
                    jetspec_candidate_pg=getattr(self.model_runner, "jetspec_candidate_pg", None),
                    jetspec_mailbox_store=getattr(self.model_runner, "jetspec_mailbox_store", None),
                    draft_runner_rank=2 if self.split_draft else self.num_tp_gpus,
                )
                if not self.split_draft:
                    # One candidate stream per process: generate() builds a new step per
                    # call, while the draft's candidate ids run for the whole process.
                    previous = getattr(self, "_jetspec_stream_owner", None)
                    if previous is not None:
                        speculator.adopt_candidate_stream(previous)
                    self._jetspec_stream_owner = speculator
                if self.split_draft:
                    from jetspec.engines.jetspec_plus_plus.verify.speculator_split import SplitSpeculator
                    async_pg, request_pg, candidate_pg, store = self.model_runner.split_draft_groups[1]
                    co_resident = SpeculatorAsync(
                        **speculator_kwargs, async_pg=async_pg,
                        jetspec_request_pg=request_pg, jetspec_candidate_pg=candidate_pg,
                        jetspec_mailbox_store=store, draft_runner_rank=1,
                    )
                    speculator = SplitSpeculator(
                        speculator, co_resident,
                        fraction=float(os.environ.get("JETSPEC_DRAFT_SPLIT_FRAC", "0.5")),
                    )
            else:
                speculator = SpeculatorSync(
                    lookahead=config.speculate_k,
                    device=config.device,
                    draft_model_runner=self.draft_runner,
                )

            verifier = Verifier(
                lookahead=config.speculate_k,
                device=config.device,
                target_model_runner=self.model_runner,
                sampler_x=config.sampler_x,
                async_fan_out=config.async_fan_out,
                jit_speculate=config.jit_speculate,
                tokenizer=self.tokenizer,
                metrics=METRICS,
            )
            return SpecDecodeStep(
                scheduler=self.scheduler,
                speculator=speculator,
                verifier=verifier,
                eagle=config.use_eagle,
                tokenizer=self.tokenizer,
                async_spec=config.draft_async,
                verbose=config.verbose,
            )
        else:
            return AutoRegressiveStep(
                scheduler=self.scheduler,
                model_runner=self.model_runner,
                tokenizer=self.tokenizer,
                verbose=config.verbose,
            )

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        stream_callback=None,
    ) -> list[str]:
        self._completed_jetspec_taps.clear()
        for k in METRICS:
            METRICS[k] = ({} if k == "jetspec_split_draft_ms_by_rank" else
                          [] if isinstance(METRICS[k], list) else 0)

        if use_tqdm:
            pbar = tqdm(total=len(prompts),
                        desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        inference_step = self.create_inference_step(self.config)
        i = 0
        max_steps = self.config.max_steps if self.config.max_steps is not None else float('inf')
        _stream_lens = {}
        while not self.is_finished() and i < max_steps:
            if self.config.verbose:
                print(f"[generate] step {i+1}" +
                    f" of {max_steps}" if max_steps != float('inf') else "",
                    flush=True,
                )
            i += 1
            t = perf_counter()
            output = self.step(inference_step)
            time_taken = perf_counter() - t
            if (getattr(self.model_runner, '_anatomy_lite', None) is not None
                    and os.environ.get('JETSPEC_DEVICE_HOST_STAMPS', '0') == '1'):
                self.model_runner._anatomy_lite.close_native_step(t, time_taken)
            METRICS["target_step_times"].append(time_taken)

            if stream_callback:
                for seq in self.scheduler.running:
                    cur = seq.num_completion_tokens
                    prev = _stream_lens.get(seq.seq_id, 0)
                    if cur > prev:
                        stream_callback(seq.seq_id, seq.completion_token_ids[prev:cur])
                        _stream_lens[seq.seq_id] = cur

            for seq_id, token_ids in output:
                if stream_callback:
                    prev = _stream_lens.get(seq_id, 0)
                    if len(token_ids) > prev:
                        stream_callback(seq_id, token_ids[prev:])
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        if self.blockhead_checkpoint_dir or self.sync_drafter_checkpoint_dir:
            complete_metric_rounds(METRICS)
        if getattr(self.model_runner, '_anatomy_lite', None) is not None:
            METRICS['anatomy_lite'].append(self.model_runner._anatomy_lite.export())
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(
            token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()

        if not stream_callback:
            self.log_metrics()

        if self.config.jetspec_taps_dump:
            dump_dir = os.path.dirname(self.config.jetspec_taps_dump)
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
            torch.save(
                {
                    "layer_ids": list(self.config.jetspec_target_layer_ids or JETSPEC_TARGET_LAYER_IDS),
                    "sequences": self._completed_jetspec_taps,
                },
                self.config.jetspec_taps_dump,
            )

        if os.environ.get("JETSPEC_C1_COMPILE_PIN", ""):
            # Bench uses this boundary so the pre-seal rank checks do not move served tok/s.
            self._generate_served_end_time = time.time()
            self.model_runner._check_compile_pin_served()
        else:
            self._generate_served_end_time = None
        return outputs, METRICS

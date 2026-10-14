import os
import time
import json
import sys
import torch
import torch.distributed as dist
import dataclasses

from jetspec.engines.jetspec_plus_plus.serving.model_runner import ModelRunner
from jetspec.engines.jetspec_plus_plus.serving.config import Config
from jetspec.engines.jetspec_plus_plus.serving.utils.context import set_context; from jetspec.engines.jetspec_plus_plus.serving.utils.context import reset_context
from jetspec.engines.jetspec_plus_plus.serving.utils.async_helpers.async_spec_helpers import get_forked_recovery_tokens_from_logits; from jetspec.engines.jetspec_plus_plus.serving.utils.async_helpers.async_spec_helpers import make_glue_decode_input_ids
from jetspec.engines.jetspec_plus_plus.serving.utils.async_helpers.nccl_pack import append_jetspec_tap_mirrors; from jetspec.engines.jetspec_plus_plus.serving.utils.async_helpers.nccl_pack import recv_int64
from jetspec.engines.jetspec_plus_plus.serving.helpers.cudagraph_helpers import flush_draft_profile
from jetspec.engines.jetspec_plus_plus.draft.drafter import DraftSuperseded; from jetspec.engines.jetspec_plus_plus.draft.drafter import JetSpecDrafter; from jetspec.engines.jetspec_plus_plus.draft.drafter import JetSpecSequence; from jetspec.engines.jetspec_plus_plus.draft.drafter import commit_sequences_batch; from jetspec.engines.jetspec_plus_plus.draft.drafter import anchor_known_correction; from jetspec.engines.jetspec_plus_plus.draft.drafter import following_candidate_from_token_block
from jetspec.engines.jetspec_plus_plus.draft.tree_serve import tree_budget; from jetspec.engines.jetspec_plus_plus.draft.tree_serve import commit_tree_ids
from jetspec.engines.jetspec_plus_plus.draft.live_rows import live_hypotheses; from jetspec.engines.jetspec_plus_plus.draft.live_rows import select_live_block; from jetspec.engines.jetspec_plus_plus.draft.live_rows import select_live_blocks_batch; from jetspec.engines.jetspec_plus_plus.draft.live_rows import cache_packed_metadata_batch; from jetspec.engines.jetspec_plus_plus.draft.live_rows import PosteriorHitCounter; from jetspec.engines.jetspec_plus_plus.draft.live_rows import posterior_mode; from jetspec.engines.jetspec_plus_plus.draft.live_rows import e04_lean
from jetspec.engines.jetspec_plus_plus.draft.live_rows import miss_redraft; from jetspec.engines.jetspec_plus_plus.draft.live_rows import miss_redraft_force; from jetspec.engines.jetspec_plus_plus.draft.live_rows import miss_redraft_passes; from jetspec.engines.jetspec_plus_plus.draft.live_rows import draft_deadline; from jetspec.engines.jetspec_plus_plus.draft.live_rows import DeadlineClock; from jetspec.engines.jetspec_plus_plus.draft.live_rows import DraftDeadlineMiss; from jetspec.engines.jetspec_plus_plus.draft.live_rows import deadline_miss_blocks; from jetspec.engines.jetspec_plus_plus.draft.live_rows import miss_redraft_anchor
from jetspec.engines.jetspec_plus_plus.draft.graph import GraphedDraftHead; from jetspec.engines.jetspec_plus_plus.draft.graph import GraphedLMHead
from jetspec.engines.jetspec_plus_plus.wire.protocol import CandidateSet; from jetspec.engines.jetspec_plus_plus.wire.protocol import unpack_request_wire; from jetspec.engines.jetspec_plus_plus.wire.protocol import candidate_set_from_blocks; from jetspec.engines.jetspec_plus_plus.wire.protocol import pack_candidate_topk; from jetspec.engines.jetspec_plus_plus.wire.protocol import anchor_mode; from jetspec.engines.jetspec_plus_plus.wire.protocol import oracle_candidate_blocks; from jetspec.engines.jetspec_plus_plus.wire.protocol import branch_count; from jetspec.engines.jetspec_plus_plus.wire.protocol import candidate_blocks_tensor; from jetspec.engines.jetspec_plus_plus.wire.protocol import pack_candidate_sets; from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width; from jetspec.engines.jetspec_plus_plus.wire.protocol import select_candidate_drafts; from jetspec.engines.jetspec_plus_plus.wire.protocol import ProtocolCounters; from jetspec.engines.jetspec_plus_plus.wire.protocol import validate_candidate_row_keys_batch; from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_verifier_kind_from_host_pins

PROFILE_DRAFT = os.environ.get("SSD_PROFILE_DRAFT", "0") == "1"

ttl = 0
ttl_hit = 0

# JETSPEC_TAIL_LEAN=1 (default off): the round's W270_HEAD / W258_KSET log lines (and the
# posterior-hit bookkeeping that feeds them) run after the candidate is sent, not between the
# commit and the adapter or between the last pass and the send; same lines, later in the log
_DEFERRED = []


def _defer_logs():
    return os.environ.get('JETSPEC_TAIL_LEAN', '0') == '1'


def _later(call):
    if _defer_logs():
        _DEFERRED.append(call)
    else:
        call()


def _flush_deferred():
    _run_idle()
    while _DEFERRED:
        _DEFERRED.pop(0)()


# JETSPEC_TAIL_PUBLISH: host work with no reader before the next round (the round's
# deferred log lines; measurement tools may queue their own writes here with idle_later) runs
# while the next round's host waits on its own passes, just before the early publish's wait,
# not between the publish and the next request.  Order is kept; the exit command drains it.
_IDLE = []




def _run_idle(event=None, keep=0):
    """Drain the queue; with ``event``, only while the device is still busy (event not done),
    then down to ``keep`` pending items (oldest first)."""
    while _IDLE and (event is None or not event.query()):
        _IDLE.pop(0)()
    if event is not None:
        while len(_IDLE) > keep:
            _IDLE.pop(0)()




def _device_publish():
    return os.environ.get('JETSPEC_DEVICE_PUBLISH', '0') == '1'


def _pack_jetspec_blocks(blocks):
    return blocks.wire if hasattr(blocks, 'wire') else candidate_blocks_tensor(blocks)


def _candidate_set_from_rows(seq_id, round_id, blocks, host_keys=None, keys_validated=False,
                             joined_metrics=None):
    # JETSPEC_TAIL_PUBLISH: host_keys = the published keys, already copied home (the same
    # values as blocks.row_keys), so the CandidateSet checks read no device value
    candidate = CandidateSet(
        seq_id=int(seq_id), round_id=int(round_id), blocks=_pack_jetspec_blocks(blocks),
        row_keys=getattr(blocks, 'row_keys', None) if host_keys is None else host_keys,
        _keys_validated=keys_validated)
    metrics = getattr(blocks, 'head_metrics', None)
    if metrics is not None:
        record = dict(seq_id=int(seq_id), round_id=int(round_id), **metrics)
        if joined_metrics is None:
            _later(lambda: print('W270_HEAD ' + json.dumps(record), flush=True))
        else:
            joined_metrics.append('W270_HEAD ' + json.dumps(record) + '\n')
    return candidate




class DraftRunner(ModelRunner):
    
    @classmethod
    def create_draft_config(cls, cfg: Config) -> Config:
        """Create a draft config from the main config without instantiating DraftRunner."""
        from jetspec.engines.jetspec_plus_plus.draft.live_rows import resolve_kset_count
        from jetspec.engines.jetspec_plus_plus.draft.routing import apply_auto_route
        route = apply_auto_route(cfg.max_num_seqs) if cfg.jetspec else None
        draft_cfg = dataclasses.replace(
            cfg,
            model=cfg.draft,
            jetspec_config_is_draft=True,
            jetspec_passes=route.passes if route is not None else cfg.jetspec_passes,
            jetspec_hyp=resolve_kset_count(cfg.jetspec_hyp) if cfg.jetspec else cfg.jetspec_hyp,
            # The rows module has its own KV and logits buffers after this
            # ordinary draft cache is allocated (prefill can exceed decode N).
            gpu_memory_utilization=((0.35 if os.environ.get("JETSPEC_ROWS_FAST", "0") == "1" else 0.5)
                                    if cfg.jetspec and os.environ.get("JETSPEC_ROWS", "0") == "1"
                                    else (0.75 if not cfg.draft_async else 0.8)),
            tokenizer_path=cfg.model if cfg.use_eagle else None,
            d_model_target=cfg.hf_config.hidden_size if cfg.use_eagle and cfg.hf_config else None,
            jetspec_tap_hidden_size=(
                cfg.hf_config.hidden_size if cfg.jetspec_taps else None
            ),
            enforce_eager=True if cfg.jetspec else cfg.enforce_eager,
            target_dtype=None,
        )
        # Config.__post_init__ skipped its target-only checks for this copy; the taps came validated.
        assert draft_cfg.jetspec_config_is_draft
        assert draft_cfg.jetspec_target_layer_ids == cfg.jetspec_target_layer_ids
        return draft_cfg

    def __init__(self, cfg: Config, rank: int = 0, init_q = None):
        self._hybrid_pair_layout = None
        if os.environ.get("JETSPEC_HYBRID_PAIR", "0") == "1":
            from jetspec.engines.jetspec_plus_plus.draft.hybrid_pair import HybridPairLayout
            if cfg.max_num_seqs != 1 or cfg.speculate_k not in (15, 31):
                raise ValueError("hybrid pair transfer requires N=1 and b16/b32")
            self._hybrid_pair_layout = HybridPairLayout.from_config(
                cfg.hf_config, lanes=cfg.speculate_k + 1)
        self._chain_mode = cfg.jetspec and os.environ.get("JETSPEC_CHAIN_GRAPH", "0") == "seg"
        if self._chain_mode:
            from jetspec.engines.jetspec_plus_plus.draft.cache_capacity import apply_chain_capacity
            apply_chain_capacity(os.environ, cfg.max_model_len, lanes=cfg.speculate_k + 1)
            if cfg.max_num_seqs != 1:
                raise ValueError("JETSPEC_CHAIN_GRAPH=seg requires N=1")
            if (os.environ.get("JETSPEC_VERIFY_DEVICE", "0") != "1"
                    and os.environ.get("JETSPEC_VERIFY_REFERENCE", "0") != "1"):
                raise ValueError("JETSPEC_CHAIN_GRAPH=seg requires device or reference verification")
        if os.environ.get("JETSPEC_DRAFT_SPLIT", "") == "verify_gpu" and rank == 2:
            os.environ.pop("CUDA_MPS_PIPE_DIRECTORY", None)
            os.environ.pop("CUDA_MPS_LOG_DIRECTORY", None)
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SSD_SPLIT_DEDICATED_UUID"]
        self.draft_cfg = self.create_draft_config(cfg)
        if os.environ.get("JETSPEC_DRAFT_SPLIT", "") == "verify_gpu" and rank == 1:
            # The target is loaded concurrently on this GPU. A small draft KV
            # fraction leaves room for the target, adapter, and row heads.
            self.draft_cfg.gpu_memory_utilization = min(
                self.draft_cfg.gpu_memory_utilization, 0.10)
            print("SPLIT_CO_KV_UTILIZATION=0.10", flush=True)
        self.is_draft = True # this is is_draft, use self.config.draft for the draft model path 
        self.prev_num_tokens = None
        super().__init__(self.draft_cfg, rank=rank, event=None, is_draft=True, num_tp_gpus=1, init_q=init_q)
        
        if self.config.use_eagle:
            assert self.config.jit_speculate, \
                "EAGLE requires jit_speculate=True (cache misses need draft activations)"

        self.jetspec_tap_mirrors = {}
        self._jetspec_tap_recv_ms = []
        self._jetspec_tap_payload_bytes = []
        self.jetspec_drafter = None
        self.jetspec_sequences = {}
        self.jetspec_candidate_sets = {}
        self.jetspec_block_objects = {}
        self.jetspec_candidate_ms = {}
        self._jetspec_candidate_send_id = 0
        self._jetspec_next_request_receive_id = 0
        self._jetspec_control_receive_id = 0
        self._jetspec_pending_rounds = {}
        self._jetspec_last_drafted_round = {}
        self._jetspec_sent_candidates = set()
        self._jetspec_wire_capacity = None
        self._chain_runtime = None
        self._pair_transport = None
        self.oracle_anchor = self.config.jetspec and anchor_mode() == 'oracle'
        if self.oracle_anchor and (os.environ.get('JETSPEC_ROWS') != '1'
                or self.config.jetspec_sync or self.config.jetspec_wait_ms != float('inf')):
            raise ValueError('oracle anchor requires asynchronous rows with infinite wait')
        self._posterior_hits = PosteriorHitCounter()
        if self.config.jetspec:
            self.jetspec_drafter = JetSpecDrafter.load(
                self.config.jetspec_checkpoint_dir,
                self.config.jetspec_adapter_config,
                self.config.jetspec_adapter_head,
                self.config.jetspec_target_release_weights,
                self.device,
            )
            if os.environ.get("JETSPEC_DRAFT_DTYPE") == "fp32":
                for module in (self.jetspec_drafter.head, self.jetspec_drafter.embed_tokens,
                               self.jetspec_drafter.lm_head, self.jetspec_drafter.adapter.head):
                    module.float()
                self.jetspec_drafter.dtype = torch.float32
            self.jetspec_rows = None
            if os.environ.get("JETSPEC_ROWS", "0") == "1":
                if self.config.jetspec_sync or (self.config.jetspec_wait_ms != float("inf")
                        and os.environ.get("JETSPEC_ROWS_FAST", "0") != "1"
                        and os.environ.get("JETSPEC_W241_ASYNC_ROWS", "0") != "1"):
                    raise ValueError("JETSPEC_ROWS requires the waiting kth protocol")
                from jetspec.engines.jetspec_plus_plus.draft.live_rows import KthRowsDrafter
                backend = "sdpa" if self.jetspec_drafter.dtype == torch.float32 else "flashinfer"
                self.jetspec_rows = KthRowsDrafter(self.jetspec_drafter, backend=backend,
                    optimized=os.environ.get("JETSPEC_ROWS_FAST", "0") == "1",
                    graphs=os.environ.get("JETSPEC_ROWS_GRAPHS", "0") == "1", branches=branch_count(),
                    f_graphs=os.environ.get("JETSPEC_F_GRAPHS", "0") == "1")
                if posterior_mode() and (self.config.max_num_seqs != 1 or branch_count() not in (1, 5)):
                    raise ValueError("posterior serving requires N=1 and F=1 or F=5")
                print(f"W234_ROWS=enabled F={branch_count()} live-kth adapter=shared-prefix", flush=True)
                # the miss fallback row and the round-ahead draft deadline (both default off)
                self._w320_redraft = miss_redraft()
                self._w320_force = self._w320_redraft and miss_redraft_force()
                self._w320_clock = DeadlineClock() if draft_deadline() else None
                self.jetspec_rows.deadline = self._w320_clock
                if (self._w320_redraft or self._w320_clock is not None) and (
                        self.config.jetspec_wait_ms != float("inf")
                        or (self._w320_clock is not None and self.config.max_num_seqs != 1)
                        or tree_budget() or anchor_mode() == 'oracle'):
                    raise ValueError("JETSPEC_MISS_REDRAFT and JETSPEC_DRAFT_DEADLINE require infinite wait, no tree budget, no oracle; "
                                     "the deadline also requires N=1")
            if self.config.jetspec_draft_graphs and self.jetspec_rows is None:
                graphed_head = GraphedDraftHead(
                    self.jetspec_drafter.head,
                    self.jetspec_drafter.lm_head,
                )
                self.jetspec_drafter.head = graphed_head
                self.jetspec_drafter.lm_head = GraphedLMHead(graphed_head)
                print(
                    "W220_DRAFT_GRAPHS=enabled key=(H,context_bucket) "
                    "block=16 adapter=eager",
                    flush=True,
                )
            else:
                print("W220_DRAFT_GRAPHS=disabled adapter=eager", flush=True)

        self.jetspec_catchup_drafter = self.jetspec_drafter
        if (self.config.jetspec and self.jetspec_rows is not None and self.jetspec_rows.optimized
                and self.config.jetspec_wait_ms != float('inf') and self.config.jetspec_draft_graphs):
            import copy
            self.jetspec_catchup_drafter = copy.copy(self.jetspec_drafter)
            catchup_head = GraphedDraftHead(self.jetspec_drafter.head, self.jetspec_drafter.lm_head)
            self.jetspec_catchup_drafter.head = catchup_head
            self.jetspec_catchup_drafter.lm_head = GraphedLMHead(catchup_head)

        if self.is_draft and self.draft_async:
            self._reset_tree_cache_tensors()
            self._init_prealloc_buffers()
            self._draft_step_times = []
            print(f'DraftRunner set up, starting draft_loop', flush=True)
            self.draft_loop()

    def draft_async_prefill(self):
        assert self.draft_async and self.is_draft
        request_pg = self.jetspec_request_pg if self.config.jetspec else self.async_pg

        # 1) Receive metadata then individual tensors
        # First recv metadata to learn sizes
        metadata = torch.zeros(5, dtype=torch.int64, device=self.device)
        dist.recv(metadata, src=0, group=request_pg)
        total_new_tokens, batch_size, max_blocks, use_eagle, eagle_act_dim = metadata.tolist()
        if use_eagle:
            assert eagle_act_dim == 3 * self.config.d_model_target, (
                f"EAGLE activation dimension {eagle_act_dim} does not match expected dimension 3 * {self.config.d_model_target}"
            )

        # 2) receive fused int64 payload (input_ids + num_tokens + draft_block_table)
        fused_total = total_new_tokens + batch_size + batch_size * max_blocks
        fused = recv_int64(request_pg, src=0, total_length=fused_total, device=self.device)
        off = 0
        input_ids = fused[off:off + total_new_tokens]; off += total_new_tokens
        num_tokens = fused[off:off + batch_size]; off += batch_size
        draft_block_table = fused[off:off + batch_size * max_blocks].view(batch_size, max_blocks).to(torch.int32); off += batch_size * max_blocks
        assert off == fused_total

        eagle_acts = None
        if use_eagle:
            eagle_acts = torch.zeros(
                total_new_tokens, eagle_act_dim, dtype=self.hf_config.torch_dtype, device=self.device,
            )
            dist.recv(eagle_acts, src=0, group=request_pg)

        if self.config.jetspec:
            fused_jetspec = recv_int64(
                request_pg,
                src=0,
                total_length=3 * batch_size,
                device=self.device,
            )
            seq_ids = fused_jetspec[:batch_size]
            prompt_indices = fused_jetspec[batch_size:2 * batch_size]
            recovery_tokens = fused_jetspec[2 * batch_size:]
            tap_lengths = torch.empty(
                batch_size, dtype=torch.int64, device=self.device
            )
            dist.recv(tap_lengths, src=0, group=request_pg)
            total_taps = int(tap_lengths.sum().item())
            tap_payload = torch.empty(
                total_taps,
                5,
                self.config.jetspec_tap_hidden_size,
                dtype=self.hf_config.torch_dtype,
                device=self.device,
            )
            if total_taps:
                dist.recv(tap_payload, src=0, group=request_pg)
            hybrid_admission = None
            if self._hybrid_pair_layout is not None:
                if batch_size != 1:
                    raise ValueError("hybrid pair admission requires N=1")
                layout = self._hybrid_pair_layout
                hybrid_admission = (
                    torch.empty(layout.recurrent_shape, dtype=torch.float32,
                                device=self.device),
                    torch.empty(layout.conv_shape, dtype=layout.dtype,
                                device=self.device),
                    torch.empty((total_taps, layout.kv_heads, layout.attention_dim),
                                dtype=layout.dtype, device=self.device),
                    torch.empty((total_taps, layout.kv_heads, layout.attention_dim),
                                dtype=layout.dtype, device=self.device),
                )
                for payload in hybrid_admission:
                    dist.recv(payload, src=0, group=request_pg)
            tap_chunks = []
            offset = 0
            for length in tap_lengths.tolist():
                next_offset = offset + int(length)
                tap_chunks.append(tap_payload[offset:next_offset])
                offset = next_offset

            from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled as eval_enabled; from jetspec.engines.jetspec_plus_plus.draft.eval_profile import release_finished_sequences
            if eval_enabled():
                # Eval runs one request at a time: free every finished request's taps and state.
                release_finished_sequences(
                    (self.jetspec_sequences, self.jetspec_tap_mirrors,
                     self.jetspec_candidate_sets, self.jetspec_block_objects),
                    seq_ids.tolist())

            id_offset = 0
            assert self.jetspec_drafter is not None
            for row, (seq_id, prompt_index, length, correction, taps) in enumerate(
                zip(
                    seq_ids.tolist(),
                    prompt_indices.tolist(),
                    num_tokens.tolist(),
                    recovery_tokens.tolist(),
                    tap_chunks,
                )
            ):
                prompt_ids = input_ids[id_offset:id_offset + int(length)]
                id_offset += int(length)
                committed_ids = torch.cat(
                    [
                        prompt_ids,
                        torch.tensor(
                            [correction], dtype=torch.int64, device=self.device
                        ),
                    ]
                ).reshape(1, -1)
                state = JetSpecSequence(
                    taps=taps,
                    committed_ids=committed_ids,
                    prompt_index=int(prompt_index),
                )
                from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled as eval_enabled
                if eval_enabled():
                    state._jetspec_seq_id = int(seq_id)
                from jetspec.engines.jetspec_plus_plus.draft.tau_lab import lab_active as e04_lab_active
                if e04_lab_active():
                    # E04_LAB_REFRESH_CUTOFF=oracle: where the reference output starts
                    state.e04_lab_prompt_length = int(length)
                    # E04_LAB_EXACT: the first (exact) round runs before the draft rank sets the
                    # seq id; its record and the oracle read this one (tau_lab.lab_seq_id)
                    state.e04_lab_seq_id = int(seq_id)
                if hybrid_admission is not None:
                    (state.hybrid_recurrent, state.hybrid_conv,
                     state.hybrid_prefix_key, state.hybrid_prefix_value) = hybrid_admission
                    if os.environ.get('JETSPEC_CHAIN_GRAPH', '0') == '0':
                        seed_eager_hybrid_state(state, capacity=self.config.max_model_len + 32)
                if self.jetspec_rows is not None:
                    self.jetspec_sequences[int(seq_id)] = state
                    self.jetspec_tap_mirrors[int(seq_id)] = taps.clone()
                    continue
                torch.cuda.synchronize()
                draft_started = time.perf_counter()
                if self.config.jetspec_sync:
                    blocks = self.jetspec_drafter.propose_sync(
                        state,
                        mode="kth",
                        hypotheses=list(range(self.config.jetspec_hyp)),
                        passes=self.config.jetspec_passes,
                    )
                else:
                    blocks = self.jetspec_drafter.propose(
                        state,
                        mode="kth",
                        k_hypotheses=list(range(self.config.jetspec_hyp)),
                        passes=self.config.jetspec_passes,
                    )
                torch.cuda.synchronize()
                self._dump_jetspec_sync_state(
                    seq_id=int(seq_id),
                    round_id=0,
                    state=state,
                    blocks=blocks,
                )
                candidate_set = CandidateSet(
                    seq_id=int(seq_id),
                    round_id=0,
                    blocks=_pack_jetspec_blocks(blocks),
                )
                self.jetspec_sequences[int(seq_id)] = state
                self.jetspec_tap_mirrors[int(seq_id)] = taps.clone()
                self.jetspec_candidate_sets[int(seq_id)] = candidate_set
                self.jetspec_block_objects[int(seq_id)] = (0, blocks)
                self.jetspec_candidate_ms[(int(seq_id), 0)] = (
                    time.perf_counter() - draft_started
                ) * 1000
                self._jetspec_last_drafted_round[int(seq_id)] = 0
            assert id_offset == input_ids.numel()
            if self.jetspec_rows is not None:
                states = [self.jetspec_sequences[int(s)] for s in seq_ids.tolist()]
                torch.cuda.synchronize()
                started = time.perf_counter()
                oracle = {}
                if self.oracle_anchor:
                    pins = torch.full((len(states), 15), -1, dtype=torch.int64, device=self.device)
                    pins[:, 0] = recovery_tokens
                    oracle['oracle_predictions'] = pins
                outputs = self.jetspec_rows.propose(states,
                    k_hypotheses=[0] if self.oracle_anchor else live_hypotheses(self.config.jetspec_hyp),
                    passes=self.config.jetspec_passes, initial=True, **oracle)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - started) * 1000
                for seq_id, blocks in zip(seq_ids.tolist(), outputs):
                    seq_id = int(seq_id)
                    self.jetspec_candidate_sets[seq_id] = (
                        candidate_set_from_blocks(seq_id, 0, blocks, hypotheses=self.config.jetspec_hyp)
                        if tree_budget() else (CandidateSet(
                            seq_id=seq_id, round_id=0,
                            blocks=oracle_candidate_blocks(blocks, hypotheses=self.config.jetspec_hyp))
                            if self.oracle_anchor else _candidate_set_from_rows(seq_id, 0, blocks)))
                    self.jetspec_block_objects[seq_id] = (0, blocks)
                    self.jetspec_candidate_ms[(seq_id, 0)] = elapsed
                    self._jetspec_last_drafted_round[seq_id] = 0
            if not self.config.jetspec_sync:
                if self._jetspec_wire_capacity is None:
                    self._jetspec_wire_capacity = max(self.config.max_num_seqs, int(batch_size))
                if int(batch_size) > self._jetspec_wire_capacity:
                    raise RuntimeError("JetSpec prefill batch exceeds wire capacity")
                if self._chain_mode:
                    self._init_chain_runtime(int(seq_ids[0]))
                self._send_jetspec_candidates(seq_ids)
            return

        prefill_ctxt = self.prepare_prefill_ctxt(num_tokens, draft_block_table)

        # 5) set up context exactly like prepare_prefill() does:
        set_context(
            is_prefill=True,
            cu_seqlens_q=prefill_ctxt["cu_seqlens_q"],
            cu_seqlens_k=prefill_ctxt["cu_seqlens_k"],
            max_seqlen_q=prefill_ctxt["max_seqlen_q"],
            max_seqlen_k=prefill_ctxt["max_seqlen_k"],
            slot_mapping=prefill_ctxt["slot_map"],
            context_lens=None,
        ) # , block_tables=block_tables, commenting this out essentially removes prefix caching

        # 6) run the draft model in prefill mode
        positions = prefill_ctxt["positions"]
        if self.config.use_eagle:
            self.run_model(input_ids, positions, is_prefill=True, last_only=True, hidden_states=eagle_acts)
        else:
            self.run_model(input_ids, positions, is_prefill=True, last_only=True, hidden_states=eagle_acts)

        # 7) clean up
        reset_context()

    def _reset_tree_cache_tensors(self):
        """Reset tensor-backed tree cache to empty."""
        # initialize as empty keys on correct device; tokens/logits set to None until first populate
        self.tree_cache_keys = torch.zeros(
            (0, 3), dtype=torch.int64, device=self.device)
        self.tree_cache_tokens = None
        self.tree_cache_logits = None
        self.tree_cache_activations = None

    def _init_prealloc_buffers(self):
        # PERFORMANCE: pre-allocate constant tensors used every draft step to avoid repeated CUDA mallocs
        K, MQ_LEN = self.config.speculate_k, self.config.MQ_LEN
        d = self.device
        self._step_pos_offsets = torch.arange(K, device=d, dtype=torch.int64)[:, None] * MQ_LEN
        self._step_rope_offsets = torch.arange(K, device=d, dtype=torch.int64)[:, None]
        self._fan_idx_hit = torch.arange(K + 1, device=d, dtype=torch.int64).repeat_interleave(self.config.fan_out_t)
        self._fan_idx_miss = torch.arange(K + 1, device=d, dtype=torch.int64).repeat_interleave(self.config.fan_out_t_miss)
        self._arange_mq = torch.arange(MQ_LEN, device=d, dtype=torch.int64)
        self._arange_kp1 = torch.arange(K + 1, device=d, dtype=torch.int64)
        self._arange_2kp1 = torch.arange(2 * K + 1, device=d, dtype=torch.int64)



    def _send_jetspec_candidates(
        self,
        seq_ids: torch.Tensor | list[int],
        *,
        candidate_sets: list[CandidateSet | None] | None = None,
        coalesced: int = 0,
        gaps: dict[tuple[int, int], int] | None = None,
    ) -> None:
        """Push one fixed-capacity response into the target's posted mailbox."""
        seq_ids = [int(seq_id) for seq_id in seq_ids]
        if candidate_sets is None:
            candidate_sets = [
                self.jetspec_candidate_sets.get(seq_id) for seq_id in seq_ids
            ]
        if len(candidate_sets) != len(seq_ids):
            raise ValueError("candidate response rows must match sequence ids")
        capacity = (
            len(seq_ids)
            if self.config.jetspec_sync
            else self._jetspec_wire_capacity
        )
        if capacity is None:
            raise RuntimeError("JetSpec candidate capacity is unknown before prefill")
        if len(candidate_sets) > capacity:
            raise ValueError("candidate response exceeds configured capacity")
        candidate_sets = list(candidate_sets) + [None] * (
            capacity - len(candidate_sets)
        )
        if self.jetspec_rows is not None and self.jetspec_rows.optimized and not self.config.jetspec_sync:
            from jetspec.engines.jetspec_plus_plus.draft.live_rows import CandidateWire
            if not hasattr(self, '_jetspec_packed_mailbox'):
                self._jetspec_packed_mailbox=CandidateWire(capacity,self.config.jetspec_hyp,device=self.device,branches=branch_count())
            mailbox = self._jetspec_packed_mailbox
            direct = (_device_publish() and getattr(mailbox, '_device_publish_ready', False)
                      and all(candidate is None or candidate.blocks.data_ptr() ==
                              mailbox.buffers[2][index].data_ptr()
                              for index, candidate in enumerate(candidate_sets)))
            if _device_publish() and getattr(mailbox, '_device_publish_ready', False) and not direct:
                raise RuntimeError('JETSPEC_DEVICE_PUBLISH candidate order differs from packet')
            packet = (mailbox.pack_published if direct else mailbox.pack)(
                candidate_sets, times=self.jetspec_candidate_ms,
                coalesced=coalesced, gaps={} if gaps is None else gaps)
            mailbox._device_publish_ready = False
            tensors=(packet,)
        else:
            available, candidate_keys, candidate_blocks = pack_candidate_sets(
                candidate_sets,
                hypotheses=self.config.jetspec_hyp,
                branches=branch_count(),
                device=self.device,
            )
            candidate_ms = torch.zeros(
                capacity, dtype=torch.float32, device=self.device
            )
            telemetry = torch.zeros(
                capacity, 2, dtype=torch.int64, device=self.device
            )
            if capacity:
                telemetry[0, 0] = int(coalesced)
            gaps = {} if gaps is None else gaps
            for row, candidate_set in enumerate(candidate_sets):
                if candidate_set is None:
                    continue
                key = (candidate_set.seq_id, candidate_set.round_id)
                candidate_ms[row] = self.jetspec_candidate_ms.get(key, 0.0)
                telemetry[row, 1] = int(gaps.get(key, 0))
            tensors = (
                available,
                candidate_keys,
                candidate_blocks,
                candidate_ms,
            )
            if not self.config.jetspec_sync:
                tensors += (telemetry,)
            if tree_budget():
                tensors += pack_candidate_topk(candidate_sets, hypotheses=self.config.jetspec_hyp,
                                               branches=branch_count(), device=self.device)
        if (len(tensors) == 1 and not tree_budget()
                and self.config.jetspec_wait_ms == float("inf")):
            packet = tensors[0]
            if (len(seq_ids) > 1 and self.jetspec_rows is not None
                    and self.jetspec_rows.optimized and torch.device(self.device).type == 'cuda'):
                # The whole candidate wire travels home once. Keep its keyed pins for the
                # following round's miss check, instead of copying each request's blocks.
                from jetspec.engines.jetspec_plus_plus.draft.live_rows import candidate_wire_views
                mailbox = self._jetspec_packed_mailbox
                source = mailbox.wire_all if direct else packet
                host_name = 'host_wire_checked' if direct else 'host_wire'
                if not hasattr(mailbox, host_name):
                    setattr(mailbox, host_name, torch.empty(source.shape,
                            dtype=source.dtype, pin_memory=True))
                host = getattr(mailbox, host_name)
                host.copy_(source, non_blocking=True)
                event = torch.cuda.Event()
                event.record()
                event.synchronize()
                packet = mailbox.published_host(host) if direct else host
                if direct:
                    for index, candidate_set in enumerate(candidate_sets):
                        if candidate_set is None:
                            continue
                        entry = self.jetspec_block_objects.get(candidate_set.seq_id)
                        if entry is None or entry[0] != candidate_set.round_id:
                            raise RuntimeError('JETSPEC_DEVICE_PUBLISH metadata owner differs from packet')
                        blocks = entry[1]
                        blocks._host_metadata = mailbox.host_metadata[index, :len(blocks)].clone()
                        fallback = getattr(blocks, '_fallback_blocks', None)
                        if fallback is not None:
                            fallback._host_metadata = mailbox.host_metadata[index, :1].clone()
                if mailbox.keyed:
                    views = candidate_wire_views(packet, capacity, self.config.jetspec_hyp,
                                                 branch_count())
                    if direct:
                        checked = getattr(self, '_jetspec_publish_checked', 0)
                        if os.environ.get('JETSPEC_DEVICE_PUBLISH_CHECK', '0') == '1' and checked < 200:
                            if not hasattr(self, '_jetspec_publish_reference'):
                                self._jetspec_publish_reference = CandidateWire(
                                    capacity, self.config.jetspec_hyp, device=self.device,
                                    branches=branch_count())
                            reference = self._jetspec_publish_reference.pack(
                                candidate_sets, times=self.jetspec_candidate_ms,
                                coalesced=coalesced, gaps={} if gaps is None else gaps).cpu()
                            same = torch.equal(packet.view(torch.uint8), reference.view(torch.uint8))
                            self._jetspec_publish_checked = checked + 1
                            print('W641_PUBLISH_CHECK ' + json.dumps(dict(
                                rounds=self._jetspec_publish_checked, mismatches=int(not same))), flush=True)
                            if not same:
                                raise AssertionError('JETSPEC_DEVICE_PUBLISH packet differs from legacy')
                    pins = torch.cat((views[5], views[2][:, :, :1]), dim=2).tolist()
                    pin_home = self.__dict__.setdefault('_jetspec_pin_home', {})
                    for index, candidate_set in enumerate(candidate_sets):
                        if candidate_set is not None:
                            pin_home[candidate_set.seq_id] = (candidate_set.round_id, pins[index])
            self.jetspec_mailbox_store.set(
                f"candidate_payload_{self._jetspec_candidate_send_id}",
                packet.cpu().numpy().tobytes())
        else:
            for tensor in tensors:
                dist.send(tensor, dst=0, group=self.jetspec_candidate_pg)
        if not self.config.jetspec_sync and self.config.jetspec_wait_ms != float("inf"):
            self.jetspec_mailbox_store.set(
                f"candidate_{self._jetspec_candidate_send_id}", "1"
            )
        for candidate_set in candidate_sets:
            if candidate_set is not None:
                self._jetspec_sent_candidates.add(
                    (candidate_set.seq_id, candidate_set.round_id)
                )
        if not self.config.jetspec_sync:
            self._jetspec_candidate_send_id += 1
        _flush_deferred()





    
























    @torch.inference_mode()
    def _warm_jetspec_session(self):
        """Exercise the real T module before any mailbox request is admitted."""
        if os.environ.get('W240_WARMUP','0') != '1' or not self.config.jetspec:
            raise RuntimeError('session warm-up requires the explicit measurement flag')
        if os.environ.get("JETSPEC_PAIR_OUTCOME_PUSH", "0") == "1":
            from jetspec.engines.jetspec_plus_plus.wire.pair_transport import PairOutcomeTransport

            self._pair_transport = PairOutcomeTransport(
                store=self.jetspec_mailbox_store, group=self.jetspec_request_pg,
                device=self.device, lanes=self.config.speculate_k + 1,
                hidden=self.config.jetspec_tap_hidden_size,
                tap_dtype=self.jetspec_drafter.dtype,
                envelope_bytes=(None if getattr(self, "_hybrid_pair_layout", None) is None else
                                self._hybrid_pair_layout.outcome_bytes))
            self._chain_inbox.transport = self._pair_transport
        d=self.jetspec_drafter
        state=JetSpecSequence(
            taps=torch.zeros(32,5,self.config.jetspec_tap_hidden_size,device=d.device,dtype=d.dtype),
            committed_ids=torch.zeros(1,33,device=d.device,dtype=torch.long),prompt_index=240)
        # Admission tail seeding is guarded by the live sequence id. Exercise
        # that path in cold warmup, before any measured request is admitted.
        state._jetspec_seq_id = -1
        ks=live_hypotheses(self.config.jetspec_hyp)
        if self.oracle_anchor:
            ks = [k for k in ks if k < 15]
        if self.jetspec_rows is not None:
            blocks=self.jetspec_rows.propose([state],k_hypotheses=ks,passes=self.config.jetspec_passes,initial=True)
            self.jetspec_rows.stage([state],[select_live_block(blocks[0],true_k=ks[0],correction=-1)])
            next_ks=live_hypotheses(self.config.jetspec_hyp,
                block=state.pending_block)
            oracle = {}
            if self.oracle_anchor:
                next_ks = [k for k in next_ks if k < 15] or [0]
                oracle['oracle_predictions'] = torch.zeros(1, 15, dtype=torch.int64, device=d.device)
            self.jetspec_rows.propose([state],k_hypotheses=next_ks,
                passes=self.config.jetspec_passes, **oracle)
            if getattr(self, '_w320_redraft', False):
                self.jetspec_rows.propose_fallback(state, true_k=0, passes=miss_redraft_passes())
            lengths = json.loads(os.environ.get('JETSPEC_WARMUP_PREFIX_LENGTHS', '[]'))
            # One prewarm pass per distinct length; a 128-prompt file has 65-77 of them.
            if (not isinstance(lengths, list)
                    or any(type(length) is not int or length < 2
                           or length + 16 > self.config.max_model_len for length in lengths)
                    or len(set(lengths)) > 256):
                raise ValueError('JETSPEC_WARMUP_PREFIX_LENGTHS must be at most 256 distinct valid prompt lengths')
            for length in sorted(set(lengths)):
                if length == 32:
                    continue
                seeded = JetSpecSequence(
                    taps=torch.zeros(length, 5, self.config.jetspec_tap_hidden_size,
                                     device=d.device, dtype=d.dtype),
                    committed_ids=torch.zeros(1, length + 1, device=d.device, dtype=torch.long),
                    prompt_index=240)
                seeded._jetspec_seq_id = -1
                seeded_blocks = self.jetspec_rows.propose(
                    [seeded], k_hypotheses=ks, passes=self.config.jetspec_passes, initial=True)
                self.jetspec_rows.stage([seeded], [select_live_block(
                    seeded_blocks[0], true_k=ks[0], correction=-1)])
                seeded_ks = live_hypotheses(self.config.jetspec_hyp,
                                             block=seeded.pending_block)
                if self.oracle_anchor:
                    seeded_ks = [k for k in seeded_ks if k < 15] or [0]
                self.jetspec_rows.propose([seeded], k_hypotheses=seeded_ks,
                    passes=self.config.jetspec_passes, **oracle)
                if getattr(self, '_w320_redraft', False):
                    self.jetspec_rows.propose_fallback(
                        seeded, true_k=0, passes=miss_redraft_passes())
            if lengths:
                print(f'W409_PREFIX_SHAPE_PREWARM lengths={sorted(set(lengths))}', flush=True)
            # Row counts are Triton constexprs: compile every size a live round can bring
            # (an anchor-prefix append of up to a full block, a rebuilt adapter cache)
            # here, not inside a measured round.
            decoder = getattr(self.jetspec_rows, 'decoder', None)
            if decoder is not None and hasattr(decoder, 'prewarm_prefix_appends'):
                def throwaway(length):
                    sequence = JetSpecSequence(
                        taps=torch.zeros(length, 5, self.config.jetspec_tap_hidden_size,
                                         device=d.device, dtype=d.dtype),
                        committed_ids=torch.zeros(1, length + 1, device=d.device,
                                                  dtype=torch.long),
                        prompt_index=240)
                    sequence._jetspec_seq_id = -1
                    return sequence
                counts = decoder.prewarm_prefix_appends(throwaway, range(1, draft_width() + 3))
                print(f'A03_DECODER_APPEND_PREWARM counts=1..{max(counts, default=0)}', flush=True)
            head = getattr(getattr(d, 'adapter', None), 'head', None)
            if head is not None and hasattr(head, 'prewarm_rebuild_rows'):
                buckets = head.prewarm_rebuild_rows(state.taps)
                print(f'A03_ADAPTER_REBUILD_PREWARM buckets={buckets}', flush=True)
            self.jetspec_rows.prewarm_flat_graph_buckets(self.config.max_model_len)
            from jetspec.engines.jetspec_plus_plus.draft.live_rows import batch_draft_graph
            if self.config.max_num_seqs > 1 and batch_draft_graph():
                self.jetspec_rows.prewarm_batch_graphs(
                    batch_sequence, max_batch=self.config.max_num_seqs,
                    max_model_len=self.config.max_model_len, initial_ks=ks,
                    round_ks=(list(range(draft_width() + 1))
                              if e04_lean() and os.environ.get('E04_ROWRULE') == 'joint'
                              else live_hypotheses(self.config.jetspec_hyp, block=state.pending_block)),
                    passes=self.config.jetspec_passes,
                    fallback_passes=(miss_redraft_passes() if getattr(self, '_w320_redraft', False)
                                     else None))
            if os.environ.get('JETSPEC_C1_GRAPH', '0') == '1':
                head = getattr(d.adapter, 'head', None)
                if head is not None and hasattr(head, 'prewarm_live_query_window'):
                    head.prewarm_live_query_window(15)
                    if os.environ.get('JETSPEC_KEYLEN_BUCKET', '0') == '1':
                        head.prewarm_keylen_graphs()
        else:
            blocks=d.propose(state,mode='kth',k_hypotheses=ks,passes=self.config.jetspec_passes)
            d.stage(state,blocks[0])
            d.propose_protocol(state,mode='kth',k_hypotheses=ks,passes=self.config.jetspec_passes)
        if os.environ.get('JETSPEC_C1_LOOP', '0') == '1':
            # The synthetic state can miss the live D1 window-mask path. Its first
            # call builds this CUDA extension (~42 s), so build it before decode.
            from jetspec.engines.jetspec_plus_plus.kernels.c1_fused import _loop_extension
            _loop_extension()
        if self._chain_mode:
            # Capture every segment before the measured request. The synthetic
            # admission only supplies pointers and shapes; real prefill re-seeds
            # the same buffers without recapturing.
            warm_state = JetSpecSequence(
                taps=torch.zeros(32, 5, self.config.jetspec_tap_hidden_size,
                                 device=d.device, dtype=d.dtype),
                committed_ids=torch.zeros(1, 33, dtype=torch.int64, device=d.device),
                prompt_index=240)
            # Eval admission stages the decoder swap; the hybrid decoder warms only seq -1.
            warm_state._jetspec_seq_id = -1
            warm_rows = self.jetspec_rows.propose(
                [warm_state], k_hypotheses=ks,
                passes=self.config.jetspec_passes, initial=True)[0]
            warm_id = -1
            self.jetspec_sequences[warm_id] = warm_state
            self.jetspec_candidate_sets[warm_id] = _candidate_set_from_rows(
                warm_id, 0, warm_rows)
            self.jetspec_block_objects[warm_id] = (0, warm_rows)
            try:
                chain_warm_started = time.perf_counter()
                self._init_chain_runtime(warm_id)
                chain_warm_seconds = time.perf_counter() - chain_warm_started
            finally:
                self.jetspec_sequences.pop(warm_id, None)
                self.jetspec_candidate_sets.pop(warm_id, None)
                self.jetspec_block_objects.pop(warm_id, None)
                self._chain_runtime = None
            from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled as eval_enabled
            # Eval replays each per-length admission graph once; the eager call is byte-equal.
            if (os.environ.get('JETSPEC_CHAIN_ADMIT_GRAPH', '0') == '1'
                    and os.environ.get('JETSPEC_CHAIN_ADMIT_BATCH', '0') == '1'
                    and not eval_enabled()):
                read = json.loads(os.environ.get('JETSPEC_CHAIN_WARM_READ') or '{}')
                prompt_lengths = read.get('prompt_lengths') or json.loads(
                    os.environ.get('JETSPEC_WARMUP_PREFIX_LENGTHS', '[]'))
                admit_rows = min(len(dict.fromkeys(int(k) for k in ks)),
                                 int(os.environ.get('JETSPEC_ROWS_CAP', '128')))
                for length in sorted(set(int(value) for value in prompt_lengths)):
                    self.jetspec_rows.prewarm_admission_graph(length, admit_rows)
            decoder = self.jetspec_rows.decoder
            window_states = getattr(decoder, '_window_graph_states', {})
            if window_states:
                window_captures = sum(len(state['graphs']) for state in window_states.values())
            else:
                state = getattr(decoder, '_window_graph_state', None)
                window_captures = 0 if state is None else len(state['graphs'])
            segment_captures = sum(len(entry.graphs) for entry in self._chain_graph.entries.values())
            allocated_stamp = (f' allocated_bytes={torch.cuda.memory_allocated(self.device)}'
                               if os.environ.get('JETSPEC_CHAIN_MISS_REDRAFT', '0') == '1'
                               else '')
            warm_stamp = (f' warm_seconds={chain_warm_seconds:.3f}'
                          if os.environ.get('JETSPEC_CHAIN_WARM_REACHABLE', '0') == '1' else '')
            print('CHAIN_GRAPH_PREWARM '
                  f'keys={len(self._chain_graph.entries)} '
                  f'segment_captures={segment_captures} '
                  f'decoder_window_captures={window_captures} '
                  f'reserved_bytes={torch.cuda.memory_reserved(self.device)}'
                  f'{allocated_stamp}'
                  f'{warm_stamp}', flush=True)
        torch.cuda.synchronize()
        print('W240_T_MODULE_WARMUP=complete synthetic_state_unpublished',flush=True)

    # new one, with true asynchrony

    @torch.inference_mode()
    def _init_chain_runtime(self, seq_id):
        """Bind the admission sequence and capture every reachable chain key."""
        from jetspec.engines.jetspec_plus_plus.wire.chain.contract import ChainKey; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import Offset; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import OutcomeOffset; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import OutcomePacket; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import allocate_token_buffers
        from jetspec.engines.jetspec_plus_plus.draft.chain.graph import ChainGraph; from jetspec.engines.jetspec_plus_plus.draft.chain.graph import chain_fixed_choice; from jetspec.engines.jetspec_plus_plus.draft.chain.graph import reachable_chain_capture; from jetspec.engines.jetspec_plus_plus.draft.chain.graph import reachable_chain_lengths
        from jetspec.engines.jetspec_plus_plus.wire.chain.intake import HASH_WEIGHTS; from jetspec.engines.jetspec_plus_plus.wire.chain.intake import ChainLengthTracker; from jetspec.engines.jetspec_plus_plus.wire.chain.intake import ChainRngTable; from jetspec.engines.jetspec_plus_plus.wire.chain.intake import ChainTapBuffers; from jetspec.engines.jetspec_plus_plus.wire.chain.intake import OutcomeReceiveRing; from jetspec.engines.jetspec_plus_plus.wire.chain.intake import PushedOutcomeRing; from jetspec.engines.jetspec_plus_plus.wire.chain.intake import PublishedRowRing; from jetspec.engines.jetspec_plus_plus.wire.chain.intake import hash_last16
        from jetspec.engines.jetspec_plus_plus.draft.chain.runtime import ChainDraftRecorder; from jetspec.engines.jetspec_plus_plus.draft.chain.runtime import ChainRoundDriver; from jetspec.engines.jetspec_plus_plus.draft.chain.runtime import configure_host_lean_entry
        from jetspec.engines.jetspec_plus_plus.draft.chain.served import build_served_entry; from jetspec.engines.jetspec_plus_plus.draft.chain.served import copy_front_state; from jetspec.engines.jetspec_plus_plus.draft.chain.served import retire_chain_runtime; from jetspec.engines.jetspec_plus_plus.draft.chain.served import reseed_served_graph; from jetspec.engines.jetspec_plus_plus.draft.chain.served import seed_front_from_admission; from jetspec.engines.jetspec_plus_plus.draft.chain.served import seed_row_ring_from_admission; from jetspec.engines.jetspec_plus_plus.draft.chain.served import shadow_state_tensors; from jetspec.engines.jetspec_plus_plus.draft.chain.served import stage_admission_rows
        from jetspec.engines.jetspec_plus_plus.models.heads.fast import dataset_entry; from jetspec.engines.jetspec_plus_plus.models.heads.fast import read_thresholds

        retire_chain_runtime(self)

        if self.jetspec_rows is None or seq_id not in self.jetspec_sequences:
            raise RuntimeError("chain admission requires one decoder row sequence")
        state = self.jetspec_sequences[seq_id]
        candidate = self.jetspec_candidate_sets[seq_id]
        _, blocks = self.jetspec_block_objects[seq_id]
        lanes = draft_width() + 1
        from jetspec.engines.jetspec_plus_plus.draft.serve_policy import rows_min_capacity; from jetspec.engines.jetspec_plus_plus.draft.serve_policy import rows_min_count
        R = rows_min_capacity(int(os.environ["JETSPEC_ROWS_RULE_CAP"]))
        if R not in (4, 8, 16):
            raise ValueError("chain served row cap must be 4, 8 or 16")
        capacity = self.config.max_model_len + lanes + 1
        initial_ids = state.committed_ids.reshape(-1).detach().clone()
        initial_taps = state.taps.detach().clone()
        L_admit = initial_taps.shape[0]
        if initial_ids.numel() != L_admit + 1:
            raise ValueError("chain admission must have one untapped query token")
        state._jetspec_seq_id = int(seq_id)  # real admission, not the synthetic warm-up state
        host_lean = (os.environ.get("JETSPEC_CHAIN_ADMIT_HOST_LEAN", "0") == "1"
                     and os.environ.get("JETSPEC_CHAIN_ADMIT_BATCH", "0") == "1")
        host_check = host_lean and os.environ.get("JETSPEC_CHAIN_ADMIT_HOST_LEAN_CHECK", "0") == "1"
        # Copy the committed IDs before the row stage queues admission work.
        initial_ids_host = initial_ids.cpu() if host_lean else None
        wire_row_cap = int(os.environ["JETSPEC_ROWS_CAP"])
        admission_hidden, admission_tau = stage_admission_rows(
            self.jetspec_rows, state, blocks, candidate, row_cap=wire_row_cap)
        if host_lean:
            heads_dir = os.environ["JETSPEC_HEADS_DIR"]
            cached = getattr(self, "_chain_admit_thresholds", None)
            if cached is None or cached[0] != heads_dir:
                cached = self._chain_admit_thresholds = (heads_dir, read_thresholds(heads_dir))
            thresholds = cached[1]
        else:
            thresholds = read_thresholds(os.environ["JETSPEC_HEADS_DIR"])
        dataset = os.environ["JETSPEC_DATASET"]
        rule = dataset_entry(thresholds["e04_rowrule_joint"], dataset)
        from jetspec.engines.jetspec_plus_plus.draft.pass_labels import chain_tau_values
        tau_values = chain_tau_values(rule['tau'], bundle=os.environ['JETSPEC_HEADS_DIR'])
        if host_lean:
            from jetspec.engines.jetspec_plus_plus.draft.rows import device_tensor
            tau_table = device_tensor(tau_values, self.device, dtype=torch.float64)
        else:
            tau_table = torch.tensor(tau_values, dtype=torch.float64, device=self.device)
        temperature = float(rule["temperature"]["value"])
        head = self.jetspec_drafter.adapter.head
        decoder = self.jetspec_rows.decoder
        max_decoder = max(decoder._window_buckets()) - lanes
        c1_capacity = getattr(head, "_c1_capacity", int(os.environ["W259_C1_CACHE_CAPACITY"]))
        from jetspec.engines.jetspec_plus_plus.draft.cache_capacity import chain_key_limit
        key_limit = chain_key_limit(self.config.max_model_len, lanes)
        if c1_capacity < key_limit or max_decoder < key_limit:
            raise ValueError(f"chain startup requires C1 capacity >= {key_limit} and "
                             f"decoder bucket >= {key_limit + lanes}; "
                             f"got C1={c1_capacity}, decoder={max_decoder + lanes}")
        reachable = os.environ.get("JETSPEC_CHAIN_WARM_REACHABLE", "0") == "1"
        if reachable:
            read = json.loads(os.environ["JETSPEC_CHAIN_WARM_READ"])
            lengths = reachable_chain_lengths(read["prompt_lengths"], read["tokens"],
                                              lanes, key_limit)
            lengths = tuple(sorted(set((*lengths, key_limit))))
            if os.environ.get("JETSPEC_CHAIN_WARM_SINGLE_KEY", "0") == "1":
                # Every shorter exact-L packet is covered by this ceiling key.
                # Keep segment pools private; avoid duplicate hybrid decoder
                # state and graph workspaces for unused smaller prefix keys.
                lengths = (key_limit,)
            row_caps, capture_names = reachable_chain_capture(
                os.environ.get("JETSPEC_CHAIN_FIXED_CONFIG"), R,
                bool(getattr(self._w320_clock, "chain_tau", None)),
                bool(getattr(self._w320_clock, "chain_tau_cap", False)))
        else:
            first = max(128, ((L_admit + 2 * lanes + 127) // 128) * 128)
            last = key_limit
            if first > last:
                raise ValueError("chain admission has no safe captured length key")
            lengths = range(first, last + 1, 128)
            row_caps, capture_names = ((R,) if R == 16 or rows_min_count() > 4 else (4, 8)), None
        last = lengths[-1]
        decoder_key_bucket = os.environ.get("JETSPEC_CHAIN_DECODER_KEY_BUCKET", "0") == "1"

        def chain_key(length, row_cap):
            window_bucket = (next(bucket for bucket in decoder._window_buckets()
                                  if bucket >= length + lanes)
                             if decoder_key_bucket else 0)
            return ChainKey(length, row_cap, 3, "after_j0_j1", lanes,
                            decoder_key_bucket=window_bucket)

        if host_lean:
            key_spec = (tuple(lengths), tuple(row_caps), decoder_key_bucket,
                        tuple(decoder._window_buckets()))
            cached = getattr(self, "_chain_admit_keys", None)
            if cached is None or cached[0] != key_spec:
                cached = self._chain_admit_keys = (
                    key_spec, tuple(chain_key(length, row_cap)
                                    for length in lengths for row_cap in row_caps))
            keys = cached[1]
        else:
            keys = tuple(chain_key(length, row_cap)
                         for length in lengths for row_cap in row_caps)
        graph = getattr(self, "_chain_graph", None)
        if graph is None:
            tokens = allocate_token_buffers(device=self.device, capacity=capacity, lanes=lanes)
            # Row passes pad their prefix to 256, including the new ceiling key.
            tap_capacity = max(capacity, ((key_limit + 255) // 256) * 256)
            tap_store = ChainTapBuffers(device=self.device, capacity=tap_capacity, lanes=lanes,
                                        hidden=initial_taps.shape[-1], dtype=initial_taps.dtype)
            row_ring = PublishedRowRing(device=self.device, rows=wire_row_cap, lanes=lanes)
        else:
            if (not graph._prewarmed or any(key not in graph.entries for key in keys)
                    or self._chain_static_capacity != capacity
                    or self._chain_static_wire_cap != wire_row_cap
                    or self._chain_static_hidden != initial_taps.shape[-1]):
                raise RuntimeError("chain warm-up graph does not cover this sequence shape")
            shared = next(iter(graph.entries.values())).buffers
            tokens, tap_store, row_ring = (shared[name] for name in
                                           ("tokens", "tap_store", "row_ring"))
        if graph is None:
            tokens.committed_ids.zero_()
            tokens.seed_committed(initial_ids)
            tokens.pending_draft_ids.zero_()
            tap_store.storage.zero_()
            tap_store.seed(initial_taps)
            seed_row_ring_from_admission(row_ring, candidate, blocks, admission_tau)
        if host_lean:
            fake_ids = torch.zeros(capacity, dtype=torch.int64)
            fake_ids[:initial_ids_host.numel()].copy_(initial_ids_host)
            host_packet = OutcomePacket(
                1, seq_id, 0, int(initial_ids_host[-1]), 0, 0, 0,
                L=L_admit,
                hash16=int(hash_last16(
                    fake_ids, torch.tensor(L_admit + 1),
                    torch.tensor(HASH_WEIGHTS, dtype=torch.int64))),
                k=-1, a=-1).pack()
            if host_check:
                device_ids = torch.zeros(capacity, dtype=torch.int64, device=self.device)
                device_ids[:initial_ids.numel()].copy_(initial_ids)
                device_packet = OutcomePacket(
                    1, seq_id, 0, int(initial_ids[-1]), 0, 0, 0,
                    L=L_admit,
                    hash16=int(hash_last16(
                        device_ids, torch.tensor(L_admit + 1, device=self.device),
                        torch.tensor(HASH_WEIGHTS, dtype=torch.int64, device=self.device))),
                    k=-1, a=-1).pack()
                mismatches = int((host_packet != device_packet.cpu()).sum())
                print(f"CHAIN_ADMIT_HOST_LEAN_CHECK mismatch={mismatches}")
                if mismatches:
                    raise RuntimeError("host round-0 packet differs from device packet")
            fake_packet = (host_packet.pin_memory().to(self.device, non_blocking=True)
                           if torch.device(self.device).type == "cuda"
                           else host_packet.to(self.device))
        else:
            fake_ids = torch.zeros(capacity, dtype=torch.int64, device=self.device)
            fake_ids[:initial_ids.numel()].copy_(initial_ids)
            fake_packet = OutcomePacket(
                1, seq_id, 0, int(initial_ids[-1]), 0, 0, 0,
                L=L_admit,
                hash16=int(hash_last16(
                    fake_ids, torch.tensor(L_admit + 1, device=self.device),
                    torch.tensor(HASH_WEIGHTS, dtype=torch.int64, device=self.device))),
                k=-1, a=-1).pack().to(self.device)

        def build(key):
            buffers, bodies = build_served_entry(
                key, rows=self.jetspec_rows, tokens=tokens, tap_store=tap_store,
                row_ring=row_ring, thresholds=thresholds, dataset=dataset,
                tau_table=tau_table, temperature=temperature,
                hypotheses=self.config.jetspec_hyp,
                wire_row_cap=wire_row_cap, admission=state,
                admission_hidden=admission_hidden, admission_tau=admission_tau)
            buffers["packet"].copy_(fake_packet)
            buffers["taps"].zero_()
            buffers["rp"].zero_()
            if os.environ.get("JETSPEC_CHAIN_ATTN3", "0") == "1":
                buffers["control_round"] = torch.zeros(
                    (), dtype=torch.int64, device=self.device)
                buffers["control_L"] = torch.full(
                    (), L_admit, dtype=torch.int64, device=self.device)
            if os.environ.get("JETSPEC_CHAIN_HOST_LEAN", "0") == "1":
                bodies = configure_host_lean_entry(buffers, bodies)
            return buffers, bodies

        def restore(key, buffers):
            tokens.committed_ids.zero_()
            tokens.seed_committed(initial_ids)
            tokens.pending_draft_ids.zero_()
            tap_store.storage.zero_()
            tap_store.seed(initial_taps)
            for old_round in range(3):
                row_ring.empty(old_round)
            seed_row_ring_from_admission(row_ring, candidate, blocks, admission_tau)
            seed_front_from_admission(buffers["front"], state, decoder,
                                      thresholds, dataset)
            if os.environ.get("JETSPEC_CHAIN_ATTN3", "0") == "1":

                # Warm each captured key's short and long exact-L plans before
                # measured requests; graph capture itself uses admission L.
                for warm_L in (max(1, key.L_bucket - 127), key.L_bucket):
                    plan_served_prefix(head, key, warm_L, refresh=True)

        if graph is None:
            kwargs = {} if capture_names is None else dict(capture_names=capture_names)
            graph = ChainGraph(
                build, capacity=len(keys), **kwargs,
                shadow_state=lambda key, buffers: shadow_state_tensors(
                    buffers, self.jetspec_rows))
            graph.prewarm(keys, after_capture=restore)
            self._chain_static_capacity = capacity
            self._chain_static_wire_cap = wire_row_cap
            self._chain_static_hidden = initial_taps.shape[-1]
        else:
            fresh_rng = ChainRngTable(
                capacity=capacity, lanes=lanes, seed=self.jetspec_drafter.seed,
                prompt_index=state.prompt_index, device=self.device)
            reseed_served_graph(
                graph, initial_ids=initial_ids, initial_taps=initial_taps,
                candidate=candidate, blocks=blocks,
                admission_hidden=admission_hidden, admission_tau=admission_tau,
                rng_rows=fresh_rng.rows, fake_packet=fake_packet)
        self._chain_graph = graph
        self._chain_last_front = None
        self._chain_last_key = None
        self._chain_preselected_choice = (2, False)
        fixed_config = os.environ.get("JETSPEC_CHAIN_FIXED_CONFIG")
        switch_value = os.environ.get("JETSPEC_CHAIN_J_SWITCH_L")
        from jetspec.engines.jetspec_plus_plus.serving.presets.budget import budget_type
        switch_L = budget_type().switch_length(switch_value)
        handoff = os.environ.get("JETSPEC_CHAIN_HANDOFF", "0") == "1"
        prepost = os.environ.get("JETSPEC_CHAIN_PREPOST", "0") == "1"
        post_lean = os.environ.get("JETSPEC_CHAIN_POST_LEAN", "0") == "1"
        anatomy_lite = os.environ.get("JETSPEC_ANATOMY_LITE", "0") != "0"
        tracker = ChainLengthTracker(capacity=last, lanes=lanes,
                                     L_known=L_admit, mirrored_round=-1)

        def covering_length(length):
            return next((covered for covered in lengths if covered >= length
                         and all(chain_key(covered, row_cap) in graph.entries
                                 for row_cap in row_caps)), None)

        def key_for(tracker, round_id):
            length = covering_length(tracker.bucket(round_id, step=128))
            if length is None:
                return None
            candidates = tuple(chain_key(length, row_cap)
                               for row_cap in row_caps)
            clock = self._w320_clock
            if fixed_config is not None:
                key = candidates[row_caps.index(R)] if R in row_caps else candidates[0]
                choice = chain_fixed_choice(fixed_config, length, switch_L)
                if not (handoff or prepost or clock is None):
                    clock.arrive(time.perf_counter(), clean=True)
                    launch_config = (f"J{choice[0]}-R{int(choice[1])}"
                                     if switch_L is not None else fixed_config)
                    choice = clock.launch_chain_fixed(launch_config, key)
            elif clock is None:
                key, choice = candidates[1 if R == 8 else 0], (2, False)
            else:
                now = time.perf_counter()
                clock.arrive(now, clean=True)
                try:
                    if len(candidates) == 1 and (reachable or not clock.chain_tau_cap):
                        key = candidates[0]
                        choice = clock.launch_chain(now, key)
                    else:
                        key, choice = clock.launch_chain_capacity(now, candidates)
                except DraftDeadlineMiss:
                    key, choice = candidates[1 if R == 8 else 0], budget_type().fallback_choice()
            self._chain_preselected_choice = choice
            return key

        def fallback(packet, taps, round_id):
            del taps
            # Last resort: the received packet supplies the exact live prefix.
            # Re-enter the driver's ordinary captured draft and publication path.
            length = covering_length(int(packet[OutcomeOffset.L].item()))
            if length is None:
                raise RuntimeError("chain capacity has no captured graph covering the live prefix")
            graph.fallback_counts["capacity"] += 1
            print("CHAIN_FALLBACK " + json.dumps(dict(
                seq_id=seq_id, round=round_id + 1, reason="capacity",
                last_resort=True, key=length,
                count=graph.fallback_counts["capacity"])), flush=True)
            self._chain_preselected_choice = (chain_fixed_choice(fixed_config, length, switch_L)
                                             if fixed_config is not None else (1, False))
            return chain_key(length, R if R in row_caps else row_caps[0])

        def select(key):
            entry = graph.entries[key]
            if handoff and fixed_config is not None and not prepost:
                clock = self._w320_clock
                if clock is not None:
                    clock.arrive(time.perf_counter(), clean=True)
                    choice = self._chain_preselected_choice
                    launch_config = (f"J{choice[0]}-R{int(choice[1])}"
                                     if switch_L is not None else fixed_config)
                    clock.launch_chain_fixed(launch_config, key)
            if self._chain_last_key != key:
                if self._chain_last_front is None:
                    seed_front_from_admission(entry.buffers["front"], state, decoder,
                                              thresholds, dataset)
                else:
                    copy_front_state(self._chain_last_front, entry.buffers["front"],
                                     prefix_length=max(0, tracker.L_known))
            self._chain_last_front = entry.buffers["front"]
            self._chain_last_key = key
            choice = self._chain_preselected_choice
            self._chain_last_choice = choice
            if os.environ.get("JETSPEC_CHAIN_HOST_LEAN", "0") != "1":
                entry.buffers["rp"][Offset.J_ROUND].fill_(choice[0])
            return choice

        def publish_redraft(buffers, round_id):
            self.jetspec_mailbox_store.set(
                f"candidate_payload_{self._jetspec_candidate_send_id}",
                buffers["redraft_home"].numpy().tobytes())
            self._jetspec_sent_candidates.add((seq_id, round_id))
            self._jetspec_candidate_send_id += 1

        def publish(buffers, round_id):
            timing = os.environ.get("JETSPEC_CHAIN_HANDOFF_TIMING", "0") == "1"
            publish_start = time.perf_counter_ns() if timing else None
            records = []
            self.jetspec_mailbox_store.set(
                f"candidate_payload_{self._jetspec_candidate_send_id}",
                buffers["wire_home"].numpy().tobytes())
            lite_publish_ns = time.monotonic_ns() if anatomy_lite else None
            if timing:
                publish_end = time.perf_counter_ns()
                records.append(("CHAIN_HANDOFF_PUBLISH ", dict(
                    seq_id=seq_id, round=round_id + 1,
                    publish_ns=publish_end,
                    store_us=round((publish_end - publish_start) / 1000, 1))))
            self._jetspec_sent_candidates.add((seq_id, round_id + 1))
            self._jetspec_candidate_send_id += 1
            if self._w320_clock is not None and not prepost:
                self._w320_clock.learn_chain_times(
                    self._chain_last_key, graph.last_ms.get(self._chain_last_key, {}), 0.0)
                deadline = self._w320_clock.published(time.perf_counter())
                records.append(('W320_DEADLINE ', dict(
                    seq_id=seq_id, round_id=round_id + 1, **deadline)))
            records.append(("CHAIN_ROUND ", dict(
                seq_id=seq_id, round=round_id, published_round=round_id + 1,
                L=int(receiver.mirrors[receiver.slot_for_round(round_id)][OutcomeOffset.L]),
                key=self._chain_last_key.L_bucket,
                config=f"J{self._chain_last_choice[0]}-R{int(self._chain_last_choice[1])}",
                row_cap=self._chain_last_key.R, sync_total=graph.round_syncs,
                **({'lite': [buffers['_lite_command_ns'], buffers['_lite_chain_end_ns'],
                    lite_publish_ns, round(graph.last_span_ms[self._chain_last_key], 6),
                    {name: round(value, 6) for name, value in
                     graph.last_ms[self._chain_last_key].items()}]} if anatomy_lite else {}),
                **({"redraft": bool(buffers.get("host_redrafted", False))}
                   if os.environ.get("JETSPEC_CHAIN_MISS_REDRAFT", "0") == "1"
                   else {}),
                mix=dict(graph.mix), cap_mix=dict(graph.cap_mix),
                fallback=dict(graph.fallback_counts))))
            if prepost or post_lean:
                def emit():
                    for prefix, payload in records:
                        print(prefix + json.dumps(payload), flush=True)
                return emit
            for prefix, payload in records:
                print(prefix + json.dumps(payload), flush=True)

        pair_push = os.environ.get("JETSPEC_PAIR_OUTCOME_PUSH", "0") == "1"
        pair_check = os.environ.get("JETSPEC_PAIR_OUTCOME_PUSH_CHECK", "0") == "1"
        shadow = (OutcomeReceiveRing(device=self.device, lanes=lanes,
                                     hidden=initial_taps.shape[-1],
                                     tap_dtype=initial_taps.dtype,
                                     hybrid_layout=getattr(self, "_hybrid_pair_layout", None))
                  if not pair_push or pair_check else None)
        receiver = (PushedOutcomeRing(
            transport=self._pair_transport, inbox=self._chain_inbox,
            lanes=lanes, hidden=initial_taps.shape[-1],
            tap_dtype=initial_taps.dtype, shadow=shadow,
            hybrid_layout=getattr(self, "_hybrid_pair_layout", None))
            if pair_push else shadow)
        def pre_stage(tracker, round_id):
            # The tracker is intentionally one packet behind here. Its length
            # bound covers the next round without waiting for this mirror.
            length = covering_length(tracker.bucket(round_id, step=128))
            if length is None:
                return None  # let the fresh mirror decide near capacity
            key = chain_key(length, R if fixed_config is not None else row_caps[0])
            self._chain_preselected_choice = chain_fixed_choice(fixed_config, length, switch_L)
            return round_id, key, graph.prepare(key)

        recorder = None
        record_root = os.environ.get("JETSPEC_CHAIN_RECORD")
        if record_root and seq_id != -1:
            seen = getattr(self, "_chain_record_requests_seen", 0)
            self._chain_record_requests_seen = seen + 1
            if seen < int(os.environ.get("JETSPEC_CHAIN_RECORD_SKIP_REQUESTS", "0")):
                record_root = None
        if record_root and seq_id != -1:  # synthetic graph prewarm is not a request
            recorder = getattr(self, "_chain_recorder", None)
            if recorder is None or not recorder.closed:
                from jetspec.engines.jetspec_plus_plus.draft.env import REGISTRY

                prefix = decoder._prefixes[id(state)]
                admission = dict(
                    environment={name: value for name, value in os.environ.items()
                                 if (name in REGISTRY or name == "SSD_ATTENTION_BACKEND")
                                 and name not in ("JETSPEC_CHAIN_RECORD", "JETSPEC_PINS")},
                    pins=json.loads(open(os.environ["JETSPEC_PINS"]).read()),
                    reference_head=os.environ["JETSPEC_CHAIN_REFERENCE_HEAD"],
                    capacity=capacity, wire_row_cap=wire_row_cap,
                    hypotheses=self.config.jetspec_hyp,
                    prompt_index=state.prompt_index,
                    initial_ids=initial_ids, initial_taps=initial_taps,
                    staged_hidden=state.staged.predicted_hidden,
                    prefix_length=prefix.length, prefix_kv=prefix.kv,
                    admission_hidden=admission_hidden, admission_tau=admission_tau,
                    fake_packet=fake_packet,
                    reserved_bytes=torch.cuda.memory_reserved(self.device),
                    captured_segments=sum(len(entry.graphs) for entry in graph.entries.values()),
                    ring=dict(rows=row_ring.rows, keys=row_ring.keys,
                              p_top1=row_ring.p_top1,
                              passes_run=row_ring.passes_run, rounds=row_ring.rounds))
            if recorder is None:
                recorder = ChainDraftRecorder(
                    record_root,
                    limit=int(os.environ.get("JETSPEC_CHAIN_RECORD_ROUNDS", "300")),
                    admission=admission)
                self._chain_recorder = recorder
            elif not recorder.closed:
                recorder.mark_request(admission)

        self._chain_runtime = ChainRoundDriver(
            store=self.jetspec_mailbox_store, request_group=self.jetspec_request_pg,
            graph=graph, receiver=receiver, tracker=tracker, seq_id=seq_id,
            key_for=key_for, select=select, publish=publish, inbox=self._chain_inbox,
            publish_redraft=publish_redraft, fallback=fallback, start_round=0,
            pre_stage=(pre_stage if handoff else None), recorder=recorder,
            rows=self.jetspec_rows)

    def draft_loop(self):
        """
        Runs the asynchronous draft model loop.
        Handles prefill (1), speculation (0), exit (2), warmup (3), and pin check (4).
        """
        assert self.draft_async, "draft_loop only runs in async-draft mode"
        chain_mode = getattr(self, "_chain_mode", False)

        if chain_mode:
            from jetspec.engines.jetspec_plus_plus.wire.chain.intake import ChainCommandInbox
            self._chain_inbox = ChainCommandInbox(self.jetspec_mailbox_store)

        while True:
            # 1) Wait for the next command (may be PREFILL, SPEC_REQUEST, or EXIT)
            command_pg = (
                self.jetspec_request_pg if self.config.jetspec else self.async_pg
            )
            if chain_mode:
                if self._chain_runtime is None:
                    _, command = self._chain_inbox.next()
                else:
                    command = self._chain_runtime.next()
                cmd = {"outcome": 0, "prefill": 1, "exit": 2,
                       "warmup": 3, "pin_served": 4}[command]
            else:
                cmd = self.recv_cmd(group=command_pg)

            # PREFILL: run the draft prefill and then loop back
            if cmd == 1:
                self.draft_async_prefill()
                continue

            elif cmd == 3:
                self._warm_jetspec_session()
                dist.send(torch.ones(1,dtype=torch.int64,device=self.device),dst=0,group=command_pg)
                continue

            # Check the draft rank before the target seals the run manifest.
            elif cmd == 4:
                try:
                    self._check_draft_compile_pin_served()
                except Exception:
                    dist.send(torch.zeros(1, dtype=torch.int64, device=self.device),
                              dst=0, group=command_pg)
                    raise
                dist.send(torch.ones(1, dtype=torch.int64, device=self.device),
                          dst=0, group=command_pg)
                continue

            # SPECULATE request: serve out-of-cache or random speculations
            elif cmd == 0:
                if chain_mode:
                    continue
                _ds0 = time.perf_counter()
                _prof = os.environ.get("SSD_PROFILE", "0") == "1"
                if _prof or PROFILE_DRAFT:
                    torch.cuda.synchronize()
                    _d0 = time.perf_counter()

                clock = getattr(self, '_w320_clock', None)
                if clock is not None:
                    clock.arrive(_ds0, clean=False)

                glue_decode_input_ids, partial_tree_decode_args = self._service_spec_request()

                if self.config.jetspec:
                    if self.config.jetspec_sync:
                        self._stage_jetspec_sync(partial_tree_decode_args)
                    elif self.config.jetspec_wait_ms == float("inf"):
                        # Preserve the accepted pre-C waiting pipeline: round 0
                        # was pushed after prefill, and every verifier request
                        # prepares and pushes round+1 before the next request.
                        seq_ids = partial_tree_decode_args["seq_ids"]
                        self._jetspec_early_published = self._jetspec_early_clock = None
                        self._jetspec_deadline_clock = None
                        self._jetspec_deadline_miss = None
                        self._advance_jetspec(partial_tree_decode_args)
                        early = self._jetspec_early_published
                        if early is not None:
                            # JETSPEC_TAIL_PUBLISH: already on the wire from inside the refinement
                            if [early] != [(int(s), int(r) + 1) for s, r in zip(
                                    seq_ids.tolist(), partial_tree_decode_args['rounds'].tolist())]:
                                raise RuntimeError('early publish does not match the request')
                            if os.environ.get('W327_PUBLISH_CHECK') == '1':
                                self._check_early_packet(seq_ids)
                            _IDLE.extend(_DEFERRED)  # after the next round's passes are queued
                            _DEFERRED.clear()
                        elif self._jetspec_deadline_miss is not None:
                            seq_id, next_round = self._jetspec_deadline_miss
                            if [(seq_id, next_round)] != [(int(s), int(r) + 1) for s, r in zip(
                                    seq_ids.tolist(), partial_tree_decode_args['rounds'].tolist())]:
                                raise RuntimeError('deadline miss does not match the request')
                            self._send_jetspec_deadline_miss(seq_id)
                        elif not self.oracle_anchor or bool((partial_tree_decode_args['rounds'] > 0).any()):
                            self._send_jetspec_candidates(seq_ids)
                        if clock is not None:
                            record = (self._jetspec_early_clock if early is not None else
                                      self._jetspec_deadline_clock if self._jetspec_deadline_miss is not None else
                                      clock.published(time.perf_counter()))
                            print('W320_DEADLINE ' + json.dumps(dict(
                                seq_id=int(seq_ids[0]), round_id=int(partial_tree_decode_args['rounds'][0]) + 1,
                                **record)), flush=True)
                    else:
                        self._serve_latest_jetspec_request(
                            partial_tree_decode_args
                        )
                    self._draft_step_times.append(time.perf_counter() - _ds0)
                    continue

                if _prof or PROFILE_DRAFT:
                    torch.cuda.synchronize()
                    _d1 = time.perf_counter()

                self._reset_tree_cache_tensors()

                tree_decode_args = self._build_tree_batch(partial_tree_decode_args, glue_decode_input_ids)

                if _prof or PROFILE_DRAFT:
                    torch.cuda.synchronize()
                    _d2 = time.perf_counter()

                # Decode the branch tree
                tokens, logits, activations = self._decode_tree(tree_decode_args)

                if _prof or PROFILE_DRAFT:
                    torch.cuda.synchronize()
                    _d3 = time.perf_counter()

                # Populate the local cache so future spec-requests can hit
                self._populate_tree_cache(tree_decode_args, tokens, logits, tree_decode_args["cache_hits"], activations)
                self._draft_step_times.append(time.perf_counter() - _ds0)

                if _prof or PROFILE_DRAFT:
                    torch.cuda.synchronize()
                    _d4 = time.perf_counter()
                    print(f"[PROFILE draft] service={(_d1-_d0)*1000:.2f}ms build_tree={(_d2-_d1)*1000:.2f}ms decode_tree={(_d3-_d2)*1000:.2f}ms populate={(_d4-_d3)*1000:.2f}ms total={(_d4-_d0)*1000:.2f}ms", flush=True)

                if PROFILE_DRAFT:
                    flush_draft_profile()

                continue

            # EXIT: clean up and break out of the loop
            elif cmd == 2:
                _run_idle()
                if self._draft_step_times:
                    avg_ms = sum(self._draft_step_times) * 1000 / len(self._draft_step_times)
                    print(f"[metrics] Avg draft step time (ms): {avg_ms:.2f}", flush=True)
                if self.config.jetspec_taps:
                    recv_sorted = sorted(self._jetspec_tap_recv_ms)
                    bytes_sorted = sorted(self._jetspec_tap_payload_bytes)
                    p90_index = max(0, (9 * len(recv_sorted) + 9) // 10 - 1)
                    mirror_bytes = sum(
                        taps.numel() * taps.element_size()
                        for taps in self.jetspec_tap_mirrors.values()
                    )
                    summary = {
                        "rounds": len(recv_sorted),
                        "recv_ms": self._jetspec_tap_recv_ms,
                        "recv_ms_mean": (
                            sum(recv_sorted) / len(recv_sorted)
                            if recv_sorted else 0.0
                        ),
                        "recv_ms_p90": recv_sorted[p90_index] if recv_sorted else 0.0,
                        "payload_bytes_mean": (
                            sum(bytes_sorted) / len(bytes_sorted)
                            if bytes_sorted else 0.0
                        ),
                        "payload_bytes_p90": (
                            bytes_sorted[p90_index] if bytes_sorted else 0
                        ),
                        "mirror_bytes": mirror_bytes,
                        "mirror_lengths": {
                            str(seq_id): taps.shape[0]
                            for seq_id, taps in self.jetspec_tap_mirrors.items()
                        },
                    }
                    print(
                        f"[jetspec_taps] draft_metrics={json.dumps(summary, sort_keys=True)}",
                        flush=True,
                    )
                self.exit()
                break

            else:
                raise RuntimeError(f"draft_loop: unknown command {cmd}")

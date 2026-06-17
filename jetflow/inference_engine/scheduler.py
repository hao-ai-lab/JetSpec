"""Continuous-batching scheduler (JetFlow N2a).

Controls admission, batching, and eviction for a batch of sequences sharing one
`PagedKVCache` pool. The engine (`JetFlowEngine.generate_batch`) owns the forward;
the scheduler owns the bookkeeping: a FIFO waiting queue, a `seq_id -> request`
running set, and the eviction policy that frees pool room when a new prefill
can't fit. It is a thin policy layer over the cache's own admit/allocate/evict
primitives (`admit_sequence` / `allocate_slots` / `evict_sequence`); the cache
is the source of truth for what physically fits, the scheduler only decides
*which* sequence yields.

Request lifecycle::

    WAITING -> (admit) -> PREFILL -> (first decode) -> DECODE -> (EOS) -> FINISHED
                                                          |
                                                     (evict) -> CHECKPOINTED -> (resume) -> WAITING

N2a scope: FCFS admission + LRU eviction (the cache's own LRU clock picks the
victim). The other policies named in the design (`priority`, `sjf`, `age` /
`priority_score`, `random`) are accepted and wired to the obvious key so callers
can opt in, but FCFS+LRU is the gated path.
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from jetflow.inference_engine.paged_kv_cache import EvictionFailed, PagedKVCache


@dataclass
class SequenceRequest:
    """One inflight request (prompt or ongoing decode).

    `input_ids` is the full prompt at admission; once decoding it carries the
    single next token. `prompt_len` caches the prompt length for telemetry /
    checkpoint restore. State fields (`output_tokens`, `logits_last`,
    `is_prefill`) are filled by the engine as the sequence advances."""

    seq_id: int
    input_ids: list                  # [prompt_tokens] (prefill) or [next_token] (decode)
    prompt_len: Optional[int] = None

    # Metadata
    arrival_time: float = field(default_factory=time.time)
    priority: float = 0.0            # lower = evict first under priority policy
    max_new_tokens: int = 256
    temperature: float = 0.0

    # Tree decode config (optional; reserved for N2b)
    use_tree: bool = False
    tree_drafter: Optional[object] = None
    tree_accept_fn: Optional[Callable] = None

    # State (filled by the engine)
    output_tokens: list = field(default_factory=list)
    logits_last: Optional[torch.Tensor] = None
    is_prefill: bool = True          # True until the first decode step

    def __post_init__(self) -> None:
        if self.prompt_len is None:
            self.prompt_len = len(self.input_ids)


@dataclass
class SchedulerOutput:
    """The scheduler's decision for one forward iteration."""

    admitted: list = field(default_factory=list)   # new reqs to prefill this step
    running: list = field(default_factory=list)     # decode + new prefill (the batch)
    evicted: list = field(default_factory=list)     # spilled (checkpointed & freed)
    finished: list = field(default_factory=list)    # EOS reached

    # Telemetry
    timestamp: float = field(default_factory=time.time)
    cache_utilization: float = 0.0   # blocks_used / blocks_total
    batch_size: int = 0              # len(running)
    prefill_tokens: int = 0          # sum of new prompts admitted this step
    decode_tokens: int = 0           # sum of existing decode steps this step
    admission_reason: str = ""       # "prefill" / "decode" / "batched"


class Scheduler:
    """Admission, batching, and eviction over a shared `PagedKVCache`.

    `step()` is called once per forward iteration: it drains the waiting queue
    into the pool (evicting the LRU running sequence when the pool is full), then
    reports the running batch the engine should forward. The engine drives the
    forward and reports back via `mark_decode_step` / `mark_finished`.
    """

    def __init__(
        self,
        cache: PagedKVCache,
        max_batch_size: int = 32,
        prefill_batch_size: int = 8,
        scheduling_policy: str = "fcfs",
        eviction_policy: str = "lru",
    ):
        self.cache = cache
        self.max_batch_size = int(max_batch_size)
        self.prefill_batch_size = int(prefill_batch_size)
        self.policy = scheduling_policy
        self.evict_policy = eviction_policy
        self.waiting_queue: deque = deque()        # FIFO of SequenceRequest
        self.running: dict = {}                    # seq_id -> SequenceRequest
        self.finished: dict = {}                   # seq_id -> SequenceRequest
        self.evicted: dict = {}                    # seq_id -> checkpoint dict

    # --- admission -----------------------------------------------------------

    def admit_request(self, req: SequenceRequest) -> bool:
        """Queue a new request for prefill (FIFO). Returns False if the pool's
        batch budget is already full (caller must back off)."""
        if len(self.waiting_queue) + len(self.running) >= self.cache._max_batch_size:
            return False
        self.waiting_queue.append(req)
        return True

    def step(self) -> SchedulerOutput:
        """Decide the batch for the next forward (admit waiting reqs, evicting to
        fit), and report the running set + telemetry.

        FCFS+LRU: pop waiting requests oldest-first; for each, ask the cache if it
        admits (`admit_sequence` accounts for blocks reclaimable by eviction). If
        it fits, reserve its blocks (`allocate_slots`, which evicts the LRU *other*
        sequence as needed) and admit; if not even eviction frees enough room, stop
        draining (the queue blocks until a running seq finishes)."""
        output = SchedulerOutput()
        output.timestamp = time.time()

        # Step 1: drain the waiting queue into the pool, evicting to fit.
        while self.waiting_queue and len(self.running) + len(output.admitted) < self.max_batch_size:
            req = self.waiting_queue[0]
            if not self.cache.admit_sequence(req.seq_id, req.prompt_len):
                break                              # cannot fit even after eviction
            # Reserve blocks; allocate_slots evicts the LRU *other* sequence when
            # the pool is full. Capture which running seqs it evicted.
            before = set(self.cache._seq_block_tables.keys())
            if not self.cache.allocate_slots(req.seq_id, req.prompt_len):
                break                              # pool can't fit even fully drained
            evicted_ids = before - set(self.cache._seq_block_tables.keys())
            for sid in evicted_ids:
                victim = self.running.pop(sid, None)
                if victim is not None:
                    self.evicted[sid] = self._checkpoint_seq(victim)
                    output.evicted.append(victim)
            output.admitted.append(self.waiting_queue.popleft())

        # Step 2: the batch is the surviving running set + the newly admitted.
        output.running = list(self.running.values()) + output.admitted

        # Step 3: promote admitted reqs to running (they prefill this step).
        for req in output.admitted:
            self.running[req.seq_id] = req
            req.is_prefill = True

        output.prefill_tokens = sum(r.prompt_len for r in output.admitted)
        output.decode_tokens = sum(1 for r in output.running if not r.is_prefill)
        output.batch_size = len(output.running)
        output.cache_utilization = self._compute_utilization()
        output.admission_reason = (
            "batched" if output.admitted and output.decode_tokens
            else "prefill" if output.admitted else "decode"
        )
        return output

    # --- state transitions ---------------------------------------------------

    def mark_finished(self, seq_id: int) -> None:
        """Sequence reached EOS / its token budget: move to finished, free blocks."""
        req = self.running.pop(seq_id, None)
        if req is not None:
            self.finished[seq_id] = req
            self.cache.free(seq_id=seq_id)

    def mark_decode_step(self, seq_id: int, next_token: int,
                         logits: Optional[torch.Tensor] = None) -> None:
        """Record a completed decode step: append the token, switch the request to
        single-token decode mode, and stash the last logits for telemetry."""
        req = self.running.get(seq_id)
        if req is None:
            return
        req.output_tokens.append(int(next_token))
        req.input_ids = [int(next_token)]          # next step feeds one token
        req.is_prefill = False
        req.logits_last = logits

    # --- eviction ------------------------------------------------------------

    def evict_sequence(self, seq_id: Optional[int] = None) -> Optional[SequenceRequest]:
        """Evict a running sequence (named, or the policy's victim), checkpoint it,
        and free its blocks. Returns the evicted request, or None if nothing was
        evictable (e.g. only one running sequence)."""
        victim = self._select_eviction_target() if seq_id is None else self.running.get(seq_id)
        if victim is None:
            return None
        try:
            self.cache.evict_sequence(seq_id=victim.seq_id)
        except EvictionFailed:
            return None
        self.running.pop(victim.seq_id, None)
        self.evicted[victim.seq_id] = self._checkpoint_seq(victim)
        return victim

    def restore_sequence(self, seq_id: int) -> Optional[SequenceRequest]:
        """Re-queue a previously-evicted sequence for prefill from its checkpoint.

        The checkpoint holds the prompt + tokens decoded so far; on resume the
        sequence re-prefills the concatenation (CHECKPOINTED -> WAITING)."""
        ckpt = self.evicted.pop(seq_id, None)
        if ckpt is None:
            return None
        full = list(ckpt["prompt_tokens"]) + list(ckpt["output_tokens"])
        req = SequenceRequest(
            seq_id=seq_id, input_ids=full, prompt_len=len(full),
            output_tokens=list(ckpt["output_tokens"]),
        )
        self.waiting_queue.append(req)
        return req

    def _select_eviction_target(self) -> Optional[SequenceRequest]:
        """Pick which running sequence to evict per `eviction_policy`.

        Prefer decode-phase sequences (a just-loaded prefill has paid its cost and
        would only be reloaded); fall back to the whole running set if every
        sequence is mid-prefill."""
        if not self.running:
            return None
        candidates = [r for r in self.running.values() if not r.is_prefill]
        if not candidates:
            candidates = list(self.running.values())
        if self.evict_policy == "priority_score":
            return min(candidates, key=lambda r: r.priority)
        if self.evict_policy == "random":
            import random
            return random.choice(candidates)
        # Default "lru": oldest arrival among the candidates.
        return min(candidates, key=lambda r: r.arrival_time)

    # --- helpers -------------------------------------------------------------

    def _checkpoint_seq(self, req: SequenceRequest) -> dict:
        """Snapshot a sequence's state for later restore (in-memory; disk later)."""
        prompt_len = req.prompt_len or len(req.input_ids)
        return {
            "seq_id": req.seq_id,
            "prompt_tokens": list(req.input_ids[:prompt_len]) if req.is_prefill
            else list(req.input_ids[:0]),
            "output_tokens": list(req.output_tokens),
            "prompt_len": prompt_len,
            "num_decoded": len(req.output_tokens),
        }

    def _compute_utilization(self) -> float:
        """Fraction of the pool's blocks currently owned by running sequences."""
        total = self.cache._num_blocks
        if total <= 0:
            return 0.0
        used = 0
        for r in self.running.values():
            tables = self.cache._seq_block_tables.get(r.seq_id, {})
            used += sum(len(t) for t in tables.values())
        return used / total

    @property
    def has_work(self) -> bool:
        """True while any sequence is waiting or running (drives the engine loop)."""
        return bool(self.waiting_queue or self.running)

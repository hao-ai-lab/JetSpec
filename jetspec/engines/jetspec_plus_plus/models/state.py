"""Mixer state and sequence ownership, independent of checkpoint identity."""
from collections import deque
import weakref

import torch


_config_states = weakref.WeakValueDictionary()


def bind_state(config, state):
    """Bind scheduler resources locally; Config must remain safe to spawn/pickle."""
    state._config = weakref.ref(config)
    _config_states[id(config)] = state


def state_for(config):
    state = _config_states.get(id(config))
    return state if state is not None and state._config() is config else None


def stage_state_slots(runner, graph_vars, count):
    """Copy live ids into static graph storage, retaining private padded rows."""
    if not has_state(runner) or runner.state.capacity == 1:
        return
    slots = graph_vars['state_slots']
    slots.copy_(runner.state.reserved[:slots.numel()])
    slots[:count].copy_(runner._state_slots[:count])


def has_state(runner):
    spec = getattr(runner, 'model_spec', None)
    return spec is not None and spec.has_state and not getattr(runner, 'is_draft', False)


def pair_enabled(owner):
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    return setting(owner, 'engine.engine_settings.hybrid_pair', '0') == '1'


def pair_layout(owner):
    """Read the new name while draft-side compatibility remains owned by F8."""
    return getattr(owner, '_pair_layout', None) or getattr(owner, '_hybrid_pair_layout', None)


class SlotAllocator:
    """Host ownership of the finite buffers belonging to a stateful mixer."""
    def __init__(self, capacity=1, *, device=None):
        self.device = device
        self.available = deque(range(capacity))

    def alloc(self, n):
        if n < 0 or n > len(self.available):
            raise ValueError('insufficient free mixer state slots')
        return torch.tensor([self.available.popleft() for _ in range(n)],
                            dtype=torch.long, device=self.device)

    def free(self, slot_ids):
        self.available.extend(slot_ids.tolist())


class StatelessState:
    """Full attention has no mixer resources to reserve or release."""
    def __init__(self, capacity=1, *, device=None):
        self.ids = torch.arange(capacity, dtype=torch.long, device=device)

    def alloc(self, n):
        if n < 0:
            raise ValueError('state slot count must be nonnegative')
        if n > self.ids.numel():
            self.ids = torch.arange(n, dtype=torch.long, device=self.ids.device)
        return self.ids if n == self.ids.numel() else self.ids[:n]

    def reset(self, slot_ids):
        pass

    def prepare_verify(self, slot_ids, width):
        pass

    def commit(self, slot_ids, accepted_counts):
        pass

    def free(self, slot_ids):
        pass


class ModelState:
    """One slot assignment shared by every mixer in a target or draft model."""
    def __init__(self, model, capacity, *, device=None):
        from jetspec.engines.jetspec_plus_plus.models.gdn import HybridGDN
        self.capacity = capacity
        self.device = device
        self.owners = {}
        self.kinds = [module for module in model.modules() if isinstance(module, HybridGDN)]
        self.stateful = bool(self.kinds)
        self.slots = (SlotAllocator if self.stateful else StatelessState)(capacity, device=device)
        # C=1 reuses its original buffers during startup, before an owner exists.
        # Larger capacities reserve one private row per possible padded/probe row.
        self.reserved = (torch.arange(1 if capacity == 1 else capacity,
                                      2 if capacity == 1 else 2 * capacity,
                                      device=device, dtype=torch.long) if self.stateful
                         else self.slots.alloc(capacity))
        if self.stateful and capacity == 1:
            self.reserved.zero_()
        # A verify holds at most one row per live sequence, so snapshots need C rows.
        # Reserved/probe slots need committed state but no extra snapshot rows.
        for mixer in self.kinds:
            if capacity == 1:
                mixer.configure_slots(1)
            else:
                mixer.configure_slots(2 * capacity, verify_rows=capacity)
            mixer._slots = SlotAllocator(capacity, device=device)
        if not self.kinds:
            self.kinds = [self.slots]
        # verify_cut_commit27b: the pointer table and kernel are built here, before graph capture.
        self._commit_all = None
        if self.stateful and any(getattr(mixer, 'verify_cut_commit27b', False) for mixer in self.kinds):
            from jetspec.engines.jetspec_plus_plus.kernels.verify_commit27b import LayerCommit
            self._commit_all = LayerCommit(self.kinds)

    def alloc(self, n):
        slots = self.slots.alloc(n)
        for mixer in self.kinds:
            if mixer is not self.slots:
                mixer.alloc(n)
        return slots

    def reset(self, slot_ids):
        for mixer in self.kinds:
            mixer.reset(slot_ids)

    def prepare_verify(self, slot_ids, width):
        for mixer in self.kinds:
            mixer.prepare_verify(slot_ids, width)

    def commit(self, slot_ids, accepted_counts):
        commit_all = getattr(self, '_commit_all', None)
        if commit_all is not None and commit_all(accepted_counts):
            return
        for mixer in self.kinds:
            mixer.commit(slot_ids, accepted_counts)

    def free(self, slot_ids):
        for mixer in self.kinds:
            if mixer is not self.slots:
                mixer.free(slot_ids)
        self.slots.free(slot_ids)

    def for_sequences(self, seqs, *, prefill=False):
        if not self.stateful:
            slots = self.alloc(len(seqs))
            if prefill:
                self.reset(slots)
            return slots
        rows = []
        for row, seq in enumerate(seqs):
            # Warm-up has no target KV allocation and must never own a live slot.
            if not seq.block_table:
                if self.capacity == 1 and self.owners:
                    raise RuntimeError('cold state probe overlaps a live sequence')
                rows.append(self.reserved[row % len(self.reserved):row % len(self.reserved) + 1])
                continue
            if seq.seq_id not in self.owners:
                self.owners[seq.seq_id] = self.alloc(1)
            rows.append(self.owners[seq.seq_id])
        slots = (rows[0] if len(rows) == 1 else torch.cat(rows) if rows
                 else torch.empty(0, device=self.device, dtype=torch.long))
        if prefill:
            self.reset(slots)
        return slots

    def release(self, seq):
        if not self.stateful:
            self.free(self.alloc(1))
            return
        slot = self.owners.pop(seq.seq_id, None)
        if slot is not None:
            self.free(slot)

    def graph_slots(self, count):
        return (self.reserved[:count] if self.stateful else self.alloc(count)).clone()

    def free_requests(self, tables, active):
        """F0 table release shares the same state release as scheduler finish."""
        from jetspec.engines.jetspec_plus_plus.jetspec.profile import release_finished_sequences
        keep = set(map(int, active))
        for seq_id in list(self.owners):
            if seq_id >= 0 and seq_id not in keep:
                self.free(self.owners.pop(seq_id))
        return release_finished_sequences(tables, active)

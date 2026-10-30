"""verify_cut_commit27b: the C=1 lazy GDN commit of every layer in one launch.

The per-layer commit runs ``gdn_lazy.commit_state`` and one conv gather for each GDN layer, so a
27B round commits in 96 launches. Here one launch reads a device table of per-layer pointers. Its
state programs run the ``_commit_state`` step body unchanged, and its conv programs copy the
accepted conv window. Every committed bit equals the per-layer commit.
"""
import torch

from jetspec.engines.jetspec_plus_plus.kernels.gdn_lazy import BV

try:
    import triton
    import triton.language as tl
except ImportError:  # The CPU suite runs without Triton.
    triton = None

# Table columns per layer: k, v, g and beta keeps, the committed state, the conv scratch and the
# committed conv.
COLUMNS = 7
CONV_BLOCK = 1024


if triton is not None:
    @triton.jit
    def _step(state, key, value, gate, beta):
        # One step of kernels/gdn_lazy.py _commit_state on one (K, BV) state tile.
        state = state * tl.exp(gate)
        delta = (value - tl.sum(state * key[:, None], 0)) * beta
        return state + key[:, None] * delta[None, :]

    @triton.jit
    def _commit_group(k_ptr, v_ptr, g_ptr, beta_ptr, state_ptr, head, group, count,
                      KEY_HEADS: tl.constexpr, VALUE_HEADS: tl.constexpr, K: tl.constexpr,
                      V: tl.constexpr, BVALUE: tl.constexpr):
        # Four state tiles of the same (K, BV) shape share one key load and norm per
        # step. Each tile keeps its own layout and reduction, so every bit matches one tile alone.
        kk = tl.arange(0, K)
        vv = group * (4 * BVALUE) + tl.arange(0, BVALUE)
        offsets = head * (K * V) + kk[:, None] * V + vv[None, :]
        s0 = tl.load(state_ptr + offsets)
        s1 = tl.load(state_ptr + offsets + BVALUE)
        s2 = tl.load(state_ptr + offsets + 2 * BVALUE)
        s3 = tl.load(state_ptr + offsets + 3 * BVALUE)
        key_head = head // (VALUE_HEADS // KEY_HEADS)
        for step in range(count):
            key = tl.load(k_ptr + (step * KEY_HEADS + key_head) * K + kk)
            key = key * tl.rsqrt(tl.sum(key * key, 0) + 1e-6)
            values = v_ptr + (step * VALUE_HEADS + head) * V + vv
            gate = tl.load(g_ptr + step * VALUE_HEADS + head)
            beta = tl.load(beta_ptr + step * VALUE_HEADS + head)
            s0 = _step(s0, key, tl.load(values), gate, beta)
            s1 = _step(s1, key, tl.load(values + BVALUE), gate, beta)
            s2 = _step(s2, key, tl.load(values + 2 * BVALUE), gate, beta)
            s3 = _step(s3, key, tl.load(values + 3 * BVALUE), gate, beta)
        tl.store(state_ptr + offsets, s0)
        tl.store(state_ptr + offsets + BVALUE, s1)
        tl.store(state_ptr + offsets + 2 * BVALUE, s2)
        tl.store(state_ptr + offsets + 3 * BVALUE, s3)


    @triton.jit
    def _commit_layers(table_ptr, count_ptr,
                       ROWS: tl.constexpr, KEY_HEADS: tl.constexpr, VALUE_HEADS: tl.constexpr,
                       K: tl.constexpr, V: tl.constexpr, BVALUE: tl.constexpr,
                       TILES: tl.constexpr, CONV_N: tl.constexpr, CONV_ROWS: tl.constexpr,
                       CONV_BLOCK: tl.constexpr, CONV_TYPE: tl.constexpr):
        # Grid (value head, value tile group or conv block, layer). The pointers carry the 16-byte
        # alignment hint that a tensor argument gets, so the state body compiles as in gdn_lazy.
        head = tl.program_id(0)
        tile = tl.program_id(1)
        row = table_ptr + tl.program_id(2) * 7
        raw = tl.load(count_ptr)
        if tile < TILES:
            count = tl.minimum(tl.maximum(raw, 0), ROWS)
            if count > 0:  # count 0 keeps the committed state: nothing to read or write
                k_ptr = tl.multiple_of(tl.load(row).to(tl.pointer_type(tl.float32)), 16)
                v_ptr = tl.multiple_of(tl.load(row + 1).to(tl.pointer_type(tl.float32)), 16)
                g_ptr = tl.multiple_of(tl.load(row + 2).to(tl.pointer_type(tl.float32)), 16)
                beta_ptr = tl.multiple_of(tl.load(row + 3).to(tl.pointer_type(tl.float32)), 16)
                state_ptr = tl.multiple_of(tl.load(row + 4).to(tl.pointer_type(tl.float32)), 16)
                _commit_group(k_ptr, v_ptr, g_ptr, beta_ptr, state_ptr, head, tile, count,
                              KEY_HEADS, VALUE_HEADS, K, V, BVALUE)
        else:
            # The accepted conv window, as raw bits: scratch_conv[index, 0] -> committed_conv.
            index = tl.minimum(tl.maximum(raw, 0), CONV_ROWS - 1)
            source = tl.multiple_of(tl.load(row + 5).to(tl.pointer_type(CONV_TYPE)), 16)
            target = tl.multiple_of(tl.load(row + 6).to(tl.pointer_type(CONV_TYPE)), 16)
            offsets = ((tile - TILES) * VALUE_HEADS + head) * CONV_BLOCK + tl.arange(0, CONV_BLOCK)
            mask = offsets < CONV_N
            tl.store(target + offsets, tl.load(source + index * CONV_N + offsets, mask=mask), mask=mask)


def _tensors(mixer):
    """The seven buffers of one layer's commit, in table column order."""
    if getattr(mixer, 'verify_cut_commitdelta27b', False):
        return mixer._delta_keep() + (mixer.lazy_beta, mixer.committed_recurrent,
                                     mixer.scratch_conv, mixer.committed_conv)
    return mixer._lazy_keep() + (mixer.committed_recurrent, mixer.scratch_conv, mixer.committed_conv)


def _geometry(mixer):
    """The layer's commit geometry, or None where the per-layer commit must run."""
    if not (getattr(mixer, 'verify_cut_commit27b', False) and mixer.verify_cut_gdnlazy
            and mixer._lazy_state()):
        return None
    k_keep, v_keep, g_keep, beta_keep, state, scratch_conv, conv = _tensors(mixer)
    rows, key_heads, key_dim = k_keep.shape
    value_heads, value_dim = v_keep.shape[1:]
    # The selected four-tile body needs complete tiles and a power-of-two key reduction.
    # Other geometries keep HybridGDN.commit, including its masked state kernel.
    if key_dim < 1 or key_dim & (key_dim - 1) or value_dim % (4 * BV):
        return None
    if (value_heads % key_heads or state.shape != (1, value_heads, key_dim, value_dim)
            or v_keep.shape[0] != rows or g_keep.shape != (rows, value_heads)
            or beta_keep.shape != g_keep.shape
            or any(t.dtype != torch.float32 for t in (k_keep, v_keep, g_keep, beta_keep, state))):
        return None
    if (conv.shape[0] != 1 or scratch_conv.shape[1:] != conv.shape or scratch_conv.dtype != conv.dtype
            or conv.element_size() not in (2, 4)):
        return None
    tensors = _tensors(mixer)
    if not all(t.is_contiguous() and t.device == state.device and t.data_ptr() % 16 == 0
               for t in tensors):
        return None
    return (rows, key_heads, value_heads, key_dim, value_dim, scratch_conv.shape[0], conv.numel(),
            conv.element_size(), state.device)


class LayerCommit:
    """The commit of every C=1 lazy GDN layer of one model in one launch (verify_cut_commit27b).

    ``models/state.py`` ``ModelState`` builds it at construction, outside graph capture, so the
    pointer table and the compiled kernel exist before ``capture_commit_graphs``. The table is
    rebuilt when a buffer moves; a move under graph capture raises.
    """

    def __init__(self, mixers):
        self.mixers = list(mixers)
        self.key = self.table = self.geometry = None
        key = self._key()
        if key is not None and not torch.cuda.is_current_stream_capturing():
            self._build(key)

    def _key(self):
        if triton is None or not self.mixers:
            return None
        delta = [getattr(m, 'verify_cut_commitdelta27b', False) for m in self.mixers]
        rawkeep = [getattr(m, 'verify_cut_conv27b_rawkeep', False) for m in self.mixers]
        if any(rawkeep) and not all(rawkeep):
            return None
        if any(delta) and not all(delta):
            return None  # Mixed modes use each layer's own commit.
        geometry = _geometry(self.mixers[0])
        if geometry is None or any(_geometry(m) != geometry for m in self.mixers[1:]):
            return None
        return geometry, tuple(t.data_ptr() for m in self.mixers for t in _tensors(m))

    def _build(self, key):
        geometry, pointers = key
        self.table = torch.tensor(pointers, dtype=torch.int64, device=geometry[-1]).reshape(-1, COLUMNS)
        self.geometry, self.key = geometry, key
        # Compile the kernel before a commit graph captures it. Count 0 touches no state, and
        # the conv window goes to a scratch tensor.
        mixer = self.mixers[0]
        out = torch.empty_like(mixer.committed_conv)
        warm = torch.tensor([list(pointers[:COLUMNS - 1]) + [out.data_ptr()]], dtype=torch.int64,
                            device=geometry[-1])
        self._launch(warm, torch.zeros(1, dtype=torch.int64, device=geometry[-1]))

    def _launch(self, table, count):
        if getattr(self.mixers[0], 'verify_cut_commitdelta27b', False):
            from jetspec.engines.jetspec_plus_plus.kernels.verify_commitdelta27b import launch_layers
            launch_layers(table, count, self.geometry,
                          **({'rawkeep': True} if getattr(self.mixers[0], 'verify_cut_conv27b_rawkeep', False) else {}))
            return
        rows, key_heads, value_heads, key_dim, value_dim, conv_rows, conv_n, conv_size, _ = self.geometry
        tiles = value_dim // (4 * BV)
        conv_tiles = triton.cdiv(conv_n, value_heads * CONV_BLOCK)
        _commit_layers[(value_heads, tiles + conv_tiles, table.shape[0])](
            table, count, rows, key_heads, value_heads, key_dim, value_dim,
            BV, tiles, conv_n, conv_rows, CONV_BLOCK,
            tl.int16 if conv_size == 2 else tl.int32,
        )

    def __call__(self, accepted_counts):
        """Commit every layer at the accepted count and return True; False runs the per-layer commit."""
        key = self._key()
        if key is None:
            return False
        if key != self.key:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('verify_cut_commit27b: a GDN buffer moved after the commit table '
                                   'was built; rebuild it before graph capture')
            self._build(key)
        count = accepted_counts.to(device=self.table.device, dtype=torch.long).reshape(-1)
        if count.numel() != 1:
            return False
        self._launch(self.table, count)
        return True

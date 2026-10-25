"""Resolve the target verification backend for JetSpec."""

from dataclasses import dataclass
from types import MappingProxyType

from jetspec.engines.jetspec_plus_plus.draft.method.model_spec import AttentionSpec

VERIFY_BACKENDS = frozenset({"flashinfer_paged", "sdpa"})


@dataclass(frozen=True)
class AttentionBackendCapability:
    head_dims: tuple[int, ...] | None
    head_shapes: tuple[tuple[int, int, int], ...] = ()
    query_lengths: tuple[int, ...] = ()
    parity: object = None
    qualification: str | None = None

    def supports(self, attention: AttentionSpec, *, query_length=None):
        shape = (attention.q_heads, attention.kv_heads, attention.head_dim)
        return (attention.q_heads > 0 and attention.kv_heads > 0
                and attention.q_heads % attention.kv_heads == 0
                and attention.head_dim > 0
                and (self.head_dims is None or attention.head_dim in self.head_dims)
                and (not self.head_shapes or shape in self.head_shapes)
                and (query_length is None or not self.query_lengths
                     or query_length in self.query_lengths))

    def parity_status(self, attention, has_state, architecture):
        if self.parity is None:
            return 'reference'
        return self.parity.get(architecture, {}).get(
            (attention.head_dim, has_state), 'unmeasured')


# Parity is independent of shape support. D256 with recurrent state amplifies
# cross-kernel rounding differences on sm80. Opt-ins name qualified recipe routes;
# they never widen a kernel's shape contract or claim unmeasured parity.
ATTENTION_BACKENDS = MappingProxyType({
    'sdpa': AttentionBackendCapability(None),
    'flashinfer': AttentionBackendCapability((64, 128, 256), parity=MappingProxyType({
        'sm80': MappingProxyType({(128, False): 'qualified', (256, True): 'failed'}),
        'sm100': MappingProxyType({(128, False): 'qualified', (256, True): 'unmeasured'}),
    }), qualification='hybrid_verify_flashinfer'),
    'triton': AttentionBackendCapability(
        head_dims=(256,),
        head_shapes=((
            24,  # local query heads
            4,   # local KV heads
            256, # head width
        ),),
        query_lengths=(1, 8, 16, 17, 32, 33),
        parity=MappingProxyType({
            'sm80': MappingProxyType({(256, True): 'recipe_opt_in'}),
            'sm100': MappingProxyType({(256, True): 'unmeasured'}),
        }), qualification='hybrid_verify_triton'),
})


def select_attention_backend(attention: AttentionSpec, has_state: bool, architecture: str,
                             *, requested='auto', qualifications=(), query_length=None,
                             reference=False):
    """Choose a supported, qualified backend without consulting model identity.

    Architecture names are CUDA SM names (sm80, sm100). Unknown architectures
    safely use the reference backend unless an explicit recipe opt-in is supplied.
    """
    names = ('triton', 'flashinfer', 'sdpa') if requested == 'auto' else (requested,)
    for name in names:
        if name not in ATTENTION_BACKENDS:
            raise ValueError(f'unknown attention backend {name!r}')
        capability = ATTENTION_BACKENDS[name]
        if not capability.supports(attention, query_length=query_length):
            continue
        status = capability.parity_status(attention, has_state, architecture)
        opted_in = (has_state and attention.head_dim == 256
                    and capability.qualification in qualifications)
        reference_exception = reference and name == 'flashinfer'
        if status in ('qualified', 'reference') or opted_in or reference_exception:
            return name
    raise ValueError(f'unsupported or unqualified attention backend {requested!r} '
                     f'for {attention} on {architecture}, has_state={has_state}')


def require_hybrid_sdpa_verify(verify_backend: str, attention_backend: str, *, run_config=None,
                               attention=None, has_state=False, architecture=None) -> None:
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    if attention is not None:
        qualifications = tuple(name for name in ('hybrid_verify_triton', 'hybrid_verify_flashinfer')
                               if setting(run_config, 'engine.engine_settings.' + name, '0') == '1')
        try:
            select_attention_backend(attention, has_state, architecture, requested=attention_backend,
                                     qualifications=qualifications,
                                     reference=current_engine().kind == 'reference')
        except ValueError as error:
            raise RuntimeError(str(error)) from error
    if (setting(run_config, 'engine.engine_settings.hybrid_verify_flashinfer', '0') == "1"
            and setting(run_config, 'engine.engine_settings.hybrid_pair', '0') == "1"
            and setting(run_config, 'engine.engine_settings.jetspec_verify_device', '0') == "1"
            and current_engine().kind != 'reference'
            and verify_backend == "flashinfer_paged" and attention_backend == "flashinfer"):
        return
    if (current_engine().kind == 'reference'
            and verify_backend == "flashinfer_paged" and attention_backend == "flashinfer"):
        return
    if (verify_backend == "sdpa" and attention_backend == "triton"
            and setting(run_config, 'engine.engine_settings.hybrid_verify_triton', '0') == "1"):
        return
    if verify_backend != "sdpa" or attention_backend != "sdpa":
        raise RuntimeError(
            "Qwen3.5 hybrid verify requires both --jetspec-verify-backend sdpa "
            "and SSD_ATTENTION_BACKEND=sdpa. FlashInfer and Triton paged verify at "
            "head_dim=256 failed the A100 cross-kernel parity gate; their "
            "attention results diverge from SDPA and amplify "
            "through later GDN layers."
        )


def resolve_jetspec_verify_backend(
    requested: str,
    *,
    jetspec_target: bool,
    flashinfer_dtype_supported: bool,
) -> str | None:
    if requested not in VERIFY_BACKENDS:
        raise ValueError(
            f"Unsupported JetSpec verify backend {requested!r}; "
            "expected 'flashinfer_paged' or 'sdpa'"
        )
    if not jetspec_target:
        return None
    if requested == "flashinfer_paged" and flashinfer_dtype_supported:
        return "flashinfer_paged"
    return "sdpa"

"""Capacity buckets for fixed drafter context caches."""

CACHE_BUCKETS = (768, 1536, 2560, 4096)

# The mirror is consumed after publication; pre-stage can be two rounds ahead.
CHAIN_MAX_AGE = 2


def chain_lanes(environment) -> int:
    """Chain lanes per round: the served block width (JETSPEC_DRAFTS + 1)."""
    from jetspec.engines.jetspec_plus_plus.draft.block_geometry import block_width
    lanes = block_width(environment)
    if lanes is None:
        raise ValueError("chain capacity requires JETSPEC_DRAFTS (the served block width)")
    return lanes


def chain_key_limit(max_model_len: int, lanes: int) -> int:
    return ((max_model_len + (CHAIN_MAX_AGE + 1) * lanes + 127) // 128) * 128


# The cache capacities that the launcher sizes from max_model_len at launch, in that order.
from jetspec.engines.jetspec_plus_plus.legacy_env import environment_name

LAUNCH_DERIVED = (
    environment_name('engine', 'engine_settings.w259_c1_cache_capacity'),
    environment_name('engine', 'engine_settings.decoder_cache_capacity'),
)


def apply_chain_capacity(environment, max_model_len: int, lanes=None) -> int:
    """Derive backing capacities, refusing undersized explicit recipe pins.

    ``lanes`` is the block width; when it is None the width comes from ``environment``."""
    if lanes is None:
        lanes = chain_lanes(environment)
    key_limit = chain_key_limit(max_model_len, lanes)
    decoder_limit = ((key_limit + 4 + lanes + 127) // 128) * 128
    required = dict(zip(LAUNCH_DERIVED, (key_limit, decoder_limit)))
    for name, value in required.items():
        if name in environment and int(environment[name]) < value:
            raise ValueError(f"{name}={environment[name]} requires at least {value} "
                             f"for chain max_model_len={max_model_len} lookahead="
                             f"{(CHAIN_MAX_AGE + 1) * lanes}")
    for name, value in required.items():
        environment.setdefault(name, str(value))
    return key_limit


def capacity_for(max_model_len: int) -> int:
    for capacity in CACHE_BUCKETS:
        if max_model_len <= capacity:
            return capacity
    raise ValueError(
        f"max_model_len {max_model_len} exceeds largest drafter cache bucket {CACHE_BUCKETS[-1]}"
    )

"""27B producer PDL controls and the early dependent-launch PTX trigger."""


def cut_mode(run_config):
    """Independent producer triggers (bit 0) and attention W1 output (bit 1)."""
    from jetspec.engines.jetspec_plus_plus.models.config import setting
    mode = setting(run_config, 'engine', 'engine_settings.verify_cut_pdl27b', '0')
    if mode not in ('0', '1', '2', '3'):
        raise ValueError('verify_cut_pdl27b must be 0, 1, 2 or 3')
    if mode != '0' and setting(
            run_config, 'engine', 'engine_settings.verify_cut_w1prod27b', '0') != '1':
        raise ValueError('pdl27b requires w1prod27b')
    return int(mode)


try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def producer_start(anchor):
        # Start the GEMM's weight prologue; this producer keeps ordinary stream ordering.
        tl.inline_asm_elementwise('griddepcontrol.launch_dependents; mov.u32 $0, $1;',
                                  '=r,r', [anchor], dtype=tl.int32, is_pure=False, pack=1)

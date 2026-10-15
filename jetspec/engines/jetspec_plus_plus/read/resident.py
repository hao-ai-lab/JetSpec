"""Request-boundary helpers shared by fresh and resident reads."""
from collections import deque

def reset_request_state(llm):
    """Drop prefix ownership without reallocating any CUDA storage or graph."""
    scheduler = llm.scheduler
    if scheduler.waiting or scheduler.running:
        raise RuntimeError('RESIDENT_ACTIVE_REQUEST')
    managers = [scheduler.block_manager]
    if hasattr(scheduler, 'draft_block_manager'):
        managers.append(scheduler.draft_block_manager)
    if any(manager.used_block_ids or any(b.ref_count for b in manager.blocks)
           for manager in managers):
        raise RuntimeError('RESIDENT_LIVE_KV_BLOCK')
    for manager in managers:
        manager.hash_to_block_id.clear()
        manager.free_block_ids = deque(range(len(manager.blocks)))
        for block in manager.blocks:
            block.hash, block.token_ids = -1, []

def rank_boundary(speculator):
    """Use an EOS-accepted control without restarting transport or counters."""
    import torch
    import torch.distributed as dist
    from jetspec.engines.jetspec_plus_plus.wire.pair_transport import announce_tcp
    store = speculator.jetspec_mailbox_store
    announce_tcp(store, 'pin_served')
    done = torch.empty(1, dtype=torch.int64, device=speculator.device)
    dist.recv(done, src=speculator.draft_runner_rank, group=speculator.jetspec_request_pg)
    if int(done.item()) != 1:
        raise RuntimeError('RESIDENT_RANK_BOUNDARY_ACK_MISMATCH')

def retain_manifest(manifest, *, run_config=None):
    from jetspec.engines.jetspec_plus_plus.jetspec import run_manifest
    run_manifest.reset(run_config=run_config)
    for entry in manifest['files']:
        if entry['role'] != 'prompts':
            run_manifest.record_file(entry['role'], entry['realpath'], run_config=run_config)
    changing = {'args', 'dataset', 'jetspec_dataset', 'prompts_sha256',
                'prompts_json_meta', 'serve_read_output_check', 'heads_dataset',
                'merged_parts', 'serve_read_prebuilt_step_seconds',
                'target_model_path', 'draft_path'}
    for key, value in manifest['values'].items():
        if key not in changing:
            run_manifest.record_value(key, value, run_config=run_config)

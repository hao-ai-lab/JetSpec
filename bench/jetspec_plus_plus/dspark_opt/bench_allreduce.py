"""Two-GPU all-reduce microbenchmark inside CUDA graphs: NCCL vs torch symmetric-memory one-shot.

72 all-reduces per replay (one verify's worth for Qwen3-8B), each on [tokens, 4096] bf16, with a small
compute kernel between them so the graph has the verify's dependency structure.

usage: CUDA_VISIBLE_DEVICES=1,6 torchrun --nproc-per-node 2 bench_allreduce.py
"""
import os
import time

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import flashinfer.comm as fc


def graph_time(fn, replays=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(replays):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / replays * 1e3


def main():
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    group = dist.group.WORLD
    symm_mem.enable_symm_mem_for_group(group.group_name)
    ws = fc.trtllm_create_ipc_workspace_for_all_reduce_fusion(rank, 2, 128, 4096, group=group)
    workspace = ws[1]
    for tokens in (16, 1):
        x = torch.randn(tokens, 4096, device='cuda', dtype=torch.bfloat16)
        buf = symm_mem.empty(tokens, 4096, device='cuda', dtype=torch.bfloat16)
        symm_mem.rendezvous(buf, group)
        out = torch.empty_like(x)

        def nccl():
            y = x
            for _ in range(72):
                y = y * 1.0001
                dist.all_reduce(y, group=group)
            return y

        def oneshot():
            y = x
            for _ in range(72):
                buf.copy_(y * 1.0001)
                torch.ops.symm_mem.one_shot_all_reduce_out(buf, 'sum', group.group_name, out)
                y = out
            return y

        def trt(pattern=fc.AllReduceFusionPattern.kAllReduce, fused=False):
            def run():
                y = x
                for _ in range(72):
                    y = y * 1.0001
                    o = torch.empty_like(y)
                    kw = dict(allreduce_out=o, residual_in=None, residual_out=None, norm_out=None,
                              rms_gamma=None, rms_eps=None)
                    if fused:
                        kw.update(allreduce_out=None, residual_in=x, residual_out=torch.empty_like(y),
                                  norm_out=o, rms_gamma=gamma, rms_eps=1e-6)
                    fc.trtllm_allreduce_fusion(
                        allreduce_in=y, world_size=2, world_rank=rank, token_num=tokens, hidden_dim=4096,
                        workspace_ptrs=workspace, launch_with_pdl=True, trigger_completion_at_end=True,
                        fp32_acc=True, pattern_code=pattern, use_oneshot=True, quant_out=None,
                        scale_out=None, scale_factor=None, layout_code=None, **kw)
                    y = o
                return y
            return run

        gamma = torch.ones(4096, device='cuda', dtype=torch.bfloat16)

        def compute_only():
            y = x
            for _ in range(72):
                y = y * 1.0001
            return y

        # correctness
        ref = x.clone()
        dist.all_reduce(ref)
        buf.copy_(x)
        torch.ops.symm_mem.one_shot_all_reduce_out(buf, 'sum', group.group_name, out)
        err = (out.float() - ref.float()).abs().max().item()
        o = torch.empty_like(x)
        fc.trtllm_allreduce_fusion(allreduce_in=x, world_size=2, world_rank=rank, token_num=tokens, hidden_dim=4096,
            workspace_ptrs=workspace, launch_with_pdl=True, trigger_completion_at_end=True, fp32_acc=True,
            pattern_code=fc.AllReduceFusionPattern.kAllReduce, use_oneshot=True, allreduce_out=o, residual_in=None,
            residual_out=None, norm_out=None, quant_out=None, scale_out=None, rms_gamma=None, rms_eps=None,
            scale_factor=None, layout_code=None)
        torch.cuda.synchronize()
        terr = (o.float() - ref.float()).abs().max().item()
        times = {n: graph_time(f) for n, f in (('compute only', compute_only), ('nccl', nccl), ('symm one-shot', oneshot),
                                               ('trtllm one-shot', trt()),
                                               ('trtllm AR+res+rmsnorm', trt(fc.AllReduceFusionPattern.kARResidualRMSNorm, True)))}
        if rank == 0:
            base = times['compute only']
            print(f'[{tokens:2d} x 4096 bf16] max|err| symm {err:.3g} trtllm {terr:.3g}; per replay of 72: ' +
                  ', '.join(f'{n} {t:.3f} ms' for n, t in times.items()) +
                  f'; per all-reduce: nccl {(times["nccl"] - base) / 72 * 1e3:.1f} us, '
                  f'symm {(times["symm one-shot"] - base) / 72 * 1e3:.1f} us, '
                  f'trtllm {(times["trtllm one-shot"] - base) / 72 * 1e3:.1f} us, '
                  f'trtllm+res+norm {(times["trtllm AR+res+rmsnorm"] - base) / 72 * 1e3:.1f} us', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

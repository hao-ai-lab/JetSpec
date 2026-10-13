"""Default-off adapter for the byte-exact PTD tree package."""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import sys
import types
import torch


def tree_budget():
    value = int(os.environ.get('JETSPEC_TREE_B', '0'))
    if value not in (0, 31, 63, 127):
        raise ValueError('JETSPEC_TREE_B must be 0, 31, 63, or 127')
    return value




def tree_api():
    # Upstream uses absolute jetspec.tree imports. Provide only that package;
    # importing jetspec's engine would pull unrelated runtime dependencies.
    path = Path(__file__).parent/'vendor/jetspec_tree'
    if 'jetspec.tree' in sys.modules:
        module = sys.modules['jetspec.tree']
        if Path(module.__file__).resolve() != (path/'__init__.py').resolve():
            raise RuntimeError('jetspec.tree is already loaded from a different source')
        return module
    if 'jetspec' not in sys.modules:
        package = types.ModuleType('jetspec')
        package.__path__ = []
        sys.modules['jetspec'] = package
    spec = importlib.util.spec_from_file_location('jetspec.tree', path/'__init__.py',
                                                submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules['jetspec.tree'] = module
    sys.modules['jetspec'].tree = module
    spec.loader.exec_module(module)
    return module




def attention_mask(tree, prefix_length):
    prefix = torch.ones(tree.num_nodes, prefix_length, dtype=torch.bool,
                        device=tree.token_ids.device)
    ghosts = tree.parent_indices < 0
    ghosts[0] = False
    prefix[ghosts] = False
    return torch.cat([prefix, tree_api().build_ancestor_matrix(tree)], dim=1)


def build_tree(anchor, topk_ids, topk_logps, budget, *, drafts=None):
    name = os.environ.get('JETSPEC_TREE_ALGO', 'accum_logp')
    if name not in ('accum_logp', 'top2gap_fanout'):
        raise ValueError('unsupported JETSPEC_TREE_ALGO')
    if topk_ids.shape != (14, 7) or topk_logps.shape != (14, 7):
        raise ValueError('tree inputs must be final-pass [14,7] top-k records')
    knobs = dict(beta=2.0, g_0=1.0) if name == 'top2gap_fanout' else {}
    built = tree_api().build_from_topk(name, int(anchor), topk_ids.cpu(),
        topk_logps.float().cpu(), budget, torch.device('cpu'),
        algo_kwargs=knobs, tree_width=7)

    if os.environ.get('JETSPEC_TREE_SPINE', '0') == '1':
        if name != 'accum_logp':
            raise ValueError('spine construction requires accum_logp')
        # topk may choose different tied maxima from the drafter's argmax,
        # including omitting that token entirely when more than seven tie.
        spine = topk_ids[:, 0].cpu() if drafts is None else torch.as_tensor(drafts, dtype=torch.long, device='cpu')
        if spine.shape != (14,):
            raise ValueError('refined spine must contain 14 drafts')
        return union_spine(built, spine, topk_logps[:, 0].float().cpu(), budget)
    return built




def spine_indices(tree, drafts):
    edges = {(int(p), int(t)): i for i, (p, t) in
             enumerate(zip(tree.parent_indices.tolist(), tree.token_ids.tolist())) if i}
    path = [0]
    for token in drafts:
        child = edges.get((path[-1], int(token)))
        if child is None:
            break
        path.append(child)
    return path


def chain_accept_depth(tree, logits, drafts):
    path = spine_indices(tree, drafts)
    if len(path) != len(drafts) + 1:
        return None
    predictions = logits[path[:-1]].argmax(-1).tolist()
    depth = 0
    for predicted, token in zip(predictions, drafts):
        if predicted != int(token):
            break
        depth += 1
    return depth




def commit_tree_ids(state, accepted_ids, correction):
    """The accepted tree can diverge from the cached linear proposal."""
    if accepted_ids.ndim != 1 or accepted_ids.numel() > 14 or bool((accepted_ids < 0).any()):
        raise ValueError('invalid accepted tree path')
    suffix = torch.cat([accepted_ids.to(state.committed_ids),
                        state.committed_ids.new_tensor([correction])]).reshape(1,-1)
    state.committed_ids = torch.cat([state.committed_ids,suffix],dim=1)
    state.prepared_context = None




def run_tree_forward(runner, seqs, trees):
    """One ragged forward into reserved pages; prefix pages remain read-only."""
    from jetspec.engines.jetspec_plus_plus.serving.utils.context import set_context; from jetspec.engines.jetspec_plus_plus.serving.utils.context import reset_context
    from time import perf_counter
    if runner.is_draft or runner.num_tp_gpus != 1:
        raise ValueError('tree serving currently requires one target GPU')
    started = perf_counter()
    original_trees = trees
    graphed = bool(getattr(runner,'_tree_graphs',None))
    # The real budget includes the root. A ghost query retains the required
    # B+1 plan/graph width and never participates in acceptance.
    trees = [pad_tree(t,tree_budget()+1) for t in trees]
    block_size = runner.block_size
    tables, slots, masks, positions, lengths, offsets = [], [], [], [], [], [0]
    for row,(seq,tree) in enumerate(zip(seqs,trees)):
        past = seq.num_cached_tokens
        first = runner._tree_scratch_start + row*runner._tree_pages_per_seq
        pages = list(range(first, first+runner._tree_pages_per_seq))
        table, node_slots = scratch_layout(seq.block_table,past,tree.num_nodes,block_size,pages)
        remainder = past%block_size
        if remainder:
            runner.kv_cache[:,:,pages[0],:remainder].copy_(
                runner.kv_cache[:,:,seq.block_table[past//block_size],:remainder])
        tables.append(table); slots.extend(node_slots)
        masks.append(attention_mask(tree,past).to(runner.device))
        positions.extend((tree.depth+past).tolist())
        lengths.append(past+tree.num_nodes)
        offsets.append(offsets[-1]+tree.num_nodes)
    device = runner.device
    block_tables = torch.zeros(len(seqs),max(runner.max_num_blocks,max(map(len,tables))),dtype=torch.int32,device=device)
    for row,table in enumerate(tables):
        block_tables[row,:len(table)] = torch.tensor(table,dtype=torch.int32,device=device)
    context_lens = torch.tensor(lengths,dtype=torch.int32,device=device)
    qo = torch.tensor(offsets,dtype=torch.int32,device=device)
    slot_tensor = torch.tensor(slots,dtype=torch.int32,device=device)
    input_ids = torch.cat([t.token_ids for t in trees]).to(device)
    pos = torch.tensor(positions,dtype=torch.int64,device=device)
    set_context(is_prefill=False,cu_seqlens_q=qo,cu_seqlens_k=None,
                max_seqlen_q=max(t.num_nodes for t in trees),max_seqlen_k=0,
                slot_mapping=slot_tensor,context_lens=context_lens,block_tables=block_tables)
    modules = [m for m in runner.model.modules() if hasattr(m,'_sdpa_paged')]
    saved = []
    wrapper_key = (len(seqs),len(input_ids))
    previous_wrapper = None
    use_fi = runner.jetspec_paged_flashinfer or os.environ.get('SSD_ATTENTION_BACKEND') == 'flashinfer'
    if use_fi and not runner.enforce_eager and not graphed:
        reset_context()
        raise RuntimeError('G tree forward lacks captured tree graphs')
    try:
        if use_fi and not graphed:
            wrapper = runner.plan_flashinfer_paged_attention(context_lens,block_tables,
                qo_indptr=qo,custom_mask=torch.cat([m.flatten() for m in masks]))
            previous_wrapper = runner.flashinfer_graph_wrappers.get(wrapper_key)
            runner.flashinfer_graph_wrappers[wrapper_key] = wrapper
            if not hasattr(runner, '_w264_eager_tree_plan_keys'):
                runner._w264_eager_tree_plan_keys = set()
            runner._w264_eager_tree_plan_keys.add(wrapper_key)
        elif not use_fi:
            # FP32 control: the established backend is SDPA (FlashInfer has no
            # FP32 path). Supply its public mask argument for each ragged tree.
            for module in modules:
                saved.append((module,module._sdpa_paged))
                module._sdpa_paged=types.MethodType(sdpa_tree,module)
        start_event,end_event=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start_event.record()
        result=(replay_tree_graph(runner,input_ids,pos,slot_tensor,context_lens,block_tables,qo,
                torch.cat([m.flatten() for m in masks])) if graphed else runner.model(input_ids,pos))
        if runner.config.jetspec_taps:
            hidden,taps=result
        else:
            hidden,taps=result,None
        logits=target_logits_flat(runner.model.compute_logits(hidden,last_only=False))
        end_event.record(); end_event.synchronize()
        if os.environ.get('W264_INDEPENDENT_RECORDS', '0') == '1' and use_fi:
            # Read the packed mask/page buffers used by the kernel, not a
            # second construction of the intended inputs. N=1 certification.
            active = runner.flashinfer_graph_wrappers[wrapper_key] if graphed else wrapper
            mask_end = int(active._mask_indptr_buf[len(seqs)].item())
            page_end = int(active._paged_kv_indptr_buf[len(seqs)].item())
            actual_pos = runner._tree_graph_vars['positions'][:len(input_ids)] if graphed else pos
            actual_slots = runner._tree_graph_vars['slots'][:len(input_ids)] if graphed else slot_tensor
            runner._w264_tree_attention_inputs = dict(
                packed_mask=active._custom_mask_buf[:mask_end].cpu().tolist(),
                mask_indptr=active._mask_indptr_buf[:len(seqs)+1].cpu().tolist(),
                kv_indices=active._paged_kv_indices_buf[:page_end].cpu().tolist(),
                kv_indptr=active._paged_kv_indptr_buf[:len(seqs)+1].cpu().tolist(),
                last_page_lengths=active._paged_kv_last_page_len_buf[:len(seqs)].cpu().tolist(),
                positions=actual_pos.cpu().tolist(), slots=actual_slots.cpu().tolist(),
                block_size=block_size, prefix_block_table=list(seqs[0].block_table),
                scratch_pages=list(range(runner._tree_scratch_start,
                    runner._tree_scratch_start+runner._tree_pages_per_seq)))
        runner._tree_forward_ms=start_event.elapsed_time(end_event)
        runner._tree_slots=slot_tensor.long()
        runner._tree_offsets=offsets
        runner._tree_forward_wall_ms=(perf_counter()-started)*1000
        if any(t.num_nodes != p.num_nodes for t,p in zip(original_trees,trees)):
            real=torch.tensor([offsets[row]+i for row,t in enumerate(original_trees)
                               for i in range(t.num_nodes)],device=device)
            logits=logits[real]
            taps=None if taps is None else taps[real]
        return logits,taps
    finally:
        for module,method in saved:
            module._sdpa_paged=method
        if use_fi and not graphed:
            if previous_wrapper is None:
                runner.flashinfer_graph_wrappers.pop(wrapper_key,None)
            else:
                runner.flashinfer_graph_wrappers[wrapper_key]=previous_wrapper
        reset_context()


def compact_tree_kv(runner,seqs,paths):
    sources,destinations=[],[]
    for row,(seq,path) in enumerate(zip(seqs,paths)):
        sources.extend(runner._tree_offsets[row]+i for i in path)
        for i in range(len(path)):
            position=seq.num_cached_tokens+i
            destinations.append(seq.block_table[position//runner.block_size]*runner.block_size+position%runner.block_size)
    source=runner._tree_slots[torch.tensor(sources,device=runner.device)]
    dest=torch.tensor(destinations,device=runner.device)
    runner.kv_cache[:,:,dest//runner.block_size,dest%runner.block_size]=runner.kv_cache[
        :,:,source//runner.block_size,source%runner.block_size].clone()






@torch.inference_mode()
def capture_tree_graphs(runner):
    """Capture every decode batch size before timing, sharing replay buffers."""
    from jetspec.engines.jetspec_plus_plus._optional import flashinfer
    from jetspec.engines.jetspec_plus_plus.serving.utils.context import set_context; from jetspec.engines.jetspec_plus_plus.serving.utils.context import reset_context
    width=tree_budget()+1
    maximum=runner.config.max_num_seqs
    rows=maximum*width
    columns=runner.max_num_blocks+runner._tree_pages_per_seq
    device=runner.device
    gv=dict(ids=torch.zeros(rows,dtype=torch.long,device=device),
        positions=torch.zeros(rows,dtype=torch.long,device=device),
        slots=torch.full((rows,),-1,dtype=torch.int32,device=device),
        lengths=torch.full((maximum,),width,dtype=torch.int32,device=device),
        tables=torch.zeros(maximum,columns,dtype=torch.int32,device=device),
        qo=torch.arange(maximum+1,dtype=torch.int32,device=device)*width)
    gv['hidden']=torch.empty(rows,runner.hf_config.hidden_size,dtype=runner.hf_config.torch_dtype,device=device)
    if runner.config.jetspec_taps:
        gv['taps']=torch.empty(rows,5,runner.hf_config.hidden_size,dtype=runner.hf_config.torch_dtype,device=device)
    runner._tree_graph_vars=gv
    runner._tree_graphs={}
    runner._tree_graph_buffers={}
    pool=None
    for bs in range(maximum,0,-1):
        count=bs*width
        buffers=dict(qo_indptr_buf=torch.empty(bs+1,dtype=torch.int32,device=device),
            paged_kv_indptr_buf=torch.empty(bs+1,dtype=torch.int32,device=device),
            paged_kv_indices_buf=torch.empty(bs*columns,dtype=torch.int32,device=device),
            paged_kv_last_page_len_buf=torch.empty(bs,dtype=torch.int32,device=device),
            custom_mask_buf=torch.empty(count*(runner.config.max_model_len+width),dtype=torch.uint8,device=device),
            mask_indptr_buf=torch.empty(bs+1,dtype=torch.int32,device=device))
        runner._tree_graph_buffers[bs]=buffers
        # Explicit FA2 avoids unused FA3 workspace on A100.
        wrapper=flashinfer.BatchPrefillWithPagedKVCacheWrapper(runner.flashinfer_attention_workspace,
            'NHD',backend='fa2',use_cuda_graph=True,**buffers)
        runner.flashinfer_graph_wrappers[bs,count]=wrapper
        mask=torch.eye(width,dtype=torch.bool,device=device).flatten().repeat(bs)
        runner.plan_flashinfer_paged_attention(gv['lengths'][:bs],gv['tables'][:bs],
            qo_indptr=gv['qo'][:bs+1],graph_batch_size=bs,custom_mask=mask)
        set_context(is_prefill=False,slot_mapping=gv['slots'][:count],context_lens=gv['lengths'][:bs],
                    block_tables=gv['tables'][:bs],cu_seqlens_q=gv['qo'][:bs+1],max_seqlen_q=width)
        def forward():
            result=runner.model(gv['ids'][:count],gv['positions'][:count])
            if runner.config.jetspec_taps:
                gv['hidden'][:count].copy_(result[0]);gv['taps'][:count].copy_(result[1])
            else:
                gv['hidden'][:count].copy_(result)
        forward();torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,pool=pool):
            forward()
        if pool is None:
            pool=graph.pool()
        runner._tree_graphs[bs]=graph
        torch.cuda.synchronize();reset_context()
    runner._tree_graph_pool=pool
    print(f'W246_TREE_GRAPHS=READY batch_sizes=1..{maximum} width={width}',flush=True)



# Adaptation: package relocation and deferred optional dependencies.

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value

import io
import pickle
import json
import time
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from transformers import AutoTokenizer, AutoConfig
import os
from jetspec.engines.jetspec_plus_plus._optional import flashinfer
from jetspec.engines.jetspec_plus_plus.serving.config import Config
from jetspec.engines.jetspec_plus_plus.serving.sequence import Sequence
from jetspec.engines.jetspec_plus_plus.models.registry import model_class as target_model_class, model_spec
from jetspec.engines.jetspec_plus_plus.models.target import TargetForCausalLM
from jetspec.engines.jetspec_plus_plus.models.sampler import Sampler
from jetspec.engines.jetspec_plus_plus.models.attention import ATTENTION_BACKEND, bind_attention_backend
from jetspec.engines.jetspec_plus_plus.models.context import set_context
from jetspec.engines.jetspec_plus_plus.models.context import reset_context
from jetspec.engines.jetspec_plus_plus.models.context import get_context
from jetspec.engines.jetspec_plus_plus.models.forward_context import set_verification
from jetspec.engines.jetspec_plus_plus.models.state import ModelState, has_state, bind_state
from jetspec.engines.jetspec_plus_plus.models.target_loader import load_model
from jetspec.engines.jetspec_plus_plus.serving.runner_helpers import prepare_decode_tensors_from_seqs
from jetspec.engines.jetspec_plus_plus.serving.runner_helpers import prepare_block_tables_from_seqs
from jetspec.engines.jetspec_plus_plus.serving.runner_helpers import prepare_prefill_tensors_from_seqs
from jetspec.engines.jetspec_plus_plus.wire.outcome_packet import announce_chain_command
from jetspec.engines.jetspec_plus_plus.serving.jetspec_backend import require_hybrid_sdpa_verify
from jetspec.engines.jetspec_plus_plus.serving.jetspec_backend import resolve_jetspec_verify_backend
from jetspec.engines.jetspec_plus_plus.serving.jetspec_backend import select_attention_backend
from jetspec.engines.jetspec_plus_plus.serving.device_placement import cuda_device_for_rank
from jetspec.engines.jetspec_plus_plus.serving.device_placement import gpu_memory_utilization
from jetspec.engines.jetspec_plus_plus.serving.cpu_affinity import pin_rank_cpu_affinity
from jetspec.engines.jetspec_plus_plus.serving.same_gpu_dist import install_cpu_staging
from jetspec.engines.jetspec_plus_plus.wire.nccl_pack import append_committed_taps
from jetspec.engines.jetspec_plus_plus.serving.graphs import run_verify_cudagraph
from jetspec.engines.jetspec_plus_plus.serving.graphs import run_decode_cudagraph
from jetspec.engines.jetspec_plus_plus.comparisons.fanout_graphs import run_fi_tree_decode_cudagraph
from jetspec.engines.jetspec_plus_plus.comparisons.fanout_graphs import run_glue_decode_cudagraph
from jetspec.engines.jetspec_plus_plus.serving.graphs import capture_cudagraph
from jetspec.engines.jetspec_plus_plus.serving.graphs import capture_verify_cudagraph
from jetspec.engines.jetspec_plus_plus.comparisons.fanout_graphs import capture_fi_tree_decode_cudagraph
from jetspec.engines.jetspec_plus_plus.comparisons.fanout_graphs import capture_glue_decode_cudagraph
from jetspec.engines.jetspec_plus_plus.serving.graphs import get_custom_mask
    

def plan_eager_verify(runner, context, *, get_mask=None):
    """Plan and register the paged wrapper for uniform eager verification rows."""
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    if get_mask is None:
        from jetspec.engines.jetspec_plus_plus.serving.mask_helpers import get_verify_chain_mask
        get_mask = get_verify_chain_mask
    batch_size = len(context.context_lens)
    rows = (context.max_seqlen_q if batch_size == 1 and getattr(context, "max_seqlen_q", None)
            else int(context.cu_seqlens_q[-1]))
    causal_n1 = (batch_size == 1 and setting(runner, 'engine.engine_settings.eager_verify_causal_n1', '0') == "1")
    mask = None if causal_n1 else get_mask(context.context_lens, rows // batch_size)
    wrapper = runner.plan_flashinfer_paged_attention(
        context.context_lens, context.block_tables,
        qo_indptr=context.cu_seqlens_q, **({} if causal_n1 else {"custom_mask": mask}))
    runner.flashinfer_graph_wrappers[(batch_size, rows)] = wrapper


class _TapFreePickler(pickle.Pickler):
    """Pickle Sequences without jetspec_taps (the TP followers never read them)."""

    def reducer_override(self, obj):
        if isinstance(obj, Sequence) and obj.jetspec_taps is not None:
            state = obj.__getstate__()
            state["jetspec_taps"] = None
            return (Sequence.__new__, (Sequence,), state)
        return NotImplemented

def shm_name():
    """The per-run shared-memory segment name. LLMEngine sets JETSPEC_SHM_NAME before it spawns
    ranks, so concurrent runs on one host never race on a fixed name."""
    name = "JETSPEC_SHM_NAME"
    return os.environ.get(name, "ssd")


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event], is_draft: bool = False, num_tp_gpus: int = -1, init_q = None, eagle3_model_class=None, engine=None, run_config=None):
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        self.run_config = (run_config if run_config is not None else
                           getattr(config, 'run_config', None))
        if self.run_config is None:
            raise ValueError('ModelRunner missing RunConfig launch_overlay')
        self.verify_mask_cache = self.run_config.engine.engine_settings.verify_mask_cache == "1"
        from jetspec.engines.jetspec_plus_plus.serving.paths import install_architecture
        install_architecture(self.run_config)
        from jetspec.engines.jetspec_plus_plus.engine import current_engine, install_engine
        self.engine = install_engine(engine if engine is not None else getattr(config, 'engine', None) or current_engine())
        from jetspec.engines.jetspec_plus_plus.draft.serve_policy import install_method_settings
        install_method_settings(getattr(config, 'method_settings', None))
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
        pin_plan_value = setting(self, 'hardware.hardware_settings.cpu_pin_plan')
        pin_plan = json.loads(pin_plan_value) if pin_plan_value is not None else None
        if pin_plan is None or pin_plan["mode"] != "off":
            pin_rank_cpu_affinity(setting(self, 'hardware.hardware_settings.rank_cpu_affinity'), rank, plan=pin_plan)
        if pin_plan is not None:
            import jetspec.engines.jetspec_plus_plus.jetspec.run_manifest as run_manifest
            row = pin_plan["ranks"][rank]
            if row["cpus"] and set(os.sched_getaffinity(0)) != set(row["cpus"]):
                raise RuntimeError(f"CPU_PIN rank={rank} applied affinity differs from requested")
            run_manifest.record_value(f"cpu_pin_rank_{rank}", row, run_config=run_config)
        if config.verbose: print(f'ModelRunner init got args: rank={rank}, is_draft={is_draft}, num_tp_gpus={num_tp_gpus}', flush=True)
        self.config = config
        self.eagle3_model_class = eagle3_model_class
        if self.engine.deterministic:
            from jetspec.engines.jetspec_plus_plus.jetspec.eval_determinism import install
            install(rank, run_config=run_config)
        # serving JETSPEC_C1_COMPILE_PIN: the pinned inductor cache goes in before any compile
        from jetspec.engines.jetspec_plus_plus.serving.compile_pin import process_install
        process_install(run_config=self.run_config)
        config.gpu_memory_utilization = gpu_memory_utilization(config.gpu_memory_utilization, run_config=self.run_config)
        
        assert is_draft in [True, False], "ERROR in ModelRunner: is_draft must be True or False"
        self.is_draft = is_draft
        if self.is_draft: 
            if config.draft_hf_config.torch_dtype != config.hf_config.torch_dtype:
                if self.verbose:
                    print(f"Warning: Draft dtype {config.draft_hf_config.torch_dtype} differs from target {config.hf_config.torch_dtype}. Casting draft to {config.hf_config.torch_dtype}.")
                config.draft_hf_config.torch_dtype = config.hf_config.torch_dtype
            assert (config.draft_hf_config.vocab_size == config.hf_config.vocab_size) or config.use_eagle, "ERROR in ModelRunner: draft_hf_config.vocab_size != hf_config.vocab_size"
        
        self.hf_config = config.hf_config if not is_draft else config.draft_hf_config
        self.model_spec = model_spec(self.hf_config) if not (is_draft and config.use_eagle) else None
        if has_state(self) and num_tp_gpus != 1:
            raise ValueError("stateful target currently requires one GPU")
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.verify_plan_cache = setting(self, 'engine.engine_settings.verify_plan_cache', '0') == "1"
        self._jetspec_verify_device = setting(self, 'engine.engine_settings.jetspec_verify_device', '0') == "1"
        self._ssd_verify_device = setting(self, 'engine.engine_settings.ssd_verify_device', '0') == "1"
        self._verify_stage_host_enabled = setting(self, 'engine.engine_settings.verify_stage_host', '0') == "1"
        self._verify_stage_early_enabled = setting(self, 'engine.engine_settings.verify_stage_early', '0') == "1"
        self._verify_plan_reuse_enabled = setting(self, 'engine.engine_settings.verify_plan_reuse', '0') == "1"
        self._verify_plan_reuse_check = setting(self, 'engine.engine_settings.verify_plan_reuse_check', '0') == "1"
        self._verify_stage_blob_enabled = setting(self, 'engine.engine_settings.verify_stage_blob', '0') == "1"
        self._verify_stage_blob_fast_enabled = setting(self, 'engine.engine_settings.verify_stage_blob_fast', '0') == "1"
        self._verify_stage_blob_fast_check = setting(self, 'engine.engine_settings.verify_stage_blob_fast_check', '0') == "1"
        self._verify_stage_blob_fast_rounds = self._verify_stage_blob_fast_mismatches = 0
        self._verify_plan_lean_enabled = kernel_enabled('verify_plan_lean', '1')
        self._verify_input_direct_enabled = setting(self, 'engine.engine_settings.verify_input_direct', '0') == "1"
        self._verify_reuse_signature = {}
        self._verify_reuse_rounds = self._verify_reuse_hits = self._verify_reuse_replans = 0
        self._verify_reuse_mismatches = 0
        self._verify_plan_cache = {}
        self._verify_plan_cache_hits = 0
        self._verify_plan_cache_misses = 0
        self._staged_verify = None
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path if config.tokenizer_path else config.model, use_fast=True)
        self.max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        assert self.hf_config is not None, "ERROR in ModelRunner: hf_config is None" # this implies boundedness to the end 
        
        # TODO: Get rid of this.
        if self.is_draft:
            should_use_dist = self.config.draft_async
        else:
            should_use_dist = self.config.num_gpus > 1

        # determines whether we create a process group and our process is aware of others, etc
        self.world_size = config.num_gpus if should_use_dist else 1
        self.rank = rank
        self.use_eagle = config.use_eagle

        if config.draft_async:
            self.draft_rank = config.num_gpus - 1
            if self.is_draft:
                # [N, 3] int64, {(seq_id, len(new_suffix), rec_token): (speculated_tokens, logits_q)}
                self.prev_fork_keys: torch.Tensor | None = None
                self.prev_fork_block_tables: torch.Tensor | None = None  # [N, M] int32 with -1 padding

        self.num_tp_gpus = num_tp_gpus # will be in [1 if no dist, num_gpus if not async, num_gpus-1 if async]
        
        if self.world_size == 1: # sync speculation or genuine single gpu 
            assert (config.speculate and not config.draft_async) or self.num_tp_gpus == 1, "ERROR in ModelRunner: draft and async must be False or num_tp_gpus=1"

        self.verbose = config.verbose
        self.draft_async = config.draft_async
        self.event = event
        self._exiting = False 
        
        self.cuda_device_index = cuda_device_for_rank(self.rank, run_config=self.run_config)
        if self.cuda_device_index >= torch.cuda.device_count():
            raise ValueError("SSD_RANK_DEVICE_MAP names an unavailable CUDA device")
        torch.cuda.set_device(self.cuda_device_index)
        self.device = torch.device(f'cuda:{self.cuda_device_index}')
        print(f'SSD_RANK_DEVICE logical_rank={self.rank} cuda_device={self.cuda_device_index}'
              f' visible={os.environ.get("CUDA_VISIBLE_DEVICES")}'
              f' mps={bool(os.environ.get("CUDA_MPS_PIPE_DIRECTORY"))}', flush=True)
        if not is_draft:
            major, minor = torch.cuda.get_device_capability(self.device)
            architecture = f'sm{major}{minor}'
            qualifications = tuple(name for name in ('hybrid_verify_triton', 'hybrid_verify_flashinfer')
                                   if setting(self, 'engine.engine_settings.' + name, '0') == '1')
            bind_attention_backend(select_attention_backend(
                self.model_spec.attention, self.model_spec.has_state, architecture,
                requested=ATTENTION_BACKEND, qualifications=qualifications,
                reference=self.engine.kind == 'reference'))
        jetspec_target = bool(config.jetspec and not is_draft)
        flashinfer_dtype_supported = self.hf_config.torch_dtype in {
            torch.float16,
            torch.bfloat16,
        }
        jetspec_verify_backend = resolve_jetspec_verify_backend(
            config.jetspec_verify_backend,
            jetspec_target=jetspec_target,
            flashinfer_dtype_supported=flashinfer_dtype_supported,
        )
        self.jetspec_paged_flashinfer = (
            jetspec_verify_backend == "flashinfer_paged")
        if has_state(self) and jetspec_target:
            require_hybrid_sdpa_verify(jetspec_verify_backend, ATTENTION_BACKEND, run_config=run_config,
                                      attention=self.model_spec.attention,
                                      has_state=self.model_spec.has_state, architecture=architecture)
        if self._verify_plan_reuse_enabled and jetspec_target and (
                self.enforce_eager or ATTENTION_BACKEND != "flashinfer" or
                flashinfer.__version__ != "0.6.18"):
            raise RuntimeError(
                "JETSPEC_VERIFY_PLAN_REUSE requires FlashInfer 0.6.18 "
                "target CUDA-graph mode")
        if self.jetspec_paged_flashinfer:
            print(
                "W217_JETSPEC_ATTENTION=flashinfer_paged "
                f"dtype={self.hf_config.torch_dtype}",
                flush=True,
            )
        if jetspec_target and config.jetspec_verify_backend == "sdpa":
            backend_label = "triton_paged" if ATTENTION_BACKEND == "triton" else "sdpa_serial"
            print(
                f"W220_JETSPEC_ATTENTION={backend_label} "
                f"dtype={self.hf_config.torch_dtype}",
                flush=True,
            )
        elif jetspec_target and not flashinfer_dtype_supported:
            print(
                "W217_FLASHINFER_FP32_CONTROL=fallback_sdpa "
                f"dtype={self.hf_config.torch_dtype} "
                "reason=flashinfer_paged_dtype_unsupported",
                flush=True,
            )
        if ATTENTION_BACKEND == "flashinfer" or self.jetspec_paged_flashinfer:
            self._init_flashinfer_attention_backend()
        
        # cudagraph logic for FlashInfer kernels, need diff wrapper for each batch size we make a graph for 
        if is_draft and config.draft_async:
            self._init_flashinfer_wrappers()
        
        if self.verbose: print(f'INSIDE MODEL RUNNER INIT, DRAFT={is_draft}', flush=True)
        self.tp_pg = None 

        if should_use_dist:
            default_port = int(os.environ.get("SSD_DIST_PORT", "1223"))
            shared_gpu = setting(self, 'hardware.hardware_settings.shared_gpu_dist') == "gloo"
            kwargs = dict(
                backend="gloo" if shared_gpu else "nccl",
                init_method=f"tcp://localhost:{default_port}",
                world_size=self.world_size,
                rank=self.rank,
            )
            if not shared_gpu:
                kwargs["device_id"] = self.device
            dist.init_process_group(**kwargs)
            if shared_gpu:
                install_cpu_staging(dist)
                print("SSD_SHARED_GPU_DIST backend=gloo staging=cpu", flush=True)

            self.tp_pg = dist.new_group(ranks=list(range(self.num_tp_gpus))) # everyone should see the new_group init even if not in group 

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
        if self.is_draft:
            assert num_tp_gpus == 1, "ERROR in ModelRunner: draft should have tp_size=1"
            self.tp_pg = None # every rank is given an object from self.tp_pg, even tho draft doesnt participate it gets GROUP_NON_MEMBER object != None back, so we can't assert None here, we 
        
        print(f'[model_runner] about to setup and warmup model and cudagraphs, is use_eagle={self.use_eagle}', flush=True)
        model_role = self.setup_and_warmup_model_and_cudagraphs(config, self.hf_config, init_q, is_draft)

        if self.verbose: print(f'-----CAPTURED {model_role}CUDAGRAPH----', flush=True)
        # None, not "cpu": factories already default to the CPU, and a "cpu" mode
        # still routes every later torch call (the verify host path) through Python dispatch.
        torch.set_default_device(None)
        # torch 2.9's set_default_device(None) leaves device_context = None, which a
        # later torch.get_default_device() (transformers from_pretrained, e.g. the
        # DSpark head load) dereferences. Without the attribute it reports the CPU.
        if getattr(getattr(torch, "_GLOBAL_DEVICE_CONTEXT", None), "device_context", False) is None:
            del torch._GLOBAL_DEVICE_CONTEXT.device_context
        torch.set_default_dtype(default_dtype)

        if self.config.draft_async:
            if self.config.fan_out_list is None:
                self.config.fan_out_list = [
                    self.config.async_fan_out] * (self.config.speculate_k + 1)

            self.config.fan_out_t = torch.tensor(self.config.fan_out_list, device=self.device)
            self.config.fan_out_t_miss = torch.tensor(self.config.fan_out_list_miss, device=self.device)
            assert len(self.config.fan_out_list) == self.config.speculate_k + \
                1, "ERROR in Config: fan_out_list must be length speculate_k + 1"
            assert any(f > 0 for f in self.config.fan_out_list), "ERROR in Config: fan_out_list must be > 0"
            self.config.MQ_LEN = sum(self.config.fan_out_list)
            print(f'F={self.config.async_fan_out}, fan_out_list={self.config.fan_out_list}, fan_out_list_miss={self.config.fan_out_list_miss}, MQ_LEN={self.config.MQ_LEN}', flush=True)

        if should_use_dist: # (draft model when async=False or just single gpu logic) doesn't even enter this loop 
            if self.is_draft and self.draft_async: 
                pass # handled on draft runner after this init, includes doing draft_loop
            elif self.rank == 0: # target in a distributed setup 
                # Try to clean up any existing shared memory first
                try:
                    existing_shm = SharedMemory(name=shm_name())
                    existing_shm.close() # here we bind it 
                    existing_shm.unlink()
                except FileNotFoundError:
                    # can proceed, nothing to clean up 
                    pass
                
                self.shm = SharedMemory(name=shm_name(), create=True, size=2**28)
                dist.barrier(group=self.tp_pg, **({} if shared_gpu else {"device_ids": [self.cuda_device_index]})) # leader on tp_group
            else: 
                dist.barrier(group=self.tp_pg, **({} if shared_gpu else {"device_ids": [self.cuda_device_index]})) # follower on tp_group, don't want them hooking onto shm before its been created
                self.shm = SharedMemory(name=shm_name())
                self.loop()
                
        if self.verbose: print(f'-----{model_role}MODEL RUNNER INITIALIZED----', flush=True)

    def _init_flashinfer_attention_backend(self):
        self.flashinfer_attention_workspace = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.flashinfer_ragged_wrapper = (
            flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                self.flashinfer_attention_workspace, "NHD"))
        self.flashinfer_paged_wrapper = (
            flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.flashinfer_attention_workspace, "NHD"))
        self.flashinfer_graph_wrappers = {}
        self._flashinfer_graph_buffers = {}

    def _attach_flashinfer_attention_backend(self):
        for module in self.model.modules():
            if hasattr(module, "flashinfer_ragged_wrapper"):
                module.flashinfer_ragged_wrapper = self.flashinfer_ragged_wrapper
                module.flashinfer_paged_wrapper = self.flashinfer_paged_wrapper
                module.flashinfer_graph_wrappers = self.flashinfer_graph_wrappers
                module.jetspec_paged_flashinfer = self.jetspec_paged_flashinfer

    def get_flashinfer_graph_wrapper(self, batch_size, query_rows, *, causal_only=False):
        # A wrapper built with a mask buffer always runs CUSTOM, so causal
        # wrappers never share a key with the masked verify wrappers.
        key = (batch_size, query_rows, "causal") if causal_only else (batch_size, query_rows)
        wrapper = self.flashinfer_graph_wrappers.get(key)
        if wrapper is not None:
            return wrapper

        max_num_blocks = self.max_num_blocks
        qo_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device=self.device)
        kv_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device=self.device)
        kv_indices = torch.empty(
            batch_size * max_num_blocks, dtype=torch.int32, device=self.device)
        kv_last_page_len = (
            self._jetspec_verify_blob_device["last_page_len"]
            if (getattr(self, "_verify_stage_blob_enabled", False) and
                batch_size == 1 and causal_only and
                hasattr(self, "_jetspec_verify_blob_device"))
            else torch.empty(batch_size, dtype=torch.int32, device=self.device))
        wrapper_kwargs = dict(
            use_cuda_graph=True,
            qo_indptr_buf=qo_indptr,
            paged_kv_indptr_buf=kv_indptr,
            paged_kv_indices_buf=kv_indices,
            paged_kv_last_page_len_buf=kv_last_page_len,
        )
        # Auto selects FA2 on Ampere but reserves unused FA3 workspace per
        # wrapper. Hundreds of large mixed shapes otherwise exhaust memory.
        if (causal_only and batch_size > 16
                and torch.cuda.get_device_capability(self.device)[0] == 8):
            wrapper_kwargs["backend"] = "fa2"
        graph_buffers = [
            qo_indptr, kv_indptr, kv_indices, kv_last_page_len]
        if self.jetspec_paged_flashinfer and query_rows != batch_size and not causal_only:
            custom_mask = torch.empty(
                query_rows * self.config.max_model_len,
                dtype=torch.uint8,
                device=self.device,
            )
            mask_indptr = torch.empty(
                batch_size + 1, dtype=torch.int32, device=self.device)
            wrapper_kwargs.update(
                custom_mask_buf=custom_mask,
                mask_indptr_buf=mask_indptr,
            )
            graph_buffers.extend([custom_mask, mask_indptr])
        self._flashinfer_graph_buffers[key] = tuple(graph_buffers)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.flashinfer_attention_workspace,
            "NHD",
            **wrapper_kwargs,
        )
        self.flashinfer_graph_wrappers[key] = wrapper
        return wrapper

    def _flashinfer_attention_heads(self):
        return (
            self.hf_config.num_attention_heads // self.num_tp_gpus,
            self.hf_config.num_key_value_heads // self.num_tp_gpus,
        )

    def plan_flashinfer_paged_attention(
        self,
        context_lens,
        block_tables,
        qo_indptr=None,
        graph_batch_size=None,
        custom_mask=None,
        static_q_len=None,
        static_kv_pages=None,
        static_kv_indptr=None,
        static_kv_last_page_len=None,
        causal_only=False,
    ):
        if graph_batch_size is not None:
            # Any explicit replan invalidates that wrapper's previous signature.
            self._verify_plan_cache.pop(graph_batch_size, None)
        batch_size = context_lens.shape[0]
        if qo_indptr is None:
            qo_indptr = torch.arange(
                batch_size + 1, dtype=torch.int32, device=self.device)
        if static_kv_pages is not None:
            if batch_size != 1 or not 0 < static_kv_pages <= block_tables.shape[1]:
                raise ValueError("static KV page count requires one valid verify row")
            kv_indptr = (static_kv_indptr if static_kv_indptr is not None else
                         torch.arange(2, dtype=torch.int32,
                                      device=context_lens.device) * static_kv_pages)
            kv_indices = block_tables[0, :static_kv_pages].contiguous()
        else:
            counts = (context_lens + self.block_size - 1) // self.block_size
            kv_indptr = torch.cat([
                torch.zeros(1, dtype=torch.int32, device=self.device),
                counts.cumsum(dim=0).to(torch.int32),
            ])
            block_mask = (
                torch.arange(block_tables.shape[1], device=self.device)[None, :]
                < counts[:, None]
            )
            kv_indices = block_tables[block_mask].to(torch.int32).contiguous()
        if static_kv_last_page_len is not None:
            kv_last_page_len = static_kv_last_page_len
        else:
            remainder = context_lens % self.block_size
            kv_last_page_len = torch.where(
                remainder == 0,
                torch.full_like(remainder, self.block_size),
                remainder,
            ).to(torch.int32)
        wrapper = (
            self.flashinfer_paged_wrapper
            if graph_batch_size is None
            else self.get_flashinfer_graph_wrapper(
                graph_batch_size,
                static_q_len if static_q_len is not None else int(qo_indptr[-1].item()),
                causal_only=causal_only,
            )
        )
        if (graph_batch_size == 1 and causal_only and
                getattr(self, "_verify_plan_reuse_enabled", False)):
            width = static_q_len if static_q_len is not None else int(qo_indptr[-1].item())
            self._verify_reuse_signature.pop(width, None)
        if (custom_mask is None and graph_batch_size is not None
                and getattr(wrapper, "_custom_mask_buf", None) is not None):
            raise RuntimeError("causal plan on a masked FlashInfer graph wrapper would run CUSTOM")
        num_heads, num_kv_heads = self._flashinfer_attention_heads()
        plan_kwargs = dict(
            sm_scale=self.hf_config.head_dim ** -0.5,
            q_data_type=self.hf_config.torch_dtype,
            kv_data_type=self.hf_config.torch_dtype,
        )
        if custom_mask is None:
            plan_kwargs["causal"] = True
        else:
            plan_kwargs["custom_mask"] = custom_mask
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_heads,
            num_kv_heads,
            self.hf_config.head_dim,
            self.block_size,
            **plan_kwargs,
        )
        return wrapper

    def plan_flashinfer_prefill_attention(self):
        context = get_context()
        num_heads, num_kv_heads = self._flashinfer_attention_heads()
        if context.block_tables is None:
            self.flashinfer_ragged_wrapper.plan(
                context.cu_seqlens_q,
                context.cu_seqlens_k,
                num_heads,
                num_kv_heads,
                self.hf_config.head_dim,
                causal=True,
                sm_scale=self.hf_config.head_dim ** -0.5,
                q_data_type=self.hf_config.torch_dtype,
                kv_data_type=self.hf_config.torch_dtype,
            )
            return

        context_lens = context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]
        self.plan_flashinfer_paged_attention(
            context_lens,
            context.block_tables,
            qo_indptr=context.cu_seqlens_q,
        )

    def plan_flashinfer_decode_attention(self):
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        context = get_context()
        if (self.enforce_eager and self.jetspec_paged_flashinfer
                and setting(self, 'engine.engine_settings.eager_verify_wrapper', '0') == "1"
                and context.cu_seqlens_q is not None):
            return plan_eager_verify(self, context)
        qo_indptr = context.cu_seqlens_q
        self.plan_flashinfer_paged_attention(
            context.context_lens,
            context.block_tables,
            qo_indptr=qo_indptr,
        )

    def _init_flashinfer_wrappers(self):
        """Initialize FlashInfer wrappers for draft async mode."""
        self.workspace_buffer = torch.zeros(
            512 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        
        if self.config.enforce_eager: 
            self.only_prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")
        else: 
            max_bs = min(self.config.max_num_seqs, 512)
            max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
            
            # FlashInfer kernel tensors
            # pages_for_max_len = (self.config.max_model_len + self.block_size - 1) // self.block_size
            last_page_len_max_len = self.config.max_model_len % self.block_size
            last_page_len_max_len = self.block_size if last_page_len_max_len == 0 else last_page_len_max_len
            MQ_LEN = self.config.async_fan_out * (self.config.speculate_k + 1)
            
            cu_seqlens_q = torch.empty(max_bs + 1, dtype=torch.int32, device=self.device)
            kv_indptr = torch.empty(max_bs + 1, dtype=torch.int32, device=self.device)
            kv_indices = torch.empty(max_bs * max_num_blocks, dtype=torch.int32, device=self.device)
            kv_last_page_len = torch.empty(max_bs, dtype=torch.int32, device=self.device)
            custom_mask_buf = torch.empty(max_bs * MQ_LEN * self.config.max_model_len, dtype=torch.uint8, device=self.device)
            mask_indptr_buf = torch.empty(max_bs + 1, dtype=torch.int32, device=self.device)
            
            # Create graph_bs_list to match what will be used in cudagraph_helpers.py
            graph_bs_list = [1]
            for bs in [2, 4, 8] + list(range(16, max_bs + 1, 16)):
                if bs <= max_bs:
                    graph_bs_list.append(bs)
            if max_bs not in graph_bs_list:
                graph_bs_list.append(max_bs)
            graph_bs_list.sort()
            
            # Create a dict of wrappers, one for each bs we will touch in cudagraph_helpers.py
            self.prefill_wrappers = {}
            print(f'[model_runner about to wrapper.init()] graph_bs_list={graph_bs_list}', flush=True)
            for bs in graph_bs_list:
                self.prefill_wrappers[bs] = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace_buffer, "NHD", 
                    use_cuda_graph=True, 
                    qo_indptr_buf=cu_seqlens_q[:bs + 1],
                    paged_kv_indptr_buf=kv_indptr[:bs + 1],
                    paged_kv_indices_buf=kv_indices[:bs * max_num_blocks],
                    paged_kv_last_page_len_buf=kv_last_page_len[:bs],
                    custom_mask_buf=custom_mask_buf[:bs * MQ_LEN * self.config.max_model_len],
                    mask_indptr_buf=mask_indptr_buf[:bs + 1],
                )
            print(f'wrapper backend is {self.prefill_wrappers[bs]._backend}', flush=True)


    def setup_and_warmup_model_and_cudagraphs(self, config: Config, hf_config: AutoConfig, init_q=None, is_draft=False):
        # cudagraphs 
        from jetspec.engines.jetspec_plus_plus.profiles import transport_enabled
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        self.graph_vars = {}
        self.graph_pools = {}
        self.graphs = {}
        self.graph_bs_list = {}

        if config.use_eagle and is_draft:
            print(f'[EAGLE3] Loading Eagle3DraftForCausalLM as model_class', flush=True)
            model_class = self.eagle3_model_class
        else:
            model_class = target_model_class(hf_config, draft=is_draft)

        # only give Qwen3 a tp_group if we're a TARGET runner
        kwargs = dict(
            config=self.hf_config,
            draft=self.is_draft,
            speculate=self.config.speculate,
            spec_k=self.config.speculate_k,
            async_fan_out=self.config.async_fan_out,
            draft_async=self.config.draft_async,
            tp_group=self.tp_pg,
            tp_size=self.num_tp_gpus,
        )
        if model_class is TargetForCausalLM:
            kwargs["jetspec_taps"] = self.config.jetspec_taps and not self.is_draft
            kwargs["jetspec_tap_layer_ids"] = self.config.jetspec_target_layer_ids
        
        if config.use_eagle:
            kwargs['use_eagle'] = True
            kwargs['eagle_layers'] = self.config.eagle_layers
            
        if model_class == self.eagle3_model_class:
            kwargs['d_model_target'] = config.d_model_target
            kwargs['debug_mode'] = config.debug_mode
            
        self.model = model_class(**kwargs, **(
            {"run_config": self.run_config} if model_class is TargetForCausalLM else {}))
        self.state = ModelState(self.model, config.max_num_seqs, device=self.device)
        self._state_slots = self.state.graph_slots(1)
        self._state_prefill_lengths = None
        if not self.is_draft:
            bind_state(config, self.state)
        if ATTENTION_BACKEND == "flashinfer" or self.jetspec_paged_flashinfer:
            self._attach_flashinfer_attention_backend()

        model_role = "DRAFT " if self.is_draft else "TARGET "
        if self.verbose:
            print(f'-----LOADING {model_role}MODEL----', flush=True)
        
        # Pass tokenizer_path as target_path if it's available (mostly for Eagle draft)
        target_path = getattr(config, 'tokenizer_path', None)
        target_hidden_size = getattr(config, 'd_model_target', None)
        load_model(self.model, config.model, target_path=target_path, target_hidden_size=target_hidden_size)
        
        if config.draft_async:  # move this here so we don't get a timeout waiting for draft rank while load_model happens?
            ranks = (self.draft_rank,)
            draft_groups = {}
            for draft_rank in ranks:
                async_pg = dist.new_group(ranks=[0, draft_rank])
                if config.jetspec:
                    request_pg = dist.new_group(ranks=[0, draft_rank])
                    candidate_pg = dist.new_group(ranks=[0, draft_rank])
                    store = dist.PrefixStore(
                        "jetspec_candidate_mailbox",
                        dist.distributed_c10d._get_default_store(),
                    )
                    if transport_enabled('reference_candidate_socket', run_config=self):
                        from jetspec.engines.jetspec_plus_plus.wire.candidate_transport import reference_candidate_store
                        store = reference_candidate_store(
                            store, is_draft=self.is_draft, max_num_seqs=config.max_num_seqs, run_config=getattr(self, 'run_config', None))
                else:
                    request_pg = candidate_pg = store = None
                draft_groups[draft_rank] = (
                    async_pg, request_pg, candidate_pg, store)
            selected = self.draft_rank
            (self.async_pg, self.jetspec_request_pg, self.jetspec_candidate_pg,
             self.jetspec_mailbox_store) = draft_groups[selected]
        if self.verbose:
            print(f'-----{model_role}MODEL LOADED----', flush=True)
        if config.sampler_x is not None:
            assert config.draft_async, "ERROR in ModelRunner: sampler_x requires draft_async"
            assert sum(config.fan_out_list) == sum(config.fan_out_list_miss) == config.async_fan_out * (config.speculate_k + 1), "ERROR in ModelRunner: fancy sampling only supported for constant fan out for now."

        self.sampler = Sampler(sampler_x=config.sampler_x, async_fan_out=config.async_fan_out)
        if self.verbose:
            print(f'-----WARMING UP {model_role}MODEL----', flush=True)
        self.warmup_model()
        if self.verbose:
            print(f'-----ALLOCATING {model_role}KV CACHE----', flush=True)
        self.allocate_kv_cache()
        if init_q is not None:
            # super().__init__() runs warmup and calculates num_kvcache_blocks, pass that up
            init_q.put(self.config.num_kvcache_blocks)
            init_q.close()

        if not self.enforce_eager:
            # if not self.is_draft or (self.is_draft and self.config.draft_async and self.config.speculate): 
            decode_graph_vars, decode_graph_pool, decode_graphs, decode_graph_bs_list = capture_cudagraph(self)  # decode cudagraph, draft needs in spec and target in normal
            self.graph_vars["decode"] = decode_graph_vars
            self.graph_pools["decode"] = decode_graph_pool
            self.graphs["decode"] = decode_graphs
            self.graph_bs_list["decode"] = decode_graph_bs_list
            if self.config.speculate and not (self.is_draft and self.config.use_eagle):  # verify CG: target always, non-EAGLE draft for fan-out; EAGLE draft uses glue_decode CG instead
                set_verification(True)
                verify_graph_vars, verify_graph_pool, verify_graphs, verify_graph_bs_list = capture_verify_cudagraph(self)
                set_verification(False)
                self.graph_vars["verify"] = verify_graph_vars
                self.graph_pools["verify"] = verify_graph_pool
                self.graphs["verify"] = verify_graphs
                self.graph_bs_list["verify"] = verify_graph_bs_list
            if self.config.speculate and self.is_draft and self.config.draft_async:
                fi_tree_decode_graph_vars, fi_tree_decode_graph_pool, fi_tree_decode_graphs, fi_tree_decode_graph_bs_list = capture_fi_tree_decode_cudagraph(self)  # fi tree decode cudagraph, draft only
                self.graph_vars["fi_tree_decode"] = fi_tree_decode_graph_vars
                self.graph_pools["fi_tree_decode"] = fi_tree_decode_graph_pool
                self.graphs["fi_tree_decode"] = fi_tree_decode_graphs
                self.graph_bs_list["fi_tree_decode"] = fi_tree_decode_graph_bs_list
            if self.config.speculate and self.is_draft and self.config.draft_async and self.config.use_eagle:
                glue_gv, glue_pool, glue_graphs, glue_bs_list = capture_glue_decode_cudagraph(self)
                self.graph_vars["glue_decode"] = glue_gv
                self.graph_pools["glue_decode"] = glue_pool
                self.graphs["glue_decode"] = glue_graphs
                self.graph_bs_list["glue_decode"] = glue_bs_list

        verify_device = setting(self, 'engine.engine_settings.jetspec_verify_device', '0') == "1"
        ssd_verify_device = setting(self, 'engine.engine_settings.ssd_verify_device', '0') == "1"
        if (not self.enforce_eager and not self.is_draft
                and (self.config.jetspec or (ssd_verify_device and self.config.jetspec_taps))
                and (ATTENTION_BACKEND == "flashinfer" or
                     ((verify_device or ssd_verify_device) and ATTENTION_BACKEND in {"sdpa", "triton"}))
                and (setting(self, 'engine.engine_settings.perseq', '0') == "1" or verify_device
                     or ssd_verify_device)):
            set_verification(True)
            self._capture_target_graphs()
            set_verification(False)

        if has_state(self):
            self.state.reset(self.state.reserved)
            if config.speculate and not self.enforce_eager:
                with torch.inference_mode():
                    from jetspec.engines.jetspec_plus_plus.serving.graphs import capture_commit_graphs
                    self._commit_graphs = capture_commit_graphs(self)
                    self.state.reset(self.state.reserved)

        # serving: every compile of the warm-up and graph captures hit the pinned cache
        from jetspec.engines.jetspec_plus_plus.serving.compile_pin import process_check
        process_check("ready " + ("draft" if self.is_draft else "target"))
        return model_role

    def _capture_target_graphs(self):
        """Target runners capture mixed verify graphs at this startup boundary."""
        pass

    @torch.inference_mode()
    def commit_state(self, accepted: torch.Tensor):
        if not has_state(self):
            return
        accepted = accepted.to(device=self._state_slots.device)
        if self.enforce_eager:
            self.state.commit(self._state_slots, accepted)
        else:
            count = self._state_slots.numel()
            bucket = next(size for size in sorted(self._commit_graphs) if size >= count)
            graph, slots, counts = self._commit_graphs[bucket]
            if self.state.capacity > 1:
                slots.copy_(self.state.reserved[:bucket])
                slots[:count].copy_(self._state_slots)
            if counts.ndim == 0:
                counts.copy_(accepted.reshape(()))
            else:
                counts.zero_()
                counts[:count].copy_(accepted.reshape(-1))
            graph.replay()

    def prepare_state(self, seqs, *, prefill=False):
        # The scheduler already reset live slots. KV-less startup probes have
        # no scheduler owner and reset the reserved slot here instead.
        reset = prefill and any(seq.seq_id not in self.state.owners for seq in seqs)
        self._state_slots = self.state.for_sequences(seqs, prefill=reset)
        self._state_prefill_lengths = (torch.tensor([len(seq) for seq in seqs], device='cpu')
                                       if prefill else None)

    def state_model_forward(self, input_ids, positions, *args, slot_ids=None, verify=None,
                            query_lengths=None):
        from jetspec.engines.jetspec_plus_plus.models.state import has_state
        if not has_state(self):
            return self.model(input_ids, positions, *args)
        from jetspec.engines.jetspec_plus_plus.models.forward_context import forward_context, get_forward_context
        verify = get_forward_context().verify if verify is None else verify
        slots = self._state_slots if slot_ids is None else slot_ids
        if verify:
            self.state.prepare_verify(slots, max(query_lengths) if query_lengths is not None
                                      else input_ids.shape[0] // slots.numel())
        with forward_context(verify=verify, slot_ids=slots,
                             prefill_lengths=(query_lengths if query_lengths is not None
                                              else self._state_prefill_lengths if get_context().is_prefill
                                              else None)):
            return self.model(input_ids, positions, *args)

    def _check_compile_pin_served(self):
        """Check both serving ranks before bench seals the run manifest."""
        from jetspec.engines.jetspec_plus_plus.profiles import transport_enabled
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
        if not setting(self, 'engine.engine_settings.c1_compile_pin', ''):
            return False
        from jetspec.engines.jetspec_plus_plus.serving.compile_pin import process_check
        import jetspec.engines.jetspec_plus_plus.jetspec.run_manifest as run_manifest
        process_check("served")
        if self.config.jetspec and self.draft_async:
            ranks = (self.draft_rank,)
            for rank in ranks:
                group = self.jetspec_request_pg
                if ((setting(self, 'engine.engine_settings.jetspec_verify_device', '0') == "1"
                     or current_engine().kind == 'reference')
                        and setting(self, 'engine.engine_settings.verify_device_transport', 'chain') != "legacy"):
                    if transport_enabled('pair_outcome_push', run_config=self):
                        from jetspec.engines.jetspec_plus_plus.wire.pair_transport import announce_tcp
                        announce_tcp(self.jetspec_mailbox_store, "pin_served")
                    else:
                        announce_chain_command(self.jetspec_mailbox_store, "pin_served", run_config=getattr(self, 'run_config', None))
                else:
                    dist.send(torch.tensor([4], dtype=torch.int64, device=self.device),
                              dst=rank, group=group)
                ack = torch.empty(1, dtype=torch.int64, device=self.device)
                dist.recv(ack, src=rank, group=group)
                if int(ack.item()) != 1:
                    raise RuntimeError(f"C1 COMPILE PIN miss on draft rank {rank} before manifest seal")
        run_manifest.record_value("compile_pin_served", True, run_config=self.run_config)
        return True

    def exit(self, hard: bool = True):
        # Idempotent
        if getattr(self, "_exiting", False):
            return
        self._exiting = True
        if self.rank == 0 and getattr(self, "_verify_prestage_path_counts", None) is not None:
            counts = self._verify_prestage_path_counts
            print("VERIFY_PRESTAGE_PATH_COUNTS "
                  f"staged_mixed={counts['staged_mixed']} "
                  f"staged_full={counts['staged_full']} "
                  f"unstaged={counts['unstaged']}", flush=True)
        # serving: the served rounds' lazy compiles must have hit the pin too (reported here)
        try:
            from jetspec.engines.jetspec_plus_plus.serving.compile_pin import process_check
            process_check("exit " + ("draft" if self.is_draft else "target"))
        except RuntimeError as error:
            print(f"W328_PIN MISS {error}", flush=True)
        if self.verbose:
            print(f"[ModelRunner] Exit start (rank={self.rank}, draft={self.is_draft}, hard={hard})", flush=True)
        # 1) Notify draft first (target rank 0 only)
        try:
            self.send_draft_exit_signal()
        except Exception:
            pass
        # 2) Best-effort local cleanup (no collectives; avoid group destroys in hard mode)
        try:
            if not self.enforce_eager and hasattr(self, "graphs"):
                del self.graphs
                if hasattr(self, "graph_pool"):
                    del self.graph_pool
            if hasattr(self, "verify_graphs"):
                del self.verify_graphs
            if hasattr(self, "verify_graph_pool"):
                del self.verify_graph_pool
            if hasattr(self, "glue_graphs"):
                del self.glue_graphs
            if hasattr(self, "glue_graph_pool"):
                del self.glue_graph_pool
        except Exception:
            pass
        # Close SHM on all ranks that have it
        try:
            if hasattr(self, "shm") and self.shm is not None:
                self.shm.close()
                if self.rank == 0:
                    try:
                        self.shm.unlink()
                    except Exception:
                        pass
        except Exception:
            pass
        # 3) Non-hard path: try to tear down process groups (may block, so do only if not hard)
        if not hard:
            for group_name in ("jetspec_candidate_pg", "jetspec_request_pg"):
                try:
                    if hasattr(self, group_name):
                        dist.destroy_process_group(getattr(self, group_name))
                except Exception:
                    pass
            try:
                if hasattr(self, "async_pg"):
                    dist.destroy_process_group(self.async_pg)
            except Exception:
                pass
            try:
                if self.world_size > 1 and hasattr(self, "tp_pg") and self.tp_pg is not None:
                    dist.destroy_process_group(self.tp_pg)
            except Exception:
                pass
            try:
                # Default group
                if self.world_size > 1 or (self.draft_async and self.is_draft):
                    dist.destroy_process_group()
            except Exception:
                pass
            return
        # 4) Hard path: do not destroy groups (avoid NCCL waits). Draft process can exit immediately.
        if self.is_draft:
            if self.verbose:
                print(f"[ModelRunner] Draft hard-exit", flush=True)
            os._exit(0)
        # Target ranks: let the process return if we're a worker (subprocess),
        # main rank will be force-exited by the engine after it joins children.
        if self.verbose:
            print(f"[ModelRunner] Exit complete (rank={self.rank})", flush=True)
        return

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def recv_cmd(self, group=None):
        t = torch.empty(1, dtype=torch.int64, device=self.device)
        dist.recv(t, src=0, group=self.async_pg if group is None else group)
        return int(t.item())

    def recv_tensor(self, shape, dtype=torch.int64, group=None):
        t = torch.empty(shape, dtype=dtype, device=self.device)
        dist.recv(t, src=0, group=self.async_pg if group is None else group)
        return t
    
    def send_draft_exit_signal(self):
        """
        Best-effort: send cmd=2 to draft (async_pg) from TARGET rank 0.
        Safe to call multiple times.
        """
        from jetspec.engines.jetspec_plus_plus.profiles import transport_enabled
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
        if not (self.draft_async and not self.is_draft and self.rank == 0):
            return
        try:
            if (self.config.jetspec and (setting(self, 'engine.engine_settings.jetspec_verify_device', '0') == "1"
                                        or current_engine().kind == 'reference')
                    and setting(self, 'engine.engine_settings.verify_device_transport', 'chain') != "legacy"):
                if transport_enabled('pair_outcome_push', run_config=self):
                    from jetspec.engines.jetspec_plus_plus.wire.pair_transport import announce_tcp
                    announce_tcp(self.jetspec_mailbox_store, "exit")
                else:
                    announce_chain_command(self.jetspec_mailbox_store, "exit", run_config=getattr(self, 'run_config', None))
                return
            cmd = torch.tensor([2], dtype=torch.int64, device=self.device)
            group = (
                self.jetspec_request_pg if self.config.jetspec else self.async_pg
            )
            ranks = (self.draft_rank,)
            for rank in ranks:
                request_pg = group
                dist.send(cmd, dst=rank, group=request_pg)
        except Exception:
            pass
    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        if self.config.jetspec_taps and not self.config.draft_async:
            # Sync-slot drafter under target TP: only rank 0 reads the accumulated taps, so the
            # followers get the sequences without them (pickling them grows with context).
            buffer = io.BytesIO()
            _TapFreePickler(buffer).dump([method_name, *args])
            data = buffer.getvalue()
        else:
            data = pickle.dumps([method_name, *args])
        n = len(data)
        assert n + 4 <= self.shm.size, f"SHM overflow: {n+4} > {self.shm.size}. Increase SHM buffer size."
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.num_tp_gpus > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        if method is None:
            raise AttributeError(f"Method '{method_name}' not found")
        return method(*args)

    def run_cold_prefill(self, lengths):
        """serve_read's cold phase: one KV-less prefill of zero tokens per prompt, on every TP rank."""
        probe_batch = self.config.max_num_seqs if has_state(self) else max(1, len(lengths))
        for offset in range(0, len(lengths), probe_batch):
            ids, positions = self.prepare_prefill([Sequence([0] * length)
                for length in lengths[offset:offset + probe_batch]])
            get_context().slot_mapping = ids.new_full(ids.shape, -1, dtype=torch.int32)
            try:
                self.run_model(ids, positions, True)
            finally:
                reset_context()

    def gather_full_vocab_weights(self):
        """All-gather the vocab-sharded embedding and LM head over the target TP group.

        A sync-slot drafter on rank 0 (block head / DFlash) reads the target's embedding and LM
        head as plain tensors, so under target TP it needs the full vocab rows; rank 0 keeps them
        as model.jetspec_full_vocab, every other rank only joins the collective."""
        full = []
        for weight in (self.model.model.embed_tokens.weight, self.model.lm_head.weight):
            parts = [torch.empty_like(weight) for _ in range(self.num_tp_gpus)]
            dist.all_gather(parts, weight.data.contiguous(), group=self.tp_pg)
            full.append(torch.cat(parts, dim=0) if self.rank == 0 else None)
            del parts
        if self.rank == 0:
            self.model.jetspec_full_vocab = tuple(full)
        torch.cuda.synchronize(self.device)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        
        hidden_states = None
        if self.config.use_eagle and self.is_draft:
            num_tokens = num_seqs * max_model_len
            d_model_target = self.config.d_model_target or 4096
            hidden_states = torch.zeros(num_tokens, 3 * d_model_target, dtype=self.hf_config.torch_dtype, device=self.device)
        
        self.run(seqs, True, hidden_states=hidden_states)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        print(f'inside allocate_kv_cache -- ', flush=True)
        config = self.config
        hf_config = self.hf_config
        
        # Simplify: just look at free memory on the GPU, and allocate up to gpu_memory_utilization * free.
        free, _ = torch.cuda.mem_get_info()
        num_kv_heads = hf_config.num_key_value_heads // self.num_tp_gpus
        # Only attention layers bind a slice of the pool (the loop below). A linear-attention layer of a hybrid
        # target keeps recurrent state instead, so the pool is sized by the attention layers.
        attention_layers = sum(1 for module in self.model.modules()
                               if hasattr(module, "k_cache") and hasattr(module, "v_cache"))
        assert attention_layers > 0, "ERROR: the model has no attention layer to bind a KV cache to"
        block_bytes = (
            2
            * attention_layers
            * self.block_size
            * num_kv_heads
            * hf_config.head_dim
            * hf_config.torch_dtype.itemsize
        )
        usable_bytes = free * config.gpu_memory_utilization

        if self.is_draft and self.draft_async:
            B = config.max_num_seqs          # max concurrent sequences
            K = config.speculate_k           # speculative steps
            F = config.async_fan_out         # branches per step
            V = hf_config.vocab_size
            dtype_size = hf_config.torch_dtype.itemsize
            # Total forks = B * (K+1) * F, each fork stores (K+1)*V logits of size dtype_size
            reserved_bytes = B * (K + 1) * F * (K + 1) * V * dtype_size
            usable_bytes = max(usable_bytes - reserved_bytes, 0)
            assert usable_bytes > 0, "ERROR: Not enough memory for draft KV cache after accounting for tree_cache for logits storage"

        config.num_kvcache_blocks = int(usable_bytes) // block_bytes
        from jetspec.engines.jetspec_plus_plus.jetspec.profile import kv_cache_blocks
        original_blocks = config.num_kvcache_blocks
        config.num_kvcache_blocks = kv_cache_blocks(config, original_blocks)
        if config.num_kvcache_blocks != original_blocks:
            print('EVAL_KV_RESERVATION ' + json.dumps(dict(
                draft=self.is_draft, original_blocks=original_blocks,
                allocated_blocks=config.num_kvcache_blocks, block_bytes=block_bytes,
                original_bytes=original_blocks * block_bytes,
                allocated_bytes=config.num_kvcache_blocks * block_bytes)), flush=True)
        if self.verbose:
            print(f'KV CACHE ALLOCATION for {"TARGET" if not self.is_draft else "DRAFT"} model', flush=True)
            print(f' free={free/1e9:.2f}GB, util={config.gpu_memory_utilization:.2f}', flush=True)
            print(f' block_bytes={block_bytes/1e9:.2f}GB,'
            + f' num_kvcache_blocks={config.num_kvcache_blocks}', flush=True)   
        
        assert config.num_kvcache_blocks > 0, "KV cache too big for free memory!"

        allocated_blocks = config.num_kvcache_blocks
        self.kv_cache = torch.zeros(
            2,
            attention_layers,
            allocated_blocks,
            self.block_size,
            num_kv_heads,
            hf_config.head_dim, 
        )
        
        print(f"allocate_kv_cache(): kv_cache shape = {self.kv_cache.shape}", flush=True)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                if self.is_draft and self.draft_async and not self.enforce_eager:
                    module.prefill_wrappers = self.prefill_wrappers
                elif self.is_draft and self.draft_async and self.enforce_eager:
                    module.only_prefill_wrapper = self.only_prefill_wrapper # this will make it not None so it can be used on fwd
                layer_id += 1

    
    def prepare_prefill(self, seqs: list[Sequence]):
        from jetspec.engines.jetspec_plus_plus.models.state import has_state
        if has_state(self):
            self.prepare_state(seqs, prefill=True)
            # Cached KV blocks have no GDN state; replay the full target prefix.
            for seq in seqs:
                seq.num_cached_tokens = 0
        input_ids, positions, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping = \
            prepare_prefill_tensors_from_seqs(seqs, self.block_size, self.is_draft) # if one big input ids, how is attn mask handled? via cu_seqlens_q/k? 

        block_tables = None
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = prepare_block_tables_from_seqs(
                seqs, self.is_draft)
        
        set_context(is_prefill=True, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k, slot_mapping=slot_mapping, context_lens=None, block_tables=block_tables)
        return input_ids, positions
    
    def prepare_decode(self, seqs: list[Sequence], verify: bool = False): 
        from jetspec.engines.jetspec_plus_plus.models.state import has_state
        if has_state(self):
            self.prepare_state(seqs)
        if self.verify_mask_cache or self.verify_plan_cache:
            self._verify_context_lens_cpu = tuple(len(seq) for seq in seqs) if verify else None
        if self.verify_plan_cache:
            self._verify_block_tables_cpu = tuple(tuple(seq.block_table) for seq in seqs) if verify else None
        input_ids, positions, slot_mapping, context_lens = \
            prepare_decode_tensors_from_seqs(seqs, self.block_size, self.is_draft, verify, self.config.speculate_k if verify else -1)

        
        block_tables = prepare_block_tables_from_seqs(seqs, self.is_draft) # if verify, set cu_seqlens_q as well

        if verify: ### what path does glue decode take? trace it. 
            # this had [not draft and draft_async] condn before
            cu_seqlens_q = torch.zeros(len(seqs) + 1, dtype=torch.int32, device=self.device)
            seqlen_q = torch.full((len(seqs),), self.config.speculate_k + 1, dtype=torch.int32, device=self.device)
            cu_seqlens_q[1:] = torch.cumsum(seqlen_q, dim=0)
            set_context(is_prefill=False, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=None, 
                       max_seqlen_q=self.config.speculate_k + 1, max_seqlen_k=0,
                       slot_mapping=slot_mapping, context_lens=context_lens, 
                       block_tables=block_tables) 
        else: # sq_decode path, draft (sync spec) or target (normal)
            set_context(is_prefill=False, cu_seqlens_q=None, cu_seqlens_k=None, 
                       max_seqlen_q=0, max_seqlen_k=0, slot_mapping=slot_mapping, 
                       context_lens=context_lens, block_tables=block_tables)
        
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            if self.is_draft and seq.draft_temperature is not None:
                temperatures.append(seq.draft_temperature)
            else:
                temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures



    def eager_tree_decode_plan(self, input_ids, positions, step, cache_hits):
        """Plan FlashInfer for tree decode in eager mode"""
        assert self.is_draft and self.config.draft_async, "ERROR in eager_tree_decode_plan: not a draft async model"
        context = get_context()
        
        K, F = self.config.speculate_k, self.config.async_fan_out
        # MQ_LEN = F * (K+1)
        MQ_LEN = self.config.MQ_LEN
        flat_batch_size = input_ids.size(0) 
        B = flat_batch_size // MQ_LEN # [N] tokens = B * sum(fan_out_list)
        
        # Convert block_tables to FlashInfer format
        block_tables = context.block_tables # [B, M]
        context_lens = context.context_lens # [B]
        
        counts = (context_lens + self.block_size - 1) // self.block_size # [B]
        kv_indptr = torch.cat([torch.tensor([0], device=block_tables.device),
                               counts.cumsum(dim=0)]).to(torch.int32)  
        mask = torch.arange(block_tables.size(1), device=block_tables.device)[None, :] < counts[:, None]
        kv_indices = block_tables[mask]                    # flattened page ids
        
        # Last-page actual token count per request
        kv_last_page_len = (context_lens % self.block_size)
        kv_last_page_len[kv_last_page_len == 0] = self.block_size
        kv_last_page_len = kv_last_page_len.to(torch.int32)
        cu_seqlens_q = torch.arange(B + 1, device=self.device, dtype=torch.int32) * MQ_LEN # assumes same MQ_LEN across batch dimension 
        custom_mask = get_custom_mask(self.config, context_lens, step, K, F, B, device=self.device, cache_hits=cache_hits)
        
        self.only_prefill_wrapper.plan(
            cu_seqlens_q,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            self.hf_config.num_attention_heads,
            self.hf_config.num_key_value_heads,
            self.hf_config.head_dim,
            self.block_size,
            custom_mask=custom_mask,
            q_data_type=self.hf_config.torch_dtype,
            kv_data_type=self.hf_config.torch_dtype,
        )

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, last_only: bool = True, tree_decode_step: int = -1, cache_hits: torch.Tensor | None = None, hidden_states: torch.Tensor | None = None):
        from jetspec.engines.jetspec_plus_plus.models.state import has_state
        if not has_state(self):
            return self._run_model(input_ids, positions, is_prefill, last_only,
                                   tree_decode_step, cache_hits, hidden_states)
        from jetspec.engines.jetspec_plus_plus.models.forward_context import forward_context
        verify = not is_prefill and self.config.speculate and not last_only
        if verify:
            self.state.prepare_verify(self._state_slots, input_ids.shape[0] // self._state_slots.numel())
        with forward_context(verify=verify, slot_ids=self._state_slots,
                             prefill_lengths=self._state_prefill_lengths):
            return self._run_model(input_ids, positions, is_prefill, last_only,
                                   tree_decode_step, cache_hits, hidden_states)

    def _run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, last_only: bool = True, tree_decode_step: int = -1, cache_hits: torch.Tensor | None = None, hidden_states: torch.Tensor | None = None):
        is_tree_decode = self.is_draft and self.config.draft_async and tree_decode_step >= 0
        is_mq_kp1 = self.config.speculate and not last_only

        assert not (is_prefill and not last_only), "ERROR in run_model: is_prefill and not last_only"
        if ATTENTION_BACKEND == "flashinfer":
            if is_prefill:
                self.plan_flashinfer_prefill_attention()
            elif self.enforce_eager and not is_tree_decode:
                self.plan_flashinfer_decode_attention()
        
        if is_prefill or self.enforce_eager:
            if is_tree_decode:
                self.eager_tree_decode_plan(input_ids, positions, tree_decode_step, cache_hits)
            
            if self.config.use_eagle: 
                if self.is_draft:
                    assert hidden_states is not None, "hidden_states required for EAGLE draft"
                    assert isinstance(self.model, self.eagle3_model_class)
                    prenorm = self.model(input_ids, positions, hidden_states)
                    logits = self.model.compute_logits(prenorm, last_only)
                    return logits, prenorm  # return prenorm as conditioning vector for next iteration
                else: # target gets outputs and hidden states
                    outputs, eagle_acts = self.model(input_ids, positions) # target fwd when eagle enabled hooks into activations for eagle conditioning
                    logits = self.model.compute_logits(outputs, last_only)
                    return logits, eagle_acts  # return eagle_acts as conditioning vector for draft
            else:
                model_out = self.model(input_ids, positions)
                if self.config.jetspec_taps and not self.is_draft:
                    outputs, jetspec_taps = model_out
                else:
                    outputs = model_out
                logits = self.model.compute_logits(outputs, last_only)
                if self.config.jetspec_taps and not self.is_draft:
                    return logits, jetspec_taps
                return logits

        elif is_tree_decode:
            return run_fi_tree_decode_cudagraph(self, input_ids, positions, last_only, self.graph_vars["fi_tree_decode"], tree_decode_step, cache_hits, hidden_states=hidden_states)
        elif is_mq_kp1 and hidden_states is not None and "glue_decode" in self.graph_vars:
            # EAGLE draft glue decode with 2K+1 per seq
            return run_glue_decode_cudagraph(self, input_ids, positions, last_only, self.graph_vars["glue_decode"], hidden_states)
        elif is_mq_kp1: # verify or non-EAGLE glue decode, "verify" ~ mq decode of len K+1
            return run_verify_cudagraph(self, input_ids, positions, last_only, self.graph_vars["verify"])
        else: # draft decoding in sync spec or JIT single-token decode
            return run_decode_cudagraph(self, input_ids, positions, last_only, self.graph_vars["decode"], hidden_states=hidden_states)


    def _guard_verify_plan_reuse(self, wrapper):
        if (flashinfer.__version__ != "0.6.18" or
                getattr(wrapper, "_backend", None) != "fa2" or
                not getattr(wrapper, "_use_cuda_graph", False) or
                getattr(wrapper, "_custom_mask_buf", None) is not None or
                getattr(wrapper, "_jit_module", None) is not None):
            raise RuntimeError(
                "JETSPEC_VERIFY_PLAN_REUSE requires FlashInfer 0.6.18 fa2 "
                "CUDA-graph mode without custom mask or JIT")

    def _plan_jetspec_n1(self, width, pages, page_ids, plan_host, gv):
        """Keep a wrapper's real plan until width, pages, or page IDs change."""
        if getattr(self, "_verify_plan_lean_enabled", False):
            wrapper = self._jetspec_n1_plan_wrappers[width]
            kwargs = self._jetspec_n1_plan_kwargs[width]
        else:
            wrapper = self.get_flashinfer_graph_wrapper(1, width, causal_only=True)
            self._guard_verify_plan_reuse(wrapper)
            kwargs = dict(qo_indptr=plan_host["qo_indptr"], graph_batch_size=1,
                          static_q_len=width, static_kv_pages=pages,
                          static_kv_indptr=plan_host["kv_indptr"],
                          static_kv_last_page_len=plan_host["kv_last_page_len"],
                          causal_only=True)
        if getattr(self, "_verify_plan_lean_enabled", False):
            kwargs["static_kv_pages"] = pages
        signature = (width, pages, page_ids)
        self._verify_reuse_rounds += 1
        if self._verify_reuse_signature.get(width) != signature:
            self.plan_flashinfer_paged_attention(
                plan_host["context_lens"], gv["block_tables"][:1], **kwargs)
            self._verify_reuse_signature[width] = signature
            self._verify_reuse_replans += 1
        else:
            self._verify_reuse_hits += 1
            if not self._verify_stage_blob_enabled:
                wrapper._paged_kv_last_page_len_buf[:1].copy_(
                    plan_host["kv_last_page_len"], non_blocking=True)
            if not getattr(self, "_verify_plan_lean_enabled", False):
                wrapper._kv_lens_buffer[:1].copy_(plan_host["context_lens"],
                                                   non_blocking=True)
            if self._verify_plan_reuse_check:
                names = ("_int_workspace_buffer", "_qo_indptr_buf",
                         "_paged_kv_indptr_buf", "_paged_kv_indices_buf",
                         "_paged_kv_last_page_len_buf", "_kv_lens_buffer")
                before = {name: getattr(wrapper, name).clone() for name in names}
                info = list(wrapper._plan_info)
                self.plan_flashinfer_paged_attention(
                    plan_host["context_lens"], gv["block_tables"][:1], **kwargs)
                mismatches = sum(not torch.equal(before[name], getattr(wrapper, name))
                                 for name in names) + int(info != list(wrapper._plan_info))
                self._verify_reuse_mismatches += mismatches
                self._verify_reuse_signature[width] = signature
                if mismatches:
                    raise RuntimeError(f"PLAN_REUSE_CHECK mismatches={mismatches}")
        if self._verify_plan_reuse_check:
            print("PLAN_REUSE_CHECK "
                  f"rounds={self._verify_reuse_rounds} reuse={self._verify_reuse_hits} "
                  f"replans={self._verify_reuse_replans} "
                  f"mismatches={self._verify_reuse_mismatches}", flush=True)




    def _run_jetspec_mixed_graph(self, input_ids, positions, context, widths):
        bs, count = len(widths), len(input_ids)
        full = sum(width > 1 for width in widths)
        gv = self._jetspec_mixed_vars
        gv["input_ids"][:count].copy_(input_ids)
        gv["positions"][:count].copy_(positions)
        gv["slot_mapping"][:count].copy_(context.slot_mapping)
        gv["context_lens"][:bs].copy_(context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1])
        gv["block_tables"][:bs].zero_()
        gv["block_tables"][:bs, :context.block_tables.shape[1]].copy_(context.block_tables)
        cu = gv["cu_seqlens_q"][:bs + 1]
        cu.copy_(context.cu_seqlens_q)
        self.plan_flashinfer_paged_attention(gv["context_lens"][:bs],
            gv["block_tables"][:bs], qo_indptr=cu, graph_batch_size=bs, causal_only=True)
        timing = getattr(self, '_anatomy_lite', None)
        if timing is not None:
            timing.gpu_begin()
        self._jetspec_mixed_graphs[bs, full].replay()
        logits = self.model.compute_logits(gv["outputs"][:count], last_only=False)
        if timing is not None:
            timing.gpu_end()
        return (logits, gv["taps"][:count]) if "taps" in gv else logits

    def _run_jetspec_mixed_staged(self, seqs, widths, staged):
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.verify.verify_plan_helpers import assemble_mixed_verify_rows
        bs = len(seqs)
        full = sum(query_width > 1 for query_width in widths)
        assembled = assemble_mixed_verify_rows(
            staged, seqs, self.config.speculate_k + 1)
        if assembled is None or (bs, full) not in self._jetspec_mixed_graphs:
            return None
        ids, positions, slots, lengths, tables, cu_q, cu_k = assembled
        gv = self._jetspec_mixed_vars
        count = len(ids)
        def copy_host(target, values, dtype):
            target.copy_(torch.tensor(values.tolist(), dtype=dtype, pin_memory=True),
                         non_blocking=True)
        copy_host(gv["input_ids"][:count], ids, torch.int64)
        copy_host(gv["positions"][:count], positions, torch.int64)
        copy_host(gv["slot_mapping"][:count], slots, torch.int32)
        copy_host(gv["context_lens"][:bs], lengths, torch.int32)
        gv["block_tables"][:bs].zero_()
        copy_host(gv["block_tables"][:bs, :tables.shape[1]], tables, torch.int32)
        copy_host(gv["cu_seqlens_q"][:bs + 1], cu_q, torch.int32)
        cu_k_device = torch.tensor(cu_k.tolist(), dtype=torch.int32, pin_memory=True).to(
            self.device, non_blocking=True)
        set_context(is_prefill=True, cu_seqlens_q=gv["cu_seqlens_q"][:bs + 1],
                    cu_seqlens_k=cu_k_device, max_seqlen_q=max(widths),
                    max_seqlen_k=int(lengths.max()),
                    slot_mapping=gv["slot_mapping"][:count], context_lens=None,
                    block_tables=gv["block_tables"][:bs, :tables.shape[1]])
        self.plan_flashinfer_paged_attention(
            gv["context_lens"][:bs], gv["block_tables"][:bs],
            qo_indptr=gv["cu_seqlens_q"][:bs + 1], graph_batch_size=bs,
            causal_only=True)
        timing = getattr(self, '_anatomy_lite', None)
        if timing is not None:
            timing.gpu_begin()
        self._jetspec_mixed_graphs[bs, full].replay()
        logits = self.model.compute_logits(gv["outputs"][:count], last_only=False)
        if timing is not None:
            timing.gpu_end()
        taps = gv["taps"][:count] if "taps" in gv else None
        if setting(self, 'engine.engine_settings.verify_prestage_mixed_check', '0') == "1":
            logits_copy = logits.clone()
            taps_copy = taps.clone() if taps is not None else None
            legacy_ids, legacy_positions = self.prepare_prefill(seqs)
            legacy_context = get_context()
            for name, got, expected in (
                ("ids", gv["input_ids"][:count], legacy_ids),
                ("positions", gv["positions"][:count], legacy_positions),
                ("slot_mapping", gv["slot_mapping"][:count], legacy_context.slot_mapping),
                ("context_lens", gv["context_lens"][:bs],
                 legacy_context.cu_seqlens_k[1:] - legacy_context.cu_seqlens_k[:-1]),
                ("block_tables", gv["block_tables"][:bs, :tables.shape[1]],
                 legacy_context.block_tables),
                ("cu_seqlens_q", gv["cu_seqlens_q"][:bs + 1],
                 legacy_context.cu_seqlens_q),
            ):
                if got.dtype != expected.dtype or got.device != expected.device or not torch.equal(got, expected):
                    raise RuntimeError(f"VERIFY_PRESTAGE_MIXED_CHECK mismatch: {name}")
            reference = self._run_jetspec_mixed_graph(
                legacy_ids, legacy_positions, legacy_context, widths)
            reference_logits, reference_taps = (reference if taps is not None
                                               else (reference, None))
            if not torch.equal(logits_copy, reference_logits):
                raise RuntimeError("VERIFY_PRESTAGE_MIXED_CHECK mismatch: logits")
            if taps_copy is not None and not torch.equal(taps_copy, reference_taps):
                raise RuntimeError("VERIFY_PRESTAGE_MIXED_CHECK mismatch: taps")
            print("VERIFY_PRESTAGE_MIXED_CHECK=PASS", flush=True)
            logits, taps = logits_copy, taps_copy
        return (logits, taps) if taps is not None else logits






    # should add spec_k that just loops this k times
    def run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        last_only: bool = True,
        draft_return_logits: bool = False,
        hidden_states: torch.Tensor | None = None
    ) -> list[int] | tuple[list[int], torch.Tensor]:
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        _pt = setting(self, 'engine.engine_settings.profile_target', '0') == "1" and not is_prefill and not last_only
        if _pt:
            torch.cuda.synchronize()
            _r0 = time.perf_counter()

        staged_verify = (self.prepare_staged_verify(seqs)
                         if not is_prefill and not last_only else None)
        if (staged_verify is None and not is_prefill and not last_only
                and getattr(self, "_verify_prestage_path_counts", None) is not None):
            self._verify_prestage_path_counts["unstaged"] += 1
        if staged_verify is not None:
            input_ids, positions, temperatures = staged_verify
        elif is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs, verify=not last_only)
        if staged_verify is None:
            temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        if _pt:
            torch.cuda.synchronize()
            _r1 = time.perf_counter()

        # Handle EAGLE returning (logits, conditioning_vector for next iter)
        conditioning = None
        jetspec_taps = None
        if self.config.use_eagle:
            logits, conditioning = self.run_model(
                input_ids, positions, is_prefill, last_only, hidden_states=hidden_states)
        elif self.config.jetspec_taps and not self.is_draft:
            logits, jetspec_taps = self.run_model(
                input_ids, positions, is_prefill, last_only, hidden_states=hidden_states)
        else:
            logits = self.run_model(input_ids, positions, is_prefill, last_only, hidden_states=hidden_states)

        if jetspec_taps is not None:
            prepare = getattr(self, '_reference_tap_prepare_hook', None)
            if prepare is not None and not is_prefill and not last_only:
                if len(seqs) != 1:
                    raise ValueError("early reference tap preparation requires N=1")
                prepare(seqs[0], jetspec_taps)
            offset = 0
            for seq in seqs:
                committed_count = seq.num_prompt_tokens if is_prefill else 1
                if is_prefill or (
                    last_only
                    and (not self.config.speculate or self.config.jetspec)
                ):
                    append_committed_taps(
                        seq,
                        jetspec_taps[offset:offset + committed_count],
                        committed_count,
                    )
                    offset += committed_count

        if _pt:
            torch.cuda.synchronize()
            _r2 = time.perf_counter()
            print(f"[PROFILE target_run] prepare_decode={(_r1-_r0)*1000:.2f}ms run_model={(_r2-_r1)*1000:.2f}ms eagle={self.config.use_eagle} n_ids={input_ids.shape[0]}", flush=True)

        if last_only:
            token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
            reset_context()
            if conditioning is not None:
                # the verifier's lossless dumper asks for logits too (JETSPEC_LOGIT_DUMP_EAGLE=1)
                return (token_ids, conditioning, logits) if draft_return_logits else (token_ids, conditioning)
            return (token_ids, logits) if draft_return_logits else token_ids
        else:
            self._staged_verify = None
            reset_context()
            if conditioning is not None:
                return logits, conditioning
            if jetspec_taps is not None:
                return logits, jetspec_taps
            return logits
    

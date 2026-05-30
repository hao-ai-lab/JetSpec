# Adapted from https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py
# Copyright (C) 2025, Tri Dao.
from __future__ import annotations

import builtins
import os
import time
import inspect
import base64
import hashlib
import json
from pathlib import Path
from functools import cached_property, partial
from typing import Dict, Tuple, List, Optional, Any, Callable

import torch
from torch import Tensor
import torch.distributed as dist

import triton
from triton import language as tl
from triton import runtime
import statistics
import math

# from . import __version__


PACKAGE_NAME = "optimus_jit"
VERSION = None


def get_home_dir():
    return os.getenv(f"{PACKAGE_NAME.upper()}_HOME", Path.home())


def default_cache_dir():
    return os.path.join(get_home_dir(), f".{PACKAGE_NAME}", "cache")


class FileCacheManager(triton.runtime.cache.FileCacheManager):
    def __init__(self, key):
        super().__init__(key)
        self.cache_dir = (
            os.getenv(f"{PACKAGE_NAME.upper()}_CACHE_DIR", "").strip() or default_cache_dir()
        )
        if self.cache_dir:
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            raise RuntimeError("Could not create or locate cache dir")


def _base32(key):
    # Assume key is a hex string.
    return base64.b32encode(bytes.fromhex(key)).decode("utf-8").rstrip("=")

def _quantile(a, q):
    n = len(a)
    a = sorted(a)

    def get_quantile(q):
        if not (0 <= q <= 1):
            raise ValueError("Quantiles must be in the range [0, 1]")
        point = q * (n - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        t = point - lower
        return (1 - t) * a[lower] + t * a[upper]

    return [get_quantile(q) for q in q]

def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = _quantile(times, quantiles)
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times
    elif return_mode == "min":
        return min(times)
    elif return_mode == "max":
        return max(times)
    elif return_mode == "mean":
        return statistics.mean(times)
    elif return_mode == "median":
        return statistics.median(times)

def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

def _dist_barrier() -> None:
    if _dist_ready():
        dist.barrier()

def _dist_allreduce_max_floats(values: List[float]) -> List[float]:
    if not _dist_ready():
        return [float(v) for v in values]
    backend = dist.get_backend()
    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and backend == "nccl"
        else torch.device("cpu")
    )
    tensor = torch.tensor([float(v) for v in values], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.cpu().tolist()

def triton_do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean"):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all". Default is "mean".
    :type return_mode: str
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    di = runtime.driver.active.get_device_interface()

    fn()
    di.synchronize()

    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    # # Estimate the runtime of the function
    # start_event = di.Event(enable_timing=True)
    # end_event = di.Event(enable_timing=True)
    # start_event.record()
    # for _ in range(5):
    #     runtime.driver.active.clear_cache(cache)
    #     print(f"clear cache {_}")
    #     fn()
    #     di.synchronize()
    #     print(f"after fn {_}")
    # end_event.record()
    # print("end record")
    # di.synchronize()
    # print("end synchronize")
    # estimate_ms = start_event.elapsed_time(end_event) / 5
    # di.synchronize()


    # compute number of warmup and repeat
    # n_warmup = max(1, int(warmup / estimate_ms))
    # n_repeat = max(1, int(rep / estimate_ms))
    n_warmup = warmup
    n_repeat = rep
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    # start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
        di.synchronize()
    # Benchmark
    start_event.record()
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        # if grad_to_none is not None:
        #     for x in grad_to_none:
        #         x.grad = None
        # we clear the L2 cache before each run
        # runtime.driver.active.clear_cache(cache)
        # record time of `fn`
        # start_event[i].record()

        fn()
        # di.synchronize()
        # end_event[i].record()
        
    # Record clocks

    end_event.record()
    di.synchronize()    
    # times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    times = start_event.elapsed_time(end_event)
    return _summarize_statistics([times], quantiles, return_mode)

class Autotuner:
    def __init__(
        self,
        fn,
        key,
        configs,
        restore_value=None,
        prune_configs_by: Optional[Dict] = None,
        do_bench=None,
        cache_results=False,
    ):
        """
        :param prune_configs_by: a dict of functions that are used to prune configs, fields:
            'perf_model': performance model used to predicate running time with different configs, returns running time
            'top_k': number of configs to bench
            'prune_num_stages_by'(optional): a function used to prune num_stages. It takes configs:List[Config] as its input, and returns pruned configs.
        """
        if not configs:
            self.configs = [AutotuneConfig()]
        else:
            self.configs = configs
        signature = inspect.signature(fn)
        self.keys = key
        self.cache: Dict[Tuple, AutotuneConfig] = {}
        self.arg_names = list(signature.parameters.keys())
        self.cache_results = (
            cache_results or os.getenv(f"{PACKAGE_NAME.upper()}_CACHE_AUTOTUNING", None) == "1"
        )

        self.restore_value = []
        if restore_value is not None:
            self.restore_value = list(restore_value)

        if len(self.restore_value) > 0:

            def _pre_hook(kwargs):
                self.restore_copies = {name: kwargs[name].clone() for name in self.restore_value}

            self.pre_hook = _pre_hook
        else:
            self.pre_hook = None

        if len(self.restore_value) > 0:

            def _post_hook(kwargs, exception):
                for name in self.restore_value:
                    kwargs[name].copy_(self.restore_copies[name])
                self.restore_copies = {}

            self.post_hook = _post_hook
        else:
            self.post_hook = None

        self.perf_model = None
        self.configs_top_k = 1.0
        self.early_config_prune = None
        if prune_configs_by:
            self.perf_model = prune_configs_by.get("perf_model", self.perf_model)
            self.configs_top_k = prune_configs_by.get("top_k", self.configs_top_k)
            self.early_config_prune = prune_configs_by.get(
                "early_config_prune", self.early_config_prune
            )

        self.fn = fn
        self._do_bench = do_bench
        # self.rank_id = int(os.environ['LOCAL_RANK'])

    @cached_property
    def do_bench(self):
        if self._do_bench is None:
            return partial(triton_do_bench, warmup=2, rep=10, return_mode="median")
        return self._do_bench

    def _bench(self, *args, config, **meta):
        verbose = os.environ.get(f"{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING", None) == "1"
        if verbose:
            print(f"rank {self.rank_id} Autotuning kernel {self.fn.__name__} with config {config}")

        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(
                f"Conflicting meta-parameters: {', '.join(conflicts)}."
                " Make sure that you don't re-define auto-tuned symbols."
            )
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.all_kwargs())
        full_nargs = {**self.nargs, **current}

        def kernel_call():
            if self.pre_hook is not None:
                self.pre_hook(full_nargs)
            try:
                self.fn.__call__(
                    *args,
                    **current,
                )
            except Exception as e:
                try:
                    if self.post_hook is not None:
                        self.post_hook(full_nargs, exception=e)
                finally:
                    # Throw exception raised by `self.fn.run`
                    raise

            if self.post_hook is not None:
                self.post_hook(full_nargs, exception=None)

        try:
            timing = self.do_bench(kernel_call)
            if verbose:
                print(f"rank {self.rank_id} timing:{timing}")
            return timing
        except Exception as e:
            if verbose:
                print(f"Autotuning failed with {e}")
            return [float("inf"), float("inf"), float("inf")]

    @torch.compiler.disable
    def check_disk_cache(self, tuning_key, configs, bench_fn):
        if not tuning_key:
            bench_fn()
            return

        fn = self.fn
        config_str_list = [str(c) for c in configs]
        assert len(config_str_list) == len(set(config_str_list)), "Config strings must be unique"
        cache_key = [str(tuning_key)] + config_str_list
        cache_key = hashlib.sha256("-".join(cache_key).encode("utf-8")).hexdigest()
        cache = FileCacheManager(_base32(cache_key))
        file_name = f"{fn.__name__[:150]}.autotune.json"
        path = cache.get_file(file_name)
        # There's an environment variable to force cache update
        if path and not os.environ.get(f"{PACKAGE_NAME.upper()}_FORCE_CACHE_UPDATE", False):
            str2config = {s: c for s, c in zip(config_str_list, configs)}
            with open(path, "r") as cached_configs:
                timings = json.load(cached_configs)["configs_timings"]
                timings = {str2config[config]: timing for config, timing in timings}
                self.cache[tuning_key] = builtins.min(timings, key=timings.get)
                self.configs_timings = timings
                self.bench_time = 0
            return

        bench_fn()
        cache.put(
            json.dumps(
                {
                    "key": tuning_key,
                    "configs_timings": [
                        (str(config), timings) for config, timings in self.configs_timings.items()
                    ],
                }
            ),
            file_name,
            binary=False,
        )

    def __call__(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        used_cached_result = True
        if len(self.configs) > 1:
            all_args = {**self.nargs, **kwargs}
            _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
            # Need "str" to make it json-serializable
            key = [str(_args[key]) for key in self.keys if key in _args]
            # print(f"key:{key}")
            # for _, arg in _args.items():
            #     if isinstance(arg, Tensor):
            #         key.append(str(arg.shape))
            #         # If stride != 0, 1, we just cache it as 2
            #         key.append(str([s if s in {0, 1} else 2 for s in arg.stride()]))
            #         key.append(str(arg.dtype))
            key = tuple(key)
            if key not in self.cache:
                # print(f"rank {self.rank_id} key not in autotune cache:{key}")
                used_cached_result = False
                pruned_configs = self.prune_configs(kwargs)

                # @torch.compiler.disable  # Don't want any tracing here
                def benchmark():
                    bench_start = time.time()
                    timings = {
                        config: self._bench(*args, config=config, **kwargs)
                        for config in pruned_configs
                    }
                    bench_end = time.time()
                    if os.getenv(f"{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING", None) == "1":
                        for config, time_ in timings.items():
                            print(f"[{config}] -> {time_[0]:.6f}ms")
                    self.bench_time = bench_end - bench_start
                    # 8卡同步：确保所有rank都完成本地bench再做聚合
                    _dist_barrier()
                    # 对每个config的timing做all-reduce max，得到8卡里最慢的那张卡的耗时
                    if _dist_ready():
                        for config in pruned_configs:
                            time_ = timings[config]
                            if not isinstance(time_, (list, tuple)):
                                time_ = [time_]
                            timings[config] = _dist_allreduce_max_floats(time_)
                    _dist_barrier()
                    if os.getenv(f"{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING", None) == "1":
                        for config, time_ in timings.items():
                            print(f"after all-reduce max [{config}] -> {time_[0]:.6f}ms")
                    best_config = builtins.min(timings, key=timings.get)

                    if os.getenv(f"{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING", None) == "1":
                        print(f"rank {self.rank_id} best_config:{best_config}")
                    self.cache[key] = best_config
                    self.configs_timings = timings
                benchmark()

                # if self.cache_results:
                #     self.check_disk_cache(key, pruned_configs, benchmark)
                # else:
                #     benchmark()
            # else:
            #     print(f"rank {self.rank_id} using cached result for key:{key}, config:{self.cache[key]}")

            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        if (
            os.getenv(f"{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING", None) == "1"
            and not used_cached_result
        ):
            print(
                f"{PACKAGE_NAME} autotuning for function {self.fn.__name__} finished after "
                f"{self.bench_time:.2f}s; best config selected: {self.best_config};"
            )
        ret = self.fn.__call__(
            *args,
            **kwargs,
            **config.all_kwargs(),
        )
        self.nargs = None
        return ret

    def prune_configs(self, kwargs: Dict) -> List[Any]:
        pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(self.configs, self.nargs, **kwargs)
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            elif not isinstance(top_k, int):
                # Slice index must be an integer
                raise TypeError(
                    "Error while pruning configs, top_k must be either 1) a float <= 1.0 or 2) an int"
                )

            if len(pruned_configs) > top_k:
                est_timing = {
                    config: self.perf_model(
                        **self.nargs,
                        **kwargs,
                        **config.all_kwargs(),
                    )
                    for config in pruned_configs
                }
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
        return pruned_configs


class AutotuneConfig:
    """
    An object that represents a possible kernel configuration for the auto-tuner to try.

    :ivar kwargs: a dictionary of meta-parameters to pass to the kernel as keyword arguments.
    :type kwargs: dict[Str, Any]
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __setstate__(self, state):
        self.kwargs = state.get("kwargs", {})

    def all_kwargs(self):
        return self.kwargs

    def __str__(self):
        res = []
        for k, v in self.kwargs.items():
            res.append(f"{k}: {v}")
        return ", ".join(res)

    def __hash__(self):
        return hash(tuple(*self.all_kwargs().items()))

    def __eq__(self, other):
        self_tuple = tuple(*self.all_kwargs().items())
        other_tuple = tuple(*other.all_kwargs().items())
        return self_tuple == other_tuple


def autotune(
    configs,
    key=None,
    prune_configs_by=None,
    restore_value=None,
    do_bench=None,
    cache_results=True,
):
    f"""
    Decorator for auto-tuning a function function.

    .. highlight:: python

    If the environment variable :code:`{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING` is set to
    :code:`"1"`, we will print a message to stdout after autotuning each
    kernel, including the time spent autotuning and the best configuration.

    :param configs: a list of :code:`AutotuneConfig` objects
    :type configs: list[AutotuneConfig]
    :param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
    :type key: list[str]
    :param prune_configs_by: a dict of functions that are used to prune configs, fields:
        'perf_model': performance model used to predicate running time with different configs, returns running time
        'top_k': number of configs to bench
        'early_config_prune'(optional): a function used to do early prune (eg, num_stages). It takes configs:List[Config] as its input, and returns pruned configs.
    :param restore_value: a list of argument names whose value will be restored after evaluating any configs.
    :type restore_value: list[str]
    :param do_bench: a benchmark function to measure the time of each run.
    :type do_bench: lambda fn, quantiles
    :param cache_results: whether to cache autotune timings to disk.  Defaults to False.
    "type cache_results: bool
    """

    if key is None:
        key = []

    def decorator(fn):
        return Autotuner(
            fn,
            key,
            configs,
            restore_value=restore_value,
            prune_configs_by=prune_configs_by,
            do_bench=do_bench,
            cache_results=cache_results,
        )

    return decorator

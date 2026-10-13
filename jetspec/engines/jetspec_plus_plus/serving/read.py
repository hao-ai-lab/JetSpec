#!/usr/bin/env python3
"""Run a manifest-pinned serving read from the repository.

JetSpec is the first shipped method.  Recipes are data under ``recipes/serve``;
the method enum is intentionally wider so later methods can add recipes without
changing the public command line.
Population invariants are timing-dependent whenever the draft deadline is enabled.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
# Script mode puts tools/ first on sys.path; the lazy `from tools.…` imports need the repo root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# The method core is a source package beside serve/; children inherit it through PYTHONPATH.
METHOD_ROOT = ROOT.parents[2]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))


class _RxProvenanceSubprocess:
    """Supply git identity from rx's synced provenance when its tree has no .git."""

    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        return getattr(self.module, name)

    def check_output(self, command, *args, **kwargs):
        provenance = ROOT.parent / "prov.json"
        if (not (ROOT / ".git").exists() and provenance.is_file()
                and command[:3] == ["git", "-C", str(ROOT)]):
            pin = json.loads(provenance.read_text(encoding="utf-8"))
            if not re.fullmatch(r"[0-9a-f]{40}", pin.get("head", "")):
                raise ValueError("rx provenance has no valid HEAD")
            if command[3:] == ["rev-parse", "HEAD"]:
                value = pin["head"] + "\n"
            elif command[3:] == ["status", "--porcelain", "--untracked-files=no"]:
                value = "" if pin.get("dirty") == "clean" else " M rx-synced-tree\n"
            else:
                return self.module.check_output(command, *args, **kwargs)
            return value if kwargs.get("text") else value.encode()
        return self.module.check_output(command, *args, **kwargs)


subprocess = _RxProvenanceSubprocess(subprocess)
MANIFEST_PATH = ROOT / "artifacts" / "manifest.json"
DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts" / "downloaded"
RECIPE_SCHEMA = "jetspec.serve_recipe/1"
HOST_VALUES_SCHEMA = "jetspec.serve_host_values/1"
# Host paths a recipe never carries: the runner reads them from the environment (each is a
# registered serving knob) or from the file named by --host-values.
HOST_VARIABLES = ("JETSPEC_TARGET_MODEL_PATH", "SSD_HF_CACHE", "SSD_DATASET_DIR")
HOST_PATH_FIELDS = ("target_model", "hf_cache", "dataset_dir", "blockhead_checkpoint")
METHODS = ("jetspecpp", "jetspec", "ar", "blockhead", "jetspec.engines.jetspec_plus_plus.serving.api", "spec_sync", "dflash",
           "dspark15", "eagle3", "jetspec_sync")
# A mask-initialised causal drafter (dflash_config.causal_head true) served synchronously: one GPU,
# verify then one drafter call, fresh context, no adapter, no heads.
SYNC_METHOD = "jetspec_sync"
SYNC_ROLES = ("drafter/config.json", "drafter/dflash.py", "drafter/model.safetensors",
              "target_release_weights")
SYNC_DRAFTER_FILES = ("config.json", "dflash.py", "model.safetensors",
                      "target_release_weights.safetensors")
REGIMES = ("eager", "graphed")
PLACEMENTS = ("pair", "one_gpu", "mps", "one_gpu_mps", "tp2")
METHOD_PLACEMENTS = {"ar": ("one_gpu", "tp2"), "blockhead": ("one_gpu", "tp2"),
                     "dflash": ("one_gpu", "tp2"), "spec_sync": ("one_gpu",)}
# tp2: the target tensor-parallel over two devices (NCCL), no async draft rank; the sync-slot
# drafter (block head / DFlash) stays on rank 0.  Only the sync-target methods above take it.
TP2_METHODS = ("ar", "blockhead", "dflash")
# Engine ranks a method starts (llm_engine.LLMEngine.__init__): a separate draft rank exists only
# with draft_async, which bench.py sets for JetSpec++ (--gpus 2).  AR runs one rank; the block head
# and the sync drafter run in rank 0's sync slot (the draft runner is bypassed, bench.py asserts
# --gpus 1 and no async).  Only a method with two ranks co-locates them on one_gpu / mps.
METHOD_ENGINE_RANKS = {"jetspecpp": 2, "jetspec.engines.jetspec_plus_plus.serving.api": 2, "eagle3": 2, "spec_sync": 1,
                       "ar": 1, "blockhead": 1, "dflash": 1, SYNC_METHOD: 1}
AR_DRAFT_REPO = "Qwen/Qwen3-0.6B"
AR_DRAFT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
AR_DRAFT_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
RUN_MANIFEST = re.compile(
    r"^RUN_MANIFEST sha256=([0-9a-f]{64}) content_sha256=([0-9a-f]{64}) "
    r"thresholds_sha256=([0-9a-f]{64}|absent)$")
EXPERIMENT_NAME = re.compile(r"^(?:JETSPEC_|E04_|SSD_|W[23][0-9]+_|PRED_)")
TRACEBACK = re.compile(r"(?:Traceback \(most recent call last\):|\bTraceback:)")
BENIGN_FINALIZER_TRACEBACK = re.compile(
    r"Exception ignored in: <Finalize object, dead>\n"
    r"Traceback \(most recent call last\):\n"
    r'  File "[^"\n]*/multiprocessing/util\.py", line [0-9]+, in __call__\n'
    r"    res = self\._callback\(\*self\._args, \*\*self\._kwargs\)\n"
    r"          \^+\n"
    r'  File "[^"\n]*/multiprocessing/synchronize\.py", line [0-9]+, in _cleanup\n'
    r"    sem_unlink\(name\)\n"
    r"FileNotFoundError: \[Errno 2\] No such file or directory")


def _identifier(record, field):
    if record.get("resolves") is not True or not record.get("value"):
        raise ValueError(f"unresolved {field}: {record!r}")
    return record["value"]


def canonical_method(method):
    if method in ("jetspecpp", "jetspec"):
        return "jetspecpp"
    if method == "dspark15":
        return "blockhead"
    return method








def target_eos_token_ids(target):
    """The target's own end-of-sequence ids: generation_config.json, else config.json."""
    found = set()
    for name in ("generation_config.json", "config.json"):
        path = Path(target) / name
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8")).get("eos_token_id")
            if value is not None:
                found.update(int(item) for item in (value if isinstance(value, list) else [value]))
            if found:
                return sorted(found)
    raise ValueError(f"no eos_token_id under {target}")








def load_recipe(root, method, regime, name):
    selected = canonical_method(method)
    if selected == SYNC_METHOD:
        return load_sync_recipe(root, regime, name)
    recipe_root = Path(root) / "configs"
    path = recipe_root / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no serve recipe for method={method} regime={regime}: {path}")
    recipe = json.loads(path.read_text(encoding="utf-8"))
    expected = dict(schema=RECIPE_SCHEMA, method=selected, regime=regime)
    for key, value in expected.items():
        if recipe.get(key) != value:
            raise ValueError(f"recipe {key} must be {value!r}, got {recipe.get(key)!r}")
    if selected == "jetspecpp" and recipe.get("display_name") not in (
            "JetSpec++", "JetSpec++ timing (e1 placeholder)"):
        raise ValueError("unsupported JetSpec++ recipe display_name")
    if recipe.get("name") != name:
        raise ValueError(f"recipe name must be {name!r}, got {recipe.get('name')!r}")
    sys.path.insert(0, str(Path(root) / 'serve'))
    from jetspec.engines.jetspec_plus_plus.draft.env import validate_lane_regime
    validate_lane_regime(recipe['environment'], regime)
    if selected == "jetspecpp":
        for field in ("drafter_artifact", "target_release_artifact"):
            if not isinstance(recipe.get(field), str) or not recipe[field]:
                raise ValueError(f"jetspecpp recipe must name {field}")
        if "adapter_artifact" in recipe and (
                not isinstance(recipe["adapter_artifact"], str) or not recipe["adapter_artifact"]):
            raise ValueError("adapter_artifact must name a manifest artifact")
    if "compile_pin_artifact" in recipe:
        if regime != "graphed":
            raise ValueError("compile_pin_artifact requires a graphed recipe")
        if not isinstance(recipe.get("compile_pin_sha256"), str) or not re.fullmatch(
                r"[0-9a-f]{64}", recipe["compile_pin_sha256"]):
            raise ValueError("compile_pin_artifact requires compile_pin_sha256")
    if selected in ("jetspec.engines.jetspec_plus_plus.serving.api", "spec_sync"):
        for field, value in (("draft_repo", AR_DRAFT_REPO),
                             ("draft_revision", AR_DRAFT_REVISION),
                             ("draft_sha256", AR_DRAFT_SHA256),
                             ("k", 4 if selected == "jetspec.engines.jetspec_plus_plus.serving.api" else 6)):
            if recipe.get(field) != value:
                raise ValueError(f"{selected} recipe {field} must be {value!r}")
        if not isinstance(recipe.get("display_name"), str) or not recipe["display_name"]:
            raise ValueError("draft recipe display_name must be nonempty")
    if selected == "jetspec.engines.jetspec_plus_plus.serving.api":
        choices = recipe.get("stock_parameters")
        if not isinstance(choices, dict) or "default" not in choices:
            raise ValueError("ssd recipe requires stock_parameters.default")
        for choice in choices.values():
            if not isinstance(choice, dict) or any(
                    type(choice.get(field)) is not int or choice[field] < 1
                    for field in ("lookahead", "fanout")):
                raise ValueError("ssd stock parameters require positive lookahead and fanout")
    if selected == "eagle3":
        k = eagle3_recipe_k(recipe)
        if recipe.get("k") != k or recipe.get("draft_artifact") != "eagle3_head":
            raise ValueError(f"eagle3 recipe requires k={k} and eagle3_head artifact")
        pins = recipe.get("draft_file_sha256")
        if not isinstance(pins, dict) or set(pins) != {"config.json", "pytorch_model.bin"} or any(
                not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in pins.values()):
            raise ValueError("eagle3 recipe requires config.json and pytorch_model.bin SHA256 pins")
    if not isinstance(recipe.get("passes"), int) or recipe["passes"] < 1:
        raise ValueError("recipe passes must be a positive integer")
    if not isinstance(recipe.get("environment"), dict):
        raise ValueError("recipe environment must be an object")
    host_fields = [field for field in HOST_PATH_FIELDS if field in recipe]
    if host_fields:
        raise ValueError(
            f"recipe {name} carries host path field(s) {', '.join(host_fields)}: a recipe names "
            f"artifacts by artifacts/manifest.json key; host paths come from "
            f"{', '.join(HOST_VARIABLES)} or --host-values")
    return recipe


def apply_sweep_overrides(recipe, args):
    """Vary only the existing JetSpec++ draft-work knobs for a measured sweep."""
    rows = args.sweep_rows_cap
    refresh = args.sweep_refresh
    j = args.sweep_j
    if rows is None and refresh is None and j is None:
        return recipe
    if (recipe["method"], recipe["regime"], args.placement) != (
            "jetspecpp", "eager", "pair"):
        raise ValueError("draft-work sweep requires eager JetSpec++ pair")
    varied = dict(recipe)
    varied["environment"] = dict(recipe["environment"])
    if rows is not None:
        varied["environment"]["JETSPEC_ROWS_CAP"] = str(rows)
        if rows < 16:
            # The ordinary admission issues 15 exact rows.  The existing budget
            # selector caps that admission before allocating the row bucket.
            varied["environment"]["JETSPEC_ROW_SELECT"] = "budget"
    if refresh is not None:
        varied["environment"]["JETSPEC_ADAPTER_REFRESH"] = refresh
    if j is not None:
        varied["passes"] = j + 1
    return varied


def apply_policy_overrides(recipe, args):
    """Resolve one policy entry through the same knob mapping as the sweep."""
    if not getattr(args, "policy", None):
        return apply_sweep_overrides(recipe, args), None
    if any(getattr(args, field, None) is not None for field in
           ("sweep_rows_cap", "sweep_refresh", "sweep_j")):
        raise ValueError("--policy cannot be combined with --sweep-rows-cap/--sweep-refresh/--sweep-j")
    path = Path(args.policy).resolve()
    raw = path.read_bytes()
    policy = json.loads(raw)
    if policy.get("schema") != "jetspec.batch_policy/1":
        raise ValueError(f"unsupported batch policy schema: {policy.get('schema')!r}")
    choice = policy.get("entries", {}).get(str(args.batch))
    if choice is None:
        raise ValueError(f"batch policy has no entry for C={args.batch}")
    rows, refresh, j = (choice.get("rows_cap"), choice.get("refresh"), choice.get("J"))
    if rows not in (2, 4, 8, 16) or refresh not in ("after_j0", "off") or j not in (1, 2):
        raise ValueError(f"invalid batch policy knobs at C={args.batch}: {choice!r}")
    overrides = argparse.Namespace(sweep_rows_cap=rows, sweep_refresh=refresh,
                                   sweep_j=j, placement=args.placement)
    varied = apply_sweep_overrides(recipe, overrides)
    return varied, {"policy": str(path), "policy_sha256": hashlib.sha256(raw).hexdigest(),
                    "policy_choice": {"rows_cap": rows, "refresh": refresh, "J": j}}


def validate_recipe_batch(recipe, batch):
    if recipe["method"] != "jetspecpp":
        return
    if "batch_only" in recipe and type(recipe["batch_only"]) is not bool:
        raise ValueError("jetspecpp recipe batch_only must be a boolean")
    if recipe.get("batch_only") and batch <= 1:
        raise ValueError("batch_only jetspecpp recipe requires --batch > 1")


def require_supported_block_size(recipe):
    value = recipe["environment"].get("JETSPEC_BLOCK_SIZE_READ", "16")
    if value not in ("4", "8", "16", "32"):
        raise ValueError("JETSPEC_BLOCK_SIZE_READ must be 4, 8, 16 or 32")


def no_host_values():
    return {"path": None, "artifacts": {}, "environment": {},
            "cuda_arch": None, "compile_pin_sha256": None,
            "rendezvous_mode": "netns", "graphed_ar_disable_dynamo": False}


def load_host_values(path):
    """Per-host paths, architecture, compile pin, and rendezvous mode (never recipe edits)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing host values file: {path}")
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or values.get("schema") != HOST_VALUES_SCHEMA:
        raise ValueError(f"host values schema must be {HOST_VALUES_SCHEMA!r}: {path}")
    unknown = sorted(set(values) - {"schema", "artifacts", "environment", "cuda_arch",
                                    "compile_pin_sha256", "rendezvous_mode",
                                    "graphed_ar_disable_dynamo"})
    if unknown:
        raise ValueError(f"host values carry unknown key(s) {', '.join(unknown)}: {path}")
    artifacts = values.get("artifacts", {})
    environment = values.get("environment", {})
    cuda_arch = values.get("cuda_arch")
    compile_pin_sha256 = values.get("compile_pin_sha256")
    rendezvous_mode = values.get("rendezvous_mode", "netns")
    graphed_ar_disable_dynamo = values.get("graphed_ar_disable_dynamo", False)
    if not isinstance(artifacts, dict) or not isinstance(environment, dict):
        raise ValueError(f"host values artifacts and environment must be objects: {path}")
    if cuda_arch is not None and (not isinstance(cuda_arch, str) or
                                  not re.fullmatch(r"[0-9]+\.[0-9]+", cuda_arch)):
        raise ValueError(f"host values cuda_arch must be a compute capability like '10.0': {path}")
    if compile_pin_sha256 is not None and (not isinstance(compile_pin_sha256, str) or
                                            not re.fullmatch(r"[0-9a-f]{64}", compile_pin_sha256)):
        raise ValueError(f"host values compile_pin_sha256 must be a 64-character SHA256: {path}")
    if rendezvous_mode not in ("netns", "host_port"):
        raise ValueError(f"host values rendezvous_mode must be 'netns' or 'host_port': {path}")
    if not isinstance(graphed_ar_disable_dynamo, bool):
        raise ValueError(f"host values graphed_ar_disable_dynamo must be boolean: {path}")
    known = set(_manifest_artifacts())
    for name, directory in artifacts.items():
        if name not in known:
            raise ValueError(f"host values name absent artifacts/manifest.json key {name!r}: {path}")
        if not isinstance(directory, str) or not Path(directory).is_absolute():
            raise ValueError(f"host values artifact {name!r} must be an absolute path: {path}")
    for name, value in environment.items():
        if name not in HOST_VARIABLES:
            raise ValueError(
                f"host values environment may name only {', '.join(HOST_VARIABLES)}, got {name!r}: {path}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"host values environment {name} must be a nonempty string: {path}")
    return {"path": str(path.resolve()), "artifacts": dict(artifacts),
            "environment": dict(environment), "cuda_arch": cuda_arch,
            "compile_pin_sha256": compile_pin_sha256,
            "rendezvous_mode": rendezvous_mode,
            "graphed_ar_disable_dynamo": graphed_ar_disable_dynamo}


def resolve_cuda_arch(recipe, host):
    return host["cuda_arch"] or recipe.get("cuda_arch", "8.0")


def resolve_compile_pin_sha256(recipe, host):
    return host["compile_pin_sha256"] or recipe.get("compile_pin_sha256")


def resolve_graphed_ar_disable_dynamo(recipe, host):
    return (host["graphed_ar_disable_dynamo"] and recipe["method"] == "ar"
            and recipe["regime"] == "graphed")


def apply_host_artifact_overrides(host, overrides):
    """Resolve an already fetched host directory by manifest key; pins are checked at load."""
    for override in overrides or ():
        name, sep, directory = override.partition("=")
        if not sep or name not in _manifest_artifacts() or not Path(directory).is_absolute():
            raise ValueError(f"--host-artifact requires a manifest key and absolute directory: {override}")
        host["artifacts"][name] = directory
    return host


def host_variable(name, host):
    """Return (value, source) for one of HOST_VARIABLES; fail closed naming the variable."""
    if host["environment"].get(name):
        return host["environment"][name], f"host_values:{name}"
    if os.environ.get(name):
        return os.environ[name], f"environment:{name}"
    raise ValueError(
        f"{name} is unset: export {name} or name it under \"environment\" in --host-values "
        f"(no default is supplied)")


def _manifest_artifacts():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    names = [row["name"] for row in manifest["artifacts"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate artifact name in artifacts/manifest.json")
    return {row["name"]: row for row in manifest["artifacts"]}


def artifact_directory(name, artifacts, host, artifacts_dir):
    """Resolve a manifest key to (directory, source): a host value, else under the fetch directory."""
    if name not in artifacts:
        raise ValueError(f"recipe names absent artifacts/manifest.json key {name!r}")
    if name in host["artifacts"]:
        return Path(host["artifacts"][name]), f"host_values:artifacts.{name}"
    row = artifacts[name]
    if row["kind"] == "generated":
        return Path(artifacts_dir) / name, f"artifacts_dir:{name}"
    destination = _identifier(row["destination_path"], f"{name} destination")
    return Path(artifacts_dir) / destination, f"artifacts_dir:{destination}"


def assert_artifact_present(name, artifacts, directory, source):
    """Every file the manifest lists for ``name`` exists under ``directory``; say why not."""
    row = artifacts[name]
    files = [_identifier(entry["destination_path"], f"{name} file destination")
             for entry in row.get("files", [])]
    missing = [file for file in files if not (Path(directory) / file).is_file()]
    if not files and not Path(directory).is_dir():
        missing = ["<directory>"]
    if not missing:
        return
    listing = ", ".join(missing[:4]) + (f" (+{len(missing) - 4} more)" if len(missing) > 4 else "")
    remedy = (f"place the artifact files under {directory} or pass the directory by flag"
              if source.startswith("artifacts_dir:") else "check the --host-values entry")
    raise FileNotFoundError(
        f"artifact {name!r} is not fetched under {directory} (resolved from {source}): "
        f"lacks {listing}; {remedy}")


def _persistent_cache_environment(recipe, cache_root):
    declared = recipe.get("cache_environment", {})
    if not isinstance(declared, dict):
        raise ValueError("recipe cache_environment must be an object")
    root = Path(cache_root).resolve()
    resolved = {}
    for name, relative in declared.items():
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError(f"{name} cache path must resolve beneath the runner cache root")
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError(f"{name} cache path must resolve beneath the runner cache root")
        resolved[name] = str(path)
    return resolved


def _condition_environment(environment, recipe):
    cache_names = set(recipe.get("cache_environment", {}))
    always = {
        "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "TORCHINDUCTOR_COMPILE_THREADS", "TORCHDYNAMO_DISABLE",
    }
    return {key: environment[key] for key in sorted(environment)
            if EXPERIMENT_NAME.match(key) or key in always or key in cache_names
            or key.startswith("PYTHON")}


def _isolated_command(command, rendezvous_mode="netns"):
    if rendezvous_mode == "host_port":
        return command
    return [
        "unshare", "-n", "--", "bash", "-c",
        'ip link set lo up; exec "$@"', "bash", *command,
    ]


def _free_local_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _run_child_with_rank_watch(command, environment, *, wall_timeout=18 * 60):
    """Bound rank pipe waits, including when the parent exits before its ranks."""
    child = subprocess.Popen(command, env=environment, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             start_new_session=True)
    started = time.monotonic()
    traceback_seen = exited_at = None

    def stop(reason, partial):
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            output, _ = child.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                output, _ = child.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                child.stdout.close()
                output = partial
        returncode = child.returncode if child.returncode is not None else 1
        if returncode == 0 and (traceback_seen is not None
                               or reason != "child exited while rank held output pipe"):
            returncode = 1
        return subprocess.CompletedProcess(command, returncode, output + "\n" + reason + "\n")

    while True:
        try:
            output, _ = child.communicate(timeout=0.25)
            return subprocess.CompletedProcess(command, child.returncode, output)
        except subprocess.TimeoutExpired as error:
            partial = error.output or b""
            if isinstance(partial, bytes):
                partial = partial.decode(errors="replace")
            scrubbed = BENIGN_FINALIZER_TRACEBACK.sub("", partial)
            if TRACEBACK.search(scrubbed):
                traceback_seen = traceback_seen or time.monotonic()
            else:
                traceback_seen = None
            now = time.monotonic()
            if child.poll() is not None:
                exited_at = exited_at or now
            if traceback_seen is not None and now - traceback_seen >= 1:
                return stop("rank traceback left parent waiting", partial)
            if exited_at is not None and now - exited_at >= 0.75:
                return stop("child exited while rank held output pipe", partial)
            if now - started >= wall_timeout:
                return stop("serve child wall timeout", partial)


def validate_method_placement(method, placement):
    allowed = METHOD_PLACEMENTS.get(canonical_method(method))
    if placement == "tp2" and canonical_method(method) not in TP2_METHODS:
        raise ValueError(f"placement tp2 is only for {', '.join(TP2_METHODS)}, got {method}")
    if allowed is not None and placement not in allowed:
        raise ValueError(
            f"method {method} requires placement {' or '.join(allowed)}, got {placement}")


def assert_measured_output_shape(outputs, *, count, tokens, stop_at_eos):
    lengths = [len(row["token_ids"]) for row in outputs]
    if len(outputs) != count or any(
            not (1 <= length <= tokens) if stop_at_eos else length != tokens
            for length in lengths):
        raise RuntimeError("output shape mismatch in measured")


# max_model_len (bench.py --max_model_len) is a served condition.  Every read before the derivation
# served 768, and 768 holds prompt + --tokens + one round's lookahead for the gsm8k, math500,
# humaneval, mtbench and mbpp prompt sets at 256 tokens, so the rule keeps 768 whenever the derived
# length fits in it (those reads' RESOLVED_CONDITION is unchanged) and serves the derived length only
# when a prompt set needs more (aime25 795 prompt tokens, livecodebench 1110).
MAX_MODEL_LEN_FLOOR = 768
# bench.py --block_sz default (config.kvcache_block_size); the runner never passes --block_sz.
KV_BLOCK_SIZE = 256
# Config asserts max_num_batched_tokens >= max_model_len; bench.py keeps the config default.
MAX_NUM_BATCHED_TOKENS = 16384
# Fixed drafter caches use these run-selected capacities; 768 retains the original graphs.
CACHE_BUCKETS = (768, 1536, 2560, 4096)
BLOCKHEAD_GRAPH_CAPACITY = CACHE_BUCKETS[0]
SYNC_STATIC_CAPACITY = CACHE_BUCKETS[0]
SYNC_DRAFTS = 15  # sync_drafter.SYNC_LOOKAHEAD; bench.py sets args.k = 15 for the sync drafter
# One 16-token verify block (anchor + 15 drafts): the least lookahead any read is sized for.
MIN_LOOKAHEAD = 16


def prompt_token_lengths(prompts_json, count):
    """(index, token count) of the rows bench.py serves: prompts[:count] of the pinned prompts JSON."""
    exported = json.loads(Path(prompts_json).read_text(encoding="utf-8"))
    rows = exported["prompts"] if isinstance(exported, dict) else exported
    rows = rows[:count]
    if not rows:
        raise ValueError(f"no prompts in {prompts_json}")
    return [(int(row.get("index", position)), len(row["token_ids"]))
            for position, row in enumerate(rows)]


def warmup_prefix_lengths(method, regime, environment, prompts_json, count):
    """The distinct prompt lengths the draft warms one by one before decode (JETSPEC_WARMUP_PREFIX_
    LENGTHS): graphed JetSpec++ with the pow2 row buckets, or with the batched draft graph
    (JETSPEC_BATCH_DRAFT_GRAPH=1), whose admission builds the anchor decoder's prefix and the
    adapter seed at the prompt's own row count (a Triton constexpr); else none."""
    if method != "jetspecpp" or regime != "graphed" or (
            environment.get("JETSPEC_G_ROW_BUCKETS") != "pow2"
            and environment.get("JETSPEC_BATCH_DRAFT_GRAPH") != "1"):
        return []
    return sorted({length for _, length in prompt_token_lengths(prompts_json, count)})


def round_lookahead(method, environment, blockhead=None):
    """Tokens the scheduler reserves per decode round before it may append (serving/
    scheduler.py, target_lookahead_len = self.K + 1 for every speculative mode, 1 without
    speculation; block_manager.can_append refuses when num_tokens + lookahead > max_model_len),
    never below MIN_LOOKAHEAD."""
    method = canonical_method(method)
    block_size_read = int(environment.get("JETSPEC_BLOCK_SIZE_READ", "16"))
    if block_size_read not in (4, 8, 16, 32):
        raise ValueError("JETSPEC_BLOCK_SIZE_READ must be 4, 8, 16 or 32")
    if method in ("ar", "jetspec.engines.jetspec_plus_plus.serving.api", "spec_sync", "eagle3"):
        return MIN_LOOKAHEAD
    if method == "jetspecpp":
        if "JETSPEC_DRAFTS" not in environment:
            raise ValueError("JETSPEC_DRAFTS must be set for JetSpec++")
        drafts = int(environment["JETSPEC_DRAFTS"])
        if drafts not in (3, 7, 14, 15, 31):
            raise ValueError("JETSPEC_DRAFTS must be 3, 7, 14, 15, or 31; minimum served block is 4")
        if block_size_read != drafts + 1:
            raise ValueError("JetSpec++ verify block must equal JETSPEC_DRAFTS + 1")
        return block_size_read
    if method == SYNC_METHOD:
        return block_size_read if block_size_read == 8 else max(block_size_read, SYNC_DRAFTS + 1)
    if method in ("blockhead", "dflash"):
        if "JETSPEC_DRAFTS" in environment:
            drafts = int(environment["JETSPEC_DRAFTS"])
            if drafts not in (3, 7, 15, 31):
                raise ValueError("batched block-head JETSPEC_DRAFTS must be 3, 7, 15 or 31")
            return drafts + 1
        if block_size_read == 8:
            return 8
        config = json.loads((Path(blockhead) / "config.json").read_text(encoding="utf-8"))
        return max(block_size_read, int(config["block_size"]) + (method == "blockhead"))
    raise ValueError(f"method {method} declares no per-round lookahead")


def context_capacity(method, environment, max_model_len):
    """(selected capacity, cache name) for a fixed drafter cache, or (None, None)."""
    method = canonical_method(method)
    from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled as eval_enabled
    if (method == "jetspecpp" and environment.get("JETSPEC_CHAIN_GRAPH") == "seg"
            and eval_enabled(environment)):
        # Eval chain caches are sized from max_model_len (apply_chain_capacity), not a bucket.
        return None, None
    if method in ("blockhead", "dflash") and environment.get("JETSPEC_BLOCKHEAD_GRAPH") == "1":
        name = "blockhead_graph.CACHE_CAPACITY (JETSPEC_BLOCKHEAD_GRAPH=1)"
    elif method == SYNC_METHOD and environment.get("JETSPEC_SYNC_STATIC", "1") != "0":
        name = "sync_drafter.STATIC_CAPACITY (JETSPEC_SYNC_STATIC=1)"
    elif method == "jetspecpp":
        name = ("W259_C1_CACHE_CAPACITY (JETSPEC_C1_KVCACHE=1)"
                if environment.get("JETSPEC_C1_KVCACHE", "0") == "1"
                else "JETSPEC_DECODER_CACHE_CAPACITY")
    else:
        return None, None
    return next((bucket for bucket in CACHE_BUCKETS if max_model_len <= bucket), None), name


def resolve_max_model_len(prompts_json, count, tokens, method, environment, target,
                          blockhead=None, forced=None):
    """The --max_model_len to serve and how it was derived.

    The compared quantity is longest prompt + --tokens + the round lookahead.  It stays
    MAX_MODEL_LEN_FLOOR (768) when that sum is at most 768, else it is the sum rounded up to the KV
    block size.  ``forced`` (recipe max_model_len or --max-model-len) replaces the derived value but
    is still checked.  Fails closed, before any model load, when the served value cannot hold some
    prompt + --tokens + lookahead or passes an engine ceiling."""
    rows = prompt_token_lengths(prompts_json, count)
    lookahead = round_lookahead(method, environment, blockhead)
    longest_index, longest = max(rows, key=lambda row: row[1])
    needed = longest + tokens + lookahead
    derived = MAX_MODEL_LEN_FLOOR if needed <= MAX_MODEL_LEN_FLOOR else \
        -(-needed // KV_BLOCK_SIZE) * KV_BLOCK_SIZE
    value = derived if forced is None else int(forced)
    for index, length in rows:
        if length + tokens + lookahead > value:
            raise ValueError(
                f"max_model_len {value} cannot hold prompt index {index} of {prompts_json}: "
                f"{length} prompt tokens + --tokens {tokens} + lookahead {lookahead} = "
                f"{length + tokens + lookahead}")
    target_config = json.loads((Path(target) / "config.json").read_text(encoding="utf-8"))
    if target_config.get("model_type") == "qwen3_5":
        target_config = target_config["text_config"]
    # The engine's position and batch limits bound the served value (Config.__post_init__ would
    # silently clamp it); a drafter's fixed context cache bounds the context it must hold.
    ceilings = [("target max_position_embeddings", value,
                 int(target_config["max_position_embeddings"])),
                ("max_num_batched_tokens", value, MAX_NUM_BATCHED_TOKENS)]
    capacity, what = context_capacity(method, environment, value)
    if what is not None:
        ceilings.append((what, value, capacity or CACHE_BUCKETS[-1]))
    for name, compared, ceiling in ceilings:
        if compared > ceiling:
            raise ValueError(
                f"prompts {prompts_json} need max_model_len {needed} (prompt index "
                f"{longest_index}: {longest} prompt tokens + --tokens {tokens} + "
                f"lookahead {lookahead}; served {value}) but {name} is {ceiling} "
                f"(largest bucket {CACHE_BUCKETS[-1]})")
    return {"value": value, "source": "derived" if forced is None else "forced",
            "derived": derived, "needed": needed, "max_prompt_tokens": longest,
            "tokens": tokens, "lookahead": lookahead, "block_size": KV_BLOCK_SIZE,
            "floor": MAX_MODEL_LEN_FLOOR, "cache_capacity": capacity}


def apply_cache_capacity(environment, method, resolved):
    """Bind optional caches to the run bucket or the chain's conservative key bound."""
    capacity = resolved["cache_capacity"]
    if canonical_method(method) == "jetspecpp":
        if environment.get("JETSPEC_CHAIN_GRAPH") == "seg":
            from jetspec.engines.jetspec_plus_plus.draft.cache_capacity import apply_chain_capacity
            apply_chain_capacity(environment, resolved["value"])
            resolved["cache_capacity"] = int(environment["W259_C1_CACHE_CAPACITY"])
            return
        if environment.get("JETSPEC_C1_KVCACHE", "0") == "1":
            environment["W259_C1_CACHE_CAPACITY"] = str(capacity)
        if resolved["value"] > 768:
            # The decoder needs 32 spare rows for its private window.
            environment["JETSPEC_DECODER_CACHE_CAPACITY"] = str(
                next(bucket for bucket in CACHE_BUCKETS if resolved["value"] <= bucket) + 32)


def forced_max_model_len(recipe, args):
    """--max-model-len, else the recipe's max_model_len, else None (derive it)."""
    value = getattr(args, "max_model_len", None)
    if value is None:
        value = recipe.get("max_model_len")
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
        raise ValueError(f"max_model_len must be a positive integer, got {value!r}")
    return value


def resolve_draft_deadline(recipe_value, override):
    if override is None:
        return recipe_value
    return "1" if override == "on" else "0"


def apply_miss_redraft_override(environment, override):
    if override == "off":
        for name in ("JETSPEC_MISS_REDRAFT", "JETSPEC_MISS_REDRAFT_PASSES",
                     "JETSPEC_MISS_REDRAFT_ANCHOR"):
            environment.pop(name, None)


def shared_gpu_kv_cap(method, placement, recipe_environment=None):
    """SSD_SHARED_GPU_KV_UTILIZATION for this method and placement, or None when it is not set.

    The cap exists because two engine ranks on one GPU each reserve KV from the same free memory
    (a JetSpec++ one_gpu OOM).  It applies only when the method starts two ranks and the
    placement puts them on one device; a recipe that sets the knob keeps its own value."""
    recipe_environment = recipe_environment or {}
    if placement == "pair":
        return None
    if "SSD_SHARED_GPU_KV_UTILIZATION" in recipe_environment:
        return recipe_environment["SSD_SHARED_GPU_KV_UTILIZATION"]
    ranks = METHOD_ENGINE_RANKS.get(canonical_method(method))
    if ranks is None:
        raise ValueError(f"method {method} declares no engine rank count")
    return "0.05" if ranks > 1 else None


def resolve_placement(placement, devices, recipe_environment=None, method="jetspecpp"):
    from jetspec.engines.jetspec_plus_plus.serving.presets.transport import resolve_placement as resolve
    return resolve(placement, devices,
                   shared_kv_cap=(None if placement in ('pair', 'tp2') else
                                  shared_gpu_kv_cap(method, placement, recipe_environment)))


def resolve_draft_split_placement(placement, recipe_environment):
    if recipe_environment.get("JETSPEC_DRAFT_SPLIT") != "verify_gpu":
        return placement
    if placement["placement"] != "pair":
        raise ValueError("draft split requires pair placement")
    split = dict(placement, environment=dict(placement["environment"]))
    # Each spawned rank sees exactly its own physical GPU. The MPS server only
    # exposes the verify GPU, so a shared two-device CUDA_VISIBLE_DEVICES would
    # make CUDA report zero devices to its clients.
    split["environment"].update(SSD_RANK_DEVICE_MAP="0,0,0",
                                SSD_SHARED_GPU_DIST="gloo")
    return split


def resolve_cpu_pin(args, uuids, method):
    mode = getattr(args, "cpu_pin", "auto")
    alias = getattr(args, "rank_cpus", None)
    if alias is not None:
        if mode != "auto":
            raise ValueError("--rank-cpus conflicts with --cpu-pin")
        mode = alias
    ranks = (3 if method == "jetspecpp" and getattr(args, "draft_split", None) == "verify_gpu"
             else 2 if args.placement == "tp2" else METHOD_ENGINE_RANKS[method])
    if args.resolve_only:
        if mode not in ("auto", "off"):
            if str(ROOT / "serve") not in sys.path:
                sys.path.insert(0, str(ROOT / "serve"))
            from jetspec.engines.jetspec_plus_plus.serving.cpu_affinity import parse_cpu_ranges
            parse_cpu_ranges(mode, ranks)
        return {"mode": "explicit" if mode not in ("auto", "off") else mode,
                "ranks": [], "spec": None, "unresolved": True}
    if str(ROOT / "serve") not in sys.path:
        sys.path.insert(0, str(ROOT / "serve"))
    from jetspec.engines.jetspec_plus_plus.serving.cpu_affinity import plan_cpu_pin
    # tp2 maps rank r to device r exactly as pair does.
    return plan_cpu_pin(mode, uuids, "pair" if args.placement == "tp2" else args.placement, ranks)


def apply_cpu_pin_environment(environment, pin):
    environment["SSD_CPU_PIN_PLAN"] = json.dumps(pin, sort_keys=True)
    environment.pop("SSD_RANK_CPU_AFFINITY", None)
    if pin["spec"] is not None:
        environment["SSD_RANK_CPU_AFFINITY"] = pin["spec"]


def apply_mps_draft_environment(environment, args, method):
    """Opt-in MPS client knobs for the spawned draft process only (llm_engine scopes them)."""
    priority = getattr(args, "mps_draft_priority", None)
    percentage = getattr(args, "mps_draft_active_thread_percentage", None)
    if priority is None and percentage is None:
        return
    if method != "jetspecpp" or args.placement != "one_gpu_mps":
        raise ValueError("draft MPS overrides require jetspecpp one_gpu_mps")
    if priority is not None:
        environment["JETSPEC_MPS_DRAFT_PRIORITY"] = priority
    if percentage is not None:
        if not 1 <= percentage <= 100:
            raise ValueError("draft MPS thread percentage must be 1..100")
        environment["JETSPEC_MPS_DRAFT_ACTIVE_THREAD_PERCENTAGE"] = str(percentage)


def assert_manifest_cpu_pin(manifest, requested):
    values = manifest.get("values", {})
    if values.get("cpu_pin") != requested["mode"]:
        raise RuntimeError(f"RUN_MANIFEST cpu_pin mismatch: {values.get('cpu_pin')!r} != "
                           f"{requested['mode']!r}")
    for row in requested["ranks"]:
        actual = values.get(f"cpu_pin_rank_{row['rank']}")
        if actual != row:
            raise RuntimeError(f"RUN_MANIFEST CPU_PIN rank {row['rank']} mismatch: "
                               f"{actual!r} != {row!r}")


def campaign_output_paths(output, repeats):
    """Return one immutable native output per one-generate process."""
    output = Path(output)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    paths = [output]
    for index in range(repeats - 1):
        paths.append(output.with_name(f"{output.stem}.measured-{index}{output.suffix}"))
    return paths




def assert_completed_run(returncode, log, output):
    if returncode:
        raise RuntimeError(f"serve read child exited {returncode}")
    scrubbed, benign_finalizer_tracebacks = BENIGN_FINALIZER_TRACEBACK.subn("", log)
    if TRACEBACK.search(scrubbed):
        raise RuntimeError("serve read refused traceback text despite child exit status")
    if not Path(output).is_file():
        raise RuntimeError(f"missing output artifact: {output}")
    return benign_finalizer_tracebacks


def assert_run_manifest(output, log, expected_threshold):
    path = Path(str(output) + ".manifest.json")
    if not path.is_file():
        raise RuntimeError(f"missing run manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    lines = [line for line in log.splitlines() if line.startswith("RUN_MANIFEST ")]
    if len(lines) != 1:
        raise RuntimeError(f"expected one RUN_MANIFEST line, found {len(lines)}")
    match = RUN_MANIFEST.fullmatch(lines[0])
    if match is None:
        raise RuntimeError("malformed RUN_MANIFEST line")
    actual = (manifest.get("manifest_sha256"), manifest.get("content_sha256"),
              _manifest_role_sha(manifest, "heads/thresholds") or "absent")
    printed = match.groups()
    if printed != actual:
        raise RuntimeError(f"RUN_MANIFEST printed/written mismatch: {printed!r} != {actual!r}")
    if actual[2] != expected_threshold:
        raise RuntimeError(
            f"RUN_MANIFEST thresholds mismatch: {actual[2]} != {expected_threshold}")
    if manifest.get("value_conflicts"):
        raise RuntimeError("RUN_MANIFEST contains value conflicts")
    return manifest


def _manifest_role_sha(manifest, role):
    found = sorted({entry["sha256"] for entry in manifest.get("files", [])
                    if entry.get("role") == role})
    return None if not found else found[0] if len(found) == 1 else found


def assert_manifest_artifacts(manifest, expected):
    for role, sha256 in sorted(expected.items()):
        actual = _manifest_role_sha(manifest, role)
        if actual != sha256:
            raise RuntimeError(f"RUN_MANIFEST artifact mismatch {role}: {actual!r} != {sha256!r}")


def assert_kpred_manifest(manifest, flag):
    expected = "legacy" if flag == "1" else "not_loaded"
    actual = manifest.get("values", {}).get("kpred_forward")
    if actual != expected:
        raise RuntimeError(f"RUN_MANIFEST kpred_forward mismatch: {actual!r} != {expected!r}")
    roles = [entry["role"] for entry in manifest.get("files", [])
             if entry["role"].startswith("heads/kpred/")]
    if flag == "0" and roles:
        raise RuntimeError(f"RUN_MANIFEST kpred roles must be absent: {roles}")
    if flag == "1" and not roles:
        raise RuntimeError("RUN_MANIFEST kpred roles must be present")


def assert_manifest_method(manifest, expected):
    actual = manifest.get("values", {}).get("method_mode")
    if actual != expected:
        raise RuntimeError(f"RUN_MANIFEST method_mode mismatch: {actual!r} != {expected!r}")


def join_populations(decisions, proposals, deadlines, *, deadline_required=True):
    proposal_index = {(row["seq_id"], row["candidate_round"]): row for row in proposals}
    deadline_index = {(row["seq_id"], row["round_id"]): row for row in deadlines}
    if len(proposal_index) != len(proposals):
        raise RuntimeError("duplicate proposal population key")
    if len(deadline_index) != len(deadlines):
        raise RuntimeError("duplicate deadline population key")
    joined = []
    for decision in decisions:
        key = (decision["seq_id"], decision.get("candidate_round"))
        if key[1] in (None, 0):
            joined.append({
                "seq_id": key[0], "candidate_round": key[1],
                "decision": decision["decision"],
                "fallback": bool(decision.get("fallback", False)),
                "accepted_length": decision.get("accepted_len"),
                "rows_requested": 0, "rows_published": decision.get("rows_published", 0),
                "rows_truncated": 0, "pass": None, "refresh_calls": 0,
                "deadline": deadline_index.get(key), "join_status": "admission",
            })
            continue
        proposal = proposal_index.get(key)
        if proposal is None:
            raise RuntimeError(f"unjoined verifier round: {key}")
        deadline = deadline_index.get(key)
        if deadline is None and deadline_required:
            raise RuntimeError(f"unjoined deadline round: {key}")
        joined.append({
            "seq_id": key[0], "candidate_round": key[1],
            "decision": decision["decision"],
            "fallback": bool(decision.get("fallback", False)),
            "accepted_length": decision.get("accepted_len"),
            "rows_requested": proposal["rows_requested"],
            "rows_published": proposal["rows_published"],
            "rows_truncated": proposal["rows_truncated"],
            "pass": proposal["pass_label"],
            "refresh_calls": proposal.get("refresh_calls", 0),
            "deadline": deadline, "join_status": "joined",
        })
    return joined


def method_populations(method, metrics, proposals, deadlines, *, deadline_required=True):
    if method == "jetspecpp":
        return join_populations(metrics["jetspec_round_records"], proposals, deadlines,
                                deadline_required=deadline_required)
    if method in ("blockhead", "dflash"):
        return list(metrics["blockhead_round_records"])
    return []


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_revision(root=ROOT):
    """Local git revision, or the rx sync receipt when the remote tree omits .git."""
    try:
        sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                      text=True, stderr=subprocess.PIPE).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            text=True, stderr=subprocess.PIPE).strip())
        return sha, dirty
    except (OSError, subprocess.SubprocessError):
        receipt = os.environ.get("RX_SOURCE_PROVENANCE") or str(Path(root).parent / "prov.json")
        provenance = json.loads(Path(receipt).read_text(encoding="utf-8"))
        sha, diff = provenance.get("head"), provenance.get("dirty")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha) or not (
                diff == "clean" or isinstance(diff, str) and re.fullmatch(r"[0-9a-f]{12}", diff)):
            raise ValueError(f"invalid rx source provenance: {receipt}")
        return sha, diff != "clean"


def _artifact_files(artifact):
    result = {}
    for row in artifact.get("files", []):
        name = _identifier(row["destination_path"], "artifact destination")
        result[name] = _identifier(row["sha256"], f"artifact sha256 for {name}")
    return result




def _resolve_artifacts(recipe, host=None, artifacts_dir=DEFAULT_ARTIFACTS_DIR, artifacts=None,
                       bundle_override=None, bundle_pin=None):
    """Resolve the recipe's artifact keys to directories.

    Returns (paths, expected run-manifest shas, pin payload, sources).  ``sources`` names where
    each path came from: a HOST_VARIABLES entry, a --host-values artifact, or the fetch directory.
    """
    host = no_host_values() if host is None else host
    artifacts = _manifest_artifacts() if artifacts is None else artifacts
    target_value, target_source = host_variable("JETSPEC_TARGET_MODEL_PATH", host)
    target = Path(target_value)
    sources = {"target": target_source}
    if not target.is_dir():
        raise FileNotFoundError(
            f"missing recipe artifact(s): target model {target} ({target_source})")
    if recipe["method"] in ("ar", "jetspec.engines.jetspec_plus_plus.serving.api", "spec_sync"):
        return {
            "adapter": None, "drafter": None, "bundle": None,
            "target": target, "compile_pin": None, "blockhead": None,
        }, {}, None, sources
    if recipe["method"] == "eagle3":
        key = recipe["draft_artifact"]
        if key not in artifacts:
            raise ValueError(f"recipe names absent artifacts/manifest.json key {key!r}")
        declared = recipe["draft_file_sha256"]
        actual = _artifact_files(artifacts[key])
        if actual != declared:
            raise ValueError(f"eagle3 recipe pins disagree with manifest entry {key!r}")
        draft, sources["draft"] = artifact_directory(key, artifacts, host, artifacts_dir)
        assert_artifact_present(key, artifacts, draft, sources["draft"])
        for name, sha in declared.items():
            found = _sha256(draft / name)
            if found != sha:
                raise ValueError(f"eagle3 draft sha256 mismatch {name}: {found} != {sha}")
        return {
            "adapter": None, "drafter": None, "bundle": None,
            "target": target, "compile_pin": None, "blockhead": None, "draft": draft.resolve(),
        }, {f"draft/{name}": sha for name, sha in declared.items()}, None, sources
    if recipe["method"] in ("blockhead", "dflash"):
        key = recipe.get("blockhead_artifact")
        if not isinstance(key, str) or not key:
            raise ValueError("blockhead recipe must name blockhead_artifact (a manifest key)")
        declared = recipe.get("blockhead_file_sha256")
        if not isinstance(declared, dict) or not declared:
            raise ValueError("blockhead recipe must declare blockhead_file_sha256")
        checkpoint, sources["blockhead"] = artifact_directory(key, artifacts, host, artifacts_dir)
        manifest_shas = _artifact_files(artifacts[key])
        if manifest_shas != declared:
            raise ValueError(
                f"blockhead recipe pins disagree with artifacts/manifest.json entry {key!r}: "
                f"recipe {sorted(declared.items())} != manifest {sorted(manifest_shas.items())}")
        assert_artifact_present(key, artifacts, checkpoint, sources["blockhead"])
        for name, expected_sha in sorted(declared.items()):
            actual_sha = _sha256(checkpoint / name)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"blockhead recipe sha256 mismatch {name}: {actual_sha} != {expected_sha}")
        return {
            "adapter": None, "drafter": None, "bundle": None,
            "target": target, "compile_pin": None, "blockhead": checkpoint,
        }, {f"blockhead/{name}": sha for name, sha in declared.items()}, None, sources
    if recipe.get("environment", {}).get("JETSPEC_HYBRID_PAIR") == "1":
        return _resolve_hybrid_placeholder(recipe, target, host, artifacts_dir,
                                           artifacts, sources, bundle_override, bundle_pin)
    drafter_key = recipe["drafter_artifact"]
    adapter_key = recipe.get("adapter_artifact", "adapter_head")
    release_key = recipe["target_release_artifact"]
    for key in (adapter_key, drafter_key, release_key, recipe["bundle_artifact"]):
        if key not in artifacts:
            raise ValueError(f"recipe names absent artifacts/manifest.json key {key!r}")
    adapter, sources["adapter"] = artifact_directory(adapter_key, artifacts, host, artifacts_dir)
    drafter, sources["drafter"] = artifact_directory(
        drafter_key, artifacts, host, artifacts_dir)
    release, sources["target_release"] = artifact_directory(
        release_key, artifacts, host, artifacts_dir)
    declared_bundle, sources["bundle"] = artifact_directory(
        recipe["bundle_artifact"], artifacts, host, artifacts_dir)
    # The recipe may serve a sibling operating point of the published bundle (same layout,
    # rewritten thresholds.json), named beside it.
    bundle = declared_bundle if host["artifacts"].get(recipe["bundle_artifact"]) else declared_bundle.with_name(recipe["bundle_name"])
    if bundle_override is not None:
        bundle = Path(bundle_override)
        sources["bundle"] = "explicit_bundle_override"
    compile_key = recipe.get("compile_pin_artifact")
    compile_pin = None
    if recipe["regime"] == "graphed" and compile_key is not None:
        if compile_key not in artifacts:
            raise ValueError("graphed recipe does not resolve compile_pin artifact")
        compile_pin, sources["compile_pin"] = artifact_directory(
            compile_key, artifacts, host, artifacts_dir)
    paths = {
        "adapter": adapter, "drafter": drafter, "bundle": bundle,
        "target": target, "target_release": release / "target_release_weights.safetensors",
        "compile_pin": compile_pin, "blockhead": None,
    }
    assert_artifact_present(adapter_key, artifacts, adapter, sources["adapter"])
    assert_artifact_present(drafter_key, artifacts, drafter, sources["drafter"])
    assert_artifact_present(release_key, artifacts, release, sources["target_release"])
    legacy_kpred = recipe.get("environment", {}).get("JETSPEC_LEGACY_KPRED_FORWARD", "0") == "1"
    bundle_key = recipe["bundle_artifact"]
    bundle_artifact = artifacts[bundle_key]
    if not legacy_kpred:
        bundle_artifact = {**bundle_artifact, "files": [
            entry for entry in bundle_artifact["files"]
            if not _identifier(entry["destination_path"], f"{bundle_key} file destination")
            .startswith("kpred/")]}
    assert_artifact_present(bundle_key, {**artifacts, bundle_key: bundle_artifact},
                            bundle, sources["bundle"])
    target_slice = False
    if bundle_override is not None:
        if not isinstance(bundle_pin, dict) or bundle_pin.get("schema") != "eng04.ablation_bundle_pin/1":
            raise ValueError("BUNDLE_PIN_REQUIRED")
        target_slice = bool(bundle_pin.get("target_files"))
        declared = bundle_pin.get("nondecoder_sha256")
        if not isinstance(declared, dict):
            raise ValueError("BUNDLE_PIN_INVALID")
        original = _artifact_files(artifacts[bundle_key])
        if set(declared) != set(original) - {"decoder/checkpoint-final-model.pt"}:
            raise ValueError("BUNDLE_NONDECODER_SET_MISMATCH")
        for name, digest in declared.items():
            if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    or not (bundle / name).is_file() or _sha256(bundle / name) != digest):
                raise ValueError(f"BUNDLE_SHA_MISMATCH: {name}")
            if name != "thresholds.json" and original.get(name) != digest:
                raise ValueError(f"BUNDLE_NONDECODER_UNPINNED: {name}")
        decoder_files = (("decoder/checkpoint-final-model.pt", bundle_pin.get("decoder_file_sha256")),
                         ("decoder/checkpoint-0.pt", bundle_pin.get("decoder_sha256")),
                         ("decoder/manifest.json", bundle_pin.get("decoder_manifest_sha256"))) if target_slice else (
                             ("decoder/checkpoint-final-model.pt", bundle_pin.get("decoder_sha256")),)
        for name, digest in (*decoder_files, ("thresholds.json", bundle_pin.get("thresholds_sha256"))):
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"BUNDLE_PIN_INVALID: {name}")
            if _sha256(bundle / name) != digest:
                raise ValueError(f"BUNDLE_SHA_MISMATCH: {name}")
        if target_slice:
            manifest = json.loads((bundle / "decoder/manifest.json").read_text())
            expected_manifest = {"source": "target_slice", "target_files": bundle_pin["target_files"]}
            if "fallback_sha256" in manifest:
                expected_manifest["fallback_sha256"] = bundle_pin["decoder_sha256"]
            if manifest != expected_manifest:
                raise ValueError("BUNDLE_TARGET_SLICE_MANIFEST_MISMATCH")
            for name, digest in bundle_pin["target_files"].items():
                if (Path(name).is_absolute() or '..' in Path(name).parts
                        or _sha256(target / name) != digest):
                    raise ValueError(f"BUNDLE_TARGET_SHA_MISMATCH: {name}")
    if compile_pin is not None:
        assert_artifact_present(compile_key, artifacts, compile_pin, sources["compile_pin"])
    expected = {
        "adapter/config": _artifact_files(artifacts[adapter_key])["config.json"],
        "adapter/head": _artifact_files(artifacts[adapter_key])["head.safetensors"],
        "heads/thresholds": recipe["thresholds_sha256"],
    }
    drafter_files = _artifact_files(artifacts[drafter_key])
    for name in ("config.json", "dflash.py", "model.safetensors"):
        expected[f"drafter/{name}"] = drafter_files[name]
    release_files = _artifact_files(artifacts[release_key])
    expected["target_release_weights"] = release_files["target_release_weights.safetensors"]
    bundle_files = _artifact_files(bundle_artifact)
    decoder_source = None
    if manifest_sha := bundle_files.get("decoder/manifest.json"):
        decoder_manifest = bundle / "decoder/manifest.json"
        actual = _sha256(decoder_manifest)
        if actual != manifest_sha:
            raise ValueError(f"decoder manifest sha256 mismatch: {actual} != {manifest_sha}")
        decoder_spec = json.loads(decoder_manifest.read_text(encoding="utf-8"))
        decoder_source = decoder_spec.get("source")
        expected["heads/decoder_manifest"] = manifest_sha
        if decoder_source == "target_slice":
            if "decoder/checkpoint-0.pt" not in bundle_files:
                raise ValueError("target-slice decoder lacks pinned fallback")
            if (decoder_spec.get("fallback_sha256") is not None
                    and decoder_spec["fallback_sha256"] != bundle_files["decoder/checkpoint-0.pt"]):
                raise ValueError("target-slice decoder fallback manifest sha256 mismatch")
            expected["heads/decoder_fallback"] = bundle_files["decoder/checkpoint-0.pt"]
            target_files = decoder_spec.get("target_files")
            if not isinstance(target_files, dict) or not target_files:
                raise ValueError("target-slice decoder lacks target file pins")
            for name, sha256 in target_files.items():
                expected[f"heads/decoder_target/{name}"] = sha256
        elif decoder_source not in ("file",):
            raise ValueError(f"unknown decoder source: {decoder_source}")
    for name, sha256 in bundle_files.items():
        if name == "decoder/checkpoint-final-model.pt" and decoder_source is None:
            expected["heads/decoder"] = sha256
        elif name == "decoder/checkpoint-0.pt" and decoder_source == "file":
            expected["heads/decoder"] = sha256
        elif name.startswith("kpred/arms/FR/") and legacy_kpred:
            expected[f"heads/kpred/{name.split('/arms/FR/', 1)[1].replace('/ckpt.pt', '')}"] = sha256
    if bundle_override is not None:
        if target_slice:
            expected.pop("heads/decoder", None)
            expected["heads/decoder_fallback"] = bundle_pin["decoder_sha256"]
            expected["heads/decoder_manifest"] = bundle_pin["decoder_manifest_sha256"]
            for name, digest in bundle_pin["target_files"].items():
                expected[f"heads/decoder_target/{name}"] = digest
        else:
            expected["heads/decoder"] = bundle_pin["decoder_sha256"]
        expected["heads/thresholds"] = bundle_pin["thresholds_sha256"]
    pin_payload = {"checkpoints": {
        "adapter_head": {"directory": str(adapter), "files": {
            name: {"sha256": value} for name, value in _artifact_files(
                artifacts[adapter_key]).items()}},
        "drafter_checkpoint": {"directory": str(drafter), "files": {
            name: {"sha256": value} for name, value in drafter_files.items()}},
        "target_release": {"directory": str(release), "files": {
            name: {"sha256": value} for name, value in release_files.items()}},
        "served_heads": {"directory": str(bundle), "files": {
            name: {"sha256": recipe["thresholds_sha256"] if name == "thresholds.json" else value}
            for name, value in bundle_files.items()}},
    }}
    if bundle_override is not None:
        files = pin_payload["checkpoints"]["served_heads"]["files"]
        files["decoder/checkpoint-final-model.pt"]["sha256"] = (
            bundle_pin["decoder_file_sha256"] if target_slice else bundle_pin["decoder_sha256"])
        if target_slice:
            files["decoder/checkpoint-0.pt"] = {"sha256": bundle_pin["decoder_sha256"]}
            files["decoder/manifest.json"] = {"sha256": bundle_pin["decoder_manifest_sha256"]}
        files["thresholds.json"]["sha256"] = bundle_pin["thresholds_sha256"]
    return paths, expected, pin_payload, sources


def _generic_draft_path(cache):
    base = Path(cache) / "models--Qwen--Qwen3-0.6B"
    candidates = [base]
    if (base / "snapshots").is_dir():
        candidates.extend(sorted((base / "snapshots").iterdir()))
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (
                (candidate / "model.safetensors").is_file()
                or (candidate / "model.safetensors.index.json").is_file()):
            return candidate.resolve()
    raise FileNotFoundError(f"generic Qwen3-0.6B draft is absent under {base}")






def _gpu_uuid(device):
    return subprocess.check_output([
        "nvidia-smi", "-i", device, "--query-gpu=uuid", "--format=csv,noheader"],
        text=True, timeout=30).strip()


def _gpu_uuids(devices, *, required=True):
    """Device identities; ``required=False`` (--resolve-only) records None where nvidia-smi is absent."""
    try:
        return [_gpu_uuid(device) for device in devices]
    except (OSError, subprocess.SubprocessError):
        if required:
            raise
        return None


def _gpu_census(uuids, allowed_pids=()):
    gpu_output = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,utilization.gpu,utilization.memory",
        "--format=csv,noheader"], text=True, timeout=30)
    app_output = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
        "--format=csv,noheader"], text=True, timeout=30)
    gpus = []
    for line in gpu_output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise RuntimeError(f"malformed GPU census row: {line!r}")
        gpus.append({
            "index": int(fields[0]), "uuid": fields[1], "used_memory": fields[2],
            "gpu_utilization": fields[3], "memory_utilization": fields[4],
        })
    applications = []
    for line in app_output.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            raise RuntimeError(f"malformed compute-app census row: {line!r}")
        applications.append({
            "gpu_uuid": fields[0], "pid": int(fields[1]),
            "used_memory": fields[2], "process_name": fields[3],
        })
    allowed = {int(pid) for pid in allowed_pids}
    other_applications = [app for app in applications if app["pid"] not in allowed]
    occupied = [app for app in other_applications if app["gpu_uuid"] in uuids]
    if occupied:
        raise RuntimeError("requested device occupied: " + " | ".join(
            f'{app["gpu_uuid"]}, {app["pid"]}, {app["used_memory"]}' for app in occupied))
    census = {
        "captured_unix": time.time(), "requested_uuids": list(uuids),
        "load_label": "quiet" if not other_applications else "loaded",
        "gpus": gpus, "compute_apps": applications,
    }
    print("DEVICE_CENSUS " + json.dumps(census, sort_keys=True), flush=True)
    return census


class MPSControl:
    """One foreground control child and its explicitly started MPS server."""
    def __init__(self, root, gpu_uuid):
        self.root = Path(root)
        self.gpu_uuid = gpu_uuid
        self.pipe = self.root / "pipe"
        self.logs = self.root / "log"
        self.process = self.output = self.server_pid = None
        self.environment = None
        self._old_sigterm = None
        self._owns_pipe = False





    def command(self, command):
        done = subprocess.run(
            ["nvidia-cuda-mps-control"], input=command + "\n", env=self.environment,
            text=True, capture_output=True, timeout=20)
        if done.returncode:
            raise RuntimeError(f"MPS command failed {command!r}: {done.stderr}")
        return done.stdout.strip()

    def __enter__(self):
        self.refuse_existing_daemon(self.gpu_uuid)
        self._old_sigterm = signal.signal(signal.SIGTERM, self._term)
        try:
            self.pipe.mkdir(parents=True, exist_ok=False)
            self._owns_pipe = True
            self.logs.mkdir()
            self.environment = dict(os.environ)
            # Client-side MPS knobs act on the process that opens a CUDA context; in the
            # control daemon's environment they would reach every client it serves.
            for key in MPS_CLIENT_ONLY_ENVIRONMENT:
                self.environment.pop(key, None)
            self.environment.update(CUDA_VISIBLE_DEVICES=self.gpu_uuid,
                                    CUDA_MPS_PIPE_DIRECTORY=str(self.pipe),
                                    CUDA_MPS_LOG_DIRECTORY=str(self.logs))
            self.output = (self.root / "foreground-control.log").open("x")
            self.process = subprocess.Popen(
                ["nvidia-cuda-mps-control", "-f"], env=self.environment,
                stdout=self.output, stderr=subprocess.STDOUT, text=True)
            for _ in range(200):
                if (self.pipe / "control").exists():
                    break
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"MPS foreground control exited during startup rc={self.process.returncode} "
                        f"log={(self.root / 'foreground-control.log').read_text(errors='replace')!r}")
                time.sleep(0.1)
            else:
                raise RuntimeError("MPS control socket did not appear")
            self.command(f"start_server -uid {os.getuid()}")
            for _ in range(200):
                servers = self.command("get_server_list")
                if servers:
                    self.server_pid = int(servers.splitlines()[-1])
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError("MPS server did not become ready")
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise


    def __exit__(self, exc_type, exc, tb):
        release_error = None
        try:
            if self.process is not None and self.process.poll() is None:
                self.command("quit")
                self.process.wait(timeout=20)
        except Exception as error:
            release_error = error
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=10)
        finally:
            if self.output is not None:
                self.output.close()
            if self._owns_pipe:
                try:
                    for entry in self.pipe.iterdir():
                        entry.unlink()
                    self.pipe.rmdir()
                except Exception as error:
                    release_error = release_error or error
            if self._old_sigterm is not None:
                signal.signal(signal.SIGTERM, self._old_sigterm)
        if self.process is not None and self.process.poll() is None:
            release_error = release_error or RuntimeError("MPS control child is still alive")
        if release_error is not None and exc is None:
            raise release_error
        return False


MPS_CLIENT_ONLY_ENVIRONMENT = (
    "CUDA_MPS_CLIENT_PRIORITY", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
    "JETSPEC_MPS_DRAFT_PRIORITY", "JETSPEC_MPS_DRAFT_ACTIVE_THREAD_PERCENTAGE")
# sun_path is 108 bytes on Linux, including the terminating NUL.
MPS_SOCKET_PATH_LIMIT = 107
# The longest socket nvidia-cuda-mps-control binds in CUDA_MPS_PIPE_DIRECTORY.
MPS_LONGEST_SOCKET = "control_privileged"


def private_mps_root(base, gpu_uuid, output):
    # AF_UNIX socket paths are short on Linux; a full UUID and output name can
    # make nvidia-cuda-mps-control exit before it writes a diagnostic.  The key is
    # fresh per launch so a retry of the same output never meets a stale log dir.
    nonce = f"{os.getpid()}:{time.time_ns()}:{os.urandom(8).hex()}"
    key = hashlib.sha256(f"{gpu_uuid}:{output}:{nonce}".encode()).hexdigest()[:16]
    root = Path(os.path.abspath(base)) / key
    socket_path = root / "pipe" / MPS_LONGEST_SOCKET
    length = len(str(socket_path).encode())
    if length > MPS_SOCKET_PATH_LIMIT:
        room = MPS_SOCKET_PATH_LIMIT - (length - len(str(Path(os.path.abspath(base))).encode()))
        raise ValueError(
            f"--mps-root is too long for private MPS sockets: {socket_path} is {length} bytes, "
            f"the Unix socket limit is {MPS_SOCKET_PATH_LIMIT}; pass an absolute --mps-root of "
            f"at most {room} bytes")
    return root


def _parse_deadlines(log):
    phases = {}
    active = None
    for line in log.splitlines():
        if line.startswith("SERVE_READ_PHASE_START "):
            active = line.split(" ", 1)[1]
            phases[active] = []
        elif line.startswith("SERVE_READ_PHASE_END "):
            active = None
        elif active is not None and "W320_DEADLINE " in line:
            # Rank-local stdout may append another process's line to this JSON record.
            # Accept one complete JSON prefix, but still fail if that prefix is malformed.
            payload = line.split("W320_DEADLINE ", 1)[1]
            record, _ = json.JSONDecoder().raw_decode(payload)
            phases[active].append(record)
    return phases


def chain_proposals(log):
    """Join chain publications using host-visible rounds without inventing row counts."""
    rows = []
    phased = "SERVE_READ_PHASE_START " in log
    active = None
    for line in log.splitlines():
        if line.startswith("SERVE_READ_PHASE_START "):
            active = line.split(" ", 1)[1]
            continue
        if line.startswith("SERVE_READ_PHASE_END "):
            active = None
            continue
        if phased and active != "measured":
            continue
        for marker in ("CHAIN_ROUND ", "CHAIN_FALLBACK "):
            if marker not in line:
                continue
            record, _ = json.JSONDecoder().raw_decode(line.split(marker, 1)[1])
            config = record.get("config", "capacity_fallback")
            candidate_round = record.get("published_round", record["round"])
            rows.append({"seq_id": record["seq_id"],
                         "candidate_round": candidate_round,
                         "pass_label": config, "rows_published": None,
                         "rows_requested": None, "rows_truncated": None,
                         "refresh_calls": int(config.endswith("-R1")), "chain": True})
            break
    return rows


def chain_t0_scope(recipe):
    """27B sm80 checks every replay in the requested read, including smoke."""
    if (recipe.get("cuda_arch") == "8.0"
            and recipe["environment"].get("JETSPEC_HYBRID_PAIR") == "1"
            and recipe["environment"].get("JETSPEC_CHAIN_GRAPH") == "seg"):
        return "read"
    return "sweep"


def chain_round_summary(log, verify_metrics, *, require_t0=False, t0_scope="sweep"):
    """Audit the served chain's single sync and expose its launch mix and LATEs."""
    records = []
    fallbacks = []
    phased = "SERVE_READ_PHASE_START " in log
    active = None
    for line in log.splitlines():
        if line.startswith("SERVE_READ_PHASE_START "):
            active = line.split(" ", 1)[1]
            continue
        if line.startswith("SERVE_READ_PHASE_END "):
            active = None
            continue
        if "W320_FALLBACK " in line:
            raise RuntimeError("chain read entered W320_FALLBACK")
        if phased and active != "measured":
            continue
        if "CHAIN_ROUND " in line:
            payload = line.split("CHAIN_ROUND ", 1)[1]
            record, _ = json.JSONDecoder().raw_decode(payload)
            records.append(record)
        if "CHAIN_FALLBACK " in line:
            payload = line.split("CHAIN_FALLBACK ", 1)[1]
            record, _ = json.JSONDecoder().raw_decode(payload)
            fallbacks.append(record)
    if not records:
        raise RuntimeError("chain read emitted no CHAIN_ROUND records")
    previous = {}
    mix = {}
    redraft_rounds = 0
    syncs = 0
    tagged = any("redraft" in record for record in records)
    for record in records:
        seq_id = record["seq_id"]
        redrafted = bool(record.get("redraft", False))
        expected_syncs = 2 if redrafted else 1
        if record["sync_total"] - previous.get(seq_id, 0) != expected_syncs:
            if tagged:
                raise RuntimeError("chain round sync count differs from hit/miss protocol")
            raise RuntimeError("chain read did not use exactly one publish sync per round")
        previous[seq_id] = record["sync_total"]
        redraft_rounds += int(redrafted)
        syncs += expected_syncs
        config = record["config"]
        mix[config] = mix.get(config, 0) + 1
    if require_t0 and t0_scope == "sweep" and (len(records) < 300
                       or not any(record["L"] <= 768 for record in records)
                       or not any(record["L"] >= 1264 for record in records)):
        raise RuntimeError("chain T0 needs 300 graph rounds across L=768 and L=1264")
    summary = {"rounds": len(records) + len(fallbacks),
               "graph_rounds": len(records), "publish_syncs": syncs,
               "syncs_per_round": ("hit=1,miss=2" if tagged else 1),
               "late_count": int(verify_metrics["jetspec_late_count"]),
               "w320_fallback": 0, "mix": mix,
               "shadow_checked_rounds": len(records) if require_t0 else 0,
               "capacity_fallback": sum(record["reason"] == "capacity"
                                        for record in fallbacks)}
    if tagged:
        summary["redraft_rounds"] = redraft_rounds
        # 5f: a verifier valve timeout (expected 0) and a redraft delivered but
        # not waited for (expected 0); both are printed with the invariants.
        summary["redraft_missing"] = int(verify_metrics.get("jetspec_chain_redraft_missing", 0))
        summary["redraft_unused"] = int(verify_metrics.get("jetspec_chain_redraft_unused", 0))
    if require_t0 and t0_scope == "read":
        # ChainGraph still compares every replay's outputs/wire/committed IDs
        # with its independent eager shadow using the existing exact checks.
        summary["t0_scope"] = "read"
    lite = [[record['seq_id'], record['published_round'], *record['lite']]
            for record in records if 'lite' in record]
    if lite:
        from jetspec.engines.jetspec_plus_plus.draft.anatomy_lite import pack
        summary['anatomy_lite'] = pack(lite)
    return summary


def apply_chain_miss_redraft_override(environment, choice, max_ms, timing=None):
    if choice is None and max_ms is None and timing is None:
        return
    if environment.get("JETSPEC_CHAIN_GRAPH") != "seg":
        raise ValueError("--chain-miss-redraft requires a segmented chain recipe")
    if choice is not None:
        environment["JETSPEC_CHAIN_MISS_REDRAFT"] = "1" if choice == "on" else "0"
    if max_ms is not None:
        if not 0 < max_ms < float("inf"):
            raise ValueError("--chain-miss-redraft-max-ms must be finite and positive")
        environment["JETSPEC_CHAIN_MISS_REDRAFT_MAX_MS"] = str(max_ms)
    if timing is not None:
        environment["JETSPEC_CHAIN_MISS_REDRAFT_TIMING"] = "1" if timing == "on" else "0"


def apply_chain_check_override(environment, choice):
    if choice is None:
        return
    if environment.get("JETSPEC_CHAIN_GRAPH") != "seg":
        raise ValueError("--chain-check requires a segmented chain recipe")
    environment["JETSPEC_CHAIN_GRAPH_CHECK"] = "1" if choice == "on" else "0"


def apply_chain_warm_override(environment, reachable, stamps):
    if reachable is None and stamps is None:
        return
    if environment.get("JETSPEC_CHAIN_GRAPH") != "seg":
        raise ValueError("chain warm overrides require a segmented chain recipe")
    if reachable is not None:
        environment["JETSPEC_CHAIN_WARM_REACHABLE"] = "1" if reachable == "on" else "0"
    if stamps is not None:
        environment["JETSPEC_CHAIN_CAPTURE_STAMPS"] = "1" if stamps == "on" else "0"


def apply_chain_decoder_key_bucket_override(environment, choice):
    if choice is None:
        return
    if environment.get("JETSPEC_CHAIN_GRAPH") != "seg":
        raise ValueError("--chain-decoder-key-bucket requires a segmented chain recipe")
    environment["JETSPEC_CHAIN_DECODER_KEY_BUCKET"] = "1" if choice == "on" else "0"


def apply_chain_sega_glue_override(environment, choice):
    if choice is None:
        return
    if environment.get("JETSPEC_CHAIN_GRAPH") != "seg":
        raise ValueError("--chain-sega-glue requires a segmented chain recipe")
    environment["JETSPEC_CHAIN_SEGA_GLUE"] = choice


def apply_chain_glue2_profile_override(environment, choice):
    if choice is None:
        return
    if environment.get("JETSPEC_CHAIN_GRAPH") != "seg":
        raise ValueError("--chain-glue2-profile requires a segmented chain recipe")
    environment["JETSPEC_CHAIN_GLUE2_PROFILE"] = "1" if choice == "on" else "0"


def apply_chain_refine_k3_override(environment, choice):
    if choice is None:
        return
    if environment.get("JETSPEC_CHAIN_GRAPH") != "seg":
        raise ValueError("--chain-refine-k3 requires a segmented chain recipe")
    environment["JETSPEC_REFINE_K3"] = "1" if choice == "on" else "0"


def apply_chain_handoff_override(environment, choice, timing):
    if choice is None and not timing:
        return
    if environment.get("JETSPEC_CHAIN_GRAPH") != "seg":
        raise ValueError("--chain-handoff requires a segmented chain recipe")
    if choice == "on" and environment.get("JETSPEC_CHAIN_FIXED_CONFIG") not in ("J1-R0", "J3-R0"):
        raise ValueError("--chain-handoff on requires a supported fixed R0 cell")
    if choice is not None:
        environment["JETSPEC_CHAIN_HANDOFF"] = "1" if choice == "on" else "0"
    if timing:
        environment["JETSPEC_CHAIN_HANDOFF_TIMING"] = "1"


def apply_chain_prepost_override(environment, choice, timing):
    if choice is None and not timing:
        return
    if environment.get("JETSPEC_CHAIN_GRAPH") != "seg":
        raise ValueError("--chain-prepost requires a segmented chain recipe")
    if choice == "on" and not environment.get("JETSPEC_CHAIN_FIXED_CONFIG"):
        raise ValueError("--chain-prepost on requires a fixed chain cell")
    if choice is not None:
        environment["JETSPEC_CHAIN_PREPOST"] = "1" if choice == "on" else "0"
    if timing:
        environment["JETSPEC_CHAIN_PREPOST_TIMING"] = "1"


VERIFY_LEAN_KNOBS = ("JETSPEC_VERIFY_STAGE_HOST", "JETSPEC_CANDIDATE_DECODE_LEAN", "JETSPEC_VERIFY_SELECT_LEAN",
                     "JETSPEC_OUTCOME_WIRE_RING", "JETSPEC_CHAIN_CMD_LOCAL_COUNTER", "JETSPEC_VERIFY_STAGE_EARLY")


def apply_verify_lean_override(environment, choice, stage_early=None):
    """--verify-lean sets the six verifier host cuts together; --verify-lean-stage-early overrides the
    one that reorders staging before the candidate wait."""
    if choice is None and stage_early is None:
        return
    if "on" in (choice, stage_early) and environment.get("JETSPEC_VERIFY_DEVICE") != "1":
        raise ValueError("--verify-lean on requires a device-verify recipe")
    if choice is not None:
        for name in VERIFY_LEAN_KNOBS:
            environment[name] = "1" if choice == "on" else "0"
    if stage_early is not None:
        environment["JETSPEC_VERIFY_STAGE_EARLY"] = "1" if stage_early == "on" else "0"


def apply_knob_overrides(environment, knobs):
    """--knob NAME=VALUE sets one registered knob for this read (repeatable; applied after the named
    overrides). The name must be registered in draft/env.py; the child validates the requirements."""
    if not knobs:
        return
    from jetspec.engines.jetspec_plus_plus.draft.env import REGISTRY
    for item in knobs:
        name, sep, value = item.partition("=")
        if not sep or not name or value == "":
            raise ValueError(f"--knob expects NAME=VALUE, got {item!r}")
        if name not in REGISTRY:
            raise ValueError(f"--knob {name} is not a registered knob (draft/env.py)")
        environment[name] = value


def apply_verify_host_census_override(environment, choice):
    if choice is None:
        return
    if choice == "on" and environment.get("JETSPEC_VERIFY_DEVICE") != "1":
        raise ValueError("--verify-host-census on requires a device-verify recipe")
    environment["JETSPEC_VERIFY_HOST_CENSUS"] = "1" if choice == "on" else "0"


def apply_chain_tail_trace_override(environment, enabled, kernel_profile=None, every=None):
    if not enabled and kernel_profile is None:
        return
    if environment.get("JETSPEC_CHAIN_GRAPH") != "seg":
        raise ValueError("--chain-tail-trace requires a segmented chain recipe")
    if enabled:
        environment["JETSPEC_CHAIN_TAIL_TRACE"] = "1"
    if kernel_profile is not None:
        environment["JETSPEC_CHAIN_KERNEL_PROFILE"] = str(Path(kernel_profile).resolve())
        environment["JETSPEC_CHAIN_KERNEL_PROFILE_EVERY"] = str(every or 20)


def apply_chain_attn3_override(environment, choice):
    if choice is None:
        return
    if environment.get("JETSPEC_CHAIN_GRAPH") != "seg":
        raise ValueError("--chain-attn3 requires a segmented chain recipe")
    environment["JETSPEC_CHAIN_ATTN3"] = "1" if choice == "on" else "0"


def apply_chain_k3_override(environment, choice):
    if choice is None:
        return
    if environment.get("JETSPEC_CHAIN_GRAPH") != "seg":
        raise ValueError("--chain-k3 requires a segmented chain recipe")
    value = "1" if choice == "on" else "0"
    environment["JETSPEC_ROWS_K3"] = value
    environment["JETSPEC_REFINE_K3"] = value


def _declared_flashinfer_matches(declared):
    from importlib import metadata
    try:
        return bool(declared) and declared == metadata.version('flashinfer-python')
    except metadata.PackageNotFoundError:
        return False


def _base_environment(recipe, placement_environment, runtime, *, eager_wrapper=False):
    sys.path.insert(0, str(ROOT / "serve"))
    from jetspec.engines.jetspec_plus_plus.draft.env import REGISTRY; from jetspec.engines.jetspec_plus_plus.draft.env import unregistered_names
    inherited_unknown = sorted(unregistered_names(os.environ))
    if inherited_unknown:
        raise ValueError("unknown inherited serving knob(s): " + ", ".join(inherited_unknown))
    result = dict(os.environ)
    for key in list(result):
        if key in REGISTRY or EXPERIMENT_NAME.match(key):
            result.pop(key)
    for key in ("TORCHDYNAMO_DISABLE", "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY",
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "CUDA_MPS_CLIENT_PRIORITY"):
        result.pop(key, None)
    result.update(recipe["environment"])
    # Observation has an environment kill switch even when a recipe is sealed.
    if 'JETSPEC_ANATOMY_LITE' in os.environ:
        result['JETSPEC_ANATOMY_LITE'] = os.environ['JETSPEC_ANATOMY_LITE']
    result.update(placement_environment)
    result.update({
        "SSD_HF_CACHE": runtime["hf_cache"], "SSD_DATASET_DIR": runtime["dataset_dir"],
        "SSD_CUDA_ARCH": runtime["cuda_arch"],
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
        "TORCHINDUCTOR_COMPILE_THREADS": "2",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
    })
    paths = [entry for entry in result.get("PYTHONPATH", "").split(os.pathsep) if entry]
    if str(METHOD_ROOT) not in paths:
        result["PYTHONPATH"] = os.pathsep.join([str(METHOD_ROOT), *paths])
    if eager_wrapper:
        result["SSD_EAGER_VERIFY_WRAPPER"] = "1"
    cache = Path(runtime["cache"])
    cache.mkdir(parents=True, exist_ok=True)
    cluster_flashinfer = result.get('FLASHINFER_WORKSPACE_BASE')
    result.update(TORCH_EXTENSIONS_DIR=str(cache / "torch_extensions"),
                  TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
                  TRITON_CACHE_DIR=str(cache / "triton"),
                  SSD_CUTE_DSL_CACHE_ROOT=str(cache / "cute"),
                  FLASHINFER_WORKSPACE_BASE=str(cache / "flashinfer"))
    result.update(_persistent_cache_environment(recipe, cache))
    if os.environ.get('CRUCIBLE_L4_ROOT'):
        from jetspec.engines.jetspec_plus_plus.serving.tools.w632_measurement.cache import configure; from jetspec.engines.jetspec_plus_plus.serving.tools.w632_measurement.cache import cleanup_locks
        cache_environment = dict(result)
        if cluster_flashinfer:
            cache_environment['FLASHINFER_WORKSPACE_BASE'] = cluster_flashinfer
        receipt = configure(ROOT, cache, cache_environment)
        result.update(receipt['environment'])
        if receipt['persistent']:
            receipt['removed_locks'] = cleanup_locks(receipt['root'])
        result['CRUCIBLE_CACHE_RECEIPT'] = json.dumps(receipt, sort_keys=True)
    elif cluster_flashinfer and _declared_flashinfer_matches(os.environ.get('CRUCIBLE_FLASHINFER_VERSION')):
        # A declared FlashInfer cache of the installed version is kept,
        # as the persistent-cache branch above does; other per-read caches stay fresh.
        result['FLASHINFER_WORKSPACE_BASE'] = cluster_flashinfer
    if os.environ.get('SERVE_READ_STARTUP_TIMING') == '1':
        result['SERVE_READ_STARTUP_TIMING'] = '1'
    return result


def compile_pin_environment(args):
    pin, sha = getattr(args, "compile_pin", None), getattr(args, "compile_pin_sha", None)
    if bool(pin) != bool(sha):
        raise ValueError("--compile-pin and --compile-pin-sha must be paired")
    if pin is None:
        return {}
    return {"JETSPEC_C1_COMPILE_PIN": str(Path(pin).resolve()),
            "JETSPEC_C1_COMPILE_PIN_SHA": sha}


def _recipe_concurrency(recipe, max_num_seqs):
    # A batch_only recipe never serves one sequence; a caller that omits its concurrency
    # must not put it under the N=1 miss-fallback policy.
    if recipe.get("batch_only") and max_num_seqs == 1:
        return None
    return max_num_seqs


def _environment(recipe, placement, paths, runtime, pins_path, records, *,
                 max_num_seqs=1, resolve_only=False):
    result = _base_environment(recipe, placement["environment"], runtime, eager_wrapper=True)
    if runtime.get("graphed_ar_disable_dynamo"):
        result["TORCHDYNAMO_DISABLE"] = "1"
    from jetspec.engines.jetspec_plus_plus.draft.env import validate_environment
    if recipe["method"] == "jetspecpp":
        result.update({
            "JETSPEC_DATASET": runtime["dataset"],
            "JETSPEC_ADAPTER_HEAD": str(paths["adapter"] / "head.safetensors"),
            "JETSPEC_ADAPTER_CONFIG": str(paths["adapter"]),
            "JETSPEC_CHECKPOINT_DIR": str(paths["drafter"]),
            "JETSPEC_HEADS_DIR": str(paths["bundle"]),
            "JETSPEC_TARGET_MODEL_PATH": str(paths["target"]),
            "JETSPEC_TARGET_RELEASE_WEIGHTS": str(paths["target_release"]),
            "JETSPEC_PINS": str(pins_path),
        })
    elif recipe["method"] in ("blockhead", "dflash"):
        result["JETSPEC_BLOCKHEAD"] = str(paths["blockhead"])
    if paths["compile_pin"] is not None:
        result.update(JETSPEC_C1_COMPILE_PIN=str(paths["compile_pin"]),
                      JETSPEC_C1_COMPILE_PIN_SHA=runtime["compile_pin_sha256"])
    validate_environment(result, jetspec=recipe["method"] == "jetspecpp",
                         max_num_seqs=_recipe_concurrency(recipe, max_num_seqs),
                         recipe_name=recipe.get("name"),
                         resolve_only=resolve_only, regime=recipe['regime'])
    return result








def _arguments(args, recipe, output, paths=None, max_model_len=MAX_MODEL_LEN_FLOOR):
    result = [
        "--qwen", "--size", "8", "--b", str(getattr(args, "batch", 1)), "--numseqs", str(args.count),
        "--output_len", str(args.tokens), "--max_model_len", str(max_model_len),
        "--temp", str(getattr(args, "temperature", 0.0)), "--dtemp", "0",
        "--seed", str(getattr(args, "seed", 0)),
        "--prompts-json", str(Path(args.prompts_json).resolve()),
    ]
    if recipe["method"] == "jetspecpp":
        result.extend((
            "--jetspec", "--gpus", "3" if recipe["environment"].get(
                "JETSPEC_DRAFT_SPLIT") == "verify_gpu" else "2", "--jetspec-hyp", "16",
            "--jetspec-passes", str(recipe["passes"]), "--jetspec-wait-ms", "inf",
            "--jetspec-miss", "ar", "--f", "1"))
        if recipe["environment"].get("JETSPEC_HYBRID_PAIR") == "1":
            result.extend(("--jetspec-verify-backend", "flashinfer_paged"
                if (recipe["environment"].get("JETSPEC_VERIFY_REFERENCE") == "1"
                    or recipe["environment"].get("JETSPEC_HYBRID_VERIFY_FLASHINFER") == "1") else "sdpa"))
    elif recipe["method"] == "ar":
        result.extend(("--gpus", "2" if getattr(args, "placement", None) == "tp2" else "1"))
    elif recipe["method"] in ("jetspec.engines.jetspec_plus_plus.serving.api", "spec_sync", "eagle3"):
        if paths is None or paths.get("draft") is None:
            raise ValueError("speculative draft arguments need the pinned snapshot")
        result.extend(("--gpus", "1" if recipe["method"] == "spec_sync" else "2",
                       "--draft", str(paths["draft"])))
        if recipe["method"] == "eagle3":
            result.extend(("--eagle", "--async", "--k", str(recipe["k"])))
            if "stock_parameters" in recipe:
                result.extend(("--f", str(stock_parameters(recipe, args.dataset)["fanout"])))
        elif recipe["method"] == "jetspec.engines.jetspec_plus_plus.serving.api":
            stock = stock_parameters(recipe, args.dataset)
            result.extend(("--spec", "--async", "--k", str(stock["lookahead"]),
                           "--f", str(stock["fanout"])))
        else:
            result.extend(("--spec", "--k", str(recipe["k"])))
    elif recipe["method"] in ("blockhead", "dflash"):
        if paths is None or paths.get("blockhead") is None:
            raise ValueError("blockhead arguments need the resolved checkpoint directory")
        result.extend(("--gpus", "2" if getattr(args, "placement", None) == "tp2" else "1",
                       "--blockhead", str(paths["blockhead"])))
    else:
        raise ValueError(f"method {recipe['method']!r} has no bench argument mapping")
    result.extend(("--output-json", str(output)))
    if args.regime == "eager":
        result.append("--eager")
    if getattr(args, "stop_at_eos", "off") == "on":
        result.append("--stop_at_eos")
    return result


def timed_request_step(llm, inference_step, original_step, request_started,
                       request_seconds, request_tokens, *, first_step_requests=None):
    """Time completed requests while preserving the engine step's outputs."""
    running_before = {seq.seq_id: seq.num_completion_tokens for seq in llm.scheduler.running}
    max_running = len(running_before)
    step_started = time.perf_counter()
    completed = original_step(inference_step)
    now = time.perf_counter()
    max_running = max(max_running, len(llm.scheduler.running))
    for seq in llm.scheduler.running:
        if seq.num_completion_tokens:
            request_started.setdefault(seq.seq_id, now)
    for seq_id, completion in completed:
        started = request_started.pop(seq_id, None)
        if started is None:
            if running_before.get(seq_id) == 0 and completion:
                # Speculative prefill stages the recovery token without appending it.
                # EOS in the first decode can remove the request before the usual
                # after-step first-token stamp. Its only decode took this whole step.
                started, phase = step_started, 'first_decode'
            elif seq_id not in running_before and len(completion) == 1:
                # AR can finish with EOS in prefill: there is no decode duration.
                started, phase = now, 'prefill_only'
            else:
                raise RuntimeError(f"request {seq_id} finished before decode timing began")
            if first_step_requests is not None:
                first_step_requests.append(dict(seq_id=seq_id, phase=phase,
                    completion_tokens=len(completion), decode_seconds=now - started))
        request_seconds[seq_id] = now - started
        request_tokens[seq_id] = len(completion)
    return completed, max_running


def batch_summary(request_seconds, *, tokens, decode_tok_s, gpus, requests=None, request_tokens=None):
    """Decode rates from first token to completion for each served request.

    request_tokens maps a request to its completion length; without it every request is
    assumed to run to the budget, which is wrong once --stop-at-eos ends requests early.
    """
    if requests is not None and len(request_seconds) != requests:
        raise RuntimeError(f"timed {len(request_seconds)} requests, expected {requests}")
    if tokens < 2 or not request_seconds:
        raise ValueError("batch decode timing needs at least two tokens and positive durations")
    if request_tokens is not None and set(request_tokens) != set(request_seconds):
        raise RuntimeError("request token counts do not cover the timed requests")
    lengths = request_tokens if request_tokens is not None else dict.fromkeys(request_seconds, tokens)
    if any(seconds < 0 or (seconds == 0 and lengths[key] >= 2)
           for key, seconds in request_seconds.items()):
        raise ValueError("batch decode timing needs at least two tokens and positive durations")
    rates = [(lengths[key] - 1) / seconds for key, seconds in request_seconds.items()
             if lengths[key] >= 2]
    if not rates:
        raise ValueError("batch decode timing needs a request with at least two tokens")
    return {
        "per_request_decode_tok_s_mean": statistics.mean(rates),
        "per_request_decode_tok_s_p50": statistics.median(rates),
        "total_decode_tok_s": decode_tok_s,
        "gpus_used": gpus,
        "decode_tok_s_per_gpu": decode_tok_s / gpus,
    }




def sequential_single_request_reads(run, args, llm, prompts, sampling_params):
    """Keep C=1 occupancy at one while reusing the loaded engine for all prompts."""
    outputs, wall, combined = [], 0.0, None
    per_request = set()
    for prompt, params in zip(prompts, sampling_params):
        part_outputs, part_wall, metrics = run(args, llm, [prompt], [params])
        if len(part_outputs) != 1:
            raise RuntimeError("sequential C=1 read did not finish exactly one request")
        outputs.extend(part_outputs)
        wall += part_wall
        metrics = copy.deepcopy(metrics)
        if combined is None:
            combined = metrics
            continue
        for key, value in metrics.items():
            if key in per_request:
                combined[key].append(value)
            # A metric can first appear on a later request (e.g. the redraft wait on a
            # request's first miss round), so a key absent so far starts empty.
            elif isinstance(value, list):
                combined.setdefault(key, []).extend(value)
            elif isinstance(value, (int, float)):
                combined[key] = combined.get(key, 0) + value
            elif key == "jetspec_kind_summary":
                continue
            elif _numeric_mapping(value) and _numeric_mapping(combined.get(key)):
                merged = dict(combined[key])
                for name, number in value.items():
                    merged[name] = merged.get(name, 0) + number
                combined[key] = merged
            else:
                # Any other per-request value (e.g. DSpark's graph self-check) is kept
                # one entry per request: never summed, never dropped.
                combined[key] = [combined.get(key), value]
                per_request.add(key)
    if combined is None:
        raise RuntimeError("empty sequential C=1 read")
    if combined["jetspec_round_records"]:
        from jetspec.engines.jetspec_plus_plus.wire.protocol import summarize_candidate_kinds
        combined["jetspec_kind_summary"] = summarize_candidate_kinds(
            combined["jetspec_round_records"])
    rounds = sum(combined[key] for key in
                 ("jetspec_hit_count", "jetspec_miss_count", "jetspec_late_count"))
    combined["jetspec_late_fraction"] = combined["jetspec_late_count"] / rounds if rounds else 0.0
    return outputs, wall, combined


def stock_parameters(recipe, dataset):
    return recipe["stock_parameters"].get(dataset, recipe["stock_parameters"]["default"])


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--regime", choices=REGIMES, required=True)
    parser.add_argument("--engine", choices=("eager", "reference", "eval", "fast"))
    parser.add_argument("--placement", choices=PLACEMENTS, required=True)
    parser.add_argument("--mps-root", help="private per-GPU MPS control and log directory; "
                        "required with one_gpu_mps")
    parser.add_argument("--mps-draft-priority", choices=("0", "1"),
                        help="opt-in MPS priority for the spawned draft client (0 normal, 1 low)")
    parser.add_argument("--mps-draft-active-thread-percentage", type=int,
                        help="opt-in MPS thread cap for the spawned draft client (1..100)")
    parser.add_argument("--devices", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prompts-json", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1,
                        help="maximum concurrent requests (default: 1)")
    parser.add_argument("--sequential-prompts", action="store_true",
                        help="serve C=1 prompts one at a time in one loaded process")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="target sampling temperature (draft remains greedy)")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--child-wall-seconds", type=float, default=18 * 60,
                        help="wall-clock cap on one serve child before SIGTERM (default 18 min)")
    parser.add_argument("--passes", type=int,
                        help="drafter passes; when supplied, must match the named recipe")
    parser.add_argument("--sweep-rows-cap", type=int, choices=(2, 4, 8, 16),
                        help="override JetSpec++ eager pair row cap for draft-work sweep")
    parser.add_argument("--sweep-refresh", choices=("after_j0", "off"),
                        help="override JetSpec++ eager pair refresh for draft-work sweep")
    parser.add_argument("--sweep-j", type=int, choices=(1, 2),
                        help="override JetSpec++ eager pair J for draft-work sweep")
    parser.add_argument("--policy", help="C-indexed draft-budget policy JSON")
    parser.add_argument("--draft-split", choices=("verify_gpu",),
                        help="place a second draft executor beside the verify under MPS")
    parser.add_argument("--draft-split-frac", type=float,
                        help="fraction of requests assigned to the co-resident executor")
    parser.add_argument("--draft-deadline", choices=("on", "off"),
                 help="explicitly override the recipe draft deadline")
    parser.add_argument("--miss-redraft", choices=("on", "off"),
                 help="override miss redrafting for a paired after-miss read")
    parser.add_argument("--chain-miss-redraft", choices=("on", "off"),
                        help="enable the J0-only same-round chain miss redraft")
    parser.add_argument("--chain-miss-redraft-max-ms", type=float,
                        help="dedicated same-round redraft wait valve")
    parser.add_argument("--chain-miss-redraft-timing", choices=("on", "off"),
                        help="print each miss round's redraft and total wall time")
    parser.add_argument("--chain-check", choices=("on", "off"),
                        help="run same-round fused eager shadow and enforce the 300-round T0 gate")
    parser.add_argument("--chain-warm-reachable", choices=("on", "off"),
                        help="capture only chain keys reachable by this read")
    parser.add_argument("--chain-capture-stamps", choices=("on", "off"),
                        help="print wall seconds for every chain key and segment capture")
    parser.add_argument("--chain-decoder-key-bucket", choices=("on", "off"),
                        help="override decoder-window key bucketing for a chain T0 read")
    parser.add_argument("--chain-sega-glue", choices=("0", "1", "2"),
                        help="override segmented chain A glue for a T0 or timing read")
    parser.add_argument("--chain-glue2-profile", choices=("on", "off"),
                        help="print captured A/B1n device milliseconds after publish")
    parser.add_argument("--chain-refine-k3", choices=("on", "off"),
                        help="override K3 refinement bookkeeping for a chain timing pair")
    parser.add_argument("--chain-handoff", choices=("on", "off"),
                        help="pre-stage the next supported fixed R0 chain key")
    parser.add_argument("--chain-handoff-timing", action="store_true",
                        help="print per-round draft and verify handoff timestamps")
    parser.add_argument("--verify-host-census", choices=("on", "off"),
                        help="stamp each host segment of the N=1 device-verify round")
    parser.add_argument("--knob", action="append", default=[], metavar="NAME=VALUE",
                        help="set one registered JETSPEC_* knob for this read (repeatable)")
    parser.add_argument("--verify-lean", choices=("on", "off"),
                        help="set the six N=1 verifier host cuts (JETSPEC_VERIFY_STAGE_HOST ... STAGE_EARLY)")
    parser.add_argument("--verify-lean-stage-early", choices=("on", "off"),
                        help="override JETSPEC_VERIFY_STAGE_EARLY alone (after --verify-lean)")
    parser.add_argument("--chain-prepost", choices=("on", "off"),
                        help="post the next chain outcome receive before Seg P sync")
    parser.add_argument("--chain-prepost-timing", action="store_true",
                        help="print P-sync to next Seg A and verifier ready-key wait")
    parser.add_argument("--chain-tail-trace", action="store_true",
                        help="print absolute host stamps for every draft chain round")
    parser.add_argument("--chain-kernel-profile",
                        help="diagnostic: append per-kernel device time of sampled chain rounds to this JSONL")
    parser.add_argument("--chain-kernel-profile-every", type=int,
                        help="profile every Nth chain round (default 20)")
    parser.add_argument("--chain-k3", choices=("on", "off"),
                        help="override both segmented-chain K3 kernels")
    parser.add_argument("--chain-attn3", choices=("on", "off"),
                        help="override split Seg A FlashInfer planning for a chain read")
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--max-model-len", type=int,
                        help="serve this max_model_len instead of the one derived from the prompts "
                             "(still refused if a prompt + --tokens + lookahead does not fit)")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR),
                        help="where tools/fetch_artifacts.py --destination put the artifacts")
    parser.add_argument("--cache-root", help="persistent compiler cache root shared by serial reads")
    parser.add_argument("--compile-pin", type=Path, help="sealed machine-local compile pin directory")
    parser.add_argument("--compile-pin-sha", help="SHA256 of --compile-pin (both flags required)")
    parser.add_argument("--bundle-override", type=Path,
                        help="pinned task-local reused decoder bundle")
    parser.add_argument("--bundle-pin", type=Path,
                        help="hash receipt for --bundle-override")
    parser.add_argument("--host-values",
                        help="per-host JSON (schema %s): artifact directories, CUDA arch, "
                             "compile pin SHA, rendezvous mode, and %s"
                             % (HOST_VALUES_SCHEMA, ", ".join(HOST_VARIABLES)))
    parser.add_argument("--host-artifact", action="append", metavar="KEY=ABS_DIR",
                        help="override one manifest artifact directory on this host")
    parser.add_argument("--roundtrace",
                        help="write rank-local host spans, CUDA event pairs, report, and SVG here")
    parser.add_argument("--device-split", action="store_true",
                        help="record adapter, decoder, row, refresh, and Jacobi CUDA spans")
    parser.add_argument("--cpu-pin", default="auto", metavar="auto|off|A-B[;C-D]",
                        help="NUMA-local CPU pinning (default: auto); off or explicit ranges")
    parser.add_argument("--rank-cpus", metavar="VERIFY_RANGE;DRAFT_RANGE[;SPLIT_RANGE]",
                        help="alias for --cpu-pin with explicit ranges")
    parser.add_argument("--stop-at-eos", choices=("on", "off"), default="off",
                        help="end each sequence at end-of-sequence instead of running to --tokens (default: off)")
    parser.add_argument("--wire-records", choices=("on", "off"), default="on",
                        help="write JetSpec proposal wire records (default: on)")
    parser.add_argument("--causal-n1", action="store_true",
                         help="use the FlashInfer causal kernel for a single verify chain")
    parser.add_argument("--anatomy", choices=("on", "off"), default="on",
                 help="write cheap per-read host anatomy (default: on)")
    parser.add_argument("--resolve-only", action="store_true",
                        help="print RESOLVED_CONDITION and exit 0 before any model load")
    parser.add_argument("--internal-plan", help=argparse.SUPPRESS)
    return parser


def _read_proposals(directory, offsets):
    rows = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        start = offsets.get(str(path), 0)
        rows.extend(json.loads(line) for line in lines[start:])
        offsets[str(path)] = len(lines)
    return rows


def _install_population_recorder():
    """Install the population recorder in every spawned process without changing decisions."""
    record_root = os.environ.get("PRED_SERVE_READ_RECORDS")
    if not record_root:
        return
    sys.path.insert(0, str(ROOT / "serve"))
    from jetspec.engines.jetspec_plus_plus.draft.live_rows import KthRowsDrafter
    if getattr(KthRowsDrafter, "_serve_read_recorder", False):
        return
    KthRowsDrafter._serve_read_recorder = True
    refresh_counts = {}
    original_refresh = KthRowsDrafter._adapter_refresh

    def recorded_refresh(self, bucket, sequence, *args, **kwargs):
        if hasattr(sequence, "_jetspec_seq_id") and hasattr(
                sequence, "_jetspec_candidate_round"):
            key = (int(sequence._jetspec_seq_id),
                   int(sequence._jetspec_candidate_round))
            refresh_counts[key] = refresh_counts.get(key, 0) + 1
        return original_refresh(self, bucket, sequence, *args, **kwargs)

    original_flat = KthRowsDrafter._flat_rows_and_passes

    def record_flat(sequence, blocks, after_miss, refresh_calls=None):
        if hasattr(sequence, "_jetspec_seq_id") and hasattr(
                sequence, "_jetspec_candidate_round"):
            key = (int(sequence._jetspec_seq_id),
                   int(sequence._jetspec_candidate_round))
            metrics = blocks.head_metrics
            record = {
                "seq_id": key[0], "candidate_round": key[1],
                "after_miss": bool(after_miss),
                "pass_label": str(sequence.pending_block.kpred_pass),
                "rows_published": metrics["rows_published"],
                "rows_requested": metrics["rows_requested"],
                "rows_truncated": metrics["rows_truncated"],
                "refresh_calls": (refresh_counts.pop(key, 0) if refresh_calls is None
                                  else refresh_calls),
                "refresh_rounds": metrics.get("refresh_rounds", 0),
                "refresh_skipped": metrics.get("refresh_skipped", 0),
                "refresh_tokens": metrics.get("refresh_tokens"),
            }
            path = Path(record_root) / f"{os.getpid()}.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def recorded_flat(self, sequence, prediction, **kwargs):
        result = original_flat(self, sequence, prediction, **kwargs)
        record_flat(sequence, result[0], kwargs["after_miss"])
        return result

    original_batch = KthRowsDrafter._flat_rows_and_passes_batch

    def recorded_batch(self, sequences, predictions, **kwargs):
        result = original_batch(self, sequences, predictions, **kwargs)
        if result is not None:
            for sequence, blocks in zip(sequences, result):
                record_flat(sequence, blocks, getattr(sequence, '_jetspec_previous_selection_miss', False),
                            blocks.head_metrics.get('refresh_rounds', 0))
        return result

    KthRowsDrafter._adapter_refresh = recorded_refresh
    KthRowsDrafter._flat_rows_and_passes = recorded_flat
    KthRowsDrafter._flat_rows_and_passes_batch = recorded_batch


_install_population_recorder()


def _install_roundtrace():
    """Attach only in explicitly traced child processes, including spawned ranks."""
    if not os.environ.get("JETSPEC_ROUNDTRACE"):
        return None
    sys.path.insert(0, str(ROOT / "serve"))
    from jetspec.engines.jetspec_plus_plus.serving.roundtrace.current import install
    return install()


_install_roundtrace()


def resolve_roundtrace(args):
    value = getattr(args, "jetspec.engines.jetspec_plus_plus.serving.roundtrace", None) or os.environ.get("JETSPEC_ROUNDTRACE")
    return None if not value else Path(value).resolve()


def source_git_sha(root):
    return source_revision(root)[0]


def source_git_dirty(root):
    return source_revision(root)[1]




def write_anatomy(run_output, method, regime, measured, populations, *,
                  refresh="off", passes=None, rows_cap=None, trace_directory=None,
                  replace=False, eos_stopped=None, cpu_pin=None):
    """Write one post-run report; no instrumentation is installed in the child."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from jetspec.engines.jetspec_plus_plus.serving.tools.cutoff_anatomy import build_anatomy; from jetspec.engines.jetspec_plus_plus.serving.tools.cutoff_anatomy import headline
    from jetspec.engines.jetspec_plus_plus.serving.tools.eos_stop import round_is_miss
    if method == "jetspecpp":
        decisions = {(row["seq_id"], row["candidate_round"]): row
                     for row in measured["metrics"]["jetspec_round_records"]}
        populations = [dict(row, fallback=round_is_miss(
            decisions.get((row["seq_id"], row["candidate_round"]), {}), row)) for row in populations]
    trace_report = None
    trace_records = None
    if trace_directory is not None:
        trace_report = json.loads((Path(trace_directory) / "report.json").read_text())
        trace_records = [json.loads(line) for path in Path(trace_directory).glob("rounds-*.jsonl")
                         for line in path.read_text().splitlines() if line]
    anatomy = build_anatomy(method, regime, measured, populations,
                            refresh=refresh, passes=passes, rows_cap=rows_cap,
                            trace_report=trace_report, trace_records=trace_records,
                            eos_stopped=eos_stopped)
    if cpu_pin is not None:
        anatomy["cpu_pin"] = cpu_pin
    path = Path(str(run_output) + ".anatomy") / "anatomy.json"
    path.parent.mkdir(parents=True, exist_ok=replace)
    if path.exists() and not replace:
        raise FileExistsError(path)
    path.write_text(json.dumps(anatomy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(headline(anatomy), flush=True)
    return str(path)


def measured_output_check(outputs, *, count, tokens, temperature, seed, stop_at_eos):
    """Keep main's greedy shape gate; sampled reads retain their raw outputs."""
    if temperature == 0:
        assert_measured_output_shape(outputs, count=count, tokens=tokens,
                                     stop_at_eos=stop_at_eos)
        return {"shape_assertion": "enforced", "seed": seed}
    if len(outputs) != count:
        raise RuntimeError("output request count mismatch in measured")
    return {
        "shape_assertion": "skipped",
        "reason": "sampled target with EOS stopping can produce different token lengths across reads",
        "temperature": temperature,
        "seed": seed,
        "token_lengths": [len(row["token_ids"]) for row in outputs],
        "token_ids": [row["token_ids"] for row in outputs],
    }


STARTUP_MARKERS = (("W240_CAPTURE", "ms"), ("W259_C1_CAPTURE", "ms"), ("KEYLEN_PREWARM", "ms"))


def _startup_stamp(stage):
    """SERVE_READ_STARTUP_TIMING=1: one host stamp per startup stage of the read child."""
    if os.environ.get("SERVE_READ_STARTUP_TIMING") == "1":
        print("SERVE_READ_STARTUP " + json.dumps({"stage": stage, "t": time.perf_counter()}),
              flush=True)


def startup_summary(log):
    """Stage walls between the child's stamps plus the capture/prewarm lines' own ms."""
    stamps = []
    for line in log.splitlines():
        if line.startswith("SERVE_READ_STARTUP "):
            stamps.append(json.loads(line.split(" ", 1)[1]))
    if not stamps:
        return None
    walls = {f"{a['stage']}->{b['stage']}_s": round(b["t"] - a["t"], 3)
             for a, b in zip(stamps, stamps[1:])}
    sums = {}
    for marker, field in STARTUP_MARKERS:
        values = [float(m) for m in re.findall(rf"{marker} .*?\b{field}=([0-9.]+)", log)]
        sums[marker] = {"count": len(values), "total_s": round(sum(values) / 1000, 3)}
    return {"stages": walls, "total_s": round(stamps[-1]["t"] - stamps[0]["t"], 3),
            "markers": sums,
            "note": "JIT (FlashInfer/Triton) compiles fall inside the stage that first needs them"}


def record_target_autotune(target):
    """Record when a hybrid target's served outputs depend on unpinned Triton choices."""
    config = json.loads((Path(target) / "config.json").read_text(encoding="utf-8"))
    config = config.get("text_config", config)
    if config.get("model_type") != "qwen3_5_text":
        return
    if ("linear_attention" not in config.get("layer_types", [])
            and os.environ.get("JETSPEC_HYBRID_FUSED_NORM") != "1"):
        return
    import jetspec.engines.jetspec_plus_plus.draft.run_manifest as run_manifest
    unpinned = not bool(os.environ.get("JETSPEC_C1_COMPILE_PIN"))
    run_manifest.record_value("target_triton_autotune_unpinned", unpinned)
    if unpinned:
        print("WARNING: target uses Triton-autotuned kernels without a compile pin; "
              "outputs depend on the Triton cache.", flush=True)




def _child(plan_path, *, resident=None):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    _startup_stamp("child_start")
    sys.path.insert(0, str(ROOT / "serve"))
    sys.path.insert(0, str(ROOT / "serve" / "jetspec.engines.jetspec_plus_plus.serving.bench.bench"))
    import jetspec.engines.jetspec_plus_plus.serving.bench.bench as bench
    import torch

    if resident is not None:
        saved_paths, saved_record, saved_run, saved_load = (
            bench.get_model_paths, bench.record_run_inputs, bench.run_benchmark, bench.LLM)
        from jetspec.engines.jetspec_plus_plus.serving.tools.w632_measurement.multiread import reset_request_state
        from jetspec.engines.jetspec_plus_plus.serving.tools.w632_measurement.resident import retain_manifest; from jetspec.engines.jetspec_plus_plus.serving.tools.w632_measurement.resident import rank_boundary
        import jetspec.engines.jetspec_plus_plus.draft.run_manifest as run_manifest
        if resident.manifest is not None:
            retain_manifest(resident.manifest)
        bench.LLM = lambda model, **kwargs: resident.engine(saved_load, model, **kwargs)
    bench.get_model_paths = lambda args: (
        "Qwen3-8B", plan["target_model"], plan["generic_draft_model"])
    original_record_inputs = bench.record_run_inputs
    def record_inputs(*args, **kwargs):
        original_record_inputs(*args, **kwargs)
        record_target_autotune(plan["target_model"])
        import jetspec.engines.jetspec_plus_plus.draft.run_manifest as run_manifest
        if os.environ.get("RX_SOURCE_PROVENANCE"):
            receipt = os.environ["RX_SOURCE_PROVENANCE"]
            run_manifest.record_file("rx_source_provenance", receipt)
            sha, dirty = source_revision()
            run_manifest.record_value("rx_source_git", {"sha": sha, "dirty": dirty})
        run_manifest.record_value("method_mode", plan["method_mode"])
        run_manifest.record_value("cpu_pin", plan["cpu_pin"]["mode"])
        if "thresholds_override" in plan:
            run_manifest.record_value("thresholds_override", plan["thresholds_override"])
        run_manifest.record_value("roundtrace_enabled", bool(os.environ.get("JETSPEC_ROUNDTRACE")))
        if os.environ.get("JETSPEC_ROUNDTRACE"):
            run_manifest.record_value("roundtrace_directory", os.environ["JETSPEC_ROUNDTRACE"])
            run_manifest.record_value("roundtrace_device", os.environ.get("JETSPEC_ROUNDTRACE_DEVICE") == "1")
        for role, path in sorted(plan.get("declared_manifest_files", {}).items()):
            run_manifest.record_file(role, path)
    bench.record_run_inputs = record_inputs
    original = bench.run_benchmark
    def run(args, llm, prompts, sampling_params):
        _startup_stamp("engine_ready")
        print("SERVE_READ_PHASE_START cold", flush=True)
        cold_started = time.perf_counter()
        if resident is not None:
            torch.cuda.synchronize()
            reset_request_state(llm)
        needs_warm = resident is None or plan.get('dataset') not in resident.warmed
        if not args.eager and needs_warm:
            if resident is not None:
                prewarm_target_prefill(llm.model_runner, prompts, plan.get("sequential_prompts", False))
            else:
                from jetspec.engines.jetspec_plus_plus.serving.sequence import Sequence
                from jetspec.engines.jetspec_plus_plus.serving.utils.context import get_context; from jetspec.engines.jetspec_plus_plus.serving.utils.context import reset_context
                runner = llm.model_runner
                from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled as eval_enabled
                # Eval splits the discarded cold prefill at the engine's prefill budget:
                # 128 long LCB prompts in one 27B prefill exceed the target card.
                cold_groups = ([[len(p) for p in prompts]] if not eval_enabled()
                               else cold_prefill_groups([len(p) for p in prompts],
                                                        llm.config.max_num_batched_tokens))
                for cold_lengths in cold_groups:
                    if runner.num_tp_gpus > 1:
                        # tp2: every target rank must join the prefill's collectives.
                        runner.call("run_cold_prefill", cold_lengths)
                    else:
                        ids, positions = runner.prepare_prefill([Sequence([0] * n) for n in cold_lengths])
                        get_context().slot_mapping = ids.new_full(ids.shape, -1, dtype=torch.int32)
                        try:
                            runner.run_model(ids, positions, True)
                        finally:
                            reset_context()
        _startup_stamp("cold_forward_done")
        # Only JetSpec++ has a session to warm; the block head and the sync drafter warm at load.
        if plan["method_mode"] == "jetspecpp" and (needs_warm or resident is not None):
            step = llm.create_inference_step(llm.config)
            if resident is not None and resident.warmed:
                rank_boundary(step.speculator)
            else:
                step.speculator.warm_jetspec_session()
        # The block head and the sync drafter load (and capture their graphs) when generate()
        # builds its inference step; build that step here so the measured window only replays.
        prebuilt = None
        if (getattr(llm, "blockhead_checkpoint_dir", None) or getattr(llm, "sync_drafter_checkpoint_dir", None)) \
                and not plan.get("sequential_prompts"):
            prebuild_started = time.perf_counter()
            prebuilt = [llm.create_inference_step(llm.config)]
            torch.cuda.synchronize()
            import jetspec.engines.jetspec_plus_plus.draft.run_manifest as run_manifest
            run_manifest.record_value("serve_read_prebuilt_step_seconds", time.perf_counter() - prebuild_started)
            build_step = llm.create_inference_step
            llm.create_inference_step = lambda config: prebuilt.pop() if prebuilt else build_step(config)
        torch.cuda.synchronize()
        _startup_stamp("warm_session_done")
        if resident is not None:
            resident.warmed.add(plan.get('dataset'))
        bench.reset_metrics()
        cold_wall = time.perf_counter() - cold_started
        print("SERVE_READ_PHASE_END cold", flush=True)
        audit_cache = (os.environ.get('SERVE_READ_STARTUP_TIMING') == '1'
                       or bool(os.environ.get('JETSPEC_C1_COMPILE_PIN', '')))
        if audit_cache:
            from jetspec.engines.jetspec_plus_plus.serving.tools.w632_measurement.cache import snapshot as cache_snapshot; from jetspec.engines.jetspec_plus_plus.serving.tools.w632_measurement.cache import changes as cache_changes
            cache_before = cache_snapshot(os.environ)
        print("SERVE_READ_PHASE_START measured", flush=True)
        request_started = {}
        request_seconds = {}
        request_tokens = {}
        first_step_requests = []
        max_running_requests = 0
        original_step = llm.step
        if "batch" in plan:
            def timed_step(inference_step):
                nonlocal max_running_requests
                completed, peak = timed_request_step(
                    llm, inference_step, original_step, request_started,
                    request_seconds, request_tokens, first_step_requests=first_step_requests)
                max_running_requests = max(max_running_requests, peak)
                return completed
            llm.step = timed_step
        try:
            if plan.get("sequential_prompts"):
                outputs, wall, metrics = sequential_single_request_reads(
                    original, args, llm, prompts, sampling_params)
            else:
                outputs, wall, metrics = original(args, llm, prompts, sampling_params)
        finally:
            llm.step = original_step
            if prebuilt is not None:
                del llm.create_inference_step
        print("SERVE_READ_PHASE_END measured", flush=True)
        output_check = measured_output_check(
            outputs, count=plan["count"], tokens=plan["tokens"],
            temperature=args.temp, seed=args.seed, stop_at_eos=args.stop_at_eos)
        import jetspec.engines.jetspec_plus_plus.draft.run_manifest as run_manifest
        run_manifest.record_value("serve_read_output_check", output_check)
        if audit_cache:
            fresh = cache_changes(cache_before, cache_snapshot(os.environ))
            metrics['compiler_cache_audit'] = dict(measured_jit_suspect=bool(fresh),
                                                  new_or_changed_files=fresh)
        snapshot = copy.deepcopy(metrics)
        proposals = _read_proposals(plan["records"],
            {} if resident is None else resident.proposal_offsets) if plan["records"] else []
        decode_tokens = sum(len(row["token_ids"]) for row in outputs) - len(outputs)
        decode_time = snapshot["decode_total_time"]
        if decode_time <= 0:
            raise RuntimeError("non-positive decode time in measured")
        snapshot["serve_read"] = {
            "process_index": plan["process_index"],
            "campaign_cold": plan["campaign_cold"],
            "process_model": ("sequential single-request generates in one process"
                              if plan.get("sequential_prompts") else
                              "one generate per process; repeats = separate processes"),
            "cold": {"phase": "cold", "wall_seconds": cold_wall},
            "measured": {
                "phase": "measured", "wall_seconds": wall,
                "output_check": output_check,
                "decode_tokens": decode_tokens, "decode_tok_s": decode_tokens / decode_time,
                "output_sha256": hashlib.sha256(json.dumps(
                    [row["token_ids"] for row in outputs], separators=(",", ":")).encode()).hexdigest(),
                "metrics": copy.deepcopy(metrics), "proposals": proposals,
            },
        }
        if "batch" in plan:
            snapshot["serve_read"]["measured"].update(batch_summary(
                request_seconds, tokens=plan["tokens"],
                decode_tok_s=decode_tokens / decode_time, gpus=plan["gpus_used"],
                requests=plan["count"], request_tokens=request_tokens))
            snapshot["serve_read"]["measured"]["max_running_requests"] = max_running_requests
            snapshot["serve_read"]["measured"]["first_step_requests"] = first_step_requests
        return outputs, wall, snapshot

    bench.run_benchmark = run
    if plan.get("warmup_prefix_lengths"):
        os.environ["JETSPEC_WARMUP_PREFIX_LENGTHS"] = json.dumps(plan["warmup_prefix_lengths"])
    if plan.get("chain_warm_read"):
        os.environ["JETSPEC_CHAIN_WARM_READ"] = json.dumps(plan["chain_warm_read"])
    if resident is not None and plan['method_mode'] == 'jetspecpp':
        Path(os.environ['KIT_RESIDENT_BOUNDARY']).write_text(json.dumps(dict(
            dataset=plan['dataset'], index=plan['resident_index'], environment={
                key: os.environ[key] for key in ('JETSPEC_DATASET',
                    'JETSPEC_WARMUP_PREFIX_LENGTHS', 'JETSPEC_CHAIN_WARM_READ')
                if key in os.environ})))
    sys.argv = [str(ROOT / "serve" / "jetspec.engines.jetspec_plus_plus.serving.bench.bench" / "bench.py"), *plan["bench_arguments"]]
    try:
        bench.main()
    finally:
        if resident is not None:
            resident.manifest = run_manifest.as_dict()
            bench.get_model_paths, bench.record_run_inputs, bench.run_benchmark, bench.LLM = (
                saved_paths, saved_record, saved_run, saved_load)






def prepare_wire_records(method, mode, output, child_environment):
    child_environment.pop("W280_RECORD", None)
    if method != "jetspecpp" or mode == "off":
        return None
    path = Path(str(output) + ".rows")
    path.mkdir()
    child_environment["W280_RECORD"] = str(path)
    return path


def _main(args, *, resident=None):
    pin_environment = compile_pin_environment(args)
    args.batch = getattr(args, "batch", 1)
    if str(ROOT / "serve") not in sys.path:
        sys.path.insert(0, str(ROOT / "serve"))
    from jetspec.engines.jetspec_plus_plus.draft.env import require_recipe_launch
    require_recipe_launch(getattr(args, "recipe", None), max_num_seqs=args.batch,
                          resolve_only=getattr(args, "resolve_only", False))
    args.sequential_prompts = getattr(args, "sequential_prompts", False)
    if min(args.count, args.tokens, args.repeats, args.batch) < 1:
        raise ValueError("count, tokens, repeats and batch must be positive")
    if args.batch > args.count:
        raise ValueError("--batch cannot exceed --count")
    if args.sequential_prompts and args.batch != 1:
        raise ValueError("--sequential-prompts requires --batch 1")
    if args.batch > 1 and args.tokens < 2:
        raise ValueError("--batch > 1 needs at least two output tokens for decode timing")
    method = canonical_method(args.method)
    if getattr(args, "policy", None) and method != "jetspecpp":
        raise ValueError("--policy requires jetspecpp")
    if getattr(args, "policy", None) and any(
            getattr(args, field, None) is not None for field in
            ("sweep_rows_cap", "sweep_refresh", "sweep_j")):
        raise ValueError("--policy cannot be combined with --sweep-rows-cap/--sweep-refresh/--sweep-j")
    if getattr(args, "draft_deadline", None) is not None and method != "jetspecpp":
        raise ValueError("--draft-deadline is supported only for jetspecpp")
    if getattr(args, "miss_redraft", None) is not None and method != "jetspecpp":
        raise ValueError("--miss-redraft is supported only for jetspecpp")
    if getattr(args, "causal_n1", False) and method != "jetspecpp":
        raise ValueError("--causal-n1 is supported only for jetspecpp")
    apply_mps_draft_environment({}, args, method)
    if method == SYNC_METHOD:
        trace_root = resolve_roundtrace(args)
        return (_main_sync(args) if trace_root is None
                else _main_sync_traced(args, trace_root))
    if method not in ("jetspecpp", "ar", "blockhead", "dflash", "jetspec.engines.jetspec_plus_plus.serving.api", "spec_sync", "eagle3"):
        raise ValueError(f"method {args.method!r} is reserved but not implemented")
    recipe, policy_record = apply_policy_overrides(
        load_recipe(ROOT, args.method, args.regime, args.recipe), args)
    from jetspec.engines.jetspec_plus_plus.serving.presets.presets import engine_name
    selected_engine = engine_name(recipe['environment'], regime=recipe['regime'])
    if getattr(args, 'engine', None) is not None:
        if args.engine != selected_engine:
            raise ValueError(f"recipe {recipe['name']} selects {selected_engine}, not {args.engine}")
        recipe = dict(recipe, environment=dict(recipe['environment'], JETSPEC_ENGINE=args.engine))
    validate_recipe_batch(recipe, args.batch)
    if args.draft_split or args.draft_split_frac is not None:
        if (method, args.placement) != ("jetspecpp", "pair") or args.regime not in ("eager", "graphed"):
            raise ValueError("draft split requires eager JetSpec++ pair")
        if args.draft_split != "verify_gpu":
            raise ValueError("--draft-split verify_gpu is required with --draft-split-frac")
        fraction = 0.5 if args.draft_split_frac is None else args.draft_split_frac
        if not 0 < fraction < 1:
            raise ValueError("draft split fraction must lie strictly between 0 and 1")
        recipe = dict(recipe, environment=dict(recipe["environment"]))
        recipe["environment"].update(JETSPEC_DRAFT_SPLIT="verify_gpu",
                                     JETSPEC_DRAFT_SPLIT_FRAC=str(fraction))
    require_supported_block_size(recipe)
    roundtrace_root = resolve_roundtrace(args)
    if args.passes is not None and args.passes != recipe["passes"]:
        raise ValueError(
            f"--passes={args.passes} contradicts recipe passes={recipe['passes']}")
    placement = resolve_placement(
        args.placement, args.devices, recipe["environment"], method)
    placement = resolve_draft_split_placement(placement, recipe["environment"])
    validate_method_placement(method, args.placement)
    prompts = Path(args.prompts_json).resolve()
    if not prompts.is_file():
        raise FileNotFoundError(f"missing prompts JSON: {prompts}")
    output = Path(args.output_json).resolve()
    run_outputs = campaign_output_paths(output, args.repeats)
    campaign_path = Path(str(output) + ".campaign.json")
    claimed = [campaign_path]
    for run_output in run_outputs:
        claimed.extend((run_output, Path(str(run_output) + ".manifest.json"),
                        Path(str(run_output) + ".populations.json")))
        if args.anatomy == "on":
            claimed.append(Path(str(run_output) + ".anatomy"))
        if method == "jetspecpp" and args.wire_records == "on":
            claimed.append(Path(str(run_output) + ".rows"))
    if roundtrace_root is not None:
        # Recorder IDs restart in every process; an earlier run's rounds files make read-N unreadable.
        claimed.extend(sorted(roundtrace_root.glob("read-*/rounds-*.jsonl")))
    existing = [str(path) for path in claimed if path.exists()]
    if existing and not args.resolve_only:
        raise FileExistsError("refusing to overwrite serve read artifact(s): " + ", ".join(existing))
    output.parent.mkdir(parents=True, exist_ok=True)
    host = apply_host_artifact_overrides(
        load_host_values(args.host_values) if args.host_values else no_host_values(),
        getattr(args, "host_artifact", None))
    artifacts_dir = Path(args.artifacts_dir).resolve()
    if bool(args.bundle_override) != bool(args.bundle_pin):
        raise ValueError("BUNDLE_PIN_REQUIRED: --bundle-override and --bundle-pin must be paired")
    if args.bundle_override and method != "jetspecpp":
        raise ValueError("BUNDLE_OVERRIDE_ONLY_JETSPECPP")
    bundle_pin = json.loads(args.bundle_pin.read_text()) if args.bundle_pin else None
    paths, expected, pins, sources = _resolve_artifacts(
        recipe, host, artifacts_dir, bundle_override=args.bundle_override, bundle_pin=bundle_pin)
    hf_cache, sources["hf_cache"] = host_variable("SSD_HF_CACHE", host)
    dataset_dir, sources["dataset_dir"] = host_variable("SSD_DATASET_DIR", host)
    if method in ("jetspec.engines.jetspec_plus_plus.serving.api", "spec_sync"):
        hub_cache = _hub_cache_root(hf_cache)
        paths["draft"] = _resolve_draft_snapshot(recipe, hub_cache)
        sources["draft"] = ("environment:HF_HUB_CACHE" if os.environ.get("HF_HUB_CACHE") else
                            "environment:HF_HOME" if os.environ.get("HF_HOME") else
                            sources["hf_cache"])
        expected["draft/model.safetensors"] = recipe["draft_sha256"]
    runtime = {
        "dataset": args.dataset, "hf_cache": hf_cache,
        "dataset_dir": dataset_dir,
        "cuda_arch": resolve_cuda_arch(recipe, host),
        "compile_pin_sha256": resolve_compile_pin_sha256(recipe, host),
        "graphed_ar_disable_dynamo": resolve_graphed_ar_disable_dynamo(recipe, host),
        "cache": str(Path(args.cache_root).resolve()) if args.cache_root else
                 str(output.parent / ".serve-read-cache"),
    }
    uuids = _gpu_uuids(placement["visible_devices"], required=not args.resolve_only)
    cpu_pin = resolve_cpu_pin(args, uuids, method)
    if not args.resolve_only:
        _gpu_census(uuids, allowed_pids=() if resident is None else resident.pids)
    with (tempfile.TemporaryDirectory(prefix="serve-read-") if resident is None
          else resident.scratch()) as scratch:
        scratch = Path(scratch)
        pins_path = scratch / "pins.json"
        if pins is not None:
            pins_path.write_text(json.dumps(pins, sort_keys=True), encoding="utf-8")
        environment = _environment(recipe, placement, paths, runtime, pins_path, scratch,
                                   max_num_seqs=args.batch, resolve_only=args.resolve_only)
        environment.update(pin_environment)
        apply_cpu_pin_environment(environment, cpu_pin)
        apply_mps_draft_environment(environment, args, method)
        if recipe["environment"].get("JETSPEC_DRAFT_SPLIT") == "verify_gpu":
            environment["SSD_SPLIT_DEDICATED_UUID"] = uuids[1]
        if args.device_split:
            if method != "jetspecpp":
                raise ValueError("--device-split requires jetspecpp")
            environment["JETSPEC_DEVICE_SPLIT"] = "1"
            from jetspec.engines.jetspec_plus_plus.draft.env import validate_environment
            validate_environment(environment, jetspec=True, max_num_seqs=args.batch,
                                 recipe_name=recipe["name"], resolve_only=args.resolve_only)
        if getattr(args, "causal_n1", False):
            environment["SSD_EAGER_VERIFY_CAUSAL_N1"] = "1"
        apply_miss_redraft_override(environment, args.miss_redraft)
        apply_chain_miss_redraft_override(
            environment, args.chain_miss_redraft, args.chain_miss_redraft_max_ms,
            args.chain_miss_redraft_timing)
        apply_chain_check_override(environment, args.chain_check)
        apply_chain_warm_override(environment, args.chain_warm_reachable,
                                  args.chain_capture_stamps)
        apply_chain_decoder_key_bucket_override(environment, args.chain_decoder_key_bucket)
        apply_chain_sega_glue_override(environment, args.chain_sega_glue)
        apply_chain_glue2_profile_override(environment, args.chain_glue2_profile)
        apply_chain_refine_k3_override(environment, args.chain_refine_k3)
        apply_chain_handoff_override(environment, args.chain_handoff,
                                     args.chain_handoff_timing)
        apply_chain_prepost_override(environment, args.chain_prepost,
                                     args.chain_prepost_timing)
        apply_chain_attn3_override(environment, args.chain_attn3)
        apply_verify_host_census_override(environment, args.verify_host_census)
        apply_verify_lean_override(environment, getattr(args, "verify_lean", None),
                                   getattr(args, "verify_lean_stage_early", None))
        apply_chain_tail_trace_override(environment, getattr(args, "chain_tail_trace", False),
                                        getattr(args, "chain_kernel_profile", None),
                                        getattr(args, "chain_kernel_profile_every", None))
        apply_knob_overrides(environment, getattr(args, "knob", None))
        if environment.get("JETSPEC_EVAL_MODE", "off") != "off":
            from jetspec.engines.jetspec_plus_plus.draft.eval_determinism import WORKSPACE
            environment["CUBLAS_WORKSPACE_CONFIG"] = WORKSPACE
        if environment.get('CRUCIBLE_CACHE_RECEIPT') and environment.get('JETSPEC_C1_COMPILE_PIN_SHA'):
            cache_receipt = json.loads(environment['CRUCIBLE_CACHE_RECEIPT'])
            if cache_receipt['persistent']:
                # Each sealed draw has its own Inductor tree; another recipe's record
                # cannot add files to this pin's audited cache.
                sha = environment['JETSPEC_C1_COMPILE_PIN_SHA']
                pinned_cache = str(Path(cache_receipt['root']) / 'inductor-pins' / sha)
                environment['TORCHINDUCTOR_CACHE_DIR'] = pinned_cache
                cache_receipt['environment']['TORCHINDUCTOR_CACHE_DIR'] = pinned_cache
                cache_receipt['compile_pin_sha256'] = sha
                environment['CRUCIBLE_CACHE_RECEIPT'] = json.dumps(cache_receipt, sort_keys=True)
        apply_chain_k3_override(environment, args.chain_k3)
        draft_deadline = resolve_draft_deadline(
            environment.get("JETSPEC_DRAFT_DEADLINE"), args.draft_deadline)
        if draft_deadline is not None:
            environment["JETSPEC_DRAFT_DEADLINE"] = draft_deadline
        if environment.get("JETSPEC_CHAIN_WIRE_RECORD") == "1":
            # Validate the destination before launch; prepare_wire_records creates
            # each read's directory and supplies its actual path to the child.
            environment.pop("W280_RECORD", None)
            if method == "jetspecpp" and args.wire_records == "on":
                environment["W280_RECORD"] = str(run_outputs[0]) + ".rows"
        if ((method == "jetspecpp" and args.batch == 1)
                or args.chain_miss_redraft is not None
                or args.chain_miss_redraft_max_ms is not None
                or args.chain_miss_redraft_timing is not None):
            from jetspec.engines.jetspec_plus_plus.draft.env import validate_environment
            validate_environment(environment, jetspec=True, max_num_seqs=args.batch,
                                 recipe_name=recipe["name"], resolve_only=args.resolve_only)
        max_model_len = resolve_max_model_len(
            prompts, args.count, args.tokens, method, environment, paths["target"],
            paths["blockhead"], forced=forced_max_model_len(recipe, args))
        apply_cache_capacity(environment, method, max_model_len)
        hybrid_target = json.loads((paths["target"] / "config.json").read_text()).get("model_type") == "qwen3_5"
        generic = (paths["target"] if hybrid_target and method in ("ar", "blockhead")
                   else paths["draft"] if method in ("jetspec.engines.jetspec_plus_plus.serving.api", "spec_sync", "eagle3")
                   else _generic_draft_path(hf_cache))
        # The backend's family guard is path-name based.  Give the pinned target an explicit
        # family-bearing alias while retaining its real path in the resolved condition.
        target_alias = scratch / "Qwen3-8B-target"
        target_alias.symlink_to(paths["target"], target_is_directory=True)
        if hybrid_target and method in ("ar", "blockhead"):
            generic = target_alias
        condition = {
            "schema": "jetspec.resolved_condition/1", "method": method,
            "rendezvous_mode": host["rendezvous_mode"],
            "method_display": recipe["display_name"],
            "method_alias_requested": args.method,
            "regime": args.regime, "placement": args.placement,
            "devices": placement["visible_devices"], "device_uuids": uuids,
            "cpu_pin": cpu_pin, "stop_at_eos": args.stop_at_eos,
            "temperature": args.temperature, "seed": args.seed,
            "rank_device_map": environment["SSD_RANK_DEVICE_MAP"],
            "dataset": args.dataset, "prompts_json": str(prompts),
            "prompts_sha256": _sha256(prompts), "count": args.count, "tokens": args.tokens,
            "repeats": args.repeats, "measured_reads": max(0, args.repeats - 1),
            "process_model": ("sequential single-request generates in one process"
                              if args.sequential_prompts else
                              "one generate per process; repeats = separate processes"),
            "sequential_prompts": args.sequential_prompts,
            "cache_root": runtime["cache"],
            "recipe": recipe["name"],
            "passes": recipe["passes"], "J": recipe["passes"] - 1,
            "draft_deadline": environment.get("JETSPEC_DRAFT_DEADLINE"),
            "max_model_len": max_model_len["value"], "max_model_len_derivation": max_model_len,
            "drafter_cache_capacity": max_model_len["cache_capacity"],
            "artifacts": {key: None if value is None else str(value) for key, value in paths.items()},
            "environment": _condition_environment(environment, recipe),
            "argv": sys.argv,
            "jetspec.engines.jetspec_plus_plus.serving.roundtrace": None if roundtrace_root is None else str(roundtrace_root),
            "git": source_git_sha(ROOT),
        }
        if args.sequential_prompts or 'CRUCIBLE_CACHE_RECEIPT' in environment:
            condition['compiler_cache'] = json.loads(environment.get('CRUCIBLE_CACHE_RECEIPT', 'null'))
        if args.batch > 1:
            condition["batch"] = args.batch
        if policy_record is not None:
            condition.update(policy_record)
        if method == "eagle3":
            condition.update(draft_artifact=recipe["draft_artifact"],
                             draft_file_sha256=recipe["draft_file_sha256"], k=recipe["k"],
                             draft_snapshot=str(paths["draft"]))
        if method in ("jetspec.engines.jetspec_plus_plus.serving.api", "spec_sync"):
            condition.update(draft_repo=recipe["draft_repo"],
                             draft_revision=recipe["draft_revision"],
                             draft_snapshot=str(paths["draft"]),
                             draft_file_sha256=recipe["draft_sha256"],
                             k=(stock_parameters(recipe, args.dataset)["lookahead"]
                                if method == "jetspec.engines.jetspec_plus_plus.serving.api" else recipe["k"]),
                             **{"async": method == "jetspec.engines.jetspec_plus_plus.serving.api"},
                             bench_argv=_arguments(args, recipe, run_outputs[0], paths,
                                                   max_model_len["value"]))
        if method == "eagle3":
            condition["bench_argv"] = _arguments(
                args, recipe, run_outputs[0], paths, max_model_len["value"])
        if method == "jetspec.engines.jetspec_plus_plus.serving.api" or (method == "eagle3" and "stock_parameters" in recipe):
            stock = stock_parameters(recipe, args.dataset)
            condition.update(stock_lookahead=stock["lookahead"], stock_fanout=stock["fanout"])
        if paths["bundle"] is not None:
            condition.update(
                bundle_artifact=recipe["bundle_artifact"], bundle=str(paths["bundle"]))
        condition["git_dirty"] = source_git_dirty(ROOT)
        condition["resolution"] = {
            "host_values": host["path"], "artifacts_dir": str(artifacts_dir),
            "sources": dict(sorted(sources.items())),
        }
        if not args.resolve_only:
            condition["eos_token_ids"] = target_eos_token_ids(paths["target"])
        print("RESOLVED_CONDITION " + json.dumps(condition, sort_keys=True), flush=True)
        if args.resolve_only:
            print("SERVE_READ_RESOLVE_ONLY " + json.dumps({
                "recipe": recipe["name"], "launched": False}), flush=True)
            return
        split_draft = recipe["environment"].get("JETSPEC_DRAFT_SPLIT") == "verify_gpu"
        is_mps = args.placement in ("mps", "one_gpu_mps") or split_draft
        if is_mps and not args.mps_root:
            raise ValueError("--mps-root is required for an MPS placement")
        mps_root = private_mps_root(args.mps_root, uuids[0], output) if is_mps else None
        mps = None
        reads = []
        manager = (MPSControl(mps_root, uuids[0]) if is_mps
                   else contextlib.nullcontext())
        with manager as mps:
            if mps is not None:
                environment.update(mps.client_environment())
            for index, run_output in enumerate(run_outputs):
                launch_census = _gpu_census(
                    uuids, allowed_pids=(resident.pids if resident is not None else
                                        () if mps is None else (mps.server_pid,)))
                records = scratch / f"proposals-{index}" if method == "jetspecpp" else None
                if records is not None:
                    records.mkdir()
                plan = {
                    "target_model": str(target_alias), "generic_draft_model": str(generic),
                    "engine": selected_engine,
                    "method_mode": {"jetspec.engines.jetspec_plus_plus.serving.api": "spec_async", "spec_sync": "spec"}.get(method, method),
                    "cpu_pin": cpu_pin,
                    "count": args.count, "tokens": args.tokens,
                    "records": None if records is None else str(records),
                    "process_index": index, "campaign_cold": index == 0,
                    "bench_arguments": _arguments(
                        args, recipe, run_output, paths, condition["max_model_len"]),
                    "warmup_prefix_lengths": warmup_prefix_lengths(
                        method, args.regime, recipe["environment"], args.prompts_json, args.count),
                    **({"chain_warm_read": {
                        "prompt_lengths": [length for _, length in prompt_token_lengths(
                            args.prompts_json, args.count)], "tokens": args.tokens}}
                       if environment.get("JETSPEC_CHAIN_WARM_REACHABLE") == "1" else {}),
                    "declared_manifest_files": ({
                        role: str(paths["blockhead"] / role.split("/", 1)[1])
                        for role in expected
                    } if method in ("blockhead", "dflash") else {}),
                }
                plan.update(batch=args.batch, gpus_used=len(uuids),
                            sequential_prompts=args.sequential_prompts)
                if pins is not None and "thresholds_override" in pins:
                    plan["thresholds_override"] = pins["thresholds_override"]
                plan_path = scratch / f"plan-{index}.json"
                plan_path.write_text(json.dumps(plan), encoding="utf-8")
                command = _isolated_command([
                    sys.executable, "-m", "jetspec.engines.jetspec_plus_plus.serving.read",
                    "--internal-plan", str(plan_path)], host["rendezvous_mode"])
                child_environment = dict(environment)
                wire_records = prepare_wire_records(
                    method, args.wire_records, run_output, child_environment)
                if host["rendezvous_mode"] == "host_port":
                    child_environment["SSD_DIST_PORT"] = str(_free_local_port())
                    print("RENDEZVOUS_PORT " + child_environment["SSD_DIST_PORT"], flush=True)
                trace_directory = None
                if roundtrace_root is not None:
                    trace_directory = roundtrace_root / f"read-{index}"
                    child_environment["JETSPEC_ROUNDTRACE"] = str(trace_directory)
                    child_environment["JETSPEC_ROUNDTRACE_DEVICE"] = os.environ.get(
                        "JETSPEC_ROUNDTRACE_DEVICE", "1")
                    if os.environ.get("JETSPEC_ROUNDTRACE_EVENT_PAIRS"):
                        child_environment["JETSPEC_ROUNDTRACE_EVENT_PAIRS"] = os.environ[
                            "JETSPEC_ROUNDTRACE_EVENT_PAIRS"]
                if records is not None:
                    child_environment["PRED_SERVE_READ_RECORDS"] = str(records)
                done = (_run_child_with_rank_watch if resident is None else resident.read)(
                    command, child_environment, wall_timeout=args.child_wall_seconds)
                sys.stdout.write(done.stdout)
                sys.stdout.flush()
                benign_finalizer_tracebacks = assert_completed_run(
                    done.returncode, done.stdout, run_output)
                if trace_directory is not None:
                    build_roundtrace(trace_directory, child_environment, regime=args.regime,
                                     method_display=recipe["display_name"], placement=args.placement,
                                     recipe_name=recipe["name"], method=method,
                                     measured_path=run_output)
                manifest = assert_run_manifest(
                    run_output, done.stdout, expected.get("heads/thresholds", "absent"))
                assert_manifest_cpu_pin(manifest, cpu_pin)
                if method == "jetspecpp":
                    assert_kpred_manifest(
                        manifest, recipe["environment"].get("JETSPEC_LEGACY_KPRED_FORWARD", "0"))
                assert_manifest_artifacts(manifest, expected)
                assert_manifest_method(
                    manifest, {"jetspec.engines.jetspec_plus_plus.serving.api": "spec_async", "spec_sync": "spec"}.get(method, method))
                raw = json.loads(run_output.read_text(encoding="utf-8"))
                run = raw["metrics"]["serve_read"]
                measured = run["measured"]
                startup = startup_summary(done.stdout)
                if startup is not None:
                    print("SERVE_READ_STARTUP_SUMMARY " + json.dumps(startup), flush=True)
                if environment.get("JETSPEC_CHAIN_GRAPH") == "seg":
                    measured["chain"] = chain_round_summary(
                        done.stdout, measured["metrics"],
                        require_t0=environment.get("JETSPEC_CHAIN_GRAPH_CHECK") == "1",
                        t0_scope=chain_t0_scope(recipe))
                deadlines = _parse_deadlines(done.stdout).get("measured", [])
                proposals = measured.pop("proposals")
                chain = environment.get("JETSPEC_CHAIN_GRAPH") == "seg"
                if chain:
                    chain_rows = chain_proposals(done.stdout)
                    chain_keys = {(row["seq_id"], row["candidate_round"])
                                  for row in chain_rows}
                    proposals = [row for row in proposals
                                 if (row["seq_id"], row["candidate_round"]) not in chain_keys]
                    proposals.extend(chain_rows)
                populations = method_populations(
                    method, measured["metrics"], proposals, deadlines,
                    deadline_required=(environment.get("JETSPEC_DRAFT_DEADLINE") == "1"
                                       and not chain))
                from jetspec.engines.jetspec_plus_plus.serving.tools.eos_stop import compute_eos_stopped; from jetspec.engines.jetspec_plus_plus.serving.tools.eos_stop import ROUND_METHODS
                eos_stopped = (compute_eos_stopped(
                    method, raw["outputs"], measured["metrics"], tokens=args.tokens,
                    eos_token_ids=condition["eos_token_ids"], populations=populations,
                    batch=args.batch,
                    prompt_indices=([index for index, _ in prompt_token_lengths(
                        args.prompts_json, args.count)] if method == "jetspecpp" else None))
                    if method in ROUND_METHODS | {"ar"} else None)
                measured["metric_population"] = {"decode_tok_s": "to-cap",
                                                 "tau_to_cap": "to-cap",
                                                 "tps_to_cap": "to-cap"}
                measured["tps_to_cap"] = measured["decode_tok_s"]
                if eos_stopped is not None:
                    measured.update(eos_stopped)
                population_path = Path(str(run_output) + ".populations.json")
                population_path.write_text(json.dumps({
                    "schema": "jetspec.serve_populations/1", "method": method,
                    "output": str(run_output), "campaign_cold": index == 0,
                    "populations": populations,
                }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                anatomy_path = (write_anatomy(
                    run_output, method, args.regime, measured, populations,
                    refresh=environment.get("JETSPEC_ADAPTER_REFRESH", "off"),
                    passes=recipe.get("passes"),
                    rows_cap=environment.get("JETSPEC_ROWS_RULE_CAP",
                                             environment.get("JETSPEC_ROWS_CAP")),
                    trace_directory=trace_directory, eos_stopped=eos_stopped, cpu_pin=cpu_pin)
                    if args.anatomy == "on" else None)
                if anatomy_path:
                    measured["tau_to_cap"] = json.loads(
                        Path(anatomy_path).read_text(encoding="utf-8"))["tau"]
                else:
                    from jetspec.engines.jetspec_plus_plus.serving.tools.cutoff_anatomy import build_anatomy
                    measured["tau_to_cap"] = build_anatomy(
                        method, args.regime, measured, populations)["tau"]
                child_roles = sorted({
                    entry["role"] for entry in manifest["files"]
                    if entry["role"].startswith(("adapter/", "drafter/", "heads/"))
                    or entry["role"].startswith(("blockhead/", "blockhead_kit/"))
                    or entry["role"] in ("pins", "target_release_weights")})
                read = {
                    "process_index": index, "campaign_cold": index == 0,
                    "included_in_timing": index != 0, "output": str(run_output),
                    "rendezvous_port": child_environment.get("SSD_DIST_PORT"),
                    "load_label": launch_census["load_label"],
                    "launch_census": launch_census,
                    "benign_finalizer_tracebacks": benign_finalizer_tracebacks,
                    "population_audit": str(population_path), "cold": run["cold"],
                    "anatomy": anatomy_path,
                    "wire_records": None if wire_records is None else str(wire_records),
                    "jetspec.engines.jetspec_plus_plus.serving.roundtrace": None if trace_directory is None else {
                        "directory": str(trace_directory),
                        "report": str(trace_directory / "report.json"),
                        "figure": str(trace_directory / "round.svg"),
                    },
                    "measured": measured,
                    "manifest": {"path": str(run_output) + ".manifest.json",
                                 "manifest_sha256": manifest["manifest_sha256"],
                                 "content_sha256": manifest["content_sha256"],
                                 "thresholds_sha256": _manifest_role_sha(
                                     manifest, "heads/thresholds"),
                                 "draft_child_contributed_roles": child_roles},
                }
                if args.batch >= 1:
                    read["summary"] = {
                        "output_sha256": measured["output_sha256"],
                        "decode_tokens": measured["decode_tokens"],
                        "decode_tok_s": measured["decode_tok_s"],
                        "max_running_requests": measured["max_running_requests"],
                        "tau_to_cap": measured["tau_to_cap"],
                        "tps_to_cap": measured["tps_to_cap"],
                        **(eos_stopped or {}),
                        **{key: measured[key] for key in (
                            "per_request_decode_tok_s_mean", "per_request_decode_tok_s_p50",
                            "total_decode_tok_s", "gpus_used", "decode_tok_s_per_gpu")},
                        **({"chain": measured["chain"]} if "chain" in measured else {}),
                    }
                reads.append(read)
        result = {
            "schema": "jetspec.serve_campaign/1", "resolved_condition": condition,
            "campaign_cold": reads[0], "measured": reads[1:], "reads": reads,
            "benign_finalizer_tracebacks": sum(
                read["benign_finalizer_tracebacks"] for read in reads),
            "mps": None if mps is None else {
                "server_pid": mps.server_pid, "control_pid": mps.process.pid,
                "control_returncode": mps.process.returncode,
                "root": str(mps.root), "pipe": str(mps.pipe), "log": str(mps.logs)},
        }
        if args.temperature > 0:
            result["sampled_outputs"] = [
                {"phase": "cold" if read["campaign_cold"] else "measured",
                 "output": read["output"], "seed": args.seed,
                 "output_sha256": read["measured"]["output_sha256"]}
                for read in reads
            ]
        campaign_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("SERVE_READ_COMPLETE " + json.dumps({
            "campaign": str(campaign_path), "first_read": str(output),
            "processes": len(reads), "measured": len(result["measured"])}), flush=True)


def main():
    if "--internal-plan" in sys.argv:
        index = sys.argv.index("--internal-plan")
        if index + 1 >= len(sys.argv):
            raise SystemExit("--internal-plan requires a path")
        _child(sys.argv[index + 1])
        return
    _main(_parser().parse_args())


if __name__ == "__main__":
    main()

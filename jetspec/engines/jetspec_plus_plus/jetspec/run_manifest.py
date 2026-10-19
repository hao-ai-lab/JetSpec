"""Run manifest: bind a serve run's numbers to the exact inputs it loaded.

Stdlib only, so it imports without torch or flashinfer. One recorder per process.
Hashing happens when a file is first recorded, which call sites place at load time;
the ``*_once`` and ``observe`` entry points cost one set lookup after their first call,
so a read site inside the decode loop never hashes or touches the filesystem again.

The engine runs the drafter in a spawned process. When ``JETSPEC_RUN_MANIFEST_DIR`` is
set (the bench sets it; it is never required) every process rewrites
``<dir>/part-<pid>.json`` whenever its records change, and the parent folds the parts
in with ``merge_parts``.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from jetspec.engines.jetspec_plus_plus.legacy_env import environment_name
from jetspec.engines.jetspec_plus_plus.jetspec.config_values import config_values

SCHEMA = "jetspec.run_manifest/1"
PARTS_ENV = environment_name('engine', 'engine_settings.run_manifest_dir')
ENV_PREFIXES = ("JETSPEC_", "E04_", "SSD_", "W270_")
# Beyond the prefixes: every campaign knob (W240_WARMUP, W258B_TRACE, W328_DIGEST, ...), the
# thread counts, and the torch / triton / inductor / CUDA / flashinfer runtime and cache-dir
# variables, and the interpreter's own (PYTHONOPTIMIZE strips asserts; PYTHONPATH decides which
# tree's ssd is imported). A launcher that sets one of these changes the served condition without
# touching a JETSPEC_ name, so the manifest records them or a later diff cannot see the change.
ENV_PATTERN = re.compile(
    r"^(W[0-9]+[A-Z]?_|OMP_|MKL_|OPENBLAS_|TORCH|TRITON_|CUDA_|CUBLAS|PYTORCH_|NCCL_|FLASHINFER_|PYTHON)")
PACKAGES = ("torch", "transformers", "flashinfer", "triton")
# Read from installed metadata and never imported: the module loads CUDA libraries at import,
# which a CPU host sealing a manifest does not have. key -> (distribution names, import name).
DISTRIBUTIONS = {"sgl-kernel": (("sgl-kernel", "sgl_kernel"), "sgl_kernel")}
_CHUNK = 8 * 1024 * 1024

_lock = threading.RLock()
_sha_cache = {}
_files = []
_file_keys = set()
_once = set()
_values = {}
_value_conflicts = {}
_observed = {}
_env = None
_git = None
_packages = None
_run_token = None


def reset(*, run_config=None):
    """Forget every record; the sha cache survives because its key carries mtime and size."""
    global _env, _git, _packages, _run_token
    with _lock:
        _files.clear()
        _file_keys.clear()
        _once.clear()
        _values.clear()
        _value_conflicts.clear()
        _observed.clear()
        _env = _git = _packages = None
        _run_token = None
        directory = config_values(run_config).get('engine.engine_settings.run_manifest_dir')
        if directory:
            _ensure_run_token(directory)


def _ensure_run_token(directory):
    """Give spawned children the parent's run identity through the parts directory."""
    global _run_token
    pid = os.getpid()
    if _run_token is not None:
        return _run_token  # a fork keeps its parent's token
    marker = Path(directory) / ".run-token.json"
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("pid") == os.getppid():
            _run_token = previous["token"]
            return _run_token
    _run_token = secrets.token_hex(16)
    scratch = marker.with_name(f"{marker.name}.{pid}.tmp")
    scratch.write_text(json.dumps({"pid": pid, "token": _run_token}), encoding="utf-8")
    os.replace(scratch, marker)
    return _run_token


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def _jsonable(value):
    """Strict JSON: containers normalised, non-finite floats and unknown types as strings."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sha256_file(real):
    status = os.stat(real)
    key = (real, status.st_mtime_ns, status.st_size)
    cached = _sha_cache.get(key)
    if cached is None:
        digest = hashlib.sha256()
        with open(real, "rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK), b""):
                digest.update(chunk)
        cached = _sha_cache[key] = digest.hexdigest()
    return cached, status.st_size


def _describe(role, path):
    given = os.fspath(path)
    if not os.path.exists(given):
        raise FileNotFoundError(f"run manifest cannot record {role!r}: no such path: {given}")
    real = os.path.realpath(given)
    entry = dict(
        role=str(role), path=given, realpath=real,
        symlink_target=os.readlink(given) if os.path.islink(given) else None)
    if os.path.isdir(real):
        listing = []
        for folder, folders, names in os.walk(real, followlinks=True):
            folders.sort()
            for name in sorted(names):
                member = os.path.join(folder, name)
                sha, size = _sha256_file(os.path.realpath(member))
                listing.append([os.path.relpath(member, real).replace(os.sep, "/"), sha, size])
        listing.sort()
        entry.update(kind="dir", entries=listing, size=sum(row[2] for row in listing),
                     sha256=_digest([row[:2] for row in listing]))
    else:
        sha, size = _sha256_file(real)
        entry.update(kind="file", sha256=sha, size=size)
    return entry


def record_file(role, path, *, run_config=None):
    """Record one file or directory under ``role``; a missing path raises."""
    with _lock:
        entry = _describe(role, path)
        key = (entry["role"], entry["realpath"], entry["sha256"])
        if key not in _file_keys:
            _file_keys.add(key)
            _files.append(entry)
            _flush_part(run_config=run_config)
        return entry


def record_file_once(role, path, *, run_config=None):
    """``record_file`` for a read site that repeats: later calls are one set lookup."""
    key = (role, os.fspath(path))
    if key in _once:
        return
    with _lock:
        record_file(role, path, run_config=run_config)
        _once.add(key)


def record_model_dir(role, path, *, run_config=None):
    """Record a model directory: config.json, the safetensors index, every weight shard."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"run manifest cannot record {role!r}: no such directory: {root}")
    names = sorted(name for name in os.listdir(root)
                   if name == "config.json" or name.endswith(".safetensors")
                   or name.endswith(".safetensors.index.json"))
    if not names:
        raise FileNotFoundError(f"run manifest found no config or safetensors under {root}")
    return [record_file(f"{role}/{name}", root / name, run_config=run_config) for name in names]


def record_value(key, value, *, run_config=None):
    """Record one resolved value; a second, different value for a key is kept as a conflict."""
    value = _jsonable(value)
    with _lock:
        if key in _values and _values[key] != value:
            seen = _value_conflicts.setdefault(key, [_values[key]])
            if value not in seen:
                seen.append(value)
        else:
            _values[key] = value
        _flush_part(run_config=run_config)


def observe(key, value, *, run_config=None):
    """Add ``value`` to the set seen under ``key``; repeats are one set lookup."""
    seen = _observed.get(key)
    if seen is not None and value in seen:
        return
    with _lock:
        _observed.setdefault(key, set()).add(value)
        _flush_part(run_config=run_config)


def snapshot_env(environ, prefixes=ENV_PREFIXES, pattern=ENV_PATTERN, *, run_config=None):
    global _env
    if run_config is not None:
        from jetspec.engines.jetspec_plus_plus.run_config import config_environment
        supplied = config_environment(run_config)
        if supplied.get('CUBLAS_WORKSPACE_CONFIG') == '' and 'CUBLAS_WORKSPACE_CONFIG' not in (environ or {}):
            supplied.pop('CUBLAS_WORKSPACE_CONFIG')
        environ = {**(environ or {}), **supplied}
    with _lock:
        _env = {name: value for name, value in sorted((environ or {}).items())
                if (name.startswith(tuple(prefixes)) or pattern.match(name))
                and name != PARTS_ENV}
        _flush_part(run_config=run_config)
        return dict(_env)


def git_revision(repo_root, *, run_config=None):
    """HEAD sha and dirty flag from ``git``; on any failure null fields plus the reason."""
    global _git
    result = dict(root=os.fspath(repo_root), sha=None, dirty=None, reason=None)
    try:
        def run(*command):
            return subprocess.run(
                ["git", "-C", os.fspath(repo_root), *command], check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60).stdout
        result["sha"] = run("rev-parse", "HEAD").strip()
        result["dirty"] = bool(run("status", "--porcelain", "--untracked-files=no").strip())
    except (OSError, subprocess.SubprocessError) as error:
        detail = (getattr(error, "stderr", "") or str(error)).strip()
        result.update(sha=None, dirty=None, reason=f"{type(error).__name__}: {detail}")
    with _lock:
        _git = result
        _flush_part(run_config=run_config)
    return dict(result)


def _distribution_version(distributions, module):
    for distribution in distributions:
        try:
            return str(importlib.metadata.version(distribution))
        except importlib.metadata.PackageNotFoundError:
            continue
    try:
        found = importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        found = False
    return "unknown" if found else "absent"


def package_versions(names=PACKAGES, distributions=DISTRIBUTIONS, *, run_config=None):
    global _packages
    versions = {}
    for name in names:
        try:
            versions[name] = str(getattr(importlib.import_module(name), "__version__", "unknown"))
        except ImportError:
            versions[name] = "absent"
    for name, (candidates, module) in distributions.items():
        versions[name] = _distribution_version(candidates, module)
    with _lock:
        _packages = versions
        _flush_part(run_config=run_config)
    return dict(versions)


def _content(files):
    # Location-free: the same bytes under the same roles give the same digest from any
    # directory, so two runs can be compared on what they loaded and nothing else.
    return sorted([entry["role"], entry["sha256"], entry["size"]] for entry in files)


def as_dict():
    with _lock:
        files = sorted((dict(entry) for entry in _files),
                       key=lambda entry: (entry["role"], entry["realpath"], entry["sha256"]))
        observed = {key: sorted((_jsonable(value) for value in seen), key=_canonical)
                    for key, seen in sorted(_observed.items())}
        manifest = dict(
            schema=SCHEMA, files=files, values=dict(sorted(_values.items())),
            value_conflicts=dict(sorted(_value_conflicts.items())), observed=observed,
            env=_env, git=_git, packages=_packages)
        return _seal(manifest)


def _seal(manifest):
    manifest["content_sha256"] = _digest(_content(manifest["files"]))
    manifest["manifest_sha256"] = _digest(
        {key: value for key, value in manifest.items()
         if key not in ("manifest_sha256", "volatile")})
    manifest["volatile"] = dict(
        written_unix=time.time(), pid=os.getpid(), host=socket.gethostname(), argv=list(sys.argv))
    return manifest


def file_sha256(role, manifest=None):
    """sha256 values recorded under ``role``: one string, a sorted list, or None."""
    manifest = as_dict() if manifest is None else manifest
    found = sorted({entry["sha256"] for entry in manifest["files"] if entry["role"] == role})
    return None if not found else found[0] if len(found) == 1 else found


def merge_parts(directory, *, run_config=None):
    """Fold the records other processes wrote under ``directory`` into this recorder."""
    merged = []
    with _lock:
        token = _ensure_run_token(directory)
        for name in sorted(os.listdir(directory)):
            if not (name.startswith("part-") and name.endswith(".json")):
                continue
            if name == f"part-{os.getpid()}.json":
                continue
            part = json.loads((Path(directory) / name).read_text(encoding="utf-8"))
            if part.get("volatile", {}).get("run_token") != token:
                continue
            for entry in part["files"]:
                key = (entry["role"], entry["realpath"], entry["sha256"])
                if key not in _file_keys:
                    _file_keys.add(key)
                    _files.append(entry)
            for key, value in part["values"].items():
                record_value(key, value, run_config=run_config)
            for key, seen in part["value_conflicts"].items():
                for value in seen:
                    record_value(key, value, run_config=run_config)
            for key, seen in part["observed"].items():
                for value in seen:
                    observe(key, tuple(value) if isinstance(value, list) else value, run_config=run_config)
            merged.append(name)
        record_value("merged_parts", merged, run_config=run_config)
    return merged


def _flush_part(*, run_config=None):
    directory = config_values(run_config).get('engine.engine_settings.run_manifest_dir')
    if not directory:
        return
    _ensure_run_token(directory)
    write(os.path.join(directory, f"part-{os.getpid()}.json"))


def write(path):
    """Write the manifest atomically and return it."""
    manifest = as_dict()
    if _run_token is not None:
        manifest["volatile"]["run_token"] = _run_token
    path = os.fspath(path)
    scratch = f"{path}.{os.getpid()}.tmp"
    with open(scratch, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(scratch, path)
    return manifest

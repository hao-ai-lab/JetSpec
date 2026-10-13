"""JETSPEC_C1_COMPILE_PIN: pinned inductor and Triton autotune caches.

JETSPEC_C1_COMPILE uses max-autotune, which picks GEMM templates and Triton configs by
timing, so a cold inductor cache can serve a different kernel set (and different
TOKENS) in each process.  The pin is an inductor cache directory recorded once (the
FX-graph cache entries, whose output code carries the chosen kernels, and the autotune
``.best_config`` files next to them) and sealed with a manifest of every file's sha256.
The optional triton_autotune/ tree seals fla's timing-selected ``.autotune.json``
results too, including in eager mode. Legacy pins keep their original file digest.

  record  run the served config once with TORCHINDUCTOR_CACHE_DIR set to an empty
          directory, then ``python -m jetspec.engines.jetspec_plus_plus.draft.compile_pin seal <that dir> <pin dir>``
          (copies it without the triton/ binaries, writes manifest.json, prints the sha)
  serve   JETSPEC_C1_COMPILE_PIN=<pin dir> JETSPEC_C1_COMPILE_PIN_SHA=<sha>: at each
          rank's ModelRunner start, before its first compile, the pin is verified file by
          file against the manifest and the sha, then copied into the process's
          TORCHINDUCTOR_CACHE_DIR (never the pin itself).  The pin covers every compile
          of the recorded run, both ranks: the target's default-mode @torch.compile
          layers (models/layers) autotune their Triton configs by timing too, and a cold
          target cache made the served taps (so the drafts and TOKENS) bimodal.  The rank
          checks after its warm-up and graph captures (``process_check``).  After each window's graph captures, ``check`` refuses any FX-graph
          cache miss or bypass, fewer FX-graph hits than newly compiled graphs, and any
          new or changed cache file (a re-benchmarked Triton config writes a new
          .best_config).

Every refusal raises: a missing pin, a sha mismatch, another torch/triton, or a miss.
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

MANIFEST = "manifest.json"
AUTOTUNE = "triton_autotune"
SKIP_DIRS = ("triton", "__pycache__")
# An FX-graph cache hit replays the recorded compile's counter deltas (select_algorithm_*,
# benchmarking.*), so those counters cannot tell a hit from a re-autotune; a miss or bypass
# can, and a real re-benchmark also writes new autotune cache files (the file check).
COUNTERS = ("fxgraph_cache_miss", "fxgraph_cache_bypass")


def tuning_choice(kind, key, choose):
    """Record once; a sealed pin replays the full plan and refuses an unseen shape."""
    pin = os.environ.get('JETSPEC_C1_COMPILE_PIN', '')
    record = os.environ.get('JETSPEC_C1_COMPILE_PIN_RECORD', '0') == '1'
    if not pin and not record:
        return choose()
    identity = json.dumps([kind, key], sort_keys=True, separators=(',', ':'))
    token = (pin, os.environ.get('JETSPEC_C1_COMPILE_PIN_SHA', ''),
             os.environ.get('TORCHINDUCTOR_CACHE_DIR'), identity)
    if token in _TUNING_CHOICES:
        return _TUNING_CHOICES[token]
    name = hashlib.sha256(identity.encode()).hexdigest() + '.json'
    root = Path(pin) / 'cache' if pin else Path(os.environ['TORCHINDUCTOR_CACHE_DIR'])
    path = root / 'tuning' / name
    if pin:
        sha = os.environ.get('JETSPEC_C1_COMPILE_PIN_SHA', '')
        # Verification at installation covers all choices, including choices never used.
        verification_token = (os.path.realpath(pin), sha)
        if verification_token not in _TUNING_VERIFIED:
            verify(pin, sha)
            _TUNING_VERIFIED.add(verification_token)
    if path.exists():
        payload = json.loads(path.read_text())
        if payload['identity'] != identity:
            raise RuntimeError('C1 COMPILE PIN tuning identity mismatch')
        _TUNING_CHOICES[token] = payload['choice']
        return payload['choice']
    if pin:
        raise RuntimeError(f'C1 COMPILE PIN tuning miss: {kind} {key}')
    choice = choose()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(dict(identity=identity, choice=choice), sort_keys=True) + '\n'
    # O_EXCL also prevents the two ranks from silently recording different draws.
    staging = path.with_suffix(f'.{os.getpid()}.tmp')
    staging.write_text(data)
    try:
        try:
            os.link(staging, path)
        except FileExistsError:
            if json.loads(path.read_text())['choice'] != choice:
                raise RuntimeError(f'C1 COMPILE PIN conflicting tuning draw: {kind} {key}')
    finally:
        staging.unlink()
    _TUNING_CHOICES[token] = choice
    return choice


_TUNING_VERIFIED = set()
_TUNING_CHOICES = {}


def install_triton_tuning_pin():
    """Replay explicit Triton autotuners without a timing draw in verify mode."""
    if not (os.environ.get('JETSPEC_C1_COMPILE_PIN', '') or
            os.environ.get('JETSPEC_C1_COMPILE_PIN_RECORD', '0') == '1'):
        return
    import inspect
    try:
        from triton.runtime.autotuner import Autotuner
    except ModuleNotFoundError as error:
        if error.name not in ('triton', 'triton.runtime', 'triton.runtime.autotuner'):
            raise
        return  # CPU cache validation has no Triton autotuners to install.
    original = Autotuner.run
    if getattr(original, '_compile_pin', False):
        return

    def run(tuner, *args, **kwargs):
        if not (os.environ.get('JETSPEC_C1_COMPILE_PIN', '') or
                os.environ.get('JETSPEC_C1_COMPILE_PIN_RECORD', '0') == '1'):
            return original(tuner, *args, **kwargs)
        arguments = dict(zip(tuner.arg_names, args), **kwargs)
        values = [arguments[name] for name in tuner.keys if name in arguments]
        dtypes = [str(value.dtype) for value in arguments.values() if hasattr(value, 'dtype')]
        source = hashlib.sha256(inspect.getsource(tuner.base_fn).encode()).hexdigest()
        identity = [source, values, dtypes]
        results = []
        def choose():
            results.append(original(tuner, *args, **kwargs))
            return tuner.best_config.all_kwargs()
        choice = tuning_choice('triton_autotune', identity, choose)
        if results:
            return results[0]
        configs = tuner.configs
        matching = [config for config in configs if config.all_kwargs() == choice]
        if len(matching) != 1:
            raise RuntimeError(f'C1 COMPILE PIN Triton config miss: {identity}')
        tuner.configs = matching
        try:
            return original(tuner, *args, **kwargs)
        finally:
            tuner.configs = configs
    run._compile_pin = True
    Autotuner.run = run


def _files(root):
    """{relative path: sha256} of every cache file under root, without the triton
    binaries (compiled deterministically from the pinned source) and lock files."""
    out = {}
    for base, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not (base == root and d in SKIP_DIRS)
                         and d != "__pycache__")
        for name in sorted(names):
            if name.endswith(".lock") or name.endswith(".tmp"):
                continue
            path = os.path.join(base, name)
            with open(path, "rb") as handle:
                out[os.path.relpath(path, root)] = hashlib.sha256(handle.read()).hexdigest()
    return out




def _autotune_files(root):
    files = {}
    for base, _, names in os.walk(root):
        for name in sorted(names):
            if name.endswith(".autotune.json"):
                path = os.path.join(base, name)
                with open(path, "rb") as handle:
                    files[os.path.relpath(path, root)] = hashlib.sha256(handle.read()).hexdigest()
    return files


def _cache_files(cache, triton=None):
    files = _files(cache)
    if triton is not None:
        files.update({os.path.join(AUTOTUNE, path): sha
                      for path, sha in _autotune_files(triton).items()})
    return files


def triton_cache_dir():
    home = os.environ.get("TRITON_HOME") or os.path.expanduser("~")
    return os.path.realpath(os.environ.get("TRITON_CACHE_DIR") or
                            os.path.join(home, ".triton", "cache"))




def seal(source, pin, triton_source=None):
    """Copy the recorded inductor cache and optional Triton autotune results into a pin."""
    if os.path.exists(pin):
        raise FileExistsError(f"C1 COMPILE PIN: {pin} exists; a pin is written once")
    if triton_source is not None and not os.path.isdir(triton_source):
        raise FileNotFoundError(f"C1 COMPILE PIN: no Triton cache directory {triton_source}")
    shutil.copytree(source, os.path.join(pin, "cache"),
                    ignore=shutil.ignore_patterns("*.lock", "*.tmp", "__pycache__"))
    triton_dir = os.path.join(pin, "cache", "triton")
    if os.path.isdir(triton_dir):
        shutil.rmtree(triton_dir)
    if triton_source is not None:
        os.makedirs(os.path.join(pin, AUTOTUNE))
        for path in _autotune_files(triton_source):
            target = os.path.join(pin, AUTOTUNE, path)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(os.path.join(triton_source, path), target)
    files = _cache_files(os.path.join(pin, "cache"), os.path.join(pin, AUTOTUNE))
    if not files:
        raise ValueError(f"C1 COMPILE PIN: {source} holds no cache files")
    versions = _versions()
    sha = digest(files, versions)
    with open(os.path.join(pin, MANIFEST), "w") as handle:
        json.dump({"sha": sha, "versions": versions, "files": files}, handle, indent=1,
                  sort_keys=True)
    return sha


def verify(pin, sha):
    """The pin's manifest and files, checked against the expected sha; returns files."""
    manifest_path = os.path.join(pin, MANIFEST)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"C1 COMPILE PIN: no {MANIFEST} in {pin}")
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    files = _cache_files(os.path.join(pin, "cache"), os.path.join(pin, AUTOTUNE))
    actual = digest(files, manifest["versions"])
    if not (manifest["sha"] == sha == actual) or files != manifest["files"]:
        raise ValueError(f"C1 COMPILE PIN: sha mismatch (expected {sha}, manifest "
                         f"{manifest['sha']}, files {actual})")
    if manifest["versions"] != _versions():
        raise ValueError(f"C1 COMPILE PIN: recorded under {manifest['versions']}, "
                         f"running {_versions()}")
    return files


_INSTALLED = {}




_PROCESS = {}


def process_install():
    """JETSPEC_C1_COMPILE_PIN (+ _SHA) at a rank's start: install the pin before any
    compile and snapshot the miss counters.  No-op when the pin is unset."""
    install_triton_tuning_pin()
    pin = os.environ.get("JETSPEC_C1_COMPILE_PIN", "")
    if not pin:
        return None
    sha = os.environ.get("JETSPEC_C1_COMPILE_PIN_SHA", "")
    if not sha:
        raise ValueError("C1 COMPILE PIN requires JETSPEC_C1_COMPILE_PIN_SHA")
    from torch._dynamo.utils import counters
    cache, files = install(pin, sha)
    triton = triton_cache_dir() if os.path.isdir(os.path.join(pin, AUTOTUNE)) else None
    _PROCESS.update(cache=cache, triton=triton, files=files, sha=sha,
                    before={key: counters["inductor"][key] for key in COUNTERS})
    return cache


def process_check(where):
    """Refuse any FX-graph miss or bypass in this process since ``process_install``, and
    any cache file the pin does not hold (the other rank's misses land in the shared dir
    too).  No-op when the pin is unset."""
    if not _PROCESS:
        return
    from torch._dynamo.utils import counters
    deltas = {key: counters["inductor"][key] - _PROCESS["before"][key] for key in COUNTERS}
    pinned = _PROCESS["files"]
    now = _cache_files(_PROCESS["cache"], _PROCESS["triton"])
    fresh = sorted(path for path, sha in now.items() if pinned.get(path) != sha)
    if any(deltas.values()) or fresh:
        raise RuntimeError(f"C1 COMPILE PIN miss at {where}: {deltas}, "
                           f"{len(fresh)} cache files outside the pin {fresh[:5]}")
    print(f"W328_PIN process {where} sha {_PROCESS['sha'][:12]} fxgraph hits "
          f"{counters['inductor']['fxgraph_cache_hit']} files {len(now)} "
          f"triton_cache {triton_cache_dir()}", flush=True)


class Pin:
    """The serve side: install the verified pin into this process's inductor cache and
    check that the compiled calls hit it."""

    def __init__(self, pin, sha):
        self.pin = os.path.realpath(pin)
        self.cache, files = install(self.pin, sha)
        self.triton = triton_cache_dir() if os.path.isdir(os.path.join(self.pin, AUTOTUNE)) else None
        self.sha = sha
        self.pinned = files
        self.checks = 0
        self.arm()

    def arm(self):
        """Snapshot the counters and the cache files before a compile window."""
        from torch._dynamo.utils import counters
        self.before = {key: counters["inductor"][key] for key in COUNTERS}
        self.hits = counters["inductor"]["fxgraph_cache_hit"]
        self.graphs = counters["stats"]["unique_graphs"]
        self.files = _cache_files(self.cache, self.triton)
        changed = {path for path, sha in self.pinned.items() if self.files.get(path) != sha}
        changed.update(path for path, sha in self.files.items()
                       if self.triton is not None and path.startswith(AUTOTUNE + os.sep)
                       and self.pinned.get(path) != sha)
        if changed:
            raise RuntimeError(f"C1 COMPILE PIN: {len(changed)} pinned files differ in "
                               f"{self.cache}: {sorted(changed)[:5]}")

    def check(self, where):
        """Refuse a miss since ``arm``: an FX-graph miss or bypass, a newly compiled graph
        without an FX-graph hit, or a new or changed cache file."""
        from torch._dynamo.utils import counters
        deltas = {key: counters["inductor"][key] - self.before[key] for key in COUNTERS}
        now = _cache_files(self.cache, self.triton)
        fresh = sorted(path for path, sha in now.items() if self.files.get(path) != sha)
        hits = counters["inductor"]["fxgraph_cache_hit"] - self.hits
        graphs = counters["stats"]["unique_graphs"] - self.graphs
        if any(deltas.values()) or fresh or hits < graphs:
            raise RuntimeError(f"C1 COMPILE PIN miss at {where}: {deltas}, graphs {graphs} "
                               f"hits {hits}, "
                               f"{len(fresh)} new or changed cache files {fresh[:5]}")
        self.checks += 1
        print(f"W328_PIN hit {where} sha {self.sha[:12]} checks {self.checks} "
              f"graphs {graphs} fxgraph hits {hits} triton_cache {triton_cache_dir()}", flush=True)
        self.arm()


_DIGESTS = {}


def digest_log(tag, env="W328_DIGEST", **tensors):
    """<env>=<n>: print the byte sha of each tensor for the first n calls (a diagnostic
    that locates where two processes' draft computations first differ)."""
    limit = int(os.environ.get(env, "0") or 0)
    count = _DIGESTS.get(env, 0)
    if count >= limit:
        return
    _DIGESTS[env] = count = count + 1
    parts = []
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        import torch
        flat = tensor.detach().cpu().reshape(-1)
        data = torch.empty_like(flat, memory_format=torch.contiguous_format).copy_(flat)
        data = data.view(torch.uint8)
        parts.append(f"{name}={hashlib.sha256(data.numpy().tobytes()).hexdigest()[:12]}"
                     f"{list(tensor.shape)}")
    print(f"W328_DIGEST {count} {tag} " + " ".join(parts), flush=True)


_DEFERRED = {}


def digest_defer(tag, env, **tensors):
    """<env>=<n>: keep device clones of the tensors for the first n calls (no host sync) and
    print their digests at call n + 20, so the diagnostic does not change the timing of
    the calls it records."""
    limit = int(os.environ.get(env, "0") or 0)
    if not limit:
        return
    state = _DEFERRED.setdefault(env, {"calls": 0, "kept": []})
    state["calls"] += 1
    if state["calls"] <= limit:
        state["kept"].append((state["calls"], tag, {name: tensor.detach().clone()
                                                    for name, tensor in tensors.items()
                                                    if tensor is not None}))
    elif state["calls"] == limit + 20:
        flush_env = f"{env}_FLUSH"
        os.environ[flush_env] = str(limit)
        for _, kept_tag, kept in state["kept"]:
            digest_log(kept_tag, env=flush_env, **kept)
        state["kept"] = []




if __name__ == "__main__":
    if len(sys.argv) in (4, 5) and sys.argv[1] == "seal":
        print(seal(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) == 5 else None))
    elif len(sys.argv) == 4 and sys.argv[1] == "verify":
        print(len(verify(sys.argv[2], sys.argv[3])), "files OK")
    else:
        raise SystemExit("usage: compile_pin.py seal <recorded cache> <pin> [Triton cache] | verify <pin> <sha>")

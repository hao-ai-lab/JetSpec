"""Opt-in, content-partitioned compiler caches for fresh serving processes."""
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

FIELDS = ('arch', 'driver', 'cuda', 'torch', 'triton', 'flashinfer', 'cutedsl',
          'python', 'venv_lock_hash')
DIRS = dict(TRITON_CACHE_DIR='triton', TORCHINDUCTOR_CACHE_DIR='inductor',
            TORCH_EXTENSIONS_DIR='torch_extensions', FLASHINFER_WORKSPACE_BASE='flashinfer',
            SSD_CUTE_DSL_CACHE_ROOT='cute', CUDA_CACHE_PATH='cuda', TVM_FFI_CACHE_DIR='tvm-ffi')


def toolchain_key(inputs):
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def partition(kind, inputs, root):
    """P1 naming implemented locally: the core is not installed in the serving kit."""
    return str(Path(root) / kind / toolchain_key(inputs))


def resolve(shared, job, inputs, source_sha, *, keyed_root=False):
    missing = [name for name in FIELDS if not inputs.get(name)]
    if not source_sha:
        missing.append('source_sha')
    persistent = bool(shared) and not missing
    key = '-'.join(re.sub(r'[^A-Za-z0-9_.]', '_', str(inputs[name]))
                   for name in FIELDS) if persistent else None
    root = (Path(shared) if keyed_root else Path(shared) / key) if persistent else Path(job)
    env = {name: str(root / relative) for name, relative in DIRS.items()}
    if persistent:
        # load_inline and FlashInfer use names rather than cross-process source keys.
        for name in ('TORCH_EXTENSIONS_DIR', 'FLASHINFER_WORKSPACE_BASE'):
            env[name] = partition(DIRS[name], dict(source_sha256=source_sha), root)
        env.update(TORCHINDUCTOR_FX_GRAPH_CACHE='1', TORCHINDUCTOR_AUTOGRAD_CACHE='1',
                   CUDA_CACHE_MAXSIZE=str(4 * 1024 ** 3))
    return dict(root=str(root), key=key, source_sha256=source_sha, persistent=persistent,
                reason=('missing cache key inputs: ' + ', '.join(missing)) if shared and missing
                       else ('CRUCIBLE_L4_ROOT unset' if not shared else None), environment=env)


def probe(root, environment):
    import torch
    cuda_home = environment.get('CUDA_HOME')
    compiler = str(Path(cuda_home) / 'bin/nvcc') if cuda_home else shutil.which('nvcc')
    if not compiler:
        raise ValueError('nvcc missing from the native environment')
    nvcc = subprocess.check_output([compiler, '--version'], text=True)
    driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
                                      '--format=csv,noheader'], text=True).splitlines()
    if len(set(driver)) != 1:
        raise ValueError('driver version is not unique')
    records = []
    for dist in metadata.distributions():
        for item in dist.files or ():
            if str(item).endswith('.dist-info/RECORD'):
                records.append((str(dist.metadata['Name']), Path(dist.locate_file(item)).read_bytes()))
    if not records:
        raise ValueError('no distribution RECORDs')
    digest = hashlib.sha256(os.path.realpath(sys.executable).encode())
    for name, data in sorted(records):
        digest.update(name.encode() + b'\0' + data + b'\0')
    from jetspec.engines.jetspec_plus_plus.read.legacy_recipes import serving_lock_bytes
    live_lock = Path(root) / 'engine/uv.lock'
    digest.update(live_lock.read_bytes() if live_lock.is_file() else serving_lock_bytes(root))
    digest.update((Path(root) / 'serve/pyproject.toml').read_bytes())
    source = hashlib.sha256()
    for path in sorted((Path(root) / 'serve').rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.h', '.hpp', '.c', '.cc', '.cpp', '.cu', '.cuh'):
            source.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes() + b'\0')
    source.update(json.dumps({name: environment.get(name) for name in
        ('CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS', 'TORCH_CUDA_ARCH_LIST')},
        sort_keys=True).encode())
    from jetspec.engines.jetspec_plus_plus.legacy_env import environment_name
    inputs = dict(arch=environment.get(environment_name('hardware', 'hardware_settings.cuda_arch')) or environment.get('TORCH_CUDA_ARCH_LIST'),
                  driver=driver[0], cuda=hashlib.sha256((nvcc + str(torch.version.cuda)).encode()).hexdigest()[:16],
                  torch=torch.__version__, triton=metadata.version('triton'),
                  flashinfer=metadata.version('flashinfer-python'),
                  cutedsl=metadata.version('nvidia-cutlass-dsl'), python=platform.python_version(),
                  venv_lock_hash=digest.hexdigest())
    return inputs, source.hexdigest()


def configure(root, job, environment):
    shared = environment.get('CRUCIBLE_L4_ROOT')
    if not shared:
        return None
    try:
        inputs, source = probe(root, environment)
        receipt = resolve(shared, job, inputs, source, keyed_root=True)
        receipt['inputs'] = inputs
        receipt['toolchain_key'] = environment.get('CRUCIBLE_TOOLCHAIN_KEY') or toolchain_key(inputs)
        if receipt['persistent']:
            target = Path(receipt['root'])
            target.mkdir(parents=True, exist_ok=True)
            identity = target / 'toolchain-identity.json'
            if not identity.exists():
                if any(target.iterdir()):
                    raise ValueError('cache identity missing in a nonempty L4 root')
                staging = target / f'identity.{os.getpid()}.tmp'
                staging.write_text(json.dumps(inputs, sort_keys=True) + '\n')
                try:
                    try:
                        os.link(staging, identity)
                    except FileExistsError:
                        pass
                finally:
                    staging.unlink()
            if json.loads(identity.read_text()) != inputs:
                raise ValueError('cache toolchain identity mismatch')
        if receipt['persistent']:
            receipt['environment'].update(CRUCIBLE_L4_ROOT=receipt['root'],
                CRUCIBLE_TOOLCHAIN_KEY=receipt['toolchain_key'], MAX_JOBS='16',
                CMAKE_BUILD_PARALLEL_LEVEL='16', NVCC_THREADS='1')
            if (environment.get('CRUCIBLE_FLASHINFER_VERSION') == inputs['flashinfer']
                    and environment.get('FLASHINFER_WORKSPACE_BASE')):
                receipt['environment']['FLASHINFER_WORKSPACE_BASE'] = environment['FLASHINFER_WORKSPACE_BASE']
                receipt['flashinfer_cache'] = 'declared version match'
            else:
                receipt['flashinfer_cache'] = 'source partition'
            receipt['implementation'] = 'local P1 partition naming; crucible_core not installed'
    except (OSError, ValueError, metadata.PackageNotFoundError, subprocess.CalledProcessError) as exc:
        receipt = resolve(shared, job, {}, None)
        receipt['reason'] = str(exc)
    print('CRUCIBLE_CACHE ' + json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def cleanup_locks(root):
    """Only remove locks with an explicit, locally dead PID. Unknown owners stay locked."""
    removed = []
    for path in Path(root).rglob('*'):
        if not path.is_file() or not (path.name == 'lock' or path.suffix == '.lock'):
            continue
        try:
            text = path.read_text().strip()
            if not text and path.name == 'lock' and sys.platform == 'linux' and shutil.which('fuser'):
                # torch FileBaton holds an open fd until release. A killed builder leaves
                # an empty sentinel; flock-based *.lock files instead release in the OS.
                inode = path.stat().st_ino
                owners = subprocess.run(['fuser', str(path)], capture_output=True)
                if owners.returncode == 1 and not owners.stderr.strip() and path.stat().st_ino == inode:
                    path.unlink()
                    removed.append(str(path))
                continue
            owner = json.loads(text)
            pid = owner.get('pid') if isinstance(owner, dict) else owner
            if type(pid) is not int or pid <= 0:
                continue
            os.kill(pid, 0)
        except ProcessLookupError:
            # Re-read before unlinking: a concurrent owner may have replaced the lock.
            if path.read_text().strip() == text:
                path.unlink()
                removed.append(str(path))
        except (OSError, ValueError, UnicodeError):
            continue
    return removed


def snapshot(environment):
    """Content snapshot outside the measured window; lock and staging files do not count."""
    files = {}
    for name in DIRS:
        if name not in environment or name == 'CUDA_CACHE_PATH':
            continue
        for path in Path(environment[name]).rglob('*'):
            if (not path.is_file() or path.name == 'lock' or path.suffix in ('.lock', '.tmp', '.pyc')):
                continue
            try:
                files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            except FileNotFoundError:
                pass
    return files


def changes(before, after):
    return sorted(path for path, sha in after.items() if before.get(path) != sha)

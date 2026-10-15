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

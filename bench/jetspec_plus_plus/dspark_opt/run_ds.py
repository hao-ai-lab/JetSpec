"""Run the Table 3 DSpark (DS) arm with extra knobs and compare against the stock DS reads.

The plan, prompts, EOS rule A reduction and validation are the Table 3 protocol's; the only change is
that the knobs given with --knob are appended to the reader's argv (the protocol forbids them for its
own DS arm). Every group is checked for token parity against the stock read at the same offset.

usage:
  python bench/jetspec_plus_plus/dspark_opt/run_ds.py --label opt --host-values host.json --device 0 \
      --stock-dir bench/outputs/dspark_gsm8k_b200 --output-dir bench/outputs/dspark_opt \
      --knob JETSPEC_BLOCKHEAD_ATTN_LEN=1 ...
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from bench.jetspec_plus_plus.paper import command, protocol  # noqa: E402

protocol.STOCK_DS_KNOBS = ()  # this harness exists to measure non-stock DS knobs


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--label', required=True)
    p.add_argument('--knob', action='append', default=[])
    p.add_argument('--offsets', default='0,32,64,96')
    p.add_argument('--device', default='0', help='GPU index, or two comma-separated indices with --placement tp2')
    p.add_argument('--placement', choices=('one_gpu', 'tp2'), default='one_gpu',
                   help='tp2: the target tensor-parallel over two GPUs, the DSpark draft on rank 0')
    p.add_argument('--host-values', type=Path, required=True)
    p.add_argument('--artifacts-dir', type=Path)
    p.add_argument('--stock-dir', type=Path, required=True, help='Table 3 output dir holding the stock DS reads')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--reduce-only', action='store_true')
    p.add_argument('--retries', type=int, default=20)
    p.add_argument('--cpu-pin', default=None, help='override the plan\'s --cpu-pin (the protocol uses off for DS)')
    p.add_argument('--share-gpu', action='store_true',
                   help='tolerate processes already on the GPU at launch (a shared GPU)')
    p.add_argument('--exclude-gpus', default='', help='comma-separated GPU indices --device auto* never picks')
    p.add_argument('--numa-groups', default='',
                   help='GPU groups on one NUMA node, e.g. "0,1,2,3;4,5,6": auto2 prefers a pair within one')
    return p


NUMA_GROUPS = ()
EXCLUDE = set()


def free_gpus():
    busy = set(subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'],
                                       text=True).split())
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    return [int(i) for i, u in (r.split(', ') for r in rows.splitlines()) if u not in busy and int(i) not in EXCLUDE]


def wait_for_devices(spec, wait_s=10800):
    """'auto1'/'auto2': wait for one/two GPUs with no compute processes (a pair on one NUMA node first)."""
    if not spec.startswith('auto'):
        return spec
    need, end = int(spec[4:]), time.time() + wait_s
    while time.time() < end:
        free = free_gpus()
        for group in NUMA_GROUPS or (tuple(free),):
            pick = [g for g in free if g in group][:need]
            if len(pick) == need:
                return ','.join(map(str, pick))
        # A cross-NUMA pair measured ~0.3 ms slower per verify; take one only after 20 minutes.
        if len(free) >= need and time.time() > end - wait_s + 1200:
            return ','.join(map(str, free[:need]))
        time.sleep(30)
    raise SystemExit('no free GPUs')


def run_read(directory, env, share=False):
    """True on success, False when the census found the GPUs occupied; raises on other failures."""
    with (directory / 'read.log').open('w') as log:
        rc = subprocess.run([sys.executable, '-m', 'bench.jetspec_plus_plus.dspark_opt.read_shared' if share else 'bench.jetspec_plus_plus.paper.read',
                             str(directory / 'plan.json')], cwd=ROOT, stdout=log,
                            stderr=subprocess.STDOUT, env=env).returncode
    if rc == 0:
        return True
    if 'requested device occupied' not in (directory / 'read.log').read_text():
        raise SystemExit(f'read failed (rc={rc}); see {directory / "read.log"}')
    return False


def parity(directory, stock):
    new = json.loads((directory / 'read.json').read_text())['outputs']
    old = json.loads((stock / 'read.json').read_text())['outputs']
    same, first = 0, []
    for i, (a, b) in enumerate(zip(new, old)):
        a, b = a['token_ids'], b['token_ids']
        if a == b:
            same += 1
        else:
            k = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            first.append((i, k, len(a), len(b)))
    return same, first


def main():
    args = parser().parse_args()
    global NUMA_GROUPS, EXCLUDE
    NUMA_GROUPS = tuple(tuple(int(g) for g in group.split(',')) for group in args.numa_groups.split(';') if group)
    EXCLUDE = {int(g) for g in args.exclude_gpus.split(',') if g}
    if args.placement == 'tp2':
        if not args.device.startswith('auto') and len(args.device.split(',')) != 2:
            raise SystemExit('--placement tp2 needs --device A,B')
        method, recipe, _ = protocol.ARMS['8b']['DS']
        protocol.ARMS['8b']['DS'] = (method, recipe, 'tp2')
        resolve = protocol.entry.resolve_read

        def resolve_tp2(names, overrides, **kw):
            # The protocol sizes hardware.devices for pair/one_gpu only; tp2 also has two ranks.
            overrides = [o if not o.startswith('hardware.devices=') else 'hardware.devices=[0, 1]'
                         for o in overrides]
            return resolve(names, overrides, **kw)
        protocol.entry.resolve_read = resolve_tp2
    out = args.output_dir.resolve() / args.label
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT), os.environ.get('PYTHONPATH')])))
    cargs = argparse.Namespace(output_dir=out, reduce_only=args.reduce_only, host_values=args.host_values,
                               artifacts_dir=args.artifacts_dir)
    reads = []
    for offset in [int(o) for o in args.offsets.split(',')]:
        directory = out / 'gsm8k' / 'DS' / f'p{offset:03d}'
        for attempt in range(1 if args.reduce_only else args.retries):
            # Pick the GPUs per attempt: a shared machine can take a free pair between check and launch.
            devices = wait_for_devices(args.device) if not args.reduce_only else '0,1'
            job = dict(model='8b', dataset='gsm8k', arm='DS', offset=offset,
                       devices=devices if args.placement == 'tp2' else devices.split(',')[0])
            source, argv = command.prepare(job, directory, cargs)
            for knob in args.knob:
                argv += ['--knob', knob]
            if args.cpu_pin is not None:
                argv[argv.index('--cpu-pin') + 1] = args.cpu_pin
            plan = command.launch_plan(job, argv)
            if args.reduce_only:
                break
            command.write_json(directory / 'plan.json', plan)
            print(f'[{args.label}] DS p{offset:03d} on GPUs {devices}', flush=True)
            if run_read(directory, env, args.share_gpu):
                break
            print(f'  GPUs {devices} occupied at launch, retry {attempt + 1}', flush=True)
            time.sleep(30)
        else:
            raise SystemExit('GPUs stayed occupied')
        result = command.reduce_directory(directory, job, source)
        same, diff = parity(directory, args.stock_dir / 'gsm8k' / 'DS' / f'p{offset:03d}')
        result['parity'] = dict(identical=same, of=32, diverged=diff[:8])
        reads.append(result)
        v = result['values']
        print(f'  tok/s {v["tok_s"]:.1f}  tau {v["tau"]:.3f}  round {v["round"]:.3f}  verify {v["verify"]:.3f}  '
              f'exposed {v["round"] - v["verify"]:.3f}  parity {same}/32', flush=True)
    command.write_json(out / 'reads.json', reads)
    if len(reads) == 4:
        stock_reads = json.loads((args.stock_dir / 'table3.json').read_text())['reads']
        rows = command.pool([r for r in stock_reads if r['arm'] == 'AR'] + reads)
        ar = next(r for r in rows if r['arm'] == 'AR')
        row = next(r for r in rows if r['arm'] == 'DS')
        command.write_json(out / 'pooled.json', row)
        print(f'[{args.label}] pooled: tok/s {row["tok_s"]:.1f} ({row["speedup"]:.2f}x vs stock AR {ar["tok_s"]:.1f})  '
              f'tau {row["tau"]:.3f}  round {row["round"]:.3f}  verify {row["verify"]:.3f}  exposed {row["exposed"]:.3f}  '
              f'parity {sum(r["parity"]["identical"] for r in reads)}/128')


if __name__ == '__main__':
    main()

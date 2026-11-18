"""paper.read with a shared-GPU allowance: compute processes already on the requested GPUs when the read
starts are tolerated by the runner's device census (the user approved sharing); new ones still refuse."""
import subprocess
import sys

from jetspec.engines.jetspec_plus_plus.launch import runner
from bench.jetspec_plus_plus.paper import read

_census = runner._gpu_census
_out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
_present = {(l.split(',')[0].strip(), int(l.split(',')[1])) for l in _out.splitlines() if l.strip()}


def _shared(uuids, allowed_pids=()):
    extra = tuple(pid for uuid, pid in _present if uuid in uuids)
    return _census(uuids, allowed_pids=tuple(allowed_pids) + extra)


runner._gpu_census = _shared
if __name__ == '__main__':
    raise SystemExit(read.main(sys.argv[1]))

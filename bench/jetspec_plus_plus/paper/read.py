"""Execute exactly the public configuration recorded by the Table 3 planner."""
import json
import os
from pathlib import Path
import sys

from jetspec.engines.jetspec_plus_plus.launch import cli
from jetspec.engines.jetspec_plus_plus.launch import runner
from jetspec.engines.jetspec_plus_plus.launch.entry import granted_devices


def main(path):
    plan = json.loads(Path(path).read_text())
    args = runner._parser().parse_args(plan['argv'][3:])
    # Plan device indices count from the first card the scheduler grants, as the serving
    # entry's do. The runner reads physical cards, so the indices map here.
    args.devices = granted_devices(args.devices, os.environ.get('CUDA_VISIBLE_DEVICES'))
    args.resolved_recipe = plan['runtime_recipe']
    args.config_read = plan['composition']
    # Published calibration directories need their registration-name alias.
    host = (json.loads(Path(args.host_values).read_text()) if args.host_values else
            {'schema': runner.HOST_VALUES_SCHEMA, 'environment': {}, 'artifacts': {}})
    key = args.resolved_recipe.get('bundle_artifact')
    alias = Path(path).parent / 'bundles' / key if key else None
    if alias and alias.is_symlink() and key in host['artifacts']:
        if alias.resolve() == Path(host['artifacts'][key]).resolve():
            host['artifacts'][key] = str(alias)
    cli.alias_heads_bundle(host, args.resolved_recipe, Path(path).parent)
    local_host = Path(path).parent / 'resolved-host.json'
    local_host.write_text(json.dumps(host) + '\n')
    args.host_values = local_host
    return runner._main(args)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))

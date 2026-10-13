"""Run one public-output mutation per converted comparison file.

Each wrapper changes one returned scalar or tensor element. Success means every
selected test rejects that output with an assertion, with the wrapper exercised.
"""
import argparse
import dataclasses
import importlib
import json
from pathlib import Path
import sys

import pytest
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE.parents[1]))
sys.path.insert(0, str(BASE))
PREFIX = 'jetspec.methods.jetspec_plus_plus.'
CASES = {
    'adapter': ('test_hybrid_norm_matches_golden', PREFIX + 'models.adapter', 'HybridC1RMSNorm.forward', None),
    'blend': ('test_blend', 'test', 'blend_features', None),
    'context': ('test_context_equal', 'test', 'build_hypothesis_contexts', None),
    'contracts': ('test_emit_block', 'test', 'emit_verified_block', None),
    'decoder': ('test_refresh_window_layout', PREFIX + 'models.decoder', 'refresh_window_layout', None),
    'drafter': ('test_lac_offers_equal_golden', 'test', 'anchor_offers', None),
    'engine_replay': ('test_replay_matching_and_corrupt_capture', 'tests.jetspec_plus_plus.parity.replay_engine_rounds', 'replay', 'equal'),
    'hooks': ('test_emit_verified_block_matches_golden', 'jetspec.core.llm', 'emit_verified_block', None),
    'init': ('test_kth_init', 'test', 'build_random_context_block_token_ids', None),
    'method': ('test_retained_decoder_leaves_equal_golden', PREFIX + 'method.decoder.core', 'AnchorDecoder.forward', None),
    'mixed_precision': ('test_decoder_committed_prefix_cast', PREFIX + 'models.decoder', 'AnchorDecoder._prefix', None),
    'paper_eval': ('test_paper_joint_pool_keeps_all_32_golden_offers', 'test', 'anchor_offers', None),
    'pass': ('test_j_pass', 'test', 'reference_pass', None),
    'prefill': ('test_prefill_taps', 'jetspec.core.model_runner', 'ModelRunner.forward', None),
    'read_comparisons': ('test_token_round_and_tau_tools_match_golden', 'tests.jetspec_plus_plus.parity.compare_reads', 'compare_reads', 'bootstrap'),
    'reader': ('test_reader_records_match_golden', PREFIX + 'reader', 'prompt_record', None),
    'real_artifacts': ('test_stock_mark_preserves_context_bits', PREFIX + 'models.drafter', 'StockAdaptedMark.forward', None),
    'refresh': ('test_refresh', 'test', 'refresh_cutoff', None),
    'round': ('test_pass_labels_and_tau_equal_golden', 'test', 'completed_label', None),
    'round_capture': ('test_reader_captures_backend_schema_without_changing_tokens', 'test', 'main', 'summary.rounds'),
    'rows': ('test_blend_mass_matches_golden', PREFIX + 'method.kmass', 'blend_mass', None),
    'schedule': ('test_fixed_schedule', None, None, None),
    'swap': ('test_swap_pins', 'test', 'swap_decision', None),
}


def perturb(value):
    if isinstance(value, torch.Tensor):
        changed = value.detach().clone()
        with torch.no_grad():
            first = changed.reshape(-1)[0]
            if changed.dtype == torch.bool:
                first.logical_not_()
            elif changed.is_floating_point():
                first.fill_(1 if not torch.isfinite(first) else float(first) + 1)
            else:
                first.bitwise_xor_(1)
        return changed
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        return value + '_changed'
    if isinstance(value, tuple):
        items = [perturb(value[0]), *value[1:]]
        return type(value)(*items) if hasattr(value, '_fields') else tuple(items)
    if isinstance(value, list):
        return [perturb(value[0]), *value[1:]]
    if isinstance(value, dict):
        result = value.copy()
        key = next(iter(result))
        result[key] = perturb(result[key])
        return result
    if dataclasses.is_dataclass(value):
        name = dataclasses.fields(value)[0].name
        return dataclasses.replace(value, **{name: perturb(getattr(value, name))})
    if hasattr(value, 'ids'):
        value.ids.reshape(-1)[0].bitwise_xor_(1)
        return value
    if hasattr(value, 'kv'):
        key = next(iter(value.kv))
        value.kv[key] = perturb(value.kv[key])
        return value
    raise TypeError(f'cannot perturb {type(value)}')


class Mutations:
    def __init__(self, selected):
        self.selected = selected
        self.seen = set()
        self.exercised = set()
        self.results = {}

    def pytest_collection_modifyitems(self, items):
        keep, drop = [], []
        for item in items:
            file = Path(item.path).stem.removeprefix('test_')
            case = CASES.get(file)
            if file in self.selected and file not in self.seen and item.originalname == case[0]:
                self.seen.add(file)
                keep.append(item)
            else:
                drop.append(item)
        items[:] = keep
        if keep:
            keep[0].config.hook.pytest_deselected(items=drop)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        file = Path(item.path).stem.removeprefix('test_')
        _, module, attribute, field = CASES[file]
        with pytest.MonkeyPatch.context() as patch:
            if file == 'schedule':
                owner = importlib.import_module(PREFIX + 'round').JetSpecPPConfig
                original = owner.operations.fget
                def operations(config):
                    self.exercised.add(file)
                    return perturb(original(config))
                patch.setattr(owner, 'operations', property(operations))
            else:
                owner = item.module if module == 'test' else importlib.import_module(module)
                parts = attribute.split('.')
                for part in parts[:-1]:
                    owner = getattr(owner, part)
                name = parts[-1]
                original = getattr(owner, name)
                def changed(*args, **kwargs):
                    result = original(*args, **kwargs)
                    self.exercised.add(file)
                    if field:
                        result = result.copy()
                        target = result
                        parts = field.split('.')
                        for part in parts[:-1]:
                            target[part] = target[part].copy()
                            target = target[part]
                        target[parts[-1]] = perturb(target[parts[-1]])
                        return result
                    return perturb(result)
                patch.setattr(owner, name, changed)
            yield

    def pytest_runtest_logreport(self, report):
        if report.when == 'call':
            file = Path(report.nodeid.split('::')[0]).stem.removeprefix('test_')
            text = report.longreprtext
            self.results[file] = {
                'nodeid': report.nodeid,
                'caught': report.failed and 'AssertionError' in text and file in self.exercised,
                'failure': next((line for line in text.splitlines() if line.startswith('E ')), text[-500:]),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', choices=sorted(CASES))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    selected = {args.file} if args.file else set(CASES)
    plugin = Mutations(selected)
    pytest.main([str(BASE), '-q'], plugins=[plugin])
    args.report.write_text(json.dumps(plugin.results, indent=2, sort_keys=True) + '\n')
    success = plugin.results.keys() == selected and all(row['caught'] for row in plugin.results.values())
    print(f'Mutations caught: {sum(row["caught"] for row in plugin.results.values())}/{len(selected)}')
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())

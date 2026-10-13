"""CPU artifact preflight or bounded four-prompt hybrid T0 smoke through the reader.

All outputs go to --output-dir, outside the synchronized source tree.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('target', 'drafter', 'adapter', 'heads-bundle', 'prompts-file', 'paper-read'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    from safetensors import safe_open
    from transformers import Qwen3Config
    from jetspec.methods.jetspec_plus_plus.models.hybrid import HybridTarget, text_config
    from jetspec.methods.jetspec_plus_plus.models.adapter import CtxKVC1Head
    from jetspec.methods.jetspec_plus_plus.models.drafter import load_qwen3_config_with_rope_compat
    from jetspec.methods.jetspec_plus_plus.artifacts import load_heads_bundle
    if args.preflight:
        config = text_config(json.loads((args.target / 'config.json').read_text()))
        drafter = json.loads((args.drafter / 'config.json').read_text())
        adapter = json.loads((args.adapter / 'config.json').read_text())
        taps = drafter['dflash_config']['target_layer_ids']
        assert taps == adapter['dflash_config']['target_layer_ids']
        assert drafter['num_target_layers'] == config.num_hidden_layers
        assert taps[-1] == config.num_hidden_layers - 3
        bundle = load_heads_bundle(args.heads_bundle)
        assert {'config.json', 'model.safetensors.index.json'} <= set(bundle.decoder_manifest['target_files'])
        index = json.loads((args.target / 'model.safetensors.index.json').read_text())['weight_map']
        with torch.device('meta'):
            target = HybridTarget(config)
            ac = Qwen3Config.from_dict(adapter)
            ac.rope_theta = adapter.get('rope_theta', adapter.get('rope_parameters', {}).get('rope_theta'))
            head = CtxKVC1Head(ac)
        expected = {name.replace('model.', 'model.language_model.', 1): tuple(t.shape)
                    for name, t in target.state_dict().items()}
        text = {name for name in index if name.startswith(('model.language_model.', 'lm_head.'))}
        assert set(expected) == text, (sorted(set(expected)-text), sorted(text-set(expected)))
        with safe_open(str(args.adapter / 'head.safetensors'), framework='pt') as handle:
            assert set(handle.keys()) == set(head.state_dict())
            assert all(tuple(handle.get_slice(name).get_shape()) == tuple(t.shape)
                       for name, t in head.state_dict().items())
        with safe_open(str(args.adapter / 'converter.safetensors'), framework='pt') as handle:
            assert set(handle.keys()) == {'fc.weight', 'hidden_norm.weight'}
        dc = load_qwen3_config_with_rope_compat(str(args.drafter), local_files_only=True)
        assert dc.rope_theta == config.rope_parameters['rope_theta']
        prompts = json.loads(args.prompts_file.read_text())
        prompts = prompts.get('prompts') if isinstance(prompts, dict) else prompts
        paper = json.loads(args.paper_read.read_text())
        assert len(prompts) >= 4 and len(paper['outputs']) >= 4
        inputs = [row if isinstance(row, str) else row.get('token_ids', row.get('prompt'))
                  for row in prompts[:len(paper['outputs'])]]
        assert all(value is not None for value in inputs)
        inputs_sha256 = hashlib.sha256(json.dumps(inputs, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        from tests.jetspec_plus_plus.parity.read_data import read_identity
        paper_inputs = read_identity(paper).get('prompt_inputs_sha256')
        assert paper_inputs is None or inputs_sha256 == paper_inputs, 'paper prompt bank differs'
        print(json.dumps(dict(preflight='PASS', target_layers=config.num_hidden_layers,
            taps=taps, target_tensors=len(expected), adapter_tensors=len(head.state_dict()),
            paper_outputs=len(paper['outputs']), prompt_inputs_sha256=inputs_sha256,
            inference_mode='reader/core decorators'), sort_keys=True))
        return 0
    from jetspec.methods.jetspec_plus_plus.reader import main as read
    from tests.jetspec_plus_plus.parity.read_data import compare_tokens, paper_credit
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = ['--target', str(args.target), '--model-family', '27b',
        '--prompts-file', str(args.prompts_file), '--prompt-count', '4',
        '--max-new-tokens', '256', '--dtype', 'bf16', '--device', 'cuda',
        '--schedule', 'J1-R1', '--seed', '9504']
    candidate = read(common + ['--drafter', str(args.drafter), '--adapter', str(args.adapter),
        '--heads-bundle', str(args.heads_bundle), '--dataset', 'gsm8k',
        '--output', str(args.output_dir / 'method.json')])
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    reference = read(common + ['--ar', '--output', str(args.output_dir / 'ar.json')])
    equality = compare_tokens(candidate, reference)
    paper = json.loads(args.paper_read.read_text())
    credit = paper_credit(paper)
    rows = []
    for index, row in enumerate(candidate['outputs']):
        tokens, rounds = sum(row['accept_lengths']), len(row['accept_lengths'])
        rows.append(dict(prompt_index=index, tokens=row['n_tokens'], method_tau=tokens / rounds,
            method_rounds=rounds, paper_tau=credit['prompt_tokens'][index] / credit['prompt_rounds'][index],
            paper_tokens=credit['prompt_tokens'][index] + 1,
            equal_ar=equality['prompts'][index]['equal']))
    report = dict(equal_ar=equality['equal'], prompts=rows, tokens=equality,
        method_read=str(args.output_dir / 'method.json'), ar_read=str(args.output_dir / 'ar.json'),
        paper_read=str(args.paper_read), comparison='smoke cap 256; paper cap 2048')
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Hybrid smoke', '', f'Exact AR tokens: {equality["equal"]}', '',
        '| Prompt | Tokens | Method tau (256 cap) | Paper tau (2048 cap) | Equal AR |',
        '|---|---|---|---|---|']
    lines += [f'| {r["prompt_index"]} | {r["tokens"]} | {r["method_tau"]:.6f} | {r["paper_tau"]:.6f} | {r["equal_ar"]} |' for r in rows]
    (args.output_dir / 'validation.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(report, sort_keys=True))
    return 0 if equality['equal'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

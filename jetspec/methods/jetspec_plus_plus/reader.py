"""Ported from the JetSpec++ reference implementation.

Public API: read_passes(llm, rows, *, drafter_factory, config=None, tokens=2048,
target_layer_ids=None) -> list[dict]; token_sha256(token_ids) -> str;
prompt_record(index, token_ids, accept_lengths, passes_run, converged_rounds,
save_output_ids=False) -> dict; summarize(prompts) -> dict;
reduce_read(raw, method='jetspecpp', loop_cut=True, *, eos_token_ids=EOS,
max_new_tokens=2048) -> dict; main(argv=None, *, drafter_factory=None,
llm_factory=None) -> dict.

read_passes calls factory(llm=llm, config=config) once, then uses the returned
tree drafter. main's factory additionally receives drafter_checkpoint,
adapter_artifact and heads_bundle paths. Explicit --draft-dtype also passes
draft_dtype (a torch dtype) to that factory. The CLI accepts that callable via
--drafter-factory module:callable, keeping model construction with the caller.
The default factory builds the public local-artifact drafter; --dataset selects
its calibration entry. --ar uses LLM.generate without loading draft artifacts.
Prompt JSON is a list (or a dict's 'prompts' list) of strings or dictionaries
with 'token_ids' or 'prompt', and optional 'index'. Text is tokenized as given.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import platform

__all__ = ['token_sha256', 'prompt_record', 'summarize', 'reduce_read', 'read_passes', 'read_ar', 'main']

TAU_POOLING = 'sum of accept_lengths over every round of every prompt / number of rounds'
EOS = (151643, 151645)


def token_sha256(token_ids: list[int]) -> str:
    # Ported from the JetSpec++ reference implementation.
    return hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def prompt_record(index: int, token_ids: list[int], accept_lengths: list[int], passes_run: list[int],
                  converged_rounds: int, save_output_ids: bool = False) -> dict:
    record = {"prompt_index": index, "accept_lengths": accept_lengths,
              "n_rounds": len(accept_lengths), "n_tokens": len(token_ids),
              "token_sha256": token_sha256(token_ids), "passes_run": passes_run,
              "converged_rounds": converged_rounds}
    if save_output_ids:
        record["output_ids"] = token_ids
    return record


def summarize(prompts: list[dict]) -> dict:
    """Pool emitted tokens over verify rounds, excluding the prefill token."""
    prompts = [row for row in prompts if not row.get("refused")]
    rounds = [length for row in prompts for length in row["accept_lengths"]]
    means = [sum(row["accept_lengths"]) / len(row["accept_lengths"]) for row in prompts if row["accept_lengths"]]
    return {
        "tau": sum(rounds) / len(rounds) if rounds else None,
        "tau_pooling": TAU_POOLING,
        "tau_per_prompt_mean": sum(means) / len(means) if means else None,
        "rounds": len(rounds),
        "accepted_sum": sum(rounds),
        "tokens": sum(row["n_tokens"] for row in prompts),
        "n_prompts": len(prompts),
    }


def reduce_read(raw, method='jetspecpp', loop_cut=True, *, eos_token_ids=EOS,
                max_new_tokens=2048):
    """Paper rule-A credit and loop cuts; no serving-clock attribution.

    Ported from the JetSpec++ reference implementation's paper-metric segment reducer.
    The last 64-token chunk must repeat at least three times, without overlap,
    in a cap-reaching output with no EOS. Stop at its first occurrence and
    credit the crossing round only for its tokens before that onset.
    """
    if method != 'jetspecpp':
        raise ValueError('this reader reduces JetSpec++ emitted round lengths')
    if max_new_tokens < 1:
        raise ValueError('max_new_tokens must be positive')
    outputs = raw['outputs']
    cuts = {}
    for i, row in enumerate(outputs):
        ids = row['token_ids'][:max_new_tokens]
        if loop_cut and len(ids) == max_new_tokens and not any(t in eos_token_ids for t in ids):
            last = ids[-64:]
            occurrences, j = [], 0
            while j + 64 <= len(ids):
                if ids[j:j+64] == last:
                    occurrences.append(j)
                    j += 64
                else:
                    j += 1
            if len(occurrences) >= 3:
                cuts[i] = occurrences[0]
    # HF reads already have emitted lengths; segment's timing arrays are absent.
    tokens, rounds = [], []
    for i, row in enumerate(outputs):
        ids = row['token_ids'][:max_new_tokens]
        first = next((j for j, token in enumerate(ids) if token in eos_token_ids), None)
        stop = len(ids) if first is None else first + 1
        stop = min(stop, cuts.get(i, stop))
        left = max(0, stop - 1)
        credit, count = 0, 0
        for length in row['accept_lengths']:
            if type(length) is not int or length < 1:
                raise ValueError('accept_lengths must contain positive integers')
            if left > 0:
                emitted = min(length, left)
                credit += emitted
                count += 1
                left -= emitted
        if left:
            raise ValueError('accept_lengths do not cover the generated output')
        tokens.append(credit)
        rounds.append(count)
    total, count = sum(tokens), sum(rounds)
    return dict(tokens=total, rounds=count, tau=total/count if count else 0,
                cut_prompts=sorted(cuts), cut_onsets=cuts,
                prompt_tokens=tokens, prompt_rounds=rounds)


def read_passes(llm, rows, *, drafter_factory, config=None, tokens=2048,
                target_layer_ids=None, capture_path=None):
    """Read a fixed J/R schedule through width-one LLM tree verification."""
    import torch
    from jetspec.core.llm import SamplingParams
    from .round import JetSpecPPConfig

    if type(tokens) is not int or tokens < 1:
        raise ValueError('tokens must be a positive integer')
    config = config or JetSpecPPConfig.paper_eval()
    drafter = drafter_factory(llm=llm, config=config)
    if capture_path is not None:
        if not hasattr(drafter, 'candidate_records'):
            raise ValueError('round capture requires a JetSpec++ drafter with candidate records')
        drafter.capture_rounds = True
        capture_path = Path(capture_path)
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        capture_path.write_text('', encoding='utf-8')
    if target_layer_ids is None:
        target_layer_ids = getattr(drafter, 'target_layer_ids', None)
    found = []
    for index, row in enumerate(rows):
        if isinstance(row, str):
            prompt, prompt_index = row, index
        else:
            prompt_index = int(row.get('index', index))
            if 'token_ids' in row:
                ids = row['token_ids']
                if not ids or any(type(token) is not int or token < 0 for token in ids):
                    raise ValueError('prompt token_ids must be a nonempty list of nonnegative integers')
                prompt = torch.tensor([ids], dtype=torch.long, device=llm.device)
            else:
                prompt = row['prompt']
        output = llm.generate_tree(
            prompt, drafter, block_size=config.block_size, tree_width=1,
            budget=config.block_size, algo='accum_logp', algo_kwargs={},
            target_layer_ids=target_layer_ids, sampling_params=SamplingParams(0.0, tokens),
            return_stats=True, tree_attn='sdpa', return_text=False,
            prompt_index=prompt_index, seed=config.seed)
        accept_lengths = [int(value) for value in output['accept_lengths']]
        token_ids = [int(value) for value in output['token_ids']]
        if sum(accept_lengths) != max(0, len(token_ids) - 1):
            raise AssertionError('emitted round credit does not equal generated tokens minus prefill')
        results = getattr(drafter, 'round_results', ())
        passes_run = [int(result.passes_run) for result in results]
        record = prompt_record(prompt_index, token_ids, accept_lengths, passes_run,
                               sum(bool(result.converged) for result in results))
        record['token_ids'] = token_ids
        found.append(record)
        if capture_path is not None:
            from .capture import verified_records
            with capture_path.open('a', encoding='utf-8') as sink:
                for candidate in verified_records(drafter, accept_lengths, prompt_index=prompt_index):
                    sink.write(json.dumps(candidate, allow_nan=False) + '\n')
    return found


def read_ar(llm, rows, *, tokens):
    """Plain target greedy generation, with one emitted token per decode round."""
    import torch
    from jetspec.core.llm import SamplingParams

    found = []
    for index, row in enumerate(rows):
        prompt_index = index if isinstance(row, str) else int(row.get('index', index))
        if isinstance(row, str):
            prompt = row
        elif 'token_ids' in row:
            ids = row['token_ids']
            if not ids or any(type(token) is not int or token < 0 for token in ids):
                raise ValueError('prompt token_ids must be a nonempty list of nonnegative integers')
            prompt = torch.tensor([ids], dtype=torch.long, device=llm.device)
        else:
            prompt = row['prompt']
        ids = [int(token) for token in llm.generate(prompt, SamplingParams(0.0, tokens))['token_ids']]
        record = prompt_record(prompt_index, ids, [1] * max(0, len(ids) - 1), [], 0)
        record['token_ids'] = ids
        found.append(record)
    return found


def main(argv=None, *, drafter_factory=None, llm_factory=None):
    """Write an uncut tau read, its rule-A reduction and a reproducibility manifest."""
    parser = argparse.ArgumentParser(description='JetSpec++ torch tau reader (greedy, B16).')
    parser.add_argument('--target', required=True)
    parser.add_argument('--drafter')
    parser.add_argument('--adapter')
    parser.add_argument('--heads-bundle')
    parser.add_argument('--prompts-file', '--prompts-json', dest='prompts_file', required=True)
    parser.add_argument('--schedule', default='J1-R1', help='fixed J/R schedule, e.g. J1-R1 or J3-R3')
    parser.add_argument('--output', required=True)
    parser.add_argument('--drafter-factory', help='module:callable accepting llm, config and artifact paths')
    parser.add_argument('--dataset', help='calibration entry in the heads bundle')
    parser.add_argument('--ar', action='store_true', help='plain target greedy AR on the same prompts')
    parser.add_argument('--capture-rounds', action='store_true',
                        help='write reference-schema publications and redrafts to <output>.candidates.jsonl')
    parser.add_argument('--prompt-start', type=int, default=0)
    parser.add_argument('--prompt-count', type=int)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=9504)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--dtype', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--draft-dtype', choices=['same', 'bf16', 'fp32'], default='same',
                        help='drafter, adapter and decoder precision (same follows --dtype)')
    args = parser.parse_args(argv)
    if args.ar and args.capture_rounds:
        parser.error('--capture-rounds requires JetSpec++ drafting')
    if not args.ar:
        if not all((args.drafter, args.adapter, args.heads_bundle)):
            parser.error('--drafter, --adapter and --heads-bundle are required unless --ar')
        if drafter_factory is None:
            if args.drafter_factory:
                if ':' not in args.drafter_factory:
                    parser.error('--drafter-factory must be module:callable')
                module, name = args.drafter_factory.rsplit(':', 1)
                drafter_factory = getattr(importlib.import_module(module), name)
            else:
                from functools import partial
                from .drafter import build_drafter
                drafter_factory = partial(build_drafter, dataset=args.dataset)

    import torch
    import transformers
    from jetspec.core.llm import LLM
    from .models.drafter import REVISION
    from .round import JetSpecPPConfig, completed_label

    config = JetSpecPPConfig.paper_eval(schedule=args.schedule, seed=args.seed)
    if args.max_new_tokens < 1:
        parser.error('--max-new-tokens must be positive')
    prompts_path = Path(args.prompts_file)
    prompts_bytes = prompts_path.read_bytes()
    rows = json.loads(prompts_bytes)
    if isinstance(rows, dict):
        rows = rows['prompts']
    if not isinstance(rows, list):
        raise ValueError('prompts file must contain a list or a prompts list')
    if args.prompt_start < 0 or args.prompt_start >= len(rows):
        parser.error('--prompt-start must select an existing prompt')
    if args.prompt_count is not None and args.prompt_count < 1:
        parser.error('--prompt-count must be positive')
    end = len(rows) if args.prompt_count is None else args.prompt_start + args.prompt_count
    if end > len(rows):
        parser.error('--prompt-count extends past the prompt bank')
    # Preserve the bank index for the reference's seed XOR prompt_index convention.
    rows = [dict(prompt=row, index=i) if isinstance(row, str) else dict(row, index=row.get('index', i))
            for i, row in enumerate(rows) if args.prompt_start <= i < end]
    # Model/load-time cast, as in the reference reader.
    # The round's fp32 probability math remains in its existing tensor steps.
    dtype = torch.float32 if args.dtype == 'fp32' else torch.bfloat16
    llm = (llm_factory or LLM)(model_name_or_path=args.target, device=args.device,
                               dtype=dtype, attn_implementation='sdpa')

    def factory(*, llm, config):
        # Keep existing injectable factories compatible on the default path.
        options = {} if args.draft_dtype == 'same' else {
            'draft_dtype': torch.float32 if args.draft_dtype == 'fp32' else torch.bfloat16}
        return drafter_factory(llm=llm, config=config, drafter_checkpoint=args.drafter,
                               adapter_artifact=args.adapter, heads_bundle=args.heads_bundle, **options)

    outputs = (read_ar(llm, rows, tokens=args.max_new_tokens) if args.ar else
               read_passes(llm, rows, drafter_factory=factory, config=config,
                           tokens=args.max_new_tokens,
                           capture_path=str(args.output) + '.candidates.jsonl' if args.capture_rounds else None))
    round_records = []
    for i, row in enumerate(outputs):
        for r, accepted in enumerate(row['accept_lengths']):
            record = dict(prompt_index=i, candidate_round=r, accepted_len=accepted)
            if r < len(row['passes_run']):
                record['pass_label'] = completed_label(row['passes_run'][r])
            round_records.append(record)
    manifest = dict(schema='jetspec.jetspec_plus_plus.manifest/1', target=args.target,
                    drafter=args.drafter, drafter_revision=REVISION, adapter=args.adapter,
                    heads_bundle=args.heads_bundle, prompts_file=str(prompts_path),
                    prompts_sha256=hashlib.sha256(prompts_bytes).hexdigest(),
                    schedule=args.schedule, config=asdict(config), seed=config.seed,
                    per_prompt_seed='seed_xor_prompt_index', temperature=0,
                    max_new_tokens=args.max_new_tokens, block_size=config.block_size,
                    dtype=args.dtype, draft_dtype=args.dtype if args.draft_dtype == 'same' else args.draft_dtype,
                    device=args.device, tree_width=1, tree_attn='sdpa',
                    method='ar' if args.ar else 'jetspecpp', dataset=args.dataset,
                    prompt_start=args.prompt_start, prompt_count=len(rows),
                    # The reference benchmark hashes selected inputs,
                    # not the prompt-file bytes recorded by prompts_sha256 above.
                    prompt_inputs_sha256=hashlib.sha256(json.dumps(
                        [row['token_ids'] if 'token_ids' in row else row['prompt'] for row in rows],
                        sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest(),
                    selected_prompts_sha256=hashlib.sha256(
                        json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                    accepted_len='emitted acc+1, truncated at EOS and cap; excludes prefill',
                    eos_token_ids=sorted(llm.eos_token_ids), terminal_round='truncated_at_emit',
                    token_sha256_rule="sha256 of json.dumps(token ids, separators=(',', ':')) encoded as ascii",
                    python=platform.python_version(), torch=torch.__version__,
                    transformers=transformers.__version__)
    read = dict(schema='jetspec.jetspec_plus_plus.read/1', outputs=outputs,
                summary=summarize(outputs), manifest=manifest,
                metrics={'jetspec_round_records': round_records},
                rule_a=reduce_read({'outputs': outputs}, eos_token_ids=llm.eos_token_ids,
                                   max_new_tokens=args.max_new_tokens))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # JSON object keys are strings, including the reducer's positional cut map.
    read = json.loads(json.dumps(read))
    output.write_text(json.dumps(read, indent=2) + '\n', encoding='utf-8')
    Path(str(output) + '.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return read


if __name__ == '__main__':
    main()

"""Find the bundled prompts of a benchmark for one model."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Evaluation exports: first-turn Qwen3-8B renders (no thinking) of each benchmark, temperature 0.
BENCHMARKS = {'gsm8k': 'gsm8k-t0', 'math500': 'math500-t0', 'aime25': 'aime25-t0',
              'humaneval': 'humaneval-t0', 'livecodebench': 'lcb-t0', 'mbpp': 'mbpp-t0',
              'mt-bench': 'mt-bench-t0'}


def rerender(source, target, destination):
    """Render each first user turn of an 8B export with the target's own chat template."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(target)
    data = json.loads(Path(source).read_text())
    rows = []
    for row in data['prompts']:
        user = row['text'].split('<|im_start|>user\n', 1)[1].split('<|im_end|>', 1)[0]
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        rows.append({'index': row['index'], 'text': text,
                     'token_ids': tokenizer(text, add_special_tokens=False).input_ids})
    destination.write_text(json.dumps(dict(data, prompts=rows)))
    return destination


def bundled_prompts(model, dataset, target, directory):
    """The bundled prompts of a dataset, rendered for the model's chat template."""
    size = '27b' if model == 'qwen3.8-27b' else '8b'
    table3 = HERE / 'paper/inputs' / size / f'prompts_{dataset}128.json'
    if table3.is_file():
        return str(table3)
    if dataset not in BENCHMARKS:
        raise SystemExit(f'no bundled prompts for --dataset {dataset} (bundled: {", ".join(BENCHMARKS)}); '
                         'pass --prompts-json')
    path = HERE / 'prompts' / f'{BENCHMARKS[dataset]}.prompts.json'
    if size == '8b':
        return str(path)
    if not target:
        raise SystemExit(f'--dataset {dataset} for {model} renders its prompts with the target tokenizer: pass --target')
    return str(rerender(path, target, Path(directory) / f'prompts_{dataset}.json'))

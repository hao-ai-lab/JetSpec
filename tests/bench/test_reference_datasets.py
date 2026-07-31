import sys
from types import SimpleNamespace

from bench.reference import benchmark


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return " | ".join(
            f"{message['role']}:{message['content']}" for message in messages
        )


def test_dataset_registry_is_dflash_superset():
    assert set(benchmark.PROMPT_FMT) == {
        "gsm8k",
        "math500",
        "aime25",
        "humaneval",
        "mbpp",
        "mt-bench",
    }


def test_mbpp_uses_sanitized_test_prompt(monkeypatch):
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return [{"prompt": "Write a function that returns zero."}]

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset)
    )
    prompts = benchmark.build_prompts(_Tokenizer(), "mbpp", 0)

    assert calls == [
        (("google-research-datasets/mbpp", "sanitized"), {"split": "test"})
    ]
    assert prompts == ["user:Write a function that returns zero."]


def test_mt_bench_preserves_turns_and_uses_generated_history(monkeypatch):
    def load_dataset(*args, **kwargs):
        return [{"prompt": ["First question", "Follow-up question"]}]

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset)
    )
    tokenizer = _Tokenizer()
    samples = benchmark.build_prompts(tokenizer, "mt-bench", 0)
    assert samples == [("First question", "Follow-up question")]

    calls = []

    def generate(prompt):
        calls.append(prompt)
        turn = len(calls)
        return {
            "token_ids": [turn],
            "text": f"answer-{turn}",
            "decode_time": float(turn),
            "accept_lengths": [turn + 1],
            "tree_sizes": [turn + 3],
            "rounds": turn,
        }

    output = benchmark._run_sample(samples[0], tokenizer, generate)

    assert calls == [
        "user:First question",
        "user:First question | assistant:answer-1 | user:Follow-up question",
    ]
    assert output["token_ids"] == [1, 2]
    assert output["decode_time"] == 3.0
    assert output["accept_lengths"] == [2, 3]
    assert output["tree_sizes"] == [4, 5]
    assert output["rounds"] == 3
    assert output["tpf"] == 2 / 3

"""Minimal demo: offline greedy generation.

    python examples/basic/simple_generate.py [model_name_or_path]
"""
import sys

from jetflow import LLM, SamplingParams


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
    llm = LLM(model)
    out = llm.generate(
        "The three primary colors are",
        SamplingParams(temperature=0.0, max_new_tokens=64),
    )
    print(out["text"])


if __name__ == "__main__":
    main()

"""JetFlow N3 end-to-end gate: the paged tree-attention triton kernel, wired into
`JetFlowEngine` behind `attn_backend="triton_paged_tree"`, must produce the SAME tokens
as the default SDPA path for N0 (`generate`), N1 (`generate_tree`), and N2a
(`generate_batch`).

SDPA is the correctness oracle: the kernel reads the exact post-RoPE K/V bytes SDPA
reads, so on a tiny fp32 Qwen3 on CUDA the two argmax streams are identical. We build
the same tiny model the CPU gates use, but on CUDA in fp32, and assert token-for-token
equality across a couple of seeds + a ragged multi-prompt batch.

Needs CUDA (triton); skipped on a CPU-only host. N2b (`generate_tree_batch`) stays on
SDPA in N3, so it is not exercised here."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)

from transformers import Qwen3Config, Qwen3ForCausalLM

from jetflow.core.llm import SamplingParams
from jetflow.core.model_runner import ModelRunner
from jetflow.draft import RandomTreeDrafter, TargetEchoTreeDrafter
from jetflow.inference_engine.engine import JetFlowEngine

DEVICE = "cuda"


class _StubTokenizer:
    """Only `.decode` is exercised when prompts are passed as input_ids tensors."""

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(t)) for t in ids)


def _tiny_model(seed: int = 0) -> Qwen3ForCausalLM:
    """A tiny fp32 Qwen3 on CUDA (head_dim=16; GQA 4/2 heads; no network)."""
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=256, tie_word_embeddings=False,
    )
    return Qwen3ForCausalLM(cfg).eval().to(torch.float32).to(DEVICE)


def _tiny_jetflow(model, attn_backend: str, block_size: int = 16) -> JetFlowEngine:
    """Wire `model` into a `JetFlowEngine` with the chosen backend (no network).

    For the kernel backend we register the interface + flip `_attn_implementation`
    here (the `__init__` does this, but the tests bypass `__init__` via
    `object.__new__` to skip `load_target`)."""
    eng = object.__new__(JetFlowEngine)
    eng.model = model
    eng.tokenizer = _StubTokenizer()
    eng.runner = ModelRunner(model)
    eng.device = DEVICE
    eng.dtype = torch.float32
    eng.block_size = block_size
    eng.eos_token_ids = set()            # no EOS -> deterministic length
    eng.attn_backend = attn_backend
    if attn_backend == "triton_paged_tree":
        from jetflow.inference_engine.paged_attn_backend import register_jetflow_paged_tree

        register_jetflow_paged_tree()
        model.config._attn_implementation = "jetflow_paged_tree"
    else:
        model.config._attn_implementation = "sdpa"
    return eng


# CPU-side prompts (moved to CUDA inside each test, so module import is safe on a
# CPU-only host where collection still runs before the skipif fires).
PROMPT = torch.tensor([[3, 14, 15, 92, 65, 35, 89, 7]])
PROMPTS = [
    torch.tensor([[3, 14, 15, 92, 65, 35, 89, 7]]),
    torch.tensor([[10, 20, 30, 40, 50]]),
    torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]),
    torch.tensor([[64, 32, 16]]),
]
SP = SamplingParams(0.0, 24)


def _both(make_engine, run):
    """Run `run(engine)` under SDPA then the kernel on the SAME weights; the kernel
    backend flips `_attn_implementation`, so build the engines per call to keep the
    SDPA reference unpolluted."""
    model = make_engine
    sdpa = run(_tiny_jetflow(model, "sdpa"))
    kern = run(_tiny_jetflow(model, "triton_paged_tree"))
    return sdpa, kern


# --- N0: generate (vanilla AR) ----------------------------------------------

@pytest.mark.parametrize("seed", [0, 1])
def test_n0_generate_kernel_matches_sdpa(seed):
    model = _tiny_model(seed)
    sdpa, kern = _both(model, lambda e: e.generate(PROMPT, SP)["token_ids"])
    assert kern == sdpa, f"N0 kernel diverged from SDPA (seed={seed})"
    assert len(sdpa) == SP.max_new_tokens


@pytest.mark.parametrize("block_size", [16, 4, 5])
def test_n0_generate_block_sizes(block_size):
    """Block sizes that don't divide head_dim exercise cross-boundary slot math."""
    model = _tiny_model(0)
    sdpa = _tiny_jetflow(model, "sdpa", block_size).generate(PROMPT, SP)["token_ids"]
    kern = _tiny_jetflow(model, "triton_paged_tree", block_size).generate(PROMPT, SP)["token_ids"]
    assert kern == sdpa, f"N0 kernel diverged from SDPA (block_size={block_size})"


# --- N1: generate_tree (single-stream tree spec) ----------------------------

@pytest.mark.parametrize("seed", [1, 7])
def test_n1_generate_tree_kernel_matches_sdpa_random(seed):
    model = _tiny_model(0)
    drafter = RandomTreeDrafter(vocab_size=model.config.vocab_size)

    def run(e):
        torch.manual_seed(seed)            # identical trees across both runs
        return e.generate_tree(PROMPT, drafter, block_size=4, tree_width=2,
                               budget=15, sampling_params=SP)["token_ids"]

    sdpa, kern = _both(model, run)
    assert kern == sdpa, f"N1 kernel diverged from SDPA (random drafter, seed={seed})"


def test_n1_generate_tree_kernel_matches_sdpa_echo():
    """TargetEchoTreeDrafter -> full-depth accepts (the multi-node-accept path)."""
    model = _tiny_model(0)
    drafter = TargetEchoTreeDrafter(model)

    def run(e):
        torch.manual_seed(1)
        return e.generate_tree(PROMPT, drafter, block_size=4, tree_width=2,
                               budget=15, sampling_params=SP)["token_ids"]

    sdpa, kern = _both(model, run)
    assert kern == sdpa, "N1 kernel diverged from SDPA (echo drafter)"


# --- N2a: generate_batch (continuous-batched AR) ----------------------------

@pytest.mark.parametrize("seed", [0, 1])
def test_n2a_generate_batch_kernel_matches_sdpa(seed):
    """Ragged multi-prompt batch (lengths 8/5/12/3): the kernel must match SDPA on
    every sequence."""
    model = _tiny_model(seed)
    sdpa = _tiny_jetflow(model, "sdpa").generate_batch(PROMPTS, SP)
    kern = _tiny_jetflow(model, "triton_paged_tree").generate_batch(PROMPTS, SP)
    for i in range(len(PROMPTS)):
        assert kern[i]["token_ids"] == sdpa[i]["token_ids"], (
            f"N2a kernel diverged from SDPA on seq {i} (seed={seed})"
        )


@pytest.mark.parametrize("block_size", [16, 4, 5])
def test_n2a_generate_batch_block_sizes(block_size):
    model = _tiny_model(0)
    sdpa = _tiny_jetflow(model, "sdpa", block_size).generate_batch(PROMPTS, SP)
    kern = _tiny_jetflow(model, "triton_paged_tree", block_size).generate_batch(PROMPTS, SP)
    for i in range(len(PROMPTS)):
        assert kern[i]["token_ids"] == sdpa[i]["token_ids"], (
            f"N2a kernel diverged from SDPA on seq {i} (block_size={block_size})"
        )

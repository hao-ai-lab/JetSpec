# Tree-algorithm roadmap

The shipped set is the curated, offline-validated minimum. The planned set is
the meaningful-but-not-yet-built algorithms, each reserved as a documented
placeholder so we don't lose track. (Triage from the design-space analysis over
the full 39-variant space; only the ones with a genuine niche are listed.)

## Shipped (registered, HF-runnable, lossless-validated)

| name | family | niche |
|---|---|---|
| `crossproduct` | baselines | full-fanout baseline |
| `top2gap_fanout` | tree_to_chain | low-budget winner (top-2 gap → fanout cap) |
| `task_router` | semantic_aware | per-prompt fanout template (prompt-adaptive) |
| `reasoning_router` | semantic_aware | rule-based reasoning→chain (zero-ML) |
| `class_histogram` | semantic_aware | per-class template; real histogram is a profiler upgrade |

## Planned (placeholder files; not registered)

| name | family | needs | notes |
|---|---|---|---|
| `depth_rank_histogram` | profile_guided | offline profiler (HF-collectable) | the only variant whose finer-than-per-depth granularity could beat `top2gap_fanout`; shares its profiler with `class_histogram`'s real version |
| `path_conditional_refresh` | layer_conditional | vLLM draft-side tree-attention hook | path-conditional draft logits feeding the gap gate; deployment-grade (~D forwards, not N) |
| `online_warmup` | profile_guided | serving-loop warmup-feedback hook | learns per-depth acceptance online; niche = long single-prompt generations |
| `template_bandit` | profile_guided | serving-loop reward-feedback hook | ε-greedy over shape templates; zero-calibration, auto-picks chain at low budget |

**Path to implement:** `depth_rank_histogram` can be done on the HF engine once
a profiler script lands (it also lights up `class_histogram`'s real histogram).
`path_conditional_refresh` / `online_warmup` / `template_bandit` need
serving-engine hooks → vLLM phase. Each placeholder's module docstring carries
its mechanism, potential, and exact dependency.

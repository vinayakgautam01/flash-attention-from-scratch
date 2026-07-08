# FlashAttention from Scratch

> Implemented a FlashAttention-style forward kernel in CUDA using tiled Q/K/V streaming and online softmax, avoiding materialization of the full attention matrix, and benchmarked against a naive CUDA baseline and PyTorch SDPA with Nsight Compute analysis.

<!-- hero-result: filled in M10 -->
**Hero result:** _pending_ — will be filled from `benchmarks/results/all.csv` in M10.

## Status

Work in progress. Roadmap: [`docs/MILESTONES.md`](docs/MILESTONES.md).

## Variants

| Variant | Description | What it demonstrates | Status |
|---|---|---|---|
| CPU reference | Direct attention in C++/NumPy | Correctness oracle | pending (M1) |
| Naive CUDA | Materializes `S` and `P` | Baseline memory cost | pending (M2) |
| Online softmax | Streaming softmax reference | Numerical stability | pending (M3) |
| Flash v1 | Tiled K/V, no `N×N` write | IO-aware attention | pending (M4) |
| Flash v2 | Better shared-mem layout, vectorized loads | Coalescing / occupancy | pending (M6) |
| Flash causal + batch | Causal mask, batching, multi-head, boundary handling | LLM-shaped inputs | pending (M7) |
| Flash fp16 | fp16 input, fp32 accumulation | Mixed precision | pending (M8) |

## Correctness strategy

_Coming in M1 — CPU / NumPy oracle in double precision, cross-checked against PyTorch SDPA. Every kernel is tested against the same oracle at pinned tolerances (see [`docs/AGENTS.md`](docs/AGENTS.md) §9)._

## Benchmarks

_Coming in M5 — runtime, HBM bytes, speedup vs naive, error vs reference, effective TFLOP/s. See [`benchmarks/results/`](benchmarks/results/)._

## Nsight Compute analysis

_Coming in M9. See [`docs/nsight_comparison.md`](docs/nsight_comparison.md), [`docs/memory_analysis.md`](docs/memory_analysis.md), and [`docs/plots/roofline.png`](docs/plots/roofline.png)._

## Writeups

_Coming in M9._

- Long-form: [`WRITEUP.md`](WRITEUP.md).
- Focused: `docs/memory_analysis.md`, `docs/online_softmax_derivation.md`, `docs/nsight_comparison.md`, `docs/backward_pass_conceptual.md`.

## Reading the gap to `flash-attn`

_Coming in M10 — analytical comparison, not a race. For each shape where this kernel is slower than `flash-attn`, we name the specific optimization whose absence explains the gap._

## Reproduce

```bash
scripts/reproduce.sh
```

Full setup instructions in [`docs/AGENTS.md`](docs/AGENTS.md).

## Future work

- Backward pass (recomputation-based).
- Tensor-core / WMMA path.
- FlashAttention-2 work partitioning.
- FlashAttention-3 Hopper features (TMA, WGMMA, FP8).
- Dropout, variable-length packed sequences, paged attention.

## References

- Original planning: [`docs/flash-attention-project.md`](docs/flash-attention-project.md).
- Milestones: [`docs/MILESTONES.md`](docs/MILESTONES.md).
- Source-of-truth conventions: [`docs/AGENTS.md`](docs/AGENTS.md).

# FlashAttention from Scratch — Long-form Writeup

> Single canonical writeup for the project. Focused writeups live under [`docs/`](docs/).
> This file is the one link you would share.

<!-- hero-result -->
**Hero result:** _pending_ — filled in M10 with a shape from `benchmarks/results/all.csv`.

Format when filled:

> On `<GPU model>` at `(B, H, N, D) = (…)`, fp32: `<X>×` faster and `<Y>×` less HBM traffic than the naive multi-kernel baseline. Within `<k>×` of `flash-attn` at the same shape; `<one-line explanation of the gap>`.

## 1. Why naive attention is memory-bound

_Coming in M2/M9. HBM traffic accounting for `S`/`P` materialization, and why the ridge on the roofline sits where it does. See also [`docs/memory_analysis.md`](docs/memory_analysis.md)._

## 2. Online softmax derivation (summary)

_Coming in M3. See [`docs/online_softmax_derivation.md`](docs/online_softmax_derivation.md) for the full derivation._

## 3. FlashAttention forward walkthrough

_Coming in M4. Tile shapes, shared-memory budget, CTA-to-query-block mapping, the recurrence in the inner loop._

## 4. v2 optimizations and roofline analysis

_Coming in M6/M9. Vectorized loads, bank-conflict avoidance, occupancy tuning; roofline plot placing each variant against the compute/memory ceilings. See [`docs/plots/roofline.png`](docs/plots/roofline.png)._

## 5. Reading the gap to `flash-attn`

_Coming in M10. Not a race. For each shape where this kernel is slower, name the specific optimization (tensor-core MMA, `cp.async` pipelining, split-K, softcap fusion, …) whose absence best explains the gap._

## 6. Conceptual backward pass (not implemented)

_Coming in M9. See [`docs/backward_pass_conceptual.md`](docs/backward_pass_conceptual.md). Recomputation trick, per-row stashed state, why it makes training feasible._

## 7. Future work

See the "Future work" section in [`README.md`](README.md).

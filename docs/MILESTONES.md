# FlashAttention from Scratch — Milestones

> **Vision.** Build an educational, honest, benchmark-driven CUDA implementation of exact
> scaled-dot-product attention that progresses from a naive multi-kernel baseline to a
> tiled, IO-aware, FlashAttention-style forward kernel. The repo should read as
> **evidence that you understand the algorithm, the memory hierarchy, and the trade-offs**,
> not as a race to beat `flash-attn`.

## One-liner (portfolio-ready target)

> Implemented a FlashAttention-style forward kernel in CUDA using tiled Q/K/V streaming
> and online softmax, avoiding materialization of the full attention matrix, and
> benchmarked against a naive CUDA baseline and PyTorch SDPA with Nsight Compute analysis.

---



## Success criteria

**Must-have (portfolio-blocking).**

- [ ] Forward-only exact attention on `float32`.
- [ ] Naive multi-kernel CUDA baseline that materializes `S` and `P`.
- [ ] Standalone online-softmax reference (CPU **and** CUDA), independent of tiling.
- [ ] FlashAttention-style tiled forward kernel (v1) — no `NxN` write to HBM.
- [ ] Causal masking variant.
- [ ] Correctness harness comparing against a CPU oracle **and** PyTorch.
- [ ] Benchmark sweep with CSV output + at least one plot per axis (N, D).
- [ ] Nsight Compute report for naive vs Flash on one canonical shape.
- [ ] **Roofline plot** placing naive and Flash variants against the compute/memory ceilings.
- [ ] Four short writeups: memory-bound analysis, online softmax derivation, Nsight comparison, and **conceptual backward-pass note** (no code).
- [ ] `WRITEUP.md` at repo root — single canonical long-form writeup with the **hero result** on top.
- [ ] `notebooks/demo.ipynb` — 30-second reproducible demo that prints the hero speedup + max abs error.
- [ ] `README.md` with variant table, correctness strategy, benchmark table, hero plot, Nsight screenshot, and links to `WRITEUP.md` + `docs/`.

**Nice-to-have (portfolio-distinguishing).**

- [ ] `fp16` input with `fp32` accumulation.
- [ ] PyTorch extension (loadable from Python).
- [ ] Comparison against `torch.nn.functional.scaled_dot_product_attention` and `flash-attn`.
- [ ] Head-dim–specialized paths (32 / 64 / 128).

**Explicitly out of scope (avoid rabbit-holes).**

- Backward pass.
- Dropout.
- Variable-length packed sequences.
- Paged attention.
- FlashAttention-2 full work partitioning.
- FlashAttention-3 Hopper features (TMA, WGMMA, FP8) — mention as future work only.

---



## Cross-cutting standards (project-wide, applies to every milestone)

- **Commits.** Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `perf:`, `docs:`, `bench:`, `chore:`). **No** `Co-authored-by:` **trailers and no tool/agent attribution in commit messages** — authorship stays clean and human.
- **Source of truth.** `docs/AGENTS.md` is authoritative. If a decision isn't there, either add it or search the codebase before assuming.
- **Search.** Use `ast-grep` over `grep` for code searches.
- **Lint/Sonar.** Zero new issues; no `NOSONAR` without approval.
- **Diffs.** Minimal and reversible; one milestone → one PR ideally, or a small stack.
- **Reuse patterns.** Follow the same variant-table + benchmarks + Nsight structure as
your `gpu-parallel-patterns` repo. Don't invent new abstractions unless the domain forces it.
- **Definition of Done per milestone.** All TODOs checked, verification plan executed and
screenshotted/logged into `docs/plots/` or `benchmarks/results/`, PR merged.



## Milestone template (used below)

Each milestone has the same shape:

- **Scope** — one paragraph, why this milestone exists.
- **Learning objectives** — what you must be able to *explain* by the end, unprompted.
- **TODOs** — checkbox list, in dependency order.
- **Verification plan** — measurable acceptance criteria; if it can't be measured, it doesn't count.
- **Understanding checkpoint** — 2–3 questions you should be able to answer out loud
before ticking the milestone. These are what separate "I typed the kernel" from
"I understand the kernel."
- **Portfolio hook** — the one sentence a reviewer will take away from this milestone.

---



## M0 — Repo scaffolding & source-of-truth setup

**Effort:** S · **Depends on:** —

**Scope.** Turn an empty repo into a project a stranger can build and test in under 5 minutes. Establish `docs/AGENTS.md` as the canonical answer-file per your own workflow rule.

**Learning objectives.**

- Justify each top-level directory in one sentence.
- State the target CUDA arch(s), toolkit version, and Python/PyTorch versions in `docs/AGENTS.md`.

**TODOs.**

- [x] Decide target GPU arch(s) (e.g. `sm_80`, `sm_86`, `sm_89`) and record in `docs/AGENTS.md`.
- [x] Create the layout from the plan doc:
  ```text
  csrc/  tests/  benchmarks/  scripts/  docs/  docs/plots/  benchmarks/results/
  ```
- [x] `CMakeLists.txt` (or Makefile) that builds a hello-world `.cu` and links against Torch (guarded).
- [x] `scripts/build.sh`, `scripts/test.sh`, `scripts/bench.sh`, `scripts/profile_ncu.sh` (stubs OK).
- [x] `.gitignore`, `.clang-format`, `.editorconfig`, `pyproject.toml` (or `requirements.txt`) pinning `torch`, `numpy`, `pytest`, `matplotlib`.
- [x] Minimal GitHub Actions workflow: build + lint on push (CPU-only runner is fine).
- [x] `docs/AGENTS.md` with: target arch, build instructions, test/bench commands, kernel-naming convention, tensor-layout convention (row-major, `[B, H, N, D]`), commit conventions, "how to add a new variant."
- [x] Empty `README.md` with variant-table skeleton (fill later).
- [x] Empty `WRITEUP.md` at repo root with the section skeleton pre-filled and a `<!-- hero-result -->` placeholder for the headline number (`{speedup}× faster, {ratio}× less HBM traffic vs naive at (B, H, N, D) = (…)`, filled in M10).
- [x] `scripts/reproduce.sh` — a single script a stranger with a GPU can run to build, test, and produce the hero plot. If GPU CI is available (self-hosted runner), wire it up; if not, `docs/AGENTS.md` documents the exact `reproduce.sh` invocation as the substitute.

**Status.** Complete on `259712c` (2026-07-20). Colab T4 build + smoke test green; CI green on `main`; hero-result placeholder still open (fills in M10).

**Verification plan.**

- Fresh clone → `scripts/build.sh` succeeds on your box in one shot.
- `docs/AGENTS.md` answers: "What arch? What layout? What's the naming rule for a new kernel?"
- `WRITEUP.md` exists with the hero-result placeholder visible.
- CI is green (CPU lint/build). Either a GPU CI job runs on push, or `scripts/reproduce.sh` is documented and runs end-to-end on your box.

**Understanding checkpoint.**

1. Why `[B, H, N, D]` and not `[B, N, H, D]` for your target kernel? (Coalescing implications.)
2. What does your `docs/AGENTS.md` say when a future contributor asks "how do I add a Flash v3 variant?"
3. If a stranger clones the repo tomorrow, which single command produces the hero result?

**Portfolio hook.** *"Clean, documented, reproducible from day zero."*

---



## M1 — CPU reference + PyTorch oracle

**Effort:** S · **Depends on:** M0

**Scope.** Own the *definition* of correctness before touching CUDA. A tiny, painfully clear C++/Python reference that you'll compare every kernel against for the rest of the project.

**Learning objectives.**

- State exactly which numerical formula you're implementing (with the `1/√d` scale, softmax stability trick, optional causal mask).
- Explain why the CPU reference must not use vendor kernels (BLAS, cuDNN) — it's the oracle.

**TODOs.**

- [x] `csrc/attention_cpu_ref.hpp` — templated, header-only, `float`/`double`. Numerically stable softmax (subtract row max).
- [x] Python mirror in `tests/reference.py` using pure NumPy (double-precision).
- [x] `tests/test_reference_matches_torch.py` — assert CPU ref matches `torch.nn.functional.scaled_dot_product_attention` within a tight tolerance on small shapes (e.g. `N=16, D=8`).
- [x] Causal variant of the CPU reference (mask upper triangle before softmax).
- [x] Fixed-seed random tensor generator (`tests/util_tensors.py`) — reuse everywhere.
- [x] GoogleTest scaffolding for the C++header (++`tests/cpp/test_attention_cpu_ref.cpp`++) — direct C++-side unit tests (AGENTS.md §8 decision).

**Status.** Complete on `main` (2026-07-22). 19 pytest cases green (18 parametrized + 1 causal sanity), 7 GoogleTest cases green on Mac-local; C++ CI green pending Colab/Modal run.

**Verification plan.**

- `pytest tests/test_reference_matches_torch.py` green.
- Max abs error CPU-ref vs PyTorch < `1e-5` (fp32) on `N ∈ {8, 16, 32}`, `D ∈ {8, 16, 32}`, causal ∈ {False, True}.
- `ctest` in the build tree runs `test_attention_cpu_ref` and passes.

**Understanding checkpoint.**

1. What happens numerically if you skip the `-max` shift in softmax? Show an example that overflows.
2. Why compare to `torch.nn.functional.scaled_dot_product_attention` instead of writing `Q@K.T` in NumPy again? (Answer both: it's a stronger cross-check; and it's what real users use.)

**Portfolio hook.** *"Correctness is defined before performance is discussed."*

---



## M2 — Naive CUDA multi-kernel baseline

**Effort:** M · **Depends on:** M1

**Scope.** Deliberately naive: three separate kernels that materialize `S = QK^T`, then `P = softmax(S)`, then `O = PV`. This is the *straw man* whose memory cost you'll later demolish.

**Learning objectives.**

- Quantify the HBM bytes read/written by the naive path as a function of `N`, `D`.
- Explain why this is memory-bound past a modest `N`.

**TODOs.**

- [x] `theory/M2.md` — beginner-friendly, self-contained theory (mirrors `theory/M1.md` style; explains three-kernel decomposition, CUDA idioms, HBM byte accounting, roofline argument).
- [x] `csrc/attention_naive.cu` — three kernels: `qk_matmul`, `row_softmax`, `pv_matmul`. API in `csrc/attention_naive.cuh`.
- [x] Simple launch wrapper `attention_naive_forward(Q, K, V, O, B, H, N, D, is_causal)` + a host-pointer overload `attention_naive_forward_host` for tests.
- [x] Correctness test: `tests/cpp/test_attention_naive.cu` (GoogleTest; decision recorded — pure C++/CUDA, no pybind11 until M4) — asserts vs CPU ref, tolerance `5e-4` abs fp32 (per `docs/AGENTS.md` §9).
- [x] Log the peak transient memory (`S` and `P` allocations) for `N ∈ {128, 256, 512, 1024, 2048, 4096}` to `benchmarks/results/naive_memory.csv`, via `benchmarks/naive_memory.py`.
- [x] Plot HBM footprint vs `N` on log-log axes → `docs/plots/naive_memory.png` (via `benchmarks/plot_naive_memory.py`).
- [x] Add a "memory-bound" note in `docs/memory_analysis.md` (rough version, finalized in M9).

**Status.** Complete on `5c55da0` (2026-08-07). Colab T4 (nvcc 12.8, torch 2.11.0+cu128) `ctest` run passes 20/20 M2 cases (16 parametrized `M2Grid/*` + 4 `AttentionNaiveEdge/*`) with max abs err `≤ 5e-7` vs the CPU ref. `benchmarks/results/naive_memory.csv` + `docs/plots/naive_memory.png` remain the standalone artifact; the M5 harness sweep (`all.csv`) also carries naive rows at every `(N, D)` in the M5 grid for cross-comparison. `docs/AGENTS.md` §9 fp32 tolerance (5e-4) held.

**Verification plan.**

- Correctness passes for `B=1, H=1, N ∈ {64, 128, 512, 1024}`, `D ∈ {32, 64}`, causal ∈ {False, True}. (16 parametrized cases + 4 edge cases = 20 GoogleTests.)
- CSV shows quadratic memory growth in `N` — confirmed: `S` at `N=4096` is 64 MB; slope-2 fit on log-log plot.
- One plot: HBM footprint vs `N` on log-log axes → `docs/plots/naive_memory.png`. ✓

**Understanding checkpoint.**

1. For `N=4096, D=64, fp32`, how many bytes does `S` alone occupy? Compare to an A100's SM shared-memory budget.
2. Which of the three kernels dominates runtime for large `N`, and why? (Predict, then verify with Nsight in M9.)

**Portfolio hook.** *"I built the straw man honestly so the improvement is measurable, not narrated."*

---



## M3 — Online softmax reference (the intellectual core)

**Effort:** M · **Depends on:** M1

**Scope.** Before fusing anything, prove to yourself and the reader that you understand the streaming recurrence. This milestone is a **standalone artifact** — a kernel that computes softmax(row) block-by-block using running `(m, l)` state and rescaling. This is what every recruiter asks about.

**Learning objectives.**

- Derive the recurrence from first principles.
- Explain why rescaling `O_old` by `exp(m_old - m_new)` (rather than recomputing) preserves exact equality.

**TODOs.**

- [x] `theory/M3.md` — beginner-friendly theory doc with block-wise diagrams (mirrors `theory/M1.md`/`M2.md` style).
- [x] `docs/online_softmax_derivation.md` — 1–2 pages, hand-derived, showing:
  - The invariant `l_i = Σ_j exp(s_j - m_i)`.
  - How `(m, l)` update when a new block arrives.
  - How to combine two partial `O` contributions correctly.
- [x] `csrc/attention_online_ref.cu` — a kernel that computes `softmax(row) @ V_row` streaming over `K/V` in blocks (still per-row, no tiling of Q yet). Bc = 64; one CTA per query row; shared-memory-resident `(m, ℓ, Õ)` state.
- [x] `tests/cpp/test_attention_online_ref.cu` — GoogleTest (parametrized grid `N ∈ {128, 512, 2048}, D ∈ {32, 64}` + 5 edge cases) asserts bit-for-bit close (`< 1e-5` fp32) to CPU ref.
- [x] Tiny toy Python script `docs/toy_online_softmax.py` — runs the recurrence in NumPy for a `N=8` example, prints every intermediate + a broken-α variant that diverges by 21% — the figure that lands in the writeup.
- [ ] ~~`tests/test_online_softmax.py`~~ — deferred to M5's harness. Python parity waits for pybind11 in M4 (`docs/AGENTS.md` §8 ADR). M3 correctness is fully covered by the C++/GoogleTest above.

**Status.** Complete on `5c55da0` (2026-08-07). Colab T4 `ctest` run passes 11/11 M3 cases (6 parametrized `M3Grid/*` at `N ∈ {128, 512, 2048}, D ∈ {32, 64}` + 5 `AttentionOnlineRefEdge/*`) with max abs err `≤ 3e-7` vs the CPU ref — well inside the 1e-5 algebraic-equivalence bound recorded in `docs/AGENTS.md` §9. The M5 pytest matrix additionally diffs `attention_online_ref` against `torch_ref` (looser 5e-4 threshold, since matmul order differs); those 10 cells also pass. Runtime behaviour matches theory: (2 + 2N)·T HBM traffic dominates → 44 ms at `N=2048, D=64` (see `benchmarks/results/all.csv`), which is *why* M4 tiles.

**Verification plan.**

- Correctness matches CPU ref within `1e-5` fp32.
- The derivation doc is understandable to someone who has only seen the naive softmax formula.
- Toy script's printed table matches by hand.

**Understanding checkpoint.**

1. If you skip the `exp(m_old - m_new)` rescale, what specifically breaks — and can you show it numerically on `N=8`?
2. Why does the recurrence let you throw away every previous block after processing it? What state must you keep per query row?

**Portfolio hook.** *"I derived and implemented the online softmax as a standalone artifact — the tiled kernel is just an application of it."*

---



## M4 — FlashAttention v1 forward kernel

**Effort:** L · **Depends on:** M2, M3

**Scope.** First real Flash kernel. One CTA per block of query rows; loop over K/V tiles in shared memory; apply the online softmax; never write `S` or `P` to HBM.

**Learning objectives.**

- Explain the tile shapes `(Br, Bc)` and the shared-memory budget that constrains them.
- Explain the CTA-to-query-block mapping and why it enables parallelism across `B*H*ceil(N/Br)` CTAs.

**TODOs.**

- [x] `theory/M4.md` — beginner-friendly theory doc with tile diagrams, launch-config decoder, and step-by-step thread-mapping walk (mirrors `theory/M2.md`/`M3.md`).
- [x] `csrc/flash_fwd_v1.cu` — start with fixed `B=1, H=1`, `D ∈ {32, 64}`, non-causal. Kernel handles any `(B, H)` in signature; tests pin to `(1, 1)` per anti-scope.
- [x] Choose `Br`, `Bc` from shared-memory budget; document the calculation in a comment header. **Picked `Br = Bc = 32`** — fits 48 KB default T4 smem AND 1024 threads/CTA cap for `D ∈ {32, 64}`.
- [x] Correctness test vs CPU ref for `N ∈ {128, 256, 512, 1024, 2048}`, `D ∈ {32, 64}`, tolerance `< 5e-4` fp32. Includes `N = 257` case (partial Q-tile *and* KV-tile tails) even though MILESTONES doesn't require it — otherwise M7 inherits an untested tail path.
- [x] First-pass benchmark vs `attention_naive`: runtime + HBM-bytes-transferred (analytical via `benchmarks/perf_model.py`; Nsight deferred to M9), delivered by the M5 harness in `benchmarks/results/all.csv` (superset of the originally-scoped `v1_vs_naive.csv` — same numbers plus `torch_ref` and `attention_online_ref` for the same grid).
- [x] Note obvious inefficiencies you're leaving for v2 (bank conflicts? uncoalesced loads? low occupancy?) in `docs/flash_attention_notes.md`.

**Status.** Complete on `5c55da0` (2026-08-07). Colab T4 `ctest` run passes 16/16 M4 cases (10 parametrized `M4Grid/*` at `N ∈ {128, 256, 512, 1024, 2048}, D ∈ {32, 64}` + 6 `FlashFwdV1Edge/*` including the `N=257` mixed-tail case) with max abs err `≤ 6.6e-7` vs the CPU ref. First-pass benchmark row now lives in `benchmarks/results/all.csv`: Flash v1 beats `attention_naive` by 3.1× at `N=128, D=64` and stays ahead through `N=512, D=64` (1.5×), then loses to naive at `N ≥ 1024` for `D=64` — the exact memory-layout / occupancy story documented in `docs/flash_attention_notes.md` and reserved for M6. `S` and `P` never appear in `torch.cuda.memory_stats()` (peak_alloc = 0 delta for Flash v1 rows), as required.

**Verification plan.**

- Correctness matches CPU ref on the shape grid above.
- Kernel outperforms `attention_naive` on at least `N ≥ 512` (if not, diagnose before moving on).
- No `S` or `P` allocation shows up in `cuda-memcheck` / memory profiling.

**Understanding checkpoint.**

1. For your chosen `(Br, Bc, D)`, how many bytes of shared memory per CTA? How many CTAs per SM does that permit?
2. What's the arithmetic intensity of your inner loop, and how does it compare to the naive kernel's?

**Portfolio hook.** *"The N×N attention matrix never touches HBM — I can point to the exact lines."*

---



## M5 — Test & benchmark harness (formalize)

**Effort:** M · **Depends on:** M4

**Scope.** Stop ad-hoc-ing. Build the harness you'll use to compare every future variant against every past one, on the same shape grid, with the same tolerance, producing the same CSV schema.

**Learning objectives.**

- Justify each column in your benchmark CSV.
- Explain why warm-up + median-of-N is the right measurement policy for CUDA.

**TODOs.**

- [x] `benchmarks/bench_attention.py` — sweeps `B, H, N, D, causal, dtype, variant`, writes one row per `(shape, variant)` to `benchmarks/results/all.csv` with columns:
  ```text
  variant, B, H, N, D, causal, dtype, runtime_ms_median, runtime_ms_p95,
  hbm_bytes_est, peak_alloc_bytes, tflops_effective,
  max_abs_err_vs_ref, max_rel_err_vs_ref, git_sha, gpu_name
  ```
- [x] Uses CUDA events for timing, `torch.cuda.synchronize`, warm-up + N=20 timed iterations, records median and p95.
- [x] `tests/test_all_variants.py` — parametrizes over `(variant, shape)`, runs correctness for every registered variant.
- [x] Plot script `benchmarks/plot.py`: runtime vs `N` per variant, speedup vs naive vs `N`, error histogram. Outputs to `docs/plots/`.
- [x] Register `attention_naive` and `flash_fwd_v1` in the harness.

**Status.** Complete on `5c55da0` (2026-08-07). Colab T4 sweep produced `benchmarks/results/all.csv` (40 rows × 16 columns — 4 variants × 5 `N` × 2 `D` × causal=False) plus three hero plots in `docs/plots/`. All 40 M5 parametrized pytest cases + 2 sanity cases pass; max abs err `≤ 6.6e-7` vs `torch_ref` across every non-reference row, well under the 5e-4 fp32-CUDA parity threshold. Measurement policy locked at `warmup=10, timed=20` CUDA-event iterations, median + p95 reported. Adding a new variant is now a three-line change (bindings shim + `variants.py` registry entry + CMake link target) — the harness is the compounding artifact for M6..M13.

**Verification plan.**

- `scripts/bench.sh` produces a fresh `all.csv` and refreshes `docs/plots/*.png` in one command.
- `scripts/test.sh` runs the full parametrized correctness matrix in < 2 minutes.
- One PR review-quality plot already exists in `docs/plots/runtime_vs_N.png`.

**Understanding checkpoint.**

1. Why median-of-N and not mean? Why warm-up?
2. What's the failure mode if you forget `torch.cuda.synchronize` before stopping the timer?

**Portfolio hook.** *"Every claim in the README traces back to a row in* `all.csv`*."*

---



## M6 — FlashAttention v2: memory-layout & occupancy tuning

**Effort:** L · **Depends on:** M4, M5

**Scope.** Same algorithm as v1, better mechanical sympathy. Vectorized loads, shared-memory layouts that avoid bank conflicts, register-pressure control to lift occupancy. **This is not FlashAttention-2 the paper** — call it Flash v2 (yours) and be explicit in `docs/` about the distinction.

**Learning objectives.**

- Read a Nsight Compute report and identify the top-two bottlenecks.
- Explain how a shared-memory swizzle avoids bank conflicts for your access pattern.

**TODOs.**

- [ ] `csrc/flash_fwd_v2_shared_kv.cu`.
- [ ] Vectorized loads (`float4` or `__ldg` where appropriate) for Q/K/V.
- [ ] Shared-memory layout revision (padded / swizzled) — document the conflict math in comments.
- [ ] Review register usage (`--ptxas-options=-v`), tune `Br/Bc` for occupancy.
- [ ] **Artifact:** `docs/ptxas_v1_vs_v2.md` — checked-in `ptxas -v` output for both kernels side-by-side (registers/thread, shared-mem/CTA, stack, spill loads/stores), with a two-paragraph interpretation.
- [ ] Correctness passes on the same grid as v1.
- [ ] Runs faster than v1 on at least `N ≥ 512, D=64`; log the improvement in `docs/flash_attention_notes.md`.
- [ ] Naming/registration in the harness so v1 and v2 appear side-by-side in plots.

**Verification plan.**

- Correctness parity with v1.
- Nsight Compute shows: fewer bank conflicts, higher achieved occupancy, or higher DRAM throughput than v1 — pick which and prove it.
- `docs/ptxas_v1_vs_v2.md` exists and its numbers are consistent with your occupancy claim (fewer registers → higher theoretical occupancy, no spills, etc.).
- New plot row in `docs/plots/runtime_vs_N.png` showing v2 above v1.

**Understanding checkpoint.**

1. Which of the three (bank conflicts / occupancy / vectorized bandwidth) actually moved the needle for you? Show the Nsight metric that changed.
2. What did you *not* fix that you'd fix in v3?

**Portfolio hook.** *"I profiled v1, formed a hypothesis, changed one thing, and can prove the metric moved."*

---



## M7 — Practical features: batch, multi-head, causal, boundaries

**Effort:** L · **Depends on:** M6

**Scope.** Make it usable for a shape that resembles a real transformer layer. Add batch and head dimensions to the CTA grid, causal masking inside the tile loop, and correct handling of `N` and `D` that aren't clean multiples of the tile sizes.

**Learning objectives.**

- Explain the CTA-grid mapping to `(batch, head, query_block)`.
- Explain how causal masking is applied *inside* the online-softmax step, not as a post-hoc filter.

**TODOs.**

- [ ] `csrc/flash_fwd_v3_causal.cu` (or feature-flagged v2).
- [ ] Support `[B, H, N, D]` with `B ∈ {1,2,4}`, `H ∈ {1,4,8,16}`.
- [ ] Causal mask applied per-tile with correct handling of the diagonal tile.
- [ ] Boundary handling: `N` not a multiple of `Br`, and last `K/V` tile smaller than `Bc`.
- [ ] `D ∈ {32, 64, 128}` supported (may be templated).
- [ ] Correctness on the full grid: `B ∈ {1,2,4}, H ∈ {1,4,8}, N ∈ {129, 256, 513, 1024, 2049, 4096}, D ∈ {32, 64, 128}, causal ∈ {False, True}`.
- [ ] Benchmarks re-run; harness now covers all these axes.

**Verification plan.**

- All parametrized tests pass.
- No silent skips for odd `N` values (`129`, `513`, `2049` prove boundary correctness).
- Causal + non-causal runtimes on a plot; explain any gap.

**Understanding checkpoint.**

1. In the diagonal tile of a causal kernel, which entries are masked and how do you avoid `exp(-inf)` polluting `l`?
2. Why does batching across `(B, H)` in the CTA grid parallelize almost for free, and where does that break down?

**Portfolio hook.** *"Handles the shapes real transformers use, including the awkward ones."*

---



## M8 — Mixed precision: fp16 input, fp32 accumulation

**Effort:** M · **Depends on:** M7

**Scope.** Match the actual precision regime used in production. Inputs and stored outputs `half`; all intermediate math (dot products, softmax stats, `O` accumulator) in `float`.

**Learning objectives.**

- Explain why *accumulation* precision matters more than *storage* precision for softmax numerical stability.
- State the tolerance you expect vs the fp32 reference and justify it.

**TODOs.**

- [ ] Template the v2/v3 kernel on input dtype, force fp32 accumulators (`float` scalars, not `__half`).
- [ ] Convert Q/K/V to `__half`, keep the CPU/torch reference in fp32.
- [ ] Correctness vs fp32 reference: tolerance `< 5e-3` abs / `< 1e-2` rel (justify in a comment).
- [ ] Benchmark fp16 vs fp32 on the full grid, add a plot.
- [ ] Note: if `mma`/tensor cores come in scope, guard them behind a compile flag; do not sink the milestone chasing WMMA.

**Verification plan.**

- All fp16 tests pass at the stated tolerance.
- Speedup over fp32 documented on the plot.
- `docs/AGENTS.md` updated with the tolerance rule.

**Understanding checkpoint.**

1. Give a concrete example where fp16 accumulation of the softmax denominator `l` would visibly drift.
2. What's the smallest change you'd need to plug in `mma.sync` here? (Just describe — don't implement.)

**Portfolio hook.** *"Mixed precision done with the right accumulation, tested against the right oracle."*

---



## M9 — Nsight Compute deep dive, roofline, and the four writeups

**Effort:** M · **Depends on:** M6 (min), better after M8

**Scope.** The "did they actually pay attention" milestone. Produce reports and prose, not just numbers. Everything gets checked into `docs/`. This is also where you visualize the memory-boundedness thesis with a roofline plot and prove — in prose — that you understand the *full* algorithm even though you only implemented the forward pass.

**Learning objectives.**

- Read a full Nsight Compute report unaided and pick the two metrics that matter for your kernel.
- Place a kernel on a roofline plot from its measured throughput and arithmetic intensity, and read off whether it's compute- or memory-bound.
- Explain, on a whiteboard, how the FlashAttention backward pass uses recomputation to avoid materializing `S`/`P` in HBM.
- Write a paragraph a hiring GPU engineer would nod at.

**TODOs.**

- [ ] `scripts/profile_ncu.sh` — profiles `attention_naive`, `flash_fwd_v1`, `flash_fwd_v2` on one canonical shape (e.g. `B=2, H=8, N=2048, D=64, causal=false, fp32`). Saves `.ncu-rep` files into `benchmarks/results/`.
- [ ] Screenshots of the section pages that matter (Memory Chart, Occupancy, Warp State) into `docs/plots/nsight/`.
- [ ] **Roofline plot** → `docs/plots/roofline.png`. Compute ceiling and memory ceiling for your target GPU; plot naive, v1, v2 as points at their measured (GFLOP/s, arithmetic intensity). The plot must visually show naive left of the ridge (memory-bound) and Flash pushing right along the memory ceiling. Script under `benchmarks/roofline.py`.
- [ ] `docs/memory_analysis.md` — finalize: HBM bytes analytical estimate, compare to measured DRAM throughput × runtime.
- [ ] `docs/online_softmax_derivation.md` — finalize with the M3 toy example.
- [ ] `docs/nsight_comparison.md` — naive vs Flash v1 vs Flash v2, one metric per paragraph, one screenshot per paragraph.
- [ ] `docs/backward_pass_conceptual.md` — 1–2 pages, no code. Cover: (a) what the backward needs (`dQ, dK, dV` from `dO`), (b) the recomputation trick (only stashed `(m, l)` per row, recompute `S`/`P` block-by-block on the backward pass), (c) why this makes training feasible without HBM blowup, (d) what a real implementation would need that this repo does not have. State explicitly at the top: *"conceptual only — not implemented in this repo."*
- [ ] Cross-link all four writeups + the roofline from `README.md` and `WRITEUP.md`.

**Verification plan.**

- Four writeups exist, each < 3 pages, each with at least one figure.
- Roofline plot exists and its axes are labeled with the actual GPU's peak FLOP/s and HBM bandwidth (sourced from datasheet, cited in the plot caption).
- Nsight reports reproducible via `scripts/profile_ncu.sh` on a fresh clone.
- Any performance claim in the writeups traces to `benchmarks/results/all.csv` or an `.ncu-rep`.
- The backward-pass note is understandable to someone who has read the online-softmax derivation but not the FlashAttention paper.

**Understanding checkpoint.**

1. On your canonical shape, what fraction of naive's runtime is spent moving `S` and `P` through HBM? Show your arithmetic.
2. Which single Nsight metric would you point at to convince a skeptic that Flash v2 > Flash v1?
3. On the roofline plot, why does naive not sit *on* the memory ceiling? (Answer: it's not saturating bandwidth — it's bottlenecked on something worse, likely latency-bound global loads. Prove it from the ncu report.)
4. If your kernel had a backward pass, what extra state per query row would you save from the forward pass and why is that enough?

**Portfolio hook.** *"There's a paragraph, a plot, and a profiler screenshot for every claim — and I can explain the backward pass without having written it."*

---



## M10 — PyTorch extension, analytical SDPA/`flash-attn` comparison, portfolio polish

**Effort:** M · **Depends on:** M7 (min), better after M9

**Scope.** Make it callable from Python, benchmark against production baselines as an **analytical exercise** (not a competition — this repo has no tensor-core path, so gaps are expected and *diagnostic*), fill in the hero result, and finish the README + `WRITEUP.md` so the repo sells itself in the first screen.

**Framing (important).** Comparing to `flash-attn` and `torch_sdpa` here is *how you learn what optimizations the remaining gap represents*. Frame the section as "reading the gap" — a retrospective analysis. Do not phrase anything as beating or losing.

**Learning objectives.**

- Explain the difference between your kernel and each PyTorch SDPA backend (math / mem-efficient / Flash).
- Given a runtime gap to `flash-attn` on a specific shape, name the top-2 optimizations (e.g. tensor-core MMA, async copy pipelining, split-K) most likely responsible.

**TODOs.**

- [ ] `torch.utils.cpp_extension`- or `pybind11`-based Python module exposing `flash_forward(Q, K, V, causal)`.
- [ ] `tests/test_against_torch.py` covering all shapes from M7 + fp16 from M8.
- [ ] `benchmarks/bench_attention.py` extended with `torch_sdpa` and (optionally, guarded) `flash_attn` variants.
- [ ] Comparison plot: your v2 vs `torch_sdpa` vs `flash_attn` on the full grid.
- [ ] `notebooks/demo.ipynb` — 30-second demo. Loads the extension, runs naive + Flash v2 on one canonical shape, prints hero speedup + max abs error vs CPU ref, renders the hero plot inline. This is the file recruiters open first.
- [ ] **Fill in the hero result** in `WRITEUP.md` (placeholder created in M0). Pick one shape from `benchmarks/results/all.csv` and freeze the number: `{speedup}× faster and {ratio}× less HBM traffic than naive at (B, H, N, D) = (…), fp32`. Also record: within `{k}×` of `flash-attn` at the same shape, with an honest one-line explanation.
- [ ] `WRITEUP.md` — the single canonical long-form writeup. Sections: hero result → why naive is memory-bound → online softmax derivation (summarized, deep link to `docs/`) → the Flash kernel walkthrough → v2 optimizations with roofline → analytical comparison to `torch_sdpa` and `flash-attn` (the "reading the gap" section) → conceptual backward pass → future work. This is the *one file* you would share as a link.
- [ ] README section "Reading the gap to `flash-attn`" — for each shape where you're slower, name the specific optimization that would close it (tensor-core MMA, async `cp.async` pipelining, split-K, softcap fusion, etc.). This is the section that most demonstrates you paid attention.
- [ ] Final `README.md`:
  - Elevator paragraph (use the one-liner above).
  - Hero result banner (one line, one number).
  - Variant table (from the plan doc).
  - Correctness strategy summary.
  - Benchmark table + hero plot + roofline plot.
  - Nsight screenshot.
  - Links to `WRITEUP.md` and the four `docs/` writeups.
  - Link to `notebooks/demo.ipynb`.
  - "Reading the gap" analytical section.
  - "Future work" section listing the out-of-scope items — this is where FA-2/FA-3, tensor cores, and the backward pass get named.
- [ ] Git tag `v1.0.0`, GitHub release with the plots attached.
- [ ] Cross-link from your `gpu-parallel-patterns` repo README ("capstone application").

**Verification plan.**

- `pip install -e .` (or equivalent) works on a fresh clone; `python -c "import flash_from_scratch"` succeeds.
- `notebooks/demo.ipynb` runs top-to-bottom in under 30 seconds on your target GPU.
- Comparison plot exists and does not hide any axis where you're slower — the gap is *labeled with a hypothesis*, not omitted.
- `WRITEUP.md`'s hero-result placeholder is replaced with a real, sourced number.
- README's first screen tells the whole story to a recruiter in 30 seconds.
- `v1.0.0` tag pushed.

**Understanding checkpoint.**

1. Which torch SDPA backend actually ran during your comparison (`torch.backends.cuda.sdp_kernel(...)`) and how did you verify?
2. Pick your worst-case shape vs `flash-attn`. Name the specific optimization (one, not a list) whose absence explains most of the gap, and justify why it's that one and not something else.
3. If someone opens `notebooks/demo.ipynb` cold, what do they learn in 30 seconds?

**Portfolio hook.** *"Callable from PyTorch, honestly compared to production baselines, and every remaining gap has a named optimization behind it."*

---



## Anti-scope guardrails (re-read before starting each milestone)

- No backward pass. If you want it later, that's a **separate repo**.
- No dropout, no packed variable-length sequences, no paged attention.
- No Hopper-only features (TMA, WGMMA, FP8). Mention in "Future work," do not implement.
- **No tensor-core / WMMA path in v1.0.0.** This repo intentionally has no `mma`/`wmma.h` variant. Treat the gap to `flash-attn` as *the thing you're analyzing*, not the thing you're closing. Naming this explicitly is a feature, not a weakness — see M10's "Reading the gap" section.
- Do **not** try to outperform `flash-attn` on its home turf. The M10 comparison is analytical: you *read* the gap, you don't race it.
- Do not skip M3 (online softmax as a standalone artifact) — it is the intellectual heart of the project.
- Do not skip the writeups or the roofline plot. The code + prose + one hero number is what makes the repo showcase-worthy.



## Progress ledger


| #   | Milestone                                  | Effort | Status |
| --- | ------------------------------------------ | ------ | ------ |
| M0  | Repo scaffolding & source-of-truth setup   | S      | [x]    |
| M1  | CPU reference + PyTorch oracle             | S      | [x]    |
| M2  | Naive CUDA multi-kernel baseline           | M      | [x]    |
| M3  | Online softmax reference                   | M      | [x]    |
| M4  | FlashAttention v1 forward kernel           | L      | [x]    |
| M5  | Test & benchmark harness (formalize)       | M      | [x]    |
| M6  | FlashAttention v2 layout & occupancy       | L      | [ ]    |
| M7  | Batch, multi-head, causal, boundaries      | L      | [ ]    |
| M8  | Mixed precision (fp16 in / fp32 accum)     | M      | [ ]    |
| M9  | Nsight deep dive & three writeups          | M      | [ ]    |
| M10 | PyTorch extension, SDPA comparison, polish | M      | [ ]    |


Update this table (and check off the TODOs inside each milestone) as you go. It's the repo's honest self-report — treat it like part of the deliverable.
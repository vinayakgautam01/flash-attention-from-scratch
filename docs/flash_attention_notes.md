# FlashAttention v1 notes — deliberate non-choices and the v2 backlog

> **What this file is:** the running list of "we could have done this in v1 but chose not to." Each item names the concrete optimization, the reason we deferred it, and the milestone where it lands.
>
> **Why it exists:** M4's success criterion (`docs/MILESTONES.md` §M4) explicitly asks us to *"note obvious inefficiencies you're leaving for v2."* This document is that note. It is also, deliberately, **M6's TODO list**.

Everything below is measured against `csrc/flash_fwd_v1.cu` at the M4 landing commit. See [`../theory/M4.md`](../theory/M4.md) for the theory these choices sit on top of.

---

## Non-choices by category

### 1. Vectorized loads (`float4`, `__ldg`) — M6

- **Where it hurts.** Steps 1 (K/V load), init (Q load), and the terminal write of `O` all do scalar `float` loads/stores. Each HBM transaction pulls at most 32 × 4 = 128 B per warp; a `float4` load would pull 512 B per warp for the same instruction count.
- **Why deferred.** Adds a `D % 4 == 0` alignment requirement to `D` and a 4-way inner unroll to the load; extra code complexity in v1 without any correctness value.
- **What M6 does.** Rewrite the load loops using `reinterpret_cast<const float4*>` on the HBM pointer, keep the smem side scalar (or switch smem to `float4` if bank-conflict analysis pans out). Guard behind a `D % 4 == 0` check; fall back to scalar path otherwise.

### 2. Shared-memory bank conflicts on `sK`/`sV`/`sO` — M6

- **Where it hurts.** Step 2 reads `sK[tx * D + d]` for varying `tx` inside a warp with `d` fixed at each unrolled step. At `D = 32`, `tx * D + d ≡ d (mod 32)` for every `tx` → **32-way bank conflict**, serializing the shared-mem read 32×. At `D = 64`, same story: `tx * 64 + d ≡ d (mod 32)`.
- **Same problem on `sV`** in step 7: `sV[c * D + d]` for varying `c`, `d` fixed → 32-way conflict.
- **Why deferred.** Fixing this requires either +1 padding on the leading dim (`sK[Bc, D+1]`) or a swizzled layout. Both change the addressing math throughout the kernel; v1 keeps the arithmetic obvious.
- **What M6 does.** Try +1 padding first — cheapest fix, adds `Bc + Br` floats to the smem budget (still fits). Measure with Nsight Compute (`smsp__inst_executed_shared_ld` bank-conflict-per-warp metric). If padding isn't enough, switch to a swizzled layout.

### 3. `Õ` in shared memory instead of registers — M6

- **Where it hurts.** `sO[Br, D] = Br * D floats` sits in shared memory across the entire block loop. At `Br = 32, D = 64` that's 8 KB — a full quarter of the smem budget. Every step-7 read/write also incurs latency instead of hitting a register.
- **Why deferred.** Register-resident `Õ` requires each thread to own a specific slice of the `Br × D` output (e.g. thread `(tx, ty)` holds `Õ[ty, tx], Õ[ty, tx+Bc], ...` in a `float` array of length `D / Bc`). Adds a fixed-size register array indexed by an outer loop; v1 keeps the mapping obvious.
- **What M6 does.** Move `Õ` to a per-thread `float O_reg[(D + Bc - 1) / Bc]` array. Freed smem (`Br * D * 4 = 8 KB`) buys either larger tiles (`Br = Bc = 64` at `D = 32` becomes reachable), 2 CTAs/SM occupancy, or both.

### 4. Register pressure and occupancy tuning — M6

- **Where it hurts.** v1 launches 1024 threads/CTA with ~37 KB smem/CTA on Colab T4. T4 caps at 1024 threads/SM and 64 KB smem/SM → **exactly one CTA per SM at v1's launch config**. The SM has no other CTA's warps to hide latency behind.
- **Diagnosis.** Read `ptxas -v` output (already enabled in `CMakeLists.txt`) to see registers/thread; a spill would push us further.
- **What M6 does.** Two levers: (a) lower thread count to 512 by using `block(Bc, Br/2)` and giving each thread two Q rows, or (b) shrink smem (per §3 above) to fit 2 CTAs/SM. Which one moves Nsight's `sm__warps_active` metric more is the empirical question M6 answers.

### 5. Async copies (`cp.async`, Ampere+) — M6/M7

- **Where it hurts.** Step 1's K/V load and step 2's compute run serially inside a tile iteration. On Ampere+, `cp.async` would overlap the next tile's K/V load with the current tile's compute (double-buffered smem tiles). Not available on Colab T4 (Turing) but real on Modal A10G (Ampere).
- **Why deferred.** Requires guarding on compute capability, double-buffering the smem layout, and pipelining the block loop.
- **What M6/M7 does.** After the smem-layout work in §2 lands, add a `sm_86`-gated `cp.async` path; keep the scalar path for T4.

### 6. FA-2 loop reversal (outer = KV-tile) — future work, not M6

- **Where it hurts.** M4 uses FA-1 loop order (outer = Q-tile per CTA). At large `D` or small `Br`, `K/V` gets re-read `Tr = ⌈N/Br⌉` times, and the HBM ratio vs naive drops toward 1× (see `theory/M4.md` §10). FA-2 flips: outer = KV-tile, K/V loaded once, running state migrates between CTAs via the `(m, ℓ, Õ)` monoid combine (see `theory/M3.md` §8).
- **Why deferred.** This is FA-2, not "v2 as I mean it here." It requires cross-CTA scheduling and a partial-result merge step across CTAs. Cleaner to name it in the writeups (M9/M10) as *the specific optimization whose absence explains the gap to `flash-attn`*.
- **What later does.** Not implemented in v1.0.0 per `docs/AGENTS.md` §10 anti-scope. Named in M10's "reading the gap" analytical section.

### 7. `D = 128` support — M7

- **Where it hurts.** At `D = 128, Br = Bc = 32`: smem = `4 * (2*32*128 + 2*32*128 + 32*32 + 64) = 4 * (8192 + 8192 + 1024 + 64) = 69 888 B` — over the 48 KB default cap AND over the 64 KB max even with `cudaFuncSetAttribute` opt-in.
- **Why deferred.** Solving it needs the smem freed by register-resident `Õ` (§3) *and* possibly a `(Br, Bc)` re-pick.
- **What M7 does.** After §3 lands in M6, `D = 128` at `Br = 32, Bc = 32` fits (`smem ≈ 60 928 B` with `Õ` in registers → opt in to 64 KB via `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536)`).

### 8. Batch and multi-head fan-out in the CTA grid — M7

- **Status in v1.** The kernel already handles arbitrary `(B, H)` correctly (grid decodes `blockIdx.y` as a linearized `(b, h)`, kernel body uses `b = bh / H`, `h = bh % H`). Only the *tests* are pinned to `B = H = 1`.
- **What M7 does.** Extend `tests/cpp/test_flash_fwd_v1.cu` (or a v3-specific test file) to cover `B ∈ {1, 2, 4}` and `H ∈ {1, 4, 8, 16}`. No kernel change expected.

### 9. Causal masking correctness testing — M7

- **Status in v1.** Kernel plumbs `is_causal` and applies the mask via a single `-inf` line in step 2 (`j_score > i` → `s_val = -inf`). Not tested in M4.
- **What M7 does.** Extend the test grid to `is_causal ∈ {false, true}`. Also exercises the diagonal-tile edge case where the mask crosses inside a single `Br × Bc` tile.

### 10. Tensor cores / WMMA — explicit anti-scope

- **Status.** Not implemented in v1.0.0 per `docs/AGENTS.md` §10 and `docs/MILESTONES.md` anti-scope.
- **Named in.** M10's "reading the gap" section — this is the single largest chunk of the gap to `flash-attn`.

### 11. Hidden `Br == Bc` coupling in the K/V load — M6

- **Where it hurts.** STEP 1's K/V load maps `ty → KV-row-within-tile`, so it fills exactly `Br` rows of `sK`/`sV`. Since `sK`/`sV` are shaped `[Bc × D]`, this only covers the tile fully when `Br == Bc`. `Br < Bc` leaves rows `[Br, Bc)` unwritten (STEP 2 reads garbage). `Br > Bc` writes past `sK` into `sV` (silent corruption). Currently guarded by a `static_assert(Br == Bc)` in the kernel; the coupling itself is a code-structure limitation, not a correctness bug.
- **Why deferred.** M4's `Br = Bc = 32` is pinned by three unrelated constraints (`Bc == warpSize`, `Br * Bc ≤ 1024`, and the smem budget), so `Br == Bc` falls out for free at v1's tile size. Generalizing costs code complexity we don't need to pay yet.
- **What M6 does.** When §3 (register-resident `Õ`) or §4 (occupancy tuning via `Br = Bc/2`) actually wants asymmetric tiles, replace the two-nested-loop load with a linearized load:
  ```cpp
  int linear_tid  = ty * Bc + tx;
  int num_threads = Br * Bc;
  for (int idx = linear_tid; idx < Bc * D; idx += num_threads) {
      int row = idx / D;
      int col = idx % D;
      // sK[row * D + col] = K[..., b_idx*Bc + row, col]  (with tail-guard)
  }
  ```
  Preserves coalescing when `D` is a multiple of 32; trades one static-assert for a bit of runtime address arithmetic. Remove the `static_assert(Br == Bc)` at the same time.

---

## Under-utilization at small `N`

Named separately because it's not fixable by any of the items above — it's a *launch-config* problem.

- **Where it hurts.** At `N = 512, Br = 32`: `Tr = 16` CTAs. T4 has 40 SMs. **24 SMs idle for the whole kernel.**
- **What v2 doesn't fix.** Even with 2 CTAs/SM from §3, 32 SMs would run — still 8 idle. To saturate at small `N` you need cross-`(B, H)` grid dims (§8) or FA-2-style sequence-parallel work partitioning (§6).
- **Consequence for M4 benchmarks.** M4 is *expected* to lose to naive at very small `N` because the fixed launch overhead + underutilization outweigh the fused-kernel win. Verification only asks for "wins on `N ≥ 512`", and we should expect it to break even (not win) at `N = 128, D = 64`.

---

## Analytical HBM footprint (for M9)

From `theory/M4.md` §10, single head:

$$
\text{HBM}_{\text{M4}} = 2 N D + \frac{2 N^2 D}{B_r} \text{ floats}
$$

At `Br = 32, D = 64, N ∈ {128, 256, 512, 1024, 2048}`:

| `N` | M4 HBM (floats) | Naive HBM (floats) | Ratio (naive / M4) |
|---|---|---|---|
| 128 | 65 536 | 65 536 | 1.0× |
| 256 | 262 144 | 262 144 + 65 536 = 327 680 | 1.25× |
| 512 | 1 048 576 | 1 048 576 + 131 072 = 1 179 648 | 1.13× |
| 1024 | 4 194 304 | 4 194 304 + 262 144 = 4 456 448 | 1.06× |
| 2048 | 16 777 216 | 16 777 216 + 524 288 = 17 301 504 | 1.03× |

At `D = 64, Br = 32`, **M4's asymptotic HBM ratio vs naive is 1× (parity)** — but naive can't run at these `N` at all because of the `O(N²)` transient allocation. That's the qualitative win M4 books; the quantitative win waits for M6 (larger `Br` via register-resident `Õ`) and beyond.

---

## Change log

- **M4 landing commit** — this file is created; sections 1–10 enumerated.
- **M6 landing commit** — sections 1–5 marked *closed* or *partially closed* with links to Nsight metrics that moved.
- **M7 landing commit** — sections 7–9 marked *closed*.
- **M10 landing commit** — section 6 explicitly *not closed*; framed as "reading the gap."

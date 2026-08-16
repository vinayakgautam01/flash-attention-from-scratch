# FlashAttention v1 notes — deliberate non-choices and the v2 backlog

> **What this file is:** the running list of "we could have done this in v1 but chose not to." Each item names the concrete optimization, the reason we deferred it, and the milestone where it lands.
>
> **Why it exists:** M4's success criterion (`docs/MILESTONES.md` §M4) explicitly asks us to *"note obvious inefficiencies you're leaving for v2."* This document is that note. It is also, deliberately, **M6's TODO list**.

Everything below is measured against `csrc/flash_fwd_v1.cu` at the M4 landing commit. See [`../theory/M4.md`](../theory/M4.md) for the theory these choices sit on top of.

---

## Non-choices by category

### 1. Vectorized loads (`float4`, `__ldg`) — **CLOSED in M6**

- **Where it hurts.** Steps 1 (K/V load), init (Q load), and the terminal write of `O` all do scalar `float` loads/stores. Each HBM transaction pulls at most 32 × 4 = 128 B per warp; a `float4` load would pull 512 B per warp for the same instruction count.
- **Why deferred.** Adds a `D % 4 == 0` alignment requirement to `D` and a 4-way inner unroll to the load; extra code complexity in v1 without any correctness value.
- **What M6 did.** Q/K/V loads use `reinterpret_cast<const float4*>` through a linearized loop over the tile's `float4` count. Two corrections to the plan above:
  - **The "512 B per warp" framing was wrong.** v1's loads were *already* perfectly coalesced (32 lanes × 4 B = one full 128 B cache line, zero waste), so `float4` cannot reduce HBM *bytes*. What it reduces is the **instruction count** (4× fewer `LDG`) and the address arithmetic behind it. Since DRAM is only ~1.8% of v1's runtime (`theory/M6.md` §3.1), this is the **least** valuable of M6's changes — a fact worth knowing, since it is the item this list put first.
  - **The scalar fallback path is dead code; it was not written.** v2 supports only `D ∈ {32, 64}`, both multiples of 4, so `D % 4 == 0` holds by construction and is a `static_assert`, not a runtime branch.
  - The terminal `O` write stays scalar on purpose: vectorizing it would force each thread to own 4 *consecutive* `d`, turning STEP 7's conflict-free `sV` read into a 4-way conflict. See `theory/M6.md` §11.1.

### 2. Shared-memory bank conflicts on `sK`/`sV`/`sO` — **CLOSED in M6** (with a correction)

- **Where it hurts.** Step 2 reads `sK[tx * D + d]` for varying `tx` inside a warp with `d` fixed at each unrolled step. At `D = 32`, `tx * D + d ≡ d (mod 32)` for every `tx` → **32-way bank conflict**, serializing the shared-mem read 32×. At `D = 64`, same story: `tx * 64 + d ≡ d (mod 32)`. **This part was correct**, and it turned out to be ~91% of all shared-memory wavefronts in v1.
- **~~Same problem on `sV`~~ — this was WRONG.** The claim above read `c` as the lane-varying term in `sV[c * D + d]`, but `c` is the **inner loop counter** and is uniform across the warp; the lane enters through `d = tx + k·Bc`, giving a lane-stride of **1**. `sV` was already conflict-free in v1 and remains so in v2. The same audit clears `sQ` (broadcast), `sS` (broadcast on read, stride-1 on write) and `sO` (stride 1). **Exactly one line in v1 was conflicted.** The full per-access audit is in `theory/M6.md` §5.1.
- **Consequence of the correction.** M6 was a one-array, one-integer change — not the three-buffer layout redesign this section implied.
- **What M6 did.** `sK` padded to `[Bc][D+1]`. The lane-stride becomes `D+1`, which is odd, and `gcd(odd, 32) = 1` → conflict-free. Cost: `Bc × 4 = 128` bytes. Predicted wavefronts/warp/tile drop from 2252 to 268 (**8.4×**). The swizzle fallback was not needed; it is retained as an option in `theory/M6.md` §5.6 alongside the transpose alternative (which merely *moves* the conflict from reads to writes and is therefore not an alternative at all).
- **Metric to confirm.** `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` — recorded in M9's Nsight comparison.

### 3. `Õ` in shared memory instead of registers — **CLOSED in M6**

- **Where it hurts.** `sO[Br, D] = Br * D floats` sits in shared memory across the entire block loop. At `Br = 32, D = 64` that's 8 KB — a full quarter of the smem budget. Every step-7 read/write also incurs latency instead of hitting a register.
- **Why deferred.** Register-resident `Õ` requires each thread to own a specific slice of the `Br × D` output (e.g. thread `(tx, ty)` holds `Õ[ty, tx], Õ[ty, tx+Bc], ...` in a `float` array of length `D / Bc`). Adds a fixed-size register array indexed by an outer loop; v1 keeps the mapping obvious.
- **What M6 did.** `Õ` lives in `float acc_o[RowsPerThread][L]` with `L = D / Bc`. The critical detail this section did not anticipate: **`D` had to become a template parameter.** With `D` a runtime `int`, `L` is not compile-time, `ptxas` cannot allocate the array in registers, and it lands in *local* memory — physically DRAM, inside the hot loop. The optimization inverts into a pessimization, silently, with correct output. Hence the explicit `switch (D)` dispatch in the launch wrapper. See `theory/M6.md` §6.2.
- **Bonus not in the original plan.** The same "is this actually shared?" audit applies to `sS`, `sm` and `sl`. Row `i` of `P` is written and read entirely within one warp, so it travels by `__shfl_sync`; and both warp reductions broadcast to all lanes, so `(m, ℓ)` can be held redundantly in registers. All three buffers are gone, along with one `__syncthreads()` per tile. `theory/M6.md` §9.
- **Freed smem.** `Br*D*4 = 8 KB` (`Õ`) + `Br*Bc*4 = 4 KB` (`sS`) + 256 B (`sm`, `sl`). Total v2 smem at `D=64`: **24,704 B**, down from v1's 37,120 B.

### 4. Register pressure and occupancy tuning — **CLOSED in M6**

- **Where it hurts.** v1 launches 1024 threads/CTA with ~37 KB smem/CTA on Colab T4. T4 caps at 1024 threads/SM and 64 KB smem/SM → **exactly one CTA per SM at v1's launch config**. The SM has no other CTA's warps to hide latency behind.
- **Diagnosis.** Read `ptxas -v` output (already enabled in `CMakeLists.txt`) to see registers/thread; a spill would push us further. → **Now a checked-in deliverable: [`ptxas_v1_vs_v2.md`](ptxas_v1_vs_v2.md).**
- **What M6 did.** *Both* levers, because they are complementary rather than alternatives: `block(Bc, Br/2)` = 512 threads with 2 Q rows per thread, **and** the smem shrink from §3. Either alone gets stuck — 512 threads with v1's 37 KB smem still only fits one CTA (2 × 37,120 > 65,536), and 24 KB smem with 1024 threads is still one CTA by the thread limit. Together they reach **2 CTAs/SM** at both `D = 32` and `D = 64`.
- **The metric named above is the wrong one.** `sm__warps_active` (and `achieved_occupancy`) read **32 warps/SM and 100% in both configurations** — v1's one CTA of 32 warps and v2's two CTAs of 16. Occupancy counts resident warps, not *independent* ones. What actually changes is the number of independent barrier domains (1 → 2), so the metric to watch is the **stall-reason breakdown** (barrier stalls) and the fraction of cycles a scheduler had no eligible warp. `theory/M6.md` §7.2.
- **Free bonus from coarsening.** Each thread's two Q rows share their `sK` and `sV` loads, halving per-row shared traffic in STEPS 2 and 7 — provided the loops are restructured to hoist the shared load out and put the row loop innermost. `theory/M6.md` §7.5.

### 5. Async copies (`cp.async`, Ampere+) — **still open, M7**

- **Where it hurts.** Step 1's K/V load and step 2's compute run serially inside a tile iteration. On Ampere+, `cp.async` would overlap the next tile's K/V load with the current tile's compute (double-buffered smem tiles). Not available on Colab T4 (Turing) but real on Modal A10G (Ampere).
- **Why deferred.** Requires guarding on compute capability, double-buffering the smem layout, and pipelining the block loop.
- **Status after M6.** Not attempted — T4 is Turing, and M6's primary dev/measurement host is T4, so there would be no way to measure the result on the machine the milestone is tuned for. The §2 smem-layout prerequisite is now done, and §3 freed 12 KB, which makes the double-buffered `sK`/`sV` this needs affordable (`2 × (sK + sV)` at `D = 64` = 16,640 B on top of `sQ`'s 8,192 B = 24,832 B, still under 48 KB). `sm_86`-gated path is M7.

### 6. FA-2 loop reversal (outer = KV-tile) — future work, not M6

- **Where it hurts.** M4 uses FA-1 loop order (outer = Q-tile per CTA). At large `D` or small `Br`, `K/V` gets re-read `Tr = ⌈N/Br⌉` times, and the HBM ratio vs naive drops toward 1× (see `theory/M4.md` §10). FA-2 flips: outer = KV-tile, K/V loaded once, running state migrates between CTAs via the `(m, ℓ, Õ)` monoid combine (see `theory/M3.md` §8).
- **Why deferred.** This is FA-2, not "v2 as I mean it here." It requires cross-CTA scheduling and a partial-result merge step across CTAs. Cleaner to name it in the writeups (M9/M10) as *the specific optimization whose absence explains the gap to `flash-attn`*.
- **What later does.** Not implemented in v1.0.0 per `docs/AGENTS.md` §10 anti-scope. Named in M10's "reading the gap" analytical section.

### 7. `D = 128` support — M7

- **Where it hurts.** At `D = 128, Br = Bc = 32`: smem = `4 * (2*32*128 + 2*32*128 + 32*32 + 64) = 4 * (8192 + 8192 + 1024 + 64) = 69 888 B` — over the 48 KB default cap AND over the 64 KB max even with `cudaFuncSetAttribute` opt-in.
- **Why deferred.** Solving it needs the smem freed by register-resident `Õ` (§3) *and* possibly a `(Br, Bc)` re-pick.
- **Status after M6.** The prerequisite is done and the budget is now **better** than the estimate above. With `Õ`, `P`, `m` and `ℓ` all in registers, v2's layout at `D = 128` is `4 * (32*128 + 32*129 + 32*128) = 49 280 B` — not the ~60,928 B this section projected. It still exceeds the 48 KB default cap, so it remains gated on `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536)`, but it now fits comfortably rather than marginally, and two CTAs are out of reach at that `D` (2 × 49,280 > 65,536).
- **What M7 does.** Add the opt-in, add `case 128:` to the launch dispatch, and extend the test grid. `L = D/Bc = 4`, so `acc_o[2][4]` = 8 accumulator registers per thread — still cheap.

### 8. Batch and multi-head fan-out in the CTA grid — M7

- **Status in v1.** The kernel already handles arbitrary `(B, H)` correctly (grid decodes `blockIdx.y` as a linearized `(b, h)`, kernel body uses `b = bh / H`, `h = bh % H`). Only the *tests* are pinned to `B = H = 1`.
- **What M7 does.** Extend `tests/cpp/test_flash_fwd_v1.cu` (or a v3-specific test file) to cover `B ∈ {1, 2, 4}` and `H ∈ {1, 4, 8, 16}`. No kernel change expected.

### 9. Causal masking correctness testing — M7

- **Status in v1.** Kernel plumbs `is_causal` and applies the mask via a single `-inf` line in step 2 (`j_score > i` → `s_val = -inf`). Not tested in M4.
- **What M7 does.** Extend the test grid to `is_causal ∈ {false, true}`. Also exercises the diagonal-tile edge case where the mask crosses inside a single `Br × Bc` tile.

### 10. Tensor cores / WMMA — explicit anti-scope

- **Status.** Not implemented in v1.0.0 per `docs/AGENTS.md` §10 and `docs/MILESTONES.md` anti-scope.
- **Named in.** M10's "reading the gap" section — this is the single largest chunk of the gap to `flash-attn`.

### 11. Hidden `Br == Bc` coupling in the K/V load — **CLOSED in M6**

- **Where it hurts.** STEP 1's K/V load maps `ty → KV-row-within-tile`, so it fills exactly `Br` rows of `sK`/`sV`. Since `sK`/`sV` are shaped `[Bc × D]`, this only covers the tile fully when `Br == Bc`. `Br < Bc` leaves rows `[Br, Bc)` unwritten (STEP 2 reads garbage). `Br > Bc` writes past `sK` into `sV` (silent corruption). Currently guarded by a `static_assert(Br == Bc)` in the kernel; the coupling itself is a code-structure limitation, not a correctness bug.
- **Why deferred.** M4's `Br = Bc = 32` is pinned by three unrelated constraints (`Bc == warpSize`, `Br * Bc ≤ 1024`, and the smem budget), so `Br == Bc` falls out for free at v1's tile size. Generalizing costs code complexity we don't need to pay yet.
- **What M6 did.** Closed as a side effect of §1 — §4's occupancy change made the thread count (512) differ from `Br * Bc` (1024), so the load loop *had* to be linearized regardless. v2 iterates over the tile's `float4` count rather than its float count, which folds the two edits into one:
  ```cpp
  for (int v4 = tid; v4 < Bc * (D/4); v4 += kThreads) {
      const int r = v4 / (D/4), c4 = v4 % (D/4);
      // sK[r * (D+1) + 4*c4 + e] = K[..., j0 + r, 4*c4 + e]  (with tail-guard)
  }
  ```
  Coalescing is preserved (consecutive `tid` cover consecutive `float4` within a row). `static_assert(Br == Bc)` is gone; v2 asserts only what it actually needs: `Bc == 32`, `Br % RowsPerThread == 0`, `D % Bc == 0`, `D % 4 == 0`, and `threads ≤ 1024`. Asymmetric tiles are now reachable without further load-loop work — which is what makes the `Br = 64` experiment in `theory/M6.md` §10 a one-constant change if a future host makes it worthwhile.

### 12. 2-D register blocking of the S matmul — **opened by M6, deferred to v3**

New entry, added at the M6 landing commit. This is the largest remaining single lever and it did not exist as a named item before, because v1's bank conflicts were masking it.

- **Where it hurts.** Even with the conflicts gone, STEP 2 issues **2 `LDS` per `FFMA`** (one `sQ`, one `sK`). A Turing SM retires 64 FP32 FMA/cycle but only about one warp-wide `LDS`/cycle, so the arithmetic *cannot* saturate — shared memory gates the math by construction, not by tuning.
- **The fix.** Have each thread compute a `t × t` micro-tile of `S`, giving `LDS/FMA = 2t/t² = 2/t`. A `2×2` tile reaches 1, a `4×4` reaches 0.5. M6's 1-D coarsening (`RowsPerThread = 2`, §4) already captures part of this — `sK` is shared across 2 rows, so v2 sits at ~1.5.
- **Why not in M6.** A 2-D micro-tile breaks the **warp-per-Q-row invariant** that the `__shfl_xor` row reductions in STEPS 3 and 5 depend on. Both would have to be redesigned as cross-lane partial reductions. That is a rewrite, not a tuning pass, and it collides with M6's stated scope ("same algebra, better mechanical sympathy").
- **What v3 does.** Redesign the thread↔`S`-element mapping and the reduction structure together, as one change, with the v2 kernel kept alongside for A/B measurement.

---

## Under-utilization at small `N`

Named separately because it's not fixable by any of the items above — it's a *launch-config* problem.

- **Where it hurts.** At `N = 512, Br = 32`: `Tr = 16` CTAs. T4 has 40 SMs. **24 SMs idle for the whole kernel.**
- **What v2 doesn't fix** *(confirmed — v2 shipped and this is unchanged)*. Even with 2 CTAs/SM from §4, 32 SMs would run — still 8 idle. To saturate at small `N` you need cross-`(B, H)` grid dims (§8) or FA-2-style sequence-parallel work partitioning (§6). Note the cross-reference above originally said "§3"; the occupancy item is §4.
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

- **M4 landing commit** — this file is created; sections 1–10 enumerated (§11 added shortly after).
- **M6 landing commit** — sections **1, 2, 3, 4, 11 closed**; section 5 restated as M7-only (T4 is Turing); section 7's budget revised downward; section **12 opened** as the largest remaining lever. **One substantive correction:** §2's claim that `sV` suffered the same 32-way conflict as `sK` was wrong — `sV`'s lane-stride is 1. The full per-access audit that settles it is in [`../theory/M6.md`](../theory/M6.md) §5.1, and the error is dissected in §5.2 because the *reason* it was wrong (mistaking a loop counter for the lane index) is the single most common way to misread a shared-memory access pattern.
- **M7 landing commit** — sections 5, 7, 8, 9 to be marked *closed*.
- **M9 landing commit** — Nsight metrics backfilled into §2 and §4 (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared`, stall-reason breakdown).
- **M10 landing commit** — section 6 explicitly *not closed*; framed as "reading the gap."

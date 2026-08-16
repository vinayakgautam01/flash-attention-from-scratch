# `ptxas -v`: Flash v1 vs Flash v2

> **What this file is:** the checked-in compiler resource report for both Flash kernels, side by side, with an interpretation. It is a required M6 deliverable (`docs/MILESTONES.md` §M6) because the occupancy argument in [`../theory/M6.md`](../theory/M6.md) §7 is a *claim about registers and shared memory*, and a claim about registers should be settled by the register allocator, not by prose.
>
> **Status: measured.** Captured on Colab **Tesla T4 (sm_75)**, `nvcc` 12.8, repo commit `5d07586`. The *predictions* in §4 were written before the measurement, on purpose, and are scored there.
>
> **Headline: no spills anywhere, and warps/SM is 32 for both kernels — identical "occupancy" — yet v2 is 13.2× faster at `N = 2048, D = 64`.** That combination is the whole point of this file; see §3.

---

## 1. How to reproduce

`--ptxas-options=-v` is already enabled on both kernel targets in `CMakeLists.txt`, so a clean build prints the report. On Colab T4:

```bash
# Fresh build so nothing is served from the CMake cache.
rm -rf build
CMAKE_CUDA_ARCHITECTURES=75 scripts/build.sh 2>&1 | tee /tmp/build_log.txt

# Pull out just the two kernels' resource lines.
grep -A3 -E "flash_fwd_v(1|2)_kernel" /tmp/build_log.txt
```

Each kernel emits a block shaped like:

```
ptxas info : Compiling entry function '_Z...flash_fwd_v1_kernel...' for 'sm_75'
ptxas info : Function properties for _Z...flash_fwd_v1_kernel...
    <S> bytes stack frame, <A> bytes spill stores, <B> bytes spill loads
ptxas info : Used <R> registers, <C> bytes cmem[0]
```

Two notes on reading it:

- **v1 emits one block** (it is templated on `<Br, Bc>` only; `D` is a runtime argument).
- **v2 emits two blocks** — one per `D` instantiation (`<32,32,32,2>` and `<32,32,64,2>`), because `D` is a template parameter. That is deliberate and is the thing that makes the register-resident accumulator possible; see `theory/M6.md` §6.2.
- Shared memory does **not** appear in this output for either kernel: both use `extern __shared__` (dynamic), which is sized at launch, not at compile time. The smem figures below come from `flash_v2_smem_bytes()` in the header, which `test_flash_fwd_v2.cu::FlashFwdV2Config` asserts against.

---

## 2. Resource table

Both halves are from the same build.

### 2.1 Registers, stack, spills

| Kernel | Instantiation | Threads/CTA | Registers/thread | Registers/CTA | Stack frame | Spill stores | Spill loads | cmem[0] |
|---|---|---|---|---|---|---|---|---|
| `flash_fwd_v1` | `<32, 32>` | 1024 | **63** | 64,512 | 0 B | **0 B** | **0 B** | 408 B |
| `flash_fwd_v2` | `<32, 32, 32, 2>` | 512 | **64** | 32,768 | 0 B | **0 B** | **0 B** | 404 B |
| `flash_fwd_v2` | `<32, 32, 64, 2>` | 512 | **64** | 32,768 | 0 B | **0 B** | **0 B** | 404 B |

Both `D` instantiations of v2 land on the same register count, which is expected: the extra accumulator registers at `D = 64` are `L = D/Bc = 2` per Q row versus 1, i.e. two more live floats, well inside the slack `ptxas` already had.

### 2.2 Shared memory and derived occupancy

The smem column is computed (arithmetic in `theory/M6.md` §7.6, constants asserted by the config guard test); the register column now uses the **measured** counts from §2.1 rather than an estimate.

| Kernel | `D` | smem/CTA | Threads/CTA | CTAs/SM (thread) | CTAs/SM (register) | CTAs/SM (smem) | **CTAs/SM** | Warps/SM |
|---|---|---|---|---|---|---|---|---|
| `flash_fwd_v1` | 32 | 20,736 B | 1024 | 1 | 1 | 3 | **1** | 32 |
| `flash_fwd_v1` | 64 | 37,120 B | 1024 | 1 | 1 | 1 | **1** | 32 |
| `flash_fwd_v2` | 32 | 12,416 B | 512 | 2 | 2 | 5 | **2** | 32 |
| `flash_fwd_v2` | 64 | 24,704 B | 512 | 2 | 2 | 2 | **2** | 32 |

T4 (sm_75) limits used: 1024 threads/SM, 32 warps/SM, 65,536 registers/SM, 64 KB smem/SM, 16 CTAs/SM.

**One Turing-specific caveat on the smem column.** A Turing SM has 96 KB of combined L1 + shared, and the shared-memory *carveout* is quantized (0 / 8 / 16 / 32 / 64 KB). Two v2 CTAs at `D = 64` need 49,408 B, which is above the 32 KB tier — so reaching 2 CTAs/SM depends on the driver selecting the 64 KB carveout. It does this automatically from the launch's dynamic-smem request, so no `cudaFuncSetAttribute` call is needed. But if a measured occupancy ever comes back at 1 CTA when this table says 2, the carveout is the first thing to check, not the register count.

**The row that carries the whole argument is `D = 64`.** v1 is pinned to one CTA per SM by the *thread* limit, with shared memory also at exactly one. v2 reaches two CTAs on both limits simultaneously. Warps per SM — and therefore reported "occupancy" — is **32 in every row**. That is the point: occupancy does not distinguish these configurations, and any writeup that leans on it is leaning on the wrong number.

---

## 3. Interpretation

**Did the register budget bind? No — and that makes the M6 story simpler, not weaker.** v1 reports **63 registers, 0 bytes of stack frame, 0 spill stores, 0 spill loads**. So there is no hidden second bottleneck underneath the bank conflicts: the register axis was never v1's problem, and the 512-thread change cannot be justified by "it stops the spilling," because there was none to stop. Per the fork pre-registered in §4, that leaves the M6 speedup attributable almost entirely to the bank-conflict fix (`theory/M6.md` §5) plus barrier independence (§7.2) — which was always the stronger argument.

One detail is worth not glossing over. 63 registers is *one under* the hard ceiling of \(65{,}536 / 1024 = 64\). v1 did not spill, but it had no headroom whatsoever; `ptxas` fit it with a single register to spare. So the escape-hatch argument in `theory/M6.md` §7.3 was not hypothetical hand-waving — v1 was genuinely parked at the edge of the cliff. It simply happened not to fall off.

**Is the occupancy claim consistent? Yes, on all three limits simultaneously.** v2 declares `__launch_bounds__(512, 2)`, which asks for two resident CTAs of 512 threads and therefore implies a 64-register ceiling. `ptxas` lands on **exactly 64 registers with zero spills**, at both `D = 32` and `D = 64`. That is the bound being honoured precisely rather than approximately: \(512 \times 64 = 32{,}768\) registers per CTA, so \(65{,}536 / 32{,}768 = 2\) CTAs fit on the register limit — and the thread limit and the shared-memory limit independently give 2 as well. The pre-registered fallback (relax the second `__launch_bounds__` argument to 1, accept one CTA and up to 128 registers) was **not needed** and should stay unused.

**The observation that carries the milestone.** Warps per SM is **32 in every row of §2.2** — v1 and v2 are identical on that axis. Any tool reporting "achieved occupancy" will show the same number for both. And yet v2 is **13.2× faster** at \(N = 2048, D = 64\) (11.69 ms → 0.88 ms). This is the concrete vindication of `theory/M6.md` §7.2: occupancy does not distinguish these two kernels, so a writeup that credits the speedup to "higher occupancy" would be citing a metric that provably did not move. What changed is the number of *independent barrier domains* (1 → 2) and the number of shared-memory wavefronts on the hot dot-product read.

**One thing these numbers do not prove.** `ptxas` output establishes that two CTAs *may* be co-resident; it does not establish that the SM actually ran two. Confirming residency, and attributing the speedup between the bank-conflict fix and barrier independence, needs Nsight Compute — specifically `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` for the former. That measurement is M9's job, and the 13.2× reported here is a runtime fact that stands independently of how the credit is eventually split.

---

## 4. Predictions, recorded before measurement

Rough register demand for `flash_fwd_v2_kernel<32,32,64,2>`, counted by hand from the source:

| Live state | Count |
|---|---|
| `acc_o[2][2]` | 4 |
| `m_run[2]`, `l_run[2]` | 4 |
| `p_val[2]`, `alpha[2]`, `s_acc[2]` | 6 |
| `q_local[2]`, `i_glob[2]`, `i_ok[2]` | ~6 |
| `v_reg[2]` | 2 |
| addresses, loop counters, unroll temporaries | ~10–15 |
| **Total** | **~35–45** |

So the `(512, 2)` launch bound should be comfortable, and **the prediction is: no spills in v2, at either `D`.**

For v1 the prediction is weaker, because its live state is smaller per thread (one Q row, `Õ` in shared memory) but its inner loops are unrolled 4× and 8×. Either outcome is informative:

- **v1 spills** → the M6 story gains a second, independent mechanism, and the 512-thread change is doing more work than the barrier argument alone claims.
- **v1 does not spill** → the register axis was never the problem, and M6's speedup should be attributable almost entirely to the bank-conflict fix (`theory/M6.md` §5) plus barrier independence (§7.2).

### 4.1 Scoring, after the fact

| Prediction | Outcome | Verdict |
|---|---|---|
| No spills in v2, at either `D` | 0 stack, 0 spill stores, 0 spill loads, both instantiations | **Correct** |
| v2 register demand ~35–45 | `ptxas` reports 64 | **Correct in spirit, wrong as a number** — see below |
| v1 spills / does not spill (open fork) | Does **not** spill (63 regs, 0 bytes) | Second branch taken |

The hand count of ~35–45 registers was not wrong about the *live state* it enumerated; it was wrong to treat that as the register demand. `ptxas` also needs registers for unroll temporaries, address staging, and the shuffle traffic, and — more importantly — it has no incentive to stop at 45 when the launch bound permits 64. Since `__launch_bounds__(512, 2)` sets the ceiling at exactly 64, landing on 64 tells us the allocator used its whole allowance, not that the kernel needed every register. The useful conclusion is the one the prediction was actually testing: **the bound is satisfiable without spilling**, with the hand count's real content being that it was never close to forcing a spill.

Writing the prediction down before the measurement is the point. If it turns out wrong, **the wrong prediction and its correction stay in this file** — that is the honest-reporting standard the repo holds itself to (`docs/MILESTONES.md`, cross-cutting standards).

---

## 5. Cross-references

- [`../theory/M6.md`](../theory/M6.md) §7 — the occupancy derivation these numbers test, including §7.3 on why halving the block size does **not** double the per-thread register budget.
- [`flash_attention_notes.md`](flash_attention_notes.md) §4 — the original M4-era observation that v1 is pinned to one CTA per SM.
- [`../csrc/flash_fwd_v2_shared_kv.cu`](../csrc/flash_fwd_v2_shared_kv.cu) — the `__launch_bounds__` declaration and the comment describing the fallback if it forces spills.
- `docs/nsight_comparison.md` (M9) — where the *runtime* consequences of these resource numbers get measured rather than derived.

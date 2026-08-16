# `ptxas -v`: Flash v1 vs Flash v2

> **What this file is:** the checked-in compiler resource report for both Flash kernels, side by side, with an interpretation. It is a required M6 deliverable (`docs/MILESTONES.md` §M6) because the occupancy argument in [`../theory/M6.md`](../theory/M6.md) §7 is a *claim about registers and shared memory*, and a claim about registers should be settled by the register allocator, not by prose.
>
> **Status: numbers pending.** The tables below are captured on a GPU host (Colab T4 first, Modal A10G for the M9 record). The *predictions* are filled in now, before the measurement, on purpose — see §4.

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

Fill both halves from the same build.

### 2.1 Registers, stack, spills

| Kernel | Instantiation | Threads/CTA | Registers/thread | Stack frame | Spill stores | Spill loads |
|---|---|---|---|---|---|---|
| `flash_fwd_v1` | `<32, 32>` | 1024 | _pending_ | _pending_ | _pending_ | _pending_ |
| `flash_fwd_v2` | `<32, 32, 32, 2>` | 512 | _pending_ | _pending_ | _pending_ | _pending_ |
| `flash_fwd_v2` | `<32, 32, 64, 2>` | 512 | _pending_ | _pending_ | _pending_ | _pending_ |

### 2.2 Shared memory and derived occupancy

These are computed, not measured — the arithmetic is in `theory/M6.md` §7.6 and the constants are asserted by the config guard test.

| Kernel | `D` | smem/CTA | Threads/CTA | CTAs/SM (thread limit) | CTAs/SM (smem limit) | Warps/SM |
|---|---|---|---|---|---|---|
| `flash_fwd_v1` | 32 | 20,736 B | 1024 | 1 | 3 | 32 |
| `flash_fwd_v1` | 64 | 37,120 B | 1024 | 1 | 1 | 32 |
| `flash_fwd_v2` | 32 | 12,416 B | 512 | 2 | 5 | 32 |
| `flash_fwd_v2` | 64 | 24,704 B | 512 | 2 | 2 | 32 |

T4 (sm_75) limits used: 1024 threads/SM, 32 warps/SM, 65,536 registers/SM, 64 KB smem/SM, 16 CTAs/SM.

**One Turing-specific caveat on the smem column.** A Turing SM has 96 KB of combined L1 + shared, and the shared-memory *carveout* is quantized (0 / 8 / 16 / 32 / 64 KB). Two v2 CTAs at `D = 64` need 49,408 B, which is above the 32 KB tier — so reaching 2 CTAs/SM depends on the driver selecting the 64 KB carveout. It does this automatically from the launch's dynamic-smem request, so no `cudaFuncSetAttribute` call is needed. But if a measured occupancy ever comes back at 1 CTA when this table says 2, the carveout is the first thing to check, not the register count.

**The row that carries the whole argument is `D = 64`.** v1 is pinned to one CTA per SM by the *thread* limit, with shared memory also at exactly one. v2 reaches two CTAs on both limits simultaneously. Warps per SM — and therefore reported "occupancy" — is **32 in every row**. That is the point: occupancy does not distinguish these configurations, and any writeup that leans on it is leaning on the wrong number.

---

## 3. Interpretation

_To be written once the tables are filled. The two paragraphs must answer:_

**Paragraph 1 — did the register budget bind?** v1 runs 1024-thread CTAs, so `ptxas` had a hard ceiling of \(65{,}536 / 1024 = 64\) registers/thread with no alternative: a kernel needing more must spill, because 65 registers/thread would make even one CTA unschedulable. State whether v1's report shows non-zero spill stores/loads. If it does, that is a *second* bottleneck sitting underneath the bank conflicts, and it is entirely an artifact of the block size. If it does not, say so plainly — the register story was a non-issue and the 512-thread change is justified purely by barrier independence (`theory/M6.md` §7.2), which is the stronger argument anyway.

**Paragraph 2 — is the occupancy claim consistent?** v2 declares `__launch_bounds__(512, 2)`, which asks `ptxas` to fit two CTAs of 512 threads, implying the same 64-register ceiling. Confirm the report honours it *without* spilling. If v2 spills where v1 did not, the coarsening (2 Q rows/thread ⇒ roughly double the per-thread live state) has overshot, and the correct response is to relax the second argument to `1` — accepting one CTA per SM and up to 128 registers — then re-measure. That fork is real and pre-registered; see §4.

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

Writing the prediction down before the measurement is the point. If it turns out wrong, **the wrong prediction and its correction stay in this file** — that is the honest-reporting standard the repo holds itself to (`docs/MILESTONES.md`, cross-cutting standards).

---

## 5. Cross-references

- [`../theory/M6.md`](../theory/M6.md) §7 — the occupancy derivation these numbers test, including §7.3 on why halving the block size does **not** double the per-thread register budget.
- [`flash_attention_notes.md`](flash_attention_notes.md) §4 — the original M4-era observation that v1 is pinned to one CTA per SM.
- [`../csrc/flash_fwd_v2_shared_kv.cu`](../csrc/flash_fwd_v2_shared_kv.cu) — the `__launch_bounds__` declaration and the comment describing the fallback if it forces spills.
- `docs/nsight_comparison.md` (M9) — where the *runtime* consequences of these resource numbers get measured rather than derived.

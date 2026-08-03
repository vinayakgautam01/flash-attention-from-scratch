# Online softmax derivation

> **Status: landed in M3. Finalized in M9** (adds the M4 tiled-forward pointer
> and the roofline placement).

This doc derives the online-softmax recurrence and shows — from first principles
— why the block-wise version used in the M3 reference kernel (and every Flash
kernel thereafter) is *algebraically equal* to the two-pass stable softmax, not
an approximation. The full pedagogical walk-through, including the CUDA-mapping
discussion and glossary, is in [`../theory/M3.md`](../theory/M3.md); this file
is the compressed *evidence* that goes with the claim.

---

## 1. Setup

Fix one query row and drop the `(b, h, i)` subscript. Given `N` scores
`s_0, …, s_{N−1}` and value rows `V_0, …, V_{N−1} ∈ ℝ^D`, the target is one row
of the attention output:

\[
O = \frac{1}{\ell}\sum_{j=0}^{N-1} \exp(s_j - m) \cdot V_j, \quad
m = \max_j s_j, \quad
\ell = \sum_j \exp(s_j - m).
\]

The naive way needs three passes over `j` — one for `m`, one for `ℓ`, one for
`O` — which forces every `s_j` (i.e. the `N×N` matrix `S`) to live in HBM.
Online softmax collapses this to **one pass**, streaming K/V in blocks, with
only `(m, ℓ, Õ)` in register/shared state — `(D + 2)` floats **independent of
N**.

---

## 2. The invariant

After processing the first `k` scores, we maintain:

\[
\boxed{\;m_k = \max_{j<k} s_j, \quad
\ell_k = \sum_{j<k} \exp(s_j - m_k), \quad
\tilde{O}_k = \sum_{j<k} \exp(s_j - m_k)\cdot V_j.\;}
\]

The critical detail — the one thing that confuses everyone the first time — is
that **every term inside `ℓ_k` and `Õ_k` is offset by the *current* running max
`m_k`, not by whatever max was active when the term was first added.**

At the terminal step `k = N`:

\[
O = \tilde{O}_N \,/\, \ell_N.
\]

---

## 3. Updating `(m, ℓ)` when a new score arrives

Suppose we've accumulated up through `k` and score `s_k` arrives.

New running max:

\[
m_{k+1} = \max(m_k, s_k).
\]

We want the new `ℓ_{k+1} = Σ_{j≤k} \exp(s_j − m_{k+1})`. The already-summed
terms inside `ℓ_k` are offset by `m_k`, not `m_{k+1}`. But this factors:

\[
\exp(s_j - m_{k+1}) = \exp(s_j - m_k)\cdot\exp(m_k - m_{k+1}).
\]

The factor `exp(m_k − m_{k+1})` is *independent of `j`*, so it pulls outside the
sum. Let `α_k = exp(m_k − m_{k+1})`. Then

\[
\ell_{k+1} = \alpha_k \cdot \ell_k + \exp(s_k - m_{k+1}).
\]

This is the whole trick: `α_k` **converts the running sum from offset `m_k` to
offset `m_{k+1}`** before adding the new (already-`m_{k+1}`-offset) term.
Because `m_{k+1} ≥ m_k`, `α_k ∈ (0, 1]` — no overflow, ever.

Proof of the invariant (induction on `k`): assume `ℓ_k = Σ_{j<k} exp(s_j − m_k)`.
Then

\[
\ell_k\,\alpha_k + \exp(s_k - m_{k+1})
= \sum_{j<k}\exp(s_j - m_k)\exp(m_k - m_{k+1}) + \exp(s_k - m_{k+1})
= \sum_{j\le k}\exp(s_j - m_{k+1}).\qquad\blacksquare
\]

---

## 4. Updating `Õ` (fusing V)

Define the unnormalized output `Õ_k = Σ_{j<k} \exp(s_j − m_k)\,V_j`. The same
factoring argument gives

\[
\tilde{O}_{k+1}
= \sum_{j\le k}\exp(s_j - m_{k+1})\,V_j
= \alpha_k \cdot \tilde{O}_k + \exp(s_k - m_{k+1})\,V_k.
\]

The full recurrence:

\[
\begin{aligned}
m_{k+1} &= \max(m_k,\; s_k) \\
\alpha_k &= \exp(m_k - m_{k+1}) \\
\beta_k &= \exp(s_k - m_{k+1}) \\
\ell_{k+1} &= \alpha_k\,\ell_k + \beta_k \\
\tilde{O}_{k+1} &= \alpha_k\,\tilde{O}_k + \beta_k\,V_k
\end{aligned}
\]

Initial state: `m_0 = −∞, ℓ_0 = 0, Õ_0 = 0`. `α_0 = 0`, so the initial zeros
are annihilated on the first update. Terminal: `O = Õ_N / ℓ_N`.

---

## 5. The block version (what the CUDA kernel actually runs)

Streaming one score at a time is wasteful on GPU. Process `Bc` scores at a time
instead. For block `b` covering columns `j ∈ [b·Bc, (b+1)·Bc)`:

1. Compute the block's `Bc` scores `s^{(b)}_c` from `Q_i · K_{b·Bc+c} / √D`.
2. **Local max** (one block reduction over `Bc` threads):
   `m̃^{(b)} = max_c s^{(b)}_c`.
3. **New running max**: `m^{(b+1)} = max(m^{(b)}, m̃^{(b)})`.
4. **Local weights** offset by the *new* max:
   `p̃^{(b)}_c = exp(s^{(b)}_c − m^{(b+1)})`.
5. **Local sum** (second block reduction): `ℓ̃^{(b)} = Σ_c p̃^{(b)}_c`.
6. **Rescale factor**: `α^{(b)} = exp(m^{(b)} − m^{(b+1)}) ∈ (0, 1]`.
7. **Merge**:
   \[
   \ell^{(b+1)} = \alpha^{(b)}\,\ell^{(b)} + \tilde{\ell}^{(b)}, \qquad
   \tilde{O}^{(b+1)} = \alpha^{(b)}\,\tilde{O}^{(b)} + \sum_c \tilde{p}^{(b)}_c\,V_{b\cdot B_c + c}.
   \]

After `⌈N / Bc⌉` blocks: `O_i = Õ^{(last)} / ℓ^{(last)}`.

This is exactly the recurrence in §4, unrolled `Bc` at a time and rearranged so
the two operations that need cross-thread visibility (`m̃`, `ℓ̃`) each become
one block reduction. Nothing new, just packaging.

**Per-row state carried across blocks:** `(m, ℓ, Õ[D])` = `(D + 2)` floats,
independent of `N`. This is the linear-in-N HBM footprint story of
FlashAttention, in one sentence.

The full block-picture with ASCII diagrams (which slice of `K/V` each block
owns, what the CTA/thread layout looks like) lives in
[`../theory/M3.md`](../theory/M3.md) §7.

---

## 6. Merging two arbitrary partial outputs (the monoid view)

Given two disjoint partial results `(m_A, ℓ_A, Õ_A)` and `(m_B, ℓ_B, Õ_B)`:

\[
m = \max(m_A, m_B), \qquad
\ell = e^{m_A-m}\,\ell_A + e^{m_B-m}\,\ell_B, \qquad
\tilde{O} = e^{m_A-m}\,\tilde{O}_A + e^{m_B-m}\,\tilde{O}_B.
\]

This is `α` applied to both sides, generalizing §5. It shows that
`(m, ℓ, Õ)` forms a **commutative monoid** under this combine — so the block
loop's serial merge can be reshaped as any parallel reduction. This is the
identity FlashAttention-2 uses for split-K work partitioning; M3 doesn't need
it (we're strictly serial in `b`), but naming it makes clear that the M4 kernel
is not doing anything algebraically new.

---

## 7. What breaks without α — the toy figure

The script [`toy_online_softmax.py`](toy_online_softmax.py) runs the recurrence
on `N=8, Bc=2, D=1` inputs

\[
s = [1.0,\; 3.0,\; 2.0,\; 5.0,\; 0.0,\; 4.0,\; 1.5,\; 2.5], \qquad
V = [0.1,\; 0.2,\; 0.3,\; 0.4,\; 0.5,\; 0.6,\; 0.7,\; 0.8]
\]

and prints two traces side-by-side.

**Baseline** (stable two-pass softmax + PV):
`m = 5.000000, ℓ = 1.690338, O* = 0.446501`.

**With α — correct online recurrence:**

| block | j     |   m_old |   m_new |         α |   ℓ_local |     ℓ_new |    Õ_new |
| ----: | :---- | ------: | ------: | --------: | --------: | --------: | -------: |
|     0 | 0..1  |    −inf |  3.0000 |  0.000000 |  1.135335 |  1.135335 | 0.213534 |
|     1 | 2..3  |  3.0000 |  5.0000 |  0.135335 |  1.049787 |  1.203438 | 0.443835 |
|     2 | 4..5  |  5.0000 |  5.0000 |  1.000000 |  0.374617 |  1.578055 | 0.667931 |
|     3 | 6..7  |  5.0000 |  5.0000 |  1.000000 |  0.112282 |  1.690338 | 0.754738 |

Final `O = Õ/ℓ = 0.754738 / 1.690338 = 0.446501`. **Bit-exact** with the
baseline (diff `= 0.0` in fp64, well under the `< 1e-12` gate).

**Without α — broken variant:**

| block | j     |   m_old |   m_new |         α |   ℓ_local |     ℓ_new |    Õ_new |
| ----: | :---- | ------: | ------: | --------: | --------: | --------: | -------: |
|     0 | 0..1  |    −inf |  3.0000 |         — |  1.135335 |  1.135335 | 0.213534 |
|     1 | 2..3  |  3.0000 |  5.0000 |         — |  1.049787 |  2.185122 | 0.628470 |
|     2 | 4..5  |  5.0000 |  5.0000 |         — |  0.374617 |  2.559740 | 0.852566 |
|     3 | 6..7  |  5.0000 |  5.0000 |         — |  0.112282 |  2.672022 | 0.939372 |

Final `O = 0.939372 / 2.672022 = 0.351559`. **`ℓ` is off by +58%, `O` is off by
−21%.** No overflow, no NaN, just silently-wrong arithmetic — because the sums
`ℓ` and `Õ` still contain terms offset by `m = 3.0` (from block 0), mixed with
terms offset by `m = 5.0` (from blocks 1–3). Apples plus oranges.

This is exactly why the M3 correctness gate uses a **tighter** tolerance (`1e-5`
abs vs M2's `5e-4`): M3 is not doing floating-point matmul accumulation, it's
doing algebraic identity plus a handful of `exp`s, and the answer should be
bit-close to the two-pass baseline.

---

## 8. Correctness gate summary

- **Reference**: `csrc/attention_cpu_ref.hpp` (fp64 math, from M1).
- **Under test**: `csrc/attention_online_ref.cu` (M3 kernel).
- **Grid**: `N ∈ {128, 512, 2048}`, `D ∈ {32, 64}`, `B = H = 1`,
  `causal = false`.
- **Tolerance**: `1e-5` abs (fp32) — see `docs/AGENTS.md` §9 row for M3.
- **Edge cases**: `N=1`, uniform Q/K, non-`Bc`-multiple `N`, peak-key,
  determinism across two runs.

If M3 passes, the recurrence is on solid ground and M4 is (as promised) layout
work, not algebra.

---

## 9. What comes next in M4

The M4 tiled forward kernel is one small change to the M3 kernel: instead of
one CTA per query row, **one CTA per block of `Br` query rows**. The `sK, sV`
shared-memory slice loaded for one K/V block is then reused across all `Br`
query rows in the CTA, and each row keeps its own copy of `(m, ℓ, Õ)`. The
recurrence is unchanged.

That is the whole M4 story from the point of view of this file. Everything else
in M4 is choosing `Br, Bc` from the shared-memory budget and drawing a
CTA-grid diagram. See [`../theory/M3.md`](../theory/M3.md) §7.7 for the exact
CTA layout that M3 already uses, and MILESTONES §M4 for the target shape.

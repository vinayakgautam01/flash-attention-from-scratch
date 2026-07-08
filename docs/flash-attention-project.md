My recommendation: **add FlashAttention as a capstone project, not just another small kernel.** Make it the bridge between your existing “GPU parallel patterns” repo and real LLM systems work.

Your current repo already looks strong because it is structured around **baseline → optimized variants → correctness tests → benchmarks → plots → Nsight profiling**. It covers convolution, stencil, histogram, reduction, scan, merge, GEMM, and sort, with techniques like shared-memory tiling, warp shuffles, privatization, decoupled look-back scan, and tiled matrix multiplication. ([GitHub][1]) FlashAttention is a perfect next project because it combines **GEMM + softmax/reduction + tiling + memory hierarchy + numerical stability**.

## Best portfolio project idea

Build a repo called something like:

**`cuda-flash-attention-lab`**
or
**`flash-attention-from-scratch`**

Position it as:

> An educational CUDA implementation of exact scaled dot-product attention, progressing from naive attention to a tiled IO-aware FlashAttention-style forward kernel, with correctness tests, PyTorch comparison, benchmarks, and Nsight profiling.

Do **not** try to beat the official FlashAttention library. The official project is already highly optimized and production-grade. FlashAttention’s original contribution was reducing HBM traffic by tiling Q/K/V and avoiding materializing the full `N x N` attention matrix. ([arXiv][2]) FlashAttention-2 then improved parallelism and work partitioning, reaching much higher GPU utilization on A100-class hardware. ([arXiv][3]) For portfolio purposes, your goal should be: **show that you understand the algorithm, memory behavior, numerical stability, and CUDA trade-offs deeply.**

## What to implement

I would implement it in stages.

### Stage 1: Naive attention baseline

Implement standard attention:

```text
S = QK^T / sqrt(d)
P = softmax(S)
O = PV
```

Have these baselines:

```text
baseline_cpu.cpp
baseline_torch.py
baseline_cuda_multi_kernel.cu
```

The CUDA baseline can be intentionally simple:

```text
1. GEMM-like kernel for QK^T
2. softmax kernel
3. GEMM-like kernel for PV
```

This gives you a clear memory problem to explain: the naive version materializes the full `N x N` score matrix and probability matrix.

### Stage 2: Online softmax reference

Before writing the FlashAttention kernel, implement the **online softmax recurrence** in CPU/CUDA reference form.

This is the key concept recruiters/interviewers will care about. You want to show that you understand how FlashAttention updates the running max and normalization term block by block:

```text
m_new = max(m_old, max(scores_block))
l_new = exp(m_old - m_new) * l_old + sum(exp(scores_block - m_new))
O_new = rescaled_old_O + contribution_from_current_block
```

This makes the project much stronger than “I copied a tiled kernel.”

### Stage 3: FlashAttention-style forward kernel

Implement **forward pass only** first. This is enough for a strong systems portfolio project.

Start with:

```text
Q, K, V: [B, H, N, D]
O:       [B, H, N, D]
dtype:   float32 first, then fp16 optional
```

Kernel design:

```text
One CTA handles a block of query rows.
Loop over K/V blocks.
Load K and V tiles into shared memory.
Compute QK^T tile.
Apply online softmax.
Accumulate O without writing S or P to global memory.
```

This directly shows the original FlashAttention idea: exact attention, but less HBM traffic by using on-chip memory and recomputation/streaming instead of storing the full attention matrix. ([arXiv][2])

### Stage 4: Make it practical

Add features in this order:

```text
1. Non-causal attention
2. Causal attention
3. Arbitrary sequence lengths with boundary handling
4. Batch and multi-head support
5. Head dimensions: 32, 64, 128
6. fp16 input with fp32 accumulation
7. Optional PyTorch extension
```

For PyTorch comparison, benchmark against `torch.nn.functional.scaled_dot_product_attention`, because PyTorch can automatically select optimized SDPA backends depending on the input and hardware. ([PyTorch Documentation][4]) Also compare against the official `flash-attn` package as a reference baseline where available; its public API implements `softmax(Q @ K^T * scale) @ V`. ([PyPI][5])

## Repo layout I would use

Mirror your existing repo style:

```text
cuda-flash-attention-lab/
├── csrc/
│   ├── attention_cpu_ref.hpp
│   ├── attention_naive.cu
│   ├── attention_online_ref.cu
│   ├── flash_fwd_v1.cu
│   ├── flash_fwd_v2_shared_kv.cu
│   ├── flash_fwd_v3_causal.cu
│   └── kernels.hpp
├── tests/
│   ├── test_correctness.cpp
│   ├── test_numerics.py
│   └── test_against_torch.py
├── benchmarks/
│   ├── bench_attention.py
│   ├── bench_cuda.cpp
│   └── results/
├── scripts/
│   ├── build.sh
│   ├── test.sh
│   ├── bench.sh
│   └── profile_ncu.sh
├── docs/
│   ├── flash_attention_notes.md
│   ├── online_softmax_derivation.md
│   ├── memory_analysis.md
│   └── plots/
├── README.md
└── CMakeLists.txt
```

The README should look like your existing GPU patterns README: variant table, correctness strategy, benchmark table, profiling screenshots, and explanation of trade-offs.

## The variant table should be the heart of the project

Use something like this:

| Variant             | Description                                  | What it demonstrates |
| ------------------- | -------------------------------------------- | -------------------- |
| CPU reference       | Direct attention in C++/Python               | Correctness oracle   |
| Naive CUDA          | Materialize `S` and `P`                      | Baseline memory cost |
| Online softmax CUDA | Streaming softmax without full matrix        | Numerical stability  |
| Flash v1            | Tiled K/V, no `N x N` write                  | IO-aware attention   |
| Flash v2            | Better shared-memory layout/vectorized loads | Memory coalescing    |
| Flash causal        | Causal masking                               | LLM-style attention  |
| Flash fp16          | fp16 input, fp32 accumulation                | Mixed precision      |

This is portfolio-friendly because it tells a story.

## What benchmarks to show

Benchmark these shapes:

```text
B = 1, 2, 4
H = 8, 16
N = 128, 256, 512, 1024, 2048, 4096
D = 32, 64, 128
causal = true/false
dtype = fp32 first, then fp16
```

Report:

```text
runtime ms
speedup over naive CUDA
peak memory allocated
max absolute error vs PyTorch
relative error
effective TFLOP/s
Nsight Compute metrics:
  - achieved occupancy
  - DRAM throughput
  - shared memory throughput
  - register count
  - global load/store efficiency
```

Your current repo already emphasizes benchmark sweeps, CSV output, plots, and Nsight profiling, so this will feel like a natural extension of your style. ([GitHub][1])

## What makes it “strong” rather than average

The strongest version would include three writeups:

```text
1. Why naive attention is memory-bound
2. Online softmax derivation
3. Nsight analysis: naive vs FlashAttention-style kernel
```

The writeup matters almost as much as the code. A recruiter or GPU engineer should be able to read it and see:

```text
This person understands CUDA.
This person understands attention.
This person understands memory hierarchy.
This person can benchmark honestly.
This person can explain performance trade-offs.
```

## My recommended scope

Do **not** start with backward pass. It will slow you down and may turn the project into a months-long rabbit hole.

Best scope:

```text
Must-have:
- Forward-only exact attention
- Naive CUDA baseline
- Online softmax explanation
- FlashAttention-style tiled CUDA kernel
- Causal mask
- PyTorch correctness comparison
- Benchmarks + Nsight profile

Nice-to-have:
- fp16 input
- PyTorch extension
- comparison with torch SDPA / official flash-attn
- head_dim 64 and 128 optimized paths

Skip initially:
- backward pass
- dropout
- variable-length packed sequences
- paged attention
- FlashAttention-2 full work partitioning
- Hopper-only FlashAttention-3 features
```

FlashAttention-3 uses Hopper-specific features like Tensor Memory Accelerator, warp specialization, WGMMA, and FP8 paths, so I would mention it in your README as “future work,” not implement it unless you have H100 access. ([arXiv][6])

## How I would present it on your portfolio

Use this title:

**FlashAttention from Scratch in CUDA: IO-Aware Exact Attention**

Use this one-liner:

> Implemented a FlashAttention-style forward kernel in CUDA using tiled Q/K/V streaming and online softmax, avoiding materialization of the full attention matrix and benchmarking against naive CUDA and PyTorch SDPA.

Portfolio bullets:

```text
- Built naive, online-softmax, and tiled FlashAttention-style CUDA kernels for scaled dot-product attention.
- Eliminated global-memory writes of the N x N attention matrix by streaming K/V tiles through shared memory.
- Added CPU/PyTorch correctness tests across causal/non-causal attention, variable sequence lengths, and head dimensions.
- Benchmarked runtime, memory usage, numerical error, and Nsight Compute metrics across N=128–4096 and D=32/64/128.
- Documented online softmax derivation, IO analysis, and optimization trade-offs.
```

## Final recommendation

Make this a **separate polished repo**, then add a link from your existing `gpu-parallel-patterns` repo as a “capstone application of the patterns.” Your existing project is broad; FlashAttention should be deep. Together, they tell a very strong story:

```text
gpu-parallel-patterns = I know the building blocks.
cuda-flash-attention-lab = I can apply them to modern LLM kernels.
```

[1]: https://github.com/vinayakgautam01/gpu-parallel-patterns/blob/main/README.md "gpu-parallel-patterns/README.md at main · vinayakgautam01/gpu-parallel-patterns · GitHub"
[2]: https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
[3]: https://arxiv.org/abs/2307.08691?utm_source=chatgpt.com "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
[4]: https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html?utm_source=chatgpt.com "torch.nn.functional.scaled_dot_product_attention — PyTorch 2.12 ..."
[5]: https://pypi.org/project/flash-attn/?utm_source=chatgpt.com "flash-attn · PyPI"
[6]: https://arxiv.org/abs/2407.08608?utm_source=chatgpt.com "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"

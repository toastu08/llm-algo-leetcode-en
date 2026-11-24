# 08. Programming Models CUDA Triton | 编程模型演进

**Difficulty:** Medium | **Environment:** GPU optional

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

**Tags:** `operator programming`, `CUDA`, `Triton` | **Audience:** learners starting with compilers and systems

---

## Section Guide

Moving from PyTorch expression to CUDA / Triton is not for the sake of being "lower level" itself, but because framework-level operator composition cannot always cling to the hardware execution granularity. As long as there are too many intermediate tensors, launches are too fragmented, or the data layout and tile organization cannot be controlled explicitly, the performance ceiling is quickly capped by memory access and scheduling overhead.

This section compares the control that the framework, CUDA, and Triton each have over execution granularity, data movement, and parallel organization, establishing a programming-model entry point for later operator optimization and kernel implementation. The learning focus is to judge which layer of control the existing abstraction lacks, rather than defaulting to descending into a custom kernel; the CUDA execution hierarchy is expanded in sections 15–16, and the Triton Program mapping is expanded in section 18.

**Keywords:** `grid`, `block`, `kernel`

![From tensor expressions to CUDA and Triton mappings](../public/01_Hardware_Math_and_Systems/08_programming_model_mapping.svg)

---
## Prerequisite Reading
**Overview:** First master tensor expression and the GPU resource model, then compare which control capabilities the framework, CUDA, and Triton each expose.
- [03. GPU Architecture and Memory](./03_GPU_Architecture_and_Memory.md)
- [Group 1D: Execution Models, Asynchrony and Operator Mapping](./1D.md)
## Q1: Why do we need to write custom operators in CUDA or Triton?


Native PyTorch operations are convenient, but by default they leave many execution details to the framework and the backend.

When a computation chain contains many small operators, repeated HBM round trips, or strong constraints on layout and tiles, the native approach often leaves more intermediate tensors and launch overhead. The value of custom CUDA or Triton is not "being lower level" itself, but realigning how the operators are organized with the hardware execution granularity.

So the key question in this kind of problem is not "should we hand-write a kernel," but "which data flows and execution flows must be controlled explicitly by us."


### Q1 mini check: Why native operations may be slow

First think about whether there are too many intermediate tensors.

```python
def memory_traffic(num_ops, tensor_bytes, fused=False):
    """Estimate the external read/write byte volume of a per-operator chain versus a fully fused chain."""
    if num_ops <= 0 or tensor_bytes <= 0:
        raise ValueError('num_ops and tensor_bytes must be positive')
    reads = num_ops if not fused else 1
    writes = num_ops if not fused else 1
    return {'reads': reads, 'writes': writes, 'traffic_bytes': (reads + writes) * tensor_bytes}

cases = [
    ('separate_ops', memory_traffic(3, 8_000_000, fused=False)),
    ('fused_ops', memory_traffic(3, 8_000_000, fused=True)),
    ('longer_chain', memory_traffic(5, 8_000_000, fused=False)),
]
for name, traffic in cases:
    print(name, '->', traffic)
assert cases[1][1]['traffic_bytes'] < cases[0][1]['traffic_bytes']


```

## Q2: How does the CUDA thread hierarchy correspond to the GPU hardware execution units?


The CUDA hierarchy is not syntactic decoration; it maps "how the task is split" onto "how the hardware is scheduled."

- grid determines how much independent work one kernel launch covers;
- block determines which group of threads can share data and synchronize;
- warp is the basic unit the hardware actually schedules;
- thread is the finest computation granularity.

After understanding this mapping, you can see why the same code can perform very differently with different thread counts, block counts, and warp utilization.


### Q2 mini check: How the hierarchy corresponds

Remember the execution hierarchy clearly, and reading kernels later will be smoother.

```python
def block_launch_stats(num_threads, warp_size=32):
    """Compute the warp count, idle lanes, and thread-slot utilization."""
    if num_threads <= 0 or warp_size <= 0:
        raise ValueError('num_threads and warp_size must be positive')
    warps = (num_threads + warp_size - 1) // warp_size
    active_lanes = num_threads
    wasted_lanes = warps * warp_size - num_threads
    occupancy = active_lanes / (warps * warp_size)
    return {'warps': warps, 'wasted_lanes': wasted_lanes, 'occupancy': round(occupancy, 2)}

for threads in [48, 64, 96, 128]:
    print(threads, 'threads ->', block_launch_stats(threads))
assert block_launch_stats(48)['wasted_lanes'] == 16
try:
    block_launch_stats(0)
except ValueError:
    pass
else:
    raise AssertionError('空 Block 必须报错')


```

## Q3: How do cross-thread sharing and synchronization mechanisms constrain and improve performance?


What cross-thread sharing really solves is "whether the same piece of data can be reused on-chip."

If the data is read only once, the overhead of shared memory and synchronization may not be worth it; if the same piece of data is used repeatedly by multiple threads inside a block, putting it in shared memory first and then synchronizing usually buys back more savings in HBM accesses.

So here it is not "the more sharing, the better," but "whether the reuse gain is enough to cover the synchronization cost."


### Q3 mini check: Why sharing is bound to synchronization

The more reuse there is, the more important collaboration becomes.

```python
def shared_tradeoff(reuse_times, sync_points, tile_bytes, bandwidth_bytes_per_us, sync_cost_us):
    """Convert the reduced HBM bytes into time and compare it with the synchronization time."""
    if reuse_times <= 0 or sync_points < 0 or tile_bytes <= 0 or bandwidth_bytes_per_us <= 0 or sync_cost_us < 0:
        raise ValueError('invalid reuse or timing inputs')
    saved_hbm_bytes = max(reuse_times - 1, 0) * tile_bytes
    saved_hbm_us = saved_hbm_bytes / bandwidth_bytes_per_us
    synchronization_us = sync_points * sync_cost_us
    return {'saved_hbm_bytes': saved_hbm_bytes, 'saved_hbm_us': saved_hbm_us, 'synchronization_us': synchronization_us, 'net_saved_us': saved_hbm_us - synchronization_us, 'worth_it': saved_hbm_us > synchronization_us}

for reuse_times, sync_points in [(1, 0), (2, 1), (4, 1), (4, 3)]:
    print((reuse_times, sync_points), '->', shared_tradeoff(reuse_times, sync_points, 64_000, 16_000, 2.0))
assert shared_tradeoff(1, 0, 64_000, 16_000, 2.0)['worth_it'] is False
assert shared_tradeoff(4, 1, 64_000, 16_000, 2.0)['worth_it'] is True


```

## Q4: Why can Triton lower the barrier to operator development?


The focus of Triton's abstraction is not "making things automatically faster for you," but hiding boilerplate details such as tiles, blocks, and layouts so that you can first write the tensor partitioning clearly.

Compared with CUDA, it does not remove the low-level constraints, but condenses them into fewer explicit concepts: layout, mask, tile, and program relationships.

So Triton's barrier is lower because it reduces boilerplate code; but its performance boundary still depends on your understanding of execution granularity and data layout.


### Q4 mini check: What Triton abstracts

First look at block organization, then at operator details.

```python
def triton_candidate(num_elements, block_size, layout_contiguous, dtype_supported, max_padding_ratio=0.25):
    """Check whether the layout, dtype, and boundary padding meet the teaching candidate threshold."""
    if num_elements <= 0 or block_size <= 0 or not 0 <= max_padding_ratio < 1:
        raise ValueError('invalid shape or padding threshold')
    programs = (num_elements + block_size - 1) // block_size
    padding_ratio = (programs * block_size - num_elements) / (programs * block_size)
    reasons = []
    if not layout_contiguous:
        reasons.append('non_contiguous_layout')
    if not dtype_supported:
        reasons.append('unsupported_dtype')
    if padding_ratio > max_padding_ratio:
        reasons.append('padding_ratio')
    return {'ready': not reasons, 'programs': programs, 'padding_ratio': padding_ratio, 'reasons': reasons}

ready = triton_candidate(1000, 128, True, True)
blocked = triton_candidate(1000, 128, False, True)
assert ready['ready'] is True
assert blocked['reasons'] == ['non_contiguous_layout']
print(ready)
print(blocked)


```

## Programming Models and Execution Paths

- Custom operators are not about showing off, but about controlling the execution path better.
- CUDA and Triton are different paradigms, but both aim to get closer to the hardware.
- Sharing and synchronization must be examined together; you cannot discuss only one.
- The core of a programming model is the mapping, not the syntax itself.

## Related Reading
**Overview:** If you want to complete the picture of programming models and kernel organization, you can continue with these pages.
- [Part 03: Triton Kernel Development](../03_Triton_Kernels/intro.md)
- [01. Triton Hello World: Vector Addition](../03_Triton_Kernels/01_Triton_Vector_Addition.md)
- [05. Triton GEMM and Autotune](../03_Triton_Kernels/05_Triton_GEMM_Tutorial.md)
---
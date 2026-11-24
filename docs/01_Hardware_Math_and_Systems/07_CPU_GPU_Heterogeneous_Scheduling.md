# 07. CPU GPU Heterogeneous Scheduling | CPU 与 GPU 异构调度

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `systems engineering`, `CPU/GPU`, `heterogeneous scheduling` | **Audience:** learners starting with system performance

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/07_CPU_GPU_Heterogeneous_Scheduling.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

The host suits control flow and data preparation, while the GPU suits large-scale parallel computation; migrating a stage to the GPU also incurs input/output transfer and synchronization costs. Therefore, faster device computation does not mean a faster end-to-end path.

This section establishes the relationship "stage placement → transfer cost → overlappable interval → end-to-end gain." When you finish, you should be able to compare the total time of a stage left on the CPU versus migrated to the GPU, and record capacity gains, transfer cost, and synchronization waiting separately; the execution semantics of Streams are expanded in section 17.

**Keywords:** `host`, `device`, `PCIe`

![Stage placement and overlap between CPU and GPU](../public/01_Hardware_Math_and_Systems/07_heterogeneous_stage_overlap.svg)

---
## Prerequisite Reading
**Overview:** First understand device resources and performance records, then compare the end-to-end cost of a task left on the host versus migrated to the device.
- [03. GPU Architecture and Memory](./03_GPU_Architecture_and_Memory.md)
- [Part 00 · 17 PyTorch Profiling Basics](../00_Prerequisites/17_PyTorch_Profiling_Basics.md)
## Q1: What roles do Host and Device each play?


The Host is usually responsible for control flow, data preparation, and scheduling, while the Device is responsible for large-scale parallel computation.

The core bottleneck between the two is often not "which one is stronger," but the cost and frequency of moving data between the two sides.

So the first thing in heterogeneous scheduling is to know which stage is better left on the CPU and which stage should be offloaded to the GPU.


### Q1 mechanism calculator: Compare the end-to-end time of a stage on the CPU or the GPU

Fix the input/output byte count of the same stage, and assume the input transfer in and the output transfer back happen serially with the same effective bandwidth in both directions. Bandwidth uses decimal GB/s (billion bytes per second), time uses milliseconds; the fixed overhead is the sum of the extra overhead of this round trip. Compare the CPU time with the "GPU compute + transfer" estimate; you cannot take a faster standalone GPU computation as an end-to-end conclusion.



```python
import math


def stage_placement(cpu_ms, gpu_ms, input_bytes, output_bytes,
                    bandwidth_gb_per_s, fixed_transfer_ms=0.0):
    """Compute a serial round-trip transfer in GB/s; performs no device computation or bandwidth measurement."""
    values = (cpu_ms, gpu_ms, input_bytes, output_bytes, fixed_transfer_ms)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('时间和字节数必须是有限非负数')
    if not math.isfinite(bandwidth_gb_per_s) or bandwidth_gb_per_s <= 0:
        raise ValueError('带宽必须是有限正数，单位 GB/s')
    transfer_ms = (
        (input_bytes + output_bytes) / (bandwidth_gb_per_s * 1e9) * 1e3
        + fixed_transfer_ms
    )
    gpu_path_ms = gpu_ms + transfer_ms
    return {
        'cpu_path_ms': cpu_ms, 'gpu_compute_ms': gpu_ms,
        'transfer_ms': transfer_ms, 'gpu_path_ms': gpu_path_ms,
        'shorter_estimated_path': 'gpu' if gpu_path_ms < cpu_ms else
                                  'cpu' if cpu_ms < gpu_path_ms else 'equal',
    }


BANDWIDTH_GB_PER_S = 16.0  # Modifiable: assumed effective byte bandwidth, not a measured value.
small_stage = stage_placement(2.0, 0.4, 64e6, 64e6, BANDWIDTH_GB_PER_S, 0.1)
large_stage = stage_placement(100.0, 8.0, 64e6, 64e6, BANDWIDTH_GB_PER_S, 0.1)
assert math.isclose(small_stage['transfer_ms'], 8.1)
assert small_stage['shorter_estimated_path'] == 'cpu'
assert large_stage['shorter_estimated_path'] == 'gpu'
print('短阶段：', small_stage)
print('长阶段：', large_stage)


```

## Q2: Which conditions allow transfer to overlap with computation?


Overlap requires that there be no unmet data dependency between transfer and computation, and that the runtime and hardware provide concurrently usable execution resources. Ideally, part of the data transfer can happen at the same time as the computation on another part of the data; the exposed time can at best be reduced to the duration of the longer stage, and the transfer cost does not disappear.

This section only computes the time difference between serial execution and ideal overlap. How Streams, Events, and synchronization boundaries express these dependencies is expanded in section 17.


### Q2 comparison check: How much transfer time overlap can hide

Assume the computation must first wait for a prefix of the transfer, the rest of the transfer is independent of the computation, and the resources allow parallelism. `dependency_ms` is the duration of that prefix and cannot exceed the total transfer time; if the computation must wait for the entire input, set it to the full transfer time. Only this dependency changes between the comparisons.



```python
def overlap_timeline(copy_ms, compute_ms, dependency_ms=0.0):
    """Compute the serial time, the dependency-constrained overlapped time, and the hidden transfer time."""
    if (any(not math.isfinite(value) or value < 0
            for value in (copy_ms, compute_ms, dependency_ms))
            or dependency_ms > copy_ms):
        raise ValueError('invalid timeline')
    serial_ms = copy_ms + compute_ms
    overlappable_copy_ms = copy_ms - dependency_ms
    elapsed_ms = dependency_ms + max(overlappable_copy_ms, compute_ms)
    hidden_ms = serial_ms - elapsed_ms
    return {'serial_ms': serial_ms, 'elapsed_ms': elapsed_ms, 'hidden_ms': hidden_ms}

full_overlap = overlap_timeline(8.0, 20.0)
dependency_limited = overlap_timeline(8.0, 20.0, dependency_ms=5.0)
assert full_overlap['hidden_ms'] == 8.0
assert dependency_limited['hidden_ms'] == 3.0
assert overlap_timeline(8.0, 20.0, dependency_ms=8.0)['hidden_ms'] == 0
print(full_overlap)
print(dependency_limited)
```

## Q3: How is CPU Offload used in training and inference respectively?


The goal of CPU Offload is to move some state that does not need to stay resident on the GPU to the CPU, relieving memory pressure.

In training, the common objects may be optimizer state, some parameters, or activations; in inference, they may be tiered weight residency or temporary state management.

Its core cost is more frequent data transfer, so it usually has to be examined together with the scheduling policy, not only by "how much memory was saved."


### Q3 mechanism calculator: Record the offloaded state size and the transfer time separately

The state size uses decimal GB, bandwidth uses GB/s, transfer time uses ms/step; one round trip counts twice the state size. The assumed hidden ratio is only used to estimate the exposed time. The size of the offloaded state is not equal to the real peak reduction: you also need to check whether offloading and reloading cross the original peak, and whether they add transfer buffers.



```python
def offload_ledger(state_gb, bandwidth_gb_per_s, round_trips_per_step=1,
                   overlap_ratio=0.0):
    """Record the offload volume, the round-trip byte volume, and the exposed time under the assumed hidden ratio separately."""
    if not math.isfinite(state_gb) or state_gb < 0:
        raise ValueError('状态大小必须是有限非负数，单位 GB')
    if not math.isfinite(bandwidth_gb_per_s) or bandwidth_gb_per_s <= 0:
        raise ValueError('带宽必须是有限正数，单位 GB/s')
    if type(round_trips_per_step) is not int or round_trips_per_step < 0:
        raise ValueError('每步往返次数必须是非负整数')
    if not math.isfinite(overlap_ratio) or not 0 <= overlap_ratio <= 1:
        raise ValueError('隐藏比例必须位于 [0, 1]')
    transfer_gb = state_gb * 2 * round_trips_per_step
    transfer_ms = transfer_gb / bandwidth_gb_per_s * 1e3
    return {
        'offloaded_state_gb': state_gb,
        'transfer_gb_per_step': transfer_gb,
        'transfer_ms': transfer_ms,
        'estimated_exposed_ms': transfer_ms * (1 - overlap_ratio),
    }


ledger = offload_ledger(4.0, bandwidth_gb_per_s=32.0, overlap_ratio=0.75)
assert ledger['transfer_gb_per_step'] == 8.0
assert ledger['transfer_ms'] == 250.0
assert ledger['estimated_exposed_ms'] == 62.5
print(ledger)


```

## Q4: How much does the end-to-end time drop once the timelines overlap?

Keep the workload and the result consistent, compare the serial time with the candidate completion time, and then check whether synchronization waiting is still exposed on the critical path. The share of the observation window occupied by the union of GPU active intervals is only a time-coverage ratio; it does not indicate the utilization of compute units, bandwidth, or instruction throughput, and the sum of the durations of several overlapping kernels also cannot be used as the active time here.


### Q4 comparison check: Record the time gain and the exposed waiting separately

Use constructed timelines to check the ratio and interval constraints, and do not set a universal adoption threshold for a single result. The GPU active interval does not overlap with the exposed synchronization waiting in this example, so the total cannot exceed the observation window.



```python
def parallelism_evidence(serial_ms, elapsed_ms, active_gpu_union_ms,
                         exposed_sync_ms):
    """Summarize the time comparison for the same workload; the active-time share is not hardware utilization."""
    values = (serial_ms, elapsed_ms, active_gpu_union_ms, exposed_sync_ms)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('时间必须是有限非负数')
    if serial_ms == 0 or elapsed_ms == 0:
        raise ValueError('串行与候选完成时间必须为正数')
    if active_gpu_union_ms + exposed_sync_ms > elapsed_ms:
        raise ValueError('互不重叠的活跃区间与暴露等待不能超过观察窗口')
    return {
        'speedup': serial_ms / elapsed_ms,
        'elapsed_saved_ms': serial_ms - elapsed_ms,
        'active_time_fraction': active_gpu_union_ms / elapsed_ms,
        'exposed_sync_fraction': exposed_sync_ms / elapsed_ms,
    }


shorter = parallelism_evidence(30.0, 20.0, 18.0, 1.0)
longer = parallelism_evidence(30.0, 32.0, 18.0, 8.0)
assert shorter['elapsed_saved_ms'] == 10.0
assert longer['elapsed_saved_ms'] == -2.0
print('时间缩短：', shorter)
print('时间增加：', longer)


```

## Check the transfer and overlap estimates

In real runtime, record the bidirectional byte count, the effective bandwidth, and the input readiness time, then use the execution trace to check which transfers are actually hidden by computation. Where offloading is involved, also check host memory, transfer buffers, and peak stages; capacity and time must be compared separately.


## Related Reading
**Overview:** If you want to complete the picture of how heterogeneous scheduling relates to launch / graph, you can continue with these pages.
- [Part 03: Triton Kernel Development](../03_Triton_Kernels/intro.md)
- [09. Triton Autotuning & Benchmarking](../03_Triton_Kernels/09_Triton_Autotune_and_Profiling.md)
- [11. CUDA Graph and JIT Compile](../04_CUDA_and_System_Optimization/11_CUDA_Graph_and_JIT_Compile.md)
---
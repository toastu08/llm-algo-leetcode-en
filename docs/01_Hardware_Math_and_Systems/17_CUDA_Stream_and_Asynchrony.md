# 17. CUDA Stream and Asynchrony | CUDA Stream 与异步执行

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `operator programming`, `CUDA Stream`, `asynchronous execution` | **Audience:** learners starting with compilers and systems

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

A throughput limit does not necessarily come from a single slow compute kernel; it may also come from copies, computation, and synchronization being arranged into an unnecessary serial chain. Starting from task dependencies, this section explains how Stream ordering, event synchronization, and asynchronous submission together determine the overlappable interval.

When you finish, you should be able to distinguish host submission order, device execution order, and real dependencies, and use a timeline to judge whether overlap occurs and whether synchronization is exposed on the critical path.

**Keywords:** CUDA Stream, asynchrony, overlap

![CUDA Stream, dependencies, and the overlap timeline](../public/01_Hardware_Math_and_Systems/17_stream_dependency_overlap.svg)

---
## Prerequisite Reading

**Overview:** First understand stage placement and the device execution order, then see how Streams and Events express asynchronous dependencies.

- [07. CPU and GPU Heterogeneous Scheduling](./07_CPU_GPU_Heterogeneous_Scheduling.md)
- [15. CUDA Execution Model](./15_CUDA_Execution_Model.md)
- [Group 1D: Execution Models, Asynchrony and Operator Mapping](./1D.md)


## Q1: When can tasks in different Streams overlap?

Tasks within the same Stream execute in order, while tasks in different Streams must also satisfy data dependencies and resource conditions. Only when the computation does not depend on the data currently being transferred, and the device can provide the corresponding resources at the same time, is there a chance to overlap. For example, transferring the input of the next batch can be independent of the computation of the current batch; the computation of the current batch, however, must wait for its own input to be ready.

The two independent stages below illustrate the ideal upper bound: ignoring contention and submission overhead, the overlapped part takes the smaller of the two durations and the total time takes the larger one; the trailing synchronization overhead is still added serially. If the computation depends on this transfer, use the sum of the two durations.


### Q1 mini check: Change only the data dependency and compare the completion time



```python
import math


def stream_timeline(copy_ms, compute_ms, sync_ms, independent=True):
    """Estimate the two-stage time; independent means the computation does not depend on this transfer."""
    values = (copy_ms, compute_ms, sync_ms)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('阶段时间必须为有限非负数')
    if type(independent) is not bool:
        raise ValueError('independent 必须是布尔值')
    serial_ms = copy_ms + compute_ms + sync_ms
    elapsed_ms = (max(copy_ms, compute_ms) if independent
                  else copy_ms + compute_ms) + sync_ms
    return {
        'serial_ms': serial_ms,
        'estimated_elapsed_ms': elapsed_ms,
        'hidden_ms': serial_ms - elapsed_ms,
        'speedup': serial_ms / elapsed_ms if elapsed_ms else 1.0,
    }


for independent in [True, False]:
    print(independent, '->', stream_timeline(10, 20, 4, independent))
assert stream_timeline(10, 20, 4)['hidden_ms'] == 10
assert stream_timeline(10, 20, 4, independent=False)['hidden_ms'] == 0
assert stream_timeline(0, 20, 4)['speedup'] == 1.0


```

## Q2: Once the asynchronous copy conditions hold, why can the gain still not be predicted directly?

For host-device transfers, pinned memory is one of the common conditions for asynchronous copies; to obtain overlap you also need independent data, a suitable queue, and parallelizable resources. If any condition is missing, the calculator does not count the transfer as hidden time.

The following takes these conditions as explicit inputs, first computing the ideal hideable time and then subtracting the assumed exposed event overhead. When the overhead is greater than the hidden time, the net saving is negative; these inputs should be verified from the actual timeline rather than guessed from the number of Streams.


### Q2 mini check: Remove the overlap conditions one by one



```python
def overlap_candidate(copy_ms, compute_ms, event_overhead_ms=0.0,
                      pinned_memory=True, independent=True,
                      concurrent_resources=True):
    """Estimate the overlap space under the given conditions; does not perform asynchronous copies."""
    values = (copy_ms, compute_ms, event_overhead_ms)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('时间必须为有限非负数')
    conditions = {
        'pinned_memory': pinned_memory,
        'independent': independent,
        'concurrent_resources': concurrent_resources,
    }
    if any(type(value) is not bool for value in conditions.values()):
        raise ValueError('重叠条件必须是布尔值')
    missing = [name for name, value in conditions.items() if not value]
    hidden = min(copy_ms, compute_ms) if not missing else 0.0
    return {
        'missing_conditions': missing,
        'ideal_hidden_ms': hidden,
        'event_overhead_ms': event_overhead_ms,
        'estimated_net_saved_ms': hidden - event_overhead_ms,
    }


print(overlap_candidate(10, 20, 1))
assert overlap_candidate(10, 20, 1)['estimated_net_saved_ms'] == 9
for condition in ['pinned_memory', 'independent', 'concurrent_resources']:
    report = overlap_candidate(10, 20, 1, **{condition: False})
    assert report['ideal_hidden_ms'] == 0
    assert report['missing_conditions'] == [condition]
assert overlap_candidate(1, 3, 3)['estimated_net_saved_ms'] == -2


```

## Q3: How do you distinguish computation, transfer, and waiting from the exposed time?

First divide the observation window into non-overlapping compute, exposed transfer, exposed synchronization, and idle intervals, then compare the time of each category. The largest category is only a clue for the next check: a long compute time still requires checking the compute volume and efficiency, while a long idle time requires tracing the input, dependencies, or host submission further.

Only interval lengths that already have overlap removed are accepted below. You cannot add up the complete event durations of multiple Streams directly, nor take the longest category directly as a confirmed bottleneck.


### Q3 mini check: Compute the shares from non-overlapping times



```python
def classify_timeline(kernel_ms, exposed_copy_ms, exposed_sync_ms, idle_ms):
    """Summarize non-overlapping exposed intervals; the longest category is not a confirmed bottleneck."""
    parts = {
        'kernel': kernel_ms,
        'copy': exposed_copy_ms,
        'sync': exposed_sync_ms,
        'idle': idle_ms,
    }
    if any(not math.isfinite(value) or value < 0 for value in parts.values()):
        raise ValueError('时间线字段必须是有限非负数')
    total_ms = sum(parts.values())
    if total_ms == 0:
        raise ValueError('时间线总时间必须大于 0')
    dominant = max(parts, key=parts.get)
    return {
        'total_ms': total_ms,
        'dominant': dominant,
        'fractions': {name: round(value / total_ms, 3) for name, value in parts.items()},
    }

plans = {
    'kernel_longest': (18, 2, 1, 1),
    'sync_longest': (6, 2, 12, 1),
    'copy_longest': (5, 14, 2, 1),
}
for name, timeline in plans.items():
    print(name, '->', classify_timeline(*timeline))

assert classify_timeline(*plans['sync_longest'])['dominant'] == 'sync'


```

## Continue checking the overlap conditions

Connect the estimate to the real runtime: record the input readiness, task submission, and device completion times separately, and note the resources used. A complete dependency chain still has to wait, and whether independent tasks really overlap must be judged from the execution trace; the actual gain is verified by repeated end-to-end timing under fixed inputs.


## Related Reading

- [29 CUDA Stream Advanced Scheduling](./29_CUDA_Stream_Advanced_Scheduling.md): compute timelines jointly constrained by explicit dependencies and queue order.
- [13 Profiling and Bottleneck Analysis](./13_Profiling_and_Bottleneck_Analysis.md): distinguish resource ceilings, inefficiency, and exposed waiting.
- [Performance Analysis: Timeline, I/O, and Critical Path](../topic_discussion/performance_analysis/03_time_breakdown_and_trace_reading.md): check waiting and overlap in the actual execution trace.


# 31. GPU Virtualization and MIG | GPU 虚拟化与 MIG

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `hardware systems`, `GPU virtualization`, `MIG` | **Audience:** learners of hardware constraints

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/31_GPU_Virtualization_and_MIG.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

When the same model has exclusive use of the device, latency is stable; once it shares the device with other tasks, tail latency may rise. Carving out independent resources for a task can reduce some contention, but may also leave one partition idle while another lacks capacity, and lower overall throughput.

This section first computes the resources per slice and the unallocated remainder using ideal equal partitioning, then compares throughput, P99, and fluctuation using constructed service metrics. After completing it, you should be able to point out both the improvements and the losses brought by isolation, and then check the resource boundaries and service behavior on real devices in the performance analysis and deployment topics.

**Keywords:** `MIG`, `MPS`, multi-tenancy

![GPU sharing, MIG slicing, and service evidence](../public/01_Hardware_Math_and_Systems/31_gpu_isolation_mig.svg)

---
## Prerequisite Reading

**Overview:** This page deals only with the multi-tenant scenario; first confirm that the candidate already satisfies the basic runtime constraints.

- [34. Heterogeneous Accelerators and Runtime Constraints](./34_Heterogeneous_Accelerators_and_Runtime_Constraints.md)

## Q1: Why is GPU virtualization first a resource isolation problem?


In a multi-tenant scenario, the GPU is not just about "who gets to use it," but "how to let many tasks share the same card without interfering with each other."

Without isolation, one task may seize too much compute, cache, VRAM, or scheduling window, causing other tasks' latency to spike and tail latency to degrade.

So the first goal of GPU virtualization is not simply to increase utilization, but to first define the interference boundaries of shared resources clearly so that performance is more controllable.

### Q1 Resource Calculation: How much integer resource ideal equal partitioning leaves

Assume memory can be divided evenly and continuously, compute units can only be allocated as integers, and each slice receives at least one unit. This does not generate MIG configurations, nor does it mean any arbitrary partitioning in the example is supported by real hardware; GB uses decimal units.


```python
import math


def partition_resources(total_memory_gb, total_compute_units, slices, reserved_memory_gb=0.0):
    """Compute the resources per slice and the integer compute-unit fragmentation after equal partitioning."""
    if (not math.isfinite(total_memory_gb) or total_memory_gb <= 0
            or type(total_compute_units) is not int or total_compute_units <= 0
            or type(slices) is not int or not 0 < slices <= total_compute_units):
        raise ValueError('resource totals and slices must be positive')
    if not math.isfinite(reserved_memory_gb) or not 0 <= reserved_memory_gb < total_memory_gb:
        raise ValueError('reserved memory must be in [0, total_memory)')
    usable_memory_gb = total_memory_gb - reserved_memory_gb
    compute_units_per_slice = total_compute_units // slices
    return {
        'memory_gb_per_slice': usable_memory_gb / slices,
        'compute_units_per_slice': compute_units_per_slice,
        'unused_compute_units': total_compute_units - compute_units_per_slice * slices,
    }

for slices in [2, 3, 7]:
    print(slices, 'slices ->', partition_resources(80, 108, slices, reserved_memory_gb=2))
assert partition_resources(80, 108, 7, 2)['unused_compute_units'] == 3

```

## Q2: What boundaries is MIG essentially slicing?


The core of MIG is not simply "splitting one GPU into several," but slicing apart compute, VRAM, and some scheduling boundaries so that different tenants run in more stable resource slices.

The direct benefit is less jitter caused by contention for cache and compute; the indirect benefit is that inference services can more easily do capacity planning and QoS control.

### Q2 Contrast Calculation: Throughput and tail-latency changes in constructed metrics

Assume the two records come from the same request set and timing method, and compare throughput and P99. The example does not measure the isolation effect on this machine; it also checks the percentile order to avoid bringing invalid records with P99 less than P50 into the judgment.


```python
def isolation_delta(shared, isolated):
    """Compare throughput, P99, and tail-latency amplification of the shared and isolated plans."""
    for record in (shared, isolated):
        if any(not math.isfinite(record[key]) or record[key] <= 0
               for key in ('throughput_rps', 'p50_ms', 'p99_ms')):
            raise ValueError('指标必须是有限正数')
        if record['p99_ms'] < record['p50_ms']:
            raise ValueError('P99 不能小于 P50')
    return {
        'throughput_change_ratio': isolated['throughput_rps'] / shared['throughput_rps'] - 1,
        'p99_reduction_ratio': 1 - isolated['p99_ms'] / shared['p99_ms'],
        'shared_tail_amplification': shared['p99_ms'] / shared['p50_ms'],
        'isolated_tail_amplification': isolated['p99_ms'] / isolated['p50_ms'],
    }

shared = {'throughput_rps': 120, 'p50_ms': 30, 'p99_ms': 150}
isolated = {'throughput_rps': 108, 'p50_ms': 32, 'p99_ms': 80}
delta = isolation_delta(shared, isolated)
assert delta['p99_reduction_ratio'] > 0
assert delta['throughput_change_ratio'] < 0
print(delta)

```

## Q3: Why is stronger isolation not necessarily higher throughput?


The stronger the isolation, the less tasks interfere with each other, but the efficiency of sharing usually decreases; the more aggressive the sharing, the higher the utilization may be, but latency and jitter are also more likely to deteriorate.

So the core of the virtualization problem is not to pursue "finer slicing" itself, but to find an acceptable trade-off among isolation, throughput, and tail latency.

### Q3 Threshold Screening: Whether the given metrics can satisfy the requirements simultaneously

The following metrics are constructed data; `jitter_ms` is defined as the difference between the maximum and minimum P99 across multiple runs and must be compared under the same repetition count and workload. The thresholds come from the goal of this example, not a general pass line.


```python
def select_isolation_plan(plans, max_p99_ms, min_throughput_rps, max_jitter_ms):
    """Apply the latency, throughput, and jitter hard thresholds first, then choose the plan with the highest throughput."""
    if any(not math.isfinite(value) or value < 0
           for value in (max_p99_ms, min_throughput_rps, max_jitter_ms)):
        raise ValueError('门槛必须是有限非负数')
    eligible = {}
    rejected = {}
    for name, metrics in plans.items():
        if any(not math.isfinite(metrics[key]) or metrics[key] < 0
               for key in ('p99_ms', 'throughput_rps', 'jitter_ms')):
            raise ValueError('候选指标必须是有限非负数')
        if metrics['p99_ms'] == 0:
            raise ValueError('P99 必须大于零')
        reasons = []
        if metrics['p99_ms'] > max_p99_ms:
            reasons.append('p99_slo')
        if metrics['throughput_rps'] < min_throughput_rps:
            reasons.append('throughput')
        if metrics['jitter_ms'] > max_jitter_ms:
            reasons.append('jitter')
        (rejected if reasons else eligible)[name] = reasons if reasons else metrics
    selected = max(eligible, key=lambda name: eligible[name]['throughput_rps']) if eligible else None
    return {'selected': selected, 'eligible': list(eligible), 'rejected': rejected}

plans = {
    'shared': {'throughput_rps': 120, 'p99_ms': 150, 'jitter_ms': 35},
    'isolated_a': {'throughput_rps': 108, 'p99_ms': 80, 'jitter_ms': 12},
    'isolated_b': {'throughput_rps': 70, 'p99_ms': 55, 'jitter_ms': 8},
}
decision = select_isolation_plan(plans, max_p99_ms=100, min_throughput_rps=90, max_jitter_ms=20)
assert decision['selected'] == 'isolated_a'
assert set(decision['rejected']) == {'shared', 'isolated_b'}
print(decision)

```

## Related Reading

**Overview:** The isolation plan needs further verification of service performance and long-term cost.

- [33. TCO and Cost Model](./33_TCO_and_Cost_Model.md)
- [Performance Analysis topic](../topic_discussion/performance_analysis/intro.md)
- [Deployment and Heterogeneous Systems topic](../topic_discussion/deployment_heterogeneous/intro.md)
---
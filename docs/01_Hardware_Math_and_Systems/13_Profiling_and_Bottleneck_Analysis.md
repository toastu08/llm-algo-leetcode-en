# 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `system performance`, `profiling`, `bottleneck analysis` | **Audience:** learners who need to locate training or inference performance bottlenecks

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

The compute volume, access bytes, stage times, data transfers, synchronization waiting, and launch overhead in a performance record must first be placed back into the same fixed workload before they can be used to explain a bottleneck.

This section establishes the attribution chain "raw record → derived metrics → candidate bottleneck → discriminating experiment". After completing it, you should be able to compare hypotheses about compute, memory access, and exposed waiting, and select one concrete variable for the next discriminating experiment. This section mainly supports performance analysis; after confirming a local bottleneck, move on to the actual comparisons in operator optimization or performance optimization.



**Keywords:** profiling, arithmetic intensity, machine balance point, bottleneck attribution

![Concept relationships in this section](../public/01_Hardware_Math_and_Systems/13_profiling_evidence_map.svg)

---

## Prerequisite Reading

**Overview:** First establish the resource model of GPU compute and memory access, and obtain stage times and trace records under a fixed workload from Part 00.

- [Part 01 · 03 GPU Architecture and Memory](./03_GPU_Architecture_and_Memory.md)
- [Part 00 · 17 PyTorch Profiling Basics](../00_Prerequisites/17_PyTorch_Profiling_Basics.md)
- [Part 01 · 02 Parameter Count and FLOPs](./02_LLM_Params_and_FLOPs.md)
---
## Where does the evidence come from

Different tools provide raw evidence at different granularities. The calculators below use explicitly constructed values to practice the attribution rules. When reading real records, first check the workload, measurement interval, and data type; theoretical FLOPs, minimum transferred volume, and hardware counter measurements should be labeled separately.

| Evidence level | Typical source | Fields used in this section |
|:---|:---|:---|
| Stages and operators | `torch.profiler` | Stage time, operator time, call count |
| System timeline | Nsight Systems | Transfers, synchronization, launches, and exposed waiting |
| Kernel metrics | Nsight Compute | FLOPs, DRAM bytes, actual throughput, and stall reasons |

## Q1: How do you convert raw records into comparable resource metrics?

Before comparing, fix the input, data type, and measurement interval, then unify the units of floating-point operations, DRAM bytes, and kernel time. Arithmetic intensity represents how many floating-point operations are completed per byte moved; achieved compute throughput and achieved bandwidth represent how much work the current kernel delivers in a given time.

| Derived metric | Computation relationship | Question answered |
|:---|:---|:---|
| Arithmetic intensity | FLOPs / DRAM bytes | Is this computation more dependent on compute or on data movement? |
| Achieved compute throughput | FLOPs / kernel time | How many TFLOP/s are currently delivered? |
| Achieved bandwidth | DRAM bytes / kernel time | How many GB/s are currently consumed? |
| Exposed time share | A kind of waiting / total time | Are transfers, synchronization, or launches on the critical path? |

### Q1 Mini Check: Derive resource metrics from the same record

Construct a serial stage: kernel execution and the already deduplicated transfer, synchronization, and launch exposed times do not overlap, so they can be added. A real timeline with overlap requires the critical path to be computed from dependencies, and this denominator cannot be applied directly.

```python
import math


def derive_resource_metrics(record):
    """Convert kernel workload and exposed time into unified resource metrics."""
    required = {'flops', 'dram_bytes', 'kernel_ms', 'transfer_ms', 'sync_ms', 'launch_ms', 'time_model'}
    if set(record) != required:
        raise ValueError(f'record 必须且只能包含 {sorted(required)}')
    if record['time_model'] != 'nonoverlapping_exposed_intervals':
        raise ValueError('本计算器只接受互不重叠、已经去重的暴露区间')
    numeric = [value for key, value in record.items() if key != 'time_model']
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError('工作量和时间必须为有限值')
    if record['flops'] <= 0 or record['dram_bytes'] <= 0 or record['kernel_ms'] <= 0:
        raise ValueError('flops、dram_bytes 和 kernel_ms 必须大于 0')
    if any(record[name] < 0 for name in ('transfer_ms', 'sync_ms', 'launch_ms')):
        raise ValueError('暴露时间不能为负数')
    total_ms = sum(record[name] for name in ('kernel_ms', 'transfer_ms', 'sync_ms', 'launch_ms'))
    return {
        'arithmetic_intensity_flop_per_byte': record['flops'] / record['dram_bytes'],
        'achieved_tflops': record['flops'] / (record['kernel_ms'] / 1000) / 1e12,
        'achieved_dram_gbps': record['dram_bytes'] / (record['kernel_ms'] / 1000) / 1e9,
        'total_ms': total_ms,
        'exposed_share': {
            name: record[f'{name}_ms'] / total_ms
            for name in ('transfer', 'sync', 'launch')
        },
    }

profile_record = {
    'flops': 4.0e10, 'dram_bytes': 8.0e9, 'kernel_ms': 20.0,
    'transfer_ms': 3.0, 'sync_ms': 2.0, 'launch_ms': 1.0,
    'time_model': 'nonoverlapping_exposed_intervals',
}
resource_metrics = derive_resource_metrics(profile_record)
print(resource_metrics)
assert resource_metrics['arithmetic_intensity_flop_per_byte'] == 5.0
assert round(resource_metrics['achieved_tflops'], 1) == 2.0
assert round(resource_metrics['achieved_dram_gbps'], 1) == 400.0

```

## Q2: How do the machine balance point and exposed time form a bottleneck hypothesis?

The machine balance point equals the peak compute throughput divided by the peak bandwidth, also in FLOP/byte. When a kernel's arithmetic intensity is below this point, the Roofline upper bound is first determined by bandwidth; above this point, it is first determined by compute throughput. This is an upper-bound classification and cannot yet confirm the actual bottleneck. The current throughput should be compared with the reference upper bound under the same precision and similar workload: when far from the upper bound, first check occupancy, layout, launch granularity, and other execution limits. The 70% in the example below is only an explicitly declared teaching screening line, not a universal hardware criterion. The exposed waiting in system stages may also be longer than the kernel time.

| Evidence | Attributed object | What still needs to be ruled out |
|:---|:---|:---|
| Arithmetic intensity below the balance point, close to the reference bandwidth, with kernel time dominating | Memory-access path | Whether layout, cache hits, and actual bandwidth are abnormal |
| Arithmetic intensity not below the balance point, close to the reference compute upper bound, with kernel time dominating | Compute path | Whether Tensor Core, occupancy, and instruction efficiency meet the target |
| Far from both reference upper bounds | Kernel main constraint cannot yet be confirmed | Workload estimate, occupancy, layout, and launch granularity |
| H2D / D2H exposed time dominating | Data transfer | Whether input preparation, pinned memory, and asynchronous copy take effect |
| Synchronization or launch time dominating | Scheduling path | Dependencies, fragmented kernels, and CPU launch overhead |

### Q2 Mini Check: Compare the machine balance point with critical-path time

First compute the balance point and efficiency using the given reference upper bounds, then compare the non-overlapping stage times. Construct the cases of being far from the upper bound, close to the bandwidth upper bound, and transfer-time dominated, and observe how the hypothesis changes.

```python
def attribute_bottleneck(record, reference_tflops, reference_dram_gbps, efficiency_screen=0.70):
    """Screen hypotheses with reference upper bounds; low-efficiency records remain paths that need further checking."""
    if any(not math.isfinite(value) or value <= 0
           for value in (reference_tflops, reference_dram_gbps)):
        raise ValueError('参考上界必须是正的有限值')
    if not math.isfinite(efficiency_screen) or not 0 < efficiency_screen <= 1:
        raise ValueError('效率筛查线必须位于 (0, 1]')
    metrics = derive_resource_metrics(record)
    machine_balance = reference_tflops * 1000 / reference_dram_gbps
    roofline_limit = min(
        reference_tflops,
        metrics['arithmetic_intensity_flop_per_byte'] * reference_dram_gbps / 1000,
    )
    efficiency = metrics['achieved_tflops'] / roofline_limit
    bound = ('memory' if metrics['arithmetic_intensity_flop_per_byte'] < machine_balance
             else 'compute')
    kernel_hypothesis = f'kernel_{bound}' if efficiency >= efficiency_screen else 'kernel_underutilized'
    exposed_ms = {
        kernel_hypothesis: record['kernel_ms'], 'transfer': record['transfer_ms'],
        'synchronization': record['sync_ms'], 'launch': record['launch_ms'],
    }
    primary = max(exposed_ms, key=exposed_ms.get)
    if efficiency > 1.05:
        primary = 'recheck_reference'
    return {
        'machine_balance_flop_per_byte': machine_balance,
        'roofline_bound': bound, 'roofline_limit_tflops': roofline_limit,
        'reference_efficiency': efficiency, 'efficiency_screen': efficiency_screen,
        'reference_consistency': 'recheck_reference' if efficiency > 1.05 else 'within_screen',
        'exposed_ms': exposed_ms, 'primary_hypothesis': primary,
    }


underutilized_case = attribute_bottleneck(profile_record, 120, 3000)
memory_record = {
    **profile_record, 'kernel_ms': 3.0, 'transfer_ms': 0.3, 'sync_ms': 0.2, 'launch_ms': 0.1
}
memory_case = attribute_bottleneck(memory_record, 120, 3000)
transfer_record = {**profile_record, 'kernel_ms': 6.0, 'transfer_ms': 14.0}
transfer_case = attribute_bottleneck(transfer_record, 120, 3000)
print('below reference:', underutilized_case)
print('near bandwidth reference:', memory_case)
print('exposed transfer:', transfer_case)
assert underutilized_case['primary_hypothesis'] == 'kernel_underutilized'
assert round(underutilized_case['reference_efficiency'], 3) == 0.133
assert memory_case['roofline_bound'] == 'memory'
assert memory_case['primary_hypothesis'] == 'kernel_memory'
assert transfer_case['primary_hypothesis'] == 'transfer'

```

## Q3: How do you use a discriminating experiment to confirm or overturn a bottleneck hypothesis?

The attribution result determines the next experiment, not the final optimization plan directly. A discriminating experiment must keep the input and target problem unchanged, change only one explicitly selected implementation condition, and record both the target metric and the cost metric. If the target metric does not change as expected, return to the timeline and revise the hypothesis.

| Hypothesis | Single change | Main observation | Falsification condition |
|:---|:---|:---|:---|
| Memory-access hypothesis | Change only the tile size | DRAM bytes, kernel time | Bytes decrease but time does not change significantly |
| Compute hypothesis | Change only the compute kernel implementation | TFLOP/s, quality error | Compute throughput rises but end-to-end time is unchanged |
| Transfer hypothesis | Change only whether host memory is pinned | Exposed transfer time | The critical path does not shorten after transfer overlap |
| Synchronization or launch hypothesis | Change only one of the synchronization position or the graph capture switch | Launch count, synchronization waiting | Events decrease but stage time is unchanged |

### Q3 Mini Check: Generate a single-variable discriminating experiment for the attribution result

Convert the main hypothesis of Q2 into a changed variable, observation metrics, and a falsification condition, avoiding changing multiple strategies at once.

```python
def build_discrimination_experiment(attribution, single_change):
    """Select one concrete variable for a main hypothesis while keeping end-to-end and correctness observations."""
    plans = {
        'kernel_memory': {
            'allowed_changes': {'tile_size', 'layout'},
            'target_metrics': ['dram_bytes', 'kernel_ms', 'stage_ms', 'max_abs_error'],
            'revisit_when': 'bytes_down_but_kernel_ms_unchanged',
        },
        'kernel_compute': {
            'allowed_changes': {'compute_kernel'},
            'target_metrics': ['achieved_tflops', 'kernel_ms', 'stage_ms', 'max_abs_error'],
            'revisit_when': 'kernel_faster_but_stage_ms_unchanged',
        },
        'kernel_underutilized': {
            'allowed_changes': {'block_size'},
            'target_metrics': ['kernel_ms', 'occupancy', 'stage_ms', 'max_abs_error'],
            'revisit_when': 'occupancy_changes_but_kernel_ms_unchanged',
        },
        'transfer': {
            'allowed_changes': {'pin_memory', 'async_copy'},
            'target_metrics': ['exposed_transfer_ms', 'stage_ms', 'host_memory_mib', 'max_abs_error'],
            'revisit_when': 'transfer_overlap_up_but_stage_ms_unchanged',
        },
        'synchronization': {
            'allowed_changes': {'sync_placement'},
            'target_metrics': ['sync_ms', 'stage_ms', 'max_abs_error'],
            'revisit_when': 'sync_ms_down_but_stage_ms_unchanged',
        },
        'launch': {
            'allowed_changes': {'graph_capture'},
            'target_metrics': ['launch_count', 'launch_ms', 'stage_ms', 'max_abs_error'],
            'revisit_when': 'launches_down_but_stage_ms_unchanged',
        },
    }
    hypothesis = attribution['primary_hypothesis']
    if hypothesis not in plans:
        raise ValueError(f'不支持的瓶颈假设: {hypothesis}')
    selected = plans[hypothesis]
    if single_change not in selected['allowed_changes']:
        raise ValueError('必须从当前假设中选择一个具体变化变量')
    return {
        'hypothesis': hypothesis, 'single_change': single_change,
        'target_metrics': selected['target_metrics'], 'revisit_when': selected['revisit_when'],
    }


assert attribute_bottleneck(profile_record, 1, 100)['primary_hypothesis'] == 'recheck_reference'

memory_plan = build_discrimination_experiment(memory_case, 'tile_size')
transfer_plan = build_discrimination_experiment(transfer_case, 'pin_memory')
print('memory plan:', memory_plan)
print('transfer plan:', transfer_plan)
assert memory_plan['single_change'] == 'tile_size'
assert 'stage_ms' in memory_plan['target_metrics']
assert transfer_plan['single_change'] == 'pin_memory'
assert 'host_memory_mib' in transfer_plan['target_metrics']
try:
    build_discrimination_experiment(memory_case, 'tile_or_layout')
except ValueError:
    pass
else:
    raise AssertionError('不能把多个备选项当成一个已选变量')
try:
    derive_resource_metrics({**profile_record, 'time_model': 'overlapping_intervals'})
except ValueError:
    pass
else:
    raise AssertionError('重叠时间线不能直接相加')

```

---
## Related Reading
Continue from resource attribution into training, inference, or end-to-end projects, and verify the discriminating experiments on a fixed workload.
- [PyTorch Profiler official documentation](https://pytorch.org/docs/stable/profiler.html)
- [Part 02 · 73 Training Performance Analysis](../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md)
- [Part 02 · 74 Profiling-Driven End-to-End Optimization](../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md)
---
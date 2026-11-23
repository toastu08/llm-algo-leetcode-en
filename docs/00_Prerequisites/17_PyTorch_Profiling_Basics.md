# 17. PyTorch Profiling Basics | PyTorch 性能分析基础

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `PyTorch`, `profiling`, `performance analysis` | **Audience:** learners preparing for Parts 2–4

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/17_PyTorch_Profiling_Basics.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## Section Guide

After confirming that the run results are valid, if a run becomes slow, you need to observe the operator summary, the stage timeline, and the distribution of repeated measurements at the same time. A single total elapsed time cannot show which stage consumes the time, nor distinguish a stable hotspot from an occasional fluctuation.

After completing this section, you should be able to build time evidence with units, distinguish the purpose of the summary table from that of the trace record, and place local hotspots back into the complete training step. First use the hotspot to choose the next check, then use a controlled experiment to test the cause.

**Keywords:** `profiler`, `trace`, `latency`

![Profiling time evidence diagram](../public/00_Prerequisites/17_profiling_evidence_map.svg)

## Prerequisite Reading
**Overview:** First learn the diagnostic order from the 0E group page, confirm that the run is valid with section 19, and then break a run into comparable time evidence.
- [0E Group Page](./0E.md)
- [Part 00 · 19 Debugging and Anomaly Localization](./19_Debugging_and_Anomaly_Localization.md)

## Q1: How do you build a comparable performance record?

Assume you already have a fixed model and input that pass the correctness check. When the waiting time of a run becomes longer, look at latency; when the samples or tokens processed per unit time decrease, look at throughput. The first run may also include initialization cost, so warm up first, then repeat the measurement, and finally put the time results and profiler hotspots into the same record.

To make two results comparable, at least fix the model, input shape, dtype, batch, and thread settings; first discard the warm-up rounds, then keep all repeated samples, and finally report the median and P95. The median describes a typical run, and P95 exposes slow samples; a single run or reporting only the mean is not enough to describe the distribution.

| Record item | Purpose |
|:---|:---|
| latency samples | Keep the time of each round so that summarized values do not hide jitter |
| median / P95 | Describe typical latency and slow-sample latency respectively |
| throughput | Determine how many samples or tokens are processed per unit time |
| profiler hotspot | Find the operator or stage where the time is concentrated, and choose the next check |
| environment record | Save the model, input, dtype, and versions to keep the result reviewable |


```python
import math
import platform
import statistics
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

# Change one input and rerun this section to compare the time record over the same scope.
SEED, WARMUP, REPEATS, BATCH_SIZE = 7, 2, 20, 32
torch.manual_seed(SEED)
model = nn.Sequential(nn.Linear(100, 200), nn.ReLU(), nn.Linear(200, 10)).eval()
inputs = torch.randn(BATCH_SIZE, 100)

with torch.no_grad():
    for _ in range(WARMUP):
        _ = model(inputs)
    latency_samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        _ = model(inputs)
        latency_samples.append((time.perf_counter() - start) * 1000)


def nearest_rank(values, fraction):
    """Compute a quantile by nearest rank; the median is computed separately with statistics.median."""
    if not values or not 0 <= fraction <= 1:
        raise ValueError('values 不能为空，fraction 必须位于 [0, 1]')
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('时间样本必须是非负有限值')
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


latency_median_ms = statistics.median(latency_samples)
latency_p95_ms = nearest_rank(latency_samples, 0.95)
throughput_samples_s = inputs.shape[0] / (latency_median_ms / 1000)
trace_dir = TemporaryDirectory()
trace_path = Path(trace_dir.name) / '17_profile_trace.json'
with profile(activities=[ProfilerActivity.CPU]) as prof:
    with torch.no_grad():
        output = model(inputs)

hotspot_table = prof.key_averages().table(sort_by='cpu_time_total', row_limit=5)
workload = {
    'model': 'Linear(100, 200) -> ReLU -> Linear(200, 10)',
    'scope': 'forward_only', 'model_mode': 'eval', 'grad_enabled': False,
    'input_shape': tuple(inputs.shape), 'dtype': str(inputs.dtype),
    'device': str(inputs.device), 'seed': SEED,
    'warmup': WARMUP, 'repeats': REPEATS,
    'measurement_method': 'cpu_perf_counter',
    'intraop_threads': torch.get_num_threads(),
    'interop_threads': torch.get_num_interop_threads(),
    'torch_version': torch.__version__, 'python_version': platform.python_version(),
}
summary = {
    'evidence_type': 'time', 'status': 'measured', 'activity_scope': 'cpu',
    'workload': workload,
    'latency_samples_ms': latency_samples,
    'latency_median_ms': latency_median_ms,
    'latency_p95_ms': latency_p95_ms,
    'throughput_samples_s': throughput_samples_s,
    'throughput_definition': 'batch_size / median_forward_seconds',
    'hotspot_table': hotspot_table,
}
print(summary)

```

## Q1 verification: Build a baseline with warm-up, repeated samples, and a latency distribution

Here it verifies that the summary keeps the raw samples and contains the median, P95, throughput, and hotspot table at the same time.


```python
assert summary['evidence_type'] == 'time' and summary['activity_scope'] == 'cpu'
assert summary['workload']['repeats'] == len(summary['latency_samples_ms']) == 20
assert summary['latency_median_ms'] > 0 and summary['throughput_samples_s'] > 0
assert summary['latency_p95_ms'] >= summary['latency_median_ms']
assert 'aten::linear' in summary['hotspot_table'] or 'aten::addmm' in summary['hotspot_table']
assert summary['workload']['scope'] == 'forward_only'
assert summary['workload']['grad_enabled'] is False
assert summary['workload']['intraop_threads'] > 0
assert nearest_rank([1, 2, 3, 100], 0.95) == 100
assert statistics.median([1, 2, 3, 100]) == 2.5
print('✅ CPU 前向基线、测量范围与热点记录通过')

```

## Q2: Why can host time not directly replace device time?

The host prepares inputs, launches operators, and waits when necessary, and the device executes kernels; data transfers and synchronization may also be interleaved between the two. CPU time can only describe host-side calls, scheduling, and waiting, and cannot replace CUDA kernel time. The current CPU-first exercise first verifies host activity recording; after entering GPU experiments, CUDA activity, transfers, and synchronization events must be collected at the same time under the same workload.

| Observation | What it represents | Common interpretation |
|:---|:---|:---|
| Host time | CPU-side calls, scheduling, and waiting | Data preparation, launch, or synchronization overhead may exist |
| Device time | GPU kernel execution intervals | Reflects device-side compute activity |
| Transfers and synchronization | Data transfer, event dependencies, and explicit waits | Explain the gaps between the host and device timelines |


```python
# The CPU-first page collects only host activity; this result cannot be extrapolated to CUDA kernel time.
with profile(activities=[ProfilerActivity.CPU]) as cpu_prof:
    with torch.no_grad():
        _ = model(inputs)
cpu_time_table = cpu_prof.key_averages().table(sort_by='cpu_time_total', row_limit=5)
print(cpu_time_table)

```

## Q2 verification: Check the CPU-side call time baseline

This verification checks the CPU-side profiling interface and the stage timing results. CUDA kernel, synchronization, and data transfer times should be collected with CUDA activity recording in the GPU experiments of Part 02.


```python
assert 'aten::linear' in cpu_time_table or 'aten::addmm' in cpu_time_table
assert len(cpu_time_table) > 0
print('✅ 主机活动记录通过；设备、搬运与同步证据需在 GPU 实验中采集')

```

## Q3: When do you need to move from the summary table to a trace?

The summary table is suitable for answering "which operator or stage has the highest cumulative time"; if you also need to know the order of stages, the waiting relationships, or the changes across iterations, you need a trace. A trace is an event record with timestamps; you can first check the event count and names, then hand it to Chrome Trace or TensorBoard Profiler to view the complete timeline.


```python
import json

with profile(activities=[ProfilerActivity.CPU]) as prof:
    for _ in range(3):
        with torch.no_grad():
            _ = model(inputs)
        prof.step()
prof.export_chrome_trace(str(trace_path))
trace_record = json.loads(trace_path.read_text(encoding='utf-8'))
trace_events = trace_record.get('traceEvents', [])
named_events = [event.get('name') for event in trace_events if event.get('name')]
print('trace path:', trace_path)
print('event count:', len(trace_events))
print('sample event names:', named_events[:8])

```

## Q3 verification: Check a trace file that can be reviewed later

The verification code first reads the trace event list, then cleans up the temporary file. When you need to inspect the complete timeline manually, you can change `trace_path` to an output directory you want to keep.


```python
assert trace_path.exists() and trace_path.stat().st_size > 0
assert len(trace_events) > 0 and len(named_events) > 0
print('✅ trace 导出与事件解析通过:', trace_path)
trace_dir.cleanup()
assert not trace_path.exists()
print('✅ 临时 trace 已清理')

```

## Q4: How do you place a local hotspot back into the complete training step?

If the total time of a training step grows longer, measuring only one Linear operator is not enough: the slow point may be in forward, loss, backward, or the optimizer step. Only by putting these stages into the same minimal training loop and marking them with labels can you place the local hotspot back into the complete training process. This example collects the stage record of one complete training step separately and does not reuse the forward latency and throughput of Q1. The labels are not nested; the denominator of the stage share is the sum of the times of these labels. It does not include unlabeled Python overhead, nor is it an end-to-end time from repeated measurements.


```python
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
targets = torch.randint(0, 10, (32,))
with profile(activities=[ProfilerActivity.CPU]) as prof:
    with torch.profiler.record_function('zero_grad'):
        optimizer.zero_grad()
    with torch.profiler.record_function('forward'):
        outputs = model(inputs)
    with torch.profiler.record_function('loss'):
        loss = criterion(outputs, targets)
    with torch.profiler.record_function('backward'):
        loss.backward()
    with torch.profiler.record_function('optimizer_step'):
        optimizer.step()
training_table = prof.key_averages().table(sort_by='cpu_time_total', row_limit=20)
stage_labels = ['zero_grad', 'forward', 'loss', 'backward', 'optimizer_step']
stage_events = [
    {'name': event.key, 'total_ms': event.cpu_time_total / 1000.0, 'calls': event.count}
    for event in prof.key_averages()
    if event.key in stage_labels
]
stage_total_ms = sum(event['total_ms'] for event in stage_events)
stage_hotspot = max(stage_events, key=lambda event: event['total_ms'])
stage_hotspot = {
    **stage_hotspot,
    'share_pct': round(stage_hotspot['total_ms'] / stage_total_ms * 100, 1),
}
time_evidence = {
    'evidence_type': 'time',
    'status': 'measured',
    'activity_scope': 'cpu',
    'workload': {
        **workload, 'scope': 'training_step', 'model_mode': 'train',
        'grad_enabled': True, 'optimizer': 'SGD', 'warmup': 0, 'repeats': 1,
        'measurement_method': 'cpu_profiler_non_nested_stages',
    },
    'stage_events': stage_events,
    'labeled_stage_sum_ms': stage_total_ms,
    'share_denominator': 'sum_of_non_nested_stage_times',
    'stage_labels': stage_labels,
    'hotspot': stage_hotspot,
    'interpretation_limit': 'hotspot_is_candidate_not_root_cause',
}
print(training_table)
print('time evidence:', time_evidence)

```

## Q4 verification: Check the training stage labels and the hotspot result

Here it reuses the result of the previous cell to confirm that all five training stages are labeled, the loss is finite, and the training-stage record does not mix in latency samples from the forward baseline.


```python
assert {event['name'] for event in stage_events} == set(stage_labels)
assert torch.isfinite(loss).item()
assert time_evidence['status'] == 'measured'
assert len(trace_events) > 0
assert time_evidence['workload']['scope'] != summary['workload']['scope']
assert 'latency_samples_ms' not in time_evidence
assert time_evidence['labeled_stage_sum_ms'] == sum(event['total_ms'] for event in stage_events)
assert time_evidence['hotspot']['name'] in stage_labels
assert 0 < time_evidence['hotspot']['share_pct'] <= 100
assert time_evidence['interpretation_limit'] == 'hotspot_is_candidate_not_root_cause'
print('✅ 训练阶段标签、时间证据与解释边界通过')

```

## Related Reading
**Overview:** After completing this section, you can continue with the memory ledger, the hardware bottleneck model, and real profiling projects.
- [Part 00 · 20 Performance Evidence and Experiment Design](./20_Performance_Evidence_and_Experiment_Design.md)
- [Part 01 · 13 Profiling and Bottleneck Analysis](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [PyTorch Profiler official tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

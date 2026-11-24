# 29. CUDA Stream Advanced Scheduling | CUDA Stream 高级调度

**Difficulty:** Hard | **Environment:** CPU-first | **Tags:** `operator programming`, `CUDA Stream`, `asynchronous scheduling` | **Audience:** learners starting with compilers and systems

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/29_CUDA_Stream_Advanced_Scheduling.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

Computation can only start after data is transferred in, and data can only be copied back after computation finishes; yet different batches may also contain independent work. Only by putting these dependencies together with the submission order within the same queue can you judge how much overlap space multi-queue scheduling leaves.

This section first computes start and end times under dependency constraints, then compares the single-batch dependency chain with a multi-batch pipeline, distinguishing what queue priority and Graph replay each change. The calculator runs on the CPU, and its results depend on the stage durations and the assumption that resources can run in parallel; real execution can continue into the performance analysis topic to check the timeline.

**Keywords:** `stream`, `event`, `graph`

![Scheduling relationship among Stream, Event, and CUDA Graph](../public/01_Hardware_Math_and_Systems/29_stream_event_graph_pipeline.svg)

---
## Prerequisite Reading

**Overview:** First master stage placement and the Stream basics, then use Events, priorities, and Graphs to organize more complex dependencies.

- [07. CPU and GPU Heterogeneous Scheduling](./07_CPU_GPU_Heterogeneous_Scheduling.md)
- [17. CUDA Stream and Asynchrony](./17_CUDA_Stream_and_Asynchrony.md)


## Q1: How do you convert task dependencies into a schedulable Stream timeline?

Tasks in the same Stream execute in the order of the given list; tasks across Streams must also wait for explicit predecessors. For each task, take the latest end time among all predecessors as the earliest start time, then add its own duration to get the end time. Advancing repeatedly over the tasks whose predecessors are already satisfied yields a timeline that satisfies the dependencies.

The following assumes that different Streams have independent execution resources, that task durations are fixed, and that there is no submission overhead or resource contention. The result is therefore an ideal scheduling estimate, not a measured CUDA time. If cross-queue dependencies are ignored and only the maximum cumulative duration of each queue is taken, work that still has to wait will be miscounted as overlappable work.


### Q1 mini check: Compute start and end times with dependencies

Keep the task durations unchanged and compare the completion time of independent tasks in different Streams against a serial dependency chain.



```python
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class TaskStage:
    name: str
    duration_us: float
    stream: str
    depends_on: Tuple[str, ...] = ()


def schedule_stages(stages: List[TaskStage]) -> Dict[str, dict]:
    """Compute the ideal timeline; same-queue order and cross-queue predecessors must both hold."""
    by_name = {}
    predecessors = {}
    previous_on_stream = {}
    for stage in stages:
        if not isinstance(stage.name, str) or not stage.name.strip():
            raise ValueError('任务名称必须是非空字符串')
        if stage.name in by_name:
            raise ValueError('任务名称不能重复')
        if not isinstance(stage.stream, str) or not stage.stream.strip():
            raise ValueError('队列名称必须是非空字符串')
        if not math.isfinite(stage.duration_us) or stage.duration_us < 0:
            raise ValueError('阶段耗时必须是有限非负数')
        if not isinstance(stage.depends_on, (tuple, list)) or any(
            not isinstance(name, str) or not name.strip()
            for name in stage.depends_on
        ):
            raise ValueError('前驱必须由任务名称组成')
        by_name[stage.name] = stage
        dependencies = set(stage.depends_on)
        if stage.stream in previous_on_stream:
            dependencies.add(previous_on_stream[stage.stream])
        predecessors[stage.name] = dependencies
        previous_on_stream[stage.stream] = stage.name
    if any(dep not in by_name for deps in predecessors.values() for dep in deps):
        raise ValueError('存在未知前驱')

    timeline = {}
    while len(timeline) < len(stages):
        progressed = False
        for name, stage in by_name.items():
            if name in timeline or not predecessors[name].issubset(timeline):
                continue
            start = max(
                (timeline[dep]['end_us'] for dep in predecessors[name]), default=0.0
            )
            timeline[name] = {
                'start_us': start, 'end_us': start + stage.duration_us,
                'stream': stage.stream,
            }
            progressed = True
        if not progressed:
            raise ValueError('显式依赖与同队列顺序形成了环')
    return timeline


def sequential_time_us(stages: List[TaskStage]) -> float:
    """Compute the total time when all tasks of a valid task set execute serially."""
    schedule_stages(stages)
    return sum(stage.duration_us for stage in stages)


def pipelined_time_us(stages: List[TaskStage]) -> float:
    """Take the latest completion time that satisfies all dependencies, without ignoring cross-queue waiting."""
    timeline = schedule_stages(stages)
    return max((event['end_us'] for event in timeline.values()), default=0.0)


def overlap_ratio(sequential_us: float, pipelined_us: float) -> float:
    """The share of the serial total time that is ideally hidden; does not represent device utilization."""
    if not all(math.isfinite(value) for value in (sequential_us, pipelined_us)):
        raise ValueError('时间必须为有限数')
    if not 0 <= pipelined_us <= sequential_us:
        raise ValueError('理想流水线时间必须位于零与串行时间之间')
    return 1 - pipelined_us / sequential_us if sequential_us else 0.0


```

## Q2: How does cross-Stream waiting enter the completion time?

After the producer finishes its work it records an Event, the consumer waits for that event, and then uses the result. Dependency edges constrain only the tasks that are genuinely related; order within the same queue exists implicitly. If the transfer-in, computation, and transfer-back of a single batch are chained head to tail, they still have to complete serially after being split into multiple Streams.

The dependency chain of four stages is kept below. The final host post-processing is also modeled as an independent logical queue; it is not a device operation on a CUDA Stream. Compare the start and end times of the stages and check whether the error "the consumer has started before the producer has finished" appears.


### Q2 mini check: A dependency chain cannot be shortened by splitting it across queues



```python
def build_dependency_edges(stages: List[TaskStage]) -> List[Tuple[str, str]]:
    """Return the explicit dependency edges; same-queue order is added separately by the timeline calculator."""
    schedule_stages(stages)
    return [(dep, stage.name) for stage in stages for dep in stage.depends_on]


def has_cross_stream_dependency(stages: List[TaskStage]) -> bool:
    schedule_stages(stages)
    name_to_stage = {stage.name: stage for stage in stages}
    return any(
        name_to_stage[dep].stream != stage.stream
        for stage in stages for dep in stage.depends_on
    )


```


```python
def test_event_dependencies():
    stages = [
        TaskStage('H2D', 12, 'copy'),
        TaskStage('Kernel', 40, 'compute', depends_on=('H2D',)),
        TaskStage('D2H', 10, 'copy', depends_on=('Kernel',)),
        TaskStage('Post', 8, 'host', depends_on=('D2H',)),
    ]
    assert build_dependency_edges(stages) == [
        ('H2D', 'Kernel'), ('Kernel', 'D2H'), ('D2H', 'Post')
    ]
    assert has_cross_stream_dependency(stages)
    timeline = schedule_stages(stages)
    assert timeline['Kernel']['start_us'] == 12
    assert timeline['D2H']['start_us'] == 52
    assert timeline['Post']['end_us'] == 70
    assert pipelined_time_us(stages) == sequential_time_us(stages) == 70
    print('依赖链时间线：', timeline)


test_event_dependencies()

```

## Q3: Does Graph replay change the dependencies or the submission cost?

CUDA Graph reuses an already captured execution structure and mainly targets the repeated submission overhead; the original data dependencies still hold. Even if the dependency chain from the previous question is captured, it cannot let the consumer execute before the producer finishes.

First choose subgraphs with a stable shape and execution path that run repeatedly, then check the conditions such as capture support, memory addresses, and resource lifetimes. Below only the simplest fixed-subgraph candidates are screened; a return of true does not yet mean the capture succeeds, nor that a speedup has been obtained. A variable-length service can bucket by shape or capture only the stable subgraph; you cannot conclude that the whole service cannot use Graph just because it is "variable length."


### Q3 mini check: Screen fixed-subgraph replay candidates



```python
def graph_suitability(is_fixed_shape: bool, is_repeated_path: bool, has_many_branches: bool) -> bool:
    """Screen simple fixed-subgraph candidates; does not verify actual capture support or performance."""
    return is_fixed_shape and is_repeated_path and not has_many_branches

assert graph_suitability(True, True, False) is True
assert graph_suitability(True, False, False) is False
assert graph_suitability(False, True, False) is False
assert graph_suitability(True, True, True) is False
print('✅ Graph 适用性测试通过')


```


```python
cases = [
    ('离线批处理推理', True, True, False),
    ('变长在线推理', False, True, False),
    ('动态分支很多的控制流', True, True, True),
]

for name, fixed_shape, repeated_path, many_branches in cases:
    print(f'{name:<18s}:', '可继续检查捕获条件' if graph_suitability(fixed_shape, repeated_path, many_branches) else '需先整理稳定子图')

```

## Q4: How does a multi-batch pipeline create overlap space?

A single batch has a dependency among transfer-in, computation, and transfer-back, but the input transfer of the next batch may be independent of the computation of the current batch. Below, two identical batches are fixed, assuming the input transfer, computation, and output transfer-back each have independent resources, to check the overlap window between batches. If the actual device shares copy resources, or these stages contend for bandwidth, the completion time will be longer than this model.

Stream priority affects the scheduling preference of ready work; it does not remove dependencies, nor is it a guarantee of immediate preemption of already-running work. Therefore, the calculator does not generate a fixed gain from priority; it first obtains the ideal saving from the dependency timeline, then adds the assumed exposed scheduling overhead separately.


### Q4 mini check: Compare the single-batch dependency chain with a two-batch pipeline



```python
def pipeline_comparison(stages: List[TaskStage], exposed_overhead_us=0.0) -> dict:
    """Record the ideal overlap and the given exposed overhead separately; does not recommend splitting queues by stage ratio."""
    if not math.isfinite(exposed_overhead_us) or exposed_overhead_us < 0:
        raise ValueError('暴露开销必须是有限非负数')
    serial = sequential_time_us(stages)
    ideal = pipelined_time_us(stages)
    return {
        'serial_us': serial,
        'ideal_pipeline_us': ideal,
        'ideal_saved_us': serial - ideal,
        'exposed_overhead_us': exposed_overhead_us,
        'estimated_elapsed_us': ideal + exposed_overhead_us,
        'net_saved_us': serial - ideal - exposed_overhead_us,
    }


stages = []
for batch in range(2):
    stages.extend([
        TaskStage(f'in{batch}', 12, 'copy_in'),
        TaskStage(f'compute{batch}', 40, 'compute', (f'in{batch}',)),
        TaskStage(f'out{batch}', 10, 'copy_out', (f'compute{batch}',)),
    ])
report = pipeline_comparison(stages, exposed_overhead_us=3)
assert report['serial_us'] == 124
assert report['ideal_pipeline_us'] == 102
assert report['net_saved_us'] == 19
assert pipeline_comparison(stages, exposed_overhead_us=25)['net_saved_us'] == -3
print('两批流水线：', schedule_stages(stages))
print('时间比较：', report)


```

## Q5: Which conditions eliminate the ideal overlap gain?

When verifying, first check whether the dependencies allow overlap, then check whether the execution resources are independent, and finally compare the actual end-to-end time; this separates "must wait" from "could have overlapped but did not."

| Condition | Effect on the model result | What to check during actual execution |
|:---|:---|:---|
| The consumer waits for the producer | Cross-queue tasks still run serially | Event timing and input availability time |
| Same-queue submission order | Later tasks cannot pass earlier ones | Whether head-of-queue waiting blocks independent work |
| Multiple queues share execution resources | The ideal parallel time is optimistic | Contention for copy engines, compute resources, and bandwidth |
| Submission or synchronization overhead is exposed | Cancels part or even all of the hidden time | Host submission and device completion times |
| Graph capture conditions are not met | The capture subgraph or execution path needs adjustment | Shape, addresses, and operation support |


## Related Reading

**Overview:** Continue the stream and event dependency judgment into real collective and communication overlap experiments.

- [Part 02 · 47 NCCL Communication Profiling](../02_PyTorch_Algorithms/47_Communication_Profiling_with_NCCL.md)
- [Part 02 · 48 Communication Hotspots and Mitigation](../02_PyTorch_Algorithms/48_Communication_Hotspots_and_Mitigation.md)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
---
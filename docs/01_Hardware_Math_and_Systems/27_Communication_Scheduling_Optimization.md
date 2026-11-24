# 27. Communication Scheduling Optimization | 通信调度优化

**Difficulty:** Hard | **Environment:** CPU-first | **Tags:** `parallel communication`, `communication optimization`, `Overlap` | **Audience:** learners starting with communication mechanisms

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/27_Communication_Scheduling_Optimization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

The total communication time is not necessarily all exposed on the training-step critical path. Message size, launch count, dependencies, and overlappable computation together determine whether communication really lengthens the step time.

This section establishes the relationship "communication load → launch and bandwidth cost → scheduling order → exposed time." When you finish, you should be able to compare the effect of bucketing, collective communication selection, and computation overlap on the critical path.

**Keywords:** communication-computation overlap, `All-Reduce`, `All-to-All`

![Concept relationships in this section](../public/01_Hardware_Math_and_Systems/27_communication_schedule_map.svg)


## Prerequisite Reading

**Overview:** First map the communication topology, parallel strategies, and NCCL synchronization primitives onto one training step, then analyze where communication is placed, at what granularity it is sent, and which computation can overlap with it.

- [05. Communication Topologies](./05_Communication_Topologies.md)
- [20. NCCL and AllReduce Basics](./20_NCCL_and_AllReduce_Basics.md)
- [26. Parallel Strategy Decision Framework](./26_Parallel_Strategy_Decision_Framework.md)

## Q1: How is communication cost determined jointly by data volume, bandwidth, latency, and synchronization?

The ideal time of one communication can be split into a fixed launch cost and a payload transfer cost. For the same total byte count, splitting into more messages does not lower the ideal transfer volume, but it pays the launch latency repeatedly.

| Quantity | Unit | Effect on communication time |
| --- | --- | --- |
| Payload | MiB | Grows linearly when bandwidth is fixed |
| Sustainable bandwidth | GiB/s | Determines the payload transfer lower bound |
| Single-launch latency | μs | More sensitive for small messages or high-frequency communication |
| Launch count | count | Determines how many times the fixed cost is paid |


### Q1 mini check

Separate the effects of data volume, bandwidth, latency, and the number of synchronizations.

```python
def communication_time_ms(payload_mib, bandwidth_gib_s, latency_us, launches=1):
    """Estimate the ideal time of one communication phase with an alpha-beta model."""
    if payload_mib < 0 or bandwidth_gib_s <= 0 or latency_us < 0 or launches <= 0:
        raise ValueError('有效载荷、带宽、延迟或启动次数非法')
    transfer_ms = payload_mib / 1024 / bandwidth_gib_s * 1000
    launch_ms = launches * latency_us / 1000
    return {
        'payload_mib': payload_mib,
        'launches': launches,
        'transfer_ms': round(transfer_ms, 4),
        'launch_ms': round(launch_ms, 4),
        'communication_ms': round(transfer_ms + launch_ms, 4),
    }

for launches in (1, 8, 64):
    print(communication_time_ms(256, bandwidth_gib_s=50, latency_us=8, launches=launches))

assert communication_time_ms(0, 50, 8, 4)['communication_ms'] == 0.032
assert communication_time_ms(256, 50, 8, 8)['transfer_ms'] == 5.0


```

## Q2: Why is merging small communications into large ones usually more stable?

In the backward pass, gradients can only enter communication after the corresponding layer is complete. Smaller buckets can start All-Reduce earlier but increase the launch count; larger buckets reduce the launch cost but may wait for more gradients to become ready.

| Bucketing change | Possible gain | Possible cost | Quantity to observe |
| --- | --- | --- | --- |
| Smaller buckets | Communication starts earlier | More launches | First bucket start time, launch count |
| Larger buckets | Lower fixed launch cost | More waiting for readiness | Communication finish time, exposed communication |
| Changed readiness order | A longer overlap window with backward computation | May break existing dependencies | Bucket readiness time and critical path |


### Q2 mini check

Compare the launch overhead and the overlap space after merging messages.

```python
def schedule_gradient_buckets(layers, bucket_cap_mib, bandwidth_gib_s=50, latency_us=8):
    """Build a communication timeline from gradient readiness time and bucket capacity."""
    if bucket_cap_mib <= 0 or not layers:
        raise ValueError('分桶容量必须为正，且层列表不能为空')
    buckets, pending, pending_mib = [], [], 0.0
    for layer in sorted(layers, key=lambda item: item['ready_ms']):
        pending.append(layer)
        pending_mib += layer['size_mib']
        if pending_mib >= bucket_cap_mib:
            buckets.append((pending, pending_mib))
            pending, pending_mib = [], 0.0
    if pending:
        buckets.append((pending, pending_mib))

    timeline, communication_available_ms = [], 0.0
    for bucket_id, (members, size_mib) in enumerate(buckets):
        ready_ms = max(member['ready_ms'] for member in members)
        start_ms = max(ready_ms, communication_available_ms)
        duration_ms = communication_time_ms(size_mib, bandwidth_gib_s, latency_us)['communication_ms']
        end_ms = start_ms + duration_ms
        timeline.append({
            'bucket_id': bucket_id,
            'layers': [member['name'] for member in members],
            'size_mib': round(size_mib, 2),
            'ready_ms': round(ready_ms, 4),
            'start_ms': round(start_ms, 4),
            'end_ms': round(end_ms, 4),
        })
        communication_available_ms = end_ms

    backward_end_ms = max(layer['ready_ms'] for layer in layers)
    return {
        'bucket_count': len(timeline),
        'backward_end_ms': backward_end_ms,
        'communication_end_ms': round(communication_available_ms, 4),
        'exposed_after_backward_ms': round(max(communication_available_ms - backward_end_ms, 0.0), 4),
        'timeline': timeline,
    }

layers = [
    {'name': 'L4', 'ready_ms': 2.0, 'size_mib': 32},
    {'name': 'L3', 'ready_ms': 5.0, 'size_mib': 32},
    {'name': 'L2', 'ready_ms': 9.0, 'size_mib': 32},
    {'name': 'L1', 'ready_ms': 14.0, 'size_mib': 32},
]
for bucket_cap_mib in (32, 64, 128):
    print(bucket_cap_mib, schedule_gradient_buckets(layers, bucket_cap_mib))

assert schedule_gradient_buckets(layers, 32)['bucket_count'] == 4
assert schedule_gradient_buckets(layers, 128)['bucket_count'] == 1


```

## Q3: Why do All-Reduce and All-to-All have different optimization targets?

All-Reduce reduces and broadcasts same-sized tensors on every rank; in the ring implementation, the logical transfer volume per rank is about `2(P-1)/P` times the payload. All-to-All instead sends different amounts of state to different ranks; when the total payload is similar, the receive volume of the busiest rank can still determine the completion time.

| Collective communication | Main load | Main source of imbalance | What to observe first when scheduling |
| --- | --- | --- | --- |
| All-Reduce | Gradients or sharded tensors | A slow rank, inconsistent bucket readiness times | Bucket readiness, launch count, exposed communication |
| All-to-All | tokens, expert inputs, or sequence shards | Skewed routing or sequence distribution | Maximum per-rank load, long tail, and idle ranks |


### Q3 mini check

Distinguish the synchronization pressure of All-Reduce from the routing pressure of All-to-All.

```python
def ring_allreduce_payload_mib(tensor_mib, world_size):
    """Return the logical transfer volume of each rank in a ring All-Reduce."""
    if tensor_mib < 0 or world_size < 2:
        raise ValueError('张量容量必须非负，world_size 至少为 2')
    return tensor_mib * 2 * (world_size - 1) / world_size

def alltoall_route_ledger(token_matrix, bytes_per_token):
    """Compute the cross-rank payload and receive imbalance of an All-to-All routing matrix."""
    world_size = len(token_matrix)
    if world_size < 2 or any(len(row) != world_size for row in token_matrix):
        raise ValueError('token_matrix 必须是至少包含两个 rank 的方阵')
    if bytes_per_token <= 0 or any(token < 0 for row in token_matrix for token in row):
        raise ValueError('token 数或每 token 字节数非法')
    received_tokens = [sum(token_matrix[src][dst] for src in range(world_size)) for dst in range(world_size)]
    remote_tokens = sum(
        token_matrix[src][dst]
        for src in range(world_size) for dst in range(world_size) if src != dst
    )
    mean_received = sum(received_tokens) / world_size
    return {
        'remote_payload_mib': round(remote_tokens * bytes_per_token / 2**20, 4),
        'received_tokens': received_tokens,
        'max_to_mean_receive_ratio': round(max(received_tokens) / mean_received, 4),
    }

balanced = [[16, 16, 16, 16] for _ in range(4)]
skewed = [[4, 4, 52, 4] for _ in range(4)]
print('Ring All-Reduce per-rank MiB:', ring_allreduce_payload_mib(256, 4))
print('Balanced All-to-All:', alltoall_route_ledger(balanced, bytes_per_token=4096))
print('Skewed All-to-All:', alltoall_route_ledger(skewed, bytes_per_token=4096))

assert ring_allreduce_payload_mib(256, 4) == 384
assert alltoall_route_ledger(skewed, 4096)['max_to_mean_receive_ratio'] > 3


```

## Q4: How do you fit communication into the gaps in computation?

Communication hiding can only come from compute intervals that have no unresolved data dependency on it. Multiple compute intervals may overlap each other, so you must first take their union within the communication interval and then compute the hidden and exposed time; overlapping windows cannot simply be summed.

| Time quantity | Meaning | Relationship to the critical path |
| --- | --- | --- |
| Total communication time | The active interval of the collective communication on the device | Does not necessarily lengthen the step in full |
| Hidden communication | Communication covered by independent compute intervals | No longer adds to the critical path separately |
| Exposed communication | The part of the communication interval that is not covered | The direct target that scheduling optimization should reduce |
| Overlap ratio | Hidden time / total communication time | Must be read together with step time |


### Q4 mini check

Observe how the critical-path cost changes once communication enters the gaps in computation.

```python
def communication_exposure(comm_interval, independent_compute_intervals):
    """Clip and merge compute intervals, and return the hidden and exposed communication time."""
    comm_start, comm_end = comm_interval
    if comm_end < comm_start:
        raise ValueError('通信区间非法')
    clipped = []
    for start, end in independent_compute_intervals:
        if end < start:
            raise ValueError('计算区间非法')
        left, right = max(start, comm_start), min(end, comm_end)
        if left < right:
            clipped.append((left, right))

    merged = []
    for start, end in sorted(clipped):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    communication_ms = comm_end - comm_start
    hidden_ms = sum(end - start for start, end in merged)
    exposed_ms = communication_ms - hidden_ms
    return {
        'communication_ms': round(communication_ms, 4),
        'hidden_ms': round(hidden_ms, 4),
        'exposed_ms': round(exposed_ms, 4),
        'overlap_ratio': round(hidden_ms / communication_ms, 4) if communication_ms else 1.0,
        'merged_overlap_intervals': [tuple(interval) for interval in merged],
    }

cases = [
    ((0, 12), []),
    ((0, 12), [(1, 4), (3, 8)]),
    ((0, 12), [(-2, 5), (5, 20)]),
]
for case in cases:
    print(case, '->', communication_exposure(*case))

assert communication_exposure((0, 12), [(1, 4), (3, 8)])['hidden_ms'] == 7
assert communication_exposure((0, 12), [(-2, 20)])['exposed_ms'] == 0


```

## Related Reading

**Overview:** Continue from theoretical scheduling to CUDA dependencies and the real communication timeline in Part 02.

- [17. CUDA Stream and Asynchrony](./17_CUDA_Stream_and_Asynchrony.md)
- [29. CUDA Stream Advanced Scheduling](./29_CUDA_Stream_Advanced_Scheduling.md)
- [Part 02 · 48 Communication Hotspots and Mitigation](../02_PyTorch_Algorithms/48_Communication_Hotspots_and_Mitigation.md)
---
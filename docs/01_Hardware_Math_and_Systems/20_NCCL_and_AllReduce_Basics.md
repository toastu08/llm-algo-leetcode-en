# 20. NCCL and AllReduce Basics | NCCL 与 AllReduce 基础

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `parallel communication`, `NCCL`, `AllReduce` | **Audience:** learners starting with communication mechanisms

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

When multi-GPU training scales poorly, the problem may come from collective communication and synchronization waiting, not just from single-GPU operator speed. Adding more GPUs also adds data and participants that must be coordinated.

This section establishes the relationship "local result → All-Reduce → ring propagation → latency and bandwidth cost." When you finish, you should be able to explain how Reduce-Scatter and All-Gather compose a ring All-Reduce, and estimate the communication volume and theoretical time of each process.


**Keywords:** `NCCL`, `AllReduce`, `DP`

![Concept relationships in this section](../public/01_Hardware_Math_and_Systems/20_nccl_collective_map.svg)

---
## Prerequisite Reading

**Overview:** First align communication with the parallelism hierarchy, then see how AllReduce and NCCL affect synchronization, the effective batch, and scaling efficiency.

- [Group 1C: Communication, Parallelism and Fault Tolerance](./1C.md)
- [05. Communication Topologies](./05_Communication_Topologies.md)
---
## Q1: What problem does NCCL actually solve in distributed training?


NCCL provides collective communication and point-to-point communication implementations for NVIDIA GPUs. It executes primitives such as All-Reduce, All-Gather, and Reduce-Scatter, but it is not responsible for choosing DP, TP, or PP, nor does it replace a higher-level scheduler.

In data parallel training, each rank first obtains local gradients and then calls collective communication to form a consistent result. If a collective is not covered by computation, the uncovered part goes directly into the critical path of the training step.


### Q1 mini check: How synchronization waiting enters the training step

Compute time and collective time cannot be multiplied linearly by the number of GPUs; first subtract the communication time that is genuinely covered by computation, and then you get the training-step critical path.

```python
def training_step_time(compute_ms, collective_ms, overlap_ms=0.0):
    """Return the critical-path time formed jointly by computation and the collective."""
    if min(compute_ms, collective_ms, overlap_ms) < 0:
        raise ValueError('时间必须非负')
    hidden_ms = min(overlap_ms, collective_ms, compute_ms)
    exposed_ms = collective_ms - hidden_ms
    return {
        'step_ms': compute_ms + exposed_ms,
        'hidden_collective_ms': hidden_ms,
        'exposed_collective_ms': exposed_ms,
    }

reports = [training_step_time(10.0, 3.0, overlap) for overlap in [0.0, 1.5, 3.0]]
assert reports[0]['exposed_collective_ms'] == 3.0
assert reports[-1]['exposed_collective_ms'] == 0.0
for report in reports:
    print(report)

```

## Q2: Why is AllReduce one of the most core communication primitives?


All-Reduce receives the same-shaped local tensor from each rank, performs a reduction, and returns the same complete result to all ranks. Its invariant is: the output shape is the same as the local input, and the outputs of all ranks are identical.

Data parallelism commonly uses it to aggregate local gradients. Summation and averaging differ only by one scaling step, but that scaling must be consistent with the loss reduction and the optimizer semantics.


### Q2 mini check: What is the output invariant of AllReduce

First think through "local result -> globally consistent" clearly.

```python
def allreduce_sum(local_vectors):
    """Reduce element-wise and return the same complete result to every rank."""
    if not local_vectors or len({len(row) for row in local_vectors}) != 1:
        raise ValueError('所有 rank 必须提供等长向量')
    reduced = [sum(items) for items in zip(*local_vectors)]
    return [reduced.copy() for _ in local_vectors]

local_gradients = [[1.0, 2.0], [10.0, 20.0], [100.0, 200.0], [1000.0, 2000.0]]
global_gradients = allreduce_sum(local_gradients)
print('rank 0 local:', local_gradients[0])
print('all ranks  :', global_gradients)
assert all(row == global_gradients[0] for row in global_gradients)

```

## Q3: Why do communication bandwidth and topology directly affect training scaling?


Communication is not only a question of "whether it can connect"; what matters more is "how fast it connects."

Ring All-Reduce contains `2 × (world_size - 1)` communication rounds. For a fixed payload, the theoretical communication volume per rank approaches twice the payload; the startup latency grows with the number of rounds, while the transfer term is constrained by the effective link bandwidth.


### Q3 mini check: The theoretical communication volume and alpha-beta cost of Ring

For a fixed full payload, compute the number of bytes each rank moves in the two phases, and separate the per-round startup latency from the transfer cost.

```python
def ring_cost(payload_bytes, world_size, alpha_ms, bandwidth_gbps):
    """Estimate the round count, per-rank communication volume, and alpha-beta time of Ring AllReduce."""
    if payload_bytes < 0 or world_size < 2 or alpha_ms < 0 or bandwidth_gbps <= 0:
        raise ValueError('检查 payload、world_size、alpha 和 bandwidth')
    rounds = 2 * (world_size - 1)
    bytes_per_rank = 2 * (world_size - 1) / world_size * payload_bytes
    bandwidth_bytes_per_ms = bandwidth_gbps * 1e9 / 8 / 1000
    latency_ms = rounds * alpha_ms
    transfer_ms = bytes_per_rank / bandwidth_bytes_per_ms
    return {
        'rounds': rounds,
        'bytes_per_rank': int(bytes_per_rank),
        'latency_ms': latency_ms,
        'transfer_ms': transfer_ms,
        'total_ms': latency_ms + transfer_ms,
    }

payload_bytes = 256 * 1024 * 1024
nvlink_like = ring_cost(payload_bytes, world_size=8, alpha_ms=0.01, bandwidth_gbps=900.0)
pcie_like = ring_cost(payload_bytes, world_size=8, alpha_ms=0.01, bandwidth_gbps=64.0)
assert nvlink_like['rounds'] == 14
assert nvlink_like['bytes_per_rank'] == 2 * 7 * payload_bytes // 8
assert pcie_like['transfer_ms'] > nvlink_like['transfer_ms']
print('高速链路：', nvlink_like)
print('低速链路：', pcie_like)

```

## Q4: Why is Ring AllReduce often used to explain the implementation idea of AllReduce?


The key point of Ring AllReduce is not that "ring is in the name," but that it splits a full aggregation into two more controllable phases:

1. **Reduce-Scatter**: first reduce the data of each GPU step by step along the ring, and split the result into several chunks scattered across the GPUs;
2. **All-Gather**: then stitch these locally reduced results back into the complete result along the ring.

The benefit of doing this is that each GPU only sends and receives a small chunk in every round, so the communication load is more even, and no single GPU has to bear the entire transfer pressure at once.

This simulator only verifies the change of algorithm state. Real NCCL chooses a concrete algorithm according to topology, message size, and configuration; actual performance must be measured in a multi-GPU environment.


### Q4 mini check: Trace chunk ownership and reduction state round by round

The first dimension of the input matrix represents the rank, and the second dimension represents the chunk. Reduce-Scatter moves only one partial chunk per round, and All-Gather then propagates the chunks whose reduction is complete.

```python
def ring_allreduce(local_chunks):
    """Simulate Ring Reduce-Scatter and All-Gather with synchronous snapshots."""
    world_size = len(local_chunks)
    if world_size < 2 or any(len(row) != world_size for row in local_chunks):
        raise ValueError('需要 world_size × world_size 的 chunk 矩阵')

    partial = [list(row) for row in local_chunks]
    trace = []
    for step in range(world_size - 1):
        snapshot = [row.copy() for row in partial]
        transfers = []
        for rank in range(world_size):
            source = (rank - 1) % world_size
            chunk = (rank - step - 1) % world_size
            partial[rank][chunk] += snapshot[source][chunk]
            transfers.append({'from': source, 'to': rank, 'chunk': chunk})
        trace.append({'phase': 'reduce_scatter', 'step': step, 'transfers': transfers})

    owned = {(rank + 1) % world_size: partial[rank][(rank + 1) % world_size] for rank in range(world_size)}
    gathered = [{(rank + 1) % world_size: partial[rank][(rank + 1) % world_size]} for rank in range(world_size)]
    for step in range(world_size - 1):
        snapshot = [dict(items) for items in gathered]
        transfers = []
        for rank in range(world_size):
            source = (rank - 1) % world_size
            chunk = (rank - step) % world_size
            gathered[rank][chunk] = snapshot[source][chunk]
            transfers.append({'from': source, 'to': rank, 'chunk': chunk})
        trace.append({'phase': 'all_gather', 'step': step, 'transfers': transfers})

    outputs = [[items[index] for index in range(world_size)] for items in gathered]
    expected = [sum(local_chunks[rank][chunk] for rank in range(world_size)) for chunk in range(world_size)]
    assert all(row == expected for row in outputs)
    return {'outputs': outputs, 'owned_after_reduce_scatter': owned, 'trace': trace}

local_chunks = [[rank * 10 + chunk for chunk in range(4)] for rank in range(4)]
result = ring_allreduce(local_chunks)
print('reduce-scatter owners:', result['owned_after_reduce_scatter'])
print('rank 0 output:', result['outputs'][0])
print('rounds:', len(result['trace']))

```

---
## Related Reading
This section only establishes the algorithm semantics and theoretical cost. Real NCCL may choose Ring, Tree, or another implementation according to message size and topology; continue to Part 02 · 47 to verify the requested algorithm, the actual algorithm, and the multi-GPU time.
- [NCCL official documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/): look up collective communication primitives, topology, and the communication environment.
- [NCCL Tests](https://github.com/NVIDIA/nccl-tests): observe collective communication bandwidth with a fixed message size and GPU count.
- [47. Communication Profiling with NCCL](../02_PyTorch_Algorithms/47_Communication_Profiling_with_NCCL.md)

---
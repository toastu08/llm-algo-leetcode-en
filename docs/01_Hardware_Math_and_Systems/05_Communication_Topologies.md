# 05. Communication Topologies | 通信拓扑与分布式基石

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `parallel communication`, `distributed training`, `communication topology` | **Audience:** learners who need to understand the communication cost of multi-GPU training

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/05_Communication_Topologies.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

Once a large model scales from a single GPU to multiple GPUs, every partitioning scheme introduces data exchange. Data parallelism (DP), tensor parallelism (TP), and pipeline parallelism (PP) partition different objects, so their communication loads and synchronization points also differ.

This section establishes the classification chain "partitioned object → collective communication interface → topology path → parallel group placement." When you finish, you should be able to point out which communication primitives a partitioning scheme may require, estimate the lower bound of link transfer, and check whether high-frequency communication is placed in a suitable topology domain; the algorithmic process of All-Reduce is expanded in section 20, and parallel candidate screening is completed in section 26.

**Keywords:** `DP`, `TP`, `PP`

![Concept relationships in this section](../public/01_Hardware_Math_and_Systems/05_communication_topology_map.svg)

---

## Prerequisite Reading
**Overview:** First review the GPU memory hierarchy and inter-device connections, then use the DP, TP, and PP examples to answer three questions: where the data is placed, which data must be exchanged, and how link bandwidth changes the parallel payoff.
- [03. GPU Architecture and Memory](./03_GPU_Architecture_and_Memory.md)
- [Group 1C: Communication, Parallelism and Fault Tolerance](./1C.md)

---
## Q1: How do DP, TP, and PP partition data, tensors, and layers respectively?


First look separately at what data parallelism (DP), tensor parallelism (TP), and pipeline parallelism (PP) each partition, then explain why their combination is called 3D parallelism, and finally trace the data exchange produced by the partitioning.

- **DP**: different GPUs process different data batches, then synchronize gradients.
- **TP**: split the large tensors inside a single layer across GPUs for joint computation.
- **PP**: place different layers on different devices or device groups to form a pipeline.

When DP, TP, and PP are used together, this is usually called 3D parallelism. Their goal is not "the more, the better," but to make the model, compute, and communication topology match together. First look separately at the partitioned object, the main data exchanged, and the common communication pattern:

| Parallelism | Partitioned object | Main data exchanged | Common communication pattern |
| --- | --- | --- | --- |
| DP | Different data batches | Gradients or parameter update results | Usually All-Reduce |
| TP | Tensors or activations within a layer | Local activations, local results | May use All-Gather, Reduce-Scatter, or All-Reduce |
| PP | Different layers or stages | Activations between adjacent stages | Point-to-point transfer, together with pipeline waiting |


### Q1 mini check: Compare the partitioned object and communication focus of each parallel dimension

Check item by item what each parallelism mode partitions, what it needs to exchange, and the corresponding communication pattern.

```python
def three_d_parallel(dp, tp, pp):
    """Combine the three parallelism dimensions DP, TP, and PP, and return the partitioning relationship and communication focus."""
    if any(value <= 0 for value in (dp, tp, pp)):
        raise ValueError('dp、tp、pp 必须为正数')
    # First record the three parallelism dimensions, then observe the worker count and communication relationship after they are combined.
    return {
        'dp_groups': dp,
        'tp_shards': tp,
        'pp_stages': pp,
        'effective_workers': dp * tp * pp,
        'partition_axis': {'dp': 'data', 'tp': 'tensor', 'pp': 'layers'},
        'communication_focus': {
            'dp': 'gradient synchronization',
            'tp': 'activation gather / reduce-scatter',
            'pp': 'stage activation transfer',
        },
    }

cases = [
    three_d_parallel(8, 1, 1),
    three_d_parallel(4, 2, 2),
    three_d_parallel(2, 4, 4),
]
for case in cases:
    print(case)
assert cases[0]['partition_axis'] == {'dp': 'data', 'tp': 'tensor', 'pp': 'layers'}
assert 'gradient' in cases[1]['communication_focus']['dp']
assert 'activation' in cases[2]['communication_focus']['pp']
print('3D parallelism = DP × TP × PP')

```

## Q2: What is the difference between All-Reduce, All-Gather, and Reduce-Scatter?


The difference between these three collective communication primitives starts with the input, the output, and the data changes on each GPU. The shapes below are only a dataflow illustration; the actual cost is also affected by message size, world size, topology, the collective algorithm, and the communication library implementation. The code only simulates these input / output shapes and does not perform real NCCL communication.

| Communication primitive | Input per GPU | Output per GPU | Data change | Common use case |
| --- | --- | --- | --- | --- |
| All-Reduce | Local tensor | Reduction result of the same shape | Reduce, then copy to every GPU | DP gradient synchronization |
| All-Gather | Local shards | Concatenated full tensor | Collect the shards from every GPU | TP activation or parameter gathering |
| Reduce-Scatter | Larger local input | Reduced local result | Reduce first, then split | Sharded gradient or parameter synchronization |


### Q2 mini check: How communication primitives change the data each GPU sees

First observe the three dataflows: "aggregate," "gather," and "split then send back."

```python
def collective_shape(kind, world_size, local_shape):
    """Return the input / output shape of a collective communication; only simulate the dataflow, without performing real communication."""
    if world_size <= 0 or not local_shape or any(dim <= 0 for dim in local_shape):
        raise ValueError('world_size 和 local_shape 必须为正数')
    table = {
        'allreduce': {'operation': 'reduce + broadcast', 'input_per_rank': local_shape, 'output_per_rank': local_shape},
        'allgather': {'operation': 'gather all pieces', 'input_per_rank': local_shape, 'output_per_rank': (local_shape[0] * world_size, *local_shape[1:])},
        'reducescatter': {'operation': 'reduce then scatter', 'input_per_rank': (local_shape[0] * world_size, *local_shape[1:]), 'output_per_rank': local_shape},
    }
    if kind not in table:
        raise ValueError(f'未知通信原语: {kind}')
    return table[kind]

world_size = 4
local_shape = (8, 1024)
for kind in ['allreduce', 'allgather', 'reducescatter']:
    print(kind, '->', collective_shape(kind, world_size, local_shape))
assert collective_shape('allreduce', 4, (8, 1024))['output_per_rank'] == (8, 1024)
assert collective_shape('allgather', 4, (8, 1024))['output_per_rank'] == (32, 1024)
assert collective_shape('reducescatter', 4, (8, 1024))['input_per_rank'] == (32, 1024)
try:
    collective_shape('unknown', 4, (8, 1024))
except ValueError:
    print('✅ 非法通信原语校验通过')
else:
    raise AssertionError('未知通信原语应报错')
```

## Q3: How does the communication topology change the communication cost?


First distinguish where communication takes place, then compare the ideal transfer time of the same message size over different links. The bandwidth in the table below is uniformly expressed in Gb/s and is only a teaching example; it does not represent every GPU, motherboard, driver, or network configuration. Real experiments should be based on `nvidia-smi topo -m`, NCCL tests, or vendor specifications.

| Communication location | Common link | Teaching-example bandwidth (Gb/s) | Main characteristics | Communication it suits |
| --- | --- | ---: | --- | --- |
| Within a GPU node | NVLink / NVSwitch | 900 class | High bandwidth, relatively low latency | High-frequency TP or collective communication |
| GPU to CPU | PCIe | 64 class | Lower bandwidth, longer path | Parameter movement, low-frequency synchronization |
| Across nodes | Network interconnect | Depends on the actual cluster | Strongly affected by the network and topology | Low-frequency or overlappable communication |


### Q3 mini check: How long the same message takes over different links

Compare the ideal transfer time of different links for the same message size.

```python
def path_transfer_lower_bound(payload_mib, links):
    """Estimate the one-way transfer lower bound from the narrowest link on the path, excluding protocol and congestion overhead."""
    if payload_mib < 0 or not links:
        raise ValueError('payload_mib 不得为负，links 不得为空')
    if any(link['bandwidth_gib_s'] <= 0 for link in links):
        raise ValueError('链路带宽必须大于 0 GiB/s')
    bottleneck = min(links, key=lambda link: link['bandwidth_gib_s'])
    lower_bound_ms = payload_mib / 1024 / bottleneck['bandwidth_gib_s'] * 1000
    return {
        'bottleneck_link': bottleneck['name'],
        'bottleneck_gib_s': bottleneck['bandwidth_gib_s'],
        'lower_bound_ms': round(lower_bound_ms, 3),
    }

intra_node = path_transfer_lower_bound(256, [{'name': 'NVLink', 'bandwidth_gib_s': 100}])
cross_node = path_transfer_lower_bound(256, [
    {'name': 'GPU-NIC', 'bandwidth_gib_s': 50},
    {'name': 'network', 'bandwidth_gib_s': 12.5},
])
assert cross_node['lower_bound_ms'] > intra_node['lower_bound_ms']
print('机内路径：', intra_node)
print('跨节点路径：', cross_node)
```

## Q4: How do you place parallel groups in suitable topology domains?


Placement of parallel groups starts with the communication frequency and the available topology domains. High-frequency communication such as TP and EP is usually confined within a node first; DP or PP can span nodes, but the actual collective communication or stage transfer still needs to be measured in later experiments.

| Observation condition | Possible risk | Placement check |
| --- | --- | --- |
| DP needs to synchronize gradients across a low-bandwidth link | Gradient communication slows down every update round | Record the cross-node bandwidth and the gradient payload |
| TP needs frequent synchronization across a low-bandwidth link | Collective communication becomes the bottleneck | Narrow the TP range and place it inside the high-speed interconnect first |
| PP transfers many activations across stages | Waiting between stages, pipeline bubble | Record the stage boundary and the cross-domain payload |
| EP exchanges tokens across nodes | All-to-All plus uneven load | First check whether the expert group can be confined within a node |
| Only an ideal-bandwidth estimate | Cannot confirm the real deployment payoff | Bring the same workload into later system experiments |


### Q4 mini check: Check whether parallel groups match their topology domains

```python
DOMAIN_ORDER = {'cross_node': 1, 'intra_node': 2}

def check_group_placement(groups):
    """Compare the topology domain required by a parallel group with its actual placement domain."""
    reports = []
    for group in groups:
        accepted = DOMAIN_ORDER[group['actual_domain']] >= DOMAIN_ORDER[group['required_domain']]
        reports.append({
            'name': group['name'],
            'required_domain': group['required_domain'],
            'actual_domain': group['actual_domain'],
            'accepted': accepted,
        })
    return reports

groups = [
    {'name': 'TP group', 'required_domain': 'intra_node', 'actual_domain': 'intra_node'},
    {'name': 'EP group', 'required_domain': 'intra_node', 'actual_domain': 'cross_node'},
    {'name': 'DP group', 'required_domain': 'cross_node', 'actual_domain': 'cross_node'},
]
reports = check_group_placement(groups)
assert next(row for row in reports if row['name'] == 'TP group')['accepted']
assert not next(row for row in reports if row['name'] == 'EP group')['accepted']
for report in reports:
    print(report)

```

---
## Related Reading
This section connects on to NCCL primitives, parallel strategy selection, and communication scheduling.
- [20. NCCL and AllReduce Basics](./20_NCCL_and_AllReduce_Basics.md)
- [26. Parallel Strategy Decision Framework](./26_Parallel_Strategy_Decision_Framework.md)
- [27. Communication Scheduling Optimization](./27_Communication_Scheduling_Optimization.md)
---
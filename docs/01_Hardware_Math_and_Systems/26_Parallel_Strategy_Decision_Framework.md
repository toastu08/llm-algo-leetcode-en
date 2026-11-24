# 26. Parallel Strategy Decision Framework | 并行策略决策框架

**Difficulty:** Hard | **Environment:** CPU-first | **Tags:** `parallel communication`, `parallel strategy`, `strategy decision` | **Audience:** learners starting with communication mechanisms

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/26_Parallel_Strategy_Decision_Framework.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

When a single device cannot meet the capacity or throughput target, you can split the training state, network layers, tensors, sequences, experts, or serving requests. Different partitioned objects solve different problems, and each also introduces different communication and load risks.

This section establishes the screening order "capacity constraint → partitioned object → interconnect condition → risk threshold." When you finish, you should be able to first exclude candidates that do not satisfy the hard constraints, then determine for the retained strategies which communication and scaling evidence needs to be verified.

**Keywords:** `DP`, `TP`, `PP`, `CP`, `EP`, `Replica`

![Concept relationships in this section](../public/01_Hardware_Math_and_Systems/26_parallel_strategy_decision_map.svg)
## Prerequisite Reading

**Overview:** First connect the basic intuitions of communication topology and memory partitioning, then look at the parallel strategy selection on this page; it will be easier to link "whether it fits" with "how to split."

- [05. Communication Topologies](./05_Communication_Topologies.md)
- [06. VRAM Calculation and ZeRO](./06_VRAM_Calculation_and_ZeRO.md)
## Q1: Is the current bottleneck first memory, communication, or model structure?


Do not memorize the names of parallel strategies first; register the constraints first:

1. **Capacity constraint**: how much model state, activation, and runtime headroom each device must keep
2. **Interconnect constraint**: whether parallel groups are inside a node or across nodes, and what the link bandwidth and latency are
3. **Workload constraint**: whether the current task is training or serving, whether the model is dense or MoE, and whether the context needs to be split

From this angle, different strategies handle different objects:
- **DP** solves "how to spread out the data," provided the model at least nearly fits
- **TP** solves "how to split the computation of a single layer," provided the intra-node interconnect is strong enough
- **PP** solves "how to segment the layers," provided the model is deep enough or the per-GPU memory is rather tight
- **EP** solves "how to dispatch the experts," provided routing and token exchange can bear it
- **CP** solves "how to split a long sequence," provided Attention state exchange can bear it
- **Replica** solves "how to distribute the requests," provided multi-replica capacity, queues, and cache locality are observable

Strategy selection is not about giving a candidate a composite score, but about recording measurable constraints first and then excluding candidates that do not satisfy the hard conditions.

| Strategy | Main partitioned object | Main pressure | Most critical precondition |
| --- | --- | --- | --- |
| DP / ZeRO | batch, parameters, gradients, or optimizer state | Gradient synchronization and state aggregation | There is a clear sharding plan for the model and training state |
| TP | Tensors and activations within a layer | High-frequency collective | Strong intra-node interconnect |
| PP | layers / stages | pipeline bubble and stage transfer | The model is deep enough and stages can be balanced |
| CP | sequence / context | Attention state exchange | The long-context gain covers the communication cost |
| EP | experts / tokens | dispatch, combine, and uneven load | MoE routing and capacity are controllable |
| Replica | serving requests | Queues, cache locality, and tail latency | Multi-replica state is observable |


### Q1 mini check

Build a resource ledger with units and check the single-device capacity gap.

```python
def build_resource_ledger(state_gib, activation_gib, runtime_reserve_gib, device_memory_gib):
    """Aggregate the single-device capacity requirement; all inputs are in GiB."""
    values = (state_gib, activation_gib, runtime_reserve_gib, device_memory_gib)
    if any(value < 0 for value in values) or device_memory_gib == 0:
        raise ValueError('容量必须为非负数，且设备显存必须大于 0 GiB')
    required_gib = state_gib + activation_gib + runtime_reserve_gib
    headroom_gib = device_memory_gib - required_gib
    return {
        'required_gib': round(required_gib, 2),
        'device_memory_gib': round(device_memory_gib, 2),
        'headroom_gib': round(headroom_gib, 2),
        'fits_one_device': headroom_gib >= 0,
    }

fits = build_resource_ledger(13.0, 4.0, 2.0, 24.0)
does_not_fit = build_resource_ledger(22.0, 5.0, 2.0, 24.0)
assert fits['fits_one_device']
assert not does_not_fit['fits_one_device']
print('可单卡容纳：', fits)
print('存在容量缺口：', does_not_fit)


```

## Q2: What communication cost does each parallel strategy introduce?


Do not look only at the "total volume" of communication cost; look at two dimensions at the same time:

- **Bandwidth term**: how much data a single transfer carries
- **Frequency term**: how many times this transfer action repeats along the whole training path

It can be roughly written as:
$$\text{cost} \approx \text{latency} \times \text{count} + \frac{\text{data volume}}{\text{bandwidth}}$$

This is also why, although both are called "communication," different strategies have completely different weak points:
- **DP** gradient synchronization is more global and low-frequency, but each synchronization covers a fuller extent
- **TP** intra-layer communication is high-frequency and usually small per call, but every layer hits it once
- **PP** communication happens at stage boundaries and is somewhat lower-frequency, but poor partitioning amplifies bubbles and waiting
- **EP** depends not only on volume but also on whether token routing and load balance are stable
- **CP** exchanges Attention state with context length, so the sequence length and the number of communication rounds must be recorded together
- **Replica** does not exchange tensors inside the model, but routing imbalance amplifies queueing and tail latency

So when judging a parallel strategy, what really needs to be asked is:
- Does this communication occur at high frequency
- Can it be absorbed by the intra-node interconnect
- Will it stall at a synchronization point and squeeze out compute overlap

If the frequency is high and the bandwidth is weak, no matter how "advanced" the strategy is, it will end up being dragged back by communication.


### Q2 mini check

Record the single-call payload, the number of calls, the link bandwidth, and the fixed latency in the same units.

```python
def communication_ledger(payload_mib, calls_per_step, bandwidth_gib_s, latency_us):
    """Estimate the communication lower bound within one training step, excluding congestion and overlap."""
    if payload_mib < 0 or calls_per_step < 0 or bandwidth_gib_s <= 0 or latency_us < 0:
        raise ValueError('载荷、次数和时延不得为负，带宽必须大于 0 GiB/s')
    transfer_ms_per_call = payload_mib / 1024 / bandwidth_gib_s * 1000
    latency_ms_per_call = latency_us / 1000
    total_ms = calls_per_step * (transfer_ms_per_call + latency_ms_per_call)
    return {
        'payload_per_call_mib': payload_mib,
        'calls_per_step': calls_per_step,
        'total_payload_mib': payload_mib * calls_per_step,
        'transfer_ms': round(transfer_ms_per_call * calls_per_step, 3),
        'latency_ms': round(latency_ms_per_call * calls_per_step, 3),
        'lower_bound_ms': round(total_ms, 3),
    }

low_frequency = communication_ledger(256, 1, 100, 5)
high_frequency = communication_ledger(32, 24, 100, 5)
assert high_frequency['calls_per_step'] > low_frequency['calls_per_step']
print('低频大载荷：', low_frequency)
print('高频小载荷：', high_frequency)


```

## Q3: How do you screen parallel strategies by constraint?


A more practical decision framework is not "pick a single-point strategy," but to exclude in order by hard constraint:

1. **Fit gate**: first see whether a single GPU can hold the model body
   - If it fits, DP has a basis; if it does not, PP / ZeRO enter the candidate set first
2. **Interconnect gate**: next see whether the intra-node interconnect can support high-frequency communication
   - A strong interconnect favors TP / EP; a weak interconnect favors reducing the synchronization frequency
3. **Structure gate**: finally see whether the model is dense or MoE
   - For dense models the common combination is DP / TP / PP
   - For MoE, EP must be brought into the main decision
4. **Context gate**: whether the context is long enough that Attention state must be split across GPUs
   - Only when the Attention state of a single GPU does not fit does CP enter the candidate set
5. **Serving gate**: whether the goal is to synchronize training state or to distribute independent requests
   - Multi-replica Serving evaluates Replica routing first, and does not mistakenly write it as training DP

A candidate page must openly state its device memory requirement, allowed workloads, model structures, and link domain. Only when these requirements are all satisfied by the current environment does the candidate enter the subsequent measurement.

### Q3 mini check

Have each candidate return a pass status and rejection reasons, and do not generate a composite score of unclear meaning.

```python
LINK_SCOPE_ORDER = {'none': 0, 'cross_node': 1, 'intra_node': 2}

def evaluate_candidate(candidate, environment):
    """Check the hard constraints declared by a candidate item by item, and return explicit rejection reasons."""
    reasons = []
    if candidate['memory_per_rank_gib'] > environment['memory_per_device_gib']:
        reasons.append('per-rank memory exceeds device memory')
    if environment['mode'] not in candidate['modes']:
        reasons.append('workload mode is unsupported')
    if environment['model_kind'] not in candidate['model_kinds']:
        reasons.append('model structure is unsupported')
    if LINK_SCOPE_ORDER[environment['link_scope']] < LINK_SCOPE_ORDER[candidate['required_link_scope']]:
        reasons.append('interconnect scope is insufficient')
    return {'name': candidate['name'], 'accepted': not reasons, 'reasons': reasons}

environment = {
    'memory_per_device_gib': 24,
    'mode': 'training',
    'model_kind': 'dense',
    'link_scope': 'intra_node',
}
candidates = [
    {'name': 'DP', 'memory_per_rank_gib': 29, 'modes': {'training'}, 'model_kinds': {'dense', 'moe'}, 'required_link_scope': 'cross_node'},
    {'name': 'ZeRO-3', 'memory_per_rank_gib': 18, 'modes': {'training'}, 'model_kinds': {'dense', 'moe'}, 'required_link_scope': 'cross_node'},
    {'name': 'TP', 'memory_per_rank_gib': 16, 'modes': {'training', 'serving'}, 'model_kinds': {'dense', 'moe'}, 'required_link_scope': 'intra_node'},
    {'name': 'EP', 'memory_per_rank_gib': 14, 'modes': {'training', 'serving'}, 'model_kinds': {'moe'}, 'required_link_scope': 'intra_node'},
]
reports = [evaluate_candidate(candidate, environment) for candidate in candidates]
assert not next(report for report in reports if report['name'] == 'DP')['accepted']
assert next(report for report in reports if report['name'] == 'TP')['accepted']
for report in reports:
    print(report)


```

## Q4: After choosing parallel strategies, how do you check the plan's risks?


A single candidate passing the hard gates does not mean the combination is still feasible. The combination check includes at least the device count, the degree of each dimension, the model structure, and the deployment domain; performance risk is still left to real communication and load evidence.

For example, `TP=4, PP=2, DP=2` requires at least 16 ranks; EP can only be used for MoE; serving replicas and training DP both replicate the model, but their synchronization semantics differ and cannot be substituted for each other by the same name.

### Q4 mini check

Check the number of ranks and the structural conditions required by the combination, and output problems that can be located.

```python
def validate_parallel_composition(degrees, available_ranks, model_kind):
    """Check the parallelism product, the available rank count, and the model structure."""
    supported = {'DP', 'TP', 'PP', 'CP', 'EP', 'Replica'}
    unknown = set(degrees) - supported
    if unknown:
        raise ValueError(f'未知并行维度：{sorted(unknown)}')
    if any(not isinstance(degree, int) or degree < 1 for degree in degrees.values()):
        raise ValueError('每个并行度必须是大于等于 1 的整数')
    required_ranks = 1
    for degree in degrees.values():
        required_ranks *= degree
    issues = []
    if required_ranks > available_ranks:
        issues.append(f'requires {required_ranks} ranks but only {available_ranks} are available')
    if degrees.get('EP', 1) > 1 and model_kind != 'moe':
        issues.append('EP degree greater than 1 requires an MoE model')
    if degrees.get('Replica', 1) > 1 and degrees.get('DP', 1) > 1:
        issues.append('Replica and training DP must use distinct execution roles')
    return {'required_ranks': required_ranks, 'valid': not issues, 'issues': issues}

valid_plan = validate_parallel_composition({'TP': 4, 'PP': 2, 'DP': 2}, 16, 'dense')
invalid_plan = validate_parallel_composition({'TP': 4, 'PP': 2, 'EP': 2}, 8, 'dense')
assert valid_plan['valid']
assert not invalid_plan['valid']
print('可行组合：', valid_plan)
print('不可行组合：', invalid_plan)


```

## Related Reading

**Overview:** Continue to the strategy selection page, the context partitioning page, and the PyTorch distributed documentation to check the real evidence each candidate requires.

- [Part 02 · 49 Parallelism Strategy Selection](../02_PyTorch_Algorithms/49_Parallelism_Strategy_Selection.md)
- [Part 02 · PAR-CONTEXT Context and Sequence Parallelism](../02_PyTorch_Algorithms/PAR-CONTEXT_Context_and_Sequence_Parallelism.md)
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
---
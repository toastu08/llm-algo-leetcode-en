# 18. Memory Profiling and Optimization | 显存账本与运行时证据

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `PyTorch`, `memory ledger`, `runtime evidence` | **Audience:** learners preparing for Parts 2–4

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/18_Memory_Profiling_and_Optimization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## Section Guide

Training memory includes not only model parameters, but also gradients, optimizer state, saved activations, and temporary buffers. These objects are created, overlap, and are released at different stages, so a model that can load does not mean a training step can complete.

After completing this section, you should be able to estimate a candidate peak by object lifetime, explain which ledger items are changed by batch, low precision, and recomputation, and use runtime records to identify omissions in the theoretical ledger. After changing one ledger assumption, observe how the peak objects and the cost still to be measured change.

**Keywords:** `memory ledger`, `lifetime`, `allocated`, `reserved`

![Training memory objects, lifetimes, and runtime reconciliation](../public/00_Prerequisites/18_memory_object_strategy_map.svg)

## Prerequisite Reading
**Overview:** First learn the diagnostic order from the 0E group page, confirm that the run is valid with section 19, and then put parameters, gradients, activations, and temporary tensors into the same lifetime ledger.
- [0E Group Page](./0E.md)
- [Part 00 · 19 Debugging and Anomaly Localization](./19_Debugging_and_Anomaly_Localization.md)

## Q1: Why must the training memory ledger record objects and lifetimes?

The fact that the model parameters fit on the GPU does not mean training will not run out of memory (OOM): the forward stage saves the activations needed for backward, the backward stage produces gradients, the optimizer state is usually initialized around the first update, and operators may also request temporary buffers. This example estimates the steady training stage after the first update: it assumes gradients are cleared at the start of each step, and uses one representative saved tensor and an assumed temporary buffer to demonstrate overlap. It is not a complete activation measurement of the current small model.


| Memory object | Lifetime | Changes with batch | Main influencing factors |
|:---|:---|:---:|:---|
| Parameter | Persists after the model is loaded | No | Parameter count, dtype |
| Gradient | Produced gradually during backward; clearing method determines whether it is kept | Usually no | Trainable parameter count, gradient dtype |
| Optimizer state | Usually created at the first update, then persists | No | Optimizer type, state precision |
| Saved activation | Produced in forward, released gradually after use in backward | Yes | Batch, sequence length, layer count |
| Temporary buffer | Exists briefly during operator execution | Yes | Operator implementation, workspace, layout |


```python
import torch
import torch.nn as nn


def pretty_mib(nbytes):
    return f"{nbytes / 1024**2:.2f} MiB"


class TinyLedgerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )

    def forward(self, x):
        return self.net(x)


model = TinyLedgerNet()
param_count = sum(p.numel() for p in model.parameters())
param_bytes = param_count * 4
grad_bytes = param_count * 4
adam_state_bytes = param_count * 8

batch_size, seq_len, hidden_size = 16, 64, 512
saved_activation_bytes = batch_size * seq_len * hidden_size * 4
temporary_bytes = saved_activation_bytes // 2

components = {
    'parameters_mib': param_bytes / 1024**2,
    'gradients_mib': grad_bytes / 1024**2,
    'optimizer_state_mib': adam_state_bytes / 1024**2,
    'saved_activations_mib': saved_activation_bytes / 1024**2,
    'temporary_mib': temporary_bytes / 1024**2,
}
phase_members = {
    'after_load': ('parameters_mib',),
    'forward_peak': ('parameters_mib', 'optimizer_state_mib', 'saved_activations_mib', 'temporary_mib'),
    'backward_peak': ('parameters_mib', 'gradients_mib', 'optimizer_state_mib', 'saved_activations_mib', 'temporary_mib'),
    'after_update': ('parameters_mib', 'gradients_mib', 'optimizer_state_mib'),
}
phase_totals = {
    phase: sum(components[name] for name in names)
    for phase, names in phase_members.items()
}
ledger = {
    'status': 'estimated',
    'assumptions': 'steady_state; one representative activation; assumed workspace',
    'objects_mib': components,
    'phase_members': phase_members,
    'phase_totals_mib': phase_totals,
    'overlap_upper_bound_mib': max(phase_totals.values()),
}
print('objects:', {name: round(value, 2) for name, value in components.items()})
print('phase totals:', {name: round(value, 2) for name, value in phase_totals.items()})

```

## Q1 verification: Are objects counted only in the stages where they may appear?

After loading, only parameters should exist; the candidate backward peak should include gradients, optimizer state, saved activations, and temporary buffers at the same time.


```python
assert phase_members['after_load'] == ('parameters_mib',)
assert {'gradients_mib', 'saved_activations_mib', 'temporary_mib'} <= set(phase_members['backward_peak'])
assert ledger['overlap_upper_bound_mib'] == phase_totals['backward_peak']
assert all(value >= 0 for value in components.values())
print('✅ 对象生命周期与候选峰值阶段通过')

```

## Q2: Why can object sizes not be added directly into a real peak?

The real peak depends on whether objects are alive at the same time. The loading stage mainly keeps parameters; the forward peak adds saved activations and temporary buffers; the backward stage may also have gradients at the same time. The theoretical ledger can give the candidate peak stage and the object contributions, but the allocator, operator workspace, and release timing may still make the measured result deviate from the theoretical value.

| Stage | Main objects newly entering or still alive | What the ledger can explain |
|:---|:---|:---|
| After loading | Parameters | The minimum residency requirement of the model state |
| Candidate forward peak | Parameters, optimizer state, saved activations, temporary buffers | The dynamic pressure caused by batch and sequence length |
| Candidate backward peak | Forward objects, gradients | The theoretical upper bound of overlapping state categories |
| After update | Parameters, gradients, optimizer state | The long-lived state before the update stage ends |


```python
def summarize_candidate_peak(objects_mib, phases):
    """Compute the candidate peak stage and the object contributions in that stage by lifetime."""
    if any(value < 0 for value in objects_mib.values()):
        raise ValueError('对象大小不能为负数')
    totals = {phase: sum(objects_mib[name] for name in names) for phase, names in phases.items()}
    peak_phase = max(totals, key=totals.get)
    contributions = {name: objects_mib[name] for name in phases[peak_phase]}
    return {
        'peak_phase': peak_phase,
        'candidate_peak_mib': totals[peak_phase],
        'phase_totals_mib': totals,
        'contributions_mib': contributions,
    }


peak_report = summarize_candidate_peak(components, phase_members)
print('candidate peak:', peak_report)

```

## Q2 verification: Is the candidate peak determined by lifetime overlap?

The candidate peak must come from an explicit stage, and the stage total must equal the sum of the objects alive in that stage.


```python
assert peak_report['peak_phase'] == 'backward_peak'
assert peak_report['candidate_peak_mib'] == sum(peak_report['contributions_mib'].values())
assert peak_report['phase_totals_mib']['after_load'] < peak_report['candidate_peak_mib']
print('✅ 生命周期重叠与候选峰值计算通过')

```

## Q3: What ledger quantities do micro-batch and gradient accumulation each change?

First distinguish two batches: `micro-batch` is the number of samples actually fed to the model at one time, and `effective batch` is the total number of samples after accumulating gradients for several rounds before updating the parameters once. Reducing the micro-batch lowers the single-pass activation peak; gradient accumulation can approximately preserve the effective batch, but it increases the number of forward / backward passes and does not reduce the long-lived occupancy of parameters, gradients, or optimizer state. Here only the ledger changes and the cost still to be measured are recorded; no strategy is chosen from them.

| Change | Ledger item affected | Cost still to be measured |
|:---|:---|:---|
| Reduce micro-batch | Single-pass activation peak | Smaller amount processed per pass |
| Increase gradient accumulation | Preserve effective batch with a smaller micro-batch | More forward / backward passes |
| Combine both | Maintain the target update scale with a lower activation peak | Throughput and training time need to be re-measured |


```python
configs = [
    {"name": "baseline", "micro_batch": 16, "accum_steps": 1, "world_size": 1},
    {"name": "accumulated", "micro_batch": 4, "accum_steps": 4, "world_size": 1},
]

seq_len, hidden_size, dtype_bytes = 64, 512, 2

def mib(nbytes):
    """Convert a byte count to a readable MiB value."""
    return nbytes / 1024**2

for cfg in configs:
    effective_batch = cfg["micro_batch"] * cfg["accum_steps"] * cfg["world_size"]
    activation_bytes = cfg["micro_batch"] * seq_len * hidden_size * dtype_bytes
    forward_backward_rounds = cfg["accum_steps"]
    cfg['effective_batch'] = effective_batch
    cfg['activation_peak_mib'] = mib(activation_bytes)
    cfg['forward_backward_rounds'] = forward_backward_rounds
    print(cfg)

```

## Q3 verification: With the same effective batch, do micro-batch and accumulation rounds trade off?

Reducing the micro-batch should lower the activation peak; to keep the same effective batch, the number of forward/backward rounds must increase.


```python
assert configs[0]['effective_batch'] == configs[1]['effective_batch']
assert configs[1]['activation_peak_mib'] < configs[0]['activation_peak_mib']
assert configs[1]['forward_backward_rounds'] > configs[0]['forward_backward_rounds']
print('✅ effective batch、activation 峰值和累计轮数关系通过')

```

## Q4: What object attributes do low precision and gradient checkpointing each change?

If the ledger shows that saved activations account for a high share, you can observe how the ledger changes after reducing the bytes per element or lowering the proportion of intermediate results saved. Low precision mainly changes the bytes per element; gradient checkpointing changes the saved proportion and recomputes part of the forward pass during backward. The "saved fraction" and "extra recomputation fraction" are given explicitly below, and only the direction is compared; they are not treated as fixed gains or selection conclusions for arbitrary models.


```python
batch_size, seq_len, hidden_size = 8, 32, 512

def activation_tradeoff(dtype_bytes, saved_fraction, extra_forward_fraction):
    """Compare activation bytes and recomputation cost under the given saved-fraction assumption."""
    if dtype_bytes <= 0 or not 0 < saved_fraction <= 1:
        raise ValueError('dtype_bytes 必须为正，saved_fraction 必须位于 (0, 1]')
    if not 0 <= extra_forward_fraction <= 1:
        raise ValueError('extra_forward_fraction 必须位于 [0, 1]')
    full_bytes = batch_size * seq_len * hidden_size * dtype_bytes
    return {
        'saved_activation_mib': full_bytes * saved_fraction / 1024**2,
        'extra_forward_fraction': extra_forward_fraction,
    }

rows = [
    ('fp32 / no checkpoint', activation_tradeoff(4, 1.0, 0.0)),
    ('bf16 / no checkpoint', activation_tradeoff(2, 1.0, 0.0)),
    ('fp32 / save 50%', activation_tradeoff(4, 0.5, 0.5)),
    ('bf16 / save 50%', activation_tradeoff(2, 0.5, 0.5)),
]

for name, result in rows:
    print(name, '->', result)

```

## Q4 verification: Do dtype and checkpoint change the byte count and the recomputation cost respectively?

Low precision should reduce the byte count at the same saved fraction; reducing the saved fraction should lower the saved activations while increasing extra forward computation.


```python
assert rows[1][1]['saved_activation_mib'] < rows[0][1]['saved_activation_mib']
assert rows[2][1]['saved_activation_mib'] < rows[0][1]['saved_activation_mib']
assert rows[3][1]['saved_activation_mib'] < rows[1][1]['saved_activation_mib']
assert rows[2][1]['extra_forward_fraction'] > rows[0][1]['extra_forward_fraction']
print('✅ 元素字节数、保存比例与重算代价关系通过')

```

## Q5: Why does memory keep growing across training rounds?

If the occupancy does not jump suddenly at some step but keeps growing after each training round, first check whether Tensors or losses that carry a computation graph are being appended to a list continuously. This is Python references extending the lifetime of the computation graph, and should not be broadly attributed to an allocator leak. `loss.item()` saves only a number, `detach()` keeps the Tensor but cuts the current computation graph, and `no_grad()` disables gradient tracking for a block of code; the three do not solve the same problem.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)
model = nn.Linear(4, 2)
x = torch.randn(8, 4)
y = torch.randint(0, 2, (8,))

loss_bucket = []
for _ in range(2):
    out = model(x)
    loss = F.cross_entropy(out, y)
    loss_bucket.append(loss)

scalar_bucket = []
for _ in range(2):
    out = model(x)
    loss = F.cross_entropy(out, y)
    scalar_bucket.append(loss.item())

with torch.no_grad():
    pred = model(x)

detached = model(x).detach()

print(f"keep tensor: type={type(loss_bucket[0]).__name__}, grad_fn={loss_bucket[0].grad_fn is not None}")
print(f"keep scalar: type={type(scalar_bucket[0]).__name__}, value={scalar_bucket[0]:.4f}")
print(f"no_grad output requires_grad={pred.requires_grad}")
print(f"detach output requires_grad={detached.requires_grad}, grad_fn={detached.grad_fn}")
print("store tensors only when you really need the graph")

```

## Q5 verification: Do saving a Tensor, saving a scalar, and cutting the computation graph produce different states?

Directly saving a loss Tensor keeps the `grad_fn`; saving a scalar, using `no_grad()`, or using `detach()` should not continue to hold the current computation graph.


```python
assert loss_bucket[0].grad_fn is not None
assert isinstance(scalar_bucket[0], float)
assert not pred.requires_grad
assert not detached.requires_grad and detached.grad_fn is None
print('✅ 计算图持有、标量保存、no_grad 和 detach 区分通过')

```

## Q6: How do you compare the theoretical ledger with allocator records?

First use the same workload and the same unit, then distinguish the occupancy at a single moment from the peak over the whole measurement interval. The difference between reserved and allocated at the same moment can help observe allocator retained space, but cannot by itself prove fragmentation; two separately recorded peaks may not occur at the same moment, so their difference cannot be treated as the cache amount at some instant. Below, a set of example values is constructed from the previous ledger to practice the reconciliation rules; it does not call the CUDA allocator.

| Comparison quantity | How to compute or read it | Next check |
|:---|:---|:---|
| Ledger difference | allocated − estimate in the same stage, keeping the sign | Check state, saved tensors, temporary buffers, and estimation assumptions |
| Same-moment retained difference | reserved − allocated | Check the cache and allocation history; do not conclude fragmentation from it |
| Device free space | total capacity − current whole-card occupancy | A real experiment must also include other processes and non-allocator occupancy |
| Process budget margin | given capacity budget − reserved | Judge whether this example's budget is tight; not equal to real device free memory |
| Interval peaks | Read the maximum of allocated and reserved separately | Keep the measurement range and peak moments; do not subtract them directly to attribute causes |


```python
import math


def reconcile_memory_record(theoretical_peak_mib, allocated_mib, reserved_mib, capacity_mib):
    """Reconcile a constructed snapshot of the same stage while keeping several signals to check."""
    values = (theoretical_peak_mib, allocated_mib, reserved_mib, capacity_mib)
    if any(not math.isfinite(value) or value < 0 for value in values) or capacity_mib <= 0:
        raise ValueError('内存数值必须有限、非负，容量预算必须为正')
    if not allocated_mib <= reserved_mib <= capacity_mib:
        raise ValueError('本例要求 allocated <= reserved <= 容量预算')
    estimate_difference_mib = allocated_mib - theoretical_peak_mib
    allocator_gap_mib = reserved_mib - allocated_mib
    budget_margin_mib = capacity_mib - reserved_mib
    signals = []
    if estimate_difference_mib > theoretical_peak_mib * 0.10:
        signals.append('inspect_estimate_difference')
    if allocator_gap_mib > capacity_mib * 0.10:
        signals.append('inspect_allocator_retained_space')
    if budget_margin_mib < capacity_mib * 0.10:
        signals.append('capacity_pressure')
    return {
        'estimate_difference_mib': estimate_difference_mib,
        'allocator_gap_mib': allocator_gap_mib,
        'budget_margin_mib': budget_margin_mib,
        'signals': signals,
        'alert_fraction': 0.10,
    }


# Construct the differences only on the same ledger above; all three alerts can appear at the same time.
theoretical_peak_mib = peak_report['candidate_peak_mib']
allocated_mib = theoretical_peak_mib + 3
reserved_mib = allocated_mib + 7
capacity_mib = reserved_mib + 1
runtime_record = reconcile_memory_record(
    theoretical_peak_mib, allocated_mib, reserved_mib, capacity_mib
)
dominant_object = max(peak_report['contributions_mib'], key=peak_report['contributions_mib'].get)
memory_evidence = {
    'evidence_type': 'memory', 'status': 'constructed_example',
    'source': 'same_ledger_with_assumed_runtime_snapshot',
    'unit': 'MiB',
    'candidate_peak_phase': peak_report['peak_phase'],
    'candidate_peak_mib': theoretical_peak_mib,
    'dominant_object': dominant_object,
    'dominant_objects_mib': peak_report['contributions_mib'],
    'allocated_mib': allocated_mib, 'reserved_mib': reserved_mib,
    'capacity_mib': capacity_mib,
    'runtime_reconciliation': runtime_record,
    'interpretation_limit': 'constructed_snapshot_not_gpu_measurement',
}
print('constructed snapshot:', runtime_record)
print('memory evidence:', memory_evidence)

```

## Q6 verification: Can reconciliation keep multiple signals at the same time?

It checks whether the ledger difference, allocator retained space, and budget pressure can appear at the same time; when below the estimate, keep the negative difference and do not attribute all unknown differences to fragmentation or leaks.


```python
assert runtime_record['estimate_difference_mib'] == 3
assert runtime_record['allocator_gap_mib'] == 7
assert runtime_record['budget_margin_mib'] == 1
assert set(runtime_record['signals']) == {
    'inspect_estimate_difference', 'inspect_allocator_retained_space', 'capacity_pressure'
}
below_estimate = reconcile_memory_record(20, 18, 19, 30)
assert below_estimate['estimate_difference_mib'] == -2
assert memory_evidence['status'] == 'constructed_example'
assert memory_evidence['unit'] == 'MiB'
assert memory_evidence['candidate_peak_phase'] == 'backward_peak'
assert memory_evidence['dominant_object'] in memory_evidence['dominant_objects_mib']
assert memory_evidence['interpretation_limit'] == 'constructed_snapshot_not_gpu_measurement'
for invalid in ((20, 22, 21, 30), (20, 22, 31, 30), (20, float('nan'), 25, 30)):
    try:
        reconcile_memory_record(*invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('无效快照不能参与对账')
print('✅ 构造快照、MiB 单位、多项告警与无效输入检查通过')

```

## Related Reading
**Overview:** After completing this section, you can consult the real allocator fields, and then bring the capacity evidence into the paired experiment of section 20; the specific training strategies are chosen by the corresponding mechanism and optimization topics.
- [PyTorch CUDA memory management documentation](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [Part 00 · 20 Performance Evidence and Experiment Design](./20_Performance_Evidence_and_Experiment_Design.md)
- [Part 02 · 76 Activation Checkpoint and Offload Comparison](../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md)

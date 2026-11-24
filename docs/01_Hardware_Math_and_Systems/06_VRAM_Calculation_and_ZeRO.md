# 06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化

**Difficulty:** Hard | **Environment:** CPU-first | **Tags:** `memory optimization`, `memory budget`, `ZeRO` | **Audience:** learners who need to build a training memory ledger

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

Training memory is composed of parameters, gradients, optimizer state, activations, communication buffers, and temporary workspace together. Different states have different replication patterns and lifetimes, so the peak cannot be estimated from the parameter count alone.

This section establishes the relationship "training state → per-GPU ledger → DDP replication → ZeRO sharding → capacity budget." When you finish, you should be able to estimate the theoretical footprint of each kind of state and judge which peak objects still need to be measured after multi-GPU sharding.

**Keywords:** `VRAM`, `ZeRO`, `AdamW`

![Concept relationships in this section](../public/01_Hardware_Math_and_Systems/06_vram_zero_map.svg)

---

## Prerequisite Reading

**Overview:** First review the byte count of each dtype, the parameter count, and the communication topology, then use the same parameter and optimizer assumptions to compare the per-GPU training state of DDP and ZeRO-1/2/3.

- [Part 01 · 01 Data Types and Precision](./01_Data_Types_and_Precision.md)
- [Part 01 · 02 Parameter Count and FLOPs](./02_LLM_Params_and_FLOPs.md)
- [Part 01 · 05 Communication Topology](./05_Communication_Topologies.md)

---
## Q1: Which training states does each GPU keep under DDP?


Under DDP, every GPU keeps the full model, the full gradients, and the full optimizer state. The example below uses BF16 + Adam with 7B parameters to put the bytes per parameter and the theoretical footprint into a single ledger table; the Adam state byte count is a teaching estimate, and it must be recomputed if an implementation also stores FP32 master weights, uses a different optimizer, or changes the parameter / gradient dtype.

| Training state | Bytes per parameter (BF16 + Adam example) | 7B theoretical footprint | Fully resident under DDP |
| --- | ---: | ---: | --- |
| Parameters | 2 bytes | 14 GB | Yes |
| Gradients | 2 bytes | 14 GB | Yes |
| Adam state | 12 bytes | 84 GB | Yes |
| Total | 16 bytes | 112 GB | Yes |

This table only covers parameters, gradients, and optimizer state; it does not include activations, communication buffers, temporary workspace, memory fragmentation, or the framework's reserved memory. The code validation should also check that the ledger changes after changing the dtype / optimizer, and that the three kinds of state are still recorded separately.


### Q1 mini check: Break down the DDP per-GPU training-state ledger

```python
def training_state_breakdown(num_params_b, model_dtype='fp16', optimizer='adam'):
    """Return the theoretical training-state ledger in decimal GB.

    The result only covers parameters, gradients, and optimizer state; it does
    not simulate activations, communication buffers, workspace, or allocator
    reserve.
    """
    if num_params_b < 0:
        raise ValueError('num_params_b must be non-negative')
    try:
        model_bytes = {'fp32': 4, 'fp16': 2, 'bf16': 2}[model_dtype]
        optimizer_bytes = {'adam': 12, 'sgd': 4}[optimizer]
    except KeyError as exc:
        raise ValueError('unsupported dtype or optimizer') from exc
    values = {
        'parameters_gb': num_params_b * model_bytes,
        'gradients_gb': num_params_b * model_bytes,
        'optimizer_state_gb': num_params_b * optimizer_bytes,
    }
    values['training_state_gb'] = sum(values.values())
    return values


def calculate_ddp_memory(num_params_b, model_dtype='fp16', optimizer='adam'):
    """Return the total DDP per-GPU training-state ledger; activations and workspace are excluded."""
    return training_state_breakdown(num_params_b, model_dtype, optimizer)['training_state_gb']

ledger = training_state_breakdown(7, 'bf16', 'adam')
print('7B parameters, BF16 + Adam theoretical training-state ledger (decimal GB):')
for name, value in ledger.items():
    print(f'  {name}: {value:.1f}')
```


```python
def test_calculate_ddp_memory():
    result = calculate_ddp_memory(7, 'fp16', 'adam')
    assert result == 112, f"错误：期望 112 GB，实际 {result} GB"

    result = calculate_ddp_memory(7, 'fp16', 'sgd')
    assert result == 56, f"错误：期望 56 GB，实际 {result} GB"

    bf16_ledger = training_state_breakdown(7, 'bf16', 'adam')
    assert bf16_ledger['parameters_gb'] == 14
    assert bf16_ledger['optimizer_state_gb'] == 84
    assert training_state_breakdown(7, 'fp32', 'adam')['training_state_gb'] == 140
    assert training_state_breakdown(7, 'bf16', 'sgd')['training_state_gb'] == 56
    try:
        training_state_breakdown(7, 'int8', 'adam')
    except ValueError:
        pass
    else:
        raise AssertionError('非法 dtype 应该被拒绝')
    print("✅ DDP 显存函数测试通过！")

test_calculate_ddp_memory()
```

## Q2: How does ZeRO share out the training state?


The core idea of ZeRO is to distribute the training state across multiple GPUs. First look at which states each stage shards, then use `calculate_zero_memory()` to compute the ideal per-GPU footprint; here we assume the state can be sharded evenly, and the table does not include activations, communication buffers, workspace, sharding granularity, or allocator reserve.

| ZeRO stage | Sharded state | State still fully resident | Ideal per-GPU ledger | Main cost |
| --- | --- | --- | --- | --- |
| DDP / ZeRO-0 | None | Parameters, gradients, optimizer state | Full state | High memory usage |
| ZeRO-1 | Optimizer state | Parameters, gradients | Parameters + gradients + optimizer shards | Optimizer state communication |
| ZeRO-2 | Optimizer state, gradients | Parameters | Parameters + gradient shards + optimizer shards | Gradient synchronization and reconstruction |
| ZeRO-3 | Parameters, gradients, optimizer state | No complete training state | All three kinds of state distributed across GPUs | Parameter reconstruction, communication, and higher scheduling complexity |

When choosing a stage, first be clear whether what you want to share out is optimizer state, gradients, or parameters; these three objects have different sharding scopes, and therefore different communication and implementation complexity.


### Q2 mini check: ZeRO memory calculation

```python
def calculate_zero_memory(num_params_b, zero_stage, num_gpus, model_dtype='fp16', optimizer='adam'):
    """Estimate the per-GPU training-state memory under ideal even sharding.

    Only parameters, gradients, and optimizer state are computed; activations,
    communication buffers, workspace, sharding granularity, and allocator
    reserve are excluded, so this cannot directly guarantee that OOM is avoided.
    """
    if num_params_b < 0 or num_gpus <= 0:
        raise ValueError('num_params_b must be non-negative and num_gpus must be positive')
    try:
        model_bytes = {'fp32': 4, 'fp16': 2, 'bf16': 2}[model_dtype]
        optimizer_bytes = {'adam': 12, 'sgd': 4}[optimizer]
    except KeyError as exc:
        raise ValueError('unsupported dtype or optimizer') from exc
    gradient_bytes = model_bytes

    if zero_stage == 0 or zero_stage == 'ddp':
        bytes_per_param = model_bytes + gradient_bytes + optimizer_bytes
    elif zero_stage == 1:
        bytes_per_param = model_bytes + gradient_bytes + optimizer_bytes / num_gpus
    elif zero_stage == 2:
        bytes_per_param = model_bytes + gradient_bytes / num_gpus + optimizer_bytes / num_gpus
    elif zero_stage == 3:
        bytes_per_param = (model_bytes + gradient_bytes + optimizer_bytes) / num_gpus
    else:
        raise ValueError('zero_stage must be 0/ddp, 1, 2 or 3')

    return num_params_b * bytes_per_param
```


```python
def test_calculate_zero_memory():
    result = calculate_zero_memory(7, 1, 8, 'fp16', 'adam')
    assert abs(result - 38.5) < 1e-9, f"错误：ZeRO-1 应该是 38.5 GB，实际 {result} GB"

    result = calculate_zero_memory(7, 2, 8, 'fp16', 'adam')
    assert abs(result - 26.25) < 1e-9, f"错误：ZeRO-2 应该是 26.25 GB，实际 {result} GB"

    result = calculate_zero_memory(7, 3, 8, 'fp16', 'adam')
    assert abs(result - 14) < 1e-9, f"错误：ZeRO-3 应该是 14 GB，实际 {result} GB"

    assert calculate_zero_memory(7, 'ddp', 8, 'fp16', 'adam') > calculate_zero_memory(7, 3, 8, 'fp16', 'adam')
    assert calculate_zero_memory(7, 1, 4, 'fp32', 'adam') > calculate_zero_memory(7, 1, 4, 'bf16', 'adam')
    try:
        calculate_zero_memory(7, 4, 8, 'fp16', 'adam')
    except ValueError:
        pass
    else:
        raise AssertionError('非法 ZeRO stage 应该被拒绝')
    print("✅ ZeRO 显存函数测试通过！")

test_calculate_zero_memory()
```

### Ledger tool: Derive the maximum model scale from capacity

This helper function converts memory capacity, ZeRO stage, and the reserve ratio into the maximum trainable parameter count, for use by the later strategy decision table.

```python
def max_trainable_params(gpu_memory_gb, num_gpus, zero_stage, overhead_ratio=0.2, model_dtype='fp16', optimizer='adam'):
    """Estimate parameter scale after a lumped safety reserve.

    ``overhead_ratio`` is a teaching knob for activations and communication;
    it is not a measured peak-memory fraction and cannot guarantee no OOM.
    """
    if gpu_memory_gb <= 0 or num_gpus <= 0:
        raise ValueError('gpu_memory_gb and num_gpus must be positive')
    if not 0 <= overhead_ratio < 1:
        raise ValueError('overhead_ratio must be in [0, 1)')
    available_memory = gpu_memory_gb * (1 - overhead_ratio)
    try:
        model_bytes = {'fp32': 4, 'fp16': 2, 'bf16': 2}[model_dtype]
        gradient_bytes = model_bytes
        optimizer_bytes = {'adam': 12, 'sgd': 4}[optimizer]
    except KeyError as exc:
        raise ValueError('unsupported dtype or optimizer') from exc

    if zero_stage == 0 or zero_stage == 'ddp':
        bytes_per_param = model_bytes + gradient_bytes + optimizer_bytes
    elif zero_stage == 1:
        bytes_per_param = model_bytes + gradient_bytes + optimizer_bytes / num_gpus
    elif zero_stage == 2:
        bytes_per_param = model_bytes + gradient_bytes / num_gpus + optimizer_bytes / num_gpus
    elif zero_stage == 3:
        bytes_per_param = (model_bytes + gradient_bytes + optimizer_bytes) / num_gpus
    else:
        raise ValueError('zero_stage must be 0/ddp, 1, 2 or 3')

    return available_memory / bytes_per_param
```


```python
def test_max_trainable_params():
    result = max_trainable_params(80, 8, 'ddp', 0.2)
    assert abs(result - 4) < 1e-9, f"错误：DDP 应该最多训练 4B，实际 {result}B"

    result = max_trainable_params(80, 8, 3, 0.2)
    assert abs(result - 32) < 1e-9, f"错误：ZeRO-3 应该最多训练 32B，实际 {result}B"

    assert max_trainable_params(80, 8, 3, 0.0) > max_trainable_params(80, 8, 3, 0.2)
    print("✅ 最大模型反推函数测试通过！")

test_max_trainable_params()
```

## Q3: How do you turn the ZeRO ledger into a capacity candidate table and distinguish the theoretical peak?

**Question:** If you only have 8 GPUs with a nominal 80GB each, and you use a 20% teaching reserve ratio to approximate activation and communication overhead, how do you put DDP, ZeRO-1, ZeRO-2, and ZeRO-3 into the same decision table?

Please put the four state layouts into the same table to compare the maximum trainable model scale, and record the communication cost and the qualitative risk of implementation complexity; you may call the earlier ledger functions to complete the scale derivation. What is generated here is capacity candidate evidence; the full parallel strategy screening is completed in section 26.


Put DDP, ZeRO-1, ZeRO-2, and ZeRO-3 into the same table and compare the per-GPU training state, the relative saving, the parameter scale that fits, as well as the communication cost and engineering complexity. The "low / medium / high" here are only first-round screening labels: the higher the ZeRO stage, the more state sharding, reconstruction, and synchronization is usually required. The memory and model scale in the table are theoretical ledger results and are not equal to the real runnable training limit; in the end, the same model, dtype, optimizer, batch, seq_len, and checkpoint configuration should be brought into the GPU experiments of 73–76 to check peak allocated, reserved, step time, and OOM.



```python
gpu_memory = 80
num_gpus = 8
overhead_ratio = 0.2
def compare_zero_strategies(gpu_memory_gb, num_gpus, overhead_ratio=0.2, model_dtype='fp16', optimizer='adam'):
    """Generate the DDP/ZeRO strategy table and compare capacity gains with qualitative communication cost.

    The memory figures are still a theoretical ledger; communication_cost and
    engineering_cost are qualitative labels that help learners make a first-round
    screening, not multi-GPU benchmark results.
    """
    strategies = [('DDP', 'ddp'), ('ZeRO-1', 1), ('ZeRO-2', 2), ('ZeRO-3', 3)]
    tradeoffs = {
        'DDP': ('低', '低'),
        'ZeRO-1': ('中', '中'),
        'ZeRO-2': ('中高', '中高'),
        'ZeRO-3': ('高', '高'),
    }
    baseline = calculate_zero_memory(1, 'ddp', num_gpus, model_dtype, optimizer)
    rows = []
    for name, stage in strategies:
        per_gpu_state_gb = calculate_zero_memory(1, stage, num_gpus, model_dtype, optimizer)
        rows.append({
            'evidence_level': 'theoretical_ledger',
            'strategy': name,
            'per_gpu_state_gb_per_1b': round(per_gpu_state_gb, 3),
            'memory_saving_ratio': round(1 - per_gpu_state_gb / baseline, 3),
            'max_trainable_params_b': round(max_trainable_params(gpu_memory_gb, num_gpus, stage, overhead_ratio, model_dtype, optimizer), 3),
            'communication_cost': tradeoffs[name][0],
            'engineering_cost': tradeoffs[name][1],
        })
    return rows

strategies = compare_zero_strategies(gpu_memory, num_gpus, overhead_ratio)
assert strategies[0]['strategy'] == 'DDP'
assert strategies[-1]['memory_saving_ratio'] > strategies[0]['memory_saving_ratio']

print('8 x A100 80GB 的 ZeRO 理论账本决策表（FP16 + Adam，预留 20% 显存）：')
print('证据级别：theoretical_ledger；不是 GPU 实测峰值。')
print(f"{'策略':<10} {'单卡状态 GB/1B':>16} {'节省比例':>10} {'最大参数 B':>12} {'通信代价':>10} {'工程代价':>10}")
print('-' * 76)
for row in strategies:
    print(f"{row['strategy']:<10} {row['per_gpu_state_gb_per_1b']:>16.3f} {row['memory_saving_ratio']:>10.3f} {row['max_trainable_params_b']:>12.3f} {row['communication_cost']:>10} {row['engineering_cost']:>10}")


```

---
## Related Reading
This section connects on to the runtime management of ZeRO/FSDP and the state sharding mechanisms of Part 02.
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/): see how the different ZeRO stages shard optimizer state, gradients, and parameters.
- [PyTorch Fully Sharded Data Parallel](https://pytorch.org/docs/stable/fsdp.html): learn about FSDP's parameter sharding, communication, and runtime management.
- [27. ZeRO Optimizer Sim](../02_PyTorch_Algorithms/27_ZeRO_Optimizer_Sim.md)

---
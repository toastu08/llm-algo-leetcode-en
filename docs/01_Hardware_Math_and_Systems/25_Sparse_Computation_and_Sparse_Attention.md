# 25. Sparse Computation and Sparse Attention | 稀疏计算与稀疏注意力

**Difficulty:** Hard | **Environment:** CPU-first | **Tags:** `model structure`, `Sparse`, `Sparse Attention` | **Audience:** learners who need to judge whether a sparse structure can be converted into a real gain

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/25_Sparse_Computation_and_Sparse_Attention.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

Sparsity can be converted into capacity, compute, or memory-access gains only when it changes the data structure and the execution path at the same time. The sparsity patterns of weights, activations, and attention connections have different skippable objects and execution constraints.
This section establishes the relationship "sparse object → structural granularity → retained connections → execution gain". After completing it, you should be able to compute the sparsity rate and theoretical workload, and judge whether random, row-column, or block sparsity has the hardware execution conditions.

**Keywords:** sparsity rate, structural granularity, density

![Concept relationships in this section](../public/01_Hardware_Math_and_Systems/25_sparse_execution_map.svg)



## Prerequisite Reading

**Overview:** First understand the memory objects of Attention and the regular computation of the Tensor Core, then judge whether a sparsity pattern can be mapped to an execution path.

- [04. Attention Variants and Memory Optimization](./04_Attention_Memory_Optimization.md)
- [23. TensorCore Deep Dive](./23_TensorCore_Deep_Dive.md)

## Q1: What are the sparse object, pattern, and density respectively?


The first task of sparsity is to state the object, pattern, and density clearly: the sparse object can be weights, activations, or Attention connections; the sparsity pattern describes how the retained elements are distributed; density represents the proportion of elements still retained.

If you only set many positions in the matrix to zero while the kernel still scans, broadcasts, and accumulates in a dense way, the zeros only change how the data is stored, and the computation and transfer may not really decrease.

A drop in density does not by itself mean the computation and transfer have already decreased; the structural judgment later will continue to check whether these elements can be skipped by the execution path.

### Q1 Mini Check: Sparsity does not bring automatic speedup

First judge whether the sparsity can actually be used by the execution path.

```python
def sparsity_definition_report(target='weights', density=0.5, structure='block'):
    """Organize the sparse object, pattern, and density without judging the execution candidate path."""
    if target not in {'weights', 'activation', 'attention'}:
        raise ValueError(f'未知稀疏对象: {target}')
    if not 0 <= density <= 1:
        raise ValueError('density 必须位于 [0, 1]')
    if structure not in {'block', 'row', 'column', 'random'}:
        raise ValueError(f'未知稀疏结构: {structure}')
    return {
        'target': target,
        'density': density,
        'zero_ratio': round(1 - density, 2),
        'structure': structure,
    }

for case in [('weights', 0.1, 'block'), ('attention', 0.5, 'block'), ('activation', 0.5, 'random')]:
    print(case, '->', sparsity_definition_report(*case))
assert sparsity_definition_report('weights', 0.5, 'block')['zero_ratio'] == 0.5
assert sparsity_definition_report('attention', 0.5, 'random')['target'] == 'attention'
try:
    sparsity_definition_report('unknown', 0.5, 'block')
except ValueError:
    print('✅ 稀疏对象校验通过')
else:
    raise AssertionError('未知稀疏对象应报错')

```

## Q2: Why is structured sparsity easier to deploy than unstructured sparsity?


Structured sparsity fixes the sparsity pattern into a shape that hardware can recognize and optimize more easily, so it is easier to map to tiles, warps, or sparse MMA.

Although unstructured sparsity is flexible, it usually requires extra indices, branches, and irregular memory access; if the kernel has no corresponding support, making the data sparse does not automatically reduce the executed work.

The comparison table below puts the pattern, execution path, and cost on the same footing.

| Sparsity pattern | Structural characteristics | Execution path | Main cost |
| --- | --- | --- | --- |
| Unstructured sparsity | Zero positions are random | Requires indices and irregular memory access | Branches, indices, memory-access fragmentation |
| Row / column structured sparsity | Pruned by row or column | Easier to map to regular access | Stronger structural constraints |
| Block sparsity | Retain or prune by block | Easy to map to tiles / warps | Granularity may affect quality |
| Hardware-constrained structure | Meets a specific sparse MMA shape | Can use a dedicated path | Depends on hardware and backend |

### Q2 Mini Check: Why regular shapes are easier to optimize

The stronger the regularity, the easier it is to map to regular tile, warp, or sparse MMA computation paths; this section first uses CPU code to check this structural relationship.

```python
def structured_path_report(shape='block', hardware='mma'):
    """Report how the sparse structure matches the hardware path without producing a fixed performance score."""
    if shape not in {'block', 'row', 'column', 'random'}:
        raise ValueError(f'未知稀疏形状: {shape}')
    if hardware not in {'mma', 'tensorcore', 'cpu'}:
        raise ValueError(f'未知硬件路径: {hardware}')
    regular = shape in {'block', 'row', 'column'}
    hardware_path = hardware in {'mma', 'tensorcore'}
    return {
        'regular_structure': regular,
        'hardware_path_available': hardware_path,
        'execution_candidate': bool(regular and hardware_path),
    }

cases = [('block', 'mma'), ('row', 'mma'), ('random', 'mma'), ('block', 'cpu')]
for shape, hw in cases:
    print((shape, hw), '->', structured_path_report(shape, hw))
assert structured_path_report('block', 'mma')['execution_candidate']
assert not structured_path_report('random', 'mma')['execution_candidate']

```

## Q3: How do the gains of weight, Attention, and MLP sparsity differ?


The three kinds of sparsity first change different objects:

| Sparse object | Main change | Cost that may be saved | Additional risk |
| --- | --- | --- | --- |
| Weight sparsity | Parameter matrix and weight loading | Weight storage, matrix multiplication workload | Sparse weights need a matching matrix kernel |
| Attention sparsity | token-token connections or score computation | Long-sequence compute volume, some intermediate results | Dynamic indexing and irregular memory access |
| MLP sparsity | FFN activations or channels | Activation path, matrix multiplication workload | Routing and load imbalance |

The code below continues to organize the object, density, structure, and theoretical execution conditions into a report; it does not produce a real speedup ratio.

### Q3 Mini Check: Whether sparsity can land on execution

First distinguish which kind of cost the sparse object changes, then check whether the structure meets the conditions to land on an execution path.

```python
def sparsity_execution_ledger(target='attention', density=0.3, structure='block'):
    """Organize the sparse object, main cost, and execution candidate conditions.

    The return value is a mechanism judgment table, not a speedup prediction; this section only verifies the object, structure, and theoretical cost.
    """
    profiles = {
        'weights': {
            'primary_cost': 'weight_storage_and_matmul',
            'possible_saving': 'storage_and_compute',
            'extra_risk': 'kernel_mapping',
        },
        'attention': {
            'primary_cost': 'token_token_score',
            'possible_saving': 'long_context_compute',
            'extra_risk': 'dynamic_indexing_and_irregular_access',
        },
        'mlp': {
            'primary_cost': 'ffn_activation_and_matmul',
            'possible_saving': 'activation_path_and_compute',
            'extra_risk': 'routing_and_load_imbalance',
        },
    }
    if target not in profiles:
        raise ValueError(f'未知稀疏对象: {target}')
    if not 0 <= density <= 1:
        raise ValueError('density 必须位于 [0, 1]')
    if structure not in {'block', 'row', 'column', 'random'}:
        raise ValueError(f'未知稀疏结构: {structure}')
    regular = structure in {'block', 'row', 'column'}
    return {
        'target': target,
        'zero_ratio': round(1 - density, 2),
        **profiles[target],
        'regular_structure': regular,
        'theoretical_path_condition': regular and density < 1,
    }

for case in [('weights', 0.25, 'block'), ('attention', 0.25, 'block'), ('mlp', 0.25, 'row')]:
    print(case, '->', sparsity_execution_ledger(*case))
assert sparsity_execution_ledger('weights', 0.25, 'block')['possible_saving'] == 'storage_and_compute'
assert sparsity_execution_ledger('attention', 0.25, 'random')['regular_structure'] is False
try:
    sparsity_execution_ledger('unknown', 0.25, 'block')
except ValueError:
    print('✅ 稀疏对象校验通过')
else:
    raise AssertionError('未知稀疏对象应报错')

```

## Q4: Why may the theoretical sparsity rate not be converted into a lower compute volume?


Q4 must turn "theoretically sparser" into a checkable comparison: fix the input, density, sparsity pattern, and computation rule, and record the retained connection count, theoretical compute volume, and output difference of dense / sparse respectively.

If sparsity only changes the storage format without reducing the actual computation, the theoretical compute volume will not automatically turn into a runtime gain. The code below first organizes the risk notes and comparison metrics that can be verified on the CPU.

The mechanism experiment can be organized according to the comparison table below: first fix the input conditions, then change only the sparse mask.

| Comparison item | Kept unchanged | Only changed condition | To record |
| --- | --- | --- | --- |
| dense baseline | Input, dtype, shape, softmax rule | Fully connected mask | Output, theoretical compute volume |
| sparse candidate | Input, dtype, shape, softmax rule | sparse mask / density | Output, retained connection count |
| quality check | Evaluation input and error convention | Whether the sparse mask is used | Output difference, numerical stability |

### Q4 Mini Check: How to design a dense / sparse comparison

First list the fixed input, dense baseline, sparse mask, and the CPU metrics to record.

```python
def sparse_failure_reason(density, structure='block'):
    """List the risk notes of the sparse-mechanism comparison without outputting hardware performance conclusions."""
    if not 0 <= density <= 1:
        raise ValueError('density 必须位于 [0, 1]')
    if structure not in {'block', 'row', 'column', 'random'}:
        raise ValueError(f'未知稀疏结构: {structure}')
    reasons = []
    if density >= 0.7:
        reasons.append('density_high_risk_for_this_demo')
    if structure == 'random':
        reasons.append('irregular_pattern')
    if structure in ['row', 'column'] and density < 0.3:
        reasons.append('weak_structure_gain')
    return {
        'usable_for_demo': len(reasons) == 0,
        'reasons': reasons or ['path_ready'],
        'comparison_metrics': ['retained_connections', 'theoretical_ops', 'max_abs_diff'],
        'requires_dense_baseline': True,
    }

cases = [(0.1, 'block'), (0.5, 'random'), (0.8, 'block'), (0.3, 'row')]
for case in cases:
    print(case, '->', sparse_failure_reason(*case))
assert 'density_high_risk_for_this_demo' in sparse_failure_reason(0.8, 'block')['reasons']
assert sparse_failure_reason(0.3, 'block')['usable_for_demo']

```

## Q5: How does Sparse Attention reduce token-token connections?


Using a boolean mask to retain connections within a local window makes it possible to clearly observe the difference between dense attention and sparse attention on the CPU: the input and softmax rule remain unchanged, and only the connections allowed to participate in the computation change. What is verified here is the mask, retained connection count, theoretical compute volume, and output difference; the theoretical reduction is not written as a runtime speedup.

### Q5 Mini Check: Verify the connections first, then observe the output

The implementation below constructs a sparse mask using a local window and compares it with the dense mask.

```python
import math
import torch

def build_local_attention_mask(seq_len, window):
    """Construct a local-window mask including the position itself, returning [1, 1, L, L]."""
    if seq_len <= 0 or window < 0:
        raise ValueError('seq_len 必须为正，window 不能为负')
    positions = torch.arange(seq_len)
    return (positions[:, None] - positions[None, :]).abs().le(window)[None, None]

def masked_attention_cpu(q, k, v, mask):
    """Compute scaled dot-product attention on the CPU according to the mask."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    return weights @ v

torch.manual_seed(0)
batch, heads, seq_len, head_dim = 1, 2, 8, 4
q = torch.randn(batch, heads, seq_len, head_dim)
k = torch.randn_like(q)
v = torch.randn_like(q)
dense_mask = torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool)
sparse_mask = build_local_attention_mask(seq_len, window=2)
dense_out = masked_attention_cpu(q, k, v, dense_mask)
sparse_out = masked_attention_cpu(q, k, v, sparse_mask)
retained = int(sparse_mask.sum().item())
dense_connections = seq_len * seq_len
report = {
    'dense_connections': dense_connections,
    'sparse_connections': retained,
    'theoretical_connection_ratio': round(retained / dense_connections, 4),
    'max_abs_diff': round((dense_out - sparse_out).abs().max().item(), 6),
}
print(report)
assert dense_out.shape == sparse_out.shape == (batch, heads, seq_len, head_dim)
assert torch.isfinite(sparse_out).all()
assert 0 < retained < dense_connections

```

## Related Reading

**Overview:** Continue into the sparse Attention mechanism, regular block execution, and mature component implementations.

- [Part 02 · 44 Local, Sliding and Sparse Attention](../02_PyTorch_Algorithms/44_Local_Sliding_and_Sparse_Attention.md)
- [18. Triton Block Model](./18_Triton_Block_Model.md)
- [xFormers | Meta's open-source efficient Transformer components](https://github.com/facebookresearch/xformers)
---
# 19. Operator Fusion Introduction | 算子融合导论

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `compiler systems`, `operator fusion`, `performance optimization` | **Audience:** compiler and systems beginners

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

When adding a bias and executing an activation after a matrix multiply, separate compute kernels may repeatedly write out and read back intermediate results. Fusion places some of these steps in the same compute kernel so that intermediate values are consumed directly by the next step, while it may also increase register demand and layout conversions.

This section compares the intermediate reads/writes and launch counts before and after fusion, then checks the semantic, shape, and on-chip resource conditions. After completing it, you should be able to compute which reads/writes were saved and point out the costs that still need to be measured; the implementation and timing of real fused compute kernels move on to the operator optimization topic.

**Keywords:** operator fusion, intermediate tensor, memory traffic, layout conversion

![Data path and verification gates before and after operator fusion](../public/01_Hardware_Math_and_Systems/19_operator_fusion_data_path.svg)

---
## Prerequisite Reading

**Overview:** First understand the objects of graph-level optimization, then check how fusion changes intermediate tensors and on-chip resources.

- [09. AI Compilers and Graph Optimization](./09_AI_Compilers_and_Graph_Optimization.md)
- [18. Triton Block Model](./18_Triton_Block_Model.md)

## Q1: How do intermediate tensors increase logical reads/writes?

Every group boundary in a serial operator chain may write out an intermediate tensor, which is then read by the next group. Under the assumption that boundary tensors are of equal size and each group corresponds to one kernel function, the number of boundaries can be used to compute the read/write bytes and launch counts. Fusion removes internal boundaries, not the model input and final output; a reduction in logical reads/writes also does not mean that actual HBM traffic or elapsed time drops by the same proportion.
### Q1 Mechanism Calculator: Compare the read/write bytes of group boundaries

```python
def fusion_traffic(num_ops, tensor_bytes, fused_groups):
    """Estimate the bytes written back and re-read for intermediate tensors caused by operator group boundaries."""
    if (type(num_ops) is not int or type(fused_groups) is not int
            or num_ops < 1 or not 1 <= fused_groups <= num_ops):
        raise ValueError('需要满足 1 <= fused_groups <= num_ops')
    if type(tensor_bytes) is not int or tensor_bytes <= 0:
        raise ValueError('tensor_bytes 必须大于 0')
    materialized_boundaries = fused_groups - 1
    intermediate_bytes = 2 * materialized_boundaries * tensor_bytes
    return {
        'launches': fused_groups,
        'materialized_boundaries': materialized_boundaries,
        'intermediate_bytes': intermediate_bytes,
    }

tensor_bytes = 64 * 1024**2
unfused = fusion_traffic(num_ops=4, tensor_bytes=tensor_bytes, fused_groups=4)
partially_fused = fusion_traffic(num_ops=4, tensor_bytes=tensor_bytes, fused_groups=2)
fully_fused = fusion_traffic(num_ops=4, tensor_bytes=tensor_bytes, fused_groups=1)
print({'unfused': unfused, 'partial': partially_fused, 'full': fully_fused})

assert unfused['intermediate_bytes'] == 6 * tensor_bytes
assert fully_fused['intermediate_bytes'] == 0

```

## Q2: Why does operator fusion improve data locality?


Fusion combines several steps that were originally separate into a more continuous execution chain.

The result of doing so is:
- intermediate results are more likely to stay on chip;
- the same batch of data is more likely to be reused consecutively;
- the gap between data movement and computation is shorter.

Once data locality improves, the hardware can more easily reuse data that is already present, instead of going back to HBM to look for it again at every step. This is why fusion is not just "a few fewer functions" but directly changes the data flow path.

### Q2 Mechanism Calculator: Check the on-chip budget and the logical reuse amount

Assume a data tile can reside on chip throughout the entire consumption stage, and compute the reduction in repeated reads. The capacity check does not include other temporary values, allocation granularity, or synchronization; this is only a local budget, not proof of actual successful fusion.


```python
def onchip_reuse(tile_bytes, reuse_count, onchip_budget_bytes):
    """Compare against the given local budget and estimate the reduced logical read volume under the full-residency assumption."""
    if any(type(value) is not int or value <= 0
           for value in (tile_bytes, reuse_count, onchip_budget_bytes)):
        raise ValueError('tile、复用次数和片上预算必须为正数')
    fits = tile_bytes <= onchip_budget_bytes
    avoided_read_bytes = tile_bytes * (reuse_count - 1) if fits else 0
    return {
        'fits_onchip': fits,
        'avoided_read_bytes': avoided_read_bytes,
        'budget_utilization': tile_bytes / onchip_budget_bytes,
    }

plans = {
    'small_tile': onchip_reuse(32 * 1024, 3, 96 * 1024),
    'large_tile': onchip_reuse(128 * 1024, 3, 96 * 1024),
}
for name, result in plans.items():
    print(name, '->', result)

assert plans['small_tile']['avoided_read_bytes'] == 64 * 1024
assert plans['large_tile']['avoided_read_bytes'] == 0

```

## Q3: Which conditions must be checked first, and which costs need to be measured?

Before fusion, check semantics, shapes, on-chip resources, and backend support, then record layout conversion, compilation, and execution costs separately. Extra layout conversions may offset part of the benefit, but they are not sufficient reason to declare the plan infeasible.

The boolean values below represent given check results and do not replace numerical comparison, compilation, or execution. As long as the premises are met, the layout bytes are retained for later comparison; a candidate is not excluded directly because the value is greater than zero.

### Q3 Contrast Verification: Layout conversion has a cost but does not automatically prevent fusion

```python
def fusion_decision(semantics_match, shape_supported, onchip_fit,
                    backend_lowered, reorder_bytes=0):
    """Record the given check results and layout cost separately; do not perform compilation or numerical verification."""
    checks = {
        'semantics_match': semantics_match,
        'shape_supported': shape_supported,
        'onchip_fit': onchip_fit,
        'backend_lowered': backend_lowered,
    }
    if any(type(value) is not bool for value in checks.values()):
        raise ValueError('检查结果必须是布尔值')
    if type(reorder_bytes) is not int or reorder_bytes < 0:
        raise ValueError('布局转换字节数必须是非负整数')
    failures = [name for name, passed in checks.items() if not passed]
    return {
        'stated_conditions_met': not failures,
        'failures': failures,
        'extra_reorder_bytes': reorder_bytes,
    }


cases = {
    'ready': (True, True, True, True, 0),
    'shape_miss': (True, False, True, True, 0),
    'reorder_cost': (True, True, True, True, 8 * 1024**2),
}
for name, config in cases.items():
    print(name, '->', fusion_decision(*config))
assert fusion_decision(*cases['ready'])['stated_conditions_met']
assert fusion_decision(*cases['reorder_cost'])['stated_conditions_met']
assert fusion_decision(*cases['shape_miss'])['failures'] == ['shape_supported']

```

## Related Reading

**Overview:** Continuing operator fusion into performance tuning, reductions, and attention kernels will make it more intuitive to see how intermediate tensors are saved.

- [09. Triton Autotuning and Benchmarking](../03_Triton_Kernels/09_Triton_Autotune_and_Profiling.md)
- [07. Triton Advanced: Cross-thread Reduction and Numerical Stability (Safe Softmax)](../03_Triton_Kernels/07_Triton_Fused_Softmax.md)
- [11. Triton Flash Attention](../03_Triton_Kernels/11_Triton_Flash_Attention.md)
---
# 30. Dynamic Shape Handling | 动态 Shape 处理

**Difficulty:** Medium-Hard | **Environment:** CPU-first | **Tags:** `systems engineering`, `dynamic shape`, `inference serving` | **Audience:** compiler and systems beginners

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/30_Dynamic_Shape_Handling.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

Even if the computation itself is unchanged, a change in the input batch or sequence length may prevent the original execution path from being reused. Padding inputs to a fixed shape can reduce the number of shape variants, but introduces wasted computation; when bucket boundaries are too wide, padding may even exceed padding directly to the maximum length of the current batch.

This section uses shape and length sequences to compare cache keys, padding amounts, and path switching, then distinguishes guard failures, recompilation, and graph breaks. Operator optimization uses these distinctions to check the reuse conditions of an implementation; inference optimization then puts padding and path selection back into the batching and waiting of actual requests.

**Keywords:** dynamic shape, guard, recompilation, graph break, bucketing

![Dynamic shapes, bucketing, and execution paths](../public/01_Hardware_Math_and_Systems/30_dynamic_shape_specialization.svg)

---
## Prerequisite Reading

First distinguish graph capture, graph transformation, and the target implementation, then check how input changes affect existing versions.

- [09 AI Compilers and Graph Optimization](./09_AI_Compilers_and_Graph_Optimization.md): programs, computation graphs, and execution paths.

## Q1: Which compiled versions can still be reused after the input changes

First consider the simplest cache model: the key consists only of the full shape, a shape misses the first time it appears and hits on subsequent appearances, and the cache never evicts. The more distinct input shapes there are, the lower the hit ratio of this model may be. The calculator below counts only these exact-shape keys and does not call a compiler.

Real compiled versions usually also have guards, which check applicability conditions such as shapes, strides, types, and related program state. Shapes can be expressed symbolically with constraints instead of generating a version for every length. For example, suppose a version allows sequence lengths between 128 and 256 for contiguous FP32 inputs; then both 128 and 192 satisfy this set of conditions; switching to a non-contiguous layout, even if the length is still 128, may make it no longer applicable. The range here is a constructed assumption, not an actual guard collected by a local compiler.

A guard failure means the current version is not applicable; it does not mean device code must necessarily be regenerated: the system may find another cached applicable version. Only when no version is available might it re-capture or recompile, and the underlying compilation may still reuse existing caches; cases that exceed limits or are unsupported may also fall back or raise an error, depending on the configuration. A graph break, on the other hand, happens during capture and means the current configuration cannot continue to include some part of the program in the same capture region; this is a different problem from one input not satisfying the conditions of an old version.

| Observation | What happened | What cannot be concluded yet |
| --- | --- | --- |
| Guard failure | The applicability conditions of an existing version are not met | It does not mean all cached versions are unavailable, or that code must be regenerated |
| Recompilation | Another version is prepared for the changed program or input conditions | It does not mean all underlying artifacts are compiled cold, nor does it guarantee being faster |
| Graph break | Capture cannot continue across some part of the program | It does not equal a shape-cache miss; modes that allow segmentation and those requiring a full graph handle this differently |

### Q1 Mechanism Calculator: Count the repetition ratio of exact shape keys


```python
def shape_reuse_report(shapes):
    """Simulate a cold cache keyed only by shape with no eviction, without calling a compiler."""
    if not shapes or any(
        not isinstance(shape, (tuple, list)) or not shape
        or any(type(dim) is not int or dim <= 0 for dim in shape)
        for shape in shapes
    ):
        raise ValueError('形状必须是非空的正整数维度序列')
    normalized = [tuple(shape) for shape in shapes]
    keys = len(set(normalized))
    hits = len(normalized) - keys
    return {
        'requests': len(normalized),
        'distinct_shape_keys': keys,
        'assumed_exact_shape_hits': hits,
        'exact_shape_hit_ratio': hits / len(normalized),
    }


fixed = shape_reuse_report([(8, 512)] * 4)
dynamic = shape_reuse_report([(8, 128), (8, 512), (8, 128), (8, 2048)])
assert fixed['exact_shape_hit_ratio'] == 0.75
assert dynamic['exact_shape_hit_ratio'] == 0.25
print('固定形状：', fixed)
print('变化形状：', dynamic)

```

## Q2: Which padding can bucketing reduce, and when does it increase padding instead?

Padding turns inputs of different lengths into regular tensors; bucketing groups similar lengths together and pads them to specified boundaries. Bucket boundaries close to the input may reduce wasted tokens, but when boundaries are too wide, it may also be more wasteful than padding directly to the maximum length of the current batch.

The following fixes the request set and computes only the padded tokens; it does not simulate request arrival, waiting for batching, or batch size limits, nor does it treat the padding ratio as a throughput gain.

### Q2 Contrast Verification: Check inputs for which bucketing is favorable and unfavorable


```python
def padding_overhead(lengths, bucket_boundaries=None):
    """Compute the number of padded tokens for a single large batch or for bucketing by length boundaries."""
    if not lengths or any(type(length) is not int or length <= 0 for length in lengths):
        raise ValueError('lengths must be positive and non-empty')
    if bucket_boundaries is None:
        targets = [max(lengths)] * len(lengths)
    else:
        if any(type(boundary) is not int or boundary <= 0
               for boundary in bucket_boundaries):
            raise ValueError('桶边界必须是正整数')
        boundaries = sorted(set(bucket_boundaries))
        if not boundaries or boundaries[-1] < max(lengths) or any(boundary <= 0 for boundary in boundaries):
            raise ValueError('bucket boundaries must cover every length')
        targets = [next(boundary for boundary in boundaries if length <= boundary) for length in lengths]
    wasted_tokens = sum(target - length for target, length in zip(targets, lengths))
    return {'useful_tokens': sum(lengths), 'wasted_tokens': wasted_tokens, 'total_tokens': sum(targets), 'padding_ratio': wasted_tokens / sum(targets)}

lengths = [64, 128, 256, 520, 800]
single_batch = padding_overhead(lengths)
bucketed = padding_overhead(lengths, [128, 256, 512, 1024])
assert bucketed['wasted_tokens'] < single_batch['wasted_tokens']
assert padding_overhead([65, 66], [128])['wasted_tokens'] > padding_overhead([65, 66])['wasted_tokens']
print('single batch:', single_batch)
print('bucketed:', bucketed)

```

## Q3: Can the number of path switches explain latency fluctuations

Given length thresholds, you can count which branches requests enter and how many times adjacent requests switch branches. But selecting a branch at runtime and generating a version at compile time are two separate actions: if both branches already have available implementations, frequent switching does not require recompilation; always entering the same branch may still trigger new applicability-condition checks due to changes in type or stride.

A graph break also cannot be inferred from the number of branches. When segmentation is allowed, captureable regions may be compiled separately while other parts continue normal execution; `fullgraph=True` requires that no graph break occurs during this capture and raises an error when an unsupported capture is encountered, but a successfully captured graph may still contain multiple device kernels and library calls. It is not a guarantee of "one GPU kernel".

The following only counts the path changes under the given thresholds. To explain real fluctuations, you need to line up the input metadata, guard/recompilation diagnostics, actual execution, and timing samples of the same call; first-time preparation and warmup should also be separated from steady-state elapsed time.

### Q3 Mechanism Calculator: Count branch changes under the given thresholds


```python
def path_stability(lengths, boundaries=(256, 1024)):
    """Count the types of runtime paths and the number of path switches between adjacent requests."""
    if not lengths or any(type(length) is not int or length <= 0 for length in lengths):
        raise ValueError('lengths must be positive and non-empty')
    if (len(boundaries) != 2
            or any(type(value) is not int for value in boundaries)
            or not 0 < boundaries[0] < boundaries[1]):
        raise ValueError('boundaries must be two increasing positive values')
    def route(length):
        return 'short' if length < boundaries[0] else 'long' if length < boundaries[1] else 'very_long'
    paths = [route(length) for length in lengths]
    transitions = sum(left != right for left, right in zip(paths, paths[1:]))
    return {'paths': paths, 'unique_paths': len(set(paths)), 'path_transitions': transitions, 'stable': transitions == 0}

stable = path_stability([128, 128, 128])
mixed = path_stability([128, 512, 128, 2048])
assert stable['stable'] is True
assert mixed['unique_paths'] == 3 and mixed['path_transitions'] == 3
print('stable:', stable)
print('mixed:', mixed)

```

## Related Reading

When you need to trace a specific specialized implementation or fixed-shape replay, look at the generated artifacts and the runtime constraints respectively.

- [Part 03 · 18 Compilation Artifacts and Actual Execution](../03_Triton_Kernels/18_Triton_Compilation_Artifacts_and_Execution.md): compile-time constants, runtime arguments, and artifact reuse.
- [Part 04 · 11 CUDA Graph and JIT Compilation](../04_CUDA_and_System_Optimization/11_CUDA_Graph_and_JIT_Compile.md): addresses, dependencies, and input updates of a fixed workflow.

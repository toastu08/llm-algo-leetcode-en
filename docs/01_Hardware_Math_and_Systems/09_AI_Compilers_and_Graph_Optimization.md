# 09. AI Compilers and Graph Optimization | AI 编译器与计算图优化

**Difficulty:** Hard | **Environment:** CPU-first | **Tags:** `compiler systems`, `AI Compiler`, `Graph Optimization` | **Audience:** compiler and systems beginners

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/09_AI_Compilers_and_Graph_Optimization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

The same "matrix multiply → add bias → activation" can be executed by calling operators one by one, or handed to a compiler to rewrite. Both approaches must preserve the agreed computation semantics and numerical requirements, yet they may use different intermediate tensors, launch counts, and backend paths.

Obtaining the computation relationships from a program, rewriting those relationships, and generating a device implementation are different actions. This section separates them along this computation chain, then compares the reads/writes and launches that graph transformations change, and checks the backend's operator, precision, and workspace conditions. Operator optimization uses this to decide whether to keep leveraging existing compilation or to change the kernel implementation; actual speed is confirmed by execution and timing on the same input.

**Keywords:** AI compiler, computation graph, intermediate tensor, backend path

![Computation graph to backend execution path](../public/01_Hardware_Math_and_Systems/09_compiler_graph_lowering.svg)

---
## Prerequisite Reading

First understand how a program maps to device execution units from [1D: Execution Model, Asynchronous Scheduling, and Operator Mapping](./1D.md). This section does not require prior knowledge of a specific compilation framework.
## Q1: How are programs, computation graphs, and device code connected

"Matrix multiply → add bias → activation" is first a program. Graph capture identifies the tensor operations and dependencies within it and stores these relationships in an intermediate representation (IR); the matrix multiply node in the graph expresses what computation happens, without yet specifying the thread tiling, instructions, or which library function. Graph transformations can then combine operations, change layouts, or retain necessary intermediate results, and the backend then selects and generates a target implementation.

Taking PyTorch's `torch.compile` as an example, it is the user entry point that connects the compilation flow, not a single fused operator. TorchDynamo captures handleable computations and establishes applicability conditions, commonly using FX graphs to represent operation relationships; during training, AOTAutograd participates in processing the forward and backward graphs. The default backend TorchInductor performs subsequent processing on the graph and, on GPU, can generate Triton implementations or retain calls to existing compute libraries. It is entirely possible to have a combination where a matrix multiply still calls a library while the adjacent bias and activation produce a fused kernel.

Therefore, capturing a graph does not equal launching a single GPU kernel, and generating Triton code does not equal every operator in the program being executed by Triton. Hand-written Triton enters the compilation chain from a block-level algorithm description, while CUDA C++ goes through its own toolchain; CUDA Graph replay reuses the submission arrangement of operations and dependencies and cannot substitute for the above graph transformations and code generation.

| Stage | What it does to the current computation | Questions you can continue to ask |
| --- | --- | --- |
| Graph capture and applicability conditions | Obtain operation dependencies, record constraints on inputs and program state | Which code is captured, which inputs can be reused? |
| Graph transformation and backend generation | Choose fusion, layout, generate kernels or library calls | Do intermediate values still need to be written back, which implementations were actually used? |
| Launch and device execution | Call generated implementations and existing libraries following dependencies | Did the call actually happen, how long did the full computation take? |
## Q2: Which launches and intermediate-tensor reads/writes can graph transformations reduce?

When four serial operators are split into four groups of execution, the results between groups need to be written out and read by the next group; merging into fewer groups has the chance to reduce these boundaries and launch counts. If the transformation also needs to adjust the layout, it will add extra reads/writes, which should be accounted for separately from the reduction.

The following assumes the operators form a single chain, the boundary tensors are all the same size, and each group corresponds to one launch. The byte volumes are logical read/write estimates, and the launch times come from the given assumptions; they cannot be added directly, nor can they replace end-to-end timing.
### Q2 Mechanism Calculator: Record launch overhead and logical reads/writes separately

Fix the operator chain and tensor size, change only the grouping, then add the layout conversion bytes as a comparison.

```python
import math


def graph_cost(num_ops, fused_groups, intermediate_elements, bytes_per_element, layout_bytes=0, launch_overhead_us=5.0):
    """Estimate kernel launch and intermediate tensor reads/writes without adding different units into an arbitrary score."""
    if (type(num_ops) is not int or type(fused_groups) is not int
            or num_ops <= 0 or not 1 <= fused_groups <= num_ops):
        raise ValueError('fused_groups must be in [1, num_ops]')
    if (type(intermediate_elements) is not int or intermediate_elements < 0
            or type(bytes_per_element) is not int or bytes_per_element <= 0
            or type(layout_bytes) is not int or layout_bytes < 0
            or not math.isfinite(launch_overhead_us) or launch_overhead_us < 0):
        raise ValueError('元素数和布局字节数须为非负整数，元素字节数须为正整数，开销须为有限非负数')
    materialized_boundaries = fused_groups - 1
    intermediate_bytes = materialized_boundaries * intermediate_elements * bytes_per_element * 2
    return {
        'launches': fused_groups,
        'launch_overhead_us': fused_groups * launch_overhead_us,
        'materialized_boundaries': materialized_boundaries,
        'intermediate_traffic_bytes': intermediate_bytes,
        'layout_traffic_bytes': layout_bytes,
        'total_extra_traffic_bytes': intermediate_bytes + layout_bytes,
    }

naive = graph_cost(4, 4, 1_000_000, 2)
fused = graph_cost(4, 1, 1_000_000, 2, layout_bytes=1_000_000)
assert fused['launches'] < naive['launches']
assert fused['total_extra_traffic_bytes'] < naive['total_extra_traffic_bytes']
print('naive:', naive)
print('fused:', fused)
```

## Q3: Once the fusion conditions are met, why still record the cost separately?

Sharing intermediate results provides a fusion opportunity, but semantics, side effects, and resource budgets still need to be checked. Here "no side effects" is used as a conservative premise for the simplified example; it does not mean all operations with side effects cannot be legally fused.

Once these premises are met, compute the avoided intermediate reads/writes minus the added layout reads/writes. Having a net reduction in reads/writes does not guarantee being faster; having no net reduction does not mean it cannot be executed, since launch counts, computation, and resource usage still affect the elapsed time. The boolean input represents already-given check results; the function itself does not prove semantic equivalence.
### Q3 Contrast Verification: Record the premises and the read/write reduction separately

```python
def fusion_candidate(shared_bytes, layout_rewrite_bytes, no_side_effect,
                     register_bytes, register_budget_bytes):
    """Record the example premises and the logical read/write difference; do not treat a cost increase as an execution failure."""
    if any(type(value) is not int or value < 0
           for value in (shared_bytes, layout_rewrite_bytes, register_bytes)):
        raise ValueError('字节数必须是非负整数')
    if type(register_budget_bytes) is not int or register_budget_bytes <= 0:
        raise ValueError('寄存器预算必须是正整数字节数')
    if type(no_side_effect) is not bool:
        raise ValueError('副作用检查结果必须是布尔值')
    reasons = []
    if not no_side_effect:
        reasons.append('side_effect_requires_analysis')
    if register_bytes > register_budget_bytes:
        reasons.append('register_budget')
    net = shared_bytes - layout_rewrite_bytes
    return {
        'stated_conditions_met': not reasons,
        'net_logical_traffic_saved_bytes': net,
        'has_net_traffic_saving': net > 0,
        'reasons': reasons,
    }


ideal = fusion_candidate(8_000_000, 0, True, 48_000, 64_000)
layout_cost = fusion_candidate(8_000_000, 10_000_000, True, 48_000, 64_000)
side_effect = fusion_candidate(8_000_000, 0, False, 48_000, 64_000)
assert ideal['stated_conditions_met'] and ideal['has_net_traffic_saving']
assert layout_cost['stated_conditions_met']
assert layout_cost['net_logical_traffic_saved_bytes'] == -2_000_000
assert not side_effect['stated_conditions_met']
print(ideal)
print(layout_cost)
print(side_effect)
```

## Q4: Why does the same graph match different backend paths?

Target backends differ in supported operators, data types, shapes, and workspaces, so the same graph may need different transformations or fallback paths. Check the declared capabilities and budgets first, then compare the times of feasible candidates; actual deployment also needs to check devices, versions, and execution results.

The following uses fictitious backend capabilities and latency inputs to practice screening; it does not probe the local backend, nor does it represent the real speed of CPU or GPU. Workspaces uniformly use MiB; a candidate that fails the shape or memory requirements cannot be selected even if its assumed latency is lower.
### Q4 Constraint Screening: Check the given conditions first, then compare assumed latencies

```python
def choose_backend(requirements, candidates):
    """Screen backends by operators, dtype, dynamic shapes, and VRAM hard constraints."""
    if (not math.isfinite(requirements['workspace_limit_mib'])
            or requirements['workspace_limit_mib'] < 0
            or type(requirements['dynamic_shape']) is not bool):
        raise ValueError('工作区预算与动态形状条件不合法')
    for candidate in candidates.values():
        if (not math.isfinite(candidate['workspace_mib'])
                or candidate['workspace_mib'] < 0
                or not math.isfinite(candidate['estimated_latency_ms'])
                or candidate['estimated_latency_ms'] <= 0
                or type(candidate['dynamic_shape']) is not bool):
            raise ValueError('候选工作区、假设延迟与动态形状条件不合法')
    eligible = []
    rejected = {}
    for name, candidate in candidates.items():
        reasons = []
        if not requirements['ops'].issubset(candidate['ops']):
            reasons.append('missing_ops')
        if requirements['dtype'] not in candidate['dtypes']:
            reasons.append('unsupported_dtype')
        if requirements['dynamic_shape'] and not candidate['dynamic_shape']:
            reasons.append('dynamic_shape')
        if candidate['workspace_mib'] > requirements['workspace_limit_mib']:
            reasons.append('workspace')
        if reasons:
            rejected[name] = reasons
        else:
            eligible.append(name)
    selected = min(eligible, key=lambda name: candidates[name]['estimated_latency_ms']) if eligible else None
    return {'selected': selected, 'eligible': eligible, 'rejected': rejected}

requirements = {'ops': {'matmul', 'softmax'}, 'dtype': 'bf16', 'dynamic_shape': True, 'workspace_limit_mib': 512}
candidates = {
    'host_example': {'ops': {'matmul', 'softmax'}, 'dtypes': {'fp32', 'bf16'}, 'dynamic_shape': True, 'workspace_mib': 64, 'estimated_latency_ms': 20.0},
    'accelerator_example': {'ops': {'matmul', 'softmax'}, 'dtypes': {'fp16', 'bf16'}, 'dynamic_shape': True, 'workspace_mib': 256, 'estimated_latency_ms': 3.0},
    'static_only': {'ops': {'matmul', 'softmax'}, 'dtypes': {'bf16'}, 'dynamic_shape': False, 'workspace_mib': 128, 'estimated_latency_ms': 1.5},
}
decision = choose_backend(requirements, candidates)
assert decision['selected'] == 'accelerator_example'
assert decision['rejected']['static_only'] == ['dynamic_shape']
print(decision)

```

## Related Reading

When inputs change, check the applicability conditions first; when you need to trace generated implementations or compare actual costs, move on to the corresponding pages.

- [30 Dynamic Shape Handling](./30_Dynamic_Shape_Handling.md): guards, shape reuse, and bucketing.
- [Part 03 · 18 Compilation Artifacts and Actual Execution](../03_Triton_Kernels/18_Triton_Compilation_Artifacts_and_Execution.md): specialized artifacts, object launch, and device events.
- [Part 04 · 11 CUDA Graph and JIT Compilation](../04_CUDA_and_System_Optimization/11_CUDA_Graph_and_JIT_Compile.md): capture replay and submission cost.
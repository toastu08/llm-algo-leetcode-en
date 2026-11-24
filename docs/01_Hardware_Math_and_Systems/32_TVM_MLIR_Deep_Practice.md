# 32. TVM MLIR Deep Practice | TVM / MLIR 编译链路

**Difficulty:** Hard | **Environment:** CPU-first | **Tags:** `compiler systems`, `TVM/MLIR`, `progressive lowering` | **Audience:** compiler and systems beginners

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/32_TVM_MLIR_Deep_Practice.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

Even with the correct shapes, a matrix multiply computation may fail to execute because the target does not support the required precision, or because the tiling exceeds the resource budget. Compilation needs to establish the symbolic dimensions, operator support, and tiling plan item by item, and the information left by each step determines what the next step can check.

This section uses Python data structures to represent these stages and checks shape binding, target support, scheduling budgets, and the final plan, without calling TVM or MLIR. After completing it, you should be able to locate the step at which a plan fails and distinguish plan generation from actual code execution; real implementations move on to operator optimization, and runtime compatibility continues to be checked in the hardware adaptation group.

**Keywords:** intermediate representation, shape binding, operator legalization, tiling schedule

![Multi-level IR, Passes, and code generation](../public/01_Hardware_Math_and_Systems/32_ir_pass_codegen.svg)

---
## Prerequisite Reading

**Overview:** First get ready the graph-level optimization, fusion, and dynamic shape constraints, then place them into the complete compilation chain.

- [09. AI Compilers and Graph Optimization](./09_AI_Compilers_and_Graph_Optimization.md)
- [19. Operator Fusion Introduction](./19_Operator_Fusion_Introduction.md)
- [30. Dynamic Shape Handling](./30_Dynamic_Shape_Handling.md)

## Q1: What does each intermediate representation transformation need to check?

Shape specialization binds symbolic dimensions to concrete sizes; legalization checks whether the current operators and data types can enter the given target; scheduling selects the tiling and checks the boundaries; finally, the target plan is generated. If the premises of any stage are not met, the pipeline should stop rather than just writing a "success" marker.

The following models only one three-dimensional matrix multiplication representation, recording the shape as $(M,N,K)$. Assume the target supports FP16 and adopts evenly divisible tiling that requires no tail-block masking; these are a set of teaching rules and do not represent general limitations of TVM, MLIR, or CUDA.
### Q1 Contrast Verification: Build the target plan only after completing shape binding

```python
source_ir = {'op': 'matmul', 'dtype': 'fp16', 'shape': ('B', 4096, 4096)}


def specialize_shape(graph, bindings=None):
    """Copy the input and bind all symbolic dimensions; this example only accepts positive integer dimensions."""
    bindings = {'B': 8} if bindings is None else bindings
    shape = tuple(bindings.get(dim, dim) for dim in graph['shape'])
    if len(shape) != 3 or any(type(dim) is not int or dim <= 0 for dim in shape):
        raise ValueError('全部 M、N、K 维度必须绑定为正整数')
    return {**graph, 'shape': shape, 'shape_specialized': True}


def legalize_ops(graph):
    """Check one operator and data type of the example target without calling a real backend."""
    if not graph.get('shape_specialized'):
        raise ValueError('先完成形状绑定')
    if graph['op'] != 'matmul' or graph['dtype'] != 'fp16':
        raise ValueError('本例目标只声明支持 FP16 matmul')
    return {**graph, 'op': 'example.matmul', 'legalized': True}


def select_schedule(graph):
    """Check the evenly divisible tiling of this example, explicitly covering the three dimensions M, N, and K."""
    if not graph.get('legalized'):
        raise ValueError('先完成算子合法化')
    shape = graph['shape']
    tiles = (8, 128, 128)
    valid = len(shape) == 3 and all(
        type(dim) is int and dim > 0 and dim % tile == 0
        for dim, tile in zip(shape, tiles)
    )
    if not valid:
        raise ValueError('形状不满足本例无尾块屏蔽的分块规则')
    return {**graph, 'schedule': {'tiles_mnk': tiles, 'warps': 4},
            'schedule_valid': True}


def build_backend_plan(graph):
    """Generate a Python target plan; it does not produce executable code or evidence of successful compilation."""
    if not all(graph.get(key) is True for key in
               ('shape_specialized', 'legalized', 'schedule_valid')):
        raise ValueError('缺少形状、合法化或调度前提')
    if graph['op'] != 'example.matmul' or graph['dtype'] != 'fp16':
        raise ValueError('目标算子或数据类型不匹配')
    # Re-check the schedule to avoid reusing an old success marker after the shape is modified.
    checked = select_schedule(graph)
    if graph.get('schedule') != checked['schedule']:
        raise ValueError('计划中的调度与本例规则不一致')
    return {**graph, 'backend': 'example_target', 'representation': 'python_plan'}


pipeline = [specialize_shape, legalize_ops, select_schedule, build_backend_plan]
ir = source_ir
for step in pipeline:
    ir = step(ir)
    print(step.__name__, '->', ir)
assert source_ir == {'op': 'matmul', 'dtype': 'fp16', 'shape': ('B', 4096, 4096)}
assert ir['representation'] == 'python_plan'
assert ir['shape'] == (8, 4096, 4096)

```

## Q2: How do you first rule out infeasible schedules, then compare times?

Tiling affects data reuse and resource usage. First rule out infeasible items according to the current problem's shape, shared memory, and warp budgets, then compare the times of the remaining candidates; if there is no feasible item, report an empty result instead of forcing a choice.

The latencies below are constructed data; no autotuning has been run. Actual measurement also requires fixed inputs, warmup, repetition counts, and correctness conditions; the ranking here must not be treated as a real GPU conclusion.
### Q2 Contrast Verification: The fastest record does not necessarily satisfy the resource conditions

```python
import math


def select_tuning_candidate(candidates, problem_size, shared_memory_limit,
                            max_warps=8):
    """Filter the constructed records by the given constraints; return None for an empty feasible set."""
    if any(type(value) is not int or value <= 0
           for value in (problem_size, shared_memory_limit, max_warps)):
        raise ValueError('问题规模与预算必须是正整数')
    feasible = []
    for row in candidates:
        if any(type(row[key]) is not int or row[key] <= 0
               for key in ('tile', 'warps', 'shared_memory_bytes')):
            raise ValueError('分块、Warp 数与共享内存字节数必须是正整数')
        if not math.isfinite(row['latency_us']) or row['latency_us'] <= 0:
            raise ValueError('延迟必须是有限正数')
        if (problem_size % row['tile'] == 0
                and row['shared_memory_bytes'] <= shared_memory_limit
                and row['warps'] <= max_warps):
            feasible.append(row)
    best = min(feasible, key=lambda row: row['latency_us']) if feasible else None
    return {'feasible': feasible, 'selected': best}


candidates = [
    {'tile': 64, 'warps': 4, 'shared_memory_bytes': 32768, 'latency_us': 82.4},
    {'tile': 128, 'warps': 4, 'shared_memory_bytes': 65536, 'latency_us': 54.7},
    {'tile': 256, 'warps': 8, 'shared_memory_bytes': 131072, 'latency_us': 49.1},
]
report = select_tuning_candidate(candidates, 4096, 99000)
assert report['selected']['tile'] == 128
assert select_tuning_candidate(candidates, 4096, 1024)['selected'] is None
print('构造数据筛选：', report)

```

## Q3: How does the hardware capability check constrain transformations?

Scheduling and code generation require the target to have the corresponding instruction, storage, and access capabilities. The following declares a set of premises for each example transformation and lists the items missing from the target capability set; meeting the declaration is only a local necessary condition, and operator semantics, layout, resources, and compiler support still need to be checked.

These premises are the rules of this example, not general conclusions. For example, fusion does not necessarily use shared memory, and layout optimization does not always require contiguous loads; the real premises are determined by the specific transformation.
### Q3 Contrast Verification: List the target capabilities missing for each example transformation

```python
compiler_passes = [
    ('fusion', {'shared_memory'}),
    ('tensor_core_vectorize', {'mma_fp16'}),
    ('layout_opt', {'contiguous_load'}),
]

hardware_capabilities = {'shared_memory', 'contiguous_load'}
decisions = []

for name, required in compiler_passes:
    missing = sorted(required - hardware_capabilities)
    decisions.append({'pass': name, 'declared_capabilities_met': not missing, 'missing': missing})

for row in decisions:
    print(row)

assert next(row for row in decisions if row['pass'] == 'fusion')['declared_capabilities_met']
assert next(row for row in decisions if row['pass'] == 'tensor_core_vectorize')['missing'] == ['mma_fp16']
```

## Q4: How do you check the stage order and the final target?

Stage checking needs to answer at the same time: whether the preceding stages are complete, whether the names are known, and whether the execution chain reaches the required endpoint. A prefix that is legal in order but stops at shape specialization is not a complete target-generation flow.

The following designates a dependency chain for the example and by default requires completion of the code generation stage; when checking only a prefix, the endpoint requirement must be explicitly disabled. The example treats duplicate stages as errors; real compilers may run some transformations repeatedly by design, so this rule must not be generalized to all pass pipelines.
### Q4 Contrast Verification: Distinguish a complete flow, a legal prefix, and an incorrect order

```python
def validate_pipeline(pipeline, require_codegen=True):
    """Check the stage dependencies and endpoint of this example; it does not mean actual compilation has been executed."""
    dependencies = {
        'shape_specialize': set(),
        'legalize_ops': {'shape_specialize'},
        'layout_transform': {'legalize_ops'},
        'scheduling': {'layout_transform'},
        'codegen': {'scheduling'},
    }
    if type(require_codegen) is not bool:
        raise ValueError('终点要求必须是布尔值')
    completed = set()
    for name in pipeline:
        if name not in dependencies:
            return False, {'pass': name, 'reason': 'unknown_stage'}
        if name in completed:
            return False, {'pass': name, 'reason': 'duplicate_stage'}
        missing = dependencies[name] - completed
        if missing:
            return False, {'pass': name, 'missing': sorted(missing)}
        completed.add(name)
    if require_codegen and 'codegen' not in completed:
        return False, {'reason': 'incomplete_pipeline', 'missing': ['codegen']}
    return True, {'completed': list(pipeline)}


valid_pipeline = ['shape_specialize', 'legalize_ops', 'layout_transform',
                  'scheduling', 'codegen']
invalid_pipeline = ['shape_specialize', 'scheduling', 'legalize_ops',
                    'layout_transform', 'codegen']
assert validate_pipeline(valid_pipeline)[0]
assert not validate_pipeline(invalid_pipeline)[0]
assert not validate_pipeline([])[0]
assert not validate_pipeline(['shape_specialize'])[0]
assert validate_pipeline(['shape_specialize'], require_codegen=False)[0]
print('完整流程：', validate_pipeline(valid_pipeline))
print('错误顺序：', validate_pipeline(invalid_pipeline))

```

## Related Reading

**Overview:** Continue by handing the generation path to concrete kernels and the runtime for verification.

- [05. Triton GEMM: Tiling and Accumulation](../03_Triton_Kernels/05_Triton_GEMM_Tutorial.md)
- [09. Triton Autotuning and Benchmarking](../03_Triton_Kernels/09_Triton_Autotune_and_Profiling.md)
- [34. Heterogeneous Accelerators and Runtime Constraints](./34_Heterogeneous_Accelerators_and_Runtime_Constraints.md)
---
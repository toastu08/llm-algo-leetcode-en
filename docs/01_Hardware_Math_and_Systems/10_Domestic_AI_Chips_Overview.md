# 10. Domestic AI Chips Overview | 算力现状与替代方案

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `hardware systems`, `AI chips`, `heterogeneous compute` | **Audience:** learners of hardware constraints

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/10_Domestic_AI_Chips_Overview.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

When preparing to move the same model to another type of accelerator, a device with larger capacity and higher peak compute may still fail to load the original artifact, or may lack some critical operation. Beyond hardware specifications, you also need to confirm whether the precision, execution framework, and software support required by the model match.

This section first lists the capacity and operation requirements of the same task, then checks the required software components, migration schedule, and power limits. After completing it, you should be able to point out exactly what a candidate lacks, instead of masking the problem with a composite score; the retained candidates continue to be verified in the runtime constraints and deployment topics.

**Keywords:** device ecosystem, software stack, migration risk

![Availability judgment for accelerator candidates](../public/01_Hardware_Math_and_Systems/10_accelerator_readiness.svg)

## Prerequisite Reading
**Overview:** First get ready with the two constraints of single-device resources and multi-device communication, then screen hardware candidates.
- [1B: Single-Device Execution, Memory and Data Access](./1B.md)
- [1C: Communication, Parallelism and Fault Tolerance](./1C.md)
## Q1: Why can't you look only at peak compute when choosing a chip?


The paper peak only says "how fast it can theoretically run," not "whether it can keep running fast in a real workload."

Actual selection also depends on:
- whether the software stack is mature;
- whether the compiler and key kernels are available;
- whether the communication and scheduling capabilities match;
- whether the migration and maintenance costs are controllable.

So selection is not a comparison of a single performance number, but a comparison of end-to-end deployability.

### Q1 Feasibility Check: Compare budgets and declared capabilities first

The capacity input is the estimated peak working set of the same stage, including weights, state, and workspace, not just the size of the weight file. The following checks the given capacity, operators, and data types, without reading local specifications or loading a model.


```python
import math


def accelerator_feasibility(required_peak_bytes, device_memory_bytes, required_ops, supported_ops, required_dtype, supported_dtypes):
    """Check the three hard conditions of capacity, operators, and dtype before comparing peaks."""
    if any(not math.isfinite(value) or value <= 0
           for value in (required_peak_bytes, device_memory_bytes)):
        raise ValueError('memory bytes must be positive')
    reasons = []
    if required_peak_bytes > device_memory_bytes:
        reasons.append('insufficient_memory')
    missing_ops = sorted(set(required_ops) - set(supported_ops))
    if missing_ops:
        reasons.append('missing_ops')
    if required_dtype not in supported_dtypes:
        reasons.append('unsupported_dtype')
    return {'feasible': not reasons, 'reasons': reasons, 'missing_ops': missing_ops}

ready = accelerator_feasibility(8e9, 16e9, {'matmul', 'softmax'}, {'matmul', 'softmax'}, 'bf16', {'fp16', 'bf16'})
blocked = accelerator_feasibility(20e9, 16e9, {'matmul', 'softmax'}, {'matmul'}, 'bf16', {'fp16'})
assert ready['feasible'] is True
assert blocked['feasible'] is False
print('ready:', ready)
print('blocked:', blocked)
```

## Q2: Which necessary components must the software stack check?

Device capability requires the driver, execution framework, target kernels, and relevant dependencies to work together. The checklist should be determined by the execution path: a single-process workload without collective communication should not be blocked because a communication library is missing; when collective communication is needed, the corresponding support must be checked.

The following only compares the declared component states. The compiler field represents the compilation or artifact support required by the chosen path; it does not mean every machine must have a local toolchain installed; specific version compatibility and actual execution still need independent verification.

### Q2 Condition Check: Whether communication support is required by the current path


```python
def stack_readiness(driver, compiler, runtime, kernel, communication,
                    kernel_coverage=1.0, requires_communication=False):
    """Check the given state according to the current path; meeting the conditions does not mean it has actually been executed."""
    layers = {
        'driver': driver, 'compiler': compiler, 'runtime': runtime,
        'kernel': kernel, 'communication': communication,
    }
    if any(type(value) is not bool for value in
           (*layers.values(), requires_communication)):
        raise ValueError('组件状态与通信需求必须是布尔值')
    if not math.isfinite(kernel_coverage) or not 0 <= kernel_coverage <= 1:
        raise ValueError('关键算子覆盖率必须位于 [0, 1]')
    required = ['driver', 'compiler', 'runtime', 'kernel']
    if requires_communication:
        required.append('communication')
    missing = [name for name in required if not layers[name]]
    if kernel_coverage != 1:
        missing.append('kernel_coverage')
    return {
        'declared_requirements_met': not missing,
        'missing': missing,
        'first_reported_gap': missing[0] if missing else None,
        'kernel_coverage': kernel_coverage,
    }


single_device = stack_readiness(True, True, True, True, False)
multi_device = stack_readiness(True, True, True, True, False,
                               requires_communication=True)
assert single_device['declared_requirements_met']
assert multi_device['missing'] == ['communication']
assert stack_readiness(True, False, True, True, True)['missing'] == ['compiler']
print('无通信需求：', single_device)
print('需要通信：', multi_device)

```

## Q3: When is an alternative worth entering subsequent verification?


When budgets, supply chain, power, deployment environment, or compliance requirements change, an alternative is no longer just a backup but may become the primary plan.

Whether a candidate can continue to be verified comes back to three things:
- whether it satisfies the hard conditions of capacity, precision, and key operators;
- whether the software stack is closed-loop;
- whether the migration schedule, supply risk, and power exceed the current limits.

Here we first form a candidate list and do not compute the full total cost of ownership; the cost per successful request is computed uniformly by [33 TCO and Cost Model](./33_TCO_and_Cost_Model.md).

### Q3 Candidate Screening: Whether the given risks and resources exceed the limits

The following schedule, power, and risk values are all assumed inputs; the risk score ranges from zero to one and is not a real failure probability derived from the device name.


```python
def admit_accelerator_candidate(feasible, stack_ready, migration_weeks, max_migration_weeks, supply_risk, max_supply_risk, power_kw, power_cap_kw):
    """Before computing the full TCO, screen device candidates by hard constraints and risk limits."""
    if type(feasible) is not bool or type(stack_ready) is not bool:
        raise ValueError('条件检查结果必须是布尔值')
    if any(not math.isfinite(value) or value < 0 for value in
           (migration_weeks, max_migration_weeks, supply_risk, max_supply_risk,
            power_kw, power_cap_kw)):
        raise ValueError('risk and resource inputs must be non-negative')
    if not 0 <= supply_risk <= 1 or not 0 <= max_supply_risk <= 1:
        raise ValueError('风险分值与上限必须位于 [0, 1]')
    reasons = []
    if not feasible:
        reasons.append('device_constraints')
    if not stack_ready:
        reasons.append('software_stack')
    if migration_weeks > max_migration_weeks:
        reasons.append('migration_schedule')
    if supply_risk > max_supply_risk:
        reasons.append('supply_risk')
    if power_kw > power_cap_kw:
        reasons.append('power_cap')
    return {'admitted': not reasons, 'reasons': reasons}

candidates = {
    'baseline': admit_accelerator_candidate(True, True, 2, 8, 0.2, 0.5, 0.7, 1.0),
    'alternative': admit_accelerator_candidate(True, True, 6, 8, 0.4, 0.5, 0.5, 1.0),
    'immature_stack': admit_accelerator_candidate(True, False, 4, 8, 0.3, 0.5, 0.6, 1.0),
}
admitted = [name for name, result in candidates.items() if result['admitted']]
assert admitted == ['baseline', 'alternative']
assert candidates['immature_stack']['reasons'] == ['software_stack']
print('admission results:', candidates)

```

## Related Reading
**Overview:** If you want to continue to complete the judgments on selection, compilation, and communication, you can continue to read these pages.
- [09. Triton Autotuning and Benchmarking](../03_Triton_Kernels/09_Triton_Autotune_and_Profiling.md)
- [01. CUDA Custom Kernel Intro](../04_CUDA_and_System_Optimization/01_CUDA_Custom_Kernel_Intro.md)
- [09. Distributed Communication Primitives](../04_CUDA_and_System_Optimization/13_Distributed_Communication_Primitives.md)
# 34. Heterogeneous Accelerators and Runtime Constraints | 异构加速器与运行时约束

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `heterogeneous systems`, `runtime`, `deployment constraints` | **Audience:** learners who need to compare the execution conditions of different accelerators

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/34_Heterogeneous_Accelerators_and_Runtime_Constraints.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

Model migration does not require every stage to use the same type of device. Text preprocessing may need only a small amount of host memory, while the generation stage must hold weights, cache, and workspace; if the requirements of the generation stage are imposed on preprocessing, a CPU that is actually usable might be incorrectly excluded.

This section splits the task into stages, compares capacity, precision, operations, and runtime requirements separately, and then checks the data handoff between stages. After completing it, you should be able to distinguish "declared match," "the real path can execute," and "performance meets the requirement"; these judgments connect to the loading and execution verification in the deployment and heterogeneous systems topic.

**Keywords:** heterogeneous accelerators, model artifact, runtime, executable contract

![Preprocessing and generation use their own device requirements](../public/01_Hardware_Math_and_Systems/34_stage_requirements_and_handoff.svg)

## Prerequisite Reading

- [10 AI Chips Overview and Alternatives](./10_Domestic_AI_Chips_Overview.md)
- [1E Compilers and Backends](./1E.md)
- [07 CPU and GPU Heterogeneous Scheduling](./07_CPU_GPU_Heterogeneous_Scheduling.md)
## Q1: Which candidates can the declared device capabilities rule out?

First record the capacity, precision, operator, and runtime requirements of the target stage, then compare them with the device declarations. The CPU, GPU, and NPU parameters below are all constructed configurations and do not read local devices; the presence of a name in a set does not mean the current version already supports it or has actually executed it.

| Judgment dimension | What to record in the declaration | What still needs to be checked in actual execution |
|:---|:---|:---|
| Memory | Peak object budget and available device capacity | Actual residency of state, workspace, and other processes |
| Precision | Data type required by the stage | Artifact format and execution data type |
| Operators | Set of operations required by the stage | Compilation, execution, and fallback under the corresponding shapes |
| Runtime and communication | Execution framework and collective communication used as needed | Version compatibility, load logs, and functional tests |


```python
import math


DEVICE_PROFILES = {
    "cpu": {"memory_gb": 128, "precisions": {"fp32", "bf16"}, "ops": {"tokenize", "route", "postprocess"}, "runtimes": {"python"}, "collective": False},
    "gpu": {"memory_gb": 48, "precisions": {"fp16", "bf16", "int8"}, "ops": {"prefill", "decode", "rope", "attention"}, "runtimes": {"torch", "vllm"}, "collective": True},
    "npu": {"memory_gb": 64, "precisions": {"bf16", "int8"}, "ops": {"prefill", "decode", "attention"}, "runtimes": {"vendor-runtime"}, "collective": True},
}


def summarize_device_profile(profile):
    """Convert the device profile into a basic ledger that is easy to compare."""
    return {
        "memory_gb": profile["memory_gb"],
        "precision_count": len(profile["precisions"]),
        "operator_count": len(profile["ops"]),
        "runtime_count": len(profile["runtimes"]),
        "collective": profile["collective"],
    }


assert summarize_device_profile(DEVICE_PROFILES["gpu"])["operator_count"] == 4
```

## Q2: What is the difference between a capability declaration and actual loading?

Candidate identity comes from the model version, artifact format, tokenizer, and template; runtime checks must confirm that the device, driver, execution framework, and key operations can work together. The following functions only compare the given declarations and cannot replace model loading or real execution.

Capacity uses decimal GB, and the requirement is the estimated peak of the compared stage. The preprocessing stage should not reuse the full model capacity and precision of the generation stage; the requirements of each stage are filled in separately in the next question.

| Check | What this example can do | What needs to be verified separately |
|:---|:---|:---|
| Capacity | Compare the given peak budget with the declared capacity | Real allocation peak and safety margin |
| Data types and operations | Compute the set difference and list missing items | Support under the target shape and specific version |
| Runtime | Check whether the name is declared as supported | Environment startup, artifact loading, and the actual path |
| Collective communication | Check the declaration only when the stage requires it | Multi-process execution and numerical results |


```python
REQUEST_PLAN = {
    "estimated_peak_gb": 40,
    "precision": "bf16",
    "required_ops": {"prefill", "decode", "attention"},
    "runtime": "vllm",
    "requires_collective": False,
    "stages": {
        "prepare": {
            "estimated_peak_gb": 0.5, "precision": "fp32",
            "required_ops": {"tokenize"}, "runtime": "python",
            "requires_collective": False,
        },
        "generate": {
            "estimated_peak_gb": 40, "precision": "bf16",
            "required_ops": {"prefill", "decode", "attention"},
            "runtime": "vllm", "requires_collective": False,
        },
    },
}


def check_device_feasibility(profile, plan):
    """Compare one stage with the given device declaration; do not probe the device or execute the model."""
    required = ("estimated_peak_gb", "precision", "required_ops",
                "runtime", "requires_collective")
    if any(key not in plan for key in required):
        raise ValueError('阶段需求缺少必要字段')
    if (not math.isfinite(profile["memory_gb"]) or profile["memory_gb"] <= 0
            or not math.isfinite(plan["estimated_peak_gb"])
            or plan["estimated_peak_gb"] < 0):
        raise ValueError('设备容量须为有限正数，阶段预算须为有限非负数')
    if (type(plan["requires_collective"]) is not bool
            or type(profile["collective"]) is not bool):
        raise ValueError('集合通信需求和支持状态必须是布尔值')
    memory_ok = profile["memory_gb"] >= plan["estimated_peak_gb"]
    precision_ok = plan["precision"] in profile["precisions"]
    missing_ops = sorted(set(plan["required_ops"]) - set(profile["ops"]))
    runtime_ok = plan["runtime"] in profile["runtimes"]
    collective_ok = not plan["requires_collective"] or profile["collective"]
    return {
        "declared_requirements_met": bool(
            memory_ok and precision_ok and not missing_ops
            and runtime_ok and collective_ok),
        "memory_ok": memory_ok, "precision_ok": precision_ok,
        "missing_ops": missing_ops, "runtime_ok": runtime_ok,
        "collective_ok": collective_ok,
    }


assert check_device_feasibility(DEVICE_PROFILES["gpu"], REQUEST_PLAN)["declared_requirements_met"]
assert not check_device_feasibility(DEVICE_PROFILES["npu"], REQUEST_PLAN)["runtime_ok"]

```

## Q3: How does each stage use its own resource and runtime requirements?

Preprocessing, generation, and output have different working sets and operations. The plan lists the peak budget, precision, runtime, and communication requirements of each stage directly, without guessing the runtime from the stage name or copying the whole-model requirements to all stages. FP32 and BF16 here are the numerical type conditions of the corresponding stages.

A candidate existing for each stage still does not mean the entire path can execute: data must also be transferred, formats converted, and synchronization performed between stages, and objects merged onto the same device may also reside at the same time. The CPU calculator only screens devices whose declarations match; actual placement needs further verification of the handoff and the joint resource budget.

| Stage | Example requirements | What needs further checking |
|:---|:---|:---|
| Preprocessing | 0.5 GB, Python, tokenize | Host memory, input protocol, and preparation time |
| Generation | 40 GB, BF16, vLLM, prefill/decode/attention | Loading, state capacity, and the actual execution path |
| Stage handoff | Determined by the chosen device combination | Data format, transfer, synchronization, and joint residency |


```python
def assign_stages(profiles, plan):
    """Screen the declarations item by item according to the explicit stage configuration; do not decide the actual placement or verify the handoff."""
    if not isinstance(plan.get("stages"), dict) or not plan["stages"]:
        raise ValueError('必须提供非空阶段需求')
    stage_trace = {}
    for stage, stage_plan in plan["stages"].items():
        checks = {
            name: check_device_feasibility(profile, stage_plan)
            for name, profile in profiles.items()
        }
        stage_trace[stage] = {
            "candidate_devices": [
                name for name, check in checks.items()
                if check["declared_requirements_met"]
            ],
            "checks": checks,
        }
    return stage_trace


STAGE_TRACE = assign_stages(DEVICE_PROFILES, REQUEST_PLAN)
assert "cpu" in STAGE_TRACE["prepare"]["candidate_devices"]
assert STAGE_TRACE["generate"]["candidate_devices"] == ["gpu"]

```

## Q4: What checks are still missing before entering performance measurement?

First check whether the required stages are complete and whether any device has a matching declaration, then confirm quality and functional tests of the entire execution path. The path test should cover loading, stage handoff, and output, and must not require that "a performance benchmark already exists" before performance measurement is allowed to start, otherwise it creates a circular premise.

The boolean values below represent checks already completed externally; the example calls only construct states and do not produce real execution evidence. Only after the premises are met do we enter the comparison of timing, throughput, and stability under a fixed workload.

| Condition | State in this example | Next step |
|:---|:---|:---|
| Stage missing or no device with a matching declaration | blocked | Fix the requirements and device declarations |
| Quality check failed | blocked | Check the model, template, and numerical results |
| The entire path has not passed a real functional test | needs_runtime_check | Verify loading, handoff, and output |
| Necessary checks declared as passed | benchmark_ready | Start performance measurement; do not directly make a deployment adoption decision |


```python
def benchmark_admission(stage_trace, required_stages, quality_gate_passed,
                        runtime_smoke_passed):
    """Decide whether timing may start based on the given check results; do not verify the authenticity of the evidence."""
    if (type(quality_gate_passed) is not bool
            or type(runtime_smoke_passed) is not bool):
        raise ValueError('质量与路径检查结果必须是布尔值')
    if (not isinstance(required_stages, (list, tuple, set)) or not required_stages
            or any(not isinstance(name, str) or not name for name in required_stages)):
        raise ValueError('必须显式给出非空的必需阶段名称集合')
    missing = sorted(set(required_stages) - set(stage_trace))
    if missing:
        return {"status": "blocked", "reason": "缺少必需阶段", "missing": missing}
    if any(not stage_trace[name].get("candidate_devices")
           for name in required_stages):
        return {"status": "blocked", "reason": "必需阶段没有声明匹配的设备"}
    if not quality_gate_passed:
        return {"status": "blocked", "reason": "质量检查未通过"}
    if not runtime_smoke_passed:
        return {"status": "needs_runtime_check", "reason": "需要整条路径的真实功能测试"}
    return {"status": "benchmark_ready", "reason": "已声明必要检查通过，可进入性能测量"}


required_stages = list(REQUEST_PLAN["stages"])
assert benchmark_admission(STAGE_TRACE, required_stages, True, False)["status"] == "needs_runtime_check"
assert benchmark_admission(STAGE_TRACE, required_stages, True, True)["status"] == "benchmark_ready"
assert benchmark_admission({}, required_stages, True, True)["status"] == "blocked"

```

## Related Reading

- [Deployment and Heterogeneous Systems topic](../topic_discussion/deployment_heterogeneous/intro.md)
- [Communication and Parallelism topic](../topic_discussion/communication_parallelism/intro.md)
- [Part 02 · 72 Multi-Model Deployment Comparison](../02_PyTorch_Algorithms/72_Multi_Model_Deployment_Comparison.md)
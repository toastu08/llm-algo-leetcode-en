# 33. TCO and Cost Model | 算力评估与 TCO 模型

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `systems engineering`, `cost evaluation`, `TCO` | **Audience:** learners of hardware constraints

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/33_TCO_and_Cost_Model.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

When two devices can both run the same model, the one with the lower purchase price may not complete requests at a lower cost. Idle power draw, adaptation labor, downtime, and a lower successful throughput can all change the final comparison.

This section computes purchase, busy and idle energy, operations, and migration costs separately within the same holding period, then divides by the number of successful requests. After completing it, you should be able to explain how utilization and success rate change the unit cost, and distinguish full purchase spending from depreciation allocation; deployment and performance comparisons still need actual data from the corresponding workloads.

**Keywords:** total cost of ownership (TCO), unit cost, energy consumption

![Total cost of ownership decision chain](../public/01_Hardware_Math_and_Systems/33_tco_decision_chain.svg)

---
## Prerequisite Reading

**Overview:** First retain the candidates that passed the executability check, then compare the full cost within the holding period.

- [10. AI Chips Overview and Alternatives](./10_Domestic_AI_Chips_Overview.md)
- [34. Heterogeneous Accelerators and Runtime Constraints](./34_Heterogeneous_Accelerators_and_Runtime_Constraints.md)

## Q1: Why is TCO more important than the per-card price?


TCO looks at the total cost of ownership, not just the money spent on the card.

It takes purchase, energy, operations, migration, downtime risk, and replacement cost into account. For a system that runs for a long time, a cheaper card does not necessarily mean a cheaper overall solution, because the subsequent electricity, maintenance, and migration may be more expensive.

So what really needs to be compared is who can keep the business running more stably over a period of time.

### Q1 Cost Calculation: Compute busy and idle energy separately

Use a unified currency unit, assuming 30 days per month and continuous power for the entire device holding period. Utilization represents the fraction of busy time; power is still billed at the idle power draw during idle time. The example prices, power, and utilization are all constructed inputs, not device quotes or measured results.


```python
import math


def tco_ledger(purchase_cost, power_kw, electricity_per_kwh,
               ops_cost_per_month, migration_cost, months, utilization,
               idle_power_kw=0.0):
    """Build a simplified holding-period cost from full purchase cash spending and busy/idle energy."""
    values = (purchase_cost, power_kw, electricity_per_kwh, ops_cost_per_month,
              migration_cost, months, utilization, idle_power_kw)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('成本与资源输入必须是有限非负数')
    if months <= 0 or utilization > 1 or idle_power_kw > power_kw:
        raise ValueError('月份须为正，利用率不得超过一，空闲功率不得高于繁忙功率')
    powered_hours = months * 30 * 24
    busy_hours = powered_hours * utilization
    idle_hours = powered_hours - busy_hours
    energy_kwh = power_kw * busy_hours + idle_power_kw * idle_hours
    components = {
        'purchase_cost': purchase_cost,
        'energy_cost': energy_kwh * electricity_per_kwh,
        'operations_cost': ops_cost_per_month * months,
        'migration_cost': migration_cost,
    }
    return {
        'components': components, 'total_cost': sum(components.values()),
        'powered_hours': powered_hours, 'busy_hours': busy_hours,
        'idle_hours': idle_hours, 'energy_kwh': energy_kwh,
    }


gpu_a = tco_ledger(30000, 0.7, 0.8, 200, 2000, 24, 0.7, idle_power_kw=0.1)
gpu_b = tco_ledger(22000, 0.9, 0.8, 350, 8000, 24, 0.7, idle_power_kw=0.15)
assert gpu_a['total_cost'] == sum(gpu_a['components'].values())
assert tco_ledger(0, 1, 1, 0, 0, 1, 0, idle_power_kw=0.1)['energy_kwh'] == 72
print('A 构造成本：', gpu_a)
print('B 构造成本：', gpu_b)

```

## Q2: Why should migration cost and risk be included in the model?


Switching hardware is not just replacing the device; it also brings software adaptation, debugging, refactoring, and performance regression risks.

If migration requires re-adapting the compiler, kernels, communication libraries, or deployment flow, that work is itself part of the cost.

So TCO is not a choice between a financial problem and a technical problem, but puts both into the same judgment framework.

### Q2 Risk Calculation: Labor cost and conditional rollback loss

Engineering and validation hours are billed at the same rate; downtime in this example refers only to the extra downtime when a rollback occurs, so it is multiplied by the rollback probability. Planned downtime or other losses need to be listed separately and must not be omitted by also multiplying them by this probability.


```python
def migration_cost(engineering_hours, hourly_rate, validation_hours, downtime_hours, downtime_cost_per_hour, rollback_probability):
    """Compute the migration engineering cost and the downtime risk cost weighted by the rollback probability."""
    if any(not math.isfinite(value) or value < 0 for value in
           (engineering_hours, hourly_rate, validation_hours, downtime_hours,
            downtime_cost_per_hour)):
        raise ValueError('migration inputs must be non-negative')
    if not math.isfinite(rollback_probability) or not 0 <= rollback_probability <= 1:
        raise ValueError('rollback_probability must be in [0, 1]')
    labor_cost = (engineering_hours + validation_hours) * hourly_rate
    expected_rollback_cost = downtime_hours * downtime_cost_per_hour * rollback_probability
    return {'labor_cost': labor_cost, 'expected_rollback_cost': expected_rollback_cost, 'total_cost': labor_cost + expected_rollback_cost}

low_risk = migration_cost(120, 100, 40, 2, 5_000, 0.05)
high_risk = migration_cost(120, 100, 40, 2, 5_000, 0.5)
assert high_risk['total_cost'] > low_risk['total_cost']
print('low risk:', low_risk)
print('high risk:', high_risk)
```

## Q3: How is the denominator of per-million successful requests computed?

First fix the cost holding period, then estimate the number of successful requests completed during it. The request rate below is the processing rate when the service is available and busy, availability represents the fraction of time the service can serve, utilization represents the busy fraction within the serviceable time, and the success rate represents the fraction of attempts that meet the quality and functional requirements; multiplying the four gives the total number of successful requests.

If the input request rate has already been averaged over the full calendar time, it must not be multiplied by availability and utilization again. When the number of successful requests is zero, the unit cost is undefined and must not be reported as zero.

### Q3 Unit Cost: Record availability, utilization, and success rate separately


```python
def cost_per_million_successful_requests(total_cost, requests_per_second,
                                          availability, months,
                                          utilization=1.0, success_rate=1.0):
    """The request rate is measured over available and busy time; the denominator separately accounts for availability, utilization, and success rate."""
    values = (total_cost, requests_per_second, availability, months,
              utilization, success_rate)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('输入必须是有限非负数')
    if months <= 0 or any(value > 1 for value in
                          (availability, utilization, success_rate)):
        raise ValueError('月份须为正，各比例须位于 [0, 1]')
    successful_requests = (requests_per_second * availability * utilization
                           * success_rate * months * 30 * 24 * 3600)
    if successful_requests <= 0:
        raise ValueError('没有成功请求，单位成本未定义')
    return total_cost / successful_requests * 1_000_000


options = {
    'stable_example': cost_per_million_successful_requests(
        gpu_a['total_cost'], 20, 0.99, 24, utilization=0.7, success_rate=0.98),
    'alternative_example': cost_per_million_successful_requests(
        gpu_b['total_cost'], 18, 0.96, 24, utilization=0.7, success_rate=0.95),
}
assert all(value > 0 for value in options.values())
print('每百万成功请求的构造成本：', options)

```

## Q4: Why do the expansion pace and depreciation cycle also change the TCO conclusion?


TCO is not just "can you afford to buy it or can you afford to run it"; it also depends on how long you need to run this system at full capacity.

If expansion is too slow, the compute purchased early may sit idle, driving up the cost per unit of output; if the depreciation cycle is too short, the hardware must be replaced before it is fully utilized, and the book cost will also rise rapidly.

Therefore a real cost model often has to consider at the same time:
- the initial purchase cost;
- the operating cost;
- the migration and switching cost;
- the expansion and depreciation pace.

Only by looking at these variables together can the decision come closer to real technology selection.

### Q4 Utilization Contrast: Distinguish depreciation allocation from purchase cash spending

This example allocates the purchase cost over the depreciation period, then divides by the hours of use; the operations and migration cost inputs are totals for the same observation period. This is a different measure from the full purchase cash spending in Q1, and the two purchase costs must not be added together; the example also assumes 30 days per month.


```python
def effective_cost_per_used_hour(purchase_cost, operations_cost, migration_cost, horizon_months, depreciation_months, utilization):
    """Allocate the purchase cost over the depreciation period and normalize by the actual hours of use."""
    if any(not math.isfinite(value) or value < 0
           for value in (purchase_cost, operations_cost, migration_cost)):
        raise ValueError('costs must be non-negative')
    if (any(not math.isfinite(value) for value in
            (horizon_months, depreciation_months, utilization))
            or horizon_months <= 0 or depreciation_months <= 0
            or not 0 < utilization <= 1):
        raise ValueError('invalid horizon, depreciation or utilization')
    allocated_purchase = purchase_cost * min(horizon_months / depreciation_months, 1.0)
    used_hours = horizon_months * 30 * 24 * utilization
    total_allocated_cost = allocated_purchase + operations_cost + migration_cost
    return {'allocated_purchase': allocated_purchase, 'used_hours': used_hours, 'cost_per_used_hour': total_allocated_cost / used_hours}

scenarios = {
    'high_utilization': effective_cost_per_used_hour(30_000, 5_000, 2_000, 24, 36, 0.8),
    'low_utilization': effective_cost_per_used_hour(30_000, 5_000, 2_000, 24, 36, 0.3),
}
assert scenarios['high_utilization']['cost_per_used_hour'] < scenarios['low_utilization']['cost_per_used_hour']
print(scenarios)

```

## Related Reading

**Overview:** Hand the cost results to the deployment and performance topics and judge them together with real workload evidence.

- [31. GPU Virtualization and MIG](./31_GPU_Virtualization_and_MIG.md)
- [Performance Optimization topic](../topic_discussion/performance_optimization/intro.md)
- [Deployment and Heterogeneous Systems topic](../topic_discussion/deployment_heterogeneous/intro.md)
---
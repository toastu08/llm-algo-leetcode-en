# 03. Python OOP and Utility Patterns | Python 面向对象与工具模式

**Difficulty:** Easy | **Environment:** CPU-first | **Tags:** `Python`, `object encapsulation`, `state management` | **Audience:** learners preparing for Parts 2–4

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/03_Python_OOP_and_Utility_Patterns.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## Section Guide

When a group of fields shares constraints and is updated across calls, scattered variables struggle to keep the state consistent. The value of an object lies in putting data, constraints, and lifecycle inside the same boundary.

After completing this section, you should be able to decide when a stateful object is needed, and check whether initialization, updates, queries, and reset jointly maintain the internal invariants.

**Keywords:** `dataclass`, `state`, `invariant`, `lifecycle`

![Python objects and state management](../public/00_Prerequisites/03_python_state_object_map.svg)

## Prerequisite Reading
**Overview:** First master Python data structures and function contracts, then decide which state should be collected into an object.
- [01. Python Essentials for LLM](./01_Python_Essentials_for_LLM.md)
- [0A Group Page](./0A.md)

## Q1: When do an object's internal fields need to jointly maintain an invariant?

Putting fields into a dataclass is not enough. When fields share constraints, validate them at object creation so that the object stays legal as soon as it is created. Below, initialization validation, state updates, and query operations maintain the object's internal contract.

```python
from dataclasses import asdict, dataclass, field


@dataclass
class TrainingOptions:
    """Store a group of related options and maintain field invariants at creation time."""

    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 3

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive')
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError('batch_size and epochs must be positive')

    def as_dict(self):
        return asdict(self)


options = TrainingOptions(batch_size=16)
print(options.as_dict())
```

## Q1 verification: Can valid and invalid configurations be distinguished at the entry point?

```python
options = TrainingOptions(learning_rate=2e-4, batch_size=8, epochs=2)
assert options.as_dict()['batch_size'] == 8

try:
    TrainingOptions(batch_size=0)
except ValueError as error:
    assert 'batch_size' in str(error)
else:
    raise AssertionError('非法 batch_size 应触发 ValueError')
print('✅ 对象内部字段不变量通过')
```

## Q2: Why are continuously updated metrics well suited to management by an object?

The weighted mean of a metric depends on the accumulated total and the total weight, and these states must be updated, queried, and reset together. Putting them in the same object prevents the caller from forgetting to synchronize one of the variables.

```python
@dataclass
class MetricTracker:
    """Accumulate weighted metrics; total and weight_sum must be updated together."""

    total: float = 0.0
    weight_sum: float = 0.0

    def update(self, value, *, weight=1):
        if weight <= 0:
            raise ValueError('weight must be positive')
        self.total += float(value) * weight
        self.weight_sum += float(weight)

    @property
    def mean(self):
        return self.total / self.weight_sum if self.weight_sum else 0.0

    def reset(self):
        self.total = 0.0
        self.weight_sum = 0.0


tracker = MetricTracker()
tracker.update(0.4, weight=2)
tracker.update(0.1)
print('mean:', tracker.mean)
```

## Q2 verification: Do update, mean, and reset share the same state?

```python
tracker = MetricTracker()
tracker.update(0.4, weight=2)
tracker.update(0.1)
assert abs(tracker.mean - 0.3) < 1e-12
tracker.reset()
assert tracker.total == 0.0 and tracker.weight_sum == 0.0 and tracker.mean == 0.0

try:
    tracker.update(0.2, weight=0)
except ValueError as error:
    assert 'weight' in str(error)
else:
    raise AssertionError('非正权重应触发 ValueError')
print('✅ 指标状态通过')
```

## Q3: When should a helper be upgraded to a stateful object?

A one-off formatting task only needs a helper; when a call must continuously maintain a step, a history, and the latest state, an object should carry the lifecycle. The `RunState` below advances the step every time it records a metric and keeps a queryable history.

```python
def format_metric(step, loss):
    """Stateless helper: format the input once."""
    return f'step={step}, loss={loss:.4f}'


@dataclass
class RunState:
    """Maintain the step and metric event history across calls."""

    step: int = 0
    history: list = field(default_factory=list)

    def record(self, loss):
        self.step += 1
        event = {'step': self.step, 'loss': float(loss)}
        self.history.append(event)
        return event

    @property
    def latest(self):
        return self.history[-1] if self.history else None


state = RunState()
state.record(0.5)
state.record(0.25)
print(format_metric(**state.latest))
```

## Q3 verification: Does the object correctly maintain state across calls?

```python
state = RunState()
assert state.latest is None
first = state.record(0.5)
second = state.record(0.25)
assert first == {'step': 1, 'loss': 0.5}
assert second == {'step': 2, 'loss': 0.25}
assert len(state.history) == 2
assert format_metric(**state.latest) == 'step=2, loss=0.2500'
print('✅ helper 与有状态对象边界通过')
```

## Related Reading

Now that you understand object state, continue by organizing configuration, paths, and run artifacts into a reproducible entry point.

- [04. Python Config and Data Entry](./04_Python_Config_and_Data_Entry.md)
- [09. PyTorch nn.Module Basics](./09_PyTorch_nn_Module_Basics.md)
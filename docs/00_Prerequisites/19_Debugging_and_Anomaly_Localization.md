# 19. Debugging and Anomaly Localization | 调试与异常定位

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `debugging`, `troubleshooting`, `tensor health` | **Audience:** learners preparing for Parts 2–4

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/19_Debugging_and_Anomaly_Localization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## Section Guide

When a program raises an error or the training result is abnormal, the symptom you see first is not necessarily the closest to the cause. What matters more is: before correctness passes, timing and memory results cannot continue to be interpreted as performance evidence. A non-finite loss may originate from an earlier activation or gradient, and a missing gradient may also come from different computation graphs and registration states.

After completing this section, you should be able to narrow the scope by tensor contract, numerical state, and gradient chain, locate the first failure point, form a replayable minimal reproduction, and output a validity conclusion on whether the current run can enter performance analysis.

**Keywords:** `shape`, `dtype`, `device`, `NaN`

![Anomaly classification and the first check item](../public/00_Prerequisites/19_debugging_anomaly_map.svg)


## Prerequisite Reading
**Overview:** First use the 0E group page to decide whether the current symptom is a structure, numerical, or gradient problem, then enter the corresponding check path.
- [0E Group Page](./0E.md)

## Q1: How do you first decide whether a run is worth further analysis?

First choose the first check by symptom: a dimension mismatch checks the structural contract, mixing CPU / CUDA checks the device, a loss that becomes NaN checks the first non-finite value, and a grad of None checks the gradient chain. If any anomaly exists, mark the current run as `invalid`, localize and reproduce first, and do not continue to interpret performance; only after all checks pass is it marked `valid` and allowed into time or memory evidence collection.


| Anomaly category | Typical symptom | First check |
|:---|:---|:---|
| shape | Dimension mismatch, broadcast error | Shapes of inputs and operators |
| dtype | Incompatible type, abnormal precision | Input and parameter dtypes |
| device / layout | Mixing CPU and CUDA, abnormal access path | device, layout, contiguous |
| numerical | NaN / Inf | The first non-finite value |
| gradient | grad is None or abnormal gradients | Computation graph and parameter registration |


```python
def classify_issue(shape_ok, dtype_ok, device_ok, has_nonfinite, grad_ok):
    """Collect anomaly categories; the list order indicates the initial teaching check order."""
    issues = []
    if not shape_ok:
        issues.append('shape')
    if not dtype_ok:
        issues.append('dtype')
    if not device_ok:
        issues.append('device')
    if has_nonfinite:
        issues.append('numerical')
    if not grad_ok:
        issues.append('gradient')
    return issues


def build_validity_summary(issues, first_failure=None, reproducible=False):
    """Turn correctness checks into a validity conclusion that later experiments can read."""
    issue_list = list(issues)
    if any(not isinstance(issue, str) or not issue.strip() for issue in issue_list):
        raise ValueError('异常类别必须是非空字符串')
    run_valid = len(issue_list) == 0
    if run_valid and first_failure is not None:
        raise ValueError('正常运行不能同时声明首个失败点')
    return {
        'evidence_type': 'correctness',
        'status': 'valid' if run_valid else 'invalid',
        'run_valid': run_valid,
        'issues': issue_list,
        'first_failure': first_failure,
        'reproducible': bool(reproducible),
        'next_action': 'collect_performance_evidence' if run_valid else 'localize_and_reproduce',
    }


sample = classify_issue(shape_ok=False, dtype_ok=True, device_ok=False, has_nonfinite=True, grad_ok=False)
primary = sample[0] if sample else 'ok'
sample_validity = build_validity_summary(sample)
healthy_validity = build_validity_summary([])
print('issues:', sample)
print('primary:', primary)
print('validity:', sample_validity)
# Output example: issues -> ['shape', 'device', 'numerical', 'gradient']; primary -> shape

```

## Q1 verification: Are the anomaly classification and the run-validity gate consistent?

It covers both the path with multiple anomalies and the fully normal path; only the normal path can continue to collect performance evidence.

```python
assert sample == ['shape', 'device', 'numerical', 'gradient']
assert primary == 'shape'
assert classify_issue(True, True, True, False, True) == []
assert sample_validity['status'] == 'invalid'
assert sample_validity['next_action'] == 'localize_and_reproduce'
assert healthy_validity['run_valid']
assert healthy_validity['next_action'] == 'collect_performance_evidence'
print('✅ 异常分类与运行有效性门槛通过')
```

## Q2: When do you look at shape, dtype, device, and layout first?

When you hit an error, check in order from easiest to hardest: shape determines the tensor's dimensions and broadcasting, dtype determines whether the data types are compatible, device determines whether the model and input are in the same compute location, and layout / contiguous affects whether an operator can access this memory directly. Before the contract is aligned, do not rush to change the optimization strategy; here a CPU Tensor is used to demonstrate how these fields are checked.


```python
import torch


def audit_tensor_contract(actual, expected):
    """Compare the tensor contract item by item and return the first field that must be fixed."""
    report = {}
    report['shape_ok'] = actual['shape'] == expected['shape']
    report['dtype_ok'] = actual['dtype'] == expected['dtype']
    report['device_ok'] = actual['device'] == expected['device']
    report['contiguous_ok'] = actual['contiguous'] == expected['contiguous']
    first_failure = next((name for name, ok in report.items() if not ok), None)
    return {'checks': report, 'first_failure': first_failure}


tensor = torch.randn(2, 4, 8)
view = tensor.transpose(1, 2)
actual = {
    'shape': tuple(view.shape),
    'dtype': str(view.dtype),
    'device': str(view.device),
    'contiguous': view.is_contiguous(),
}
expected = {'shape': (2, 8, 4), 'dtype': 'torch.float32', 'device': 'cpu', 'contiguous': True}
report = audit_tensor_contract(actual, expected)
print('contract:', report)
print('must_fix:', [k for k, v in report['checks'].items() if not v])
# Output example: contiguous_ok=False in contract, must_fix=['contiguous_ok']

```

## Q2 verification: Can the tensor contract locate the first failing field?

Construct a view that violates only the contiguity requirement, so that multiple errors do not hide the localization result at the same time.


```python
assert report['first_failure'] == 'contiguous_ok'
assert report['checks']['shape_ok'] and report['checks']['dtype_ok'] and report['checks']['device_ok']
assert not report['checks']['contiguous_ok']
print('✅ 张量契约的首个失败字段定位通过')
```

## Q3: When do you check NaN / Inf first, and how do you find the first abnormal source?

If the loss finally becomes NaN, the real problem may appear earlier in an activation or gradient. Check in the actual execution order of input → forward activation → loss → backward gradients, and record the stage and object of the first NaN / Inf; the first abnormal position is usually closer to the cause than the NaN seen last, and can also point directly to the next verification.


```python
def first_nonfinite_tensor(records):
    """Return the first Tensor containing NaN/Inf in execution order."""
    if not isinstance(records, list):
        raise TypeError('records 必须是按执行顺序排列的列表')
    for idx, (source, value) in enumerate(records):
        if not isinstance(value, torch.Tensor):
            raise TypeError('每条记录的值必须是 Tensor')
        finite = torch.isfinite(value)
        if not finite.all():
            return {
                'index': idx,
                'source': source,
                'nonfinite_count': int((~finite).sum().item()),
            }
    return None


trace = [
    ('embedding', torch.tensor([0.2, 0.8])),
    ('attention', torch.tensor([1.0, 2.0])),
    ('mlp', torch.tensor([0.5, float('nan')])),
    ('loss', torch.tensor(float('nan'))),
]
failure = first_nonfinite_tensor(trace)
print('trace:', trace)
print('first_nonfinite_tensor:', failure)
print('source:', failure['source'] if failure else 'ok')

```

## Q3 verification: Can both the first non-finite value and the normal path be identified?

```python
assert failure == {'index': 2, 'source': 'mlp', 'nonfinite_count': 1}
assert first_nonfinite_tensor([('loss', torch.tensor(1.0))]) is None
print('✅ 第一处非有限值定位通过')
```

## Q4: Why is the gradient always `None`, and how do you distinguish the reasons for the break?

"grad is None" does not necessarily have only one cause: it may be that detach() cut off the current Tensor, that no_grad() disabled gradient tracking, that a non-leaf Tensor does not keep gradients by default, or that the parameter was not registered in the module at all. You also need to distinguish "grad is None" from "the value of grad is 0": the former means no gradient was produced or kept, and the latter means the gradient already exists and only its current value is zero. First determine which case it is from the position where it appears, then decide whether to check the computation graph, the gradient attributes, or the module parameters.


```python
import warnings
import torch.nn as nn

leaf = torch.tensor([1.0], requires_grad=True)
detached = leaf.detach()
with torch.no_grad():
    no_grad_value = leaf * 2
non_leaf = leaf * 3
non_leaf.sum().backward()
with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    non_leaf_grad_without_retain = non_leaf.grad

zero_grad_leaf = torch.tensor([2.0], requires_grad=True)
(zero_grad_leaf * 0).sum().backward()

class BrokenModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_weight = torch.randn(2, 2, requires_grad=True)

broken = BrokenModule()
gradient_cases = {
    'detached_requires_grad': detached.requires_grad,
    'no_grad_requires_grad': no_grad_value.requires_grad,
    'non_leaf_grad': non_leaf_grad_without_retain,
    'zero_grad': zero_grad_leaf.grad.clone(),
    'registered_parameters': dict(broken.named_parameters()),
}
print('gradient cases:', gradient_cases)

```

## Q4 verification: Can `grad=None`, `grad=0`, and unregistered parameters be distinguished?

```python
assert not gradient_cases['detached_requires_grad']
assert not gradient_cases['no_grad_requires_grad']
assert gradient_cases['non_leaf_grad'] is None
assert torch.equal(gradient_cases['zero_grad'], torch.zeros_like(zero_grad_leaf))
assert 'raw_weight' not in gradient_cases['registered_parameters']
print('✅ grad=None、grad=0、非叶子 Tensor 和未注册参数机制通过')
```

## Q5: How do you use module hooks to find the first anomaly inside the model?

When the input contract is normal but the final loss is still non-finite, the checkpoints need to be placed inside the model. A forward hook can record whether the output of each module is finite, helping find the first failing module; if the forward result is finite but backward fails, then use `torch.autograd.detect_anomaly()` or gradient hooks to check the backward path. Hooks should be removed after localization to avoid duplicate registration.


```python
class InjectNaN(nn.Module):
    def forward(self, x):
        return x / 0


probe_model = nn.Sequential(
    nn.Linear(2, 2),
    nn.ReLU(),
    InjectNaN(),
    nn.Linear(2, 1),
)
events = []
handles = []

def record_finiteness(name):
    def hook(_module, _inputs, output):
        events.append({'module': name, 'finite': bool(torch.isfinite(output).all())})
    return hook

for name, module in probe_model.named_children():
    handles.append(module.register_forward_hook(record_finiteness(name)))

_ = probe_model(torch.ones(1, 2))
for handle in handles:
    handle.remove()
first_bad_module = next(event['module'] for event in events if not event['finite'])
print('events:', events)
print('first bad module:', first_bad_module)

```

## Q5 verification: Does the hook locate the first non-finite module and get removed?

```python
assert first_bad_module == '2'
assert events[0]['finite'] and events[1]['finite']
assert not events[2]['finite'] and not events[3]['finite']
assert all(not module._forward_hooks for module in probe_model.children())
print('✅ 前向 hook 定位与清理通过')
```

## Q6: How do you shrink the first failure point into a replayable minimal reproduction?

After locating the failing module, keep the minimum information needed to trigger the failure: random seed, input values and shape, dtype, operation name, framework version, and the first failure position. A minimal reproduction does not carry the complete training pipeline; it should be able to retrigger the same failure in an independent function, making it easy to confirm consistency before and after the fix. Finally, write the anomaly category, the first failure point, and the reproduction state into the validity conclusion, and hand it to section 20 to decide whether to stop the experiment or continue collecting performance evidence.

| Reproduction field | Why keep it |
|:---|:---|
| seed and input | Fix the trigger conditions |
| shape / dtype | Fix the tensor contract |
| operation | Keep the minimal failing operation |
| first failure | Compare the localization result before and after the fix |
| framework version | Explain runtime differences |


```python
def build_minimal_repro(seed=7):
    """Construct and record a non-finite value case that can be replayed independently."""
    torch.manual_seed(seed)
    inputs = torch.tensor([-1.0, 4.0], dtype=torch.float32)
    outputs = torch.sqrt(inputs)
    return {
        'seed': seed,
        'input_values': inputs.tolist(),
        'shape': tuple(inputs.shape),
        'dtype': str(inputs.dtype),
        'operation': 'torch.sqrt',
        'first_failure': 'output',
        'nonfinite_count': int((~torch.isfinite(outputs)).sum().item()),
        'torch_version': torch.__version__,
    }

def replay_minimal_repro(record):
    """Replay only the sqrt/FP32 operation declared by this example and check the input shape."""
    required = {'input_values', 'shape', 'dtype', 'operation'}
    if not required <= set(record):
        raise KeyError('复现记录缺少输入、形状、类型或操作')
    if record['operation'] != 'torch.sqrt' or record['dtype'] != 'torch.float32':
        raise ValueError('本例只支持 torch.sqrt 和 torch.float32')
    inputs = torch.tensor(record['input_values'], dtype=torch.float32)
    if tuple(inputs.shape) != tuple(record['shape']):
        raise ValueError('记录形状与输入值不一致')
    outputs = torch.sqrt(inputs)
    return int((~torch.isfinite(outputs)).sum().item())

repro = build_minimal_repro()
replayed_nonfinite_count = replay_minimal_repro(repro)
validity_report = build_validity_summary(
    ['numerical'],
    first_failure=repro['first_failure'],
    reproducible=replayed_nonfinite_count == repro['nonfinite_count'],
)
print('minimal repro:', repro)
print('validity report:', validity_report)

```

## Q6 verification: Can the minimal reproduction retrigger the same failure?

```python
assert repro['operation'] == 'torch.sqrt'
assert repro['shape'] == (2,) and repro['dtype'] == 'torch.float32'
assert repro['nonfinite_count'] == 1
assert replayed_nonfinite_count == repro['nonfinite_count']
assert validity_report['status'] == 'invalid'
assert validity_report['first_failure'] == 'output'
assert validity_report['reproducible']
assert validity_report['next_action'] == 'localize_and_reproduce'
for change in ({'operation': 'torch.log'}, {'dtype': 'torch.float64'}, {'shape': (3,)}):
    try:
        replay_minimal_repro({**repro, **change})
    except ValueError:
        pass
    else:
        raise AssertionError('操作、类型或形状不一致时必须停止重放')
try:
    build_validity_summary([], first_failure='loss')
except ValueError:
    pass
else:
    raise AssertionError('正常运行不能带有失败点')
print('✅ 最小复现、输入一致性与有效性结论通过')
```

## Related Reading
**Overview:** After completing this section, you can go back and reinforce Autograd semantics, and bring reproducible correctness evidence into performance experiments.
- [Part 00 · 07 Autograd and Backward Propagation](./07_PyTorch_Autograd_and_Backward.md)
- [PyTorch anomaly detection official documentation](https://docs.pytorch.org/docs/stable/autograd.html#debugging-and-anomaly-detection)
- [Part 00 · 20 Performance Evidence and Experiment Design](./20_Performance_Evidence_and_Experiment_Design.md)

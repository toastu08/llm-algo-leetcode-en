# 09. PyTorch nn Module Basics | PyTorch nn.Module 基础

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `PyTorch`, `model encapsulation`, `state management` | **Audience:** learners preparing to read model definitions and training code

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/09_PyTorch_nn_Module_Basics.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## Section Guide

A model module organizes parameters, buffers, submodules, forward computation, and run mode into an object that can be composed recursively. The way state is initialized and registered determines whether it can participate in updates, migration, and saving.

After completing this section, you should be able to distinguish parameters, buffers, and ordinary attributes, check the initialization scale, and determine whether the frozen state is consistent with the model's registration state.

**Keywords:** `nn.Module`, `initialization`, `Parameter`, `train/eval`, `requires_grad`

![The object and registration relationships of nn.Module](../public/00_Prerequisites/09_module_registry_map.svg)

## Prerequisite Reading
**Overview:** Once tensors and gradients enter a model object, parameter registration determines which state the optimizer can update.
- [08. PyTorch Grad Hygiene and No-Grad](./08_PyTorch_Grad_Hygiene_and_No_Grad.md)
- [0C Group Page](./0C.md)

## Q1: How do you organize layers and submodules into a callable `nn.Module`?

A model is not a set of scattered functions, but a callable object that can be composed recursively. `__init__()` registers parameters and submodules, `forward()` describes the dataflow, and `model(x)` triggers the forward pass through the module call entry point; after the module is constructed, the parameters also need an initial scale that matches the size of the layer.


```python
import torch
import torch.nn as nn


class SimpleLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # A `Parameter` is automatically registered as a model parameter.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # `forward()` defines the computation logic; when actually calling, write `model(x)`.
        return x @ self.weight.t() + self.bias


class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Submodules participate recursively in parameter registration and state management.
        self.fc1 = SimpleLinear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = SimpleLinear(hidden_dim, output_dim)

    def forward(self, x):
        # A typical module composition: linear layer -> activation -> linear layer.
        return self.fc2(self.act(self.fc1(x)))


def count_parameters(module):
    """Count the total number of elements of registered parameters in the module and its submodules."""
    return sum(param.numel() for param in module.parameters())


model = TwoLayerMLP(4, 8, 2)
print('参数总量：', count_parameters(model))
print('named_parameters：', [name for name, _ in model.named_parameters()])

```

## Q1 verification: Are parameter registration, submodules, and forward output correct?

Two things are checked directly here: whether the custom layer runs, and whether the MLP's output shape and total parameter count match expectations.


```python
def test_simple_linear():
    layer = SimpleLinear(2, 1)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2.0, -1.0]]))
        layer.bias.copy_(torch.tensor([0.5]))
    x = torch.tensor([[3.0, 4.0]])
    y = layer(x)
    assert torch.allclose(y, torch.tensor([[2.5]]))
    print('✅ SimpleLinear 通过')


def test_two_layer_mlp():
    model = TwoLayerMLP(4, 8, 2)
    x = torch.randn(3, 4)
    y = model(x)
    assert y.shape == (3, 2)
    expected_params = (4 * 8 + 8) + (8 * 2 + 2)
    assert count_parameters(model) == expected_params
    print('✅ TwoLayerMLP 通过')


test_simple_linear()
test_two_layer_mlp()

```

## Q2: Why does parameter initialization need to consider both scale and reproducibility?

Initialization determines the scale of signals and gradients at the start of training. A linear layer can set the Xavier range based on the input and output dimensions, and the bias usually starts at zero; a fixed random seed makes the same initialization scheme reviewable. The initialization strategy should still be chosen together with the downstream activation functions and residual structure.


```python
def reset_linear_parameters(layer, seed):
    """Reset a CPU linear layer with Xavier uniform and return an initialization scale summary."""
    if not isinstance(layer, nn.Linear):
        raise TypeError('layer 必须是 nn.Linear')
    if not isinstance(seed, int) or seed < 0:
        raise ValueError('seed 必须是非负整数')
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        nn.init.xavier_uniform_(layer.weight, generator=generator)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    bound = (6.0 / (layer.in_features + layer.out_features)) ** 0.5
    return {
        'fan_in': layer.in_features,
        'fan_out': layer.out_features,
        'xavier_bound': bound,
        'weight_std': float(layer.weight.std()),
    }


linear_a = nn.Linear(4, 8)
linear_b = nn.Linear(4, 8)
linear_c = nn.Linear(4, 8)
initialization = reset_linear_parameters(linear_a, seed=7)
reset_linear_parameters(linear_b, seed=7)
reset_linear_parameters(linear_c, seed=11)
print('初始化摘要：', initialization)

```

## Q2 verification: Are the random seed, Xavier range, and bias starting point correct?

The same seed should produce the same parameters, and different seeds should change the weights; all weights should lie within the theoretical Xavier range, and the bias should start at zero.


```python
assert torch.equal(linear_a.weight, linear_b.weight)
assert not torch.equal(linear_a.weight, linear_c.weight)
assert torch.count_nonzero(linear_a.bias) == 0
assert float(linear_a.weight.abs().max()) <= initialization['xavier_bound'] + 1e-6
assert initialization['weight_std'] > 0
assert linear_a(torch.ones(2, 4)).shape == (2, 8)
print('✅ 参数初始化尺度与可复现性通过')

```

## Q3: Where do `Parameter`, `buffer`, and ordinary attributes each go?

When reading a module, first use three questions to determine where an object belongs: "is it trained, is it saved, does it migrate with the module". A parameter is updated only when it is given to the optimizer and has a gradient; a buffer is saved by default but can be excluded with `persistent=False`.

| Object | Updated by the optimizer | Enters `state_dict()` | Migrates device with the module | Common use |
|:---|:---:|:---:|:---:|:---|
| `Parameter` | When given to the optimizer and has a gradient | Yes | Yes | weights, biases |
| `buffer` | No | Saved by default | Yes | masks, statistics, fixed scale values |
| Ordinary attribute | No | No | No | configuration, names, temporary flags |


```python
class ToyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        # A `buffer` is often used for state such as masks, statistics, and caches that "should be saved but not trained".
        self.register_buffer('scale', torch.tensor([2.0]))
        # An ordinary attribute does not enter state_dict and is not automatically migrated to another device with the module.
        self.name = 'toy'

    def forward(self, x):
        return x * self.weight * self.scale


m = ToyModule()
print('parameters：', [n for n, _ in m.named_parameters()])
print('buffers：', [n for n, _ in m.named_buffers()])
print('state_dict keys：', list(m.state_dict().keys()))

```

## Q3 verification: Does state_dict contain both parameters and buffers?

Here it confirms that `state_dict()` includes parameters and buffers, but does not treat ordinary attributes as savable state.


```python
m = ToyModule()
keys = list(m.state_dict().keys())
assert 'weight' in keys
assert 'scale' in keys
assert 'name' not in keys
m.register_buffer('temporary_mask', torch.ones(1), persistent=False)
assert 'temporary_mask' in dict(m.named_buffers())
assert 'temporary_mask' not in m.state_dict()
print('✅ state_dict 结构通过')

```

## Q4: What do `train/eval` and gradient mode control respectively?

`train()` and `eval()` change the run behavior of modules such as Dropout and BatchNorm, and `torch.no_grad()` controls whether the computation graph is recorded; the two solve different problems. The validation stage usually uses `model.eval()` and `torch.no_grad()` together, and the training stage restores `model.train()`.


```python
class ModeAwareBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=1.0)

    def forward(self, x):
        # The multiplication ensures the eval path also produces a new result, so gradient mode can be checked separately.
        return self.dropout(x) * 1.0


mode_model = ModeAwareBlock()
mode_input = torch.ones(2, 3, requires_grad=True)
mode_model.train()
train_output = mode_model(mode_input)
mode_model.eval()
with torch.no_grad():
    eval_output = mode_model(mode_input)
print('train / eval output：', train_output, eval_output)

```

## Q4 verification: Are module mode and gradient mode independent of each other?

Dropout in training mode changes the output; evaluation mode restores deterministic behavior, and `no_grad()` additionally ensures the evaluation output is not connected to the computation graph.


```python
assert mode_model.training is False
assert torch.equal(train_output, torch.zeros_like(train_output))
assert torch.equal(eval_output, mode_input)
assert train_output.requires_grad
assert not eval_output.requires_grad
print('✅ train/eval 与梯度模式检查通过')

```

## Q5: After freezing parameters, how does registration state differ from trainable state?

`requires_grad=False` only turns off gradient computation for that parameter; it does not remove the parameter from the module or make it disappear from `state_dict()`. When constructing the optimizer, you should explicitly filter the trainable parameters so that "registered" and "should be updated" are not conflated.

| Check dimension | Entry point | Result for a frozen parameter |
|:---|:---|:---|
| Is it registered | `named_parameters()` | Still present |
| Is it saved | `state_dict()` | Still saved |
| Does it compute gradients | `requires_grad` and `.grad` | Off; `.grad` is `None` after backward |
| Is it given to the optimizer | `optimizer.param_groups` | Depends on whether it was filtered when constructing the optimizer |


```python
def split_parameter_roles(module):
    """Split registered parameters by requires_grad and return lists of names."""
    trainable = []
    frozen = []
    for name, parameter in module.named_parameters():
        (trainable if parameter.requires_grad else frozen).append(name)
    return {'trainable': trainable, 'frozen': frozen}


model = TwoLayerMLP(4, 8, 2)
for parameter in model.fc1.parameters():
    parameter.requires_grad_(False)

roles = split_parameter_roles(model)
optimizer = torch.optim.SGD(
    [parameter for parameter in model.parameters() if parameter.requires_grad],
    lr=0.1,
)
print('parameter roles：', roles)

```

## Q5 verification: Are frozen parameters still saved without producing gradients?

After freezing, `fc1` should still appear in the parameter registry and in `state_dict()`; it has no gradient after backward propagation, and the optimizer in this example also holds only the trainable parameters of `fc2`.


```python
x = torch.randn(3, 4)
loss = model(x).pow(2).mean()
loss.backward()

assert roles['frozen'] == ['fc1.weight', 'fc1.bias']
assert roles['trainable'] == ['fc2.weight', 'fc2.bias']
assert {'fc1.weight', 'fc1.bias'} <= dict(model.named_parameters()).keys()
assert {'fc1.weight', 'fc1.bias'} <= set(model.state_dict())
assert all(parameter.grad is None for parameter in model.fc1.parameters())
assert all(parameter.grad is not None for parameter in model.fc2.parameters())
assert sum(len(group['params']) for group in optimizer.param_groups) == 2
print('✅ 冻结与可训练参数语义通过')

```

## Related Reading

With the module object established, the next page continues with state serialization, restoration, and structural migration.

- [10. PyTorch State, Checkpoint, and Recovery](./10_PyTorch_State_dict_and_Persistence.md)
# 14. Activation Functions | 激活函数

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `PyTorch`, `nonlinearity`, `gated activation` | **Audience:** learners preparing to understand MLPs and gated structures

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/14_Activation_Functions.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## Section Guide

Activation functions change the forward output and the local gradient, and keep a stack of linear transformations from degenerating into a single linear mapping. In a Transformer MLP, the activation can sit on a single projection path, or participate in element-wise modulation of the gate/value branches.

After completing this section, you should be able to compare the outputs and local gradients of ReLU, GELU, and SiLU, and trace the `H → H_ffn → H` shape path of a plain FFN and a gated FFN.

**Keywords:** `ReLU`, `GELU`, `SiLU`, `SwiGLU`, `FFN`

![Activation output, gradient, and gating relationships](../public/00_Prerequisites/14_activation_gradient_map.svg)

## Prerequisite Reading
**Overview:** Tensor operations give the forward values, Autograd gives the local gradients, and section 13 gives the position where the MLP branch writes back to the residual stream.
- [07. PyTorch Autograd and Backward](./07_PyTorch_Autograd_and_Backward.md)
- [13. Transformer Block Dataflow and Residual Stream](./13_Transformer_Block_Dataflow_and_Residual_Stream.md)
- [0D Group Page](./0D.md)

## Q1: What problem do activation functions solve in a model?

Without an activation function, multiple layers of linear transformations can still be merged into a single linear transformation. ReLU, GELU, and SiLU all introduce nonlinearity, but they handle negative values, the region near zero, and large positive values differently.


```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def relu(x):
    return torch.where(x > 0, x, torch.zeros_like(x))


def gelu_exact(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def silu(x):
    return x * torch.sigmoid(x)


def activation_summary(x):
    return {
        'relu_mean': float(relu(x).mean().item()),
        'gelu_mean': float(gelu_exact(x).mean().item()),
        'silu_mean': float(silu(x).mean().item()),
    }


x = torch.linspace(-3, 3, steps=7)
print('输入：', x.tolist())
print('ReLU：', relu(x).tolist())
print('GELU：', [round(v, 4) for v in gelu_exact(x).tolist()])
print('SiLU：', [round(v, 4) for v in silu(x).tolist()])

```

## Q1 verification: Are the output shapes of the different activations visible?

Here the same set of inputs is fed directly to different activations to see whether their handling of negative, zero, and positive values matches expectations.


```python
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
assert torch.allclose(relu(x), F.relu(x))
assert torch.allclose(gelu_exact(x), F.gelu(x), atol=1e-6, rtol=1e-6)
assert torch.allclose(silu(x), F.silu(x))
print('✅ 基本激活函数通过')

```

## Q2: When the output shapes are similar, how do the gradient behaviors differ?

An activation function changes not only the output but also the local derivative. ReLU has zero gradient on the negative half-axis and is not differentiable at zero; this example uses PyTorch ReLU's zero-point gradient convention, which is 0. GELU and SiLU retain smooth changes near zero. Comparing outputs and gradients on the same input explains training behavior better than remembering formulas alone.


```python
def value_and_gradient(function, values):
    """Return the activation output and the gradient of the output with respect to the input."""
    inputs = values.clone().detach().requires_grad_(True)
    outputs = function(inputs)
    outputs.sum().backward()
    return outputs.detach(), inputs.grad.detach()


points = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0])
for name, function in {'relu': relu, 'gelu': gelu_exact, 'silu': silu}.items():
    values, gradients = value_and_gradient(function, points)
    print(name, 'values=', values.tolist(), 'gradients=', gradients.tolist())

```

## Q2 verification: Are the cut-off region and the smooth region observable?

It checks that the ReLU gradient is zero on the negative half-axis, and confirms that the outputs and gradients of GELU and SiLU remain finite.


```python
relu_values, relu_gradients = value_and_gradient(relu, points)
gelu_values, gelu_gradients = value_and_gradient(gelu_exact, points)
silu_values, silu_gradients = value_and_gradient(silu, points)
_, reference_relu_gradients = value_and_gradient(F.relu, points)
assert torch.equal(relu_gradients, reference_relu_gradients)
assert relu_gradients[2].item() == 0.0
assert torch.equal(relu_gradients[:2], torch.zeros(2))
assert torch.isfinite(gelu_values).all() and torch.isfinite(gelu_gradients).all()
assert torch.isfinite(silu_values).all() and torch.isfinite(silu_gradients).all()
assert not torch.equal(gelu_gradients[:2], torch.zeros(2))
print('✅ 激活输出与梯度通过')

```

## Q3: How do activation functions enter a plain FFN and a gated FFN?

A plain FFN first expands the hidden dimension to `H_ffn`, applies the activation, and then projects back to `H`. A gated FFN generates gate and value at the same time, and the two branches must have the same shape; SwiGLU uses `SiLU(gate) × value` to modulate the content, and the down projection then restores the hidden shape of the residual stream.


```python
def swiglu(gate, value):
    """Gate the value branch with SiLU; the two inputs must have the same shape."""
    if gate.shape != value.shape:
        raise ValueError('gate and value must have the same shape')
    return silu(gate) * value


class DenseFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.down(F.gelu(self.up(hidden_states)))


class GatedFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.down(swiglu(self.gate(hidden_states), self.up(hidden_states)))


torch.manual_seed(0)
ffn_input = torch.randn(2, 4, 8)
dense_ffn = DenseFFN(hidden_size=8, intermediate_size=16)
gated_ffn = GatedFFN(hidden_size=8, intermediate_size=16)
print('input / dense / gated：', ffn_input.shape, dense_ffn(ffn_input).shape, gated_ffn(ffn_input).shape)

```

## Q3 verification: Do both kinds of FFN restore the hidden shape of the residual stream?

The internal intermediate dimension can expand, but the input shape must be restored before writing back to the Block; when the gate and value of the gated branch do not have the same shape, it should fail immediately.


```python
dense_output = dense_ffn(ffn_input)
gated_output = gated_ffn(ffn_input)
assert dense_output.shape == gated_output.shape == ffn_input.shape

gate = torch.tensor([[-2.0, 0.0, 2.0]])
value = torch.tensor([[3.0, 3.0, 3.0]])
torch.testing.assert_close(swiglu(gate, value), F.silu(gate) * value)
assert swiglu(gate, value)[0, 1].item() == 0.0
try:
    swiglu(torch.ones(2, 3), torch.ones(2, 4))
except ValueError:
    pass
else:
    raise AssertionError('shape mismatch should fail')
print('✅ Dense FFN / Gated FFN shape contract 通过')

```

## Q4: How do you compare the local behavior of activation functions with the same set of statistics?

A local comparison can record the output scale, the zero fraction, and the gradient scale at the same time. Looking only at the mean misses ReLU's hard cut-off and cannot show whether a smooth activation keeps gradients on the negative half-axis; these statistics describe the mechanism and do not directly replace model training and quality comparisons.


```python
def activation_profile(function, values):
    """Summarize activation outputs and local gradients without converting statistics into preset labels."""
    outputs, gradients = value_and_gradient(function, values)
    return {
        'output_mean': float(outputs.mean()),
        'output_std': float(outputs.std(unbiased=False)),
        'zero_fraction': float(outputs.eq(0).float().mean()),
        'gradient_abs_mean': float(gradients.abs().mean()),
        'gradient_abs_max': float(gradients.abs().max()),
    }


sample = torch.linspace(-4, 4, steps=17)
profiles = {name: activation_profile(function, sample) for name, function in {
    'relu': relu,
    'gelu': gelu_exact,
    'silu': silu,
}.items()}
for name, profile in profiles.items():
    print(name, profile)

```

## Q4 verification: Can the statistics distinguish cut-off from smooth activations?

ReLU should show a clear zero fraction; GELU and SiLU should not produce the same hard cut-off on this input. All statistics must remain finite.


```python
assert profiles['relu']['zero_fraction'] > 0.4
assert profiles['gelu']['zero_fraction'] < profiles['relu']['zero_fraction']
assert profiles['silu']['zero_fraction'] < profiles['relu']['zero_fraction']
for profile in profiles.values():
    assert all(torch.isfinite(torch.tensor(value)) for value in profile.values())
print('✅ 激活统计对照通过')

```

## Related Reading

Continue by observing how normalization controls the scale of branch inputs, and how activations and gating enter a real Transformer Block.

- [15. Normalization Techniques](./15_Normalization_Techniques.md)
- [Part 02 · 02 SwiGLU](../02_PyTorch_Algorithms/02_SwiGLU_Activation.md)
- [Part 02 · 05 LLaMA3 Block](../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.md)
- [LLM Architecture Evolution · MLP / FFN evolution](../topic_discussion/llm_architecture_evolution/07_mlp_ffn_evolution.md)

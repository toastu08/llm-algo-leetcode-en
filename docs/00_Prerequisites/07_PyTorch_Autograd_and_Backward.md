# 07. PyTorch Autograd and Backward | PyTorch 自动求导与反向传播

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `PyTorch`, `autograd`, `backward propagation` | **Audience:** learners preparing to study the training loop, backward state, or memory mechanisms

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/07_PyTorch_Autograd_and_Backward.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## Section Guide

The forward pass connects tensors and operators into computational dependencies, and the backward pass sends upstream gradients back to the parameters along these dependencies. Non-scalar outputs, branches, residuals, and shared parameters change how gradients flow and merge.

After completing this section, you should be able to explain the vector-Jacobian product, the state saved for backward, and gradient accumulation, and determine why a gradient path exists, merges, or is interrupted.

**Keywords:** `requires_grad`, `backward()`, `VJP`, `saved_tensors`

![Concept relationships in this section](../public/00_Prerequisites/07_autograd_backward_map.svg)

## Prerequisite Reading
**Overview:** First understand tensor shape, layout, and indexing, then turn your attention to computational dependencies and gradient flow.
- [06. PyTorch Tensor Layout and Indexing](./06_PyTorch_Tensor_Layout_and_Indexing.md)
- [0B Group Page](./0B.md)

## Q1: How does autograd send a scalar gradient back along the computation graph?

After a leaf tensor with `requires_grad=True` participates in operations, the result records a `grad_fn`. When `backward()` is called on a scalar result, PyTorch starts from the upstream gradient 1, applies the chain rule, and accumulates the result into the leaf tensor's `.grad`.

```python
import torch


x = torch.tensor(2.0, requires_grad=True)
y = x * x + 3 * x
print('y:', y.item(), 'grad_fn:', type(y.grad_fn).__name__)
y.backward()
print('x.grad:', x.grad.item())
```

## Q1 verification: Are the analytic gradient and the accumulation behavior correct?

```python
x = torch.tensor(2.0, requires_grad=True)
(x * x + 3 * x).backward()
assert x.grad.item() == 7.0
(x * x + 3 * x).backward()
assert x.grad.item() == 14.0
x.grad.zero_()
assert x.grad.item() == 0.0
print('✅ 标量反向传播与梯度累积通过')
```

## Q2: What do leaf tensors, `grad_fn`, and `retain_grad()` each describe?

A leaf tensor is where the gradient is ultimately written; a non-leaf tensor produced by an operation records its source through `grad_fn` but does not keep `.grad` by default. When debugging intermediate gradients, you can call `retain_grad()`, which does not change the differentiation formula and only adds an observable state.

```python
leaf = torch.tensor(2.0, requires_grad=True)
intermediate = leaf * 3
intermediate.retain_grad()
loss = intermediate.square()
loss.backward()

print({
    'leaf_is_leaf': leaf.is_leaf,
    'intermediate_is_leaf': intermediate.is_leaf,
    'intermediate_grad_fn': type(intermediate.grad_fn).__name__,
    'leaf_grad': leaf.grad.item(),
    'intermediate_grad': intermediate.grad.item(),
})
```

## Q2 verification: Do the gradients of the leaf and the intermediate result follow the chain rule?

```python
leaf = torch.tensor(2.0, requires_grad=True)
intermediate = leaf * 3
intermediate.retain_grad()
intermediate.square().backward()
assert leaf.is_leaf is True and intermediate.is_leaf is False
assert leaf.grad.item() == 36.0
assert intermediate.grad.item() == 12.0
print('✅ 叶子张量、grad_fn 与 retain_grad 通过')
```

## Q3: Why does a non-scalar output need an explicit upstream gradient?

A vector output has no unique "derivative with respect to all elements" result. `backward(gradient)` actually computes a vector-Jacobian product: given an upstream vector `v`, it returns `vᵀJ`. Different upstream gradients mean that the downstream computation assigns different weights to each output position.

```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x.square()
upstream = torch.tensor([1.0, 0.5])
y.backward(upstream)

print('output:', y.tolist())
print('upstream:', upstream.tolist())
print('vjp:', x.grad.tolist())
```

## Q3 verification: Does the VJP equal the upstream gradient multiplied by the local derivative?

```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
upstream = torch.tensor([1.0, 0.5])
x.square().backward(upstream)
expected = upstream * 2 * x.detach()
assert torch.allclose(x.grad, expected)

try:
    torch.tensor([2.0, 3.0], requires_grad=True).square().backward()
except RuntimeError as error:
    assert 'scalar' in str(error)
else:
    raise AssertionError('非标量输出缺少上游梯度时应失败')
print('✅ 向量—雅可比积通过')
```

## Q4: Why does backward propagation need to save forward tensors?

Some local derivatives depend on the forward input or output, so autograd saves the tensors needed for backward propagation and reads them during the backward stage. `saved_tensors_hooks` can observe this save-read chain; it illustrates the backward dependency and is not the same as an intermediate tensor's `.grad`. The number and lifetime of saved objects further affect training memory.

```python
events = []


def pack_hook(tensor):
    """Record the shape of a tensor saved by the forward pass and return the saved object unchanged."""
    events.append(('save', tuple(tensor.shape)))
    return tensor


def unpack_hook(tensor):
    """Record the shape of a tensor read by the backward pass and restore the saved object."""
    events.append(('load', tuple(tensor.shape)))
    return tensor


x = torch.tensor([2.0, 3.0], requires_grad=True)
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    loss = x.square().sum()
    loss.backward()

print('saved tensor events:', events)
```

## Q4 verification: Do saving and reading occur in the forward and backward stages respectively?

```python
events = []
x = torch.tensor([2.0, 3.0], requires_grad=True)
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    x.square().sum().backward()

event_names = [name for name, _ in events]
assert 'save' in event_names and 'load' in event_names
assert event_names.index('save') < event_names.index('load')
assert torch.allclose(x.grad, torch.tensor([4.0, 6.0]))
print('✅ 反向保存张量的生命周期通过')
```

## Q5: How does a custom `Function` close the forward-backward contract?

The custom `Function`'s `forward()` computes the output and uses `ctx.save_for_backward()` to save the state needed for the local derivative; `backward()` reads this state, multiplies the upstream gradient by the local derivative, and returns the input gradient. The number of returned gradients must correspond to the inputs that require gradients.

```python
from torch.autograd import Function


class Square(Function):
    """A minimal Function that demonstrates saving state in forward and consuming the upstream gradient in backward."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.square()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * 2 * x


x = torch.tensor(3.0, requires_grad=True)
Square.apply(x).backward()
print('custom gradient:', x.grad.item())
```

## Q5 verification: Does the custom gradient pass the analytic value and gradcheck?

```python
x = torch.tensor(3.0, requires_grad=True)
Square.apply(x).backward()
assert x.grad.item() == 6.0

gradcheck_input = torch.tensor([1.5, -2.0], dtype=torch.double, requires_grad=True)
assert torch.autograd.gradcheck(Square.apply, (gradcheck_input,))
print('✅ 自定义 autograd 通过')
```

## Q6: How do the gradients of branches, residuals, and shared parameters merge?

When a tensor is used by multiple branches, backward propagation adds the gradients returned by each branch. The residual connection `main + x` provides an identity path with local derivative 1 for `x`; when a shared parameter participates in both the main branch and the regularization branch, the contributions of both paths accumulate into the parameter's `.grad`. Three kinds of quantities need to be distinguished here: the upstream gradient is the backward input received by a node, an intermediate tensor's gradient is only kept if `retain_grad()` is called, and a leaf parameter's `.grad` is the accumulated result of the contributions of all reachable branches.

```python
x = torch.tensor(2.0, requires_grad=True)
shared_weight = torch.tensor(3.0, requires_grad=True)

main = shared_weight * x
merged = main + x  # residual identity path
merged.retain_grad()
main_loss = merged.square()
regularization_loss = shared_weight.square()
loss = main_loss + regularization_loss
loss.backward()

print({
    'upstream_at_merged': merged.grad.item(),
    'x_grad_after_residual_merge': x.grad.item(),
    'shared_weight_grad_after_branch_merge': shared_weight.grad.item(),
})
```

## Q6 verification: Are the identity path and the branch contributions of the shared parameter added correctly?

```python
upstream_to_merged = 2 * merged.detach()
expected_x_grad = upstream_to_merged * (shared_weight.detach() + 1)
weight_grad_from_main = upstream_to_merged * x.detach()
weight_grad_from_regularization = 2 * shared_weight.detach()

assert torch.allclose(merged.grad, upstream_to_merged)
assert torch.allclose(x.grad, expected_x_grad)
assert torch.allclose(
    shared_weight.grad,
    weight_grad_from_main + weight_grad_from_regularization,
)
print('✅ 残差恒等路径与共享参数的梯度汇合通过')
```

### Section Summary

- Scalar `backward()` starts from the upstream gradient 1, and non-scalar outputs need an explicit upstream gradient.
- `retain_grad()` keeps intermediate gradients; saved-for-backward tensors save the forward state required by the local backward formula, and the two have different responsibilities.
- A custom `Function` must close the loop between the information saved in forward, the inputs and outputs of backward, and the number of gradients.
- Branches merge gradients during the backward stage; the residual identity path and shared parameters both change the contribution finally accumulated into `.grad`.

## Related Reading
**Overview:** Continue with gradient boundaries in the training loop, and connect saved-for-backward tensors to training state and the memory ledger.
- [PyTorch Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [08. PyTorch Grad Hygiene and No-Grad](./08_PyTorch_Grad_Hygiene_and_No_Grad.md)
- [Backpropagation and Training Mechanics](../topic_discussion/backpropagation_training_mechanism/intro.md)

# 08. PyTorch Grad Hygiene and No Grad | PyTorch 梯度边界与状态控制

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `PyTorch`, `gradient control`, `parameter freezing` | **Audience:** learners who know basic backward and are preparing to study the training loop and fine-tuning

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/08_PyTorch_Grad_Hygiene_and_No_Grad.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## Section Guide

Training code needs to manage module behavior, computation graph recording, historical gradients, and parameter trainability at the same time. These four kinds of state are independent of one another, and conflating them causes unintended graph construction, residual gradients, or incorrect freezing.

After completing this section, you should be able to distinguish the four control boundaries and determine which states should change during training, validation, local cut-off, and parameter freezing.

**Keywords:** `zero_grad`, `no_grad`, `detach`, `requires_grad`

![Gradient control concept relationships](../public/00_Prerequisites/08_grad_hygiene_map.svg)

## Prerequisite Reading
**Overview:** First understand backward, VJP, and saved tensors, then study the gradient switches used in training, validation, and parameter freezing.
- [07. PyTorch Autograd and Backward](./07_PyTorch_Autograd_and_Backward.md)
- [0B Group Page](./0B.md)

## Q1: What do module mode and gradient mode control respectively?

`model.train()` / `eval()` change the behavior of modules such as Dropout and BatchNorm, but do not automatically enable or disable gradients; `no_grad()` and `inference_mode()` are what control whether the current context records the computation graph. Validation and inference usually need the correct module mode and gradient mode set at the same time.

```python
import torch
import torch.nn as nn


dropout = nn.Dropout(p=1.0)
x = torch.ones(1, 2, requires_grad=True)

dropout.train()
train_output = dropout(x)
dropout.eval()
eval_output = dropout(x) * 2
with torch.no_grad():
    no_grad_output = dropout(x) * 2
with torch.inference_mode():
    inference_output = dropout(x) * 2

print({
    'train_output': train_output.tolist(),
    'eval_output': eval_output.tolist(),
    'eval_requires_grad': eval_output.requires_grad,
    'no_grad_requires_grad': no_grad_output.requires_grad,
    'inference_tensor': torch.is_inference(inference_output),
})
```

## Q1 verification: Does `eval()` still build the graph?

```python
dropout = nn.Dropout(p=1.0)
x = torch.ones(1, 2, requires_grad=True)

dropout.train()
train_output = dropout(x)
dropout.eval()
eval_output = dropout(x) * 2
with torch.no_grad():
    no_grad_output = dropout(x) * 2
with torch.inference_mode():
    inference_output = dropout(x) * 2

assert torch.equal(train_output, torch.zeros_like(x))
assert torch.equal(eval_output, x * 2)
assert eval_output.requires_grad is True
assert no_grad_output.requires_grad is False and not torch.is_inference(no_grad_output)
assert inference_output.requires_grad is False and torch.is_inference(inference_output)
print('✅ 模块模式与梯度模式通过')
```

## Q2: Why do gradients accumulate, and what does `set_to_none=True` change?

By default, PyTorch adds new gradients to the existing `.grad`. Before a new parameter update, the old gradients usually need to be cleared. In-place zeroing keeps a zero tensor; `optimizer.zero_grad(set_to_none=True)` sets `.grad` to `None`, and the next backward creates the gradient again. Both start a new accumulation window, but the state representation differs.

```python
parameter = nn.Parameter(torch.tensor(1.0))
optimizer = torch.optim.SGD([parameter], lr=0.1)

(parameter * 2).backward()
(parameter * 3).backward()
print('accumulated grad:', parameter.grad.item())

optimizer.zero_grad(set_to_none=True)
print('after set_to_none:', parameter.grad)
(parameter * 4).backward()
print('new grad:', parameter.grad.item())
```

## Q2 verification: Can the accumulation window be restarted in both ways?

```python
parameter = nn.Parameter(torch.tensor(1.0))
optimizer = torch.optim.SGD([parameter], lr=0.1)
(parameter * 2).backward()
(parameter * 3).backward()
assert parameter.grad.item() == 5.0

parameter.grad.zero_()
assert parameter.grad.item() == 0.0
(parameter * 4).backward()
assert parameter.grad.item() == 4.0

optimizer.zero_grad(set_to_none=True)
assert parameter.grad is None
(parameter * 5).backward()
assert parameter.grad.item() == 5.0
print('✅ 梯度累积与清理通过')
```

## Q3: How do the scopes of `detach()`, `no_grad()`, and `inference_mode()` differ?

`detach()` only cuts the connection between the specified tensor and the existing computation graph; `no_grad()` keeps new operations in the context from being recorded; `inference_mode()` also disables more Autograd maintenance state and is better suited to pure inference that will not re-enter the training graph. After leaving the context, default gradient recording is restored.

```python
source = torch.tensor(2.0, requires_grad=True)
tracked = source * 2
detached = tracked.detach()

with torch.no_grad():
    no_grad_result = source * 3
with torch.inference_mode():
    inference_result = source * 4
restored = source * 5

print({
    'tracked': tracked.requires_grad,
    'detached': detached.requires_grad,
    'no_grad': no_grad_result.requires_grad,
    'no_grad_is_inference': torch.is_inference(no_grad_result),
    'inference': inference_result.requires_grad,
    'inference_is_inference': torch.is_inference(inference_result),
    'restored': restored.requires_grad,
})
```

## Q3 verification: Are local cut-off and context cut-off kept distinct?

```python
source = torch.tensor(2.0, requires_grad=True)
tracked = source * 2
detached = tracked.detach()
with torch.no_grad():
    no_grad_result = source * 3
with torch.inference_mode():
    inference_result = source * 4
restored = source * 5

assert tracked.requires_grad is True
assert detached.requires_grad is False and detached.item() == tracked.item()
assert no_grad_result.requires_grad is False and not torch.is_inference(no_grad_result)
assert inference_result.requires_grad is False and torch.is_inference(inference_result)
assert restored.requires_grad is True
restored.backward()
assert source.grad.item() == 5.0
print('✅ detach 与梯度上下文通过')
```

## Q4: Why is freezing parameters not the same as putting the entire forward pass inside `no_grad()`?

The parameters of a frozen layer do not need `.grad`, but its output may still feed into subsequent trainable layers. Putting the entire forward pass inside `no_grad()` cuts off all subsequent training graphs; the correct approach is to set the frozen parameters to `requires_grad=False` and pass only the trainable parameters to the optimizer.

```python
class TwoStageModel(nn.Module):
    """Freeze the first-stage parameters while preserving the trainable computation graph of the second stage."""

    def __init__(self):
        super().__init__()
        self.frozen_stage = nn.Linear(2, 2)
        self.trainable_stage = nn.Linear(2, 1)

    def forward(self, x):
        return self.trainable_stage(torch.relu(self.frozen_stage(x)))


model = TwoStageModel()
for parameter in model.frozen_stage.parameters():
    parameter.requires_grad_(False)

trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
optimizer = torch.optim.SGD(trainable_parameters, lr=0.1)
loss = model(torch.ones(2, 2)).sum()
loss.backward()

print('frozen grads:', [parameter.grad for parameter in model.frozen_stage.parameters()])
print('trainable grads present:', [parameter.grad is not None for parameter in model.trainable_stage.parameters()])
```

## Q4 verification: When the frozen layer has no gradients, can the subsequent trainable layers still be updated?

```python
model = TwoStageModel()
for parameter in model.frozen_stage.parameters():
    parameter.requires_grad_(False)

trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
optimizer = torch.optim.SGD(trainable_parameters, lr=0.1)
before = [parameter.detach().clone() for parameter in model.trainable_stage.parameters()]
model(torch.ones(2, 2)).sum().backward()

assert all(parameter.grad is None for parameter in model.frozen_stage.parameters())
assert all(parameter.grad is not None for parameter in model.trainable_stage.parameters())
optimizer.step()
after = list(model.trainable_stage.parameters())
assert any(not torch.allclose(old, new) for old, new in zip(before, after))
print('✅ 参数冻结与可训练路径通过')
```

### Section Summary

- `train/eval` controls module behavior, and `no_grad/inference_mode` controls whether new computation graphs are recorded.
- Gradients accumulate by default; both zeroing to 0 and setting to `None` start a new accumulation window, but the state differs.
- `detach()` cuts off a single tensor, and parameter freezing only prevents the corresponding parameters from receiving gradients.
- When freezing some parameters, the computation graph must still be preserved for subsequent trainable layers.

## Related Reading
**Overview:** Next, place gradient state into model objects, optimizers, and the complete training loop.
- [09. PyTorch nn.Module Basics](./09_PyTorch_nn_Module_Basics.md)
- [11. PyTorch Optimizers and Loss](./11_PyTorch_Optimizers_and_Loss.md)
- [Backpropagation and Training Mechanics](../topic_discussion/backpropagation_training_mechanism/intro.md)

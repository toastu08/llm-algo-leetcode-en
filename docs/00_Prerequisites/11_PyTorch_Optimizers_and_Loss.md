# 11. PyTorch Optimizers and Loss | PyTorch 优化器与损失函数

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `PyTorch`, `loss`, `optimizer` | **Audience:** learners preparing to understand the training update mechanism

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/11_PyTorch_Optimizers_and_Loss.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## Section Guide

The loss function turns the task objective into a differentiable scalar, and the optimizer then combines the gradient, learning rate, and internal state into a parameter update. Different update rules save different state and produce different parameter trajectories.

After completing this section, you should be able to hand-compute a basic update, explain the role of the optimizer state, and determine whether gradient accumulation, parameter updates, and learning-rate advancement happen at the correct moments.

**Keywords:** `loss`, `SGD`, `AdamW`, `optimizer step`, `scheduler`

![Gradient accumulation, optimizer, and scheduler timeline](../public/00_Prerequisites/11_optimizer_update_timeline.svg)

## Prerequisite Reading
**Overview:** Parameters must first be registered in the model, and backward propagation must already have produced gradients, before the optimizer can perform an update.
- [09. PyTorch nn.Module Basics](./09_PyTorch_nn_Module_Basics.md)
- [08. PyTorch Grad Hygiene and No-Grad](./08_PyTorch_Grad_Hygiene_and_No_Grad.md)
- [0C Group Page](./0C.md)

## Q1: What problems do loss and optimizer solve respectively?

Regression and classification use different target contracts, but both must reduce the error of a batch to a scalar before backward propagation can start from that scalar. The optimizer does not define the task objective; it only consumes the gradients on the parameters.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def mse_loss(pred, target):
    # In a regression task, the loss usually reduces the error to a single scalar.
    return torch.mean((pred - target) ** 2)


def cross_entropy_loss(logits, target):
    # In a classification task, logits are unnormalized scores and target is the class id.
    return F.cross_entropy(logits, target)


pred = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
target = torch.tensor([[1.0, 1.0], [3.0, 5.0]])
print('MSE 示例：', mse_loss(pred, target).item())

logits = torch.tensor([[2.0, 0.5], [0.1, 1.2]])
labels = torch.tensor([0, 1])
print('CrossEntropy 示例：', cross_entropy_loss(logits, labels).item())

```

## Q1 verification: Are the values of the basic losses correct?

Here it directly checks whether the MSE and cross-entropy calls return the expected values.


```python
pred = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
target = torch.tensor([[1.0, 1.0], [3.0, 5.0]])
loss = mse_loss(pred, target)
assert torch.allclose(loss, torch.tensor(1.5))

logits = torch.tensor([[2.0, 0.5], [0.1, 1.2]])
labels = torch.tensor([0, 1])
ce = cross_entropy_loss(logits, labels)
assert torch.allclose(ce, F.cross_entropy(logits, labels))
print('✅ loss 基本验证通过')

```

## Q2: Which quantities determine one SGD parameter update?

The most basic SGD update is determined by the current parameters, the gradients, and the learning rate; with momentum, the optimizer must also save momentum state across steps. Only after aligning the hand-computed result with the PyTorch update can you clearly separate parameters, gradients, and optimizer state.


```python
def manual_sgd_step(parameter, gradient, learning_rate):
    """Compute one SGD update without momentum and weight decay."""
    return parameter - learning_rate * gradient


weight = nn.Parameter(torch.tensor([2.0]))
weight.grad = torch.tensor([3.0])
optimizer = torch.optim.SGD([weight], lr=0.1)
expected = manual_sgd_step(weight.detach(), weight.grad, learning_rate=0.1)
optimizer.step()
print('更新后参数：', weight.item(), '手算结果：', expected.item())

```

## Q2 verification: Is PyTorch SGD consistent with the hand-computed update?

Without momentum, the two update paths should produce the same parameter; after enabling momentum, a buffer saved across steps should appear in the optimizer state.


```python
weight = nn.Parameter(torch.tensor([2.0]))
weight.grad = torch.tensor([3.0])
optimizer = torch.optim.SGD([weight], lr=0.1)
optimizer.step()
assert torch.allclose(weight.detach(), torch.tensor([1.7]))
assert optimizer.state_dict()['state'] == {}

momentum_weight = nn.Parameter(torch.tensor([2.0]))
momentum_optimizer = torch.optim.SGD([momentum_weight], lr=0.1, momentum=0.9)
momentum_weight.grad = torch.tensor([3.0])
momentum_optimizer.step()
state = momentum_optimizer.state[momentum_weight]
assert 'momentum_buffer' in state
print('✅ SGD 更新与 momentum state 通过')

```

## Q3: Why does AdamW need to save first-order and second-order moment state?

AdamW does not only read the current gradient; it also maintains first-order and second-order moment estimates of the gradient and applies weight decay independently. This state explains why a complete checkpoint must save the optimizer state, and why optimizer state increases training memory.


```python
parameter = nn.Parameter(torch.tensor([1.0, -1.0]))
optimizer = torch.optim.AdamW([parameter], lr=0.01, weight_decay=0.1)
for gradient in (torch.tensor([0.5, -0.25]), torch.tensor([0.25, -0.5])):
    optimizer.zero_grad(set_to_none=True)
    parameter.grad = gradient.clone()
    optimizer.step()

state = optimizer.state[parameter]
print('AdamW state keys：', sorted(state.keys()))
print('step：', state['step'].item())

```

## Q3 verification: Does the AdamW state update with optimizer steps?

After two consecutive steps, the step count, first moment, and second moment should all exist, and the state shapes should match the parameters.


```python
assert int(state['step'].item()) == 2
assert state['exp_avg'].shape == parameter.shape
assert state['exp_avg_sq'].shape == parameter.shape
assert torch.count_nonzero(state['exp_avg']) == parameter.numel()
assert torch.count_nonzero(state['exp_avg_sq']) == parameter.numel()
print('✅ AdamW optimizer state 通过')

```

## Q4: How do micro-batches, optimizer steps, and scheduler steps align?

Gradients can accumulate across multiple micro-batches, but reaching the accumulation boundary only means an update attempt is initiated. Only when the parameter update succeeds do optimizer-step and scheduler-step advance together. In this example every micro-batch has the same number of samples; when the last window has fewer than the accumulation step count, the loss is scaled by the actual window size; when the sample counts differ, the loss should be weighted by sample count. The code uses a given boolean sequence to demonstrate a successful or skipped update and does not detect real mixed-precision overflow.

| Clock | When it advances | What it represents |
|:---|:---|:---|
| micro-step | On consuming each micro-batch | Data and backward propagation progress |
| update attempt | On reaching the accumulation window boundary | An attempt to apply the current accumulated gradients |
| optimizer-step | On performing one optimizer update that was not skipped | Number of effective updates; the parameters may also be unchanged when the gradient is zero |
| scheduler-step | After a successful optimizer-step | The learning-rate schedule aligns with effective updates |


```python
def run_accumulated_updates(
    model, batches, optimizer, scheduler, accumulation_steps, update_success_pattern
):
    """Run optimizer and scheduler at accumulation boundaries and return a checkable timeline."""
    if accumulation_steps <= 0:
        raise ValueError('accumulation_steps must be positive')
    optimizer.zero_grad(set_to_none=True)
    timeline = []
    optimizer_steps = 0
    update_attempts = 0
    total_micro_steps = len(batches)
    for micro_step, (x, target) in enumerate(batches, start=1):
        window_start = ((micro_step - 1) // accumulation_steps) * accumulation_steps
        window_size = min(accumulation_steps, total_micro_steps - window_start)
        loss = mse_loss(model(x), target) / window_size
        loss.backward()
        reaches_boundary = micro_step % accumulation_steps == 0 or micro_step == len(batches)
        update_succeeded = None
        if reaches_boundary:
            if update_attempts >= len(update_success_pattern):
                raise ValueError('update_success_pattern is shorter than update attempts')
            update_succeeded = bool(update_success_pattern[update_attempts])
            update_attempts += 1
            if update_succeeded:
                optimizer.step()
                scheduler.step()
                optimizer_steps += 1
            optimizer.zero_grad(set_to_none=True)
        timeline.append({
            'micro_step': micro_step,
            'update_attempt': reaches_boundary,
            'update_succeeded': update_succeeded,
            'optimizer_step': optimizer_steps,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'window_size': window_size,
        })
    return timeline


model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
batches = [(torch.tensor([[float(i)]]), torch.tensor([[2.0 * i + 1.0]])) for i in range(1, 6)]
timeline = run_accumulated_updates(
    model, batches, optimizer, scheduler, accumulation_steps=2,
    update_success_pattern=[True, False, True],
)
print(timeline)

```

## Q4 verification: Does the scheduler advance only after a parameter update?

Five micro-batches with one accumulation every two form three update attempts; the second attempt is skipped, so only two optimizer-steps and two scheduler-steps occur. The last window has only one micro-batch, and its loss is no longer divided by 2.


```python
assert [item['update_attempt'] for item in timeline] == [False, True, False, True, True]
assert [item['update_succeeded'] for item in timeline] == [None, True, None, False, True]
assert [item['optimizer_step'] for item in timeline] == [0, 1, 1, 1, 2]
assert [item['window_size'] for item in timeline] == [2, 2, 2, 2, 1]
assert timeline[0]['learning_rate'] == 0.1
assert timeline[1]['learning_rate'] == 0.05
assert timeline[3]['learning_rate'] == 0.05
assert timeline[4]['learning_rate'] == 0.025
assert all(parameter.grad is None for parameter in model.parameters())
print('✅ micro-batch / optimizer / scheduler 对齐通过')

```

## Q5: Why does the optimizer need parameter groups?

Parameters in the same model do not necessarily share the same update rule. A common practice is to apply weight decay to matrix weights while using zero decay for biases and normalization parameters; parameter groups bind a set of parameters to their own learning rate and decay coefficient.


```python
weight = nn.Parameter(torch.tensor([1.0]))
bias = nn.Parameter(torch.tensor([1.0]))
optimizer = torch.optim.AdamW([
    {'params': [weight], 'weight_decay': 0.1},
    {'params': [bias], 'weight_decay': 0.0},
], lr=0.1)

# Zero gradients isolate the effect of weight decay, making the two rules directly observable.
weight.grad = torch.zeros_like(weight)
bias.grad = torch.zeros_like(bias)
optimizer.step()
print('weight / bias：', weight.item(), bias.item())

```

## Q5 verification: Do different parameter groups apply different decay rules?

Both groups share the learning rate, but only the group containing weight enables decay; under zero gradients, weight should shrink while bias stays unchanged.


```python
assert len(optimizer.param_groups) == 2
assert optimizer.param_groups[0]['weight_decay'] == 0.1
assert optimizer.param_groups[1]['weight_decay'] == 0.0
assert torch.allclose(weight.detach(), torch.tensor([0.99]))
assert torch.allclose(bias.detach(), torch.tensor([1.0]))
print('✅ parameter groups 与 weight decay 规则通过')

```

## Related Reading

With the update rules mastered, continue by checking how batches are formed and how the update rules enter the complete training loop.

- [12. PyTorch Data Interface and Training Lifecycle](./12_PyTorch_Minimal_Training_Interface.md)
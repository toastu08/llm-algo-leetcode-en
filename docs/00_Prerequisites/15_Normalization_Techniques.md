# 15. Normalization Techniques | 归一化技术

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `PyTorch`, `normalization`, `statistical axis` | **Audience:** learners preparing to understand the statistical path of a Transformer

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/15_Normalization_Techniques.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## Section Guide

Normalization must answer three questions at the same time: along which dimensions the statistics are computed, whether the formula performs centering, and where the Norm sits in the residual branch. BatchNorm, LayerNorm, and RMSNorm are not equivalent in these respects.

After completing this section, you should be able to determine which dimension the statistics come from, distinguish the state dependency of training mode and inference mode, compare LayerNorm and RMSNorm, and explain the dataflow difference between Pre-Norm and Post-Norm.

**Keywords:** `BatchNorm`, `LayerNorm`, `RMSNorm`, `Pre-Norm`, `running stats`

![Normalization statistical sources, formulas, and residual positions](../public/00_Prerequisites/15_normalization_paths_map.svg)

## Prerequisite Reading
**Overview:** Section 13 gives the position of the Norm in the Block, and section 14 shows how nonlinearity changes the branch values; this page focuses on the statistics and placement rules of normalization.
- [13. Transformer Block Dataflow and Residual Stream](./13_Transformer_Block_Dataflow_and_Residual_Stream.md)
- [14. Activation Functions](./14_Activation_Functions.md)
- [0D Group Page](./0D.md)

## Q1: How do the statistical axes used by BatchNorm and LayerNorm differ?

BatchNorm uses the statistics of the same feature across the batch, and LayerNorm uses the statistics of the last dimension of a single sample. Different statistical axes change their dependence on batch size, sequence position, and training/inference mode; a Transformer usually needs per-token normalization that does not depend on other samples.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_norm_train(x, gamma, beta, eps=1e-5):
    batch_mean = x.mean(dim=0)
    batch_var = x.var(dim=0, unbiased=False)
    normalized = (x - batch_mean) / torch.sqrt(batch_var + eps)
    return normalized * gamma + beta, batch_mean, batch_var


def batch_norm_eval(x, gamma, beta, running_mean, running_var, eps=1e-5):
    normalized = (x - running_mean) / torch.sqrt(running_var + eps)
    return normalized * gamma + beta


def layer_norm_last_dim(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    normalized = (x - mean) / torch.sqrt(var + eps)
    return normalized * gamma + beta


x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
gamma = torch.ones(3)
beta = torch.zeros(3)
print('BatchNorm 输出：')
y, mean, var = batch_norm_train(x, gamma, beta)
print(y)
print('LayerNorm 输出：')
print(layer_norm_last_dim(x, gamma, beta))

```

## Q1 verification: Are the outputs of the two normalizations visible?

Here it directly prints the BatchNorm and LayerNorm outputs, to first see what they do to the same set of inputs.


```python
x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
gamma = torch.tensor([1.0, 1.5, 2.0])
beta = torch.tensor([0.0, 1.0, -1.0])
y, mean, var = batch_norm_train(x, gamma, beta)
expected_mean = torch.tensor([1.5, 3.0, 4.5])
expected_var = torch.tensor([0.25, 1.0, 2.25])
expected = (x - expected_mean) / torch.sqrt(expected_var + 1e-5) * gamma + beta
assert torch.allclose(mean, expected_mean)
assert torch.allclose(var, expected_var)
assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)
print('✅ BatchNorm 基本验证通过')

```

## Q2: When must training mode and inference mode be distinguished?

During training, BatchNorm uses the statistics of the current batch and updates the running stats; during inference it switches to the accumulated statistics. LayerNorm computes along dimensions inside the sample, so it usually does not depend on batch statistics.


```python
batch_norm = nn.BatchNorm1d(3, affine=False, momentum=0.5, eps=1e-5)
train_batch = torch.tensor([[1.0, 2.0, 3.0], [3.0, 6.0, 9.0]])
batch_norm.train()
train_output = batch_norm(train_batch)
running_mean_after_train = batch_norm.running_mean.clone()
running_var_after_train = batch_norm.running_var.clone()

eval_batch = torch.tensor([[10.0, 20.0, 30.0]])
batch_norm.eval()
eval_output = batch_norm(eval_batch)
print('running mean:', running_mean_after_train.tolist())
print('eval output:', eval_output.tolist())

```

## Q2 verification: Do training mode and inference mode use different statistical sources?

The training output should use the current batch statistics; the running variance is updated with PyTorch's unbiased variance convention, and the inference output then uses the accumulated statistics.


```python
expected_running_mean = 0.5 * train_batch.mean(dim=0)
expected_running_var = 0.5 * torch.ones(3) + 0.5 * train_batch.var(dim=0, unbiased=True)
expected_eval = (eval_batch - expected_running_mean) / torch.sqrt(expected_running_var + batch_norm.eps)
assert torch.allclose(train_output.mean(dim=0), torch.zeros(3), atol=1e-6)
assert torch.allclose(running_mean_after_train, expected_running_mean)
assert torch.allclose(running_var_after_train, expected_running_var)
assert torch.allclose(eval_output, expected_eval, atol=1e-6, rtol=1e-6)
print('✅ BatchNorm 训练态 / 推理态统计通过')

```

## Q3: Why is `eps` part of the normalization formula?

When a group of features is identical or the variance is extremely small, dividing directly by the standard deviation produces a division by zero or amplified error. `eps` sets a numerical lower bound on the denominator so that constant inputs also produce finite outputs.


```python
constant = torch.ones(2, 4)
centered = constant - constant.mean(dim=-1, keepdim=True)
variance = centered.pow(2).mean(dim=-1, keepdim=True)
without_eps = centered / torch.sqrt(variance)
with_eps = centered / torch.sqrt(variance + 1e-6)
print('without eps：', without_eps)
print('with eps：', with_eps)

```

## Q3 verification: Does `eps` prevent non-finite outputs?

The variance of a constant input is 0; without `eps` a non-finite value appears, and after adding `eps` the output should be a finite 0.


```python
assert variance.eq(0).all()
assert not torch.isfinite(without_eps).all()
assert torch.isfinite(with_eps).all()
assert torch.equal(with_eps, torch.zeros_like(with_eps))
print('✅ eps 数值稳定性通过')

```

## Q4: Which operations do LayerNorm and RMSNorm each keep?

LayerNorm first subtracts the mean of the last dimension, then scales by the standard deviation; RMSNorm does not perform mean centering and only scales by the root mean square. Both preserve the input shape and usually use one set of scale parameters along the hidden dimension. RMSNorm's output does not guarantee a zero mean, so when comparing them you should check "zero mean" and "unit root mean square" separately rather than treating them as the same formula. `eps` is used to avoid division by zero when the input scale is close to 0.


```python
def rms_norm_last_dim(x, weight, eps=1e-6):
    """Perform RMSNorm along the last dimension without mean centering."""
    mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(mean_square + eps)
    return normalized * weight


x = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.0, 2.0]])
weight = torch.ones(3)
layer_output = layer_norm_last_dim(x, weight, torch.zeros(3), eps=1e-6)
rms_output = rms_norm_last_dim(x, weight, eps=1e-6)

print('LayerNorm row mean:', layer_output.mean(dim=-1).tolist())
print('RMSNorm row mean:', rms_output.mean(dim=-1).tolist())
print('RMSNorm row RMS:', rms_output.pow(2).mean(dim=-1).sqrt().tolist())
```

## Q4 verification: Do the invariants of the two normalizations match their own definitions?

Here it verifies LayerNorm's zero mean and RMSNorm's unit root mean square respectively, and compares the implementation result with `torch.nn.functional.rms_norm`.

```python
x = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.0, 2.0]])
weight = torch.ones(3)
layer_output = layer_norm_last_dim(x, weight, torch.zeros(3), eps=1e-6)
rms_output = rms_norm_last_dim(x, weight, eps=1e-6)
expected_rms = F.rms_norm(x, (x.shape[-1],), weight=weight, eps=1e-6)

assert torch.allclose(layer_output.mean(dim=-1), torch.zeros(2), atol=1e-6)
assert torch.allclose(rms_output, expected_rms, atol=1e-6, rtol=1e-6)
assert torch.allclose(rms_output.pow(2).mean(dim=-1), torch.ones(2), atol=1e-5)
assert not torch.allclose(rms_output[0].mean(), torch.tensor(0.0), atol=1e-6)
print('✅ LayerNorm / RMSNorm 对照通过')
```

## Q5: Where do Pre-Norm and Post-Norm place statistical control?

Pre-Norm normalizes the branch input first, then adds the branch output back to the original residual; Post-Norm merges the residual and the branch first, then normalizes the result. Both use the same normalization formula, but preserve the identity path in different ways.


```python
def pre_norm_block(x, branch_scale=0.5, eps=1e-6):
    """Normalize the branch input first, then preserve the original residual."""
    normalized = F.layer_norm(x, (x.shape[-1],), eps=eps)
    return x + branch_scale * normalized


def post_norm_block(x, branch_scale=0.5, eps=1e-6):
    """Merge the residual and the branch first, then normalize the result."""
    combined = x + branch_scale * x
    return F.layer_norm(combined, (x.shape[-1],), eps=eps)


x = torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 8.0]])
print('Pre-Norm block：', pre_norm_block(x))
print('Post-Norm block：', post_norm_block(x))

```

## Q5 verification: Do the two placements preserve different residual statistics?

The Post-Norm output is re-centered along the hidden dimension; Pre-Norm preserves the row mean of the original residual. Both have the same shape, but the statistical results should not be mistaken for the same.


```python
pre_output = pre_norm_block(x)
post_output = post_norm_block(x)
assert pre_output.shape == post_output.shape == x.shape
assert torch.allclose(post_output.mean(dim=-1), torch.zeros(2), atol=1e-6)
assert torch.allclose(pre_output.mean(dim=-1), x.mean(dim=-1), atol=1e-6)
assert not torch.allclose(pre_output, post_output)
print('✅ Pre-Norm / Post-Norm 放置差异通过')

```

## Related Reading

Once the intuition for statistical axes, formulas, and residual positions is established, you can continue by observing how RMSNorm and Attention enter a real Transformer Block.

- [16. Attention Mechanism Intro](./16_Attention_Mechanism_Intro.md)
- [Part 02 · 01 RMSNorm](../02_PyTorch_Algorithms/01_RMSNorm_Tutorial.md)
- [Part 02 · 05 LLaMA3 Block](../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.md)
- [LLM Architecture Evolution · normalization evolution](../topic_discussion/llm_architecture_evolution/03_norm_evolution.md)

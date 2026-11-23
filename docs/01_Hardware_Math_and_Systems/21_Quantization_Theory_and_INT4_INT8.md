# 21. Quantization Theory and INT4 INT8 | 量化理论与 INT4/INT8

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `quantization and compression`, `INT4/INT8`, `quantization theory` | **Audience:** beginners in system performance

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

Quantization represents weights, activations, or caches with a limited set of discrete values, with the goal of reducing storage, transfer, or compute cost within a quality gate. Different objects and granularities produce different quantization errors and metadata costs.

This section establishes the basic relationship "representation range → scale factor (scale) and granularity → error → execution conditions", providing a numerical entry point for the calibration, training, and inference branches in the quantization and compression topic.

**Keywords:** `INT8`, `INT4`, scale factor

![Quantization objects and intervention timing](../public/02_PyTorch_Algorithms/21_quantization_object_timing.svg)
## Prerequisite Reading

**Overview:** First review data formats, parameter counts, and the GPU memory hierarchy, then understand how low-bit quantization changes storage, bandwidth, and inference throughput.

- [01. Data Types and Precision](./01_Data_Types_and_Precision.md)
- [02. LLM Params and FLOPs](./02_LLM_Params_and_FLOPs.md)
- [03. GPU Architecture and Memory](./03_GPU_Architecture_and_Memory.md)

## Q1: Why can quantization significantly reduce memory and bandwidth pressure?


If the weights are compressed from FP16 to INT8, the storage of a single parameter changes from 2 Bytes to 1 Byte, and the weight memory is theoretically about halved; if compressed to INT4, it theoretically drops further to 0.5 Byte, and the weight volume continues to decrease.

Below, a 7B model example computes the weight storage under different bit widths:

| Weight format | Bytes per parameter | 7B model weight memory | Relative to FP16 |
| --- | --- | --- | --- |
| FP16 | 2 Bytes | 14 GB | 1x |
| INT8 | 1 Byte | 7 GB | 0.5x |
| INT4 | 0.5 Byte | 3.5 GB | 0.25x |

This matters for inference, and the reason is not only that "the model is smaller", but also that:
- With lower memory usage, a larger batch or a longer context fits more easily
- Less data needs to be transferred when reading the weights, so HBM bandwidth pressure also drops
- In some bandwidth-bound scenarios, throughput improves noticeably

But note that quantization usually does not reduce all costs linearly with the bit count:
- Activations may still be kept at higher precision
- Some layers keep FP16 / BF16 accumulation
- Dequantization and scale handling also have extra overhead
- Whether INT8 / INT4 actually speed things up also depends on whether the hardware natively supports low-bit matrix operations; if the hardware has no corresponding Tensor Core / MMA support, quantization sometimes only saves memory and does not necessarily save time

So the benefit of quantization is usually "smaller memory + lower bandwidth + higher throughput", not simply "how much the bit width shrinks is how much the speed improves".

### Q1 Mini Check: Implement the model weight memory function

Implement a function that computes the model weight memory for a given parameter count and data format.

```python
def calculate_weight_memory(num_params_b, dtype):
    """
    Compute the memory usage of the model weights.

    Args:
        num_params_b: parameter count (unit: B, i.e. billions)
        dtype: data type, one of 'fp16', 'bf16', 'int8', 'int4'

    Returns:
        memory_gb: memory usage (unit: GB)
    """
    bytes_per_param = {'fp16': 2, 'bf16': 2, 'int8': 1, 'int4': 0.5}[dtype]
    return num_params_b * bytes_per_param


def test_calculate_weight_memory():
    result = calculate_weight_memory(7, 'fp16')
    assert abs(result - 14.0) < 1e-9
    result = calculate_weight_memory(7, 'int8')
    assert abs(result - 7.0) < 1e-9
    result = calculate_weight_memory(7, 'int4')
    assert abs(result - 3.5) < 1e-9
    print('✅ calculate_weight_memory tests passed')

# Run the test
test_calculate_weight_memory()
```

### Q1 Extended Check: Compare different data formats

Use the function above to compare the weight memory of a 7B model under different data formats.

```python
# Directly compute the weight memory of a 7B model under different formats
num_params = 7
dtypes = ['fp16', 'bf16', 'int8', 'int4']

print('7B 模型权重显存占用对比：')
print('-' * 40)
for dtype in dtypes:
    memory = calculate_weight_memory(num_params, dtype)
    print(f'{dtype.upper():<6} {memory:>6.1f} GB')

```

## Q2: What is the difference between symmetric, asymmetric, per-tensor, and per-channel quantization?


The most common quantization form can be written as:

```text
q = round(x / scale) + zero_point
```

where:
- `scale` determines the numerical mapping ratio
- `zero_point` determines whether the zero point is shifted

There are four common combinations:
- **Symmetric quantization**: `zero_point = 0`, simple to implement, often used for weights
- **Asymmetric quantization**: keeps `zero_point`, better suited to data with an obvious distribution shift
- **per-tensor**: the whole tensor shares one set of scales
- **per-channel**: each channel has its own set of scales, usually with better precision but more metadata

From experience:
- Weight quantization is often better suited to per-channel
- Activation quantization often relies more on calibration data
- At extremely low bit widths, the error comes mainly not from the mean but from outliers and distribution skew

Why are activations harder to quantize?
- Weight distributions are usually relatively stable and have fewer outliers
- Activations change with the token, layer, and context, so the distribution fluctuates more
- Some activation channels may have obvious outliers, which are more easily distorted when pressed directly into low bits
- This is also why many schemes perform joint weight-activation handling or introduce preprocessing ideas such as SmoothQuant

This is also why, in real deployment, quantization is not simply "make the dtype smaller", but must simultaneously decide the scale, zero point, grouping granularity, and accumulation precision.

### Q2 Mini Check: Implement the minimal per-tensor symmetric quantization

This question implements only the minimal branch: the whole tensor shares one scale, and the zero-point is fixed at 0. Asymmetric and per-channel quantization are understood conceptually first and are not required to be fully implemented here. `num_bits=4` only simulates the 4-bit numerical range; the returned PyTorch Tensor is still `torch.int8` and does not represent real packed INT4 storage.


```python
import torch


def quantize_per_tensor(x, num_bits=8):
    """Perform symmetric per-tensor quantization on a tensor.

    `num_bits=4` only controls the logical quantization range; the return value is still stored in an int8 Tensor, and bit packing is not implemented.
    """
    qmax = 2 ** (num_bits - 1) - 1
    scale = x.abs().max() / qmax if x.numel() > 0 else torch.tensor(1.0, device=x.device, dtype=x.dtype)
    scale = torch.clamp(scale, min=1e-8)
    # An int8 tensor carries the quantized integers here, making the mapping and error easy to observe; real INT4 needs extra packing.
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax).to(torch.int8)
    return q, scale


def dequantize_per_tensor(q, scale):
    """Restore quantized integers to float32; does not simulate the accumulation path of a real kernel."""
    return q.to(torch.float32) * scale


def per_channel_scales(x, axis=0, num_bits=8):
    """Return the scales of per-channel symmetric quantization, only to observe the granularity difference."""
    if x.ndim != 2 or axis not in (0, 1):
        raise ValueError('示例只接受二维张量，axis 必须为 0 或 1')
    qmax = 2 ** (num_bits - 1) - 1
    reduce_dims = (1,) if axis == 0 else (0,)
    scale = x.abs().amax(dim=reduce_dims, keepdim=True) / qmax
    return torch.clamp(scale, min=1e-8)


def test_quantize_per_tensor():
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    q, scale = quantize_per_tensor(x, 8)
    x_hat = dequantize_per_tensor(q, scale)
    assert q.dtype == torch.int8 and x_hat.shape == x.shape
    print('q:', q.tolist(), 'scale:', float(scale))
    print('x_hat:', x_hat.tolist())
    print('✅ quantize_per_tensor tests passed')


test_quantize_per_tensor()

```

### Q2 Extended Check: Observe the quantization error

Compare the error between the original tensor and the dequantized tensor.

```python
# Directly observe the difference in quantization error between 8-bit and 4-bit
torch.manual_seed(0)
x = torch.randn(1024) * 2

for bits in [8, 4]:
    q, scale = quantize_per_tensor(x, bits)
    x_hat = dequantize_per_tensor(q, scale)
    mse = torch.mean((x - x_hat) ** 2).item()
    max_err = torch.max(torch.abs(x - x_hat)).item()
    print(f'{bits}-bit -> MSE={mse:.6f}, max_err={max_err:.6f}')

matrix = torch.tensor([[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]])
print('per-tensor scale shape:', tuple(quantize_per_tensor(matrix, 8)[1].shape))
print('per-channel scale shape:', tuple(per_channel_scales(matrix, axis=0, num_bits=8).shape))

```


```python
def test_quantization_practice():
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    q8, s8 = quantize_per_tensor(x, 8)
    x8 = dequantize_per_tensor(q8, s8)
    q4, s4 = quantize_per_tensor(x, 4)
    x4 = dequantize_per_tensor(q4, s4)
    assert q8.dtype == torch.int8 and q4.dtype == torch.int8
    assert x8.shape == x.shape and x4.shape == x.shape
    assert torch.mean((x - x4).abs()) >= torch.mean((x - x8).abs())
    print('✅ 21 Quantization tests passed')

test_quantization_practice()
```

## Q3: Why do the quantization object, processing timing, and delivery form affect deployment?

A quantization scheme cannot be named only by the INT4 or INT8 bit width; you must also look at which kind of state it changes, at which stage it happens, and which backend finally interprets the quantized artifact. You can judge along the following chain:

| Quantization object | Processing timing | Typical artifact | What to verify first |
|---|---|---|---|
| Weights | Before deployment | GPTQ / AWQ / GGUF files | Whether it loads, plus weight memory and quality |
| Weights and activations | During training or at runtime | QAT model / FP8 path | Kernels, numerical stability, and throughput |
| KV Cache | During request execution | Low-bit cache representation | Context capacity, concurrency, latency, and quality |

The processing timing determines the calibration or training cost, and the delivery form determines backend compatibility; therefore "smaller memory" is only a candidate gain, and whether it is worthwhile still requires quality and performance verification on a fixed workload. The judgment order can be remembered as: the quantization object determines which cost is affected, the processing timing determines how the quantization parameters are obtained, and the delivery form determines who interprets the quantized artifact.

Here PTQ means quantization after training is complete, and QAT means simulating quantization error during training or fine-tuning; GPTQ and AWQ are common weight-calibration routes, and GGUF is a file format and delivery packaging, not equivalent to a general quantization algorithm.

### Q3 Mini Check: Organize the quantization methods by processing timing and delivery form

Put PTQ, QAT, GPTQ, AWQ, and GGUF back into their respective layers, distinguishing when quantization happens, how the error is handled, and the final delivery form.


```python
def quantization_method_catalog():
    """Return the quantization-path metadata; does not represent the full configuration of a specific backend."""
    return {
        'PTQ': {'stage': '训练完成后', 'artifact': '量化权重', 'backend': '取决于部署引擎'},
        'QAT': {'stage': '训练或微调中', 'artifact': '量化感知模型', 'backend': '取决于部署引擎'},
        'GPTQ': {'stage': '部署前校准', 'artifact': 'GPTQ 权重', 'backend': '支持 GPTQ 的引擎'},
        'AWQ': {'stage': '部署前校准', 'artifact': 'AWQ 权重', 'backend': '支持 AWQ 的引擎'},
        'GGUF': {'stage': '模型交付', 'artifact': '文件格式', 'backend': 'llama.cpp 等'},
    }


def select_quantization_path(method: str) -> dict:
    """Return the stage, artifact, and backend that later experiments should focus on, according to the method name."""
    catalog = quantization_method_catalog()
    if method not in catalog:
        raise ValueError(f'未知量化方法：{method}，可选值：{sorted(catalog)}')
    return {'method': method, **catalog[method]}


for method in ['PTQ', 'GPTQ', 'AWQ', 'GGUF']:
    print(select_quantization_path(method))

assert select_quantization_path('GGUF')['backend'] == 'llama.cpp 等'

```

## Q4: When should quantization consider QAT?


QAT (Quantization-Aware Training) is not the first choice, but it is valuable in the following scenarios:

- After PTQ the accuracy loss is too large, for example a noticeable drop in perplexity or task metrics
- The model itself is particularly sensitive to low bits, especially smaller models or generative tasks
- You have enough training data and compute to accept the retraining or fine-tuning cost

One-sentence judgment:
- If the goal is "fast deployment", use PTQ first
- If the goal is "recover low-bit accuracy as much as possible", then consider QAT

The cost of QAT is a more complex and more expensive training pipeline, but it can incorporate the quantization error directly into the training process and is an important remedy beyond PTQ. The judgment below is only used to organize experiments and does not automatically prove that QAT is necessarily better than PTQ; the quality and training cost on a fixed validation set still have to be compared.

### Q4 Mini Check: When should QAT be considered?

Put the quantization error and the training budget together to judge whether to move from PTQ to QAT.


```python
def qat_recommendation(ptq_drop, acceptable_drop=0.5, retrain_budget_hours=0, sensitivity='medium'):
    score = 0
    if ptq_drop > acceptable_drop:
        score += 2
    if sensitivity == 'high':
        score += 1
    if retrain_budget_hours >= 10:
        score += 1
    if ptq_drop > acceptable_drop and retrain_budget_hours >= 10:
        recommendation = 'QAT'
    else:
        recommendation = 'PTQ'
    return {
        'recommendation': recommendation,
        'risk_score': score,
        'ptq_drop': ptq_drop,
        'acceptable_drop': acceptable_drop,
    }

cases = [
    (0.2, 0.5, 0, 'low'),
    (0.8, 0.5, 12, 'high'),
    (0.6, 0.5, 8, 'high'),
]
for case in cases:
    print(case, '->', qat_recommendation(*case))
print('QAT is worth considering when PTQ drop is too large and retraining budget exists')

```

## Q5: Why can theoretical compression not directly prove that a low-bit scheme is usable?


The bit-width ledger can only give a theoretical storage lower bound. To hand the low-bit representation to later topics for verification, four kinds of conditions also need to be recorded:

- **"INT4 is always better than INT8"**
  Not correct. A lower bit width is not automatically better; both error and hardware support may make INT4 harder to use.

- **"Quantization is just changing the dtype"**
  Not correct. Real quantization involves calibration, grouping, dequantization, accumulation precision, and kernel support.

- **"Quantization definitely does not affect quality"**
  Not correct. Different layers, channels, and models tolerate low bits very differently.

- **"Quantization only affects weights"**
  It is also incomplete. What actually limits inference performance often includes activations, caches, and bandwidth as well.

Therefore, this section only organizes the numerical and execution risks of quantization candidates; quality, real memory, execution paths, and throughput must be verified on fixed workloads in the quantization and compression topic. The risk function below is only a teaching checklist, not a predictor of hardware performance or model quality.

### Q5 Mini Check: The most common problems in quantization configurations

Look at which configurations are more likely to cause problems, instead of understanding quantization simply as "lowering the dtype".


```python
def quantization_risk(bitwidth, hardware_support=True, calibration_quality=1.0, activation_sensitive=False):
    """Use interpretable conditions for teaching-oriented screening of quantization risk; does not predict real quality or throughput."""
    score = 0
    if bitwidth <= 4:
        score += 2
    if not hardware_support:
        score += 2
    if calibration_quality < 0.7:
        score += 1
    if activation_sensitive:
        score += 1
    if score >= 4:
        level = 'high'
    elif score >= 2:
        level = 'medium'
    else:
        level = 'low'
    return {
        'risk_level': level,
        'risk_score': score,
        'bitwidth': bitwidth,
        'evidence_level': 'teaching_checklist',
    }

cases = [
    (8, True, 0.9, False),
    (4, True, 0.8, True),
    (4, False, 0.6, True),
]
for case in cases:
    print(case, '->', quantization_risk(*case))
print('quantization fails when bitwidth, calibration, and hardware support are all under pressure')

```

---
## Related Reading

**Overview:** After this section, you can continue with weight representation, runtime low precision, and deployment verification.

- [25. Quantization W8A16](../02_PyTorch_Algorithms/25_Quantization_W8A16.md)
- [41. FP8 Runtime Low-Precision Paths](../02_PyTorch_Algorithms/41_FP8_Runtime_Low_Precision_Paths.md)
- [67. Quantized Inference and Deployment](../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md)
---
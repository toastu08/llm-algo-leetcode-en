# 01. Data Types and Precision | 大模型的数据格式与混合精度

**Difficulty:** Easy | **Environment:** CPU-first | **Tags:** `numerical foundations`, `data types`, `mixed precision` | **Audience:** learners who need to understand data formats and numerical stability

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

Parameters, gradients, activations, and caches in a model must all be stored and computed in a specific data type (dtype). When the same tensor uses FP32, BF16, FP16, or INT8, the storage bytes, numerical range, error risk, and available hardware paths all change.
This section establishes the relationship "numerical representation → state bytes → precision choice". After completing it, you should be able to compute the theoretical capacity of the model state and, combined with the numerical range, judge whether a precision is worth entering real hardware verification.

**Keywords:** `FP16`, `BF16`, `INT8`

![Concept relationships in this section](../public/01_Hardware_Math_and_Systems/01_dtype_precision_map.svg)

---
## Prerequisite Reading
**Overview:** Before entering this section, review tensors and autograd; then use the dtype ledger to understand the relationship between mixed precision, low-bit representations, and training memory.

- [Group 0B: PyTorch Tensors and Autograd](../00_Prerequisites/0B.md)

---
## Q1: Basic concepts — how much memory does each common data format occupy?


At the lowest level, 1 Byte = 8 bits. First compute the theoretical storage lower bound below; a real model may also contain scale, zero-point, packing, padding, and runtime workspace.

- **FP32 (single-precision floating point)**: 32 bits = **4 Bytes**
- **FP16 (half-precision floating point)**: 16 bits = **2 Bytes**
- **BF16 (BFloat16)**: 16 bits = **2 Bytes**
- **INT8 (8-bit integer)**: 8 bits = **1 Byte**
- **INT4 (4-bit integer)**: 4 bits = **0.5 Byte** (usually used for extremely compressed quantization such as AWQ/GPTQ)

**Practical estimate:**
When estimating the weight lower bound, you can multiply the parameter count by the bytes per parameter. For example, for a 7B (7 billion) parameter model, if loaded in FP16/BF16, the pure weight occupancy is about: $7 \times 10^9 \times 2 \text{ Bytes} \approx 14 \text{ GB}$. This is not the complete peak memory for inference or training.

### Q1 Mechanism Calculator: Convert bit width to theoretical weight capacity

First predict the capacity of a 7B model in different formats, then run the calculator. Afterward, change only `MODEL_PARAMS_B` or `MODEL_DTYPE` and check whether the capacity changes linearly with the parameter count and bit width.


```python
DTYPE_BITS = {
    'fp32': 32,
    'fp16': 16,
    'bf16': 16,
    'int8': 8,
    'int4': 4,
}
BITS_PER_BYTE = 8
PARAMS_PER_BILLION = 1_000_000_000
BYTES_PER_DECIMAL_GB = 1_000_000_000


def calculate_model_memory(num_params_b, dtype):
    """
    Compute the memory occupancy of the model parameters
    
    Args:
        num_params_b: parameter count (unit: B, i.e. billions)
        dtype: data type, one of 'fp32', 'fp16', 'bf16', 'int8', 'int4'
    
    Returns:
        memory_gb: theoretical weight occupancy (decimal GB, excluding quantization metadata and runtime buffers)
    
    Examples:
        >>> calculate_model_memory(7, 'fp16')
        14.0
        >>> calculate_model_memory(7, 'int8')
        7.0
    """
    if num_params_b < 0:
        raise ValueError("num_params_b must be non-negative")

    try:
        bits_per_element = DTYPE_BITS[dtype]
    except KeyError as exc:
        raise ValueError(f'unsupported dtype: {dtype}') from exc
    num_params = num_params_b * PARAMS_PER_BILLION
    total_bits = num_params * bits_per_element
    total_bytes = total_bits / BITS_PER_BYTE
    memory_gb = total_bytes / BYTES_PER_DECIMAL_GB
    return memory_gb
```


```python
def test_calculate_model_memory():
    """Verify the bit-width ratio, the parameter-scale ratio, and invalid inputs."""
    assert calculate_model_memory(7, 'fp16') == 14
    assert calculate_model_memory(7, 'int8') == 7
    assert calculate_model_memory(13, 'fp16') == 26
    assert calculate_model_memory(70, 'int4') == 35
    assert calculate_model_memory(14, 'fp16') == 2 * calculate_model_memory(7, 'fp16')
    try:
        calculate_model_memory(7, 'unknown')
    except ValueError:
        pass
    else:
        raise AssertionError('未知 dtype 必须报错')

test_calculate_model_memory()
```

### Q1 Comparison Check: Fix the model scale, change only the data format

Below are the explicit editable inputs of this section. When the model scale is kept unchanged, the capacity difference comes only from the bit width per element; this is still the theoretical weight lower bound.


```python
MODEL_NAME = '7B 教学模型'  # Editable: output label only.
MODEL_PARAMS_B = 7          # Editable: model parameter count, unit B.
DTYPE_CANDIDATES = ['fp32', 'fp16', 'bf16', 'int8', 'int4']  # Editable: candidate formats.

print(f"{MODEL_NAME} 的理论权重容量：")
print("-" * 40)
for dtype in DTYPE_CANDIDATES:
    memory = calculate_model_memory(MODEL_PARAMS_B, dtype)
    print(f"{dtype.upper():<8} {memory:>6.1f} GB")
```

## Q2: Both are 16-bit, so how do the bit distributions of FP16 and BF16 affect range and precision?


This involves the bit distribution design of floating point numbers at the lowest level: a floating point number consists of a **sign bit (Sign)** + **exponent bits (Exponent)** + **mantissa bits / fraction bits (Mantissa/Fraction)**.
The core rule is: **the exponent bits determine the size of the numerical range, and the mantissa bits determine the precision of the value.**

1. **The structure of FP16**: 1 sign bit + **5 exponent bits** + 10 mantissa bits.
   - With 5 exponent bits, the largest value it can represent is only **65504**.
   - Because the mantissa is long, it represents the fractional part very "finely".

2. **The structure of BF16 (Brain Float 16)**: 1 sign bit + **8 exponent bits** + 7 mantissa bits.
   - It was invented specifically for deep learning by Google Brain. It is actually FP32 (8 exponent bits) with the following 16 mantissa bits cut off directly!
   - Because it has 8 exponent bits, the exponent range of BF16 is the same as FP32 (the largest finite value is about $3.4 \times 10^{38}$), so in many training workloads it overflows less easily because of insufficient range than FP16; this does not mean Inf/NaN cannot occur. The cost is that the mantissa bits drop from 10 to 7, so the rounding precision is relatively lower.

The bit distribution first explains the numerical range and precision; the training-state ledger, optimizer state, and Master Weights are covered later in Q4 and the memory-ledger section.

### Q2 Comparison Check: Range and rounding error after real conversion

Use the same set of FP32 values, convert them to FP16 and BF16 respectively, and observe the largest finite value, machine precision, number of finite values, and conversion error. The only changed variable is the target dtype.


```python
import torch

TORCH_FLOAT_DTYPES = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


def cast_error_report(values, dtype_name):
    """Convert the same FP32 input to the target dtype and report range and rounding error."""
    if dtype_name not in TORCH_FLOAT_DTYPES:
        raise ValueError(f'unsupported float dtype: {dtype_name}')
    source = torch.tensor(values, dtype=torch.float32)
    target_dtype = TORCH_FLOAT_DTYPES[dtype_name]
    converted = source.to(target_dtype)
    restored = converted.to(torch.float32)
    info = torch.finfo(target_dtype)
    finite_mask = torch.isfinite(converted)
    return {
        'dtype': dtype_name,
        'max_finite': info.max,
        'eps_near_one': info.eps,
        'finite_count': int(finite_mask.sum().item()),
        'total_count': source.numel(),
        'max_abs_error': float((restored[finite_mask] - source[finite_mask]).abs().max().item()),
        'restored': restored.tolist(),
    }
```


```python
def test_cast_error_report():
    """Verify the range-precision trade-off between FP16 and BF16 when both are 16-bit."""
    values = [1.0, 1.001, 60_000.0, 70_000.0]
    fp16 = cast_error_report(values, 'fp16')
    bf16 = cast_error_report(values, 'bf16')
    assert fp16['finite_count'] == 3
    assert bf16['finite_count'] == 4
    assert fp16['eps_near_one'] < bf16['eps_near_one']
    try:
        cast_error_report(values, 'int8')
    except ValueError:
        pass
    else:
        raise AssertionError('非浮点 dtype 必须报错')

test_cast_error_report()
```


```python
PROBE_VALUES = [1.0, 1.001, 60_000.0, 70_000.0]  # Editable: includes both precision and range pressure.
for dtype_name in ['fp16', 'bf16']:
    report = cast_error_report(PROBE_VALUES, dtype_name)
    print(dtype_name, '->', report)
```

## Q3: What problems can loss scaling solve for FP16?

FP16 has a narrow range: overly large forward values may become `Inf`, and overly small gradients may become zero during conversion. Loss scaling first enlarges the gradients and then shrinks them after the low-precision conversion, mainly mitigating underflow of small gradients; it cannot repair overflow that has already occurred during the forward stage.

BF16 has a larger exponent range and usually relies less on loss scaling, but its mantissa is shorter, so the rounding error may be higher than FP16. Both need fixed inputs to check finite values, the zero fraction, and error; stability cannot be judged from the format name alone.

| Observation | FP16 | BF16 |
|:---|:---|:---|
| Range risk | Overflow and underflow of small gradients | Range close to FP32 |
| Resolution | Longer mantissa | Shorter mantissa |
| Common training control | Dynamic loss scaling | Usually no loss scaling, but finite-value checks are still needed |

### Q3 Comparison Check: Fix the gradients, change only the dtype and scale factor

```python
def gradient_survival_report(gradients, dtype_name, loss_scale=1.0):
    """Simulate scaling, low-precision conversion, and unscaling, and count the gradients that can still participate in an update."""
    if loss_scale <= 0:
        raise ValueError('loss_scale must be positive')
    if dtype_name not in {'fp16', 'bf16'}:
        raise ValueError('dtype_name must be fp16 or bf16')
    source = torch.tensor(gradients, dtype=torch.float32)
    target_dtype = TORCH_FLOAT_DTYPES[dtype_name]
    low_precision = (source * loss_scale).to(target_dtype)
    restored = low_precision.to(torch.float32) / loss_scale
    return {
        'dtype': dtype_name,
        'loss_scale': loss_scale,
        'nonzero_count': int((restored != 0).sum().item()),
        'finite_count': int(torch.isfinite(restored).sum().item()),
        'restored': restored.tolist(),
    }
```


```python
TINY_GRADIENTS = [1e-4, 1e-6, 1e-8, 1e-10]  # Editable: observe when smaller gradients disappear.
fp16_unscaled = gradient_survival_report(TINY_GRADIENTS, 'fp16', loss_scale=1.0)
fp16_scaled = gradient_survival_report(TINY_GRADIENTS, 'fp16', loss_scale=1024.0)
bf16_unscaled = gradient_survival_report(TINY_GRADIENTS, 'bf16', loss_scale=1.0)

assert fp16_scaled['nonzero_count'] >= fp16_unscaled['nonzero_count']
assert bf16_unscaled['finite_count'] == len(TINY_GRADIENTS)
print('FP16 without scaling:', fp16_unscaled)
print('FP16 with scaling:   ', fp16_scaled)
print('BF16 without scaling:', bf16_unscaled)
```

## Q4: Why may mixed-precision training still keep FP32 Master Weights?


In a common mixed-precision training configuration, part of the forward and backward computation uses 16-bit, but the parameter update may continue to use FP32 state. Whether Master Weights are kept depends on the optimizer, the framework, and the AMP implementation.

But the problem occurs at the **parameter update (Optimizer Step)** step:
$$ W_{new} = W_{old} - \text{Learning\_Rate} \times \text{Gradient} $$

In the later stages of large-model training, the learning rate (LR) is usually very small (for example $10^{-5}$), and the computed gradients are usually also small. Their product is an extremely tiny update $\Delta W$.
If $W_{old}$ is also stored only in a 16-bit format, the limited mantissa precision may make some very small updates disappear during rounding (for example, $1.0 + 0.0000001$ may still be close to $1.0$ at low precision). Therefore, some implementations accumulate updates on an FP32 copy and then convert the result to the dtype used by forward computation.

Therefore, some full-parameter mixed-precision training implementations keep a copy of **FP32 Master Weights**: gradients participate in the update on the high-precision copy, and the result is then converted to the dtype used by forward computation. It is part of the training-state ledger and should not be confused with the occupancy of static model weights.
The capacity estimate below is only used to observe the theoretical increment of Master Weights and the static weight capacity; it does not simulate activations, communication buffers, allocator peaks, or real OOM boundaries.

### Q4 Mechanism Calculator: Build the training-state ledger item by item

Compute parameters, gradients, Adam first moment, Adam second moment, and optional Master Weights separately. Explicit components avoid adding `12 bytes/parameter` and Master Weights together again.


```python
def training_state_ledger(
    num_params_b,
    parameter_dtype='fp16',
    gradient_dtype='fp16',
    moment_dtype='fp32',
    keep_master_weights=True,
):
    """Return the static training-state ledger; does not include activations, temporary buffers, or allocator reservations."""
    if num_params_b <= 0:
        raise ValueError('num_params_b must be positive')
    for dtype_name in (parameter_dtype, gradient_dtype, moment_dtype):
        if dtype_name not in DTYPE_BITS:
            raise ValueError(f'unsupported dtype: {dtype_name}')

    def component_gb(dtype_name, copies=1):
        return calculate_model_memory(num_params_b, dtype_name) * copies

    components = {
        'parameters_gb': component_gb(parameter_dtype),
        'gradients_gb': component_gb(gradient_dtype),
        'adam_first_moment_gb': component_gb(moment_dtype),
        'adam_second_moment_gb': component_gb(moment_dtype),
        'master_weights_gb': component_gb('fp32') if keep_master_weights else 0.0,
    }
    return {
        'components': components,
        'total_gb': sum(components.values()),
        'scope': 'static_training_states_only',
    }


def test_training_state_ledger():
    """Verify that each training state is counted only once and check the Master Weights increment."""
    with_master = training_state_ledger(7, keep_master_weights=True)
    without_master = training_state_ledger(7, keep_master_weights=False)
    assert with_master['components']['parameters_gb'] == 14
    assert with_master['components']['adam_first_moment_gb'] == 28
    assert with_master['components']['adam_second_moment_gb'] == 28
    assert with_master['components']['master_weights_gb'] == 28
    assert with_master['total_gb'] - without_master['total_gb'] == 28

test_training_state_ledger()
```


```python
TRAINING_PARAMS_B = 7               # Editable: parameter count, unit B.
KEEP_MASTER_WEIGHTS = True          # Editable: whether to keep an extra FP32 parameter copy.
ledger = training_state_ledger(
    TRAINING_PARAMS_B,
    parameter_dtype='fp16',
    gradient_dtype='fp16',
    moment_dtype='fp32',
    keep_master_weights=KEEP_MASTER_WEIGHTS,
)
for name, size_gb in ledger['components'].items():
    print(f'{name:<28} {size_gb:>6.1f} GB')
print('-' * 40)
print(f"{'total_gb':<28} {ledger['total_gb']:>6.1f} GB")
```

## Q5: Why did A100 introduce both the BF16 Tensor Core and the TF32 path?


The key change in A100 is not simply adding another storage dtype, but providing two different compute paths at the same time: BF16 can serve as a 16-bit Tensor Core input, and TF32 lets FP32 inputs have a chance to enter Tensor Core matrix multiplication. First distinguish "how the tensor is stored" from "how matrix multiplication is executed", then discuss throughput, numerical error, and configuration.

**1. Native BF16 Tensor Core support**
- On the Tensor Core matrix multiply-add path of V100, FP16 inputs are mainly used; this does not mean all computation on V100 can only use FP16.
- The A100 Tensor Core adds a BF16 multiply-add path. BF16 keeps the same exponent bit width as FP32 and usually has a larger dynamic range than FP16; whether a gain is obtained still depends on the matrix size, operator implementation, and workload.

**2. TF32 is a compute path for FP32 inputs**
- TF32 is not a 19-bit model dtype that needs to be stored separately; inputs and outputs can still be stored as FP32, and the matrix multiplication internally uses the Tensor Core path with lower effective precision.
- Configurations such as `torch.backends.cuda.matmul.allow_tf32` affect whether some FP32 matrix multiplications use TF32; the actual behavior also depends on PyTorch, CUDA, the operator, and the hardware. A configuration switch cannot be treated directly as a measured throughput conclusion.
- Therefore, the core judgment of Q5 is: FP32 / TF32 mainly change the compute path, whereas BF16 / FP16 change both the storage width and the compute path; real Tensor Core usage requires a fixed workload and profiler verification.

### Q5 Mini Check: Candidate compute paths on A100

Compare the storage representation and candidate compute paths of FP32, TF32, BF16, and FP16; the results only represent theoretical paths, and real Tensor Core usage requires GPU profiler verification.


```python
def a100_precision_path(requested_dtype, allow_tf32=True):
    """Return the theoretical path candidates on A100; does not estimate speedup or prove real Tensor Core execution."""
    table = {
        'fp32': {'storage_dtype': 'fp32', 'compute_path': 'tf32' if allow_tf32 else 'fp32', 'accumulation_dtype': 'fp32', 'tensor_core_candidate': allow_tf32},
        'tf32': {'storage_dtype': 'fp32', 'compute_path': 'tf32', 'accumulation_dtype': 'fp32', 'tensor_core_candidate': True},
        'bf16': {'storage_dtype': 'bf16', 'compute_path': 'bf16_tensor_core', 'accumulation_dtype': 'fp32', 'tensor_core_candidate': True},
        'fp16': {'storage_dtype': 'fp16', 'compute_path': 'fp16_tensor_core', 'accumulation_dtype': 'fp32', 'tensor_core_candidate': True},
    }
    if requested_dtype not in table:
        raise ValueError(f'unsupported dtype: {requested_dtype}')
    return {**table[requested_dtype], 'requires_gpu_validation': True}

for dtype in ['fp32', 'tf32', 'bf16', 'fp16']:
    print(dtype, '->', a100_precision_path(dtype))
assert a100_precision_path('fp32', allow_tf32=False)['compute_path'] == 'fp32'
assert a100_precision_path('fp32')['storage_dtype'] == 'fp32'
assert a100_precision_path('bf16')['tensor_core_candidate'] is True
try:
    a100_precision_path('int8')
except ValueError:
    print('✅ dtype 边界校验通过')
else:
    raise AssertionError('本题只比较 A100 的 FP32 / TF32 / BF16 / FP16 路径')
```

## Q6: Why does the H100 FP8 Tensor Core usually use both E4M3 and E5M2?


As model scale and throughput requirements increase, FP8 provides lower storage and compute costs than 16-bit; the H100 Tensor Core provides a native matrix computation path for FP8. Here "format choice" and "hardware support" should be understood separately: the format determines the numerical range and precision, and the hardware and framework determine whether it actually enters the FP8 compute path.

An 8-bit format must trade off exponent range against mantissa precision, so common FP8 formats include two variants with different emphases:

1. **E4M3 format** (4 exponent bits + 3 mantissa bits):
   - **Emphasis: precision.**
   - **Use:** often used for forward propagation and activations, but the actual choice depends on the model, quantization strategy, and framework implementation.
2. **E5M2 format** (5 exponent bits + 2 mantissa bits):
   - **Emphasis: dynamic range.**
   - **Use:** often used for tensors such as gradients that need a larger dynamic range; the actual mapping also depends on the training implementation and numerical calibration.

A common `HYBRID` recipe uses E4M3 for forward and E5M2 for backward: forward activations value precision more, and backward gradients value dynamic range more. But this is not a fixed rule for all models; the specific choice is also affected by the tensor distribution, scaling factors, and the framework recipe. FP8 usually needs a scaling factor computed from the `amax` history to map high-precision tensors into the limited FP8 range.

The code below only verifies the format policy and the scaling requirement; it does not perform FP8 quantization and cannot prove that H100 has used the FP8 Tensor Core; the real hardware path needs Transformer Engine, the CUDA profiler, or a fixed-workload benchmark.

### Q6 Mini Check: FP8 format policy and scaling requirement

Compare the `HYBRID`, `E4M3`, and `E5M2` policies, and record the forward / backward format and whether a scaling factor is required.


```python
FP8_MAX_FINITE = {'E4M3': 448.0, 'E5M2': 57_344.0}


def fp8_scale_from_amax(amax, fp8_format):
    """Compute the theoretical scaling factor that maps to the FP8 range from the observed absolute maximum."""
    if amax <= 0:
        raise ValueError('amax must be positive')
    if fp8_format not in FP8_MAX_FINITE:
        raise ValueError(f'unsupported FP8 format: {fp8_format}')
    return FP8_MAX_FINITE[fp8_format] / amax


def fp8_variant_policy(recipe='hybrid', forward_amax=1.0, backward_amax=1.0):
    """Return the forward/backward formats and the theoretical scaling factors derived from their respective amax."""
    recipes = {
        'hybrid': {
            'forward': {'format': 'E4M3', 'exp_bits': 4, 'mantissa_bits': 3, 'focus': 'precision'},
            'backward': {'format': 'E5M2', 'exp_bits': 5, 'mantissa_bits': 2, 'focus': 'range'},
        },
        'e4m3': {
            'forward': {'format': 'E4M3', 'exp_bits': 4, 'mantissa_bits': 3, 'focus': 'precision'},
            'backward': {'format': 'E4M3', 'exp_bits': 4, 'mantissa_bits': 3, 'focus': 'precision'},
        },
        'e5m2': {
            'forward': {'format': 'E5M2', 'exp_bits': 5, 'mantissa_bits': 2, 'focus': 'range'},
            'backward': {'format': 'E5M2', 'exp_bits': 5, 'mantissa_bits': 2, 'focus': 'range'},
        },
    }
    if recipe not in recipes:
        raise ValueError(f'未知 FP8 recipe: {recipe}')
    stages = recipes[recipe]
    return {
        'recipe': recipe,
        'stages': stages,
        'forward_scale': fp8_scale_from_amax(forward_amax, stages['forward']['format']),
        'backward_scale': fp8_scale_from_amax(backward_amax, stages['backward']['format']),
        'scaling_required': True,
        'scaling_source': 'amax history',
        'requires_gpu_validation': True,
    }

FORWARD_AMAX = 12.0   # Editable: absolute maximum of the forward activations.
BACKWARD_AMAX = 800.0 # Editable: absolute maximum of the backward gradients.
for recipe in ['hybrid', 'e4m3', 'e5m2']:
    print(recipe, '->', fp8_variant_policy(recipe, FORWARD_AMAX, BACKWARD_AMAX))
assert fp8_variant_policy('hybrid')['stages']['forward']['format'] == 'E4M3'
assert fp8_variant_policy('hybrid')['stages']['backward']['format'] == 'E5M2'
assert fp8_variant_policy('hybrid')['scaling_required'] is True
assert fp8_variant_policy('hybrid', 2.0, 2.0)['forward_scale'] == 224.0
try:
    fp8_variant_policy('unknown')
except ValueError:
    print('✅ 非法 FP8 recipe 校验通过')
else:
    raise AssertionError('未知 FP8 recipe 应报错')
```

---

## Related Reading
After this section you can continue into mixed-precision execution and the training-memory budget; consult the PyTorch AMP documentation when you need to check framework behavior.
- [PyTorch Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html): see how autocast and GradScaler participate in mixed-precision execution.
- [06. VRAM Calculation and ZeRO](./06_VRAM_Calculation_and_ZeRO.md)
- [12. TensorCore and Mixed Precision](./12_TensorCore_and_Mixed_Precision.md)

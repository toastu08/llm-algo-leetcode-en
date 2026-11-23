# 12. TensorCore and Mixed Precision | Tensor Core 与混合精度

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `hardware systems`, `Tensor Core`, `mixed precision` | **Audience:** learners who need to understand the cost of low-precision training and inference

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

Mixed precision chooses different data types for storage, computation, and accumulation, and tries to use the Tensor Core to accelerate matrix operations. A lower bit width may reduce capacity and transfer, but the hardware path and numerical stability still need to be checked separately.

This section establishes the relationship "data type → matrix shape → accumulation precision → execution conditions". After completing it, you should be able to distinguish theoretical low-precision gains from real Tensor Core paths and judge when throughput and numerical error need further verification.


**Keywords:** `FP16`, `BF16`, `Tensor Core`

![Concept relationships in this section](../public/01_Hardware_Math_and_Systems/12_tensorcore_precision_map.svg)

---

## Prerequisite Reading

**Overview:** First master data types and the GPU memory hierarchy, then see how the Tensor Core handles matrix multiply-add and how mixed precision trades off throughput, memory, and stability.

- [Part 01 · 01 Data Types and Precision](./01_Data_Types_and_Precision.md)
- [Part 01 · 03 GPU Architecture and Memory](./03_GPU_Architecture_and_Memory.md)
- [Part 01 · 06 VRAM Calculation and ZeRO](./06_VRAM_Calculation_and_ZeRO.md)

## Q1: What exactly is the Tensor Core, and why is it better suited to matrix computation than ordinary CUDA Cores?


The Tensor Core is not a "faster scalar arithmetic unit", but a hardware path designed specifically for matrix multiply-add. Ordinary CUDA Cores are more like executing a scalar FMA per element, whereas the Tensor Core packs a small block of matrix multiply-add into one MMA (Matrix Multiply-Accumulate).

The importance of this lies in the fact that the most expensive computation in large models almost always comes from GEMM, that is, matrix multiplication. If a compute unit can handle more multiply-add operations at once and the data reuse path is shorter, the same clock budget can complete more work.

The relationship between mixed precision and the Tensor Core is here as well: low-precision inputs can make multiplication throughput higher, while a high-precision accumulator preserves the stability of the result. In other words, the Tensor Core is not simply "boosting speed"; it raises throughput by organizing data in a more suitable way.

### Q1 Mini Check: Why matrix computation is better suited to packed execution

Compare the idea of scalar FMA with blocked MMA and observe how matrix-block processing changes the amount of computation per unit time.

```python
def gemm_flops(m, n, k):
    """Compute the number of multiply-add FLOPs of a dense GEMM; does not represent the actual hardware elapsed time."""
    if min(m, n, k) <= 0:
        raise ValueError('m、n、k 必须为正数')
    return 2 * m * n * k

# A 1024x1024 matrix multiplication
m = n = k = 1024
flops = gemm_flops(m, n, k)
print(f'GEMM FLOPs: {flops / 1e9:.2f} GFLOPs')
print('Tensor Core 的意义不是改变 FLOPs 数量，而是提高单位时间可完成的矩阵乘加密度。')
```

## Q2: What are the differences among FP16, BF16, and FP32, and why is mixed precision not simply "the lower the worse"?


Precision selection must consider the representation range, bytes per element, and compute path at the same time. Mixed precision does not change all data to the lowest precision; it lets inputs suited to low precision take the low-precision path and keeps sensitive accumulation or state at higher precision. The table below first puts the common dtypes on the same footing.

| dtype | Bytes per element | Main characteristics | Common use | What needs GPU verification |
| --- | ---: | --- | --- | --- |
| FP32 | 4 | Relatively high range and precision | Numerical stability baseline, accumulation | Actual throughput and peak memory |
| FP16 | 2 | Lower storage and transfer cost, narrower dynamic range | Tensor Core input, inference | Whether it overflows, whether hardware acceleration is used |
| BF16 | 2 | Exponent range close to FP32, lower mantissa precision | Training and some inference paths | Whether the hardware natively supports it, throughput change |
| FP8 | 1 | Fewer bytes, but stronger range and precision constraints | Specific hardware and quantization paths | Kernel support and task quality |

### Q2 Mini Check: How much does the memory usage differ between precisions?

For the same tensor, changing only the dtype directly shows the change in memory and bandwidth pressure.

```python
def tensor_storage_bytes(numel, dtype_bytes):
    """Compute the theoretical storage bytes of a tensor, excluding the allocator and temporary workspace."""
    if numel < 0 or dtype_bytes <= 0:
        raise ValueError('numel 不能为负数，dtype_bytes 必须为正数')
    return numel * dtype_bytes

shape = (4096, 4096)
numel = shape[0] * shape[1]
for name, bytes_per_elem in [('FP32', 4), ('BF16/FP16', 2), ('FP8', 1)]:
    size_mb = tensor_storage_bytes(numel, bytes_per_elem) / 1024 / 1024
    print(f'{name:8s}: {size_mb:8.2f} MB')

assert tensor_storage_bytes(1024, 4) == 4096
assert tensor_storage_bytes(1024, 2) == 2048
print('✅ dtype 存储量计算通过；实际 GPU peak memory 仍需单独测量')
```

## Q3: Why does precision selection affect memory, throughput, and the compute path at the same time?


Precision is not a pure numerical choice; it rewrites three costs at the same time:

1. **Memory cost**: how many bytes each element occupies determines the volume of model parameters, activations, and the KV cache.
2. **Transfer cost**: for the same total byte count, the transfer time directly affects whether the bandwidth bottleneck is obvious.
3. **Compute path cost**: some hardware paths have dedicated acceleration for specific low-precision formats, and the Tensor Core is a typical example.

This is also why low-precision inference, mixed-precision training, and throughput comparison are often discussed together: dtype changes the memory, bandwidth, and compute path of model execution at the same time.

Therefore, when looking at precision problems, you cannot only ask "can it still compute correctly", but also ask "is this path more memory-efficient, moves less data, and is easier to fill the hardware". The INT8 result below only represents the theoretical lower bound of 1 byte per element; a real quantized model must also consider scale, zero-point, packing, metadata, and the corresponding kernel.

### Q3 Mini Check: How the byte count affects the model size

Fix the parameter count and see the direct impact of different dtypes on the model size.

```python
def precision_impact_ledger(params, activation_elements, kv_elements, dtype_bytes):
    """Estimate the theoretical storage of weights, activations, and the KV Cache under the same dtype.

    The result does not include optimizer state, the CUDA allocator, workspace, or actual kernel throughput.
    """
    values = (params, activation_elements, kv_elements)
    if any(value < 0 for value in values) or dtype_bytes <= 0:
        raise ValueError('元素数量不能为负数，dtype_bytes 必须为正数')
    return {
        'weights_gb': round(tensor_storage_bytes(params, dtype_bytes) / 1e9, 3),
        'activation_gb': round(tensor_storage_bytes(activation_elements, dtype_bytes) / 1e9, 3),
        'kv_cache_gb': round(tensor_storage_bytes(kv_elements, dtype_bytes) / 1e9, 3),
    }

params = 7_000_000_000
activation_elements = 2_000_000_000
kv_elements = 1_000_000_000
for name, bytes_per_elem in [('FP32', 4), ('BF16/FP16', 2), ('INT8', 1)]:
    ledger = precision_impact_ledger(params, activation_elements, kv_elements, bytes_per_elem)
    print(name, '->', ledger)

fp32 = precision_impact_ledger(params, activation_elements, kv_elements, 4)
bf16 = precision_impact_ledger(params, activation_elements, kv_elements, 2)
assert bf16['weights_gb'] == fp32['weights_gb'] / 2
assert bf16['activation_gb'] == fp32['activation_gb'] / 2
assert bf16['kv_cache_gb'] == fp32['kv_cache_gb'] / 2
try:
    precision_impact_ledger(params, activation_elements, kv_elements, 0)
except ValueError:
    print('✅ dtype 字节数校验通过')
else:
    raise AssertionError('dtype_bytes 为 0 时应报错')
```

## Related Reading
This section continues from data types into mixed-precision training and low-bit adaptation; when reading, distinguish theoretical byte counts, hardware execution paths, and task quality.
- [01. Data Types and Precision](./01_Data_Types_and_Precision.md)
- [PyTorch Automatic Mixed Precision documentation](https://pytorch.org/docs/stable/amp.html)
- [26. QLoRA and 4-bit Quantization](../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.md)
---
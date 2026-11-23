# 05. PyTorch Tensor Fundamentals | PyTorch 张量基础操作

**Difficulty:** Easy | **Environment:** CPU-first | **Tags:** `PyTorch`, `Tensor`, `dtype`, `device` | **Audience:** learners moving from NumPy to PyTorch

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/05_PyTorch_Tensor_Fundamentals.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## Section Guide

On top of arrays, tensors add data type, device, gradient, and storage semantics. Tensors with the same shape may still produce different results because of different data types, devices, or sharing relationships.

After completing this section, you should be able to read the basic contract of a tensor, distinguish copying from shared storage, and judge whether numeric tensors and index tensors meet the input requirements.

**Keywords:** `tensor`, `dtype`, `device`, `storage sharing`

![Basic tensor concept relationships](../public/00_Prerequisites/05_tensor_fundamentals_map.svg)

## Prerequisite Reading
**Overview:** First master NumPy shapes and broadcasting, then observe the dtype, device, and storage semantics that tensors add.
- [02. NumPy and Einsum](./02_NumPy_and_Einsum.md)
- [0B Group Page](./0B.md)

## Q1: How do you read a tensor's shape, dtype, and device?

shape describes the structure, dtype describes the numeric interpretation and per-element size, and device describes the compute location of the data. When checking inputs, record all three instead of only looking at the printed values.

```python
import numpy as np
import torch


def describe_tensor(tensor):
    """Return the tensor's structure, numeric type, device, and per-element bytes."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('expected a torch.Tensor')
    return {
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype).replace('torch.', ''),
        'device': str(tensor.device),
        'element_size_bytes': tensor.element_size(),
        'requires_grad': tensor.requires_grad,
    }


tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(describe_tensor(tensor))
```

## Q1 verification: Are the structural and numeric attributes complete?

```python
tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
description = describe_tensor(tensor)
assert description == {
    'shape': (2, 2),
    'dtype': 'float32',
    'device': 'cpu',
    'element_size_bytes': 4,
    'requires_grad': False,
}

try:
    describe_tensor(np.zeros(2))
except TypeError as error:
    assert 'Tensor' in str(error)
else:
    raise AssertionError('非 Tensor 输入应触发 TypeError')
print('✅ Tensor 属性通过')
```

## Q2: Which construction methods share the underlying data?

`torch.tensor(array)` copies the input; `torch.from_numpy(array)` and `torch.as_tensor(array)` with a compatible dtype on CPU usually share the NumPy memory; `clone()` creates an independent tensor. Whether data is shared determines whether modifying one side changes the other, so after conversion you cannot only check that the values are equal.

```python
array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
copied = torch.tensor(array)
from_numpy = torch.from_numpy(array)
as_tensor = torch.as_tensor(array)
cloned = from_numpy.clone()

array[0] = 99.0
print({
    'numpy': array.tolist(),
    'torch.tensor': copied.tolist(),
    'from_numpy': from_numpy.tolist(),
    'as_tensor': as_tensor.tolist(),
    'clone': cloned.tolist(),
})
```

## Q2 verification: Do the sharing and copying relationships match expectations?

```python
array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
copied = torch.tensor(array)
shared = torch.from_numpy(array)
wrapped = torch.as_tensor(array)
cloned = shared.clone()

array[1] = 77.0
assert copied[1].item() == 2.0
assert shared[1].item() == 77.0
assert wrapped[1].item() == 77.0
assert cloned[1].item() == 2.0
print('✅ Tensor 复制与共享语义通过')
```

## Q3: When does a dtype or device conversion reuse the original tensor?

`.to()` usually returns the original tensor when both dtype and device are unchanged; when the type or device changes, it creates new storage, and `copy=True` can explicitly request a copy. Conversion must preserve shape and numeric semantics, and a real device migration also requires the target runtime environment to be available.

```python
def convert_tensor(tensor, *, dtype=None, device=None, copy=False):
    """Convert a tensor's dtype/device; reuse is allowed when nothing changes, copy=True forces a copy."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('expected a torch.Tensor')
    target_dtype = tensor.dtype if dtype is None else dtype
    target_device = tensor.device if device is None else device
    return tensor.to(dtype=target_dtype, device=target_device, copy=copy)


source = torch.tensor([1.0, 2.0], dtype=torch.float32)
reused = convert_tensor(source)
converted = convert_tensor(source, dtype=torch.float64)
forced_copy = convert_tensor(source, copy=True)
print({
    'reused_object': reused is source,
    'converted_dtype': str(converted.dtype),
    'forced_copy_storage': forced_copy.data_ptr() != source.data_ptr(),
})
```

## Q3 verification: Do reuse, type conversion, and forced copying each preserve their semantics?

```python
source = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
reused = convert_tensor(source)
converted = convert_tensor(source, dtype=torch.float64)
forced_copy = convert_tensor(source, copy=True)

assert reused is source
assert converted.shape == source.shape and converted.dtype == torch.float64
assert converted.data_ptr() != source.data_ptr()
assert forced_copy.shape == source.shape and forced_copy.data_ptr() != source.data_ptr()
assert torch.equal(forced_copy, source)

try:
    convert_tensor([1.0, 2.0])
except TypeError as error:
    assert 'Tensor' in str(error)
else:
    raise AssertionError('非 Tensor 输入应触发 TypeError')
print('✅ dtype/device 转换语义通过')
```

## Q4: How do the dtype/device contracts of numeric tensors and index tensors differ?

Floating-point tensors participate in model computation; Embedding, table lookup, and position selection usually require `torch.long` indices. Tensors participating in the same operator must also be on compatible devices. The entry function can explicitly convert the index dtype and device, but should not silently accept floating-point fractional indices.

```python
def prepare_indices(indices, *, device):
    """Move integer indices to the target device and unify them as torch.long."""
    indices = torch.as_tensor(indices)
    if indices.dtype.is_floating_point or indices.dtype == torch.bool:
        raise TypeError('indices must contain integers')
    return indices.to(device=device, dtype=torch.long)


embedding_table = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
indices = prepare_indices([3, 1, 4], device=embedding_table.device)
selected = embedding_table.index_select(0, indices)
print('indices:', indices.dtype, indices.device)
print('selected shape:', tuple(selected.shape))
```

## Q4 verification: Are the index dtype, device, and output shape correct?

```python
table = torch.arange(20, dtype=torch.float32).reshape(5, 4)
indices = prepare_indices(np.array([3, 1], dtype=np.int32), device=table.device)
selected = table.index_select(0, indices)
assert indices.dtype == torch.long
assert indices.device == table.device
assert selected.shape == (2, 4)
assert torch.equal(selected[0], table[3])

try:
    prepare_indices([1.5, 2.0], device=table.device)
except TypeError as error:
    assert 'integers' in str(error)
else:
    raise AssertionError('浮点索引应触发 TypeError')
print('✅ dtype、device 与索引契约通过')
```

### Section Summary

- shape, dtype, device, and per-element bytes together describe the basic contract of a tensor.
- `torch.tensor`, `from_numpy`, `as_tensor`, and `clone` have different copy/share semantics.
- `.to()` can reuse the original tensor when the conditions are unchanged; a type or device change produces a new representation.
- Index tensors should use an integer dtype and be on a device compatible with the object being indexed.
## Related Reading

With the tensor entry point mastered, continue with layout, indexing, and backward propagation.

- [PyTorch Tensor official documentation](https://pytorch.org/docs/stable/tensors.html)
- [06. PyTorch Tensor Layout and Indexing](./06_PyTorch_Tensor_Layout_and_Indexing.md)
- [07. PyTorch Autograd and Backward](./07_PyTorch_Autograd_and_Backward.md)
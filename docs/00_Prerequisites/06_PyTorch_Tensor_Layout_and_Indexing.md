# 06. PyTorch Tensor Layout and Indexing | PyTorch 张量布局与索引

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `PyTorch`, `layout`, `indexing`, `broadcast` | **Audience:** learners who can already create tensors and are ready to read sequence and multi-head tensor transformations

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/06_PyTorch_Tensor_Layout_and_Indexing.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## Section Guide

Shape describes the logical dimensions, stride describes the access layout, and indexing determines which positions are taken. The same shape does not imply the same layout, and dimension transformations may also change contiguity and shared storage relationships.

After completing this section, you should be able to check shape, stride, contiguity, and indexing rules together, and judge whether reshaping, broadcasting, and position selection preserve the original data semantics.

**Keywords:** `shape`, `stride`, `contiguous`, `gather`

![Tensor layout and indexing concept relationships](../public/00_Prerequisites/06_tensor_layout_indexing_map.svg)

## Prerequisite Reading
**Overview:** First master the tensor's dtype, device, and copy/share semantics, then move on to layout and indexing.
- [05. PyTorch Tensor Fundamentals](./05_PyTorch_Tensor_Fundamentals.md)
- [0B Group Page](./0B.md)

## Q1: What do shape, stride, contiguous, and shared storage each describe?

shape is the logical dimension, stride is the number of elements skipped when moving one position along each dimension, and contiguous indicates whether the layout follows the default contiguous order. Slicing and transposing usually still share storage with the original tensor, but may become non-contiguous views.

```python
import torch


def describe_layout(tensor):
    """Return shape, stride, contiguity, and the underlying storage address."""
    return {
        'shape': tuple(tensor.shape),
        'stride': tuple(tensor.stride()),
        'contiguous': tensor.is_contiguous(),
        'storage_ptr': tensor.untyped_storage().data_ptr(),
    }


base = torch.arange(12).reshape(3, 4)
sliced = base[:, 1:3]
transposed = base.t()
print('base:', describe_layout(base))
print('sliced:', describe_layout(sliced))
print('transposed:', describe_layout(transposed))
```

## Q1 verification: Do the views share storage but have different layouts?

```python
base = torch.arange(12).reshape(3, 4)
sliced = base[:, 1:3]
transposed = base.t()
assert base.is_contiguous()
assert not sliced.is_contiguous()
assert not transposed.is_contiguous()
assert describe_layout(base)['storage_ptr'] == describe_layout(sliced)['storage_ptr']
assert describe_layout(base)['storage_ptr'] == describe_layout(transposed)['storage_ptr']
print('✅ layout 与共享存储通过')
```

## Q2: Why can `view()` fail after transpose?

`transpose/permute` only changes the dimension interpretation and strides, and does not automatically rearrange the storage. `view()` requires the current layout to be directly interpretable as the target shape; `reshape()` copies when necessary, and `contiguous()` explicitly produces a contiguous layout.

```python
base = torch.arange(24).reshape(2, 3, 4)
transposed = base.transpose(1, 2)

view_failed = False
try:
    transposed.view(-1)
except RuntimeError:
    view_failed = True

reshaped = transposed.reshape(-1)
contiguous_tensor = transposed.contiguous()
print({
    'view_failed': view_failed,
    'reshape_shape': tuple(reshaped.shape),
    'contiguous': contiguous_tensor.is_contiguous(),
})
```

## Q2 verification: Do the different reshaping paths preserve the element count and order?

```python
base = torch.arange(24).reshape(2, 3, 4)
transposed = base.transpose(1, 2)
reshaped = transposed.reshape(-1)
contiguous_tensor = transposed.contiguous()
assert not transposed.is_contiguous()
assert reshaped.numel() == base.numel()
assert torch.equal(reshaped, contiguous_tensor.view(-1))
assert contiguous_tensor.is_contiguous()
assert contiguous_tensor.untyped_storage().data_ptr() != base.untyped_storage().data_ptr()
print('✅ view、reshape 与 contiguous 通过')
```

## Q3: What shape conditions must masks and broadcasting satisfy?

A per-position mask should match the target tensor shape or be clearly broadcastable; a feature bias usually requires the last dimension to equal the hidden size. Checking these conditions at the entry point avoids a successful broadcast with incorrect semantics.

```python
def mask_and_add_bias(values, visible_mask, feature_bias, fill_value=0):
    """Broadcast the feature bias first, then write fill_value into invisible positions."""
    if visible_mask.dtype != torch.bool or visible_mask.shape != values.shape:
        raise ValueError('visible_mask must be boolean and match values.shape')
    if feature_bias.ndim != 1 or feature_bias.shape[0] != values.shape[-1]:
        raise ValueError('feature_bias must match the last dimension')
    return (values + feature_bias).masked_fill(~visible_mask, fill_value)


values = torch.arange(12).reshape(3, 4)
visible_mask = values % 2 == 0
bias = torch.tensor([10, 20, 30, 40])
print(mask_and_add_bias(values, visible_mask, bias))
```

## Q3 verification: Can the mask values and illegal broadcasting be checked?

```python
values = torch.arange(8).reshape(2, 4)
visible_mask = values % 2 == 0
bias = torch.tensor([10, 20, 30, 40])
result = mask_and_add_bias(values, visible_mask, bias)
assert result.tolist() == [[10, 0, 32, 0], [14, 0, 36, 0]]

try:
    mask_and_add_bias(values, visible_mask[:, :3], bias)
except ValueError as error:
    assert 'visible_mask' in str(error)
else:
    raise AssertionError('错误 mask shape 应触发 ValueError')
print('✅ mask 与广播契约通过')
```

## Q4: By what rules do advanced indexing, `index_select`, and `gather` take values?

Advanced indexing and `index_select` select entire rows or columns along one dimension; `gather` instead requires the index to align with the output shape and specifies the source index for each position. Advanced indexing usually produces a new tensor that no longer shares storage with the original table.

```python
table = torch.tensor([
    [10, 11, 12, 13],
    [20, 21, 22, 23],
    [30, 31, 32, 33],
])
row_ids = torch.tensor([2, 0], dtype=torch.long)
positions = torch.tensor([[0, 3], [1, 2], [3, 0]], dtype=torch.long)

advanced = table[row_ids]
selected = torch.index_select(table, dim=0, index=row_ids)
gathered = torch.gather(table, dim=1, index=positions)
print('selected:', selected)
print('gathered:', gathered)
```

## Q4 verification: Are whole-row selection, per-position selection, and copy semantics correct?

```python
table = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]])
row_ids = torch.tensor([2, 0], dtype=torch.long)
positions = torch.tensor([[0, 3], [1, 2], [3, 0]], dtype=torch.long)
advanced = table[row_ids]
selected = torch.index_select(table, 0, row_ids)
gathered = torch.gather(table, 1, positions)
assert torch.equal(advanced, selected)
assert selected.tolist() == [[30, 31, 32, 33], [10, 11, 12, 13]]
assert gathered.tolist() == [[10, 13], [21, 22], [33, 30]]
assert advanced.untyped_storage().data_ptr() != table.untyped_storage().data_ptr()
print('✅ 索引与 gather 通过')
```

## Q5: How do you turn position selection and head splitting into stable inputs?

The combined workflow first selects `[B, S, D]` by token position, then splits the hidden dimension into heads, and finally arranges it as `[B, H, S, D_h]`. At each step you must check the index range, whether hidden is divisible, and whether the final layout satisfies the downstream interface.

```python
def prepare_head_input(features, positions, num_heads):
    """Select positions from [B, T, D] and arrange them into a contiguous [B, H, S, D_h]."""
    if features.ndim != 3:
        raise ValueError('features must have shape [B, T, D]')
    if positions.dtype != torch.long or positions.ndim != 1:
        raise ValueError('positions must be a one-dimensional long tensor')
    if positions.numel() and (positions.min() < 0 or positions.max() >= features.shape[1]):
        raise IndexError('positions are out of range')
    if num_heads <= 0 or features.shape[-1] % num_heads != 0:
        raise ValueError('hidden size must be divisible by num_heads')

    selected = features.index_select(1, positions)
    batch, selected_tokens, hidden = selected.shape
    head_dim = hidden // num_heads
    heads = selected.reshape(batch, selected_tokens, num_heads, head_dim).permute(0, 2, 1, 3)
    return heads.contiguous()


features = torch.arange(2 * 4 * 8).reshape(2, 4, 8)
ready = prepare_head_input(features, torch.tensor([0, 2]), num_heads=2)
print('ready shape:', tuple(ready.shape), 'contiguous:', ready.is_contiguous())
```

## Q5 verification: Does the combined preparation preserve the position contents and the target layout?

```python
features = torch.arange(2 * 4 * 8).reshape(2, 4, 8)
positions = torch.tensor([0, 2], dtype=torch.long)
ready = prepare_head_input(features, positions, num_heads=2)
assert ready.shape == (2, 2, 2, 4)
assert ready.is_contiguous()
restored = ready.permute(0, 2, 1, 3).reshape(2, 2, 8)
assert torch.equal(restored, features.index_select(1, positions))
print('✅ 输入整理闭环通过')
```

### Section Summary

- shape, stride, contiguity, and shared storage together determine a tensor's layout semantics.
- `transpose()` only changes the dimension interpretation; `reshape()` copies when necessary, and `contiguous()` explicitly produces a contiguous layout.
- Masks, biases, and indices all need explicit shape/dtype contracts, and invisible positions should preserve the fill-value semantics.
- Position selection and head splitting should be verified step by step for content conservation and the final layout.
## Related Reading

With the layout and indexing contracts established, you can continue by observing how they enter the computation graph and the attention mechanism.

- [PyTorch Tensor Views official documentation](https://pytorch.org/docs/stable/tensor_view.html)
- [07. PyTorch Autograd and Backward](./07_PyTorch_Autograd_and_Backward.md)
- [16. Attention Mechanism Intro](./16_Attention_Mechanism_Intro.md)
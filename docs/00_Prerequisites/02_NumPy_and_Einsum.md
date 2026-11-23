# 02. NumPy and Einsum | NumPy 与 Einsum

**Difficulty:** Easy | **Environment:** CPU-first | **Tags:** `NumPy`, `broadcasting`, `einsum` | **Audience:** learners preparing for Parts 2–4

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/02_NumPy_and_Einsum.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## Section Guide

The result of a multidimensional array depends not only on the values, but also on dimensions, indexing, broadcasting, and contraction relationships. Even when the shapes appear compatible, element order may change or broadcasting may be wrong.

After completing this section, you should be able to trace the elements and dimensions before and after array reshaping, judge whether broadcasting is legal, and verify matrix operations using explicit dimension relationships.

**Keywords:** `ndarray`, `broadcast`, `einsum`, `shape contract`

![NumPy shape and dimension operations](../public/00_Prerequisites/02_numpy_shape_einsum_map.svg)

## Prerequisite Reading
**Overview:** First have Python sequences, dicts, and function contracts in place, then move on to multidimensional array expressions.
- [01. Python Essentials for LLM](./01_Python_Essentials_for_LLM.md)
- [0A Group Page](./0A.md)

## Q1: How do indexing and broadcasting change an array's contents and shape?

Indexing selects positions, and broadcasting expands a smaller array into the computation along compatible dimensions. When adding a feature bias, the last dimension of the bias must match the feature dimension; when selecting samples, the length of the boolean mask must match the indexed dimension.

```python
import numpy as np


def add_feature_bias(values, bias):
    """Add a one-dimensional feature bias to a [..., feature] array."""
    values = np.asarray(values)
    bias = np.asarray(bias)
    if bias.ndim != 1 or values.shape[-1] != bias.shape[0]:
        raise ValueError('bias length must match the last feature dimension')
    return values + bias


values = np.arange(2 * 3 * 4).reshape(2, 3, 4)
bias = np.array([1, 10, 100, 1000])
adjusted = add_feature_bias(values, bias)
selected = adjusted[np.array([True, False])]
print('adjusted shape:', adjusted.shape)
print('selected shape:', selected.shape)
```

## Q1 verification: Can the broadcast result and the invalid feature dimension be checked?

```python
values = np.arange(24).reshape(2, 3, 4)
bias = np.array([1, 10, 100, 1000])
adjusted = add_feature_bias(values, bias)
assert adjusted.shape == values.shape
assert np.array_equal(adjusted[0, 0], values[0, 0] + bias)

try:
    add_feature_bias(values, np.ones(3))
except ValueError as error:
    assert 'feature' in str(error)
else:
    raise AssertionError('不匹配的 feature bias 应触发 ValueError')
print('✅ 索引与广播通过')
```

## Q2: What must reshape, transpose, and head splitting preserve?

Splitting heads changes the interpretation of dimensions and their order, but must not lose or duplicate elements. The hidden size must be divisible by the number of heads; when merging, first transpose the dimensions back to `[B, T, H, D_h]`, then restore `[B, T, D]`.

```python
def split_heads(x, num_heads):
    """Turn [B, T, D] into [B, H, T, D_h]."""
    if x.ndim != 3 or num_heads <= 0 or x.shape[-1] % num_heads != 0:
        raise ValueError('hidden size must be divisible by num_heads')
    batch, tokens, hidden = x.shape
    head_dim = hidden // num_heads
    return x.reshape(batch, tokens, num_heads, head_dim).transpose(0, 2, 1, 3)


def merge_heads(x):
    """Restore [B, H, T, D_h] to [B, T, H * D_h]."""
    if x.ndim != 4:
        raise ValueError('head tensor must be rank 4')
    batch, heads, tokens, head_dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, tokens, heads * head_dim)


x = np.arange(2 * 3 * 8).reshape(2, 3, 8)
heads = split_heads(x, num_heads=2)
print('split:', heads.shape, 'merged:', merge_heads(heads).shape)
```

## Q2 verification: Do splitting and merging preserve the elements?

```python
x = np.arange(2 * 3 * 8).reshape(2, 3, 8)
heads = split_heads(x, num_heads=2)
assert heads.shape == (2, 2, 3, 4)
assert np.array_equal(merge_heads(heads), x)

try:
    split_heads(np.zeros((2, 3, 7)), num_heads=2)
except ValueError as error:
    assert 'divisible' in str(error)
else:
    raise AssertionError('不能整除的 hidden size 应触发 ValueError')
print('✅ shape contract 通过')
```

## Q3: When should you use `einsum` to express dimension relationships?

When matrix multiplication involves named dimensions such as batch, query, and key, `einsum` can write the contraction dimensions directly into the expression. Before using it, you still need to confirm that the sizes of the dimensions being contracted match; the concise notation must not hide shape errors.

```python
def matrix_multiply(a, b):
    """Compute [M, K] × [K, N] -> [M, N]."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError('expected compatible [M, K] and [K, N] arrays')
    return np.einsum('mk,kn->mn', a, b)


def batched_pair_scores(query, key):
    """Compute the pairwise scores [B, Q, K] of [B, Q, D] and [B, K, D]."""
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError('query and key must be rank-3 arrays')
    if query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]:
        raise ValueError('batch and feature dimensions must match')
    return np.einsum('bqd,bkd->bqk', query, key)


query = np.array([[[1.0, 0.0], [0.0, 1.0]]])
key = np.array([[[1.0, 2.0], [3.0, 4.0]]])
print(batched_pair_scores(query, key))
```

## Q3 verification: Is Einsum consistent with ordinary matrix multiplication?

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
assert np.array_equal(matrix_multiply(a, b), a @ b)

query = np.array([[[1, 0], [0, 1]]])
key = np.array([[[1, 2], [3, 4]]])
expected = np.array([[[1, 3], [2, 4]]])
assert np.array_equal(batched_pair_scores(query, key), expected)
print('✅ Einsum 维度关系通过')
```

## Related Reading

After mastering array dimensions, continue with object encapsulation, or move on to PyTorch tensor layout and device semantics.

- [03. Python OOP and Utility Patterns](./03_Python_OOP_and_Utility_Patterns.md)
- [05. PyTorch Tensor Fundamentals](./05_PyTorch_Tensor_Fundamentals.md)
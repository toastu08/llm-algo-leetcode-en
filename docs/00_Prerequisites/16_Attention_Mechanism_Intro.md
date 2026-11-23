# 16. Attention Mechanism Intro | Attention 机制导论

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `PyTorch`, `Attention`, `mask` | **Audience:** learners preparing to enter Transformer structure and optimization content

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/16_Attention_Mechanism_Intro.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## Section Guide

Attention first uses Q/K to form match scores, then applies visibility constraints and Softmax to obtain read weights, and finally aggregates V. The multi-head layout changes the tensor view, while sequence length and positional mechanisms limit what standard Attention can express and carry.

After completing this section, you should be able to trace the minimal Attention loop, combine causal and padding masks, and determine the shape contract of self/cross Attention and the multi-head layout.

**Keywords:** `Q`, `K`, `V`, `mask`, `shape contract`

![The representation, visibility, and aggregation dataflow of Attention](../public/00_Prerequisites/16_attention_dataflow_map.svg)

## Prerequisite Reading
**Overview:** Section 13 gives the position of the Attention branch in the Block, and section 15 explains the statistical control before the branch reads the residual stream; this page focuses on information aggregation across tokens.
- [13. Transformer Block Dataflow and Residual Stream](./13_Transformer_Block_Dataflow_and_Residual_Stream.md)
- [15. Normalization Techniques](./15_Normalization_Techniques.md)
- [0D Group Page](./0D.md)

## Q1: How do Q/K/V form the complete Attention loop?

Q represents the query of the current token, K provides the match index, and V provides the content being aggregated. When implementing, trace `QKᵀ / √D_h → mask → softmax → weights × V` continuously, instead of splitting the four steps into unrelated formulas.


```python
import math
import torch
import torch.nn.functional as F


def build_causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))


def masked_softmax(scores, mask=None, dim=-1):
    if mask is not None:
        mask = mask.to(dtype=torch.bool)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    shifted = scores - scores.max(dim=dim, keepdim=True).values
    probs = torch.softmax(shifted, dim=dim)
    if mask is not None:
        mask_f = mask.to(dtype=probs.dtype)
        probs = probs * mask_f
        denom = probs.sum(dim=dim, keepdim=True)
        probs = torch.where(denom > 0, probs / denom.clamp_min(1e-12), torch.zeros_like(probs))
    return probs


def attention_weights(q, k, mask=None):
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    return masked_softmax(scores, mask=mask, dim=-1)


def scaled_dot_product_attention(q, k, v, mask=None):
    weights = attention_weights(q, k, mask=mask)
    output = torch.matmul(weights, v)
    return output, weights


q = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]])
k = torch.tensor([[[1.0, 2.0, 0.0], [3.0, 0.0, 1.0]]])
v = torch.tensor([[[10.0, 0.0], [0.0, 20.0]]])
mask = build_causal_mask(2).unsqueeze(0)
weights = attention_weights(q, k, mask=mask)
output, _ = scaled_dot_product_attention(q, k, v, mask=mask)
print('weights：')
print(weights)
print('output：')
print(output)

```

## Q1 verification: Are the match weights and the V aggregation both correct?

Under the causal constraint, the first query can only read the first value; the second query can read both positions. The weights and output should both keep the expected shapes and match the PyTorch reference implementation.


```python
q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
k = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
v = torch.tensor([[[10.0, 0.0], [0.0, 20.0]]])
mask = build_causal_mask(2).unsqueeze(0)
output, weights = scaled_dot_product_attention(q, k, v, mask=mask)
expected_weights = torch.tensor([[[1.0, 0.0], [0.19557033, 0.80442965]]])

assert weights.shape == (1, 2, 2)
assert output.shape == (1, 2, 2)
assert torch.allclose(weights, expected_weights, atol=1e-5, rtol=1e-5)
torch.testing.assert_close(output, torch.matmul(expected_weights, v), atol=1e-5, rtol=1e-5)
print('✅ Q/K 匹配与 V 聚合闭环通过')

```

## Q2: How do the visibility mask and a stable Softmax work together?

Autoregressive Attention uses a causal mask to hide future tokens, and a padding mask to exclude padded invalid keys; the two constraints can be combined into a `[batch, heads, query_length, key_length]` visibility mask. Subtracting the row maximum before Softmax avoids exponential overflow, and a fully masked row should return finite zero weights.


```python
def build_attention_mask(key_padding_mask, query_length, causal):
    """Build a [B, 1, S_q, S_k] visibility mask that can broadcast to multi-head scores."""
    batch_size, key_length = key_padding_mask.shape
    visible = key_padding_mask[:, None, None, :].expand(batch_size, 1, query_length, key_length)
    if causal:
        if query_length != key_length:
            raise ValueError('causal self-attention requires query_length == key_length')
        visible = visible & build_causal_mask(query_length)[None, None, :, :]
    return visible


key_padding_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
self_mask = build_attention_mask(key_padding_mask, query_length=4, causal=True)
cross_mask = build_attention_mask(key_padding_mask, query_length=2, causal=False)

scores = torch.tensor([[1.0, 2.0, 3.0], [9.0, 8.0, 7.0]])
score_mask = torch.tensor([[True, True, False], [False, False, False]])
probs = masked_softmax(scores, mask=score_mask)
print('self / cross mask shape：', self_mask.shape, cross_mask.shape)
print('masked probabilities：', probs)

```

## Q2 verification: Are the mask semantics, broadcast shape, and probability normalization correct?

The causal constraint should form lower-triangular visibility, and padding positions should always be unreadable; rows with visible entries sum to 1, and fully masked rows stay finite and return all zeros.


```python
expected_causal = torch.tensor([
    [True, False, False, False],
    [True, True, False, False],
    [True, True, True, False],
    [True, True, True, True],
])
assert torch.equal(build_causal_mask(4), expected_causal)
assert self_mask.shape == (2, 1, 4, 4)
assert cross_mask.shape == (2, 1, 2, 4)
assert not self_mask[0, 0, 0, 1]
assert not self_mask[0, 0, 3, 3]
assert cross_mask[0, 0, 0].tolist() == [True, True, True, False]
assert torch.allclose(probs[0].sum(), torch.tensor(1.0))
assert torch.equal(probs[1], torch.zeros(3))
assert torch.isfinite(probs).all()
print('✅ causal/padding mask 与稳定 Softmax 通过')

```

## Q3: How do the shape contracts of self-attention and cross-attention differ?

Both require the batch, head, and Q/K feature dimensions to support matrix multiplication, and K and V must also share the source sequence length. Self-attention usually satisfies `S_q = S_k = S_v`; cross-attention allows the query sequence length `S_q` to differ from the source sequence length `S_k`, so "all sequence lengths are the same" cannot be written as a general contract.


```python
def attention_contract_ok(q_shape, k_shape, v_shape, mode):
    """Check a [B, H, S, D]-style self-attention or cross-attention contract."""
    if mode not in {'self', 'cross'}:
        return False
    if any(len(shape) != 4 or any(type(size) is not int or size <= 0 for size in shape)
           for shape in (q_shape, k_shape, v_shape)):
        return False
    same_batch = q_shape[0] == k_shape[0] == v_shape[0]
    same_head = q_shape[1] == k_shape[1] == v_shape[1]
    score_feature_match = q_shape[3] == k_shape[3]
    source_length_match = k_shape[2] == v_shape[2]
    self_length_match = q_shape[2] == k_shape[2] if mode == 'self' else True
    return same_batch and same_head and score_feature_match and source_length_match and self_length_match


print('self contract：', attention_contract_ok((2, 4, 8, 16), (2, 4, 8, 16), (2, 4, 8, 32), mode='self'))
print('cross contract：', attention_contract_ok((2, 4, 3, 16), (2, 4, 8, 16), (2, 4, 8, 32), mode='cross'))
```

## Q3 verification: Can valid and invalid shapes of self / cross attention be distinguished?

Self-attention requires the query and source sequence lengths to match; cross-attention allows them to differ, but the K/V source lengths must match.


```python
assert attention_contract_ok((2, 4, 8, 16), (2, 4, 8, 16), (2, 4, 8, 32), mode='self')
assert attention_contract_ok((2, 4, 3, 16), (2, 4, 8, 16), (2, 4, 8, 32), mode='cross')
assert not attention_contract_ok((2, 4, 3, 16), (2, 4, 8, 16), (2, 4, 7, 32), mode='cross')
assert not attention_contract_ok((2, 4, 8), (2, 4, 8, 16), (2, 4, 8, 32), mode='self')
assert not attention_contract_ok((2, 4, 8, 16), (2, 4, 8, 16), (2, 4, 8, 32), mode='unknown')
print('✅ self / cross attention shape contract 通过')

```

## Q4: How does multi-head Attention convert between the head dimension and the hidden dimension?

Multi-head Attention splits `[B, S, D]` into `[B, H, S, D_h]`, lets different heads compute independently, and merges the results back into the hidden dimension. Splitting and merging must preserve the element order and total dimension; this layout contract is also the basis for later reading about GQA, FlashAttention, and fused kernels.


```python
def split_heads(x, num_heads):
    """Turn [B, S, D] into [B, H, S, D_h]."""
    batch_size, seq_len, hidden_size = x.shape
    if hidden_size % num_heads != 0:
        raise ValueError('hidden_size must be divisible by num_heads')
    head_dim = hidden_size // num_heads
    return x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


def merge_heads(x):
    """Merge [B, H, S, D_h] back into [B, S, D]."""
    batch_size, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)


x = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
heads = split_heads(x, num_heads=2)
restored = merge_heads(heads)
print('head shape:', tuple(heads.shape))
print('round trip:', torch.equal(restored, x))

```

## Q4 verification: Is the hand-written multi-head dataflow consistent with the PyTorch reference implementation?

Besides the layout round-trip, you should compare the hand-written Attention of each head with `scaled_dot_product_attention` to confirm that score scaling, softmax, and V aggregation have no semantic deviation.


```python
x = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8) / 10
q = split_heads(x, num_heads=2)
k = split_heads(x + 0.1, num_heads=2)
v = split_heads(x - 0.2, num_heads=2)
manual_output, _ = scaled_dot_product_attention(q, k, v)
reference_output = F.scaled_dot_product_attention(q, k, v)

assert q.shape == (1, 2, 3, 4)
assert torch.equal(merge_heads(split_heads(x, 2)), x)
assert torch.allclose(manual_output, reference_output, atol=1e-6, rtol=1e-6)
print('✅ 多头布局与 PyTorch SDPA 对照通过')

```

## Q5: Which two problems does standard Attention still not solve?

Every query forms scores with all keys, so the score matrix grows quadratically with sequence length. On the other hand, without position encoding or positional bias, self-attention matches only by content; permuting the token order in the same way only permutes the output in the same way, and the model cannot distinguish the original order from Attention alone.


```python
def attention_score_elements(batch_size, num_heads, seq_len):
    """Return the number of elements of the standard self-attention score tensor [B, H, S, S]."""
    return batch_size * num_heads * seq_len * seq_len


short_scores = attention_score_elements(batch_size=2, num_heads=4, seq_len=8)
long_scores = attention_score_elements(batch_size=2, num_heads=4, seq_len=16)

tokens = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
permutation = torch.tensor([2, 0, 1])
output, _ = scaled_dot_product_attention(tokens, tokens, tokens)
permuted = tokens[:, permutation]
permuted_output, _ = scaled_dot_product_attention(permuted, permuted, permuted)
print('score elements：', short_scores, '->', long_scores)

```

## Q5 verification: Do doubling the length and permuting the order match the mechanism's expectations?

When the sequence length doubles, the number of score elements should become four times larger; without a positional mechanism, permuting the input only permutes the output in the same way.


```python
assert long_scores == 4 * short_scores
assert torch.allclose(permuted_output, output[:, permutation], atol=1e-6)
print('✅ Attention 二次增长与位置缺失特性通过')

```

## Related Reading

With the minimal Attention semantics mastered, continue with position, attention representation, memory, and inference-state problems separately.

- [Part 01 · 04 Attention Memory Optimization](../01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization.md)
- [Part 02 · 03 RoPE rotary position encoding](../02_PyTorch_Algorithms/03_RoPE_Tutorial.md)
- [Part 02 · 04 MHA / GQA](../02_PyTorch_Algorithms/04_Attention_MHA_GQA.md)
- [Part 02 · 22 vLLM PagedAttention](../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md)
- [LLM Architecture Evolution · attention representation and KV Cache](../topic_discussion/llm_architecture_evolution/04_attention_evolution.md)

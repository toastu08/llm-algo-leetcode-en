# 11. KV Cache and Memory Growth | KV Cache 与显存增长

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `inference optimization`, `KV Cache`, `memory growth` | **Audience:** learners who need to understand the memory cost of long-context inference

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

The attention mechanism saves keys (K) and values (V) for historical tokens; the longer the context and the more concurrent requests, the larger the key-value cache (KV Cache) occupancy. Capacity growth directly limits the number of requests that can be served.
This section establishes the ledger "layer count and KV head count → sequence length → concurrency → cache capacity". After completing it, you should be able to estimate the KV state volume of a single request and of concurrent scenarios, and determine whether the problem first comes from the representation, the physical organization, or the reuse strategy.


**Keywords:** `KV Cache`, sequence length, capacity growth

![Concept relationships in this section](../public/01_Hardware_Math_and_Systems/11_kv_cache_growth_map.svg)


---
## Prerequisite Reading

**Overview:** First master data types, Attention state, and the memory ledger, then compute how the KV Cache changes as the sequence length and concurrency increase.
- [Part 01 · 01 Data Types and Precision](./01_Data_Types_and_Precision.md)
- [Part 01 · 04 Attention and Memory Optimization](./04_Attention_Memory_Optimization.md)
---
## Q1: Why does the KV cache grow with the context length?


The KV cache essentially saves the Key and Value of historical tokens during the decoding stage. Each time a new token is generated, the model must append the K/V of this token at every layer and every KV head into the cache, so its growth is not "increasing a little occasionally" but a continuous linear growth with the context length.

More specifically, in standard autoregressive inference, the cache size can usually be written as:

$$
\text{KV Cache Bytes} \approx 2 \times L \times B \times H_{kv} \times D \times S
$$

where:
- $L$ is the number of layers
- $B$ is the batch size
- $H_{kv}$ is the number of KV heads
- $D$ is the head dim
- $S$ is the context length
- the leading $2$ means K and V are both stored

This formula tells us a key fact: **the KV cache is a "historical state cost"**. The longer the context, the more historical tokens there are, and the larger the cache; the larger the batch, the cache also scales up in sync; the more layers, the more this cost repeats at every layer.

Therefore, long contexts and high concurrency continuously raise the KV Cache capacity requirement; whether it becomes the limiting factor before compute, bandwidth, or other runtime state still needs to be combined with the model structure, concurrency, and backend measurements. Here we first clarify the growth rule that "the K/V state must be kept for every historical token".

### Q1 Mini Check: KV cache growth intuition

Double the context length and check whether the cache size also approximately doubles.

```python
def kv_cache_bytes(seq_len, num_layers, num_kv_heads, head_dim, batch_size=1, dtype_bytes=2):
    """Estimate the theoretical storage of K and V; does not include the allocator, workspace, or fragmentation."""
    values = (seq_len, num_layers, num_kv_heads, head_dim, batch_size, dtype_bytes)
    if any(value <= 0 for value in values):
        raise ValueError('序列长度、层数、头数、维度、batch 和 dtype 字节数必须为正数')
    return 2 * seq_len * num_layers * num_kv_heads * head_dim * batch_size * dtype_bytes

examples = [(1024, 32, 32, 128), (2048, 32, 32, 128), (4096, 32, 32, 128)]
for seq_len, layers, kv_heads, head_dim in examples:
    size_gb = kv_cache_bytes(seq_len, layers, kv_heads, head_dim) / 1e9
    print(f"seq_len={seq_len:4d} -> KV cache ≈ {size_gb:5.2f} GB")
```

### Order-of-Magnitude Overview

| Changed item | Impact on the KV cache | Intuition |
| --- | --- | --- |
| `seq_len` doubles | Approximately doubles | More historical tokens, so the cache grows linearly |
| `batch_size` doubles | Approximately doubles | Multiple requests share the same layer structure, but the cache is copied per sample |
| `num_layers` doubles | Approximately doubles | Each layer keeps its own copy of K/V |
| `num_kv_heads` increases | Approximately linear increase | More KV heads, larger cache |

The purpose of this table is to first fix the "growth direction" in mind, then look at MHA / MQA / GQA and paging management later.
## Q2: Why do MHA, MQA, and GQA affect KV cache pressure?


They affect the **head-count dimension of the KV cache**.

- **MHA (Multi-Head Attention)**: every query head has its own group of K/V, so the cache pressure is the largest.
- **MQA (Multi-Query Attention)**: multiple query heads share the same group of K/V, so the KV cache shrinks immediately.
- **GQA (Grouped-Query Attention)**: between MHA and MQA, grouping query heads to share K/V, making a trade-off between memory and expressive capacity.

From the cache perspective, what really determines the memory size is not how many query heads there are, but **how many groups of K/V must be stored**. So as soon as the number of KV heads decreases, the cache decreases proportionally.

This is also why many long-context models adopt MQA or GQA: they do not just "change the form of attention", but directly lower the KV cache cost during inference.

### Q2 Mini Check: Head count and memory intuition

Fix the context length and layer count, change only the number of KV heads, and observe the memory change.

```python
def kv_cache_gb(seq_len, num_layers, num_kv_heads, head_dim, batch_size=1, dtype_bytes=2):
    """Convert the theoretical KV Cache byte count to decimal GB."""
    return kv_cache_bytes(seq_len, num_layers, num_kv_heads, head_dim, batch_size, dtype_bytes) / 1e9

seq_len = 4096
num_layers = 32
head_dim = 128
for name, kv_heads in [("MHA", 32), ("GQA", 8), ("MQA", 1)]:
    print(f"{name:>3s}: kv_heads={kv_heads:2d}, KV cache ≈ {kv_cache_gb(seq_len, num_layers, kv_heads, head_dim):5.2f} GB")
```

## Q3: When the KV Cache grows large, should the organization or the representation of the cache be optimized?


First determine where the pressure comes from: if the cache ledger of a single token is too large, give priority to changing the cache representation; if the cache capacity is acceptable but reservation and fragmentation cause waste under many requests, give priority to changing the cache organization. The representation scale and page count in the table are still mechanism models; to judge the memory, latency, and throughput gains of a real backend, fixed-workload measurements are needed. A decision table is used below to separate the three kinds of problems.

| Observed problem | Object to change first | Representative mechanism | Main observation |
| --- | --- | --- | --- |
| Per-token KV cost is too large | Cache representation | MQA / GQA / MLA | Cache bytes per token, per layer |
| Obvious reservation or fragmentation under many requests | Cache organization | PagedAttention | Page count, tail-page waste, reuse |
| Repeated computation caused by duplicate prefixes | Cache reuse | Prefix Cache (extension) | Hit rate, number of reused tokens |


### Q3 Mini Check: Cache management versus representation compression

Look at "cache management" and "representation compression" separately, then judge their respective impact on memory.

```python
def paged_attention_pages(seq_len, page_size):
    """Compute the number of pages needed for paging; only observes the allocation granularity and does not simulate a real allocator."""
    if seq_len <= 0 or page_size <= 0:
        raise ValueError('seq_len 和 page_size 必须为正数')
    return (seq_len + page_size - 1) // page_size

def paged_cache_waste_tokens(seq_len, page_size):
    """Compute the unused token slots in the last page."""
    pages = paged_attention_pages(seq_len, page_size)
    return pages * page_size - seq_len

def mla_cache_bytes(seq_len, num_layers, latent_dim, batch_size=1, dtype_bytes=2):
    """Estimate the theoretical storage of the latent representation; not the complete MLA kernel ledger."""
    values = (seq_len, num_layers, latent_dim, batch_size, dtype_bytes)
    if any(value <= 0 for value in values):
        raise ValueError('序列长度、层数、latent_dim、batch 和 dtype 字节数必须为正数')
    return seq_len * num_layers * latent_dim * batch_size * dtype_bytes

def compare_cache_strategies(seq_len, page_size, base_kv_heads, head_dim, latent_dim, prefix_len=0, num_layers=32, batch_size=1, dtype_bytes=2):
    """Display the observable quantities of cache organization and representation compression side by side.

    PagedAttention focuses on page count and tail-page waste; the MLA proxy focuses on the representation size per token;
    prefix_len only records the number of reusable prefix tokens. The three are not the same metric and are not equal to the end-to-end gain of a real backend.
    """
    if prefix_len < 0 or prefix_len > seq_len:
        raise ValueError('prefix_len 必须位于 [0, seq_len]')
    base_bytes = kv_cache_bytes(seq_len, num_layers, base_kv_heads, head_dim, batch_size, dtype_bytes)
    compressed_bytes = mla_cache_bytes(seq_len, num_layers, latent_dim, batch_size, dtype_bytes)
    return {
        'management': {
            'pages': paged_attention_pages(seq_len, page_size),
            'waste_tokens': paged_cache_waste_tokens(seq_len, page_size),
        },
        'representation': {
            'base_kv_bytes': base_bytes,
            'latent_bytes_proxy': compressed_bytes,
            'toy_representation_saving_ratio': round(1 - compressed_bytes / base_bytes, 4),
        },
        'reuse': {
            'prefix_reuse_tokens': prefix_len,
        },
    }

result = compare_cache_strategies(4096, 128, 32, 128, 64, prefix_len=1024)
print('management ->', result['management'])
print('representation ->', result['representation'])
assert result['management']['pages'] == 32
assert result['management']['waste_tokens'] == 0
assert result['reuse']['prefix_reuse_tokens'] == 1024
assert 0 < result['representation']['toy_representation_saving_ratio'] < 1
try:
    paged_attention_pages(4096, 0)
except ValueError:
    print('✅ 分页参数校验通过')
else:
    raise AssertionError('page_size 为 0 时应报错')
```

---
## Related Reading

From the KV Cache formula, this section can return to the head-count design of Attention, then continue into paging management, prefix reuse, and request scheduling.

- [22. vLLM and PagedAttention](../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md)
- [24. Prefix Cache Matching and Reuse](../02_PyTorch_Algorithms/24_Prefix_Cache_Matching_and_Reuse.md)
- [34. SGLang and RadixAttention](../02_PyTorch_Algorithms/34_SGLang_RadixAttention.md)
---
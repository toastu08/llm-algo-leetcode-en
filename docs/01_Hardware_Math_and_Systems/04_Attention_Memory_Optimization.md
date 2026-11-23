# 04. Attention Memory Optimization | 注意力机制变体与显存优化

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `inference optimization`, `Attention`, `KV Cache` | **Audience:** beginners in system performance

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

The memory pressure of the attention mechanism comes from the key-value cache (KV Cache) kept for a long time, the temporary working set during computation, and the cache organization at serving runtime. Reducing the amount of state, reducing temporary reads and writes, and improving physical placement solve different problems.
This section first classifies the problem: head sharing or representation compression changes the long-lived state volume, FlashAttention changes the temporary working set and reads/writes, and PagedAttention changes the physical organization of the cache. After completing it, you should be able to distinguish which kind of cost MHA, MQA, GQA, MLA, FlashAttention, and PagedAttention each change; the long-lived KV Cache capacity ledger is expanded in section 11, and the temporary working-set model is expanded in section 14.

**Keywords:** `MHA`, `MQA`, `GQA`

![Concept relationships in this section](../public/01_Hardware_Math_and_Systems/04_attention_kv_cache_map.svg)


---

## Prerequisite Reading
**Overview:** First review data representations, the GPU memory hierarchy, and scaled dot-product attention, then compare the three kinds of memory levers.

- [Part 01 · 01 Data Types and Precision](./01_Data_Types_and_Precision.md)
- [Part 01 · 03 GPU Architecture and Memory](./03_GPU_Architecture_and_Memory.md)
- [Part 00 · 16 Attention Mechanism Intro](../00_Prerequisites/16_Attention_Mechanism_Intro.md)
  
---
## Q1: In autoregressive generation, how does the KV Cache of standard multi-head attention (MHA) create memory and bandwidth pressure?


Standard multi-head attention (MHA) keeps K and V for every layer and every historical token. The theoretical byte count can be written as:

```text
KV Cache bytes = 2 × batch × layers × seq_len × kv_heads × head_dim × dtype_bytes
```

where `2` corresponds to K and V. Taking `batch=1`, `layers=32`, `seq_len=2048`, `kv_heads=32`, `head_dim=128`, and FP16 as an example, the KV Cache for a single sequence is about 1.07 GB; when the number of concurrent sequences increases to 32, the theoretical capacity requirement also grows by about 32 times.

The cache goes through three phases: Prefill, Decode, and reclamation: Prefill establishes the initial K/V, Decode appends one token and reads the existing cache each time it generates a token, and the space is released or reclaimed after the request ends. Growth in sequence length and concurrency brings capacity pressure, and the repeated reads in Decode may also bring bandwidth pressure; whether it becomes memory bound still depends on the model, dtype, concurrency, and implementation. The mini-check in this section first records how capacity changes with sequence length; real latency, bandwidth impact, and reclamation strategies are left to the inference projects for verification.

| Variable | Meaning | Impact on the theoretical KV Cache |
| --- | --- | --- |
| `layers` | Number of Transformer layers | Approximately linear growth |
| `seq_len` | Cached context length | Approximately linear growth |
| `batch` / concurrency | Number of requests resident at the same time | Approximately linear growth |
| `kv_heads` | Number of K/V heads | Approximately linear growth; GQA / MQA changes it |
| `head_dim` | Dimension of a single head | Approximately linear growth |
| `dtype_bytes` | Bytes per element | Approximately linear growth; usually 2 for FP16 / BF16 |

### Q1 Mini Check: Observe how the KV Cache changes with sequence length

Double the context length, then check whether the cache size also approximately doubles.

```python
def kv_cache_bytes(seq_len, layers, heads, head_dim, dtype_bytes=2, batch_size=1):
    """Estimate the theoretical byte count of the KV Cache in the decode stage.

    Does not include block metadata, padding, quantization scales, allocator reserve, or
    runtime workspace; heads here should be KV heads, not query heads.
    """
    if min(seq_len, layers, heads, head_dim, dtype_bytes, batch_size) <= 0:
        raise ValueError('shape values 和 dtype_bytes 必须为正数')
    return 2 * batch_size * seq_len * layers * heads * head_dim * dtype_bytes

for seq_len in [1024, 2048, 4096]:
    size_gb = kv_cache_bytes(seq_len, 32, 32, 128) / 1e9
    print(f'seq_len={seq_len:4d} -> KV cache ≈ {size_gb:5.2f} GB')
base = kv_cache_bytes(2048, 32, 32, 128)
doubled = kv_cache_bytes(4096, 32, 32, 128)
assert doubled == 2 * base
print('seq_len doubled -> theoretical KV Cache ratio:', doubled / base)

```

## Q2: How do MQA (Multi-Query Attention) and GQA (Grouped-Query Attention) alleviate KV Cache pressure through architectural improvements?


To reduce the amount of data that needs to be moved, researchers changed the projection structure of Attention:

1. **MQA (Multi-Query Attention)**:
   - **Mechanism**: no matter how many Query heads there are, all Query heads **share only 1 Key head and 1 Value head**.
   - **Benefit**: the number of heads related to Key/Value in the KV Cache drops from `H` to `1`, so the cache size is approximately reduced to `1/H` of MHA while also lowering the potential KV read volume; whether the end-to-end speed improves still depends on the kernel, backend, and workload.
   - **Cost**: sharing K/V heads changes the model's expressive capacity and may affect quality on some tasks; the degree of impact depends on the model structure, training method, and the specific checkpoint.

2. **GQA (Grouped-Query Attention)**:
   - **Mechanism**: a compromise. The Query heads are grouped (for example, 32 Query heads split into 8 groups), and the Query heads within each group share 1 pair of Key/Value heads.
   - **Benefit**: the number of heads related to Key/Value in the KV Cache drops from `H` to `G`, so with other conditions equal, the cache size is approximately reduced to `G/H` of MHA. For example, when `H = 32, G = 8`, the KV Cache is about `1/4` of MHA, and the potential read volume also decreases.

The table below compares the three structures under the same accounting. The ratios in the table are theoretical capacity ratios; actual speed, quality, and backend gains still need to be verified on a fixed workload.

| Mechanism | Query heads | KV heads | Theoretical KV Cache ratio | Main change | Main cost |
| --- | ---: | ---: | ---: | --- | --- |
| MHA | H | H | 1 | Each Query head has its own K/V head | Largest KV Cache |
| GQA | H | G | G/H | Multiple Query heads share a group of K/V heads | Requires a matching model structure or checkpoint |
| MQA | H | 1 | 1/H | All Query heads share one group of K/V heads | Expressive capacity and task quality may be affected |

### Q2 Mini Check: Why does a change in the head count directly affect the cache

Fix the context length and change only the number of KV heads to see how the cache shrinks.

```python
def kv_cache_gb(seq_len, layers, kv_heads, head_dim, dtype_bytes=2, batch_size=1):
    """Return the KV Cache in decimal GB, for comparing different KV head counts."""
    return kv_cache_bytes(seq_len, layers, kv_heads, head_dim, dtype_bytes, batch_size) / 1e9

seq_len = 4096
layers = 32
head_dim = 128
mha_gb = kv_cache_gb(seq_len, layers, 32, head_dim)
for name, kv_heads in [('MHA', 32), ('GQA', 8), ('MQA', 1)]:
    current_gb = kv_cache_gb(seq_len, layers, kv_heads, head_dim)
    print(f'{name:>3s}: kv_heads={kv_heads:2d}, KV cache ≈ {current_gb:5.2f} GB, 压缩比例={current_gb / mha_gb:.3f}')
ratios = {name: kv_cache_gb(seq_len, layers, heads, head_dim) / mha_gb for name, heads in [('MHA', 32), ('GQA', 8), ('MQA', 1)]}
assert ratios == {'MHA': 1.0, 'GQA': 0.25, 'MQA': 1 / 32}
print('head-sharing ratios are theoretical cache ratios; speed and quality require a real workload')

```

## Q3: Why can MLA compress the KV Cache through a latent representation?


MLA (Multi-Head Latent Attention) is a class of latent attention design represented by DeepSeek-V2/V3, whose goal is to compress the KV Cache representation while preserving effective attention information. The specific cache format, RoPE handling, and gains depend on the model implementation.

**Mechanism and principle**:
1. **Low-rank compression**: the model does not directly cache the complete K and V, but stores a more compact latent representation.
2. **Dynamic recovery**: when computing Attention, the representation needed for the dot product is recovered through a projection path, reducing the amount of long-lived data.
3. **RoPE decoupling**: positional and content information are handled through different paths, and the specific cache entries depend on the implementation.

MLA trades additional representation transformation and recovery computation for a smaller cache, and a fixed speed or quality gain should not be derived directly. The ratio model here does not use the latent dimension of a specific DeepSeek checkpoint; the result can only illustrate the "compression-recovery computation" relationship and cannot predict the memory of a real model. Whether it is worthwhile must be verified together with the latent dimension, RoPE handling, backend, and a fixed workload.

### Q3 Mini Check: The trade-off between compression ratio and extra computation

Change the compression ratio of the latent representation and compare the cache unit count and the reconstruction proxy volume; the result only expresses the mechanism's order of magnitude and is not equal to the memory, latency, or quality gain of a real model.

```python
def mla_gain(seq_len, kv_heads, compression_ratio=0.5):
    """Use a ratio model to show the order-of-magnitude change from MLA latent representation compression.

    This is not the complete cache formula of MLA, nor does it represent actual quality, throughput, or kernel FLOPs.
    reconstruction_units_proxy is only a teaching proxy for the extra recovery path.
    """
    if seq_len <= 0 or kv_heads <= 0 or not 0 < compression_ratio <= 1:
        raise ValueError('seq_len、kv_heads 必须为正数，compression_ratio 必须在 (0, 1]')
    base = 2 * seq_len * kv_heads
    compressed = base * compression_ratio
    reconstruction_units = compressed * 2
    return {
        'base_units': base,
        'compressed_units': round(compressed, 2),
        'saving_ratio': round(1 - compression_ratio, 2),
        'reconstruction_units_proxy': round(reconstruction_units, 2),
        'evidence_level': 'teaching_proxy',
    }

for case in [(1024, 32, 0.5), (4096, 32, 0.25), (4096, 16, 0.25)]:
    print(case, '->', mla_gain(*case))
assert mla_gain(4096, 32, 0.25)['saving_ratio'] == 0.75
print('reconstruction_units_proxy 只是额外恢复路径的教学代理，不是 MLA kernel FLOPs')

```

## Q4: How does PagedAttention use paging management to reduce wasted KV Cache space?


Besides changing the model architecture, cache management at the system level is also important. Early inference implementations often reserved contiguous space for each request, but the generation length cannot be known in advance, so mixing long and short requests easily causes reservation waste and difficulties in space organization.

**Mechanism and principle**:
1. **Fixed-size blocks**: split the KV Cache into fixed-size blocks by token.
2. **Non-contiguous storage**: blocks in physical memory are not required to be contiguous; a block table maps logical tokens to physical positions.
3. **Growth on demand**: request more blocks when a request needs more tokens, reducing the waste caused by reserving for the maximum length.

**Benefit**:
PagedAttention changes the allocation and addressing of the cache, not the KV representation itself; the actual concurrency benefit also depends on the block size, scheduling, kernel, prefix sharing, and workspace.

Two kinds of system mechanisms need to be distinguished: PagedAttention mainly addresses the allocation, addressing, and reclamation of physical blocks; Prefix Cache / RadixAttention mainly addresses shared-prefix reuse across different requests. The former answers "where the cache is placed", and the latter answers "whether an existing prefix can be reused".

| Mechanism | Main problem solved | Observation metric |
| --- | --- | --- |
| PagedAttention | Block allocation, addressing, and reclamation within a request | Space utilization, fragmentation, number of requests that fit |
| Prefix Cache / RadixAttention | Shared-prefix reuse across requests | Hit rate, number of reused tokens, prefill savings |

### Q4 Mini Check: Memory utilization after paging

Compare contiguous reservation with the slots rounded up by block; the result only observes space utilization and is not equal to the real block allocator or serving throughput of vLLM.

```python
def contiguous_reserved_tokens(lengths, max_len):
    """Simulate reserving max_len contiguous token slots per request."""
    if not lengths or max_len <= 0:
        raise ValueError('lengths 不能为空，max_len 必须为正数')
    if any(length <= 0 or length > max_len for length in lengths):
        raise ValueError('每个请求长度必须在 (0, max_len] 内')
    return len(lengths) * max_len


def paged_reserved_tokens(lengths, block_size):
    """Simulate the paged token slots after rounding up to a fixed block_size."""
    if not lengths or block_size <= 0:
        raise ValueError('lengths 不能为空，block_size 必须为正数')
    if any(length <= 0 for length in lengths):
        raise ValueError('请求长度必须为正数')
    return sum(((length + block_size - 1) // block_size) * block_size for length in lengths)

lengths = [64, 96, 128, 320, 512]
max_len = max(lengths)
block_size = 128
contiguous = contiguous_reserved_tokens(lengths, max_len)
paged = paged_reserved_tokens(lengths, block_size)
utilization = sum(lengths) / paged
waste = 1 - utilization

print(f'Contiguous reservation: {contiguous} tokens')
print(f'Paged reservation: {paged} tokens')
print(f'Paged utilization: {utilization:.1%}')
print(f'Paged waste: {waste:.1%}')
assert paged <= contiguous
assert 0 < utilization <= 1

```

---

## Related Reading
After this section, continue with KV capacity, the Attention working set, and head-count representation mechanisms respectively.
- [Part 01 · 11 KV Cache and Memory Growth](./11_KV_Cache_and_Memory_Growth.md)
- [Part 01 · 14 FlashAttention Memory Model](./14_FlashAttention_Memory_Model.md)
- [Part 02 · 04 Attention (MHA / MQA / GQA)](../02_PyTorch_Algorithms/04_Attention_MHA_GQA.md)

---
# 14. FlashAttention Memory Model | FlashAttention 显存模型

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `inference optimization`, `FlashAttention`, `memory model` | **Audience:** learners who need to understand the memory-access cost of Attention

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

Standard attention may materialize an intermediate result close to $N \times N$; as the sequence grows longer, the temporary working set and memory reads and writes grow quickly. FlashAttention avoids fully saving these intermediate results through tiling and online softmax.

This section establishes the relationship "tile size → on-chip working set → memory reads and writes → capacity and time". After completing it, you should be able to estimate the I/O change brought by tiling and distinguish the temporary working set from the long-lived KV Cache.


**Keywords:** `FlashAttention`, `tiling`, `SRAM`

![Concept relationships in this section](../public/01_Hardware_Math_and_Systems/14_flashattention_memory_flow.svg)


---
## Prerequisite Reading

**Overview:** First understand the score, softmax, and output computation of Attention, then observe how FlashAttention uses a tiled working set to reduce the materialization of intermediate results, and judge the memory pressure and data movement cost accordingly.

- [Part 01 · 03 GPU Architecture and Memory](./03_GPU_Architecture_and_Memory.md)
- [Part 01 · 04 Attention and Memory Optimization](./04_Attention_Memory_Optimization.md)
---
## Q1: Why does standard Attention quickly become a memory bottleneck on long sequences?


The problem with standard Attention is not that "it cannot compute", but that the intermediate result is too large.

In long-sequence scenarios, $QK^T$ generates an attention matrix close to $N \times N$. If this matrix has to be written back to HBM frequently and then read out again for Softmax and subsequent multiplication, the number of memory accesses becomes very large.

This means two things:
- The memory occupied by the intermediate result expands quickly;
- Data movement more easily becomes the bottleneck than the computation itself.

So what FlashAttention has to solve is not "make the matrix smaller", but "do not let the large matrix land in HBM for a long time".

### Q1 Mini Check: Compute the size of the intermediate matrix for a sequence length

Compute the number of elements and bytes of the score matrix from the sequence length, then judge the capacity and transfer cost after it is written to HBM.

```python
def attention_score_bytes(seq_len, batch_size=1, num_heads=1, dtype_bytes=2):
    """Estimate the theoretical size of the materialized score tensor; does not represent real HBM traffic."""
    values = (seq_len, batch_size, num_heads, dtype_bytes)
    if any(value <= 0 for value in values):
        raise ValueError('序列长度、batch、头数和 dtype 字节数必须为正数')
    return batch_size * num_heads * seq_len * seq_len * dtype_bytes

for n in [1024, 2048, 4096]:
    naive_gb = attention_score_bytes(n) / 1024 / 1024 / 1024
    print(f'seq_len={n:4d} -> score matrix ≈ {naive_gb:6.2f} GB')

```

## Q2: Why does FlashAttention use tiling and online softmax?


The change in FlashAttention can be split into two actions: use tiling to control the working set processed at one time, and use online softmax to complete the reduction during tiling. The sizes and working sets in the table are theoretical models and do not represent the shared memory, registers, or CUDA workspace of a complete kernel; below, standard Attention and the tiled working set are placed on the same footing.

| Comparison | Standard Attention | FlashAttention working-set model |
| --- | --- | --- |
| score result | A complete matrix close to $N \times N$ | A single score tile |
| softmax | Processed uniformly over the complete matrix | Local statistics updated per block |
| Intermediate result | May be written back to HBM for a long time | Kept in the on-chip working set as much as possible |
| Main parameter | Sequence length determines the matrix size | tile size, head dim, dtype |
| Conclusion type | Theoretical materialization size | Theoretical working set, not a kernel measurement |

### Q2 Mini Check: Why it is more stable after tiling

Split a large matrix into multiple small blocks, then distinguish the 1D tile count, the 2D score tile count, and the theoretical size of a single score tile. The values here are a working-set model, not the shared memory or register usage of a complete kernel.

```python
def num_1d_tiles(seq_len, tile_size):
    """Return the number of tiles along one sequence dimension; does not represent the 2D tile total of Attention."""
    if seq_len <= 0 or tile_size <= 0:
        raise ValueError('seq_len 和 tile_size 必须为正数')
    return (seq_len + tile_size - 1) // tile_size

def num_score_tiles(seq_len, tile_size):
    """Return the number of 2D score tiles formed by combining Q tiles and K tiles."""
    tiles_1d = num_1d_tiles(seq_len, tile_size)
    return tiles_1d * tiles_1d

def score_tile_bytes(tile_size, dtype_bytes=2):
    """Estimate only one score tile; does not represent the complete FlashAttention working set."""
    if tile_size <= 0 or dtype_bytes <= 0:
        raise ValueError('tile_size 和 dtype_bytes 必须为正数')
    return tile_size * tile_size * dtype_bytes

def flashattention_working_set_bytes(tile_size, head_dim, dtype_bytes=2):
    """Estimate the teaching working set of Q/K/V, score, and output blocks.

    This is a block-level capacity model, not equal to the shared memory, registers,
    or CUDA workspace usage of a complete kernel.
    """
    if tile_size <= 0 or head_dim <= 0 or dtype_bytes <= 0:
        raise ValueError('tile_size、head_dim 和 dtype_bytes 必须为正数')
    qkv = 3 * tile_size * head_dim
    score = tile_size * tile_size
    output = tile_size * head_dim
    return (qkv + score + output) * dtype_bytes


seq_len = 4096
for tile in [64, 128, 256]:
    tiles_1d = num_1d_tiles(seq_len, tile)
    score_tiles = num_score_tiles(seq_len, tile)
    score_tile_kb = score_tile_bytes(tile) / 1024
    working_set_kb = flashattention_working_set_bytes(tile, head_dim=128) / 1024
    print(f'tile={tile:3d} -> 1D tiles={tiles_1d:3d}, score tiles={score_tiles:4d}, score tile ≈ {score_tile_kb:6.1f} KB, working set ≈ {working_set_kb:6.1f} KB')

assert flashattention_working_set_bytes(128, 128) > score_tile_bytes(128)
try:
    flashattention_working_set_bytes(0, 128)
except ValueError:
    print('✅ 工作集参数校验通过')
else:
    raise AssertionError('tile_size 为 0 时应报错')

```

## Q3: How do you judge the benefit of FlashAttention from the working set, tile size, and data movement?


HBM has a large capacity but high access cost, while SRAM has a small capacity but is closer to the computation. The tile size must trade off between the single-block working set and the number of tiles: a smaller tile has lower on-chip usage but more scheduling passes; a larger tile reduces the number of tiles but increases on-chip storage pressure.

| tile size | Single score tile / working set | Number of tiles | Main trade-off |
| ---: | --- | ---: | --- |
| Small | Smaller | More | Low on-chip pressure, but loop and scheduling overhead may increase |
| Medium | Medium | Medium | Better balance between working set and scheduling cost |
| Large | Larger | Fewer | Fewer scheduling passes, but higher on-chip storage pressure |

The IO-aware value of FlashAttention lies in reducing the HBM round trips of large-scale intermediate results; the specific gain still depends on the sequence length, dtype, hardware, and kernel implementation, and an end-to-end speedup cannot be inferred from the theoretical working set alone.

### Q3 Mini Check: Materialized matrix versus tiled working set

Change the sequence length and tile size, compare the order of magnitude of the materialized score matrix and the tiled working set, then judge whether data movement may become the main cost.

```python
def tile_tradeoff(seq_len, tile_size, head_dim=128, dtype_bytes=2):
    """Return the tile count, single score tile, and working set, for comparing tile sizes."""
    return {
        'tile_count': num_score_tiles(seq_len, tile_size),
        'score_tile_bytes': score_tile_bytes(tile_size, dtype_bytes),
        'working_set_bytes': flashattention_working_set_bytes(tile_size, head_dim, dtype_bytes),
    }

def score_materialization_ratio(seq_len, tile_size, dtype_bytes=2):
    """Compare the theoretical storage size of the complete score matrix with a single score tile.

    This is not actual HBM traffic, a kernel speedup, or an end-to-end performance metric.
    """
    if seq_len <= 0 or tile_size <= 0 or dtype_bytes <= 0:
        raise ValueError('seq_len、tile_size 和 dtype_bytes 必须为正数')
    full = attention_score_bytes(seq_len, dtype_bytes=dtype_bytes)
    tile = score_tile_bytes(tile_size, dtype_bytes)
    return full / tile

tile_rows = {}
for tile in [64, 128, 256]:
    tile_rows[tile] = tile_tradeoff(4096, tile)
    print(f'tile={tile:3d} -> {tile_rows[tile]}, materialization ratio ≈ {score_materialization_ratio(4096, tile):.0f}x')
print('smaller tile => smaller working set, but more tiles and more scheduling work')
assert score_materialization_ratio(4096, 64) > score_materialization_ratio(4096, 256)

```

---
## Related Reading

From the IO-aware Attention paper, this section can return to working-set simulation, then continue into Triton implementation and inference-side prefill verification.

- [FlashAttention paper](https://arxiv.org/abs/2205.14135)
- [20. FlashAttention Simulation](../02_PyTorch_Algorithms/20_FlashAttention_Sim.md)
- [11. Triton Flash Attention forward operator](../03_Triton_Kernels/11_Triton_Flash_Attention.md)
---
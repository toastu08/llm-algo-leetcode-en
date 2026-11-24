# 16. Warp Block SharedMemory Basics | Warp、Block 与 Shared Memory 基础

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `operator programming`, `CUDA`, `Shared Memory` | **Audience:** learners starting with compilers and systems

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/16_Warp_Block_SharedMemory_Basics.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

When the same data is used repeatedly by multiple threads inside a Block, reading it from global memory again and again produces unnecessary external traffic. Shared memory keeps this data inside the Block, but it is constrained at the same time by capacity and by synchronization correctness.

This section computes the thread-slot coverage ratio within a single block and the logical read byte volume before and after reuse, then checks capacity, synchronization, and reuse gain separately. It mainly supports the operator optimization route, and also provides the memory-access foundation for Triton Tiles, fusion, and on-chip reductions.

**Keywords:** `CUDA`, `Warp`, `Shared Memory`

![Shared memory reuse chain within a Block](../public/01_Hardware_Math_and_Systems/16_block_shared_memory_reuse.svg)

---
## Prerequisite Reading

First distinguish the thread block defined by the program from the SM that carries it on the device, then observe the data reuse inside the block.

- [15 CUDA Execution Model](./15_CUDA_Execution_Model.md): grid coverage, Warp grouping, and per-block resource limits.


## Q1: What are Warp, Block, and Thread each responsible for?

A Thread handles the elements assigned to it, the threads of the same Block share a collaboration scope, and the hardware groups 32 consecutive threads within a block into an NVIDIA Warp. Threads in a Warp may temporarily not participate in the same instruction because of differing conditions; you cannot understand "belonging to the same Warp" as all memory accesses being naturally synchronized, nor treat the Warp count as the number of physical cores.

For example, when a Block has 48 threads, it occupies the thread slots of two Warps, and the second Warp has only 16 valid threads. Two such Blocks are still each split into two Warps, and the remaining threads are not stitched into a third full Warp. What is computed below is the thread-slot coverage ratio, not the measured occupancy of some SM.

Shared memory is used to reuse data among threads of the same block; in the ordinary thread-block model of this section, other Blocks do not directly use this copy. Between the producer finishing the write, the consumer reading, and the next round overwriting, the synchronization and visibility conditions must still be satisfied. Device generations change the available resources and supported mechanisms, but they do not turn all threads of one launch into one shared-memory collaboration group.


### Q1 coverage calculation: How the threads of a single Block are packed into Warps

The thread count here belongs to the same Block. Tail threads of different Blocks are not merged into one Warp; the thread-slot coverage ratio is also not device occupancy.



```python
def execution_coverage(num_threads, warp_size=32):
    """Compute the Warp count, the tail-Warp thread count, and the slot coverage ratio of a single Block."""
    if any(type(value) is not int or value <= 0 for value in (num_threads, warp_size)):
        raise ValueError('单块线程数与 Warp 大小必须是正整数')
    warps = (num_threads + warp_size - 1) // warp_size
    thread_slots = warps * warp_size
    last_warp_active_threads = num_threads - (warps - 1) * warp_size
    return {
        'threads': num_threads,
        'warps': warps,
        'last_warp_active_threads': last_warp_active_threads,
        'thread_slot_utilization': num_threads / thread_slots,
    }

for threads in [32, 48, 65]:
    print(execution_coverage(threads))
assert execution_coverage(48)['last_warp_active_threads'] == 16


```

## Q2: How does shared memory change repeated reads?

When the same Block uses one piece of data multiple times, you can move it from global memory into shared memory and let the threads read the on-chip copy repeatedly. What this reduces is the repeated global read requests; the actual HBM byte volume also depends on the cache hits of the original scheme. The load-in, the synchronization, and the on-chip capacity are new costs and must be recorded separately.

Below we compare the logical byte volume of a full re-read each time versus loading once per block, and we do not treat that ratio as a speedup.


### Q2 comparison check: Compute the logical read reduction of whole-block reuse

Cache hits are ignored for now; assume each Block loads the data block independently and uses it in full. Only the use count changes, to observe the logical global read volume; the real HBM traffic is not measured here.



```python
def shared_reuse_gain(tile_bytes, reuse_count, blocks):
    """Return the logical byte volume of the whole-block read model and the reduction ratio; does not measure HBM."""
    if any(type(value) is not int or value <= 0 for value in (tile_bytes, reuse_count, blocks)):
        raise ValueError('数据块字节数、使用次数与块数必须是正整数')
    baseline_global_read_bytes = tile_bytes * reuse_count * blocks
    shared_global_read_bytes = tile_bytes * blocks
    return {
        'baseline_global_read_bytes': baseline_global_read_bytes,
        'shared_global_read_bytes': shared_global_read_bytes,
        'saved_global_read_bytes': baseline_global_read_bytes - shared_global_read_bytes,
        'global_read_reduction_fraction': 1 - shared_global_read_bytes / baseline_global_read_bytes,
    }

for reuse_count in [1, 2, 4]:
    print('reuse', reuse_count, '->', shared_reuse_gain(8192, reuse_count, 8))
assert shared_reuse_gain(8192, 1, 8)['saved_global_read_bytes'] == 0


```

## Q3: Once the capacity and synchronization checks pass, do you already have a reuse gain?

A shared memory scheme first needs to hold the data block and let consumers read after the data is written. Both checks passing only shows that the capacity and synchronization preconditions of this example are satisfied; if the data is used only once, the scheme can still execute correctly but does not reduce the logical global reads. The synchronization correctness of an actual kernel requires checking the control flow and the access relationships; it cannot be replaced by one boolean input.


### Q3 comparison check: Check the execution preconditions and the reuse volume separately

Fix the block count and the budget, and construct the cases of exceeding capacity, unsatisfied synchronization conditions, and no repeated use separately.



```python
def shared_memory_candidate(tile_bytes, reuse_count, blocks,
                            shared_limit_bytes, synchronization_valid):
    """Check the given capacity and synchronization preconditions; the synchronization input is a known assumption, not a verification result."""
    traffic = shared_reuse_gain(tile_bytes, reuse_count, blocks)
    if type(shared_limit_bytes) is not int or shared_limit_bytes <= 0:
        raise ValueError('共享内存预算必须是正整数字节数')
    if type(synchronization_valid) is not bool:
        raise ValueError('同步前提必须是布尔值')
    reasons = []
    if tile_bytes > shared_limit_bytes:
        reasons.append('shared_capacity')
    if not synchronization_valid:
        reasons.append('synchronization')
    return {
        'capacity_and_sync_conditions_met': not reasons,
        'reasons': reasons,
        'has_logical_read_reduction': traffic['saved_global_read_bytes'] > 0,
        **traffic,
    }


good = shared_memory_candidate(32 * 1024, 4, 8, 64 * 1024, True)
too_large = shared_memory_candidate(96 * 1024, 4, 8, 64 * 1024, True)
no_reuse = shared_memory_candidate(32 * 1024, 1, 8, 64 * 1024, True)
assert good['capacity_and_sync_conditions_met']
assert not too_large['capacity_and_sync_conditions_met']
assert no_reuse['capacity_and_sync_conditions_met']
assert not no_reuse['has_logical_read_reduction']
assert shared_memory_candidate(1024, 4, 8, 65536, False)['reasons'] == ['synchronization']
print(good)
print(too_large)
print(no_reuse)


```

### From logical reuse to an actual kernel

First judge whether the thread collaboration and synchronization are correct, then check the shared memory capacity and the logical read reduction. When continuing to enlarge the tiling, you also need to verify it together with registers, the number of resident blocks, and the compilation results; these resource relationships can be computed further in [SRAM Optimization Techniques](./24_SRAM_Optimization_Techniques.md).


## Related Reading

First connect in-block collaboration to an actual shared memory matrix multiplication, then continue computing the trade-off between resources and residency.

- [Part04·06 Shared Memory GEMM](../04_CUDA_and_System_Optimization/06_CUDA_Shared_Memory_Optimization.md): collaborative loading, reading, and synchronization before overwriting.
- [24 SRAM Optimization Techniques](./24_SRAM_Optimization_Techniques.md): shared memory, registers, and the number of resident blocks.


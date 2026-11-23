# 24. SRAM Optimization Techniques | SRAM 优化技术

**Difficulty:** Hard | **Environment:** CPU-first | **Tags:** `hardware systems`, `SRAM`, `memory-access optimization` | **Audience:** learners who need to understand the cost of on-chip GPU memory access

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

On-chip shared memory lets one data transfer be reused by multiple threads or multiple computation steps, but it also introduces constraints on capacity, synchronization, access conflicts, and register usage. More reuse does not automatically equal an end-to-end speedup.
This section establishes the relationship "matrix tiling → shared-memory reuse → synchronization and conflicts → resource admission". After completing it, you should be able to estimate the reduced memory traffic and judge whether a tiling design satisfies the on-chip resource limits.

**Keywords:** shared memory, matrix tiling, bank conflicts

![Concept relationships in this section](../public/01_Hardware_Math_and_Systems/24_sram_optimization_chain.svg)



## Prerequisite Reading

**Overview:** First understand how shared memory reduces repeated transfers through data reuse, then observe how the access layout, synchronization, and register usage affect the actual benefit of a tile.

- [15. CUDA Execution Model](./15_CUDA_Execution_Model.md)
- [16. Warp Block SharedMemory Basics](./16_Warp_Block_SharedMemory_Basics.md)
- [23. TensorCore Deep Dive](./23_TensorCore_Deep_Dive.md)

## Q1: How many global reads can shared-memory reuse eliminate?

When the same data block is used multiple times, it can either be read from global memory each time, or be moved into shared memory first and then reused by threads. First assume that each global read must transfer the whole data block, and ignore cache hits for now: the data block size multiplied by the number of uses is the number of read bytes in the non-reused scheme; the shared-memory scheme transfers it once and adds on-chip writes, reads, and synchronization.

The byte counts and synchronization counts of each kind are recorded separately below. The reduction in global reads is a result of this access model, and actual HBM traffic is also affected by the cache; byte counts and synchronization counts cannot be subtracted directly to obtain elapsed time.

### Q1 Mini Check: Fix the data block, change only the number of uses

First predict the global read volume for one use and multiple uses, then check the increment in shared-memory writes, reads, and synchronization.

```python
def reuse_traffic_ledger(tile_bytes, uses, sync_points=1):
    """Compute the byte counts of the whole-block repeated-read model, excluding cache and elapsed-time estimates."""
    for name, value in {'tile_bytes': tile_bytes, 'uses': uses}.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} 必须是正整数')
    if type(sync_points) is not int or sync_points < 0:
        raise ValueError('sync_points 必须是非负整数')
    return {
        'global_read_bytes_without_reuse': tile_bytes * uses,
        'global_read_bytes_with_reuse': tile_bytes,
        'global_read_bytes_saved': tile_bytes * (uses - 1),
        'shared_write_bytes': tile_bytes,
        'shared_read_bytes': tile_bytes * uses,
        'sync_points': sync_points,
    }


TILE_BYTES = 4096  # Editable: the size of the data block fully used each time, in bytes.
for uses in [1, 2, 4]:
    print(uses, '次使用 ->', reuse_traffic_ledger(TILE_BYTES, uses))
assert reuse_traffic_ledger(4096, 1)['global_read_bytes_saved'] == 0
assert reuse_traffic_ledger(4096, 4)['global_read_bytes_saved'] == 12288
assert reuse_traffic_ledger(4096, 4, 3)['sync_points'] == 3
try:
    reuse_traffic_ledger(4096, 0)
except ValueError:
    pass
else:
    raise AssertionError('使用次数为零时应报错')

```

## Q2: Why do bank conflicts serialize accesses that were originally parallel?


A bank conflict occurs when multiple threads access different addresses at the same time but they fall into the same bank; the hardware must serve these requests in multiple rounds, so accesses that were originally parallel may degrade into serial ones. Judging conflicts requires looking not only at the number of banks but also at the stride, padding, and alignment; broadcast reads of the same address cannot simply be treated as ordinary conflicts.

### Q2 Mini Check: Why the access pattern matters

Let the threads of a warp access shared memory with different strides, padding, or broadcast patterns, and observe which banks they fall into and the most severe conflict degree.

```python
def bank_access_report(stride, banks=32, threads=32, element_words=1, padding_words=0, access_mode='strided'):
    """Report the bank mapping of one warp without converting conflicts into invented elapsed time.

    padding_words simulates changing the row stride; broadcast distinguishes reads of the same address
    from different addresses falling into the same bank. Real conflicts are also affected by the data type,
    broadcast rules, and the specific access instruction, so only the mapping mechanism is verified here.
    """
    if stride <= 0 or banks <= 0 or threads <= 0 or element_words <= 0 or padding_words < 0:
        raise ValueError('stride、banks、threads 和 element_words 必须为正数，padding_words 不能为负数')
    if access_mode not in {'strided', 'broadcast'}:
        raise ValueError("access_mode 必须是 'strided' 或 'broadcast'")
    effective_stride = stride + padding_words
    bank_ids = ([0] * threads if access_mode == 'broadcast' else
                [(lane * effective_stride * element_words) % banks for lane in range(threads)])
    counts = {bank: bank_ids.count(bank) for bank in sorted(set(bank_ids))}
    max_conflict = max(counts.values())
    return {
        'bank_ids': bank_ids,
        'unique_banks': len(counts),
        'effective_stride': effective_stride,
        'access_mode': access_mode,
        'max_conflict_degree': max_conflict,
        'serialized_risk': access_mode == 'strided' and max_conflict > 1,
    }

for stride in [1, 2, 16, 32, 33]:
    report = bank_access_report(stride)
    print(stride, '->', {k: report[k] for k in ['unique_banks', 'max_conflict_degree', 'serialized_risk']})
assert bank_access_report(1)['max_conflict_degree'] == 1
assert bank_access_report(32)['max_conflict_degree'] == 32
assert bank_access_report(2)['max_conflict_degree'] == 2
assert bank_access_report(32, padding_words=1)['max_conflict_degree'] == 1
assert bank_access_report(32, access_mode='broadcast')['serialized_risk'] is False
try:
    bank_access_report(0)
except ValueError:
    print('✅ bank 映射参数校验通过')
else:
    raise AssertionError('stride 为 0 时应报错')

```

## Q3: How do tiling, padding, and buffering limit the number of resident blocks?

A thread block of a matrix multiplication can hold the $M\times K$ sub-block of the left matrix and the $K\times N$ sub-block of the right matrix at the same time. Let each element occupy $b$ bytes, let each row of the right matrix be additionally padded with $p$ elements, and let the number of buffer copies be $q$; the shared-memory usage is:

$$
S_{\mathrm{block}}=q\,b\,[MK+K(N+p)].
$$

Here $M,N,K$ are the tile dimensions, $p$ is used to change the row stride of the right matrix, and $q=2$ means reserving double buffering for two groups of data. Double buffering is only a capacity condition; whether computation and transfer actually overlap still depends on the execution schedule. The code uses an assumed per-SM budget to compute the shared-memory, thread-count, and block-count limits separately, and takes the minimum as the upper bound on resident blocks; this upper bound does not yet include limits such as registers and allocation granularity.

| Design item | Computable change | What must be checked at the same time |
|:---|:---|:---|
| Tile dimensions | Element count and shared-memory bytes of the left and right sub-blocks | A larger working set may reduce the number of resident blocks |
| Row padding | Capacity increased by $Kp$ elements | Whether the bank mapping improves, see the previous question |
| Buffer copies | Shared-memory usage grows with the number of copies | Whether the schedule actually overlaps transfer and computation |
| Threads per block | Resident blocks allowed by the thread budget | Take the minimum of multiple resource upper bounds |

### Q3 Mini Check: Compare single buffering, double buffering, and row padding

Fix the tile dimensions, change the number of buffer copies or the padding amount respectively, and check the shared-memory bytes and the resident-block upper bound.

```python
def tile_resource_ledger(tile_m, tile_n, tile_k, element_bytes=4, buffers=1,
                         padding=0, threads_per_block=256,
                         smem_per_sm_bytes=65536, max_threads_per_sm=2048,
                         max_blocks_per_sm=16):
    """Estimate the residency upper bound under the given budgets; the budgets are teaching inputs and do not represent a specific GPU."""
    positive = {
        'tile_m': tile_m, 'tile_n': tile_n, 'tile_k': tile_k,
        'element_bytes': element_bytes, 'buffers': buffers,
        'threads_per_block': threads_per_block,
        'smem_per_sm_bytes': smem_per_sm_bytes,
        'max_threads_per_sm': max_threads_per_sm,
        'max_blocks_per_sm': max_blocks_per_sm,
    }
    if any(type(value) is not int or value <= 0 for value in positive.values()):
        raise ValueError('维度、字节数、份数与资源预算必须是正整数')
    if type(padding) is not int or padding < 0:
        raise ValueError('padding 必须是非负整数')
    shared_bytes = buffers * element_bytes * (
        tile_m * tile_k + tile_k * (tile_n + padding)
    )
    limits = {
        'shared_memory': smem_per_sm_bytes // shared_bytes,
        'threads': max_threads_per_sm // threads_per_block,
        'blocks': max_blocks_per_sm,
    }
    resident_blocks = min(limits.values())
    return {
        'shared_bytes_per_block': shared_bytes,
        'padding_bytes_per_block': buffers * element_bytes * tile_k * padding,
        'resident_block_limits': limits,
        'resident_blocks_upper_bound': resident_blocks,
        'active_thread_fraction_upper_bound':
            resident_blocks * threads_per_block / max_threads_per_sm,
    }


for buffers, padding in [(1, 0), (2, 0), (1, 1)]:
    print(buffers, padding, '->', tile_resource_ledger(
        32, 32, 32, buffers=buffers, padding=padding
    ))
single = tile_resource_ledger(32, 32, 32)
double = tile_resource_ledger(32, 32, 32, buffers=2)
padded = tile_resource_ledger(32, 32, 32, padding=1)
assert single['shared_bytes_per_block'] == 8192
assert double['shared_bytes_per_block'] == 16384
assert single['resident_blocks_upper_bound'] == 8
assert double['resident_blocks_upper_bound'] == 4
assert padded['padding_bytes_per_block'] == 128
assert padded['resident_blocks_upper_bound'] == 7
assert tile_resource_ledger(128, 128, 128)['resident_blocks_upper_bound'] == 0

```

## Q4: After register demand increases, which resource limits appear first?

As tiling and temporary values increase, the per-thread register demand may rise. Under a given per-thread budget, first record the amount by which the demand is exceeded; then multiply the per-thread demand by the number of threads per block to estimate how many registers one block needs. Dividing the total registers per SM by the per-block demand gives another upper bound on resident blocks; it constrains residency together with the shared-memory and thread upper bounds from the previous question.

Below it is assumed that the input demand is the per-thread allocation, without considering register allocation granularity for now. Exceeding the set budget means the compilation result needs to be checked, and does not mean the same number of overflow reads and writes has already occurred: the compiler may shorten variable lifetimes, recompute, or store values in local memory. The actual number of spilled bytes and the elapsed time must be seen in the compilation report and execution metrics.

### Q4 Mini Check: Fix the thread count, increase only the register demand

Observe the total registers per block, the budget excess, and the resident-block upper bound, then take the minimum with the shared-memory upper bound from the previous question.

```python
def register_resource_ledger(registers_per_thread, threads_per_block=256,
                             per_thread_budget=64, registers_per_sm=65536,
                             max_threads_per_sm=2048, max_blocks_per_sm=16):
    """Compute the register constraint from the assumed allocation without inferring actual spills or speedup."""
    values = (registers_per_thread, threads_per_block, per_thread_budget,
              registers_per_sm, max_threads_per_sm, max_blocks_per_sm)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError('寄存器数量、线程数与预算必须是正整数')
    registers_per_block = registers_per_thread * threads_per_block
    return {
        'registers_per_block': registers_per_block,
        'excess_registers_per_thread': max(
            registers_per_thread - per_thread_budget, 0
        ),
        'resident_blocks_upper_bound': min(
            registers_per_sm // registers_per_block,
            max_threads_per_sm // threads_per_block,
            max_blocks_per_sm,
        ),
    }


SHARED_LIMIT = tile_resource_ledger(32, 32, 32)['resident_blocks_upper_bound']
for registers in [32, 64, 80, 96]:
    record = register_resource_ledger(registers)
    record['combined_resident_blocks_upper_bound'] = min(
        SHARED_LIMIT, record['resident_blocks_upper_bound']
    )
    print(registers, '寄存器/线程 ->', record)
assert register_resource_ledger(32)['resident_blocks_upper_bound'] == 8
assert register_resource_ledger(64)['resident_blocks_upper_bound'] == 4
assert register_resource_ledger(80)['excess_registers_per_thread'] == 16
assert register_resource_ledger(80)['resident_blocks_upper_bound'] == 3
assert register_resource_ledger(64)['excess_registers_per_thread'] == 0

```

## Related Reading

**Overview:** Continue connecting on-chip reuse to the Triton block and FlashAttention, and observe how tile design reduces high-bandwidth memory access.

- [CUDA C Programming Guide: Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory): consult shared memory, bank conflicts, and synchronization semantics.
- [18. Triton Block Model](./18_Triton_Block_Model.md)
- [20. FlashAttention Sim](../02_PyTorch_Algorithms/20_FlashAttention_Sim.md)
---
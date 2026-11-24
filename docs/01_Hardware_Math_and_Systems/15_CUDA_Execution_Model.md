# 15. CUDA Execution Model | CUDA 执行模型

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `operator programming`, `CUDA`, `execution model` | **Audience:** learners starting with compilers and systems

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/15_CUDA_Execution_Model.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

CUDA organizes the threads of one kernel launch into Grids and Blocks; threads inside the same Block are then grouped into Warps for execution. The program chooses the grid and block sizes, and the device decides how this work resides and executes on the SMs. The task organization is not a layer-by-layer unfolding of the number of GPU cores.

This section builds the execution ledger from task coverage, Warp packing, and Block resource constraints. When you finish, you should be able to derive the launch geometry from the element count and judge whether a Block configuration satisfies the thread, shared memory, and register limits; these capabilities directly support the operator optimization route and the Kernel implementations in Parts 03–04.

**Keywords:** `grid`, `block`, `warp`

![CUDA execution hierarchy and resource mapping](../public/01_Hardware_Math_and_Systems/15_cuda_execution_hierarchy.svg)

---
## Prerequisite Reading

**Overview:** First understand GPU resources and the CUDA programming model, then map a tensor task onto Grid, Block, Warp, and Thread.

- [03. GPU Architecture and Memory](./03_GPU_Architecture_and_Memory.md)
- [08. Programming Models and CUDA/Triton](./08_Programming_Models_CUDA_Triton.md)


## Q1: Why is a kernel on the GPU not "one thread running"?

A Thread executes local computation, several threads form a Block, and all Blocks of one launch form a Grid. These are the units by which the program organizes tasks, not a number of Grids or Blocks that the chip has fixed in advance. NVIDIA groups 32 consecutively numbered threads in the same block into one Warp, and Lane denotes the thread's position within that group; branches and boundary conditions cause some Lanes not to participate in a given instruction.

For example, if each thread handles one element, then for 1000 elements with 256 threads per block you need 4 Blocks with 8 Warps each, and the last 24 threads process no valid elements. If you change to 128 threads per block, you need 8 Blocks with 4 Warps each. Both cover the same input, but they change the in-block collaboration and resource allocation; you cannot judge that the time is the same just because the total thread count is the same.

The SM (streaming multiprocessor) is the hardware unit that carries this work. One SM can hold multiple Blocks that satisfy the resource constraints, and the Block count of a Grid can also exceed the SM count, with the remaining work waiting for resources; a Block number does not designate a fixed SM, and a Warp is not 32 dedicated physical compute cores either. Threads can access global memory by address; the execution hierarchy is not a memory permission that narrows layer by layer. See the [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html) for the official definition.


### Q1 mechanism calculator: Derive Grid, Block, and Warp from the element count

Assume each thread handles one element; first compute the minimum covering grid, then distinguish the threads in the tail block that exceed the input from the empty slots produced by packing each block into Warps; the two are not the same quantity.



```python
def launch_geometry(num_elements, threads_per_block, warp_size=32):
    """Compute the blocks, warps, and valid threads in the tail block of a one-dimensional kernel."""
    if any(type(value) is not int or value <= 0 for value in (num_elements, threads_per_block, warp_size)):
        raise ValueError('元素数、块线程数和 Warp 大小必须是正整数')
    blocks = (num_elements + threads_per_block - 1) // threads_per_block
    warps_per_block = (threads_per_block + warp_size - 1) // warp_size
    launched_threads = blocks * threads_per_block
    return {
        'grid_blocks': blocks,
        'warps_per_block': warps_per_block,
        'total_warps': blocks * warps_per_block,
        'launched_threads': launched_threads,
        'inactive_tail_threads': launched_threads - num_elements,
        'warp_thread_slots': blocks * warps_per_block * warp_size,
        'unassigned_warp_slots': blocks * (warps_per_block * warp_size - threads_per_block),
    }

for threads_per_block in [64, 128, 256]:
    print(threads_per_block, '->', launch_geometry(1000, threads_per_block))
assert launch_geometry(1000, 256)['inactive_tail_threads'] == 24
assert launch_geometry(66, 33)['unassigned_warp_slots'] == 62
assert launch_geometry(66, 33)['inactive_tail_threads'] == 0


```

## Q2: How is data reused inside a Block through shared memory?

A Block provides threads with shared memory and a collaboration boundary, and a Warp is the execution grouping of threads. When producers write data and consumers read it, the corresponding synchronization and visibility conditions must be satisfied; belonging to the same Warp does not mean that all synchronization checks can be omitted.

Moving repeatedly used data into shared memory first can reduce logical global reads. Whether it reduces actual HBM traffic and whether it shortens the time also depends on cache hits, synchronization, and on-chip resource occupancy.

![Shared memory reuse path within a Block](../public/01_Hardware_Math_and_Systems/15_memory_reuse_path.svg)


### Q2 traffic ledger: How much reuse reduces the global read requests

Assume the data block is read in full every time and cache hits are ignored for now, then compare the byte volume of repeated reads with loading it once per block. Actual HBM traffic is also affected by the cache, so these logical read bytes cannot be used directly in place of hardware counters.



```python
def block_reuse_traffic(blocks, tile_elements, bytes_per_element, reuse_count):
    """Compare the logical byte volume of repeated full-block global reads with loading once per block."""
    if any(type(value) is not int or value <= 0 for value in (blocks, tile_elements, bytes_per_element, reuse_count)):
        raise ValueError('块数、元素数、元素字节数和使用次数必须是正整数')
    tile_bytes = tile_elements * bytes_per_element
    baseline_global_read_bytes = blocks * tile_bytes * reuse_count
    tiled_global_read_bytes = blocks * tile_bytes
    return {
        'baseline_global_read_bytes': baseline_global_read_bytes,
        'tiled_global_read_bytes': tiled_global_read_bytes,
        'global_read_bytes_saved': baseline_global_read_bytes - tiled_global_read_bytes,
        'shared_bytes_per_block': tile_bytes,
    }

for reuse_count in [1, 2, 4]:
    print('reuse', reuse_count, '->', block_reuse_traffic(8, 4096, 2, reuse_count))
assert block_reuse_traffic(8, 4096, 2, 1)['global_read_bytes_saved'] == 0


```

## Q3: How does the resource requirement of a single Block constrain the configuration?

The same task can choose different thread counts, shared memory, and register usage, but these resources are all limited by the device and the compilation configuration. First compute the per-block requirement item by item, then compare it with the given budget; too many threads, shared memory over the limit, or too high a register requirement can all prevent the configuration from executing as expected.

The block size is chosen by the program, while the allowed range is limited jointly by the device and the kernel; the shared memory and register budgets are also not fixed numbers shared by all generations. The example budget below checks the per-block requirement; multi-block residency and allocation granularity continue to be computed in the SRAM optimization subsection. The resource check and the tensor coverage check must both hold before entering actual kernel verification; you should also check the compiled register and shared memory usage, the launch restrictions, and the output, and then measure the time.

![Mapping from the execution model to kernel implementation](../public/01_Hardware_Math_and_Systems/15_execution_model_to_kernel.svg)


### Q3 resource gate: Check whether a per-block configuration exceeds the given budget

The budget is an example input and does not represent a specific GPU. Check the threads per block, the shared memory, the registers per thread, and the total registers per block separately; allocation granularity and multi-block residency are ignored for now, and staying within budget does not equal being able to actually launch or reach high throughput.



```python
def block_resource_gate(threads_per_block, shared_bytes_per_block,
                        registers_per_thread, limits):
    """Check per-block resources against a given integer budget; does not replace compilation and device launch verification."""
    if type(threads_per_block) is not int or threads_per_block <= 0:
        raise ValueError('块线程数必须是正整数')
    if any(type(value) is not int or value < 0
           for value in (shared_bytes_per_block, registers_per_thread)):
        raise ValueError('共享内存字节数与寄存器数量必须是非负整数')
    required = ('max_threads_per_block', 'max_shared_bytes_per_block',
                'max_registers_per_thread', 'register_file_bytes_per_block')
    if any(key not in limits or type(limits[key]) is not int or limits[key] <= 0
           for key in required):
        raise ValueError('必须提供全部资源预算，且预算为正整数')
    reasons = []
    if threads_per_block > limits['max_threads_per_block']:
        reasons.append('threads_per_block')
    if shared_bytes_per_block > limits['max_shared_bytes_per_block']:
        reasons.append('shared_memory')
    if registers_per_thread > limits['max_registers_per_thread']:
        reasons.append('registers_per_thread')
    register_bytes = threads_per_block * registers_per_thread * 4
    if register_bytes > limits['register_file_bytes_per_block']:
        reasons.append('register_file')
    return {'within_stated_limits': not reasons, 'reasons': reasons,
            'register_bytes_per_block': register_bytes}


limits = {
    'max_threads_per_block': 1024,
    'max_shared_bytes_per_block': 64 * 1024,
    'max_registers_per_thread': 128,
    'register_file_bytes_per_block': 256 * 1024,
}
valid = block_resource_gate(256, 32 * 1024, 64, limits)
invalid = block_resource_gate(2048, 96 * 1024, 64, limits)
assert valid['within_stated_limits']
assert not invalid['within_stated_limits']
assert 'registers_per_thread' in block_resource_gate(32, 0, 129, limits)['reasons']
print(valid)
print(invalid)


```

## Related Reading

Continue from logical task coverage to real CUDA indexing, or compare the Triton data-block abstraction.

- [Part04·01 CUDA Vector Addition](../04_CUDA_and_System_Optimization/01_CUDA_Custom_Kernel_Intro.md): thread indexing, tail blocks, and actual launches.
- [18 Triton Block Model](./18_Triton_Block_Model.md): the difference between Programs, logical element slots, and physical threads.


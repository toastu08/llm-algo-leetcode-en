# 18. Triton Block Model | Triton Block 模型

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `operator programming`, `Triton`, `Block Model` | **Audience:** learners starting with compilers and systems

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/18_Triton_Block_Model.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

Triton uses Programs and Tiles to block up the tensor space. When the tensor size is not divisible by the block size, the logical data block of the last Program contains element slots beyond the input, and a Mask is needed to guard the boundary accesses.

This section builds the spatial mapping from one-dimensional coverage, two-dimensional Tile layout, and boundary Mask. When you finish, you should be able to compute the Program count, the boundary padding, and the valid indices; these capabilities connect directly to the operator optimization route and the Triton Kernels of Part 03.

**Keywords:** `program`, `block`, `tile`

![Mapping of Triton Program, Tile, and Mask](../public/01_Hardware_Math_and_Systems/18_triton_program_tile_mask.svg)

---

## Prerequisite Reading

**Overview:** First understand the programming model and the device execution hierarchy, then see how a Triton Program covers the tensor space with tiles and masks.

- [08. Programming Models and CUDA/Triton](./08_Programming_Models_CUDA_Triton.md)
- [15. CUDA Execution Model](./15_CUDA_Execution_Model.md)


## Q1: What is the relationship between a Program, a data block, and threads?

A Program is an execution instance that handles one block of logical data, and a Tile describes the tensor elements it covers. The Program ID determines the region position, and the in-block offset determines the element index; these are divisions of the data space, not a physical thread count. Triton's launch grid specifies the number of Program instances, not the SM count.

For example, a Program handles 1024 elements with `num_warps=4`, using 4 Warps and 128 threads on the ordinary NVIDIA single-CTA path of this section. 1024 is the number of logical element slots and 128 is the thread count; the compiled layout distributes the work of multiple elements to the threads. You cannot allocate one thread directly per element, nor call this logical block a "1024-thread CUDA Block." The CTA here is a CUDA thread block; special multi-CTA or Warp-specialized configurations need their mapping checked separately.

Changing `BLOCK_SIZE` changes the data coverage, and changing `num_warps` changes the execution configuration; both may in turn affect registers, shared memory, and the actual instructions. This example first uses ceiling division to obtain the minimum grid, and only verifies the Program count, the valid elements of the tail block, and the masked element slots; the physical layout and resources are left to the generated code and the actual kernel verification.


### Q1 mechanism calculator: How a Program covers a one-dimensional tensor

Compute the minimum covering grid and the logical element slot count. One element slot is not equal to one CUDA thread or hardware lane; the actual thread mapping is determined by the compiled layout, and this example only verifies the data coverage.



```python
def program_coverage(num_elements, block_size):
    """Compute the Program count, logical element slots, and tail-block valid elements of the minimum grid."""
    if any(type(value) is not int or value <= 0 for value in (num_elements, block_size)):
        raise ValueError('输入元素数与块大小必须是正整数')
    programs = (num_elements + block_size - 1) // block_size
    covered_slots = programs * block_size
    last_program_valid = num_elements - (programs - 1) * block_size
    return {
        'programs': programs,
        'covered_slots': covered_slots,
        'last_program_valid': last_program_valid,
        'masked_slots': covered_slots - num_elements,
    }

for block_size in [64, 128, 256]:
    print(block_size, '->', program_coverage(1000, block_size))
assert program_coverage(1000, 256)['masked_slots'] == 24


```

## Q2: Why must Triton define the data layout first?

The layout determines how the elements of the same tensor are distributed to Programs, and then how they are loaded and computed by threads. For example, a two-dimensional matrix tiled by row allows the same Program to handle consecutive columns; if the input is a transposed view, the same logical columns may not be contiguous in physical addresses. Looking only at the data block shape, you cannot tell whether the loads are coalesced, nor whether an element is reused repeatedly on-chip.

Enlarging a Tile may reduce the Program count and increase the reuse opportunities, but it may also increase intermediate values, tail-block padding, and resource requirements. Different devices and compilers may generate different execution arrangements for the same logical layout, so "a larger logical block" does not mean "more threads" or "faster execution." Below only the geometric coverage and padding are compared; the actual addresses, instructions, and time are left to implementation and measurement.


### Q2 layout calculation: How much boundary padding the tile shape brings

Compare the geometric coverage of the same matrix; the element slot count does not represent the thread count. Any positive-integer block shape can participate in this geometric calculation, but it is not guaranteed that the corresponding shape can be compiled directly into a Triton kernel.



```python
def tile_layout(rows, cols, block_rows, block_cols):
    """Compute the two-dimensional tiling grid, the logical element slot count, and the boundary padding ratio."""
    if any(type(value) is not int or value <= 0 for value in (rows, cols, block_rows, block_cols)):
        raise ValueError('矩阵和块形状必须是正整数')
    row_programs = (rows + block_rows - 1) // block_rows
    col_programs = (cols + block_cols - 1) // block_cols
    useful_elements = rows * cols
    covered_elements = row_programs * col_programs * block_rows * block_cols
    return {
        'program_grid': (row_programs, col_programs),
        'useful_elements': useful_elements,
        'covered_elements': covered_elements,
        'padding_elements': covered_elements - useful_elements,
        'useful_ratio': useful_elements / covered_elements,
    }

for tile in [(16, 16), (32, 32), (64, 64)]:
    print(tile, '->', tile_layout(100, 130, *tile))
assert tile_layout(128, 128, 32, 32)['padding_elements'] == 0


```

## Q3: How do the Program ID, the offset, and the Mask keep the boundary safe?

The Program ID determines the position of the current block, the in-block offset gives the element index, and the Mask shields the loads and stores that exceed the input range. First check whether the valid indices in the minimum grid of this example are complete and non-repeating, then check the elements masked in the tail block.

The CPU calculator only verifies the logical indices; an actual Triton implementation also needs to use block shapes and layouts supported by the compiler, apply the Mask to loads and stores, and check the numerical results and the resource usage.


### Q3 boundary verification: Whether the indices in the minimum grid are complete and non-repeating

First check whether the Program ID belongs to the minimum covering grid of this example, then generate the offsets and the Mask. Traverse all Programs and verify that the valid indices cover the input exactly once; finally check the tail-block and divisible cases.



```python
def program_offsets(program_id, block_size, num_elements):
    """Generate the offsets and boundary Mask of one Program within the minimum covering grid."""
    coverage = program_coverage(num_elements, block_size)
    if type(program_id) is not int or not 0 <= program_id < coverage['programs']:
        raise ValueError('Program ID 必须属于本例的最小覆盖网格')
    offsets = [program_id * block_size + slot for slot in range(block_size)]
    mask = [0 <= offset < num_elements for offset in offsets]
    valid_offsets = [offset for offset, valid in zip(offsets, mask) if valid]
    return {
        'offsets': offsets, 'mask': mask, 'valid_offsets': valid_offsets,
        'valid_count': len(valid_offsets),
        'safe': all(0 <= offset < num_elements for offset in valid_offsets),
    }


tail = program_offsets(program_id=3, block_size=32, num_elements=100)
assert tail['valid_offsets'] == [96, 97, 98, 99]
assert tail['valid_count'] == 4 and tail['safe']
for size in [1, 32, 64, 100]:
    coverage = program_coverage(size, 32)
    all_valid = [
        offset
        for program_id in range(coverage['programs'])
        for offset in program_offsets(program_id, 32, size)['valid_offsets']
    ]
    assert all_valid == list(range(size))
assert program_offsets(1, 32, 64)['valid_count'] == 32
print(tail)


```

## Related Reading

Connect the data-space coverage to an actual kernel, then check how the logical block corresponds to the generated code and resources.

- [Part03·01 Vector Addition](../03_Triton_Kernels/01_Triton_Vector_Addition.md): Program ID, offsets, and tail-block masks.
- [Part03·18 Compilation Artifacts and Execution Check](../03_Triton_Kernels/18_Triton_Compilation_Artifacts_and_Execution.md): specialization conditions, resources, and the actual launch target.
- [Triton vector addition tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html): the official example of logical blocks and launch grids.


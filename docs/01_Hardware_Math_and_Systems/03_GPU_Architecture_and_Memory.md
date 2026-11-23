# 03. GPU Architecture and Memory | GPU 物理架构与内存层级

**Difficulty:** Hard | **Environment:** CPU-first | **Tags:** `hardware systems`, `GPU`, `memory hierarchy` | **Audience:** learners who have mastered tensor operations

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

Model computation on a GPU is constrained at the same time by compute throughput, memory bandwidth, and memory capacity. The set of data that the current stage needs to access frequently or keep resident temporarily is called the working set, and it determines how data should move between registers, on-chip storage, and memory.

Even though they are all called GPUs, different products have different memory, on-chip resources, and instruction capabilities; an architecture name also cannot be taken directly as the complete specification of a device. This section connects these differences to compute units, memory levels, and the working set, supporting tiling, matrix computation, and resource judgment in operator optimization. After completing it, you should be able to make an initial judgment, based on compute volume, data volume, and access path, whether an operator is more likely limited by compute, bandwidth, or capacity.

**Keywords:** `Tensor Core`, `SRAM`, `HBM`

![Concept relationships in this section](../public/01_Hardware_Math_and_Systems/03_gpu_architecture_memory_map.svg)

---
## Prerequisite Reading
**Overview:** First review the order-of-magnitude relationships among dtype, parameter count, and FLOPs, then start from the GPU execution hierarchy and memory hierarchy to judge where compute, bandwidth, and capacity pressure each come from.

- [Part 01 · 01 Data Types and Precision](./01_Data_Types_and_Precision.md)
- [Part 01 · 02 Parameter Count and FLOPs](./02_LLM_Params_and_FLOPs.md)

---
## Q1: Why does a Transformer not necessarily speed up proportionally when GPU compute improves?


The peak compute in the GPU specification only describes how many operations the compute units can complete at best; it does not mean the data has been delivered to the compute units in time. A Transformer consists of multiple repeated computation blocks, and both Attention and MLP produce inputs, weights, activations, and intermediate results; if the reading, write-back, and reuse of these data cannot keep up, the compute units wait, and peak compute cannot be converted into an equal program speedup.
Therefore, to judge what limits an operator, you can first ask two questions: how much computation it must complete, and how much data it must move from memory. The FLOPs, Bytes, and arithmetic intensity later quantify these two questions.

Different GPU generations make different trade-offs in dtype support, on-chip storage, asynchronous transfer, and interconnect capability. Here we first explain how these capabilities change operator execution, then distinguish architecture names, specific models, and device capabilities.

You can first use the following three-way classification to locate the phenomenon:

| Phenomenon | Main limitation | Common symptom |
| --- | --- | --- |
| Data does not fit | Memory capacity | OOM; offload, sharding, or a shorter sequence is needed |
| Data moves slowly | Memory bandwidth or access path | Compute units wait, and bandwidth is close to the limit |
| Computation is slow | Compute throughput | CUDA Cores or Tensor Cores are close to saturation |

### Q1 Mini Check: First distinguish the compute path from the data supply

First observe what the compute path and the memory path each handle, then use the hierarchy table and arithmetic intensity below to judge the specific bottleneck.

```python
from typing import Dict

# Teaching proxy values: used to compare orders of magnitude, not the measured specification of a specific GPU.
TEACHING_BANDWIDTH = {'shared_memory': 19e12, 'l2_cache': 1.5e12, 'hbm': 1.5e12}
COMPUTE_PATHS = {'cuda_core': '线程级标量 / 向量计算', 'tensor_core': '矩阵乘加与低精度路径'}


def bytes_to_gb(bytes_val: float) -> float:
    """Convert bytes to decimal GB; for teaching display only."""
    if bytes_val < 0:
        raise ValueError('bytes_val 不能为负数')
    return bytes_val / 1e9


def analyze_memory_hierarchy() -> Dict[str, Dict[str, float]]:
    """Compute the order-of-magnitude bandwidth of the memory hierarchy and a proxy time for a 1KB transfer."""
    result = {}
    for mem_type, bandwidth in TEACHING_BANDWIDTH.items():
        result[mem_type] = {'bandwidth_tb_s': bandwidth / 1e12, 'time_for_1kb_ns': 1024 / bandwidth * 1e9}
    return result


mem_analysis = analyze_memory_hierarchy()
print('计算路径：', COMPUTE_PATHS)
for name, stats in mem_analysis.items():
    print(f"{name:15s}: {stats['bandwidth_tb_s']:6.1f} TB/s, 1KB transfer proxy: {stats['time_for_1kb_ns']:6.2f} ns")
def test_gpu_memory_practice():
    assert set(COMPUTE_PATHS) == {'cuda_core', 'tensor_core'}
    assert mem_analysis['shared_memory']['bandwidth_tb_s'] > mem_analysis['hbm']['bandwidth_tb_s']
    print('✅ 03 hardware path tests passed')


test_gpu_memory_practice()

```

### From architecture names to specific device capabilities

An architecture name (such as Ampere, Hopper, Blackwell) describes a class of hardware design, a product model points to a specific device, and Compute Capability (CC) identifies CUDA hardware features and instruction characteristics. The three cannot be substituted for one another: products that both belong to Blackwell may also correspond to different CCs; the same CC also does not mean the same memory capacity, SM count, bandwidth, and interconnect configuration.

For example, NVIDIA's [device compute capability table](https://developer.nvidia.com/cuda/gpus) lists A100 as CC 8.0, H100 as 9.0, and B200 as 10.0, while GeForce RTX 5070 Ti is listed as 12.0. CC is a capability classification, not a score that can be compared directly for speed; you cannot infer that a consumer product is faster on a given training or inference task just because 12.0 is greater than 10.0.

To judge whether an optimization can be used, first check the specific model and CC, then look up the target instruction, data type, and resource limits; to judge whether it is worth using, you also need the input, data layout, and real execution results. The CUDA version supported by the driver, the CUDA version used by the PyTorch build, and the local Toolkit version are software conditions, not the device CC. Record resources and versions separately, to avoid treating "the device can be recognized" as "the target kernel can run".

| Question to judge | What to check | Impact on the operator implementation |
| --- | --- | --- |
| Whether a matrix or transfer instruction can be used | CC, the target requirement of that instruction, compiler support | Choose the supported path and keep the necessary fallback implementation |
| Whether a tile fits | Thread, register, and shared-memory limits of the specific device and kernel | Determine tiling, buffering, and residency conditions |
| Whether the data fits in memory and moves fast enough | Product memory, bandwidth, and access method | Judge capacity, reuse, and data-supply pressure |
| How multiple GPUs exchange data | Actual links, host topology, and interconnect configuration | Determine the communication path; do not infer bandwidth from the architecture name |

## Q2: How does matrix computation in a Transformer use the Tensor Core?


Both Attention and MLP in a Transformer contain matrix multiplication. The Tensor Core is a compute unit specifically for increasing matrix multiply-add throughput, better suited to answering "is the matrix multiplication fast enough" than to directly answering "is the whole Transformer block fast enough".

CUDA Cores handle more general thread-level computation; Tensor Cores mainly serve matrix multiply-add under low-precision data types such as FP16, BF16, and FP8. The specifically supported dtypes, data block shapes, and throughput depend on the GPU architecture; a single fixed specification cannot summarize all devices.

The Tensor Core covers only the matrix multiply-add segment; data preparation, layout conversion, Softmax, and result write-back still need other execution units and memory levels to cooperate. Therefore, the fact that matrix multiplication has a Tensor Core candidate path does not mean the whole Transformer block will speed up by the same ratio.
The mini-check in this section only compares the impact of different matrix shapes and dtypes on FLOPs, input/output scale, and the Tensor Core candidate path. The relationship between the Attention working set and sequence length is observed separately in Q4.

Compute resources and thread execution groups need to be viewed separately; they are not the same kind of hierarchy:

| Object | Role |
| --- | --- |
| SM | Schedules thread blocks and organizes on-chip compute resources |
| Warp | Executes instructions as a group of threads |
| CUDA Core | Executes general thread-level computation |
| Tensor Core | Executes matrix multiply-add of specific shapes |

### Q2 Mini Check: Matrix shapes and Tensor Core candidate paths

First compare matrix shapes, dtypes, and FLOPs, then judge whether there is a Tensor Core candidate path for low-precision matrix multiplication; this is still a mechanism simulation, not a hardware throughput test.

```python
def estimate_matmul_workload(m, n, k, dtype='bf16'):
    """Estimate the matrix multiplication workload and mark the Tensor Core candidate path."""
    if min(m, n, k) <= 0:
        raise ValueError('矩阵维度必须为正数')
    if dtype not in {'fp16', 'bf16', 'fp8', 'fp32'}:
        raise ValueError('unsupported dtype')
    return {
        'flops': 2 * m * n * k,
        'input_elements': m * k + k * n,
        'output_elements': m * n,
        'dtype': dtype,
        'tensor_core_candidate': dtype in {'fp16', 'bf16', 'fp8'},
    }

for shape, dtype in [((1024, 1024, 4096), 'bf16'), ((1024, 1024, 4096), 'fp32'), ((2048, 4096, 4096), 'fp16')]:
    print(shape, dtype, estimate_matmul_workload(*shape, dtype))


def test_matmul_workload():
    low_precision = estimate_matmul_workload(1024, 1024, 4096, 'bf16')
    fp32 = estimate_matmul_workload(1024, 1024, 4096, 'fp32')
    assert low_precision['flops'] == fp32['flops']
    assert low_precision['tensor_core_candidate'] is True
    assert fp32['tensor_core_candidate'] is False
    print('✅ Tensor Core candidate tests passed')


test_matmul_workload()

```

## Q3: How does the GPU memory hierarchy affect data movement?


Think of the memory hierarchy first as "different distances from the compute units": the closer to the compute units, the faster the access is usually and the smaller the capacity; the farther away, the larger the capacity, but the more obvious the transfer cost. The key to optimization is not to keep all data in the fastest level, but to keep data that will be reused as close to the compute units as possible.

When an operator completes only a small amount of computation for many bytes moved, it is more likely limited by global memory bandwidth. **Arithmetic intensity** can quantify this judgment:

```text
Arithmetic Intensity = FLOPs / Bytes
```

A low arithmetic intensity means less computation is completed per data transfer, making bandwidth limitation more likely; only with a high arithmetic intensity are the compute units more likely to become the limiting factor. If threads in the same warp access consecutive addresses, coalesced access forms more easily; when the access span is large or the layout does not match, more global memory transactions are added. Triton's block abstraction can help organize tiles and data movement, but it does not automatically decide the optimal tile size or reuse pattern; the specific choice still needs to be judged together with the operator shape and profiling. Excessive resource usage may also reduce the number of warps that can reside simultaneously in one SM, that is, occupancy. The table below grounds these judgments in each storage level.

| Level | Main role | Main observation | Typical risk |
| --- | --- | --- | --- |
| Registers | Hold the small amount of data a thread is using | Temporary variables and register usage | Register spilling, reduced occupancy |
| Shared Memory (on-chip SRAM) | Reuse data within a thread block | Whether the tile fits, whether data is reused repeatedly | Bank conflicts, excessive capacity or occupancy |
| L2 Cache | Buffer accesses by multiple SMs to global data | Whether different thread blocks can reuse the same data | Insufficient hit rate, repeated memory access |
| Global memory (such as HBM or GDDR) | Hold weights, activations, and intermediate results | Access count, bandwidth, and capacity | Bandwidth-bound, insufficient capacity |

### Q3 Mini Check: Judge the bottleneck with arithmetic intensity

Put the FLOPs of an operator together with the number of bytes it must move, and observe whether it is more likely limited by compute throughput or memory bandwidth.


```python
def estimate_arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """Compute arithmetic intensity: the FLOPs corresponding to each 1 Byte of data moved."""
    if flops < 0 or bytes_moved <= 0:
        raise ValueError('FLOPs 不能为负数，bytes_moved 必须为正数')
    return flops / bytes_moved


def roofline_estimate(flops: float, bytes_moved: float, peak_compute_tflops: float, memory_bandwidth_gb_s: float) -> Dict[str, float | str]:
    """Estimate attainable performance and the bottleneck with a simplified Roofline; not a profiler counter result."""
    if flops < 0 or bytes_moved <= 0 or peak_compute_tflops <= 0 or memory_bandwidth_gb_s <= 0:
        raise ValueError('输入必须满足 FLOPs>=0，字节数、峰值算力和带宽均为正数')
    intensity = estimate_arithmetic_intensity(flops, bytes_moved)
    bandwidth_roof = intensity * memory_bandwidth_gb_s / 1e3
    return {'arithmetic_intensity': intensity, 'bandwidth_roof_tflops': bandwidth_roof, 'attainable_tflops': min(peak_compute_tflops, bandwidth_roof), 'bottleneck': 'compute' if peak_compute_tflops <= bandwidth_roof else 'memory'}


cases = {'elementwise_like': (2_000_000, 1_000_000), 'matmul_like': (2_000_000_000, 100_000_000)}
for name, (flops, moved) in cases.items():
    r = roofline_estimate(flops, moved, 300, 1500)
    print(f"{name}: intensity={r['arithmetic_intensity']:.1f} FLOPs/Byte, {r['bottleneck']} bound")


def test_roofline_estimate():
    assert roofline_estimate(2e6, 1e6, 300, 1500)['bottleneck'] == 'memory'
    result = roofline_estimate(2e12, 1e8, 300, 1500)
    assert result['bottleneck'] == 'compute' and result['attainable_tflops'] <= 300
    print('✅ roofline tests passed')


test_roofline_estimate()

```

## Q4: Why does the Attention working set expose a memory bottleneck?


When a model runs, it produces weights, activations, and intermediate tensors that must move between compute units and different memory levels. Attention is a typical example: as the sequence grows longer, the complete score matrix grows with the square of the sequence length, and both the working set and data movement may become limiting.

This section first uses the Attention working set to explain the necessity of tiling; to continue learning about tiled working sets and Online Softmax, read [Part 01 · 14 FlashAttention Memory Model](./14_FlashAttention_Memory_Model.md) and [Part 02 · 20 FlashAttention Simulation](../02_PyTorch_Algorithms/20_FlashAttention_Sim.md).
The mini-check first estimates the working sets of Q/K/V, the score matrix, and the output, then compares the size of the complete score matrix with a single tile; it explains the change in residency, and does not represent the end-to-end performance of a real kernel.

### Q4 Mini Check: Attention working set and tiling scale

```python
def calculate_attention_vram(seq_len: int, num_heads: int, head_dim: int, dtype_bytes: int = 2, batch_size: int = 1) -> Dict[str, float]:
    """Estimate the working-set memory of standard Attention; not the CUDA peak memory of a complete model."""
    if min(seq_len, num_heads, head_dim, dtype_bytes, batch_size) <= 0:
        raise ValueError('所有形状和 dtype_bytes 都必须为正数')
    qkv = 3 * batch_size * seq_len * num_heads * head_dim * dtype_bytes
    matrix = batch_size * num_heads * seq_len * seq_len * dtype_bytes
    output = batch_size * seq_len * num_heads * head_dim * dtype_bytes
    flops = 4 * batch_size * num_heads * seq_len * seq_len * head_dim
    return {'qkv': bytes_to_gb(qkv), 'attention_matrix': bytes_to_gb(matrix), 'output': bytes_to_gb(output), 'total': bytes_to_gb(qkv + matrix + output), 'score_flops': flops, 'score_intensity': flops / matrix}


def attention_sensitivity(seq_lens, batch_size=1, num_heads=32, head_dim=128, dtype_bytes=2):
    """Generate the working-set table under different sequence lengths; not equal to the full-model peak memory."""
    rows = []
    for seq_len in seq_lens:
        r = calculate_attention_vram(seq_len, num_heads, head_dim, dtype_bytes, batch_size)
        rows.append({'seq_len': seq_len, 'qkv_gb': round(r['qkv'], 4), 'attention_matrix_gb': round(r['attention_matrix'], 4), 'total_working_set_gb': round(r['total'], 4), 'attention_share': round(r['attention_matrix'] / r['total'], 4), 'score_flops': r['score_flops'], 'score_intensity': round(r['score_intensity'], 2)})
    return rows


def attention_score_bytes(seq_len: int, dtype_bytes: int = 2) -> int:
    """Estimate the size of the complete Attention score matrix."""
    if seq_len <= 0 or dtype_bytes <= 0:
        raise ValueError('seq_len 和 dtype_bytes 必须为正数')
    return seq_len * seq_len * dtype_bytes


def tile_bytes(block_size: int, dtype_bytes: int = 2) -> int:
    """Estimate the size of one square tile."""
    if block_size <= 0 or dtype_bytes <= 0:
        raise ValueError('block_size 和 dtype_bytes 必须为正数')
    return block_size * block_size * dtype_bytes


def compare_attention_working_set(seq_len: int, block_size: int = 128, dtype_bytes: int = 2):
    """Compare the complete matrix with a single-tile working set; does not simulate a real FlashAttention kernel."""
    score, tile = attention_score_bytes(seq_len, dtype_bytes), tile_bytes(block_size, dtype_bytes)
    return {'naive_score_mb': score / 1024 / 1024, 'tile_kb': tile / 1024, 'tile_count': (seq_len + block_size - 1) // block_size, 'working_set_reduction': score / tile}


for row in attention_sensitivity([512, 2048, 4096, 8192]):
    print(row)

for seq_len in [2048, 4096, 8192]:
    print(f"seq_len={seq_len}: {compare_attention_working_set(seq_len)}")


def test_attention_working_set():
    rows = attention_sensitivity([512, 1024, 2048])
    totals = [r['total_working_set_gb'] for r in rows]
    matrices = [r['attention_matrix_gb'] for r in rows]
    assert totals == sorted(totals) and matrices == sorted(matrices)
    assert rows[-1]['attention_share'] > rows[0]['attention_share']
    short, long = compare_attention_working_set(2048), compare_attention_working_set(4096)
    assert long['naive_score_mb'] > short['naive_score_mb']
    assert long['tile_kb'] == short['tile_kb'] and long['tile_count'] > short['tile_count']
    print('✅ attention working-set tests passed')


test_attention_working_set()

```

## Extended Q5: After spreading memory across GPUs, where does the communication cost come from?


When a single GPU cannot hold the model or cross-GPU cooperation is needed, parameters, activations, or the KV Cache may need to be exchanged between GPUs. Spreading memory does not eliminate the cost; it transfers part of the cost to communication bandwidth, communication latency, and synchronization waiting.

PCIe, NVLink, and NVSwitch can provide different connection bandwidths and topology conditions; the specific differences depend on the devices, link count, and platform implementation. This extended section is used to judge: if spreading requires frequent All-Reduce, All-Gather, or state synchronization, communication may become the new bottleneck. The code below only estimates proxy times for PCIe / NVLink link transfer and repeated synchronization, and does not simulate the real algorithm, topology, or communication volume of collective communication; these need to be verified in distributed projects.

| Communication mode | Typical use | Main cost |
| --- | --- | --- |
| PCIe | Data transfer between GPU and CPU, or GPU and GPU | Bandwidth and latency affected by host topology |
| NVLink | High-speed interconnect between GPUs within a node | Requires matching GPUs, links, and topology |
| NVSwitch | Provides a more uniform interconnect path in multi-GPU systems | Still needs to consider synchronization and communication volume |
| All-Reduce / All-Gather | Aggregate or exchange parameters, gradients, or activations in parallel computation | Communication volume, synchronization waiting, and critical-path blocking |


```python
def pcie_vs_nvlink(payload_mb: float, pcie_gbps: float = 64, nvlink_gbps: float = 900, communication_rounds: int = 1, sync_overhead_ms: float = 0.0):
    """Estimate PCIe / NVLink link time and repeated synchronization cost; not measured.

    All-Reduce, All-Gather, NVSwitch topology, and real collective communication volume are not in this proxy model."""
    if payload_mb <= 0 or pcie_gbps <= 0 or nvlink_gbps <= 0:
        raise ValueError('payload 和带宽必须为正数')
    if communication_rounds < 1 or sync_overhead_ms < 0:
        raise ValueError('communication_rounds 至少为 1，sync_overhead_ms 不能为负数')
    pcie_once = payload_mb * 8 / pcie_gbps
    nvlink_once = payload_mb * 8 / nvlink_gbps
    return {'pcie_once_ms': round(pcie_once, 3), 'nvlink_once_ms': round(nvlink_once, 3), 'pcie_total_ms': round((pcie_once + sync_overhead_ms) * communication_rounds, 3), 'nvlink_total_ms': round((nvlink_once + sync_overhead_ms) * communication_rounds, 3), 'communication_rounds': communication_rounds, 'evidence_level': 'link_transfer_proxy', 'collectives_modeled': False}


for payload in [64, 256, 1024]:
    print(payload, 'MB ->', pcie_vs_nvlink(payload, communication_rounds=4, sync_overhead_ms=0.05))
print('带宽差异只有在通信位于关键路径时才会影响端到端时间')


def test_interconnect_model():
    one, many = pcie_vs_nvlink(256, communication_rounds=1), pcie_vs_nvlink(256, communication_rounds=4)
    assert many['pcie_total_ms'] > one['pcie_total_ms'] and many['nvlink_total_ms'] > one['nvlink_total_ms']
    assert many['collectives_modeled'] is False and many['evidence_level'] == 'link_transfer_proxy'
    print('✅ interconnect model tests passed')


test_interconnect_model()

```

## Related Reading

After separating device capabilities from execution organization, trace how threads cover tensors and what constrains on-chip reuse.

- [15 CUDA Execution Model](./15_CUDA_Execution_Model.md): Grid, Block, Warp, and thread coverage.
- [24 SRAM Optimization Techniques](./24_SRAM_Optimization_Techniques.md): how shared-memory and register requirements constrain residency.
- [NVIDIA CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/): official definitions of compute capability, instructions, and resource limits.

# Part 01: Hardware, Math, and Systems

## Part Overview

Part 01 connects model scale, device resources, and execution constraints. Starting from data types, parameter count, and compute volume, learners gradually understand how single-device memory access, multi-device communication, program execution, compiler backends, and device runtimes jointly constrain training and inference; after completing it, you can turn a model or workload into resource assumptions and know what to verify next.

## Learning Path

First use 1A and 1B to establish orders of magnitude, the hardware hierarchy, and the single-device working set, then choose the communication-parallelism, execution-mapping, compiler-backend, or device-adaptation branch according to the actual problem.

![Part 01 learning path](../docs/public/01_Hardware_Math_and_Systems/part01_learning_route.svg)

| Learning mode | Recommended entry | Judgment you can complete |
|:---|:---|:---|
| Build a common foundation | [1A Numerics and Scale](./1A.md) → [1B Single-Device Execution, Memory and Data Access](./1B.md) | Whether parameters, state, compute volume, and the working set fall within the available resources |
| Training or multi-GPU problems | [1C Communication, Parallelism and Fault Tolerance](./1C.md) | What should be split, what should be exchanged, and whether the recovery cost is acceptable |
| Operator or execution-path problems | [1D Execution Models, Asynchronous Scheduling and Operator Mapping](./1D.md) → [1E Compilers and Backends](./1E.md) | How the program is mapped to the device, and whether the backend can generate the target execution path |
| Deployment and device-selection problems | [1F Hardware Adaptation and System Constraints](./1F.md) | Whether the model artifact, runtime, and cost allow entering the target device |

## Group Overview

| Learning group | Question to answer | Core entry | Input passed to later topics |
|:---|:---|:---|:---|
| [1A Numerics and Scale](./1A.md) | How much storage and compute do the model, training state, and sparse structure each need? | [01 Data Types and Precision](./01_Data_Types_and_Precision.ipynb) → [02 Parameter Count and FLOPs](./02_LLM_Params_and_FLOPs.ipynb) | Resource ledger for quantization, model architecture, and performance candidates |
| [1B Single-Device Execution, Memory and Data Access](./1B.md) | How do the working set, KV state, matrix computation, and memory-access constraints form a single-GPU bottleneck? | [03 GPU Architecture and Memory](./03_GPU_Architecture_and_Memory.ipynb) | Capacity and critical-path assumptions for inference, performance analysis, and operator optimization |
| [1C Communication, Parallelism and Fault Tolerance](./1C.md) | How is state split, when does communication become exposed, and how do you recover after a system interruption? | [05 Communication Topologies](./05_Communication_Topologies.ipynb) → [06 VRAM Calculation and ZeRO](./06_VRAM_Calculation_and_ZeRO.ipynb) | Strategies, payloads, and recovery conditions for communication parallelism |
| [1D Execution Models, Asynchronous Scheduling and Operator Mapping](./1D.md) | How do tasks land on the host, device, thread, warp, block, or program? | [07 CPU/GPU Heterogeneous Scheduling](./07_CPU_GPU_Heterogeneous_Scheduling.ipynb) → [08 CUDA/Triton Programming Models](./08_Programming_Models_CUDA_Triton.ipynb) | Execution dependencies for operator implementation and performance optimization |
| [1E Compilers and Backends](./1E.md) | How do graphs, shapes, and fusion become execution paths that a backend can run? | [09 Graph Optimization and Compilers](./09_AI_Compilers_and_Graph_Optimization.ipynb) | Backend constraints in operator optimization and deployment |
| [1F Hardware Adaptation and System Constraints](./1F.md) | Can device, runtime, isolation, and cost jointly meet the delivery conditions? | [10 Chips and Software Stacks](./10_Domestic_AI_Chips_Overview.ipynb) → [34 Heterogeneous Accelerators and Runtime](./34_Heterogeneous_Accelerators_and_Runtime_Constraints.ipynb) | Admission conditions for deployment and heterogeneous systems |

## Topic Connections

Part 01 first gives the theoretical ledger, resource upper bounds, and execution conditions. The foundational support topics organize these objects into analysis methods, and the mainline and lateral topics then choose mechanisms and verify real gains accordingly.

| Object taken from Part 01 | Enter first | Later use |
|:---|:---|:---|
| Data types, training-state bytes, and training compute volume | [Backpropagation and Training Mechanics](../topic_discussion/backpropagation_training_mechanism/intro.md) | [Post-Training Optimization](../topic_discussion/post_training_optimization/intro.md) |
| Ledger of parameter count, active compute, KV state, and sparse structure | [LLM Architecture Evolution](../topic_discussion/llm_architecture_evolution/intro.md) | [Inference Optimization](../topic_discussion/inference_optimization/intro.md), [Communication and Parallelism](../topic_discussion/communication_parallelism/intro.md) |
| Assumptions for compute, bandwidth, capacity, and the critical path | [Performance Analysis](../topic_discussion/performance_analysis/intro.md) | [Performance Optimization](../topic_discussion/performance_optimization/intro.md) |
| Quantized representation, accumulation precision, and low-precision execution conditions | [Quantization and Compression](../topic_discussion/quantization_compression/intro.md) | [Inference Optimization](../topic_discussion/inference_optimization/intro.md), [Deployment and Heterogeneous Systems](../topic_discussion/deployment_heterogeneous/intro.md) |
| Execution hierarchy, matrix path, on-chip reuse, and graph transformation conditions | [Operator Optimization](../topic_discussion/operator_optimization/intro.md) | [Performance Optimization](../topic_discussion/performance_optimization/intro.md) |
| Device capabilities, backend support, isolation, and cost constraints | [Deployment and Heterogeneous Systems](../topic_discussion/deployment_heterogeneous/intro.md) | Loading, serving, and regression verification on the target device |

## Environment Notes

- **Default environment:** Use [`BASE-CPU`](../docs/guide.md#base-cpu) to run the resource-ledger and mechanism exercises.
- **Extended experiments:** For content that needs a GPU, a compiler toolchain, or a multi-GPU runtime, prepare the environment according to the corresponding topic Quickstart.

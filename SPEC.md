# SPEC.md — English Translation Specification

> **Status:** bootstrap specification. This file is the authoritative scope
> and batch definition for the translation workstream. Keep it in sync with
> `PROGRESS.md` if the inventory changes.

## 1. Purpose

Define exactly what will be translated, how the work is divided into
milestones and batches, which files are excluded, and how a batch is
validated. Read `AGENTS.md` first for the translation rules.

## 2. Locked decisions

| Decision | Choice |
| --- | --- |
| Docs tree | Translate source trees, then regenerate `docs/` with the existing tools |
| Notebook H1s | Leave numbered `# NN. English \| Chinese` H1s in Parts 00–04 untouched; translate Part 05 notebook H1s |
| Code cells | Translate comments and docstrings only; leave other strings and logic alone |
| Homepage | Exclude `README.md` and generated `docs/index.md` |
| Generated boilerplate | Accept the small Chinese cloud-run banner in regenerated docs |

## 3. Scope

### 3.1 In-scope inventory

| Group | Files | CJK characters |
| --- | ---: | ---: |
| Part 00 — Prerequisites | 26 | 24,903 |
| Part 01 — Hardware, Math, and Systems | 41 | 55,676 |
| Part 02 — PyTorch Algorithms | 136 | 413,874 |
| Part 03 — Triton Kernels | 26 | 53,581 |
| Part 04 — CUDA and System Optimization | 25 | 51,401 |
| Part 05 — Integrated Projects and Decisions | 13 | 8,394 |
| Topic discussions | 180 | 240,702 |
| Team study | 6 | 2,994 |
| Docs-only pages | 14 | 25,810 |
| Ancillary development/environment docs | 5 | 3,487 |
| **Total** | **472** | **880,822** |

CJK counts include markdown prose and code-cell comments/docstrings/strings
as they exist today. They are an effort estimate, not a translation
acceptance metric.

### 3.2 Explicitly out of scope

| Path / pattern | Reason |
| --- | --- |
| `README.md` | Excluded by decision |
| `docs/index.md` | Generated from `README.md` by `tools/sync_docs_index.py` |
| `docs/00_Prerequisites/` … `docs/04_CUDA_and_System_Optimization/` | Generated mirrors; rebuild from sources |
| `docs/topic_discussion/` | Generated mirror; rebuild from `topic_discussion/` |
| `docs/team_study/` | Generated mirror; rebuild from `team_study/` |
| All `docs/05_*/` mirror pages | Generated from Part 05 sources during final QA |
| `docs/public/**` images and SVGs | Image assets; only alt text/captions in Markdown are translated |
| `.ipynb_checkpoints/` | Local checkpoint artifacts |
| `tools/*.py`, `verify.py`, `.github/` | Code and CI |
| `src/`, `pyproject.toml`, `uv.lock`, `.python-version` | Local/user workspace, not tutorial content |
| `docs/.vitepress/**`, `package.json`, `package-lock.json` | Site code and configuration |
| `requirements*`, `environment.yml`, `.cnb.yml` | Environment configuration |
| `benchmarks/artifacts/**/README.md` | Model cards, already English/zero CJK |
| `benchmarks/results/legacy_part02/README.md` | Already English |

### 3.3 Source-to-docs pipeline

| Source | Generated output | Tool |
| --- | --- | --- |
| Parts 00–04 notebooks and prose | `docs/<part>/` mirrors | `tools/convert_notebook.py` |
| Intro/group pages, `topic_discussion/`, `team_study/` | `docs/` mirrors | `tools/sync_docs_navigation.py` |
| `README.md` | `docs/index.md` | `tools/sync_docs_index.py` — **do not run** |
| Part 05 sources | `docs/05_*/` sub-pages | One-off copy/conversion during final QA |

## 4. Milestone overview

| ID | Milestone | Scope | Files | CJK | Batches |
| --- | --- | --- | ---: | ---: | ---: |
| M0 | Bootstrap and dry run | Protocol only | 0 | — | 1 |
| M1 | Part 00 — Prerequisites | `00_Prerequisites/` | 26 | 24,903 | 3 |
| M2 | Part 01 — Hardware, Math, and Systems | `01_Hardware_Math_and_Systems/` | 41 | 55,676 | 4 |
| M3 | Part 02 — PyTorch Algorithms | `02_PyTorch_Algorithms/` | 136 | 413,874 | 11 |
| M4 | Part 03 — Triton Kernels | `03_Triton_Kernels/` | 26 | 53,581 | 3 |
| M5 | Part 04 — CUDA and System Optimization | `04_CUDA_and_System_Optimization/` | 25 | 51,401 | 3 |
| M6 | Part 05 — Integrated Projects and Decisions | `05_Integrated_Projects_and_Decisions/` | 13 | 8,394 | 1 |
| M7 | Topic discussions | `topic_discussion/` | 180 | 240,702 | 16 |
| M8 | Team study | `team_study/` | 6 | 2,994 | 1 |
| M9 | Docs-only pages | `docs/` standalone pages | 14 | 25,810 | 2 |
| M10 | Ancillary development and environment docs (optional) | `tools/`, `cnb/` | 5 | 3,487 | 1 |
| M11 | Regeneration and final QA | Derived docs + checks | 0 | — | 1 |
| **Total** | | | **472** | **880,822** | |

M0 is a protocol dry run on `00_Prerequisites/intro.md` and
`00_Prerequisites/01_Python_Essentials_for_LLM.ipynb`; both files are counted
in M1, not twice. M11 produces no new translations; it rebuilds the mirrors
and runs the final checks.

## 5. Detailed milestones and batches

Each batch is sized for one agent session: roughly 10–15 notebooks or
2,000–45,000 CJK characters. Batch IDs are stable; use them in commits and
in `PROGRESS.md`.

### M1 — Part 00 — Prerequisites

**Source:** `00_Prerequisites/`  
**Files:** 26  
**CJK:** 24,903

| Batch | Files | CJK | Contents |
| --- | ---: | ---: | --- |
| `00-P` | 6 | 2,705 | Prose and navigation |
| `00-1` | 10 | 8,690 | Notebooks 01–10 |
| `00-2` | 10 | 13,508 | Notebooks 11–20 |

<details>
<summary><strong>00-P</strong> — 6 files, 2,705 CJK</summary>

- `00_Prerequisites/intro.md`
- `00_Prerequisites/0A.md`
- `00_Prerequisites/0B.md`
- `00_Prerequisites/0C.md`
- `00_Prerequisites/0D.md`
- `00_Prerequisites/0E.md`

</details>

<details>
<summary><strong>00-1</strong> — 10 files, 8,690 CJK</summary>

- `00_Prerequisites/01_Python_Essentials_for_LLM.ipynb`
- `00_Prerequisites/02_NumPy_and_Einsum.ipynb`
- `00_Prerequisites/03_Python_OOP_and_Utility_Patterns.ipynb`
- `00_Prerequisites/04_Python_Config_and_Data_Entry.ipynb`
- `00_Prerequisites/05_PyTorch_Tensor_Fundamentals.ipynb`
- `00_Prerequisites/06_PyTorch_Tensor_Layout_and_Indexing.ipynb`
- `00_Prerequisites/07_PyTorch_Autograd_and_Backward.ipynb`
- `00_Prerequisites/08_PyTorch_Grad_Hygiene_and_No_Grad.ipynb`
- `00_Prerequisites/09_PyTorch_nn_Module_Basics.ipynb`
- `00_Prerequisites/10_PyTorch_State_dict_and_Persistence.ipynb`

</details>

<details>
<summary><strong>00-2</strong> — 10 files, 13,508 CJK</summary>

- `00_Prerequisites/11_PyTorch_Optimizers_and_Loss.ipynb`
- `00_Prerequisites/12_PyTorch_Minimal_Training_Interface.ipynb`
- `00_Prerequisites/13_Transformer_Block_Dataflow_and_Residual_Stream.ipynb`
- `00_Prerequisites/14_Activation_Functions.ipynb`
- `00_Prerequisites/15_Normalization_Techniques.ipynb`
- `00_Prerequisites/16_Attention_Mechanism_Intro.ipynb`
- `00_Prerequisites/17_PyTorch_Profiling_Basics.ipynb`
- `00_Prerequisites/18_Memory_Profiling_and_Optimization.ipynb`
- `00_Prerequisites/19_Debugging_and_Anomaly_Localization.ipynb`
- `00_Prerequisites/20_Performance_Evidence_and_Experiment_Design.ipynb`

</details>

### M2 — Part 01 — Hardware, Math, and Systems

**Source:** `01_Hardware_Math_and_Systems/`  
**Files:** 41  
**CJK:** 55,676

| Batch | Files | CJK | Contents |
| --- | ---: | ---: | --- |
| `01-P` | 7 | 4,438 | Prose and navigation |
| `01-A` | 13 | 23,921 | Groups 1A + 1B |
| `01-B` | 13 | 17,056 | Groups 1C + 1D |
| `01-C` | 8 | 10,261 | Groups 1E + 1F |

<details>
<summary><strong>01-P</strong> — 7 files, 4,438 CJK</summary>

- `01_Hardware_Math_and_Systems/intro.md`
- `01_Hardware_Math_and_Systems/1A.md`
- `01_Hardware_Math_and_Systems/1B.md`
- `01_Hardware_Math_and_Systems/1C.md`
- `01_Hardware_Math_and_Systems/1D.md`
- `01_Hardware_Math_and_Systems/1E.md`
- `01_Hardware_Math_and_Systems/1F.md`

</details>

<details>
<summary><strong>01-A</strong> — 13 files, 23,921 CJK</summary>

- `01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.ipynb`
- `01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.ipynb`
- `01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb`
- `01_Hardware_Math_and_Systems/22_MoE_Parameter_and_Compute.ipynb`
- `01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb`
- `01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization.ipynb`
- `01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.ipynb`
- `01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.ipynb`
- `01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.ipynb`
- `01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.ipynb`
- `01_Hardware_Math_and_Systems/23_TensorCore_Deep_Dive.ipynb`
- `01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.ipynb`
- `01_Hardware_Math_and_Systems/25_Sparse_Computation_and_Sparse_Attention.ipynb`

</details>

<details>
<summary><strong>01-B</strong> — 13 files, 17,056 CJK</summary>

- `01_Hardware_Math_and_Systems/05_Communication_Topologies.ipynb`
- `01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.ipynb`
- `01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.ipynb`
- `01_Hardware_Math_and_Systems/26_Parallel_Strategy_Decision_Framework.ipynb`
- `01_Hardware_Math_and_Systems/27_Communication_Scheduling_Optimization.ipynb`
- `01_Hardware_Math_and_Systems/28_Fault_Tolerance_and_Checkpointing.ipynb`
- `01_Hardware_Math_and_Systems/07_CPU_GPU_Heterogeneous_Scheduling.ipynb`
- `01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton.ipynb`
- `01_Hardware_Math_and_Systems/15_CUDA_Execution_Model.ipynb`
- `01_Hardware_Math_and_Systems/16_Warp_Block_SharedMemory_Basics.ipynb`
- `01_Hardware_Math_and_Systems/18_Triton_Block_Model.ipynb`
- `01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony.ipynb`
- `01_Hardware_Math_and_Systems/29_CUDA_Stream_Advanced_Scheduling.ipynb`

</details>

<details>
<summary><strong>01-C</strong> — 8 files, 10,261 CJK</summary>

- `01_Hardware_Math_and_Systems/09_AI_Compilers_and_Graph_Optimization.ipynb`
- `01_Hardware_Math_and_Systems/32_TVM_MLIR_Deep_Practice.ipynb`
- `01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.ipynb`
- `01_Hardware_Math_and_Systems/30_Dynamic_Shape_Handling.ipynb`
- `01_Hardware_Math_and_Systems/10_Domestic_AI_Chips_Overview.ipynb`
- `01_Hardware_Math_and_Systems/34_Heterogeneous_Accelerators_and_Runtime_Constraints.ipynb`
- `01_Hardware_Math_and_Systems/31_GPU_Virtualization_and_MIG.ipynb`
- `01_Hardware_Math_and_Systems/33_TCO_and_Cost_Model.ipynb`

</details>

### M3 — Part 02 — PyTorch Algorithms

**Source:** `02_PyTorch_Algorithms/`  
**Files:** 136  
**CJK:** 413,874

| Batch | Files | CJK | Contents |
| --- | ---: | ---: | --- |
| `02-P` | 11 | 9,760 | Prose and navigation |
| `02-1` | 15 | 37,146 | Notebooks 00–14 |
| `02-2` | 11 | 39,978 | Notebooks 15–25 |
| `02-3` | 11 | 39,405 | Notebooks 26–36 |
| `02-4` | 12 | 40,835 | Notebooks 37–48 |
| `02-5` | 14 | 39,173 | Notebooks 49–62 |
| `02-6` | 10 | 41,526 | Notebooks 63–72 |
| `02-7` | 14 | 41,537 | Notebooks 73–86 |
| `02-8` | 14 | 40,476 | Reserved tail + ARCH/DISTILL/INF-A…P |
| `02-9` | 14 | 44,997 | INF-R…S + PAR + PERF-C…R + QUANT-QAT + TRAIN-MIXED |
| `02-10` | 10 | 39,041 | PERF-TRACE + PT-* + QUANT-MIXED-BIT |

<details>
<summary><strong>02-P</strong> — 11 files, 9,760 CJK</summary>

- `02_PyTorch_Algorithms/intro.md`
- `02_PyTorch_Algorithms/2_1.md`
- `02_PyTorch_Algorithms/2_2.md`
- `02_PyTorch_Algorithms/2_3.md`
- `02_PyTorch_Algorithms/2_4.md`
- `02_PyTorch_Algorithms/2_5.md`
- `02_PyTorch_Algorithms/2_6.md`
- `02_PyTorch_Algorithms/2_7.md`
- `02_PyTorch_Algorithms/2_8.md`
- `02_PyTorch_Algorithms/2_9.md`
- `02_PyTorch_Algorithms/2_10.md`

</details>

<details>
<summary><strong>02-1</strong> — 15 files, 37,146 CJK</summary>

- `02_PyTorch_Algorithms/00_PyTorch_Warmup.ipynb`
- `02_PyTorch_Algorithms/01_RMSNorm_Tutorial.ipynb`
- `02_PyTorch_Algorithms/02_SwiGLU_Activation.ipynb`
- `02_PyTorch_Algorithms/03_RoPE_Tutorial.ipynb`
- `02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb`
- `02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb`
- `02_PyTorch_Algorithms/06_MoE_Router.ipynb`
- `02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss.ipynb`
- `02_PyTorch_Algorithms/08_Architecture_Tricks.ipynb`
- `02_PyTorch_Algorithms/09_Activation_and_Loss_Backward.ipynb`
- `02_PyTorch_Algorithms/10_Attention_Backward_and_Custom_Autograd.ipynb`
- `02_PyTorch_Algorithms/11_Gradient_Accumulation.ipynb`
- `02_PyTorch_Algorithms/12_Optimizer_Updates_and_Learning_Rate_Scheduling.ipynb`
- `02_PyTorch_Algorithms/13_SFT_Training_Loop.ipynb`
- `02_PyTorch_Algorithms/14_LoRA_Tutorial.ipynb`

</details>

<details>
<summary><strong>02-2</strong> — 11 files, 39,978 CJK</summary>

- `02_PyTorch_Algorithms/15_End_to_End_Fine_Tuning_Experiment.ipynb`
- `02_PyTorch_Algorithms/16_RLHF_PPO_Memory.ipynb`
- `02_PyTorch_Algorithms/17_DPO_Loss_Tutorial.ipynb`
- `02_PyTorch_Algorithms/18_GRPO_Loss_Tutorial.ipynb`
- `02_PyTorch_Algorithms/19_Activation_Checkpointing.ipynb`
- `02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb`
- `02_PyTorch_Algorithms/21_Decoding_Strategies.ipynb`
- `02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb`
- `02_PyTorch_Algorithms/23_Speculative_Decoding.ipynb`
- `02_PyTorch_Algorithms/24_Prefix_Cache_Matching_and_Reuse.ipynb`
- `02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb`

</details>

<details>
<summary><strong>02-3</strong> — 11 files, 39,405 CJK</summary>

- `02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.ipynb`
- `02_PyTorch_Algorithms/27_ZeRO_Optimizer_Sim.ipynb`
- `02_PyTorch_Algorithms/28_Pipeline_Parallelism_MicroBatch.ipynb`
- `02_PyTorch_Algorithms/29_Tensor_Parallelism_Sim.ipynb`
- `02_PyTorch_Algorithms/30_Long_Context_Fine_Tuning.ipynb`
- `02_PyTorch_Algorithms/31_LoRA_Variants_Theory.ipynb`
- `02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.ipynb`
- `02_PyTorch_Algorithms/33_Fine_Tuning_Readiness.ipynb`
- `02_PyTorch_Algorithms/34_SGLang_RadixAttention.ipynb`
- `02_PyTorch_Algorithms/35_Multi_Token_Decoding.ipynb`
- `02_PyTorch_Algorithms/36_Decode_Scheduling.ipynb`

</details>

<details>
<summary><strong>02-4</strong> — 12 files, 40,835 CJK</summary>

- `02_PyTorch_Algorithms/37_KV_Cache_Lifecycle_and_Capacity_Control.ipynb`
- `02_PyTorch_Algorithms/38_Prefill_Decode_Scheduling.ipynb`
- `02_PyTorch_Algorithms/39_Hetero_PD_and_Serving_Tiers.ipynb`
- `02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb`
- `02_PyTorch_Algorithms/41_FP8_Runtime_Low_Precision_Paths.ipynb`
- `02_PyTorch_Algorithms/42_Activation_Offload.ipynb`
- `02_PyTorch_Algorithms/43_Unified_Training_Memory_Ledger.ipynb`
- `02_PyTorch_Algorithms/44_Local_Sliding_and_Sparse_Attention.ipynb`
- `02_PyTorch_Algorithms/45_Linear_Attention_and_Recurrent_State.ipynb`
- `02_PyTorch_Algorithms/46_MoE_Expert_Parallel.ipynb`
- `02_PyTorch_Algorithms/47_Communication_Profiling_with_NCCL.ipynb`
- `02_PyTorch_Algorithms/48_Communication_Hotspots_and_Mitigation.ipynb`

</details>

<details>
<summary><strong>02-5</strong> — 14 files, 39,173 CJK</summary>

- `02_PyTorch_Algorithms/49_Parallelism_Strategy_Selection.ipynb`
- `02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.ipynb`
- `02_PyTorch_Algorithms/51_Online_DPO.ipynb`
- `02_PyTorch_Algorithms/52_Quantization_Calibration_and_Error.ipynb`
- `02_PyTorch_Algorithms/53_Activation_Quantization_and_SmoothQuant.ipynb`
- `02_PyTorch_Algorithms/54_KV_Cache_Quantization_Strategies.ipynb`
- `02_PyTorch_Algorithms/55_Training_Update_Dynamics.ipynb`
- `02_PyTorch_Algorithms/56_Continual_Pretraining_and_Retention.ipynb`
- `02_PyTorch_Algorithms/57_Reserved_57.ipynb`
- `02_PyTorch_Algorithms/58_Reserved_58.ipynb`
- `02_PyTorch_Algorithms/59_Reserved_59.ipynb`
- `02_PyTorch_Algorithms/60_Decoder_Block_Stability_Benchmark.ipynb`
- `02_PyTorch_Algorithms/61_Model_Architecture_Exploration.ipynb`
- `02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.ipynb`

</details>

<details>
<summary><strong>02-6</strong> — 10 files, 41,526 CJK</summary>

- `02_PyTorch_Algorithms/63_LoRA_Variants_Benchmark.ipynb`
- `02_PyTorch_Algorithms/64_SFT_Data_Quality_Project.ipynb`
- `02_PyTorch_Algorithms/65_QLoRA_Selection_Project.ipynb`
- `02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb`
- `02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb`
- `02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.ipynb`
- `02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.ipynb`
- `02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.ipynb`
- `02_PyTorch_Algorithms/71_MLA_KV_Cache_Architecture_Benchmark.ipynb`
- `02_PyTorch_Algorithms/72_Multi_Model_Deployment_Comparison.ipynb`

</details>

<details>
<summary><strong>02-7</strong> — 14 files, 41,537 CJK</summary>

- `02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb`
- `02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb`
- `02_PyTorch_Algorithms/75_Optimizer_Execution_Path_Benchmark.ipynb`
- `02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb`
- `02_PyTorch_Algorithms/77_Long_Sequence_Memory_Architecture_Benchmark.ipynb`
- `02_PyTorch_Algorithms/78_Attention_Access_Pattern_Benchmark.ipynb`
- `02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.ipynb`
- `02_PyTorch_Algorithms/80_MoE_Expert_Parallel_Benchmark.ipynb`
- `02_PyTorch_Algorithms/81_Distributed_Inference_Project.ipynb`
- `02_PyTorch_Algorithms/82_Quantization_Artifact_Evaluation_Project.ipynb`
- `02_PyTorch_Algorithms/83_Quantized_Serving_Profile_Benchmark.ipynb`
- `02_PyTorch_Algorithms/84_DPO_Preference_Project.ipynb`
- `02_PyTorch_Algorithms/85_GRPO_Groupwise_Alignment_Project.ipynb`
- `02_PyTorch_Algorithms/86_DPO_Online_Benchmark.ipynb`

</details>

<details>
<summary><strong>02-8</strong> — 14 files, 40,476 CJK</summary>

- `02_PyTorch_Algorithms/87_Reserved_87.ipynb`
- `02_PyTorch_Algorithms/88_Distilled_Student_Delivery_Project.ipynb`
- `02_PyTorch_Algorithms/89_Reserved_89.ipynb`
- `02_PyTorch_Algorithms/ARCH-HYBRID-MEMORY_Attention_SSM_Hybrid.ipynb`
- `02_PyTorch_Algorithms/ARCH-MLA_Multihead_Latent_Attention.ipynb`
- `02_PyTorch_Algorithms/DISTILL-OPD_On_Policy_Distillation_and_GKD.ipynb`
- `02_PyTorch_Algorithms/INF-AFD_Attention_FFN_Disaggregation.ipynb`
- `02_PyTorch_Algorithms/INF-CONTINUOUS-BATCHING_Continuous_Batching.ipynb`
- `02_PyTorch_Algorithms/INF-EAGLE_Feature_Drafter_and_Candidate_Tree.ipynb`
- `02_PyTorch_Algorithms/INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch.ipynb`
- `02_PyTorch_Algorithms/INF-ITERATION-BUDGET_Serving_Iteration_Cost_and_Batch_Bounds.ipynb`
- `02_PyTorch_Algorithms/INF-MOE-EXECUTION_MoE_Inference_Dispatch_and_Expert_Execution.ipynb`
- `02_PyTorch_Algorithms/INF-PARALLEL-DRAFTING_DFlash_and_DSpark.ipynb`
- `02_PyTorch_Algorithms/INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.ipynb`

</details>

<details>
<summary><strong>02-9</strong> — 14 files, 44,997 CJK</summary>

- `02_PyTorch_Algorithms/INF-REQUEST-LIFECYCLE_Inference_Request_Lifecycle.ipynb`
- `02_PyTorch_Algorithms/INF-SERVING-OVERLAP_Serving_Overlap_and_State_Handoff.ipynb`
- `02_PyTorch_Algorithms/PAR-CONTEXT_Context_and_Sequence_Parallelism.ipynb`
- `02_PyTorch_Algorithms/PAR-MODEL-INTERNAL_Model_Internal_Parallel_Benchmark.ipynb`
- `02_PyTorch_Algorithms/PAR-REPLICA_Serving_Replica_Routing.ipynb`
- `02_PyTorch_Algorithms/PERF-CANDIDATE_Constraint_to_Candidate_and_Cost_Transfer.ipynb`
- `02_PyTorch_Algorithms/PERF-COMPOSITION_Strategy_Interaction_and_Bottleneck_Shift.ipynb`
- `02_PyTorch_Algorithms/PERF-INPUT_Training_Input_Pipeline_Performance.ipynb`
- `02_PyTorch_Algorithms/PERF-IO_Artifact_State_and_Output_IO_Critical_Path.ipynb`
- `02_PyTorch_Algorithms/PERF-MEMORY-TIMELINE_Memory_Residency_and_Peak_Overlap.ipynb`
- `02_PyTorch_Algorithms/PERF-REGRESSION_Paired_Benchmark_and_Regression_Thresholds.ipynb`
- `02_PyTorch_Algorithms/PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.ipynb`
- `02_PyTorch_Algorithms/QUANT-QAT_Quantization_Aware_Training.ipynb`
- `02_PyTorch_Algorithms/TRAIN-MIXED-PRECISION-BACKWARD_Mixed_Precision_Backward.ipynb`

</details>

<details>
<summary><strong>02-10</strong> — 10 files, 39,041 CJK</summary>

- `02_PyTorch_Algorithms/PERF-TRACE_Trace_Dependencies_and_Critical_Path.ipynb`
- `02_PyTorch_Algorithms/PT-AGENTIC-RL_Environment_Trajectory_and_Credit_Assignment.ipynb`
- `02_PyTorch_Algorithms/PT-KTO_Binary_Feedback_Alignment.ipynb`
- `02_PyTorch_Algorithms/PT-REWARD-MODEL_Reward_Model_Training_and_Calibration.ipynb`
- `02_PyTorch_Algorithms/PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.ipynb`
- `02_PyTorch_Algorithms/PT-RL-SYSTEM_Rollout_Training_and_Reward_System.ipynb`
- `02_PyTorch_Algorithms/PT-RLOO_Leave_One_Out_Policy_Optimization.ipynb`
- `02_PyTorch_Algorithms/PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.ipynb`
- `02_PyTorch_Algorithms/PT-SYNTHETIC-DATA_Instruction_Generation_and_Verifier_Admission.ipynb`
- `02_PyTorch_Algorithms/QUANT-MIXED-BIT_Mixed_Bit_Allocation.ipynb`

</details>

### M4 — Part 03 — Triton Kernels

**Source:** `03_Triton_Kernels/`  
**Files:** 26  
**CJK:** 53,581

| Batch | Files | CJK | Contents |
| --- | ---: | ---: | --- |
| `03-P` | 6 | 3,180 | Prose and navigation |
| `03-1` | 10 | 24,331 | Notebooks 01–10 |
| `03-2` | 10 | 26,070 | Notebooks 11–20 |

<details>
<summary><strong>03-P</strong> — 6 files, 3,180 CJK</summary>

- `03_Triton_Kernels/intro.md`
- `03_Triton_Kernels/3_1.md`
- `03_Triton_Kernels/3_2.md`
- `03_Triton_Kernels/3_3.md`
- `03_Triton_Kernels/3_4.md`
- `03_Triton_Kernels/3_5.md`

</details>

<details>
<summary><strong>03-1</strong> — 10 files, 24,331 CJK</summary>

- `03_Triton_Kernels/01_Triton_Vector_Addition.ipynb`
- `03_Triton_Kernels/02_Triton_Layouts_and_Memory_Access.ipynb`
- `03_Triton_Kernels/03_Triton_Fused_SwiGLU.ipynb`
- `03_Triton_Kernels/04_Triton_Fused_RMSNorm.ipynb`
- `03_Triton_Kernels/05_Triton_GEMM_Tutorial.ipynb`
- `03_Triton_Kernels/06_Triton_Persistent_GEMM_and_Tile_Scheduling.ipynb`
- `03_Triton_Kernels/07_Triton_Fused_Softmax.ipynb`
- `03_Triton_Kernels/08_Triton_Softmax_Backward.ipynb`
- `03_Triton_Kernels/09_Triton_Autotune_and_Profiling.ipynb`
- `03_Triton_Kernels/10_Triton_Fused_RoPE.ipynb`

</details>

<details>
<summary><strong>03-2</strong> — 10 files, 26,070 CJK</summary>

- `03_Triton_Kernels/11_Triton_Flash_Attention.ipynb`
- `03_Triton_Kernels/12_Triton_Variable_Length_GQA.ipynb`
- `03_Triton_Kernels/13_Triton_PagedAttention.ipynb`
- `03_Triton_Kernels/14_Triton_Quantization.ipynb`
- `03_Triton_Kernels/15_Triton_Multi_LoRA.ipynb`
- `03_Triton_Kernels/16_Triton_KV_Cache_Quantized_Decode.ipynb`
- `03_Triton_Kernels/17_Triton_Memory_Model_and_Debug.ipynb`
- `03_Triton_Kernels/18_Triton_Compilation_Artifacts_and_Execution.ipynb`
- `03_Triton_Kernels/19_Triton_PyTorch_Op_Integration.ipynb`
- `03_Triton_Kernels/20_Triton_Llama3_Block_Project.ipynb`

</details>

### M5 — Part 04 — CUDA and System Optimization

**Source:** `04_CUDA_and_System_Optimization/`  
**Files:** 25  
**CJK:** 51,401

| Batch | Files | CJK | Contents |
| --- | ---: | ---: | --- |
| `04-P` | 6 | 4,701 | Prose and navigation |
| `04-1` | 10 | 22,890 | Notebooks 01–10 |
| `04-2` | 9 | 23,810 | Notebooks 11–19 |

<details>
<summary><strong>04-P</strong> — 6 files, 4,701 CJK</summary>

- `04_CUDA_and_System_Optimization/intro.md`
- `04_CUDA_and_System_Optimization/4_1.md`
- `04_CUDA_and_System_Optimization/4_2.md`
- `04_CUDA_and_System_Optimization/4_3.md`
- `04_CUDA_and_System_Optimization/4_4.md`
- `04_CUDA_and_System_Optimization/4_5.md`

</details>

<details>
<summary><strong>04-1</strong> — 10 files, 22,890 CJK</summary>

- `04_CUDA_and_System_Optimization/01_CUDA_Custom_Kernel_Intro.ipynb`
- `04_CUDA_and_System_Optimization/02_CUDA_Global_Memory_Coalescing.ipynb`
- `04_CUDA_and_System_Optimization/03_Warp_Level_Primitives.ipynb`
- `04_CUDA_and_System_Optimization/04_Bank_Conflict_Deep_Dive.ipynb`
- `04_CUDA_and_System_Optimization/05_CUDA_Block_Resources_and_Residency.ipynb`
- `04_CUDA_and_System_Optimization/06_CUDA_Shared_Memory_Optimization.ipynb`
- `04_CUDA_and_System_Optimization/07_Tensor_Core_MMA_Programming.ipynb`
- `04_CUDA_and_System_Optimization/08_Async_Data_Prefetch_and_Double_Buffering.ipynb`
- `04_CUDA_and_System_Optimization/09_CUDA_Async_Copy_Pipeline.ipynb`
- `04_CUDA_and_System_Optimization/10_PyTorch_CUDA_Streams_and_Transfer.ipynb`

</details>

<details>
<summary><strong>04-2</strong> — 9 files, 23,810 CJK</summary>

- `04_CUDA_and_System_Optimization/11_CUDA_Graph_and_JIT_Compile.ipynb`
- `04_CUDA_and_System_Optimization/12_Memory_Pool_and_VRAM_Management.ipynb`
- `04_CUDA_and_System_Optimization/13_Distributed_Communication_Primitives.ipynb`
- `04_CUDA_and_System_Optimization/14_Ring_AllReduce_Deep_Dive.ipynb`
- `04_CUDA_and_System_Optimization/15_Communication_Computation_Overlap_Advanced_Scheduling.ipynb`
- `04_CUDA_and_System_Optimization/16_DeepSpeed_Zero_Config.ipynb`
- `04_CUDA_and_System_Optimization/17_Heterogeneous_Training_CPU_Offload_NVMe_Offload.ipynb`
- `04_CUDA_and_System_Optimization/18_CUDA_vs_Triton_vs_PyTorch.ipynb`
- `04_CUDA_and_System_Optimization/19_TCO_and_Hardware_Selection.ipynb`

</details>

### M6 — Part 05 — Integrated Projects and Decisions

**Source:** `05_Integrated_Projects_and_Decisions/`  
**Files:** 13  
**CJK:** 8,394

| Batch | Files | CJK | Contents |
| --- | ---: | ---: | --- |
| `05-1` | 13 | 8,394 | All project pages and notebooks |

<details>
<summary><strong>05-1</strong> — 13 files, 8,394 CJK</summary>

- `05_Integrated_Projects_and_Decisions/01_Inference_Service/Inference_Service_Project.ipynb`
- `05_Integrated_Projects_and_Decisions/01_Inference_Service/index.md`
- `05_Integrated_Projects_and_Decisions/01_Inference_Service_Integrated_Project.md`
- `05_Integrated_Projects_and_Decisions/02_Post_Training_Delivery/Post_Training_Delivery_Project.ipynb`
- `05_Integrated_Projects_and_Decisions/02_Post_Training_Delivery/index.md`
- `05_Integrated_Projects_and_Decisions/02_Post_Training_Delivery_Project.md`
- `05_Integrated_Projects_and_Decisions/03_Performance_Optimization_Delivery/Performance_Optimization_Delivery_Project.ipynb`
- `05_Integrated_Projects_and_Decisions/03_Performance_Optimization_Delivery/index.md`
- `05_Integrated_Projects_and_Decisions/04_Quantization_Delivery/Quantization_Delivery_Project.ipynb`
- `05_Integrated_Projects_and_Decisions/04_Quantization_Delivery/index.md`
- `05_Integrated_Projects_and_Decisions/intro.md`
- `05_Integrated_Projects_and_Decisions/open_practice/_template/README.md`
- `05_Integrated_Projects_and_Decisions/open_practice/_template/index.md`

</details>

### M7 — Topic discussions

**Source:** `topic_discussion/`  
**Files:** 180  
**CJK:** 240,702

| Batch | Files | CJK | Contents |
| --- | ---: | ---: | --- |
| `07-0` | 1 | 2,795 | Topic overview |
| `07-01` | 8 | 6,611 | Topic: agent |
| `07-02` | 13 | 15,386 | Topic: backpropagation_training_mechanism |
| `07-03` | 13 | 11,388 | Topic: communication_parallelism |
| `07-04` | 12 | 9,598 | Topic: data_engineering |
| `07-05` | 14 | 12,822 | Topic: deployment_heterogeneous |
| `07-06` | 17 | 35,435 | Topic: inference_optimization |
| `07-07` | 16 | 17,158 | Topic: llm_architecture_evolution |
| `07-08` | 1 | 299 | Topic: memory_performance_tuning |
| `07-09` | 2 | 4,034 | Topic: model_architecture_and_training |
| `07-10` | 12 | 7,279 | Topic: multimodal |
| `07-11` | 14 | 33,945 | Topic: operator_optimization |
| `07-12` | 14 | 16,368 | Topic: performance_analysis |
| `07-13` | 13 | 18,290 | Topic: performance_optimization |
| `07-14` | 17 | 31,849 | Topic: post_training_optimization |
| `07-15` | 13 | 17,445 | Topic: quantization_compression |

<details>
<summary><strong>07-0</strong> — 1 files, 2,795 CJK</summary>

- `topic_discussion/intro.md`

</details>

<details>
<summary><strong>07-01</strong> — 8 files, 6,611 CJK</summary>

- `topic_discussion/agent/00_agent_workload_and_trace.md`
- `topic_discussion/agent/01_agent_trajectory_data.md`
- `topic_discussion/agent/02_agent_sandbox_environment.md`
- `topic_discussion/agent/casebook.md`
- `topic_discussion/agent/experiment_assets.md`
- `topic_discussion/agent/glossary.md`
- `topic_discussion/agent/intro.md`
- `topic_discussion/agent/walkthrough.md`

</details>

<details>
<summary><strong>07-02</strong> — 13 files, 15,386 CJK</summary>

- `topic_discussion/backpropagation_training_mechanism/01_training_loop_and_computation_graph.md`
- `topic_discussion/backpropagation_training_mechanism/02_autograd_and_transformer_backward.md`
- `topic_discussion/backpropagation_training_mechanism/03_loss_reduction_gradient_accumulation_effective_update.md`
- `topic_discussion/backpropagation_training_mechanism/04_training_state_lifecycle.md`
- `topic_discussion/backpropagation_training_mechanism/05_checkpointing_and_offload.md`
- `topic_discussion/backpropagation_training_mechanism/06_correctness_and_performance_evidence.md`
- `topic_discussion/backpropagation_training_mechanism/casebook.md`
- `topic_discussion/backpropagation_training_mechanism/experiment_assets.md`
- `topic_discussion/backpropagation_training_mechanism/formula_reference.md`
- `topic_discussion/backpropagation_training_mechanism/glossary.md`
- `topic_discussion/backpropagation_training_mechanism/intro.md`
- `topic_discussion/backpropagation_training_mechanism/quickstart.md`
- `topic_discussion/backpropagation_training_mechanism/walkthrough.md`

</details>

<details>
<summary><strong>07-03</strong> — 13 files, 11,388 CJK</summary>

- `topic_discussion/communication_parallelism/01_why_parallel_and_communication.md`
- `topic_discussion/communication_parallelism/02_data_parallel_and_synchronization.md`
- `topic_discussion/communication_parallelism/03_state_sharding_and_zero.md`
- `topic_discussion/communication_parallelism/04_pipeline_and_tensor_parallel.md`
- `topic_discussion/communication_parallelism/05_expert_parallel_and_communication_hotspots.md`
- `topic_discussion/communication_parallelism/06_communication_profiling_and_overlap.md`
- `topic_discussion/communication_parallelism/07_benchmark_and_parallel_decision.md`
- `topic_discussion/communication_parallelism/casebook.md`
- `topic_discussion/communication_parallelism/experiment_assets.md`
- `topic_discussion/communication_parallelism/glossary.md`
- `topic_discussion/communication_parallelism/intro.md`
- `topic_discussion/communication_parallelism/quickstart.md`
- `topic_discussion/communication_parallelism/walkthrough.md`

</details>

<details>
<summary><strong>07-04</strong> — 12 files, 9,598 CJK</summary>

- `topic_discussion/data_engineering/00_data_lifecycle_and_contract.md`
- `topic_discussion/data_engineering/01_collection_parsing_contract.md`
- `topic_discussion/data_engineering/02_cleaning_quality_chain.md`
- `topic_discussion/data_engineering/03_data_mixture_and_curriculum.md`
- `topic_discussion/data_engineering/04_synthetic_data_and_trajectory_validation.md`
- `topic_discussion/data_engineering/05_post_training_data_interfaces.md`
- `topic_discussion/data_engineering/06_pipeline_lineage_and_downstream_validation.md`
- `topic_discussion/data_engineering/casebook.md`
- `topic_discussion/data_engineering/experiment_assets.md`
- `topic_discussion/data_engineering/glossary.md`
- `topic_discussion/data_engineering/intro.md`
- `topic_discussion/data_engineering/walkthrough.md`

</details>

<details>
<summary><strong>07-05</strong> — 14 files, 12,822 CJK</summary>

- `topic_discussion/deployment_heterogeneous/01_model_artifact_and_deployment_contract.md`
- `topic_discussion/deployment_heterogeneous/02_backend_compatibility_and_single_node_execution.md`
- `topic_discussion/deployment_heterogeneous/03_device_coordination_and_data_path.md`
- `topic_discussion/deployment_heterogeneous/04_multi_device_and_service_tiers.md`
- `topic_discussion/deployment_heterogeneous/05_observability_regression_and_release.md`
- `topic_discussion/deployment_heterogeneous/06_heterogeneous_selection_and_cost_decision.md`
- `topic_discussion/deployment_heterogeneous/07_accelerator_runtime_compatibility.md`
- `topic_discussion/deployment_heterogeneous/08_edge_cloud_placement_and_handoff.md`
- `topic_discussion/deployment_heterogeneous/casebook.md`
- `topic_discussion/deployment_heterogeneous/experiment_assets.md`
- `topic_discussion/deployment_heterogeneous/glossary.md`
- `topic_discussion/deployment_heterogeneous/intro.md`
- `topic_discussion/deployment_heterogeneous/quickstart.md`
- `topic_discussion/deployment_heterogeneous/walkthrough.md`

</details>

<details>
<summary><strong>07-06</strong> — 17 files, 35,435 CJK</summary>

- `topic_discussion/inference_optimization/01_request_path_and_metrics.md`
- `topic_discussion/inference_optimization/02_prefill_and_attention_kernel.md`
- `topic_discussion/inference_optimization/03_decoding_strategies.md`
- `topic_discussion/inference_optimization/04_kv_cache_lifecycle_and_reuse.md`
- `topic_discussion/inference_optimization/05_serving_scheduling_and_pd.md`
- `topic_discussion/inference_optimization/06_quantized_inference_and_deployment.md`
- `topic_discussion/inference_optimization/07_benchmark_and_decision.md`
- `topic_discussion/inference_optimization/08_distributed_inference_and_parallelism.md`
- `topic_discussion/inference_optimization/casebook.md`
- `topic_discussion/inference_optimization/experiment_assets.md`
- `topic_discussion/inference_optimization/glossary.md`
- `topic_discussion/inference_optimization/intro.md`
- `topic_discussion/inference_optimization/mini_sglang_cache_scheduling.md`
- `topic_discussion/inference_optimization/nano_vllm_request_walkthrough.md`
- `topic_discussion/inference_optimization/quickstart.md`
- `topic_discussion/inference_optimization/teaching_engines.md`
- `topic_discussion/inference_optimization/walkthrough.md`

</details>

<details>
<summary><strong>07-07</strong> — 16 files, 17,158 CJK</summary>

- `topic_discussion/llm_architecture_evolution/01_transformer_decoder.md`
- `topic_discussion/llm_architecture_evolution/02_tokenization_embedding.md`
- `topic_discussion/llm_architecture_evolution/03_norm_evolution.md`
- `topic_discussion/llm_architecture_evolution/04_attention_evolution.md`
- `topic_discussion/llm_architecture_evolution/05_rope_position_encoding.md`
- `topic_discussion/llm_architecture_evolution/06_block_residual_path.md`
- `topic_discussion/llm_architecture_evolution/07_mlp_ffn_evolution.md`
- `topic_discussion/llm_architecture_evolution/08_representative_models.md`
- `topic_discussion/llm_architecture_evolution/09_moe_sparsity_evolution.md`
- `topic_discussion/llm_architecture_evolution/10_linear_attention_ssm_and_hybrid.md`
- `topic_discussion/llm_architecture_evolution/casebook.md`
- `topic_discussion/llm_architecture_evolution/experiment_assets.md`
- `topic_discussion/llm_architecture_evolution/glossary.md`
- `topic_discussion/llm_architecture_evolution/intro.md`
- `topic_discussion/llm_architecture_evolution/quickstart.md`
- `topic_discussion/llm_architecture_evolution/walkthrough.md`

</details>

<details>
<summary><strong>07-08</strong> — 1 files, 299 CJK</summary>

- `topic_discussion/memory_performance_tuning/intro.md`

</details>

<details>
<summary><strong>07-09</strong> — 2 files, 4,034 CJK</summary>

- `topic_discussion/model_architecture_and_training/intro.md`
- `topic_discussion/model_architecture_and_training/walkthrough.md`

</details>

<details>
<summary><strong>07-10</strong> — 12 files, 7,279 CJK</summary>

- `topic_discussion/multimodal/00_visual_representation_cnn_to_vit.md`
- `topic_discussion/multimodal/01_detection_yolo_to_detr.md`
- `topic_discussion/multimodal/02_realtime_detr_evolution.md`
- `topic_discussion/multimodal/03_vlm_architecture_and_input_contract.md`
- `topic_discussion/multimodal/04_vlm_workload_and_cache.md`
- `topic_discussion/multimodal/05_multimodal_profiling_and_deployment.md`
- `topic_discussion/multimodal/06_multimodal_delivery_decision.md`
- `topic_discussion/multimodal/casebook.md`
- `topic_discussion/multimodal/experiment_assets.md`
- `topic_discussion/multimodal/glossary.md`
- `topic_discussion/multimodal/intro.md`
- `topic_discussion/multimodal/walkthrough.md`

</details>

<details>
<summary><strong>07-11</strong> — 14 files, 33,945 CJK</summary>

- `topic_discussion/operator_optimization/01_optimization_targets_and_baselines.md`
- `topic_discussion/operator_optimization/02_graph_compilation_and_fusion.md`
- `topic_discussion/operator_optimization/03_kernel_semantics_and_memory.md`
- `topic_discussion/operator_optimization/04_attention_and_inference_data_paths.md`
- `topic_discussion/operator_optimization/05_measurement_profiling_and_autotuning.md`
- `topic_discussion/operator_optimization/06_cuda_execution_and_runtime_constraints.md`
- `topic_discussion/operator_optimization/07_integration_benchmarking_and_adoption.md`
- `topic_discussion/operator_optimization/casebook.md`
- `topic_discussion/operator_optimization/experiment_assets.md`
- `topic_discussion/operator_optimization/formula_reference.md`
- `topic_discussion/operator_optimization/glossary.md`
- `topic_discussion/operator_optimization/intro.md`
- `topic_discussion/operator_optimization/quickstart.md`
- `topic_discussion/operator_optimization/walkthrough.md`

</details>

<details>
<summary><strong>07-12</strong> — 14 files, 16,368 CJK</summary>

- `topic_discussion/performance_analysis/01_why_profiling_matters.md`
- `topic_discussion/performance_analysis/02_reproducible_baseline.md`
- `topic_discussion/performance_analysis/03_time_breakdown_and_trace_reading.md`
- `topic_discussion/performance_analysis/04_memory_timeline_and_residency.md`
- `topic_discussion/performance_analysis/05_bottleneck_attribution.md`
- `topic_discussion/performance_analysis/06_benchmark_design_and_regression_validation.md`
- `topic_discussion/performance_analysis/07_diagnosis_and_action_decision.md`
- `topic_discussion/performance_analysis/casebook.md`
- `topic_discussion/performance_analysis/experiment_assets.md`
- `topic_discussion/performance_analysis/formula_reference.md`
- `topic_discussion/performance_analysis/glossary.md`
- `topic_discussion/performance_analysis/intro.md`
- `topic_discussion/performance_analysis/quickstart.md`
- `topic_discussion/performance_analysis/walkthrough.md`

</details>

<details>
<summary><strong>07-13</strong> — 13 files, 18,290 CJK</summary>

- `topic_discussion/performance_optimization/01_performance_problem_and_benchmark.md`
- `topic_discussion/performance_optimization/02_observability_and_profiling.md`
- `topic_discussion/performance_optimization/03_training_memory_ledger.md`
- `topic_discussion/performance_optimization/04_inference_memory_and_kv_cache.md`
- `topic_discussion/performance_optimization/05_compute_communication_and_io.md`
- `topic_discussion/performance_optimization/06_cost_model_and_production_decision.md`
- `topic_discussion/performance_optimization/07_end_to_end_performance_engineering.md`
- `topic_discussion/performance_optimization/casebook.md`
- `topic_discussion/performance_optimization/experiment_assets.md`
- `topic_discussion/performance_optimization/glossary.md`
- `topic_discussion/performance_optimization/intro.md`
- `topic_discussion/performance_optimization/quickstart.md`
- `topic_discussion/performance_optimization/walkthrough.md`

</details>

<details>
<summary><strong>07-14</strong> — 17 files, 31,849 CJK</summary>

- `topic_discussion/post_training_optimization/01_sft_data_and_loss.md`
- `topic_discussion/post_training_optimization/02_lora_peft_design.md`
- `topic_discussion/post_training_optimization/03_training_control.md`
- `topic_discussion/post_training_optimization/04_end_to_end_experiment.md`
- `topic_discussion/post_training_optimization/05_dpo_and_preference_optimization.md`
- `topic_discussion/post_training_optimization/06_rlhf_and_ppo_system_cost.md`
- `topic_discussion/post_training_optimization/07_grpo_and_groupwise_alignment.md`
- `topic_discussion/post_training_optimization/08_trajectory_lifecycle_and_environment_interaction.md`
- `topic_discussion/post_training_optimization/09_rl_execution_system_and_efficiency.md`
- `topic_discussion/post_training_optimization/10_evaluation_and_adoption_decision.md`
- `topic_discussion/post_training_optimization/casebook.md`
- `topic_discussion/post_training_optimization/experiment_assets.md`
- `topic_discussion/post_training_optimization/formula_reference.md`
- `topic_discussion/post_training_optimization/glossary.md`
- `topic_discussion/post_training_optimization/intro.md`
- `topic_discussion/post_training_optimization/quickstart.md`
- `topic_discussion/post_training_optimization/walkthrough.md`

</details>

<details>
<summary><strong>07-15</strong> — 13 files, 17,445 CJK</summary>

- `topic_discussion/quantization_compression/01_quantization_object_and_error.md`
- `topic_discussion/quantization_compression/02_ptq_and_qat_timing.md`
- `topic_discussion/quantization_compression/03_low_bit_training_adaptation.md`
- `topic_discussion/quantization_compression/04_weight_only_compression.md`
- `topic_discussion/quantization_compression/05_fp8_and_kv_cache_quantization.md`
- `topic_discussion/quantization_compression/06_deployment_and_benchmark_decision.md`
- `topic_discussion/quantization_compression/07_pruning_distillation_and_structured_sparsity.md`
- `topic_discussion/quantization_compression/casebook.md`
- `topic_discussion/quantization_compression/experiment_assets.md`
- `topic_discussion/quantization_compression/glossary.md`
- `topic_discussion/quantization_compression/intro.md`
- `topic_discussion/quantization_compression/quickstart.md`
- `topic_discussion/quantization_compression/walkthrough.md`

</details>

### M8 — Team study

**Source:** `team_study/`  
**Files:** 6  
**CJK:** 2,994

| Batch | Files | CJK | Contents |
| --- | ---: | ---: | --- |
| `08-1` | 6 | 2,994 | All study-group pages |

<details>
<summary><strong>08-1</strong> — 6 files, 2,994 CJK</summary>

- `team_study/intro.md`
- `team_study/part2_l1_202606/group_topic_1.md`
- `team_study/part2_l1_202606/group_topic_2.md`
- `team_study/part2_l1_202606/intro.md`
- `team_study/part2_l1_202607/intro.md`
- `team_study/part2_l2_202607/intro.md`

</details>

### M9 — Docs-only pages

**Source:** `docs/` standalone pages  
**Files:** 14  
**CJK:** 25,810

| Batch | Files | CJK | Contents |
| --- | ---: | ---: | --- |
| `09-1` | 6 | 7,328 | Guides, maintenance, environment, CNB history |
| `09-2` | 8 | 18,482 | Registries, audits, verification, terminology |

<details>
<summary><strong>09-1</strong> — 6 files, 7,328 CJK</summary>

- `docs/guide.md`
- `docs/contributing.md`
- `docs/maintenance.md`
- `docs/template_guidelines.md`
- `docs/gpu_environment_assets.md`
- `docs/cnb/README.md`

</details>

<details>
<summary><strong>09-2</strong> — 8 files, 18,482 CJK</summary>

- `docs/part02_result_assets.md`
- `docs/part02_visual_assets_audit.md`
- `docs/topic_asset_registry.md`
- `docs/topic_discussion_visual_assets_audit.md`
- `docs/verification/fine_tuning_projects.md`
- `docs/verification/inference_projects.md`
- `docs/verification/memory_projects.md`
- `docs/terminology.md`

</details>

### M10 — Ancillary development and environment docs (optional)

**Source:** `tools/`, `cnb/`  
**Files:** 5  
**CJK:** 3,487

| Batch | Files | CJK | Contents |
| --- | ---: | ---: | --- |
| `10-1` | 5 | 3,487 | Tooling and environment docs |

<details>
<summary><strong>10-1</strong> — 5 files, 3,487 CJK</summary>

- `tools/README.md`
- `tools/diagram_style_guide.md`
- `tools/tests/README.md`
- `tools/_scratch/README.md`
- `cnb/README.md`

</details>

## 6. Generation pipeline

### 6.1 During a batch

Do not regenerate the whole docs tree. For a completed part batch, rebuild
only that part:

```bash
python tools/convert_notebook.py --dir 00_Prerequisites
python tools/sync_docs_navigation.py --part 00_Prerequisites
```

`--dir` writes/overwrites the target mirrors without deleting stale files.
For a topic:

```bash
python tools/sync_docs_navigation.py --topic <topic-name> --prune
```

### 6.2 Final QA rebuild

A no-argument `convert_notebook.py` run **deletes and rebuilds**
`docs/00_Prerequisites` through `docs/04_CUDA_and_System_Optimization`. Run it
only when you intend a full rebuild:

```bash
python tools/convert_notebook.py
python tools/sync_docs_navigation.py
```

Then produce the Part 05 sub-page mirrors once: copy the translated
`index.md` pages with the same link rewrites used by
`sync_docs_navigation.py`, and convert the four Part 05 notebooks with the
same Markdown-cell/code-cell logic as `convert_notebook.py`. Do not hand-edit
the generated copies afterwards.

### 6.3 Do not run

```bash
python tools/sync_docs_index.py   # overwrites docs/index.md from README.md
```

## 7. Validation and acceptance criteria

### 7.1 Per file

- [ ] Every Chinese prose passage is translated.
- [ ] Code-cell policy applied (comments + docstrings only).
- [ ] Structure, tables, headings, links, anchors, and image paths preserved.
- [ ] No unintended Chinese prose remains.
- [ ] Notebook JSON/schema valid; cell order, types, metadata, tags, and outputs preserved.
- [ ] Code-cell AST comparison shows no logic change.
- [ ] `PROGRESS.md` checkbox updated.

### 7.2 Per batch

- [ ] All files meet the per-file criteria.
- [ ] Residual-CJK scan reviewed and recorded.
- [ ] Affected docs mirrors rebuilt.
- [ ] `python tools/check_source_docs_mirror.py` passes.
- [ ] Link checks run; anchor changes manually verified.
- [ ] Dashboard and session log updated.

### 7.3 Final QA (M11)

- [ ] Full docs rebuild completed.
- [ ] Part 05 sub-page mirrors produced.
- [ ] No stale generated files remain.
- [ ] Residual-CJK scan across all in-scope files reviewed.
- [ ] `python tools/check_docs_links.py` passes (or its environment limitation is recorded).
- [ ] CPU-safe notebook tests pass; GPU-only tests are recorded as skipped.
- [ ] Spot review of at least one file per milestone by a second agent.
- [ ] `PROGRESS.md` shows every file as translated/reviewed or explicitly blocked.

## 8. Known quirks and limitations

1. **Generated mirrors.** Never translate `docs/00–04`, `docs/topic_discussion`,
   or `docs/team_study` directly. Fix the source and regenerate.
2. **`docs/index.md`.** It is generated from the excluded `README.md`; it will
   remain Chinese. Its Chinese anchor links must keep working, so preserve
   compatibility anchors on translated headings where needed.
3. **Notebook H1.** For Parts 00–04, leave the numbered
   `# NN. English | Chinese` line untouched; `verify.py` and the docs generator
   depend on it. Translate Part 05 notebook H1s, which are not numbered.
4. **Chinese validator expectations.** `audit_chapter0_1_notebooks.py`,
   `validate_notebook_format.py`, `test_framework_conformance.py`, and
   `check_mechanism_step4_tables.py` encode Chinese headings. Their
   heading/title failures are expected after translation; do not edit the
   tools to silence them.
5. **Stale docs files.** `docs/02_PyTorch_Algorithms/` contains stale generated
   pages with no source counterpart. The full rebuild removes them; do not
   translate those files.
6. **Part 05 mirrors.** Only `intro.md` and two summary pages are synced by
   the tooling. The sub-page `index.md` and four notebook mirrors need the
   one-off final-QA step.
7. **Chinese diagrams.** 611 of 649 SVGs under `docs/public/` contain Chinese
   text. They are out of scope for now; alt text and captions will be English,
   but the diagrams will remain Chinese.
8. **Cloud-run boilerplate.** `convert_notebook.py` injects a Chinese
   “cloud run” block into generated mirrors. This is accepted for now; do not
   hand-edit the generated output.
9. **Anchor checks.** `check_docs_links.py` ignores anchors. Any heading you
   translate must be manually checked for inbound anchor links.
10. **Terminology table.** When translating `docs/terminology.md` in M9, keep
    the Chinese and English term columns and the table structure intact.
    Translate only the explanatory prose; the bilingual mapping is the
    project-wide reference.
11. **Site chrome.** VitePress nav/sidebar labels in
    `docs/.vitepress/config.mts` are configuration, not Markdown, and remain
    Chinese in this phase.
12. **Landing page.** `README.md` and its generated mirror `docs/index.md`
    remain Chinese by decision. The English edition starts from the tutorial
    pages; the README already contains an English section.

## 9. Change control

If a file is added, removed, or moved during the translation effort:

1. Update the batch definition in this file.
2. Update the corresponding checklist in `PROGRESS.md`.
3. Keep the total file count and CJK estimate in sync.
4. Do not silently drop a file from a batch.


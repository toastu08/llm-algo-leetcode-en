# PROGRESS.md — English Translation Progress

> **Last updated:** 2026-11-24  
> **Overall:** 46 / 472 source files translated (9.7%)  
> **Current focus:** M2 Part 01 — `01-P` and `01-A` are complete; next is `01-B` (groups 1C + 1D notebooks)

This file is the single source of truth for translation status. Update it
after every file and every batch. Do not delete completed rows.

## Status legend

| Marker | Meaning |
| --- | --- |
| `[ ]` | Not started |
| `[~]` | In progress (optional; use for a long batch) |
| `[x]` | Translated and self-checked |
| `[x] · reviewed` | Translated and reviewed by a second agent |
| `[!]` | Blocked (append `· blocked: reason`) |
| `[-]` | N/A or derived (do not translate) |

## Dashboard

| ID | Milestone | Files | Translated | Reviewed | Blocked | Not started | % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 | Bootstrap and dry run | 0 | 0 | 0 | 0 | 0 | ✅ |
| M1 | Part 00 — Prerequisites | 26 | 26 | 0 | 0 | 0 | 100% |
| M2 | Part 01 — Hardware, Math, and Systems | 41 | 20 | 0 | 0 | 21 | 48.8% |
| M3 | Part 02 — PyTorch Algorithms | 136 | 0 | 0 | 0 | 136 | 0% |
| M4 | Part 03 — Triton Kernels | 26 | 0 | 0 | 0 | 26 | 0% |
| M5 | Part 04 — CUDA and System Optimization | 25 | 0 | 0 | 0 | 25 | 0% |
| M6 | Part 05 — Integrated Projects and Decisions | 13 | 0 | 0 | 0 | 13 | 0% |
| M7 | Topic discussions | 180 | 0 | 0 | 0 | 180 | 0% |
| M8 | Team study | 6 | 0 | 0 | 0 | 6 | 0% |
| M9 | Docs-only pages | 14 | 0 | 0 | 0 | 14 | 0% |
| M10 | Ancillary development and environment docs (optional) | 5 | 0 | 0 | 0 | 5 | 0% |
| M11 | Regeneration and final QA | 0 | 0 | 0 | 0 | 0 | — |
| **Total** | | **472** | **46** | **0** | **0** | **426** | **9.7%** |

Update the counts as you go. A file counts as “Translated” only when it
meets the definition of done in `AGENTS.md` §13. “Reviewed” means a second
agent checked it and confirmed the same criteria.

## Current focus

**`01-P` and `01-A` are complete.** The Part 01 prose pages and the 13 group 1A/1B notebooks are translated and mirrored.

Next focus:

- Translate batch `01-B` (groups 1C + 1D, 13 notebooks: communication, parallel, execution, and asynchrony).
- Then `01-C`, followed by the Part 01 checks and full mirror rebuild.

### M0 dry-run notes

- The numbered notebook H1 was left untouched, as required by `verify.py`.
- Display metadata labels (`Difficulty`, `Environment`, `Tags`, `Audience`,
  `Keywords`) and display tag values were translated; cell metadata and tags
  were not touched.
- Code docstrings were translated; print/assert string literals were left in
  Chinese per the comments+docstrings-only policy.
- Markdown headings, link targets, image paths, table rows, and cell structure
  were preserved.
- `intro.md` contains no remaining Chinese; the notebook's only Markdown CJK is
  the required H1 subtitle.
- A dry-run docs conversion in a temp directory confirmed that the generator
  keeps the H1, rewrites notebook links to `.md`, and emits the accepted
  Chinese cloud-run banner.

## Per-file checklists

Mark a file `[x]` only after translating, self-checking, and updating the
notes. Append `· reviewed`, `· blocked: reason`, or a short note as needed.

### M1 — Part 00 — Prerequisites (26 files, 24,903 CJK)

#### 00-P — Prose and navigation (6 files, 2,705 CJK)

- [x] `00_Prerequisites/intro.md`
- [x] `00_Prerequisites/0A.md`
- [x] `00_Prerequisites/0B.md`
- [x] `00_Prerequisites/0C.md`
- [x] `00_Prerequisites/0D.md`
- [x] `00_Prerequisites/0E.md`

#### 00-1 — Notebooks 01–10 (10 files, 8,690 CJK)

- [x] `00_Prerequisites/01_Python_Essentials_for_LLM.ipynb`
- [x] `00_Prerequisites/02_NumPy_and_Einsum.ipynb`
- [x] `00_Prerequisites/03_Python_OOP_and_Utility_Patterns.ipynb`
- [x] `00_Prerequisites/04_Python_Config_and_Data_Entry.ipynb`
- [x] `00_Prerequisites/05_PyTorch_Tensor_Fundamentals.ipynb`
- [x] `00_Prerequisites/06_PyTorch_Tensor_Layout_and_Indexing.ipynb`
- [x] `00_Prerequisites/07_PyTorch_Autograd_and_Backward.ipynb`
- [x] `00_Prerequisites/08_PyTorch_Grad_Hygiene_and_No_Grad.ipynb`
- [x] `00_Prerequisites/09_PyTorch_nn_Module_Basics.ipynb`
- [x] `00_Prerequisites/10_PyTorch_State_dict_and_Persistence.ipynb`

#### 00-2 — Notebooks 11–20 (10 files, 13,508 CJK)

- [x] `00_Prerequisites/11_PyTorch_Optimizers_and_Loss.ipynb`
- [x] `00_Prerequisites/12_PyTorch_Minimal_Training_Interface.ipynb`
- [x] `00_Prerequisites/13_Transformer_Block_Dataflow_and_Residual_Stream.ipynb`
- [x] `00_Prerequisites/14_Activation_Functions.ipynb`
- [x] `00_Prerequisites/15_Normalization_Techniques.ipynb`
- [x] `00_Prerequisites/16_Attention_Mechanism_Intro.ipynb`
- [x] `00_Prerequisites/17_PyTorch_Profiling_Basics.ipynb`
- [x] `00_Prerequisites/18_Memory_Profiling_and_Optimization.ipynb`
- [x] `00_Prerequisites/19_Debugging_and_Anomaly_Localization.ipynb`
- [x] `00_Prerequisites/20_Performance_Evidence_and_Experiment_Design.ipynb`

### M2 — Part 01 — Hardware, Math, and Systems (41 files, 55,676 CJK)

#### 01-P — Prose and navigation (7 files, 4,438 CJK)

- [x] `01_Hardware_Math_and_Systems/intro.md`
- [x] `01_Hardware_Math_and_Systems/1A.md`
- [x] `01_Hardware_Math_and_Systems/1B.md`
- [x] `01_Hardware_Math_and_Systems/1C.md`
- [x] `01_Hardware_Math_and_Systems/1D.md`
- [x] `01_Hardware_Math_and_Systems/1E.md`
- [x] `01_Hardware_Math_and_Systems/1F.md`

#### 01-A — Groups 1A + 1B (13 files, 23,921 CJK)

- [x] `01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.ipynb`
- [x] `01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.ipynb`
- [x] `01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb`
- [x] `01_Hardware_Math_and_Systems/22_MoE_Parameter_and_Compute.ipynb`
- [x] `01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb`
- [x] `01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization.ipynb`
- [x] `01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.ipynb`
- [x] `01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.ipynb`
- [x] `01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.ipynb`
- [x] `01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.ipynb`
- [x] `01_Hardware_Math_and_Systems/23_TensorCore_Deep_Dive.ipynb`
- [x] `01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.ipynb`
- [x] `01_Hardware_Math_and_Systems/25_Sparse_Computation_and_Sparse_Attention.ipynb`

#### 01-B — Groups 1C + 1D (13 files, 17,056 CJK)

- [ ] `01_Hardware_Math_and_Systems/05_Communication_Topologies.ipynb`
- [ ] `01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.ipynb`
- [ ] `01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.ipynb`
- [ ] `01_Hardware_Math_and_Systems/26_Parallel_Strategy_Decision_Framework.ipynb`
- [ ] `01_Hardware_Math_and_Systems/27_Communication_Scheduling_Optimization.ipynb`
- [ ] `01_Hardware_Math_and_Systems/28_Fault_Tolerance_and_Checkpointing.ipynb`
- [ ] `01_Hardware_Math_and_Systems/07_CPU_GPU_Heterogeneous_Scheduling.ipynb`
- [ ] `01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton.ipynb`
- [ ] `01_Hardware_Math_and_Systems/15_CUDA_Execution_Model.ipynb`
- [ ] `01_Hardware_Math_and_Systems/16_Warp_Block_SharedMemory_Basics.ipynb`
- [ ] `01_Hardware_Math_and_Systems/18_Triton_Block_Model.ipynb`
- [ ] `01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony.ipynb`
- [ ] `01_Hardware_Math_and_Systems/29_CUDA_Stream_Advanced_Scheduling.ipynb`

#### 01-C — Groups 1E + 1F (8 files, 10,261 CJK)

- [ ] `01_Hardware_Math_and_Systems/09_AI_Compilers_and_Graph_Optimization.ipynb`
- [ ] `01_Hardware_Math_and_Systems/32_TVM_MLIR_Deep_Practice.ipynb`
- [ ] `01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.ipynb`
- [ ] `01_Hardware_Math_and_Systems/30_Dynamic_Shape_Handling.ipynb`
- [ ] `01_Hardware_Math_and_Systems/10_Domestic_AI_Chips_Overview.ipynb`
- [ ] `01_Hardware_Math_and_Systems/34_Heterogeneous_Accelerators_and_Runtime_Constraints.ipynb`
- [ ] `01_Hardware_Math_and_Systems/31_GPU_Virtualization_and_MIG.ipynb`
- [ ] `01_Hardware_Math_and_Systems/33_TCO_and_Cost_Model.ipynb`

### M3 — Part 02 — PyTorch Algorithms (136 files, 413,874 CJK)

#### 02-P — Prose and navigation (11 files, 9,760 CJK)

- [ ] `02_PyTorch_Algorithms/intro.md`
- [ ] `02_PyTorch_Algorithms/2_1.md`
- [ ] `02_PyTorch_Algorithms/2_2.md`
- [ ] `02_PyTorch_Algorithms/2_3.md`
- [ ] `02_PyTorch_Algorithms/2_4.md`
- [ ] `02_PyTorch_Algorithms/2_5.md`
- [ ] `02_PyTorch_Algorithms/2_6.md`
- [ ] `02_PyTorch_Algorithms/2_7.md`
- [ ] `02_PyTorch_Algorithms/2_8.md`
- [ ] `02_PyTorch_Algorithms/2_9.md`
- [ ] `02_PyTorch_Algorithms/2_10.md`

#### 02-1 — Notebooks 00–14 (15 files, 37,146 CJK)

- [ ] `02_PyTorch_Algorithms/00_PyTorch_Warmup.ipynb`
- [ ] `02_PyTorch_Algorithms/01_RMSNorm_Tutorial.ipynb`
- [ ] `02_PyTorch_Algorithms/02_SwiGLU_Activation.ipynb`
- [ ] `02_PyTorch_Algorithms/03_RoPE_Tutorial.ipynb`
- [ ] `02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb`
- [ ] `02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb`
- [ ] `02_PyTorch_Algorithms/06_MoE_Router.ipynb`
- [ ] `02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss.ipynb`
- [ ] `02_PyTorch_Algorithms/08_Architecture_Tricks.ipynb`
- [ ] `02_PyTorch_Algorithms/09_Activation_and_Loss_Backward.ipynb`
- [ ] `02_PyTorch_Algorithms/10_Attention_Backward_and_Custom_Autograd.ipynb`
- [ ] `02_PyTorch_Algorithms/11_Gradient_Accumulation.ipynb`
- [ ] `02_PyTorch_Algorithms/12_Optimizer_Updates_and_Learning_Rate_Scheduling.ipynb`
- [ ] `02_PyTorch_Algorithms/13_SFT_Training_Loop.ipynb`
- [ ] `02_PyTorch_Algorithms/14_LoRA_Tutorial.ipynb`

#### 02-2 — Notebooks 15–25 (11 files, 39,978 CJK)

- [ ] `02_PyTorch_Algorithms/15_End_to_End_Fine_Tuning_Experiment.ipynb`
- [ ] `02_PyTorch_Algorithms/16_RLHF_PPO_Memory.ipynb`
- [ ] `02_PyTorch_Algorithms/17_DPO_Loss_Tutorial.ipynb`
- [ ] `02_PyTorch_Algorithms/18_GRPO_Loss_Tutorial.ipynb`
- [ ] `02_PyTorch_Algorithms/19_Activation_Checkpointing.ipynb`
- [ ] `02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb`
- [ ] `02_PyTorch_Algorithms/21_Decoding_Strategies.ipynb`
- [ ] `02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb`
- [ ] `02_PyTorch_Algorithms/23_Speculative_Decoding.ipynb`
- [ ] `02_PyTorch_Algorithms/24_Prefix_Cache_Matching_and_Reuse.ipynb`
- [ ] `02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb`

#### 02-3 — Notebooks 26–36 (11 files, 39,405 CJK)

- [ ] `02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.ipynb`
- [ ] `02_PyTorch_Algorithms/27_ZeRO_Optimizer_Sim.ipynb`
- [ ] `02_PyTorch_Algorithms/28_Pipeline_Parallelism_MicroBatch.ipynb`
- [ ] `02_PyTorch_Algorithms/29_Tensor_Parallelism_Sim.ipynb`
- [ ] `02_PyTorch_Algorithms/30_Long_Context_Fine_Tuning.ipynb`
- [ ] `02_PyTorch_Algorithms/31_LoRA_Variants_Theory.ipynb`
- [ ] `02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.ipynb`
- [ ] `02_PyTorch_Algorithms/33_Fine_Tuning_Readiness.ipynb`
- [ ] `02_PyTorch_Algorithms/34_SGLang_RadixAttention.ipynb`
- [ ] `02_PyTorch_Algorithms/35_Multi_Token_Decoding.ipynb`
- [ ] `02_PyTorch_Algorithms/36_Decode_Scheduling.ipynb`

#### 02-4 — Notebooks 37–48 (12 files, 40,835 CJK)

- [ ] `02_PyTorch_Algorithms/37_KV_Cache_Lifecycle_and_Capacity_Control.ipynb`
- [ ] `02_PyTorch_Algorithms/38_Prefill_Decode_Scheduling.ipynb`
- [ ] `02_PyTorch_Algorithms/39_Hetero_PD_and_Serving_Tiers.ipynb`
- [ ] `02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb`
- [ ] `02_PyTorch_Algorithms/41_FP8_Runtime_Low_Precision_Paths.ipynb`
- [ ] `02_PyTorch_Algorithms/42_Activation_Offload.ipynb`
- [ ] `02_PyTorch_Algorithms/43_Unified_Training_Memory_Ledger.ipynb`
- [ ] `02_PyTorch_Algorithms/44_Local_Sliding_and_Sparse_Attention.ipynb`
- [ ] `02_PyTorch_Algorithms/45_Linear_Attention_and_Recurrent_State.ipynb`
- [ ] `02_PyTorch_Algorithms/46_MoE_Expert_Parallel.ipynb`
- [ ] `02_PyTorch_Algorithms/47_Communication_Profiling_with_NCCL.ipynb`
- [ ] `02_PyTorch_Algorithms/48_Communication_Hotspots_and_Mitigation.ipynb`

#### 02-5 — Notebooks 49–62 (14 files, 39,173 CJK)

- [ ] `02_PyTorch_Algorithms/49_Parallelism_Strategy_Selection.ipynb`
- [ ] `02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.ipynb`
- [ ] `02_PyTorch_Algorithms/51_Online_DPO.ipynb`
- [ ] `02_PyTorch_Algorithms/52_Quantization_Calibration_and_Error.ipynb`
- [ ] `02_PyTorch_Algorithms/53_Activation_Quantization_and_SmoothQuant.ipynb`
- [ ] `02_PyTorch_Algorithms/54_KV_Cache_Quantization_Strategies.ipynb`
- [ ] `02_PyTorch_Algorithms/55_Training_Update_Dynamics.ipynb`
- [ ] `02_PyTorch_Algorithms/56_Continual_Pretraining_and_Retention.ipynb`
- [ ] `02_PyTorch_Algorithms/57_Reserved_57.ipynb`
- [ ] `02_PyTorch_Algorithms/58_Reserved_58.ipynb`
- [ ] `02_PyTorch_Algorithms/59_Reserved_59.ipynb`
- [ ] `02_PyTorch_Algorithms/60_Decoder_Block_Stability_Benchmark.ipynb`
- [ ] `02_PyTorch_Algorithms/61_Model_Architecture_Exploration.ipynb`
- [ ] `02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.ipynb`

#### 02-6 — Notebooks 63–72 (10 files, 41,526 CJK)

- [ ] `02_PyTorch_Algorithms/63_LoRA_Variants_Benchmark.ipynb`
- [ ] `02_PyTorch_Algorithms/64_SFT_Data_Quality_Project.ipynb`
- [ ] `02_PyTorch_Algorithms/65_QLoRA_Selection_Project.ipynb`
- [ ] `02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb`
- [ ] `02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb`
- [ ] `02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.ipynb`
- [ ] `02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.ipynb`
- [ ] `02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.ipynb`
- [ ] `02_PyTorch_Algorithms/71_MLA_KV_Cache_Architecture_Benchmark.ipynb`
- [ ] `02_PyTorch_Algorithms/72_Multi_Model_Deployment_Comparison.ipynb`

#### 02-7 — Notebooks 73–86 (14 files, 41,537 CJK)

- [ ] `02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb`
- [ ] `02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb`
- [ ] `02_PyTorch_Algorithms/75_Optimizer_Execution_Path_Benchmark.ipynb`
- [ ] `02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb`
- [ ] `02_PyTorch_Algorithms/77_Long_Sequence_Memory_Architecture_Benchmark.ipynb`
- [ ] `02_PyTorch_Algorithms/78_Attention_Access_Pattern_Benchmark.ipynb`
- [ ] `02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.ipynb`
- [ ] `02_PyTorch_Algorithms/80_MoE_Expert_Parallel_Benchmark.ipynb`
- [ ] `02_PyTorch_Algorithms/81_Distributed_Inference_Project.ipynb`
- [ ] `02_PyTorch_Algorithms/82_Quantization_Artifact_Evaluation_Project.ipynb`
- [ ] `02_PyTorch_Algorithms/83_Quantized_Serving_Profile_Benchmark.ipynb`
- [ ] `02_PyTorch_Algorithms/84_DPO_Preference_Project.ipynb`
- [ ] `02_PyTorch_Algorithms/85_GRPO_Groupwise_Alignment_Project.ipynb`
- [ ] `02_PyTorch_Algorithms/86_DPO_Online_Benchmark.ipynb`

#### 02-8 — Reserved tail + ARCH/DISTILL/INF-A…P (14 files, 40,476 CJK)

- [ ] `02_PyTorch_Algorithms/87_Reserved_87.ipynb`
- [ ] `02_PyTorch_Algorithms/88_Distilled_Student_Delivery_Project.ipynb`
- [ ] `02_PyTorch_Algorithms/89_Reserved_89.ipynb`
- [ ] `02_PyTorch_Algorithms/ARCH-HYBRID-MEMORY_Attention_SSM_Hybrid.ipynb`
- [ ] `02_PyTorch_Algorithms/ARCH-MLA_Multihead_Latent_Attention.ipynb`
- [ ] `02_PyTorch_Algorithms/DISTILL-OPD_On_Policy_Distillation_and_GKD.ipynb`
- [ ] `02_PyTorch_Algorithms/INF-AFD_Attention_FFN_Disaggregation.ipynb`
- [ ] `02_PyTorch_Algorithms/INF-CONTINUOUS-BATCHING_Continuous_Batching.ipynb`
- [ ] `02_PyTorch_Algorithms/INF-EAGLE_Feature_Drafter_and_Candidate_Tree.ipynb`
- [ ] `02_PyTorch_Algorithms/INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch.ipynb`
- [ ] `02_PyTorch_Algorithms/INF-ITERATION-BUDGET_Serving_Iteration_Cost_and_Batch_Bounds.ipynb`
- [ ] `02_PyTorch_Algorithms/INF-MOE-EXECUTION_MoE_Inference_Dispatch_and_Expert_Execution.ipynb`
- [ ] `02_PyTorch_Algorithms/INF-PARALLEL-DRAFTING_DFlash_and_DSpark.ipynb`
- [ ] `02_PyTorch_Algorithms/INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.ipynb`

#### 02-9 — INF-R…S + PAR + PERF-C…R + QUANT-QAT + TRAIN-MIXED (14 files, 44,997 CJK)

- [ ] `02_PyTorch_Algorithms/INF-REQUEST-LIFECYCLE_Inference_Request_Lifecycle.ipynb`
- [ ] `02_PyTorch_Algorithms/INF-SERVING-OVERLAP_Serving_Overlap_and_State_Handoff.ipynb`
- [ ] `02_PyTorch_Algorithms/PAR-CONTEXT_Context_and_Sequence_Parallelism.ipynb`
- [ ] `02_PyTorch_Algorithms/PAR-MODEL-INTERNAL_Model_Internal_Parallel_Benchmark.ipynb`
- [ ] `02_PyTorch_Algorithms/PAR-REPLICA_Serving_Replica_Routing.ipynb`
- [ ] `02_PyTorch_Algorithms/PERF-CANDIDATE_Constraint_to_Candidate_and_Cost_Transfer.ipynb`
- [ ] `02_PyTorch_Algorithms/PERF-COMPOSITION_Strategy_Interaction_and_Bottleneck_Shift.ipynb`
- [ ] `02_PyTorch_Algorithms/PERF-INPUT_Training_Input_Pipeline_Performance.ipynb`
- [ ] `02_PyTorch_Algorithms/PERF-IO_Artifact_State_and_Output_IO_Critical_Path.ipynb`
- [ ] `02_PyTorch_Algorithms/PERF-MEMORY-TIMELINE_Memory_Residency_and_Peak_Overlap.ipynb`
- [ ] `02_PyTorch_Algorithms/PERF-REGRESSION_Paired_Benchmark_and_Regression_Thresholds.ipynb`
- [ ] `02_PyTorch_Algorithms/PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.ipynb`
- [ ] `02_PyTorch_Algorithms/QUANT-QAT_Quantization_Aware_Training.ipynb`
- [ ] `02_PyTorch_Algorithms/TRAIN-MIXED-PRECISION-BACKWARD_Mixed_Precision_Backward.ipynb`

#### 02-10 — PERF-TRACE + PT-* + QUANT-MIXED-BIT (10 files, 39,041 CJK)

- [ ] `02_PyTorch_Algorithms/PERF-TRACE_Trace_Dependencies_and_Critical_Path.ipynb`
- [ ] `02_PyTorch_Algorithms/PT-AGENTIC-RL_Environment_Trajectory_and_Credit_Assignment.ipynb`
- [ ] `02_PyTorch_Algorithms/PT-KTO_Binary_Feedback_Alignment.ipynb`
- [ ] `02_PyTorch_Algorithms/PT-REWARD-MODEL_Reward_Model_Training_and_Calibration.ipynb`
- [ ] `02_PyTorch_Algorithms/PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.ipynb`
- [ ] `02_PyTorch_Algorithms/PT-RL-SYSTEM_Rollout_Training_and_Reward_System.ipynb`
- [ ] `02_PyTorch_Algorithms/PT-RLOO_Leave_One_Out_Policy_Optimization.ipynb`
- [ ] `02_PyTorch_Algorithms/PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.ipynb`
- [ ] `02_PyTorch_Algorithms/PT-SYNTHETIC-DATA_Instruction_Generation_and_Verifier_Admission.ipynb`
- [ ] `02_PyTorch_Algorithms/QUANT-MIXED-BIT_Mixed_Bit_Allocation.ipynb`

### M4 — Part 03 — Triton Kernels (26 files, 53,581 CJK)

#### 03-P — Prose and navigation (6 files, 3,180 CJK)

- [ ] `03_Triton_Kernels/intro.md`
- [ ] `03_Triton_Kernels/3_1.md`
- [ ] `03_Triton_Kernels/3_2.md`
- [ ] `03_Triton_Kernels/3_3.md`
- [ ] `03_Triton_Kernels/3_4.md`
- [ ] `03_Triton_Kernels/3_5.md`

#### 03-1 — Notebooks 01–10 (10 files, 24,331 CJK)

- [ ] `03_Triton_Kernels/01_Triton_Vector_Addition.ipynb`
- [ ] `03_Triton_Kernels/02_Triton_Layouts_and_Memory_Access.ipynb`
- [ ] `03_Triton_Kernels/03_Triton_Fused_SwiGLU.ipynb`
- [ ] `03_Triton_Kernels/04_Triton_Fused_RMSNorm.ipynb`
- [ ] `03_Triton_Kernels/05_Triton_GEMM_Tutorial.ipynb`
- [ ] `03_Triton_Kernels/06_Triton_Persistent_GEMM_and_Tile_Scheduling.ipynb`
- [ ] `03_Triton_Kernels/07_Triton_Fused_Softmax.ipynb`
- [ ] `03_Triton_Kernels/08_Triton_Softmax_Backward.ipynb`
- [ ] `03_Triton_Kernels/09_Triton_Autotune_and_Profiling.ipynb`
- [ ] `03_Triton_Kernels/10_Triton_Fused_RoPE.ipynb`

#### 03-2 — Notebooks 11–20 (10 files, 26,070 CJK)

- [ ] `03_Triton_Kernels/11_Triton_Flash_Attention.ipynb`
- [ ] `03_Triton_Kernels/12_Triton_Variable_Length_GQA.ipynb`
- [ ] `03_Triton_Kernels/13_Triton_PagedAttention.ipynb`
- [ ] `03_Triton_Kernels/14_Triton_Quantization.ipynb`
- [ ] `03_Triton_Kernels/15_Triton_Multi_LoRA.ipynb`
- [ ] `03_Triton_Kernels/16_Triton_KV_Cache_Quantized_Decode.ipynb`
- [ ] `03_Triton_Kernels/17_Triton_Memory_Model_and_Debug.ipynb`
- [ ] `03_Triton_Kernels/18_Triton_Compilation_Artifacts_and_Execution.ipynb`
- [ ] `03_Triton_Kernels/19_Triton_PyTorch_Op_Integration.ipynb`
- [ ] `03_Triton_Kernels/20_Triton_Llama3_Block_Project.ipynb`

### M5 — Part 04 — CUDA and System Optimization (25 files, 51,401 CJK)

#### 04-P — Prose and navigation (6 files, 4,701 CJK)

- [ ] `04_CUDA_and_System_Optimization/intro.md`
- [ ] `04_CUDA_and_System_Optimization/4_1.md`
- [ ] `04_CUDA_and_System_Optimization/4_2.md`
- [ ] `04_CUDA_and_System_Optimization/4_3.md`
- [ ] `04_CUDA_and_System_Optimization/4_4.md`
- [ ] `04_CUDA_and_System_Optimization/4_5.md`

#### 04-1 — Notebooks 01–10 (10 files, 22,890 CJK)

- [ ] `04_CUDA_and_System_Optimization/01_CUDA_Custom_Kernel_Intro.ipynb`
- [ ] `04_CUDA_and_System_Optimization/02_CUDA_Global_Memory_Coalescing.ipynb`
- [ ] `04_CUDA_and_System_Optimization/03_Warp_Level_Primitives.ipynb`
- [ ] `04_CUDA_and_System_Optimization/04_Bank_Conflict_Deep_Dive.ipynb`
- [ ] `04_CUDA_and_System_Optimization/05_CUDA_Block_Resources_and_Residency.ipynb`
- [ ] `04_CUDA_and_System_Optimization/06_CUDA_Shared_Memory_Optimization.ipynb`
- [ ] `04_CUDA_and_System_Optimization/07_Tensor_Core_MMA_Programming.ipynb`
- [ ] `04_CUDA_and_System_Optimization/08_Async_Data_Prefetch_and_Double_Buffering.ipynb`
- [ ] `04_CUDA_and_System_Optimization/09_CUDA_Async_Copy_Pipeline.ipynb`
- [ ] `04_CUDA_and_System_Optimization/10_PyTorch_CUDA_Streams_and_Transfer.ipynb`

#### 04-2 — Notebooks 11–19 (9 files, 23,810 CJK)

- [ ] `04_CUDA_and_System_Optimization/11_CUDA_Graph_and_JIT_Compile.ipynb`
- [ ] `04_CUDA_and_System_Optimization/12_Memory_Pool_and_VRAM_Management.ipynb`
- [ ] `04_CUDA_and_System_Optimization/13_Distributed_Communication_Primitives.ipynb`
- [ ] `04_CUDA_and_System_Optimization/14_Ring_AllReduce_Deep_Dive.ipynb`
- [ ] `04_CUDA_and_System_Optimization/15_Communication_Computation_Overlap_Advanced_Scheduling.ipynb`
- [ ] `04_CUDA_and_System_Optimization/16_DeepSpeed_Zero_Config.ipynb`
- [ ] `04_CUDA_and_System_Optimization/17_Heterogeneous_Training_CPU_Offload_NVMe_Offload.ipynb`
- [ ] `04_CUDA_and_System_Optimization/18_CUDA_vs_Triton_vs_PyTorch.ipynb`
- [ ] `04_CUDA_and_System_Optimization/19_TCO_and_Hardware_Selection.ipynb`

### M6 — Part 05 — Integrated Projects and Decisions (13 files, 8,394 CJK)

#### 05-1 — All project pages and notebooks (13 files, 8,394 CJK)

- [ ] `05_Integrated_Projects_and_Decisions/01_Inference_Service/Inference_Service_Project.ipynb`
- [ ] `05_Integrated_Projects_and_Decisions/01_Inference_Service/index.md`
- [ ] `05_Integrated_Projects_and_Decisions/01_Inference_Service_Integrated_Project.md`
- [ ] `05_Integrated_Projects_and_Decisions/02_Post_Training_Delivery/Post_Training_Delivery_Project.ipynb`
- [ ] `05_Integrated_Projects_and_Decisions/02_Post_Training_Delivery/index.md`
- [ ] `05_Integrated_Projects_and_Decisions/02_Post_Training_Delivery_Project.md`
- [ ] `05_Integrated_Projects_and_Decisions/03_Performance_Optimization_Delivery/Performance_Optimization_Delivery_Project.ipynb`
- [ ] `05_Integrated_Projects_and_Decisions/03_Performance_Optimization_Delivery/index.md`
- [ ] `05_Integrated_Projects_and_Decisions/04_Quantization_Delivery/Quantization_Delivery_Project.ipynb`
- [ ] `05_Integrated_Projects_and_Decisions/04_Quantization_Delivery/index.md`
- [ ] `05_Integrated_Projects_and_Decisions/intro.md`
- [ ] `05_Integrated_Projects_and_Decisions/open_practice/_template/README.md`
- [ ] `05_Integrated_Projects_and_Decisions/open_practice/_template/index.md`

### M7 — Topic discussions (180 files, 240,702 CJK)

#### 07-0 — Topic overview (1 files, 2,795 CJK)

- [ ] `topic_discussion/intro.md`

#### 07-01 — Topic: agent (8 files, 6,611 CJK)

- [ ] `topic_discussion/agent/00_agent_workload_and_trace.md`
- [ ] `topic_discussion/agent/01_agent_trajectory_data.md`
- [ ] `topic_discussion/agent/02_agent_sandbox_environment.md`
- [ ] `topic_discussion/agent/casebook.md`
- [ ] `topic_discussion/agent/experiment_assets.md`
- [ ] `topic_discussion/agent/glossary.md`
- [ ] `topic_discussion/agent/intro.md`
- [ ] `topic_discussion/agent/walkthrough.md`

#### 07-02 — Topic: backpropagation_training_mechanism (13 files, 15,386 CJK)

- [ ] `topic_discussion/backpropagation_training_mechanism/01_training_loop_and_computation_graph.md`
- [ ] `topic_discussion/backpropagation_training_mechanism/02_autograd_and_transformer_backward.md`
- [ ] `topic_discussion/backpropagation_training_mechanism/03_loss_reduction_gradient_accumulation_effective_update.md`
- [ ] `topic_discussion/backpropagation_training_mechanism/04_training_state_lifecycle.md`
- [ ] `topic_discussion/backpropagation_training_mechanism/05_checkpointing_and_offload.md`
- [ ] `topic_discussion/backpropagation_training_mechanism/06_correctness_and_performance_evidence.md`
- [ ] `topic_discussion/backpropagation_training_mechanism/casebook.md`
- [ ] `topic_discussion/backpropagation_training_mechanism/experiment_assets.md`
- [ ] `topic_discussion/backpropagation_training_mechanism/formula_reference.md`
- [ ] `topic_discussion/backpropagation_training_mechanism/glossary.md`
- [ ] `topic_discussion/backpropagation_training_mechanism/intro.md`
- [ ] `topic_discussion/backpropagation_training_mechanism/quickstart.md`
- [ ] `topic_discussion/backpropagation_training_mechanism/walkthrough.md`

#### 07-03 — Topic: communication_parallelism (13 files, 11,388 CJK)

- [ ] `topic_discussion/communication_parallelism/01_why_parallel_and_communication.md`
- [ ] `topic_discussion/communication_parallelism/02_data_parallel_and_synchronization.md`
- [ ] `topic_discussion/communication_parallelism/03_state_sharding_and_zero.md`
- [ ] `topic_discussion/communication_parallelism/04_pipeline_and_tensor_parallel.md`
- [ ] `topic_discussion/communication_parallelism/05_expert_parallel_and_communication_hotspots.md`
- [ ] `topic_discussion/communication_parallelism/06_communication_profiling_and_overlap.md`
- [ ] `topic_discussion/communication_parallelism/07_benchmark_and_parallel_decision.md`
- [ ] `topic_discussion/communication_parallelism/casebook.md`
- [ ] `topic_discussion/communication_parallelism/experiment_assets.md`
- [ ] `topic_discussion/communication_parallelism/glossary.md`
- [ ] `topic_discussion/communication_parallelism/intro.md`
- [ ] `topic_discussion/communication_parallelism/quickstart.md`
- [ ] `topic_discussion/communication_parallelism/walkthrough.md`

#### 07-04 — Topic: data_engineering (12 files, 9,598 CJK)

- [ ] `topic_discussion/data_engineering/00_data_lifecycle_and_contract.md`
- [ ] `topic_discussion/data_engineering/01_collection_parsing_contract.md`
- [ ] `topic_discussion/data_engineering/02_cleaning_quality_chain.md`
- [ ] `topic_discussion/data_engineering/03_data_mixture_and_curriculum.md`
- [ ] `topic_discussion/data_engineering/04_synthetic_data_and_trajectory_validation.md`
- [ ] `topic_discussion/data_engineering/05_post_training_data_interfaces.md`
- [ ] `topic_discussion/data_engineering/06_pipeline_lineage_and_downstream_validation.md`
- [ ] `topic_discussion/data_engineering/casebook.md`
- [ ] `topic_discussion/data_engineering/experiment_assets.md`
- [ ] `topic_discussion/data_engineering/glossary.md`
- [ ] `topic_discussion/data_engineering/intro.md`
- [ ] `topic_discussion/data_engineering/walkthrough.md`

#### 07-05 — Topic: deployment_heterogeneous (14 files, 12,822 CJK)

- [ ] `topic_discussion/deployment_heterogeneous/01_model_artifact_and_deployment_contract.md`
- [ ] `topic_discussion/deployment_heterogeneous/02_backend_compatibility_and_single_node_execution.md`
- [ ] `topic_discussion/deployment_heterogeneous/03_device_coordination_and_data_path.md`
- [ ] `topic_discussion/deployment_heterogeneous/04_multi_device_and_service_tiers.md`
- [ ] `topic_discussion/deployment_heterogeneous/05_observability_regression_and_release.md`
- [ ] `topic_discussion/deployment_heterogeneous/06_heterogeneous_selection_and_cost_decision.md`
- [ ] `topic_discussion/deployment_heterogeneous/07_accelerator_runtime_compatibility.md`
- [ ] `topic_discussion/deployment_heterogeneous/08_edge_cloud_placement_and_handoff.md`
- [ ] `topic_discussion/deployment_heterogeneous/casebook.md`
- [ ] `topic_discussion/deployment_heterogeneous/experiment_assets.md`
- [ ] `topic_discussion/deployment_heterogeneous/glossary.md`
- [ ] `topic_discussion/deployment_heterogeneous/intro.md`
- [ ] `topic_discussion/deployment_heterogeneous/quickstart.md`
- [ ] `topic_discussion/deployment_heterogeneous/walkthrough.md`

#### 07-06 — Topic: inference_optimization (17 files, 35,435 CJK)

- [ ] `topic_discussion/inference_optimization/01_request_path_and_metrics.md`
- [ ] `topic_discussion/inference_optimization/02_prefill_and_attention_kernel.md`
- [ ] `topic_discussion/inference_optimization/03_decoding_strategies.md`
- [ ] `topic_discussion/inference_optimization/04_kv_cache_lifecycle_and_reuse.md`
- [ ] `topic_discussion/inference_optimization/05_serving_scheduling_and_pd.md`
- [ ] `topic_discussion/inference_optimization/06_quantized_inference_and_deployment.md`
- [ ] `topic_discussion/inference_optimization/07_benchmark_and_decision.md`
- [ ] `topic_discussion/inference_optimization/08_distributed_inference_and_parallelism.md`
- [ ] `topic_discussion/inference_optimization/casebook.md`
- [ ] `topic_discussion/inference_optimization/experiment_assets.md`
- [ ] `topic_discussion/inference_optimization/glossary.md`
- [ ] `topic_discussion/inference_optimization/intro.md`
- [ ] `topic_discussion/inference_optimization/mini_sglang_cache_scheduling.md`
- [ ] `topic_discussion/inference_optimization/nano_vllm_request_walkthrough.md`
- [ ] `topic_discussion/inference_optimization/quickstart.md`
- [ ] `topic_discussion/inference_optimization/teaching_engines.md`
- [ ] `topic_discussion/inference_optimization/walkthrough.md`

#### 07-07 — Topic: llm_architecture_evolution (16 files, 17,158 CJK)

- [ ] `topic_discussion/llm_architecture_evolution/01_transformer_decoder.md`
- [ ] `topic_discussion/llm_architecture_evolution/02_tokenization_embedding.md`
- [ ] `topic_discussion/llm_architecture_evolution/03_norm_evolution.md`
- [ ] `topic_discussion/llm_architecture_evolution/04_attention_evolution.md`
- [ ] `topic_discussion/llm_architecture_evolution/05_rope_position_encoding.md`
- [ ] `topic_discussion/llm_architecture_evolution/06_block_residual_path.md`
- [ ] `topic_discussion/llm_architecture_evolution/07_mlp_ffn_evolution.md`
- [ ] `topic_discussion/llm_architecture_evolution/08_representative_models.md`
- [ ] `topic_discussion/llm_architecture_evolution/09_moe_sparsity_evolution.md`
- [ ] `topic_discussion/llm_architecture_evolution/10_linear_attention_ssm_and_hybrid.md`
- [ ] `topic_discussion/llm_architecture_evolution/casebook.md`
- [ ] `topic_discussion/llm_architecture_evolution/experiment_assets.md`
- [ ] `topic_discussion/llm_architecture_evolution/glossary.md`
- [ ] `topic_discussion/llm_architecture_evolution/intro.md`
- [ ] `topic_discussion/llm_architecture_evolution/quickstart.md`
- [ ] `topic_discussion/llm_architecture_evolution/walkthrough.md`

#### 07-08 — Topic: memory_performance_tuning (1 files, 299 CJK)

- [ ] `topic_discussion/memory_performance_tuning/intro.md`

#### 07-09 — Topic: model_architecture_and_training (2 files, 4,034 CJK)

- [ ] `topic_discussion/model_architecture_and_training/intro.md`
- [ ] `topic_discussion/model_architecture_and_training/walkthrough.md`

#### 07-10 — Topic: multimodal (12 files, 7,279 CJK)

- [ ] `topic_discussion/multimodal/00_visual_representation_cnn_to_vit.md`
- [ ] `topic_discussion/multimodal/01_detection_yolo_to_detr.md`
- [ ] `topic_discussion/multimodal/02_realtime_detr_evolution.md`
- [ ] `topic_discussion/multimodal/03_vlm_architecture_and_input_contract.md`
- [ ] `topic_discussion/multimodal/04_vlm_workload_and_cache.md`
- [ ] `topic_discussion/multimodal/05_multimodal_profiling_and_deployment.md`
- [ ] `topic_discussion/multimodal/06_multimodal_delivery_decision.md`
- [ ] `topic_discussion/multimodal/casebook.md`
- [ ] `topic_discussion/multimodal/experiment_assets.md`
- [ ] `topic_discussion/multimodal/glossary.md`
- [ ] `topic_discussion/multimodal/intro.md`
- [ ] `topic_discussion/multimodal/walkthrough.md`

#### 07-11 — Topic: operator_optimization (14 files, 33,945 CJK)

- [ ] `topic_discussion/operator_optimization/01_optimization_targets_and_baselines.md`
- [ ] `topic_discussion/operator_optimization/02_graph_compilation_and_fusion.md`
- [ ] `topic_discussion/operator_optimization/03_kernel_semantics_and_memory.md`
- [ ] `topic_discussion/operator_optimization/04_attention_and_inference_data_paths.md`
- [ ] `topic_discussion/operator_optimization/05_measurement_profiling_and_autotuning.md`
- [ ] `topic_discussion/operator_optimization/06_cuda_execution_and_runtime_constraints.md`
- [ ] `topic_discussion/operator_optimization/07_integration_benchmarking_and_adoption.md`
- [ ] `topic_discussion/operator_optimization/casebook.md`
- [ ] `topic_discussion/operator_optimization/experiment_assets.md`
- [ ] `topic_discussion/operator_optimization/formula_reference.md`
- [ ] `topic_discussion/operator_optimization/glossary.md`
- [ ] `topic_discussion/operator_optimization/intro.md`
- [ ] `topic_discussion/operator_optimization/quickstart.md`
- [ ] `topic_discussion/operator_optimization/walkthrough.md`

#### 07-12 — Topic: performance_analysis (14 files, 16,368 CJK)

- [ ] `topic_discussion/performance_analysis/01_why_profiling_matters.md`
- [ ] `topic_discussion/performance_analysis/02_reproducible_baseline.md`
- [ ] `topic_discussion/performance_analysis/03_time_breakdown_and_trace_reading.md`
- [ ] `topic_discussion/performance_analysis/04_memory_timeline_and_residency.md`
- [ ] `topic_discussion/performance_analysis/05_bottleneck_attribution.md`
- [ ] `topic_discussion/performance_analysis/06_benchmark_design_and_regression_validation.md`
- [ ] `topic_discussion/performance_analysis/07_diagnosis_and_action_decision.md`
- [ ] `topic_discussion/performance_analysis/casebook.md`
- [ ] `topic_discussion/performance_analysis/experiment_assets.md`
- [ ] `topic_discussion/performance_analysis/formula_reference.md`
- [ ] `topic_discussion/performance_analysis/glossary.md`
- [ ] `topic_discussion/performance_analysis/intro.md`
- [ ] `topic_discussion/performance_analysis/quickstart.md`
- [ ] `topic_discussion/performance_analysis/walkthrough.md`

#### 07-13 — Topic: performance_optimization (13 files, 18,290 CJK)

- [ ] `topic_discussion/performance_optimization/01_performance_problem_and_benchmark.md`
- [ ] `topic_discussion/performance_optimization/02_observability_and_profiling.md`
- [ ] `topic_discussion/performance_optimization/03_training_memory_ledger.md`
- [ ] `topic_discussion/performance_optimization/04_inference_memory_and_kv_cache.md`
- [ ] `topic_discussion/performance_optimization/05_compute_communication_and_io.md`
- [ ] `topic_discussion/performance_optimization/06_cost_model_and_production_decision.md`
- [ ] `topic_discussion/performance_optimization/07_end_to_end_performance_engineering.md`
- [ ] `topic_discussion/performance_optimization/casebook.md`
- [ ] `topic_discussion/performance_optimization/experiment_assets.md`
- [ ] `topic_discussion/performance_optimization/glossary.md`
- [ ] `topic_discussion/performance_optimization/intro.md`
- [ ] `topic_discussion/performance_optimization/quickstart.md`
- [ ] `topic_discussion/performance_optimization/walkthrough.md`

#### 07-14 — Topic: post_training_optimization (17 files, 31,849 CJK)

- [ ] `topic_discussion/post_training_optimization/01_sft_data_and_loss.md`
- [ ] `topic_discussion/post_training_optimization/02_lora_peft_design.md`
- [ ] `topic_discussion/post_training_optimization/03_training_control.md`
- [ ] `topic_discussion/post_training_optimization/04_end_to_end_experiment.md`
- [ ] `topic_discussion/post_training_optimization/05_dpo_and_preference_optimization.md`
- [ ] `topic_discussion/post_training_optimization/06_rlhf_and_ppo_system_cost.md`
- [ ] `topic_discussion/post_training_optimization/07_grpo_and_groupwise_alignment.md`
- [ ] `topic_discussion/post_training_optimization/08_trajectory_lifecycle_and_environment_interaction.md`
- [ ] `topic_discussion/post_training_optimization/09_rl_execution_system_and_efficiency.md`
- [ ] `topic_discussion/post_training_optimization/10_evaluation_and_adoption_decision.md`
- [ ] `topic_discussion/post_training_optimization/casebook.md`
- [ ] `topic_discussion/post_training_optimization/experiment_assets.md`
- [ ] `topic_discussion/post_training_optimization/formula_reference.md`
- [ ] `topic_discussion/post_training_optimization/glossary.md`
- [ ] `topic_discussion/post_training_optimization/intro.md`
- [ ] `topic_discussion/post_training_optimization/quickstart.md`
- [ ] `topic_discussion/post_training_optimization/walkthrough.md`

#### 07-15 — Topic: quantization_compression (13 files, 17,445 CJK)

- [ ] `topic_discussion/quantization_compression/01_quantization_object_and_error.md`
- [ ] `topic_discussion/quantization_compression/02_ptq_and_qat_timing.md`
- [ ] `topic_discussion/quantization_compression/03_low_bit_training_adaptation.md`
- [ ] `topic_discussion/quantization_compression/04_weight_only_compression.md`
- [ ] `topic_discussion/quantization_compression/05_fp8_and_kv_cache_quantization.md`
- [ ] `topic_discussion/quantization_compression/06_deployment_and_benchmark_decision.md`
- [ ] `topic_discussion/quantization_compression/07_pruning_distillation_and_structured_sparsity.md`
- [ ] `topic_discussion/quantization_compression/casebook.md`
- [ ] `topic_discussion/quantization_compression/experiment_assets.md`
- [ ] `topic_discussion/quantization_compression/glossary.md`
- [ ] `topic_discussion/quantization_compression/intro.md`
- [ ] `topic_discussion/quantization_compression/quickstart.md`
- [ ] `topic_discussion/quantization_compression/walkthrough.md`

### M8 — Team study (6 files, 2,994 CJK)

#### 08-1 — All study-group pages (6 files, 2,994 CJK)

- [ ] `team_study/intro.md`
- [ ] `team_study/part2_l1_202606/group_topic_1.md`
- [ ] `team_study/part2_l1_202606/group_topic_2.md`
- [ ] `team_study/part2_l1_202606/intro.md`
- [ ] `team_study/part2_l1_202607/intro.md`
- [ ] `team_study/part2_l2_202607/intro.md`

### M9 — Docs-only pages (14 files, 25,810 CJK)

#### 09-1 — Guides, maintenance, environment, CNB history (6 files, 7,328 CJK)

- [ ] `docs/guide.md`
- [ ] `docs/contributing.md`
- [ ] `docs/maintenance.md`
- [ ] `docs/template_guidelines.md`
- [ ] `docs/gpu_environment_assets.md`
- [ ] `docs/cnb/README.md`

#### 09-2 — Registries, audits, verification, terminology (8 files, 18,482 CJK)

- [ ] `docs/part02_result_assets.md`
- [ ] `docs/part02_visual_assets_audit.md`
- [ ] `docs/topic_asset_registry.md`
- [ ] `docs/topic_discussion_visual_assets_audit.md`
- [ ] `docs/verification/fine_tuning_projects.md`
- [ ] `docs/verification/inference_projects.md`
- [ ] `docs/verification/memory_projects.md`
- [ ] `docs/terminology.md`

### M10 — Ancillary development and environment docs (optional) (5 files, 3,487 CJK)

#### 10-1 — Tooling and environment docs (5 files, 3,487 CJK)

- [ ] `tools/README.md`
- [ ] `tools/diagram_style_guide.md`
- [ ] `tools/tests/README.md`
- [ ] `tools/_scratch/README.md`
- [ ] `cnb/README.md`

## Session log

| Date | Agent | Milestone / batch | Files touched | Validation run | Notes |
| --- | --- | --- | ---: | --- | --- |
| 2026-11-23 | OpenCode | M0 dry run (M1 `00-P`/`00-1`) | 2 | JSON/schema, AST, residual-CJK, structure | Conventions confirmed; code print/assert strings left Chinese per policy |
| 2026-11-23 | OpenCode | M1 `00-P` (0A–0E) | 5 | Structure diff, residual-CJK, mirror check, chapter links | Zero residual CJK; heading/table/link counts match baseline; mirrors rebuilt for the six `00-P` pages |
| 2026-11-23 | OpenCode | M1 `00-1` (02–10) | 9 | Structure diff, AST (docstrings normalized), cell metadata/outputs, residual-CJK scan, mirror check, chapter links | 0 CJK in markdown prose and 0 in code comments/docstrings; the 56 remaining markdown CJK characters are the required numbered H1 subtitles; no notebook outputs changed; mirrors rebuilt for the nine notebooks |
| 2026-11-23 | OpenCode | M1 `00-2` (11–20) + M1 close-out | 10 | Structure diff, AST (docstrings normalized), cell metadata/outputs, residual-CJK scan, full Part 00 mirror rebuild, mirror and chapter-link checks, `verify.py` | 0 CJK in markdown prose and 0 in code comments/docstrings; remaining code CJK is print/assert literals plus the numbered H1 subtitles; `verify.py` reported 34 passed / 20 failed, all failures being `ModuleNotFoundError: numpy`/`torch` in this environment; out-of-scope mirrors rewritten by `verify.py` were reverted |
| 2026-11-23 | OpenCode | M2 `01-P` (intro, 1A–1F) | 7 | Structure diff, residual-CJK, mirror check, chapter links | Zero residual CJK; heading levels, table column counts, link counts, and line/blank-line counts match the baseline; `1E.md` was Chinese-only and received an English H1 and headings; no inbound heading anchors existed |
| 2026-11-24 | OpenCode | M2 `01-A` (groups 1A + 1B, 13 notebooks) | 13 | Structure diff, AST (docstrings normalized), cell metadata/outputs, comment/docstring CJK scan, mirror check, chapter links | 0 CJK in markdown prose beyond the numbered H1 subtitles and 0 in code comments/docstrings; heading levels, table column counts, and link/image targets match the baseline; code-cell dict values that are displayed as data were left in Chinese per the string-literal policy; no notebook outputs changed; mirrors rebuilt for the 13 notebooks |

## Roadblocks and quirks

Pre-filled known items; add new ones as they appear.

| ID | Area | Observation | Action |
| --- | --- | --- | --- |
| Q1 | Generated docs | `docs/00–04`, `docs/topic_discussion`, `docs/team_study` are generated mirrors | Translate sources; regenerate; never hand-edit mirrors |
| Q2 | Homepage | `docs/index.md` is generated from the excluded `README.md` | Do not edit; preserve Chinese anchor compatibility where needed |
| Q3 | Notebook H1 | `verify.py` requires numbered `# NN. English \| Chinese` H1s in Parts 00–04 | Leave those untouched; translate Part 05 notebook H1s |
| Q4 | Validators | Several tools expect Chinese headings (`本节导读`, `## 测试`, etc.) | Treat heading/title failures as expected; do not edit tools |
| Q5 | Stale docs | `docs/02_PyTorch_Algorithms/` has stale generated pages | Do not translate; full rebuild removes them |
| Q6 | Part 05 docs | Only intro and two summary pages are synced by tooling | Produce sub-page mirrors once during M11 |
| Q7 | Diagrams | 611/649 SVGs contain Chinese text | Out of scope for now; translate alt text/captions only |
| Q8 | Cloud banner | `convert_notebook.py` injects a Chinese cloud-run block | Accepted; do not hand-edit generated output |
| Q9 | Anchors | `check_docs_links.py` ignores anchors | Manually verify changed heading anchors |
| Q10 | Code strings | Code-cell strings may be asserted elsewhere | Leave non-comment/non-docstring strings unchanged |
| Q11 | Site chrome | VitePress nav/sidebar labels in `docs/.vitepress/config.mts` remain Chinese | Out of scope for this phase |
| Q12 | Landing page | `README.md` and generated `docs/index.md` remain Chinese | English edition starts from tutorial pages; README already has an English section |
| Q13 | `verify.py` side effects | `verify.py` runs `convert_chapter0_1.py` and `convert_notebook.py --dir 03/04`, rewriting `docs/01`, `docs/03`, and `docs/04` mirrors; it also executes notebooks | Run it only as a diagnostic, then revert out-of-scope mirror changes and re-check that notebook outputs are unchanged |
| Q14 | Local Python env | The default interpreter here has no `numpy`/`torch`, so notebook-executing validators fail at import | Treat these as environment limitations, not translation defects; record the count |

## Decisions log

| Date | Decision | Rationale |
| --- | --- | --- |
| bootstrap | Translate sources, regenerate docs | Avoids duplicate work and docs drift |
| bootstrap | Keep numbered bilingual H1s in Parts 00–04 | Keeps `verify.py` and the docs generator working without code changes; Part 05 H1s are translated |
| bootstrap | Code cells: comments + docstrings only | Safest; avoids changing asserted strings or behavior |
| bootstrap | Exclude `README.md` and `docs/index.md` | `docs/index.md` is generated; homepage already has an English section |
| bootstrap | Accept Chinese cloud-run boilerplate | No code changes; tutorial content is still English |
| bootstrap | No separate manifest | `PROGRESS.md` is the single status source |

## Glossary additions

Add new terms discovered during translation. Later, merge them into
`docs/terminology.md` during M9.

| Chinese | English / abbreviation | Where first seen | Notes |
| --- | --- | --- | --- |
| 显存账本 | memory ledger | `00_Prerequisites/0E.md`, `18_Memory_Profiling_and_Optimization` | Matches the `Unified_Training_Memory_Ledger` naming in Part 02 |
| 对象驻留 | object residency | `00_Prerequisites/0E.md` | Lifecycle of objects that stay resident in memory |
| 复测 | re-measurement | `00_Prerequisites/0E.md` | Re-running a measurement to confirm a change or regression |
| 上游梯度 | upstream gradient | `00_Prerequisites/0B.md` | Already in `docs/terminology.md`; used consistently |
| 梯度边界 | gradient boundary | `00_Prerequisites/08_PyTorch_Grad_Hygiene_and_No_Grad` | Distinguishes `train/eval`, `no_grad`, `detach`, and `requires_grad` |
| 保序 / 内容守恒 | content conservation | `00_Prerequisites/06_PyTorch_Tensor_Layout_and_Indexing` | Used for reshape/index checks preserving elements |

## Content issues

Record content that looks wrong, outdated, inconsistent, or untranslatable.
Translate it faithfully anyway; do not fix it.

| File | Issue | Decision / follow-up |
| --- | --- | --- |
| — | — | — |

## How to update this file

1. Mark the file checkbox.
2. Update the dashboard counts and percentage.
3. Add a session-log row for the batch.
4. Add any new roadblock or glossary entry.
5. Update the “Last updated” and “Overall” lines at the top.
6. Keep `SPEC.md` in sync if the inventory changes.


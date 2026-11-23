# Part 00: Python, PyTorch, and Experimental Foundations

## Part Overview

Shape errors, interrupted gradients, missing state, and timing bias in model code often come from misaligned foundational interfaces. Part 00 starts with data and functions, then gradually covers tensors, model objects, training updates, and model components, using small CPU examples to check the results.

## Learning Path

Beginners should read 0A → 0B → 0C → 0D; if you are familiar with Python and NumPy, you can start from 0B. When a run fails, you need to measure time, or you are preparing to compare changes, go to 0E at any time; you do not need to finish all the other groups first. Solid lines in the figure indicate the main learning order; dashed lines indicate entering diagnosis and measurement based on a problem.

![Part 00 learning path and diagnostic entry points](../docs/public/00_Prerequisites/part00_learning_route.svg)

## Five Group Entry Points

| Learning group | Problem it addresses | What you can check after completing it |
|:---|:---|:---|
| [0A Python, NumPy, and Engineering Entry](./0A.md) | How is data organized, how are array dimensions expressed, and how are configuration and files managed? | Function inputs, shape transformations, object state, and runtime paths |
| [0B PyTorch Tensors and Autograd](./0B.md) | Do tensors share storage, how do dimensions change, and where do gradients come from? | Tensor layout, indexing results, backward gradients, and gradient modes |
| [0C Models and the Training Lifecycle](./0C.md) | Which state is registered, saved, and updated, and how are training and validation connected? | Model state, resume-from-checkpoint, parameter updates, and evaluation metrics |
| [0D Transformer Block and Model Components](./0D.md) | How do attention, activations, and normalization update the same hidden state? | Branch increments, component outputs, local gradients, and multi-head shapes |
| [0E Runtime Diagnosis and Experimental Foundations](./0E.md) | Is a run valid, and how are time and capacity measured and compared? | First failure point, timing samples, capacity estimates, and experimental conditions |

## Next Steps

Choose the next entry point based on the code objects you have already mastered: Part 01 explains resources and execution conditions, while mechanism pages and topic discussions further expand specific methods.

| Problem to solve next | Preferred entry point |
|:---|:---|
| Understand gradients, loss reduction, and effective parameter updates | [Backpropagation and Training Mechanics](../topic_discussion/backpropagation_training_mechanism/intro.md) |
| Understand how model components are combined and evolve | [Part 02 Model Components](../02_PyTorch_Algorithms/2_1.md), [LLM Architecture Evolution](../topic_discussion/llm_architecture_evolution/intro.md) |
| Locate performance problems from time and capacity symptoms | [Performance Analysis](../topic_discussion/performance_analysis/intro.md) |
| Estimate scale, memory access, and communication, or start writing compute kernels | [Part 01 Hardware, Math, and Systems](../01_Hardware_Math_and_Systems/intro.md) |

## Environment

Use [`BASE-CPU`](../docs/guide.md#base-cpu) to run this part's reference code and verification. For later GPU, compiler toolchain, or multi-GPU experiments, prepare the environment according to the environment notes on the target page.

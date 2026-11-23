# 02. LLM Params and FLOPs | 大模型参数量与算力推导

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `numerical foundations`, `parameter estimation`, `FLOPs` | **Audience:** learners who need to estimate model scale and compute budgets

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

The 7B, 14B, or 70B in a model name is only a scale label. To judge whether a model fits in memory and how much compute one training or inference run needs, you also need the layer count, hidden dimension, vocabulary, feed-forward network structure, and token count.
This section establishes the estimation chain "model structure → parameter count → workload compute volume → model FLOPs utilization (MFU)". After completing it, you should be able to convert a model configuration into a parameter and compute budget and distinguish theoretical compute volume from measured throughput.

**Keywords:** parameter count, `FLOPs`, `MFU`

![Concept relationships in this section](../public/01_Hardware_Math_and_Systems/02_params_flops_chain.svg)

---
## Prerequisite Reading
**Overview:** Before entering this section, confirm how the parameter count enters the FLOPs computation, then put forward computation, training, and hardware utilization on the same estimation chain; these values will continue to be used in later fine-tuning, performance analysis, and GPU experiments.

- [01. Data Types and Precision](./01_Data_Types_and_Precision.md)
- [Group 0B: PyTorch Tensors and Autograd](../00_Prerequisites/0B.md)
  
---
## Q1: Assume the hidden dimension is $d$ and the vocabulary size is $V$. Derive the total parameter count of one Decoder, and compare a standard FFN with SwiGLU.


We break the Transformer into three parts (ignoring the very small bias and LayerNorm weights, whose share of a hundred-billion-parameter model is less than one thousandth):

**1. Embedding Layer and Output Layer (LM Head)**
- Token Embedding: shape $[V, d]$, parameter count $V \times d$.
- LM Head (output projection): shape $[d, V]$, parameter count $V \times d$.
- *(Note: many models such as Gemma/Qwen share these two weights, halving the parameter count. Here we assume they are not shared)*.

**2. Attention Mechanism (Multi-Head Attention, MHA)**
In each Decoder block:
- Project Q, K, V: three matrices of shape $[d, d]$. Parameter count $3d^2$.
- Project Output (O): one matrix of shape $[d, d]$. Parameter count $d^2$.
- **Total MHA parameter count = $4d^2$**.
*(If GQA is used, the parameter count of K and V decreases significantly; here it is computed as the original MHA)*.

**3. Feed Forward Network (FFN / MLP)**
In the standard GPT architecture, the hidden layer is first expanded to $4d$ and then projected back to $d$:
- Expansion matrix $W_{up}$: $[d, 4d]$, parameter count $4d^2$.
- Projection matrix $W_{down}$: $[4d, d]$, parameter count $4d^2$.
- **Total FFN parameter count = $8d^2$**.
*(If SwiGLU is used in LLaMA, the intermediate dimension is usually scaled to about $\frac{8}{3}d$; although there are 3 matrices, the total parameter count is still about $3 \times \frac{8}{3}d^2 = 8d^2$. If $4d$ is used directly as the SwiGLU intermediate dimension, the parameter count becomes $12d^2$ and cannot be mixed with the standard FFN formula.)*

**In summary:**
- Parameter count of one Block = $4d^2$ (Attn) + $8d^2$ (MLP) = **$12d^2$**.
- Total parameter count $\approx 2Vd + L \times 12d^2$.

*Take a configuration with about 7B parameters as an example: $d=4096, L=32, V=32000$*
*Block parameters = $32 \times 12 \times 4096^2 \approx 6.4 \text{ Billion}$*
*Embedding = $2 \times 32000 \times 4096 \approx 0.26 \text{ Billion}$*
*The total is about 6.7B, which is the so-called 7B model!*

### Q1 Mini Check: Parameter count computation

Run the reference function, explicitly choose the standard FFN or SwiGLU, and compare the parameter counts with and without shared word embeddings. A failed assertion raises an error directly, making it easy to locate a formula or implementation deviation.


```python
def calculate_transformer_params(vocab_size, hidden_dim, num_layers, intermediate_size=None, tie_embeddings=False, ffn_type='swiglu'):
    """Estimate the parameter count from the simplified structure of a dense Decoder-only Transformer.

    Ignores bias, specific norm variants, and MoE routing differences; the result is for an order-of-magnitude budget
    and does not replace the parameter statistics of a specific model config or checkpoint.
    ffn_type='dense' uses two FFN matrices; ffn_type='swiglu' uses three matrices.
    """
    if min(vocab_size, hidden_dim, num_layers) <= 0:
        raise ValueError('vocab_size、hidden_dim 和 num_layers 必须为正数')
    if ffn_type not in {'dense', 'swiglu'}:
        raise ValueError("ffn_type 必须为 'dense' 或 'swiglu'")
    if intermediate_size is not None and intermediate_size <= 0:
        raise ValueError('intermediate_size 必须为正数')
    if intermediate_size is None:
        intermediate_size = 4 * hidden_dim if ffn_type == 'dense' else round(8 * hidden_dim / 3)

    embedding_params = vocab_size * hidden_dim
    attention_params = num_layers * (4 * hidden_dim * hidden_dim)
    ffn_matrix_count = 2 if ffn_type == 'dense' else 3
    ffn_params = num_layers * (ffn_matrix_count * hidden_dim * intermediate_size)
    layernorm_params = num_layers * (2 * hidden_dim)
    lm_head_params = 0 if tie_embeddings else vocab_size * hidden_dim

    total_params = embedding_params + attention_params + ffn_params + layernorm_params + lm_head_params
    return total_params
```


```python
def test_calculate_transformer_params():
    result = calculate_transformer_params(1000, 64, 2, 256, tie_embeddings=True)
    assert result == 195328, f"错误：期望 195328，实际 {result}"

    result = calculate_transformer_params(1000, 64, 2, 256, tie_embeddings=False)
    assert result == 259328, f"错误：期望 259328，实际 {result}"
    dense = calculate_transformer_params(1000, 64, 2, 256, tie_embeddings=True, ffn_type='dense')
    assert dense == 162560, f"标准 FFN 参数量不符合预期，实际 {dense}"

    result = calculate_transformer_params(32000, 4096, 32, 11008, tie_embeddings=False)
    assert result > 6000000000, f"错误：LLaMA 级别模型参数量应大于 6B，实际 {result}"

    print("✅ 参数量函数测试通过！")

test_calculate_transformer_params()
```

### Q1 Extended Check: Estimate the parameter count of a 7B-level configuration

Change the vocabulary, hidden dimension, layer count, or feed-forward intermediate dimension and observe the total parameter count; change only one item at a time when comparing.


```python
vocab_size = 32000
hidden_dim = 4096
num_layers = 32
intermediate_size = 11008

total_params = calculate_transformer_params(vocab_size, hidden_dim, num_layers, intermediate_size, tie_embeddings=False)
print(f"估算参数量: {total_params / 1e9:.2f}B")
print("提示：不同实现会因为是否共享词嵌入、是否计入偏置而略有差异。")
```

## Q2: Given the parameter count and token count, how many FLOPs does one forward pass need?


This question computes only the FLOPs of one forward pass; why the training stage needs more compute is broken down in the next question. The formula here is an order-of-magnitude estimate for a dense Transformer and does not treat forward FLOPs as the complete training cost.

Having understood the parameter count, let us look at how much compute a large model needs for inference (forward propagation).

**Core order-of-magnitude estimate: for a dense Transformer, 1 parameter processing 1 token is commonly estimated at about 2 floating-point operations (FLOPs).**
Why 2? Because in the matrix multiplication $Y = W \times X$, for each weight element we need one **multiplication** and one **addition** (Multiply-Accumulate, MAC).

**Inference FLOPs formula:**
$$ C_{forward} \approx 2 \times P \times T $$
where:
- $C_{forward}$ is the compute needed for forward propagation (FLOPs)
- $P$ is the total parameter count of the model (Parameters)
- $T$ is the number of tokens processed (Tokens)

*(Note: this ignores the Attention matrix products, normalization, activation functions, padding, and other operators; their share changes with sequence length, model structure, and batch, and cannot be fixed at 99%.)*

### Q2 Mini Check: Compute forward FLOPs from the parameter count and token count

First estimate one forward pass with `2 × parameter count × token count`. The breakdown of training FLOPs is left to the next question, to avoid treating the formula's conclusion as a mechanism explanation in advance.


```python
def calculate_forward_flops(num_params_b, num_tokens, flops_per_param_token=2):
    """Roughly estimate the FLOPs of one forward pass of a dense Transformer.

    The default 2 FLOPs/parameter/token is a teaching approximation for multiply-accumulate operations,
    excluding sequence-length-dependent attention terms, normalization, and other operators.
    """
    if num_params_b < 0 or num_tokens < 0 or flops_per_param_token < 0:
        raise ValueError('参数量、token 数和 FLOPs 系数不能为负数')
    return num_params_b * 1_000_000_000 * num_tokens * flops_per_param_token

```


```python
def test_calculate_flops():
    result = calculate_forward_flops(7, 1_000_000_000_000)
    assert result == 14_000_000_000_000_000_000_000, f"错误：期望 1.4e22，实际 {result}"

    print("✅ FLOPs 函数测试通过！")

test_calculate_flops()
```

### Q2 Extended Check: Observe the sensitivity of forward FLOPs to the token count

Fix the parameter count and compare the forward FLOPs under different token counts; the training-time estimate is left to Q3.


```python
def compare_forward_token_scales(num_params_b, token_counts):
    """Compare the forward FLOPs under different token counts; the result is not measured elapsed time."""
    if not token_counts or any(tokens < 0 for tokens in token_counts):
        raise ValueError('token_counts 不能为空且不能包含负数')
    return {tokens: calculate_forward_flops(num_params_b, tokens) for tokens in token_counts}

print('7B 模型不同 token 数的前向 FLOPs：')
for tokens, flops in compare_forward_token_scales(7, [1_000, 8_000, 32_000]).items():
    print(f'{tokens:>6} tokens -> {flops:.2e} FLOPs')

```

## Q3: Why does full training need more FLOPs than one forward pass?


Training not only includes forward propagation to compute the loss, but also backward propagation to compute gradients.

In backward propagation, we need to:
1. Compute the gradients of the activations, so that the error can continue to be propagated backward (about $2 \times P \times T$ FLOPs).
2. Compute the gradients of the weights, used to update the model parameters (about $2 \times P \times T$ FLOPs as well).

Therefore, the compute of backward propagation is about **2 times** that of forward propagation.

**Training FLOPs formula (order-of-magnitude approximation):**
$$ C_{train} = C_{forward} + C_{backward} \approx 2PT + 4PT = 6 \times P \times T $$

**Accounting for different training modes:**
| Training mode | Forward and backward | Optimizer state | Base weights | Treatment in this section |
|---|---|---|---|---|
| Full-parameter training / fine-tuning | The whole model participates | Covers all parameters | Usually updatable | Used as the `6PT` baseline |
| LoRA | Passes through the base and updates the Adapter | Mainly maintains the Adapter | Usually frozen | Only a conceptual comparison |
| QLoRA | The quantized base participates in computation and the Adapter is updated | Mainly maintains the Adapter | Low-bit frozen | Only a resource-boundary description |

The gains of LoRA and QLoRA cannot be judged from the FLOPs formula alone; activations, quantize-dequantize, kernels, and the actual workload still need to be measured in the Part 02 projects.

**Practical estimate:**
Suppose we want to pretrain a LLaMA-7B (7 billion parameters) model from scratch with 1T (1 trillion) tokens of training data.
The required total theoretical compute is $C = 6 \times (7 \times 10^9) \times (1 \times 10^{12}) = 4.2 \times 10^{22}$ FLOPs.

If we estimate with 312 TFLOPs theoretical peak per A100, 35% effective efficiency, and 1000 GPUs, the effective cluster compute is about $1000 \times 312 \times 0.35$ TFLOPs, and the training time is about 4.5 days. This result changes with the efficiency assumption and is only used to build order-of-magnitude intuition.

### Q3 Mini Check: Break down training FLOPs and estimate scenario time

First split forward, activation gradients, and weight gradients, then feed the total FLOPs into training-time estimates for different GPU scales.


```python
def calculate_training_flops(num_params_b, num_tokens, flops_per_param_token=6):
    """Roughly estimate the training FLOPs of a dense Transformer; not the real training elapsed time."""
    if num_params_b < 0 or num_tokens < 0 or flops_per_param_token < 0:
        raise ValueError('参数量、token 数和 FLOPs 系数不能为负数')
    return num_params_b * 1_000_000_000 * num_tokens * flops_per_param_token


def estimate_training_time(num_params_b, num_tokens, gpu_tflops, num_gpus, efficiency=0.35):
    """Estimate the time from theoretical training FLOPs and an assumed efficiency; not measured GPU time."""
    if gpu_tflops <= 0 or num_gpus <= 0 or not 0 < efficiency <= 1:
        raise ValueError('gpu_tflops、num_gpus 必须为正数，efficiency 必须在 (0, 1]')
    total_flops = calculate_training_flops(num_params_b, num_tokens)
    effective_flops = gpu_tflops * 1e12 * num_gpus * efficiency
    return total_flops / effective_flops / 3600


def training_flops_breakdown(num_params_b, num_tokens):
    """Split out the forward, backward, and total training FLOPs; an order-of-magnitude model for a dense Transformer."""
    forward = calculate_forward_flops(num_params_b, num_tokens)
    activation_backward = 2 * num_params_b * 1_000_000_000 * num_tokens
    weight_backward = 2 * num_params_b * 1_000_000_000 * num_tokens
    return {
        'forward': forward,
        'activation_backward': activation_backward,
        'weight_backward': weight_backward,
        'total': forward + activation_backward + weight_backward,
    }

breakdown = training_flops_breakdown(7, 1_000_000_000_000)
assert breakdown['total'] == calculate_training_flops(7, 1_000_000_000_000)
assert calculate_training_flops(7, 1_000_000_000_000) == 3 * calculate_forward_flops(7, 1_000_000_000_000)
print('训练 FLOPs 拆分:', {key: f'{value:.2e}' for key, value in breakdown.items()})

scenarios = [
    ('1x A100 80GB', 312, 1, 0.35),
    ('8x A100 80GB', 312, 8, 0.35),
    ('1000x A100 80GB', 312, 1000, 0.35),
    ('8x H100 80GB', 500, 8, 0.45),
]

print('LLaMA-7B 训练时间粗估（1T tokens）：')
print('-' * 70)
for name, gpu_tflops, num_gpus, eff in scenarios:
    hours = estimate_training_time(7, 1_000_000_000_000, gpu_tflops, num_gpus, eff)
    print(f"{name:<16} {hours:>12.1f} 小时  ({hours/24:>8.1f} 天)")
```

## Q4: When training large models, what is compute utilization (MFU, Model FLOPs Utilization)?


Through Q3 above we computed the **theoretically required compute**. But in real engineering, the hardware does not spend all its time on matrix multiplication. This brings in MFU; it is an important indicator for observing model compute utilization, but it cannot by itself represent training engineering quality.

- **Theoretical compute (Peak FLOPs)**: the compute written on the graphics card specification. For example, the theoretical BF16 peak of A100 is 312 TFLOPs (312 trillion floating-point operations per second).
- **Actual compute (Observed FLOPs)**: that is, the theoretical computation volume $6PT$ required for the whole training divided by the **actual time** spent completing these steps.
- **MFU = actual model FLOPs / theoretical hardware peak FLOPs**. Actual model FLOPs usually need to be estimated together with the model structure and token count; a time-share ratio cannot be treated directly as the MFU.

**Why is MFU hard to reach 100%?**
Because in a real training cluster there are the **Memory-bound (memory wall)** and **Communication (communication bottleneck)** problems. The GPU spends a lot of time waiting for data to be moved from memory, or waiting for the All-Reduce data from other machines, without doing effective multiply-add operations.

MFU has no universal passing line independent of the model, precision, hardware, parallel mode, and statistical convention. A low MFU only indicates that further checks are needed and cannot by itself prove that communication or I/O is the main bottleneck; it should be handed to performance analysis together with timeline, memory-access, communication, and input-pipeline records.

### Q4 Mini Check: Compute time-share and bottleneck decomposition

Separate compute, memory waiting, and communication waiting to observe which parts of the timeline are slowing down execution. Both the MFU formula estimate and the timeline share are given below: the former uses model FLOPs, actual elapsed time, and hardware peak, and the latter only describes the proportion of compute and waiting on the timeline; the two cannot replace each other.


```python
def calculate_mfu(model_flops, elapsed_seconds, peak_tflops):
    """Estimate MFU from model FLOPs, actual elapsed time, and hardware peak; not a profiler measurement."""
    if model_flops < 0 or elapsed_seconds <= 0 or peak_tflops <= 0:
        raise ValueError('model_flops 不能为负数，elapsed_seconds 和 peak_tflops 必须为正数')
    achieved_flops = model_flops / elapsed_seconds
    return achieved_flops / (peak_tflops * 1e12)


def compute_time_fraction(compute_ms, memory_wait_ms, comm_wait_ms):
    total_ms = compute_ms + memory_wait_ms + comm_wait_ms
    if total_ms <= 0:
        return {'compute_time_fraction': 0.0, 'dominant_stall': 'none', 'stall_ratio': 0.0}
    compute_fraction = compute_ms / total_ms
    stalls = {'memory': memory_wait_ms, 'communication': comm_wait_ms}
    dominant = max(stalls, key=stalls.get)
    return {
        'compute_time_fraction': round(compute_fraction, 3),
        'dominant_stall': dominant,
        'stall_ratio': round((memory_wait_ms + comm_wait_ms) / total_ms, 3),
    }

cases = [
    (100, 30, 20),
    (100, 60, 40),
    (100, 10, 5),
]
for case in cases:
    print(case, '->', compute_time_fraction(*case))
print('MFU example:', round(calculate_mfu(1e15, 10, 312), 4))
assert 0 < calculate_mfu(1e15, 10, 312) < 1
assert compute_time_fraction(0, 0, 0)['dominant_stall'] == 'none'
print('A high compute-time fraction does not by itself prove high MFU')

```

---

## Related Reading
This section can continue into model configuration, the memory budget, and the MoE parameter-compute ledger.
- [Hugging Face Transformers model configuration](https://huggingface.co/docs/transformers/main_classes/configuration): see how hidden size, layer count, vocabulary, and attention configuration feed into the parameter estimate.
- [06. VRAM Calculation and ZeRO](./06_VRAM_Calculation_and_ZeRO.md)
- [22. MoE Parameter and Compute](./22_MoE_Parameter_and_Compute.md)

  
---
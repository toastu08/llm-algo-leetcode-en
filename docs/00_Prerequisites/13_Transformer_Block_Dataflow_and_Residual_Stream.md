# 13. Transformer Block Dataflow and Residual Stream | Transformer Block 数据流与残差通路

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `Transformer`, `Block`, `Residual` | **Audience:** learners preparing to understand how modern LLM components are assembled

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/13_Transformer_Block_Dataflow_and_Residual_Stream.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## Section Guide

A Transformer Block continuously reads and updates the same residual stream. The Attention branch is responsible for information aggregation across tokens, and the MLP branch for per-token channel transformation; both branches must restore the uniform hidden shape before writing back to the main path.

After completing this section, you should be able to trace the two residual updates of one Block, distinguish the main-path state from the branch updates, and check whether multi-layer stacking preserves the shape and advances the state.

**Keywords:** `hidden state`, `residual stream`, `branch update`, `Block stack`

![Transformer Block dataflow and residual path](../public/00_Prerequisites/13_transformer_block_dataflow.svg)

## Prerequisite Reading
**Overview:** The model object and training lifecycle explain how a Block is called and updated; this page further observes how the hidden state contract is maintained inside the Block.
- [12. PyTorch Data Interface and Training Lifecycle](./12_PyTorch_Minimal_Training_Interface.md)
- [0D Group Page](./0D.md)

## Q1: Why do a Block's input and output usually keep the same hidden shape?

Residual addition requires the main path and the update branch to have the same shape. A Block may internally split heads or expand the MLP intermediate dimension, but before writing back to the residual stream it must restore `[batch, sequence, hidden]`.


```python
import torch
import torch.nn as nn


def residual_update(hidden_states, branch_update):
    """Check the residual contract, then write the branch update back to the main path."""
    if hidden_states.shape != branch_update.shape:
        raise ValueError(
            f'residual shape mismatch: {tuple(hidden_states.shape)} != {tuple(branch_update.shape)}'
        )
    return hidden_states + branch_update


hidden_states = torch.randn(2, 4, 8)
branch_update = torch.randn_like(hidden_states)
updated_states = residual_update(hidden_states, branch_update)
print('residual shape：', tuple(updated_states.shape))

```

## Q1 verification: Does residual addition preserve the shape and reject invalid updates?

A valid update should preserve the main-path shape; when the hidden dimension differs, an explicit error should be raised before the addition.


```python
assert updated_states.shape == hidden_states.shape
torch.testing.assert_close(updated_states, hidden_states + branch_update)
try:
    residual_update(hidden_states, torch.randn(2, 4, 4))
except ValueError as error:
    assert 'residual shape mismatch' in str(error)
else:
    raise AssertionError('shape 不一致时 residual update 应失败')
print('✅ residual shape contract 通过')

```

## Q2: How do Attention and MLP form two residual updates?

A modern decoder Block usually first writes the Attention output back to the residual stream, and then writes the MLP output back to the updated main path. Below, shape-preserving branches represent token mixing and channel mixing respectively, to verify the order of the two updates and the hidden shape contract.


```python
class ResidualBlockSkeleton(nn.Module):
    """Represent only the interface of two shape-preserving branches; does not replace real Attention/MLP."""

    def __init__(self, hidden_size):
        super().__init__()
        self.attention_branch = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_branch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size, bias=False),
        )

    def forward(self, hidden_states):
        attention_update = self.attention_branch(hidden_states)
        after_attention = residual_update(hidden_states, attention_update)
        mlp_update = self.mlp_branch(after_attention)
        output = residual_update(after_attention, mlp_update)
        return {
            'input': hidden_states,
            'attention_update': attention_update,
            'after_attention': after_attention,
            'mlp_update': mlp_update,
            'output': output,
        }


torch.manual_seed(0)
block = ResidualBlockSkeleton(hidden_size=8)
block_states = block(torch.randn(2, 4, 8))
print({name: tuple(value.shape) for name, value in block_states.items()})

```

## Q2 verification: Are the two branch updates written back to the residual stream in order?

The five observed quantities should keep the same shape; `after_attention` must equal the input plus the Attention update, and the final output must equal the updated main path plus the MLP update.


```python
assert all(value.shape == block_states['input'].shape for value in block_states.values())
torch.testing.assert_close(
    block_states['after_attention'],
    block_states['input'] + block_states['attention_update'],
)
torch.testing.assert_close(
    block_states['output'],
    block_states['after_attention'] + block_states['mlp_update'],
)
print('✅ Attention/MLP 两次 residual update 通过')

```

## Q3: Why distinguish the residual state from the branch update?

The residual state is the main-path state passed to the next branch and the next layer; the branch update is the increment produced by the current sublayer. The two have the same shape but different meanings. Only by recording the increment separately can you check where each write-back happens, instead of mistaking "shape preserved" for "state unchanged".


```python
def root_mean_square(tensor):
    return tensor.pow(2).mean().sqrt()


state_and_update_rms = {
    name: float(root_mean_square(block_states[name]))
    for name in ('input', 'attention_update', 'after_attention', 'mlp_update', 'output')
}
for name, value in state_and_update_rms.items():
    print(f'{name:>18}: RMS={value:.4f}')

```

## Q3 verification: Can the main path and the two increments be reconstructed separately?

Both updates should be finite values, and they should separately reconstruct the state after the Attention write-back and after the MLP write-back. What is verified here is the state relationship; deep training stability is not judged from it.


```python
for name in ('attention_update', 'mlp_update'):
    assert torch.isfinite(block_states[name]).all()
assert not torch.equal(block_states['input'], block_states['after_attention'])
assert not torch.equal(block_states['after_attention'], block_states['output'])
torch.testing.assert_close(
    block_states['output'] - block_states['after_attention'],
    block_states['mlp_update'],
)
print('✅ residual state 与 branch update 已分离验证')

```

## Q4: When multiple Blocks are stacked, how does the residual stream evolve layer by layer?

The Block stack passes the same hidden-state contract between layers, and each layer reads the previous layer's output and produces a new residual state. Recording the layer-by-layer states can check the shape, the number of layers, and whether the state actually advances; whether the deep-layer values and gradients are stable requires further verification in architecture and training experiments.


```python
class BlockStack(nn.Module):
    def __init__(self, hidden_size, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlockSkeleton(hidden_size) for _ in range(num_blocks)
        ])

    def forward(self, hidden_states):
        layer_states = [hidden_states]
        for block in self.blocks:
            hidden_states = block(hidden_states)['output']
            layer_states.append(hidden_states)
        return layer_states


stack = BlockStack(hidden_size=8, num_blocks=3)
stack_input = torch.randn(2, 4, 8)
layer_states = stack(stack_input)
print('layer state shapes：', [tuple(state.shape) for state in layer_states])

```

## Q4 verification: Are the layer count, shape, and state advancement consistent?

Three Blocks should produce four observable states; each state keeps the uniform shape, and the output of adjacent layers should change.


```python
assert len(layer_states) == 4
assert all(state.shape == stack_input.shape for state in layer_states)
assert all(not torch.equal(left, right) for left, right in zip(layer_states, layer_states[1:]))
assert len(stack.blocks) == 3
print('✅ Block stack 与 residual state 推进通过')

```

## Related Reading

After the Block skeleton, you can go deeper into nonlinearity, the statistical path, and token information aggregation separately; design evolution and deep-layer stability are expanded in the architecture topic.

- [14. Activation Functions](./14_Activation_Functions.md)
- [15. Normalization Techniques](./15_Normalization_Techniques.md)
- [16. Attention Mechanism Intro](./16_Attention_Mechanism_Intro.md)
- [LLM Architecture Evolution · Transformer block and residual backbone](../topic_discussion/llm_architecture_evolution/06_block_residual_path.md)

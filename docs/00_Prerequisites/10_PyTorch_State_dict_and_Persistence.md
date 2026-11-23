# 10. PyTorch State dict and Persistence | PyTorch 状态、Checkpoint 与恢复

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `PyTorch`, `state_dict`, `persistence` | **Audience:** learners preparing to save, restore, or migrate model state

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/10_PyTorch_State_dict_and_Persistence.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## Section Guide

Model persistence must handle the state contents, independent snapshots, structural compatibility, and training progress at the same time. Saving only the model weights cannot fully restore the optimization trajectory and validation state.

After completing this section, you should be able to distinguish a model artifact from a training checkpoint, judge whether state can be migrated and restored, and identify the objects missing for resuming training.

**Keywords:** `state_dict`, `artifact`, `training checkpoint`, `resume`

![Saving, migrating, and restoring model state](../public/00_Prerequisites/10_state_persistence_map.svg)

## Prerequisite Reading
**Overview:** First confirm how parameters and buffers are registered, then observe how this state leaves the current model instance and is restored.
- [09. PyTorch nn.Module Basics](./09_PyTorch_nn_Module_Basics.md)
- [0C Group Page](./0C.md)

## Q1: What problem does `state_dict` solve?

A model not only contains trainable parameters, but may also contain buffers such as running statistics, masks, or scale values. `state_dict` uses stable keys to collect these tensors into the model state, but does not save the Python class definition or ordinary attributes.


```python
import copy
import random
import torch
import torch.nn as nn


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        # A buffer is often used for state such as scale, masks, statistics, or caches that "should be saved but not trained".
        self.register_buffer('scale', torch.tensor(1.0))

    def forward(self, x):
        return self.linear(x) * self.scale


model = TinyNet()
state = model.state_dict()
print('state_dict keys：', sorted(state.keys()))

```

## Q1 verification: Does `state_dict` contain both parameters and buffers?

Look at the keys directly here to confirm that both parameters and buffers are present, while ordinary attributes are not.


```python
model = TinyNet()
state = model.state_dict()
assert 'linear.weight' in state
assert 'linear.bias' in state
assert 'scale' in state
print('✅ state_dict keys 通过')

```

## Q2: How do you write model state to an independent medium and restore it into a new instance?

Loading a model's own `state_dict` straight back into it cannot verify the serialization path. A more complete check should write the state to an independent byte stream, and then have a newly created model instance read and restore it.


```python
import io

model = TinyNet()
with torch.no_grad():
    model.linear.weight.fill_(2.0)
    model.linear.bias.fill_(0.5)
    model.scale.fill_(3.0)

state_buffer = io.BytesIO()
torch.save(model.state_dict(), state_buffer)
state_buffer.seek(0)
loaded_state = torch.load(state_buffer, weights_only=True, map_location='cpu')
restored_model = TinyNet()
restored_model.load_state_dict(loaded_state)
print('序列化字节数：', state_buffer.getbuffer().nbytes)

```

## Q2 verification: Are the new instance's parameters, buffers, and outputs identical?

After restoring the state, you should compare not only the keys but also the tensor values and the model output on the same input.


```python
for key, value in model.state_dict().items():
    assert torch.allclose(value, restored_model.state_dict()[key])
sample = torch.tensor([[1.0, -1.0]])
assert torch.allclose(model(sample), restored_model(sample))
assert restored_model.scale.item() == 3.0
print('✅ 独立序列化与恢复通过')

```

## Q3: When the structure changes, how do strict loading and partial loading differ?

Strict loading requires the current model and the saved state to have exactly the same keys. During structural migration you can use `strict=False`, but you must check the returned missing and unexpected keys, and must not treat "no exception was raised" as a successful migration.


```python
class TinyNetWithHead(TinyNet):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(1, 2)


base_state = TinyNet().state_dict()
migrated_model = TinyNetWithHead()
try:
    migrated_model.load_state_dict(base_state, strict=True)
except RuntimeError as error:
    strict_error = str(error)
else:
    raise AssertionError('strict loading should fail for the added head')

load_result = migrated_model.load_state_dict(base_state, strict=False)
print('missing keys：', load_result.missing_keys)
print('unexpected keys：', load_result.unexpected_keys)

```

## Q3 verification: Does the migration report accurately point out the newly added structure?

The added head should become a missing key, and the original model state should not produce unexpected keys. The strict-mode error message should also point out the missing entries.


```python
assert set(load_result.missing_keys) == {'head.weight', 'head.bias'}
assert load_result.unexpected_keys == []
assert 'head.weight' in strict_error and 'head.bias' in strict_error
print('✅ strict / non-strict 加载报告通过')

```

## Q4: How do you make a model artifact carry checkable version information?

A bare `state_dict` can only show which tensors exist. A maintainable model artifact should also record the format version, the model type, and the construction parameters; the load function should return a structured report so that the caller can see exactly which state was restored.


```python
def build_model_artifact(model, format_version, model_config):
    """Package the model state together with the minimal metadata needed to rebuild the model."""
    return {
        'format_version': format_version,
        'model_class': type(model).__name__,
        'model_config': dict(model_config),
        'model_state': copy.deepcopy(model.state_dict()),
    }


def load_model_artifact(artifact, model, strict=True):
    """Load the model state and return an auditable compatibility report."""
    result = model.load_state_dict(artifact['model_state'], strict=strict)
    return {
        'format_version': artifact['format_version'],
        'model_class': artifact['model_class'],
        'missing_keys': list(result.missing_keys),
        'unexpected_keys': list(result.unexpected_keys),
    }


artifact_model = TinyNet()
artifact = build_model_artifact(artifact_model, format_version=1, model_config={'in_features': 2})
report = load_model_artifact(artifact, TinyNet(), strict=True)
print(report)

```

## Q4 verification: Are the model artifact and the load report consistent?

The report needs to preserve the format version and the model type, and make clear that strict loading has no missing or unexpected keys. As the model keeps changing, artifacts that have already been generated must also remain independent snapshots.


```python
assert artifact['format_version'] == 1
assert artifact['model_class'] == 'TinyNet'
assert artifact['model_config'] == {'in_features': 2}
assert report['missing_keys'] == []
assert report['unexpected_keys'] == []
saved_weight = artifact['model_state']['linear.weight'].clone()
with torch.no_grad():
    artifact_model.linear.weight.add_(1.0)
assert torch.equal(artifact['model_state']['linear.weight'], saved_weight)
print('✅ 模型产物与加载报告通过')

```

## Q5: How do you make the best metric, early stopping, and a resumable checkpoint use the same training state?

When the validation metric improves, you need to update the best metric, save the corresponding model state, and reset the no-improvement counter; only on consecutive non-improvements does the early-stopping counter advance. A checkpoint used to resume training must save the computational state, update state, and training progress at the same point in time.

| State category | What this example saves | Direct impact if missing |
|:---|:---|:---|
| Model state | parameters and buffers | The forward results differ after restoring |
| Optimizer state | AdamW steps and moment estimates | The parameter update trajectory restarts |
| Scheduler state | current learning-rate stage | The learning-rate advancement position is wrong |
| Random state | Python, PyTorch CPU; CUDA too when already initialized | Random inputs, sampling, or random operators cannot be reproduced |
| Training progress | epoch, micro-step, optimizer-step | Data position and update count cannot be aligned |
| Validation state | best metric, no-improvement epochs, best model | Model selection and early-stopping judgment are lost |


```python
def update_validation_state(val_loss, state, model):
    """Update the best checkpoint and the early-stopping counter based on the validation loss."""
    updated = dict(state)
    if val_loss < state['best_val_loss']:
        updated['best_val_loss'] = float(val_loss)
        updated['bad_epochs'] = 0
        updated['best_model_state'] = copy.deepcopy(model.state_dict())
    else:
        updated['bad_epochs'] = state['bad_epochs'] + 1
    updated['should_stop'] = updated['bad_epochs'] >= updated['patience']
    return updated


def capture_rng_state():
    """Save the Python and PyTorch random state actually used by this process."""
    return {
        'python': random.getstate(),
        'torch_cpu': torch.get_rng_state().clone(),
        'torch_cuda': (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() and torch.cuda.is_initialized()
            else None
        ),
    }


def restore_rng_state(rng_state):
    random.setstate(rng_state['python'])
    torch.set_rng_state(rng_state['torch_cpu'])
    if rng_state['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all(rng_state['torch_cuda'])


def build_training_checkpoint(model, optimizer, scheduler, progress, validation_state):
    """Save the model, updaters, progress, and RNG state needed to continue training."""
    return {
        'format_version': 1,
        'model': copy.deepcopy(model.state_dict()),
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'scheduler': copy.deepcopy(scheduler.state_dict()),
        'progress': {key: int(value) for key, value in progress.items()},
        'validation_state': copy.deepcopy(validation_state),
        'rng_state': capture_rng_state(),
    }


def restore_training_checkpoint(checkpoint, model, optimizer, scheduler):
    """Restore the training state and return the progress fields."""
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    restore_rng_state(checkpoint['rng_state'])
    return {
        'progress': copy.deepcopy(checkpoint['progress']),
        'validation_state': copy.deepcopy(checkpoint['validation_state']),
    }


def run_random_training_step(model, optimizer, scheduler):
    """Build a reproducible resume check step using the Python and PyTorch RNGs."""
    random_scale = random.uniform(0.5, 1.5)
    inputs = torch.randn(3, 2) * random_scale
    target = inputs.sum(dim=1, keepdim=True)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(inputs), target)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach()), inputs.clone()


random.seed(17)
torch.manual_seed(17)
training_model = TinyNet()
training_optimizer = torch.optim.AdamW(training_model.parameters(), lr=1e-2)
training_scheduler = torch.optim.lr_scheduler.StepLR(training_optimizer, step_size=1, gamma=0.5)
training_optimizer.zero_grad(set_to_none=True)
training_model(torch.ones(2, 2)).mean().backward()
training_optimizer.step()
training_scheduler.step()
validation_state = {
    'best_val_loss': float('inf'), 'bad_epochs': 0, 'patience': 2,
    'should_stop': False, 'best_model_state': None,
}
validation_state = update_validation_state(0.4, validation_state, training_model)
training_checkpoint = build_training_checkpoint(
    training_model, training_optimizer, training_scheduler,
    progress={'epoch': 3, 'micro_step': 12, 'optimizer_step': 6},
    validation_state=validation_state,
)
checkpoint_weight_before_continuation = training_checkpoint['model']['linear.weight'].clone()
reference_loss, reference_inputs = run_random_training_step(
    training_model, training_optimizer, training_scheduler
)
reference_model_state = copy.deepcopy(training_model.state_dict())
reference_optimizer_state = copy.deepcopy(training_optimizer.state_dict())
print('checkpoint fields：', sorted(training_checkpoint))

```

## Q5 verification: Can the best state and the training progress be restored together?

Equal fields only prove that the objects were read back. A stronger check is: after restoring the random state, run one more step, and the random inputs, loss, and updated parameters should all match the uninterrupted path; the checkpoint itself must also remain an independent snapshot.


```python
saved_weight = checkpoint_weight_before_continuation
assert torch.equal(training_checkpoint['model']['linear.weight'], saved_weight)
assert not torch.equal(training_model.linear.weight, saved_weight)
random.random()
torch.rand(4)

restored_training_model = TinyNet()
restored_optimizer = torch.optim.AdamW(restored_training_model.parameters(), lr=1e-2)
restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1, gamma=0.5)
restored_state = restore_training_checkpoint(
    training_checkpoint, restored_training_model, restored_optimizer, restored_scheduler
)
assert torch.equal(restored_training_model.state_dict()['linear.weight'], saved_weight)
assert restored_state['progress'] == {'epoch': 3, 'micro_step': 12, 'optimizer_step': 6}
assert restored_state['validation_state']['best_val_loss'] == 0.4
assert restored_state['validation_state']['best_model_state'] is not None
restored_loss, restored_inputs = run_random_training_step(
    restored_training_model, restored_optimizer, restored_scheduler
)
assert torch.equal(restored_inputs, reference_inputs)
assert restored_loss == reference_loss
for key, value in reference_model_state.items():
    assert torch.equal(restored_training_model.state_dict()[key], value)
for parameter_id, state_values in reference_optimizer_state['state'].items():
    for state_name, state_value in state_values.items():
        restored_value = restored_optimizer.state_dict()['state'][parameter_id][state_name]
        assert torch.equal(restored_value, state_value) if torch.is_tensor(state_value) else restored_value == state_value
assert restored_scheduler.state_dict() == training_scheduler.state_dict()
first_bad = update_validation_state(0.5, validation_state, training_model)
second_bad = update_validation_state(0.6, first_bad, training_model)
assert first_bad['bad_epochs'] == 1 and not first_bad['should_stop']
assert second_bad['bad_epochs'] == 2 and second_bad['should_stop']
assert validation_state['bad_epochs'] == 0
improved = update_validation_state(0.3, second_bad, training_model)
assert improved['bad_epochs'] == 0 and not improved['should_stop']
assert improved['best_val_loss'] == 0.3
print('✅ 最佳状态、早停字段与可续训 checkpoint 通过')

```

## Related Reading

Now that state can be restored reliably, continue with optimizer updates, and connect them into a complete training lifecycle on the data interface page.

- [11. PyTorch Optimizers and Loss](./11_PyTorch_Optimizers_and_Loss.md)
- [12. PyTorch Data Interface and Training Lifecycle](./12_PyTorch_Minimal_Training_Interface.md)
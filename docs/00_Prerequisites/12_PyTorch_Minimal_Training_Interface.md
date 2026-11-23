# 12. PyTorch Minimal Training Interface | PyTorch 数据接口与训练生命周期

**Difficulty:** Medium | **Environment:** CPU-first | **Tags:** `PyTorch`, `batch contract`, `training lifecycle` | **Audience:** learners preparing to connect the data entry point and model updates into a training loop

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/12_PyTorch_Minimal_Training_Interface.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## Section Guide

The training lifecycle starts from raw samples, goes through tokenization, batch assembly, model computation, loss, backward propagation, and parameter updates, and finally enters independent validation and metric evaluation. Any inconsistent interface will let errors propagate through the subsequent stages.

After completing this section, you should be able to trace how a single sample becomes model input, check the training and validation boundaries, and confirm that evaluation metrics are aggregated over the complete sample set.

**Keywords:** `Dataset`, `DataLoader`, `batch contract`, `train/validate`, `evaluation metrics`

![The batch contract from a single sample to model input](../public/00_Prerequisites/12_batch_contract_map.svg)

## Prerequisite Reading
**Overview:** First review how the optimizer and loss function consume model outputs, then establish the data entry point of the training loop on this page.
- [11. PyTorch Optimizers and Loss](./11_PyTorch_Optimizers_and_Loss.md)
- [0C Group Page](./0C.md)

## Q1: What problems do Dataset and DataLoader solve respectively?

`Dataset` defines the fields of a single sample, and `DataLoader` handles sampling, batching, and iteration. Once sample-level and batch-level logic are separated, fields such as `attention_mask`, `labels`, and `position_ids` can be generated and validated at well-defined positions.


```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ToyDataset(Dataset):
    def __init__(self, sample_ids=(0, 1, 2, 3)):
        self.sample_ids = tuple(sample_ids)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        idx = self.sample_ids[idx]
        # The sample id determines the input and label; training and validation use different ids.
        return {
            'input_ids': torch.tensor([idx], dtype=torch.long),
            'attention_mask': torch.tensor([1], dtype=torch.long),
            'labels': torch.tensor([idx % 2], dtype=torch.long),
        }


loader = DataLoader(ToyDataset(), batch_size=2, shuffle=False)
for batch in loader:
    print('batch keys:', list(batch.keys()))
    print('input_ids shape:', batch['input_ids'].shape, 'labels shape:', batch['labels'].shape)

```

## Q1 verification: Is the batch output shape stable?

Here it directly confirms whether the dimensions of the batch taken from `DataLoader` match expectations. What you should remember is that a batch is not always a tuple; a dict batch is more common in large-model training.


```python
loader = DataLoader(ToyDataset(), batch_size=2, shuffle=False)
batches = list(loader)
assert len(batches) == 2
batch = batches[0]
assert batch['input_ids'].shape == (2, 1)
assert batch['attention_mask'].shape == (2, 1)
assert batch['labels'].shape == (2, 1)
print('✅ Dataset / DataLoader 通过')

```

## Q2: How does `collate_fn` organize single samples into a batch?

`Dataset` returns a single sample, and `collate_fn` determines how samples in the same batch are stacked by field. The output must specify the field names, batch dimension, and dtype so that the training loop can unpack it stably.


```python
def collate_pair(batch):
    # With a custom collate, first unpack the single samples, then reassemble them according to the current task.
    # torch.stack adds a batch dimension while preserving the original sequence dimension of each sample.
    input_ids = torch.stack([item['input_ids'] for item in batch], dim=0)
    attention_mask = torch.stack([item['attention_mask'] for item in batch], dim=0)
    labels = torch.stack([item['labels'] for item in batch], dim=0)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


loader = DataLoader(ToyDataset(), batch_size=2, shuffle=False, collate_fn=collate_pair)
for batch in loader:
    print('collate 后 keys:', list(batch.keys()))
    print('input_ids:', batch['input_ids'].tolist(), 'labels:', batch['labels'].tolist())

```

## Q2 verification: Does the custom collate preserve the batch and sequence dimensions?

A custom `collate_fn` should still produce `[batch, sequence]` and must not flatten the sequence dimension of single samples during batching.


```python
loader = DataLoader(ToyDataset(), batch_size=2, shuffle=False, collate_fn=collate_pair)
batch = next(iter(loader))
assert batch['input_ids'].shape == (2, 1)
assert batch['attention_mask'].shape == (2, 1)
assert batch['labels'].shape == (2, 1)
print('✅ collate_fn 通过')

```

## Q3: How do variable-length tokenizer outputs become a uniform batch?

The tokenizer turns text into token sequences of different lengths; the batch layer then performs truncation, padding, and field derivation. The three kinds of fields have different responsibilities: `input_ids` stores the token ids, `attention_mask` distinguishes valid tokens from padding, and `labels` specifies which positions participate in the loss. Below, starting from the tokenizer output, you verify whether length alignment, the padding mask, and label masking satisfy the training interface contract.


```python
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


def tokenize_for_contract(text, max_length):
    """Generate a deterministic token sequence to demonstrate the truncation contract rather than reproduce a real tokenization algorithm."""
    content_ids = [10 + (ord(char) % 50) for char in text]
    token_ids = [BOS_ID, *content_ids, EOS_ID]
    return token_ids[:max_length]


def collate_tokenized_texts(texts, max_length):
    """Arrange variable-length tokenizer outputs into input_ids, attention_mask, and labels."""
    tokenized = [tokenize_for_contract(text, max_length) for text in texts]
    batch_length = max(len(token_ids) for token_ids in tokenized)

    input_rows = []
    mask_rows = []
    label_rows = []
    for token_ids in tokenized:
        pad_length = batch_length - len(token_ids)
        input_rows.append(token_ids + [PAD_ID] * pad_length)
        mask_rows.append([1] * len(token_ids) + [0] * pad_length)
        label_rows.append(token_ids + [-100] * pad_length)

    return {
        'input_ids': torch.tensor(input_rows, dtype=torch.long),
        'attention_mask': torch.tensor(mask_rows, dtype=torch.long),
        'labels': torch.tensor(label_rows, dtype=torch.long),
    }


tokenized_batch = collate_tokenized_texts(['batch', 'io'], max_length=6)
print({key: value.tolist() for key, value in tokenized_batch.items()})
```

## Q3 verification: Do truncation, padding, and ignored labels use the same coordinates?

The three fields should have the same shape; the `attention_mask` at padding positions is 0, and the corresponding `labels` must be `-100`.


```python
assert tuple(tokenized_batch['input_ids'].shape) == (2, 6)
assert tokenized_batch['input_ids'].shape == tokenized_batch['attention_mask'].shape == tokenized_batch['labels'].shape
assert torch.equal(tokenized_batch['attention_mask'].eq(0), tokenized_batch['labels'].eq(-100))
assert tokenized_batch['attention_mask'][1].tolist() == [1, 1, 1, 1, 0, 0]
print('✅ tokenizer 输出与 padding 契约通过')

```

## Q4: Why do valid input positions and supervised positions need to be represented separately?

`attention_mask` indicates which positions are valid inputs, and `labels != -100` indicates which positions participate in the loss. A valid input position can provide context without directly contributing supervision, so the two kinds of masks cannot replace each other.

| Field or condition | What it controls | Typical values |
|:---|:---|:---|
| `attention_mask` | Whether a position is a valid input | 1 for valid positions, 0 for padding |
| `labels` | The target value at each position | target id or `-100` |
| `labels != -100` | Whether a position contributes to the supervised loss | Determined by the task's data contract |


```python
def pad_to_max_len(tensors, pad_value):
    # First pad to the longest sample, then stack along the batch dimension.
    max_len = max(t.size(0) for t in tensors)
    padded = []
    for tensor in tensors:
        pad_len = max_len - tensor.size(0)
        if pad_len == 0:
            padded.append(tensor)
            continue
        pad = torch.full((pad_len,), pad_value, dtype=tensor.dtype)
        padded.append(torch.cat([tensor, pad], dim=0))
    return torch.stack(padded, dim=0)


def build_sequence_sample(token_ids, supervised_positions):
    """Build a generic sequence input and supervision labels from a boolean position table."""
    if len(token_ids) != len(supervised_positions):
        raise ValueError('token_ids and supervised_positions must have the same length')
    input_ids = torch.tensor(token_ids, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([
        token_id if is_supervised else -100
        for token_id, is_supervised in zip(token_ids, supervised_positions)
    ], dtype=torch.long)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

sequence_samples = [
    build_sequence_sample([11, 12, 13, 14, 15], [False, True, True, False, True]),
    build_sequence_sample([21, 22, 23], [True, False, True]),
]

# Padding only aligns variable-length samples to the same batch shape.
sequence_batch = {
    'input_ids': pad_to_max_len([item['input_ids'] for item in sequence_samples], pad_value=0),
    'attention_mask': pad_to_max_len([item['attention_mask'] for item in sequence_samples], pad_value=0),
    'labels': pad_to_max_len([item['labels'] for item in sequence_samples], pad_value=-100),
}

print('input_ids shape:', tuple(sequence_batch['input_ids'].shape))
print('attention_mask row1:', sequence_batch['attention_mask'][1].tolist())
print('labels row0:', sequence_batch['labels'][0].tolist())
print('labels row1:', sequence_batch['labels'][1].tolist())

```

## Q4 verification: Do valid positions, supervised positions, and padding use the same coordinates?

The three fields must have the same shape; unsupervised but valid positions keep `attention_mask=1` and `labels=-100`, while padding uses both `attention_mask=0` and `labels=-100`.


```python
assert tuple(sequence_batch['input_ids'].shape) == (2, 5)
assert sequence_batch['input_ids'].shape == sequence_batch['attention_mask'].shape == sequence_batch['labels'].shape
assert sequence_batch['input_ids'][0].tolist() == [11, 12, 13, 14, 15]
assert sequence_batch['labels'][0].tolist() == [-100, 12, 13, -100, 15]
assert sequence_batch['input_ids'][1].tolist() == [21, 22, 23, 0, 0]
assert sequence_batch['attention_mask'][1].tolist() == [1, 1, 1, 0, 0]
assert sequence_batch['labels'][1].tolist() == [21, -100, 23, -100, -100]
assert sequence_batch['attention_mask'][0, 0].item() == 1
assert sequence_batch['labels'][0, 0].item() == -100
print('✅ 输入位置、监督位置与 padding 对齐通过')

```

## Q5: How is a batch adapted and validated into input that the model can consume?

A data batch can contain both model inputs and supervision labels. The adaptation layer handles moving to the device, filters out the fields needed by `forward()`, and passes `labels` separately to the loss function; the contract check validates the fields, first dimension, shape, and dtype before execution.


```python
class MaskedMeanRegressor(nn.Module):
    def forward(self, input_ids, attention_mask):
        mask = attention_mask.to(dtype=torch.float32)
        values = input_ids.to(dtype=torch.float32)
        return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def adapt_batch_for_model(batch, device):
    """Move the batch and split it into model inputs and supervision labels."""
    moved = {key: value.to(device) for key, value in batch.items()}
    model_inputs = {
        'input_ids': moved['input_ids'],
        'attention_mask': moved['attention_mask'],
    }
    return model_inputs, moved['labels'].squeeze(-1)


def validate_batch_contract(batch, required_keys, expected_batch_size):
    """Check the fields, first dimension, shape alignment, and integer dtype of a training batch."""
    missing_keys = [key for key in required_keys if key not in batch]
    if missing_keys:
        return {'valid': False, 'errors': [f'missing keys: {missing_keys}']}

    errors = []
    shapes = {key: tuple(batch[key].shape) for key in required_keys}
    for key in required_keys:
        if batch[key].ndim < 1 or batch[key].shape[0] != expected_batch_size:
            errors.append(f'{key}: invalid batch dimension {shapes[key]}')
        if batch[key].dtype != torch.long:
            errors.append(f'{key}: expected torch.long, got {batch[key].dtype}')
    if len(set(shapes.values())) != 1:
        errors.append(f'field shapes are not aligned: {shapes}')
    return {'valid': not errors, 'errors': errors, 'shapes': shapes}


model = MaskedMeanRegressor()
batch = next(iter(DataLoader(ToyDataset(), batch_size=2, shuffle=False)))
model_inputs, labels = adapt_batch_for_model(batch, torch.device('cpu'))
predictions = model(**model_inputs)
print('predictions / labels shape：', predictions.shape, labels.shape)

```

## Q5 verification: Are both the adapted result and the failure report sufficiently explicit?

The adapted result should pass only the fields the model needs into `forward()`, while keeping labels for loss computation; a valid batch should pass the check, and a missing field should produce an explicit report before the model forward pass.


```python
required_keys = ('input_ids', 'attention_mask', 'labels')
report = validate_batch_contract(batch, required_keys, expected_batch_size=2)
assert report['valid']
assert report['shapes']['input_ids'] == (2, 1)
assert set(model_inputs) == {'input_ids', 'attention_mask'}
assert predictions.shape == labels.shape == (2,)
broken_batch = {key: value for key, value in batch.items() if key != 'attention_mask'}
broken_report = validate_batch_contract(broken_batch, required_keys, expected_batch_size=2)
assert not broken_report['valid']
assert 'missing keys' in broken_report['errors'][0]
print('✅ batch 适配与输入契约检查通过')

```

## Q6: How do you connect data, updates, and validation into a complete training lifecycle?

The same batch contract should serve both training and validation. The training stage enables gradients and performs an optimizer step, and the validation stage switches to `eval()` and disables gradient recording; both return a comparable average loss.


```python
class TrainableRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 1)

    def forward(self, input_ids, attention_mask):
        mask = attention_mask.to(dtype=torch.float32)
        values = input_ids.to(dtype=torch.float32)
        pooled = (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.projection(pooled.unsqueeze(-1)).squeeze(-1)


def run_epoch(model, loader, optimizer=None):
    """Run a training or validation epoch using the same batch contract."""
    is_training = optimizer is not None
    model.train(is_training)
    loss_sum = 0.0
    sample_count = 0
    for batch in loader:
        report = validate_batch_contract(batch, ('input_ids', 'attention_mask', 'labels'), batch['input_ids'].shape[0])
        if not report['valid']:
            raise ValueError(report['errors'])
        model_inputs, labels = adapt_batch_for_model(batch, torch.device('cpu'))
        with torch.set_grad_enabled(is_training):
            predictions = model(**model_inputs)
            batch_loss_sum = torch.nn.functional.mse_loss(
                predictions, labels.to(torch.float32), reduction='sum'
            )
            loss = batch_loss_sum / labels.numel()
        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        loss_sum += float(batch_loss_sum.detach())
        sample_count += labels.numel()
    if sample_count == 0:
        raise ValueError('训练或验证数据不能为空')
    return loss_sum / sample_count


lifecycle_loader = DataLoader(ToyDataset(), batch_size=3, shuffle=False, collate_fn=collate_pair)
lifecycle_model = TrainableRegressor()
lifecycle_optimizer = torch.optim.SGD(lifecycle_model.parameters(), lr=0.05)
before_training = {name: value.detach().clone() for name, value in lifecycle_model.named_parameters()}
train_loss = run_epoch(lifecycle_model, lifecycle_loader, lifecycle_optimizer)
after_training = {name: value.detach().clone() for name, value in lifecycle_model.named_parameters()}
validation_dataset = ToyDataset(sample_ids=(4, 5, 6, 7))
validation_loader = DataLoader(validation_dataset, batch_size=3, shuffle=False, collate_fn=collate_pair)
validation_loss = run_epoch(lifecycle_model, validation_loader)
print({'train_loss': train_loss, 'validation_loss': validation_loss})

```

## Q6 verification: Are the training updates and the validation boundary correct?

The training epoch should update at least one parameter; the validation epoch should return a finite loss, keep the parameters unchanged, and leave the model in evaluation mode at the end.


```python
assert any(not torch.equal(before_training[name], after_training[name]) for name in before_training)
before_validation = {name: value.detach().clone() for name, value in lifecycle_model.named_parameters()}
repeat_validation_loss = run_epoch(lifecycle_model, validation_loader)
after_validation = {name: value.detach().clone() for name, value in lifecycle_model.named_parameters()}
assert all(torch.equal(before_validation[name], after_validation[name]) for name in before_validation)
assert torch.isfinite(torch.tensor([train_loss, validation_loss, repeat_validation_loss])).all()
assert abs(validation_loss - repeat_validation_loss) < 1e-8
assert lifecycle_model.training is False
assert set(lifecycle_loader.dataset.sample_ids).isdisjoint(validation_dataset.sample_ids)
print('✅ 数据、更新与验证生命周期通过')

```

## Q7: After training, how do you evaluate the model with loss and task metrics?

The loss is the optimization objective, and task metrics answer how well the model performs on the target task. Evaluation should use independent validation or test data, disable gradients, and aggregate numerators and denominators by sample or token, rather than directly averaging the means of batches of different sizes. Validation metrics are used for tuning and selecting the best checkpoint, and test metrics are used for the final report; the reverse must not happen. The regression example below computes MAE, MSE, and RMSE.

| Task | Common evaluation signals | Aggregation unit |
|:---|:---|:---|
| Regression | MAE, RMSE | Sample |
| Classification | loss, accuracy, and precision / recall / F1 when needed | Sample or class |
| Language model | token loss, perplexity | Non-padding tokens |


```python
def evaluate_regressor(model, loader):
    """Aggregate regression errors by sample and return comparable evaluation metrics."""
    model.eval()
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in loader:
            report = validate_batch_contract(
                batch, ('input_ids', 'attention_mask', 'labels'), batch['input_ids'].shape[0]
            )
            if not report['valid']:
                raise ValueError(report['errors'])
            model_inputs, labels = adapt_batch_for_model(batch, torch.device('cpu'))
            predictions = model(**model_inputs)
            errors = predictions - labels.to(torch.float32)
            absolute_error_sum += float(errors.abs().sum())
            squared_error_sum += float(errors.square().sum())
            sample_count += errors.numel()
    if sample_count == 0:
        raise ValueError('评价数据不能为空')
    mae = absolute_error_sum / sample_count
    mse = squared_error_sum / sample_count
    return {
        'mae': mae,
        'mse': mse,
        'rmse': mse ** 0.5,
        'sample_count': sample_count,
    }


evaluation_loader = DataLoader(
    validation_dataset, batch_size=3, shuffle=False, collate_fn=collate_pair
)
before_evaluation = {
    name: value.detach().clone() for name, value in lifecycle_model.named_parameters()
}
evaluation_metrics = evaluate_regressor(lifecycle_model, evaluation_loader)
after_evaluation = {
    name: value.detach().clone() for name, value in lifecycle_model.named_parameters()
}
print('evaluation metrics：', evaluation_metrics)

```

## Q7 verification: Are the metrics aggregated over all samples without modifying the model?

The two batches of the evaluation loader have sizes 3 and 1; the metrics should still be aggregated over 4 samples. The evaluation process does not update parameters and keeps the model in evaluation mode.


```python
assert evaluation_metrics['sample_count'] == len(validation_dataset)
# One batch has three samples and the other has one; recompute directly over all predictions to check the aggregation denominator.
all_validation = collate_pair([validation_dataset[i] for i in range(len(validation_dataset))])
all_inputs, all_labels = adapt_batch_for_model(all_validation, torch.device('cpu'))
with torch.no_grad():
    expected_mse = (lifecycle_model(**all_inputs) - all_labels.float()).square().mean().item()
assert abs(evaluation_metrics['mse'] - expected_mse) < 1e-6
assert torch.isfinite(torch.tensor([
    evaluation_metrics['mae'], evaluation_metrics['mse'], evaluation_metrics['rmse']
])).all()
assert abs(evaluation_metrics['rmse'] ** 2 - evaluation_metrics['mse']) < 1e-6
assert abs(evaluation_metrics['mse'] - validation_loss) < 1e-6
assert evaluation_metrics['mae'] <= evaluation_metrics['rmse'] + 1e-6
assert all(torch.equal(before_evaluation[name], after_evaluation[name]) for name in before_evaluation)
assert lifecycle_model.training is False
print('✅ 模型评价指标与参数不变性通过')

```

## Related Reading

Now that the general loop from data to training is complete, you can continue by observing the Block dataflow inside the model; when you need to construct prompt-response supervision labels, go on to the SFT training loop.

- [13. Transformer Block Dataflow and Residual Stream](./13_Transformer_Block_Dataflow_and_Residual_Stream.md)
- [13. SFT Training Loop](../02_PyTorch_Algorithms/13_SFT_Training_Loop.md)
- [73. Training Performance Analysis](../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md)
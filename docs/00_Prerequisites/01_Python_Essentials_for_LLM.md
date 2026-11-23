# 01. Python Essentials for LLM | Python 基础与数据表示

**Difficulty:** Easy | **Environment:** CPU-first | **Tags:** `Python`, `engineering fundamentals`, `data structures` | **Audience:** learners preparing for Parts 2–4

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/01_Python_Essentials_for_LLM.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## Section Guide

Later tutorials repeatedly use ordered data, fixed dimensions, and named fields to pass tokens, shapes, configurations, and metrics. After choosing the wrong data structure, field meaning, mutability, and input boundaries become ambiguous.

After completing this section, you should be able to choose a data structure based on ordering, mutability, and field access, and use input/output contracts to identify missing fields and invalid inputs early.

**Keywords:** `list`, `tuple`, `dict`, `function contract`

![Python data and function contracts](../public/00_Prerequisites/01_python_contract_stream_map.svg)

## Prerequisite Reading
**Overview:** This page is the code entry point for Part 00; if you have basic Python experience, you can start directly with the questions and verification code.
- [0A Group Page](./0A.md)

## Q1: What are `list`, `tuple`, and `dict` each best suited to express?

`list` is suited to samples that are added, removed, or processed in order; `tuple` is suited to shapes with fixed meaning; `dict` is suited to configurations and metrics accessed by name. When choosing a data structure, first ask whether the information is ordered, whether it is fixed, and whether it needs to be read by field.

```python
def describe_batch(tokens, shape, metrics):
    """Summarize the token sequence, fixed shape, and named metrics."""
    return {
        'token_count': len(tokens),
        'unique_tokens': len(set(tokens)),
        'shape': tuple(shape),
        'loss': metrics['loss'],
    }


tokens = ['llm', 'infra', 'llm']
shape = (2, 3, 4)
metrics = {'loss': 0.25, 'learning_rate': 1e-3}
print(describe_batch(tokens, shape, metrics))
```

## Q1 verification: Do the three data types preserve their respective semantics?

This checks that order, fixed dimensions, and named fields are not mixed up during summarization.

```python
summary = describe_batch(['a', 'b', 'a'], (2, 4), {'loss': 0.5})
assert summary == {
    'token_count': 3,
    'unique_tokens': 2,
    'shape': (2, 4),
    'loss': 0.5,
}
assert describe_batch([], (0, 4), {'loss': 0.0})['token_count'] == 0
print('✅ Python 数据结构通过')
```

## Q2: How does a function form a reusable input/output contract?

A function signature should distinguish required inputs from optional policies. Below, `batch_size` and `drop_last` are keyword arguments so the call site directly expresses their meaning; the returned dictionary preserves the full batch count, the remainder, and the number of execution steps, avoiding the need to return a single number that is hard to interpret.

```python
def build_batch_plan(sample_count, *, batch_size=4, drop_last=False):
    """Compute the batch plan from a non-negative sample count; return full batches, the remainder, and execution steps."""
    if not isinstance(sample_count, int) or sample_count < 0:
        raise ValueError('sample_count must be a non-negative integer')
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')

    full_batches, remainder = divmod(sample_count, batch_size)
    steps = full_batches if drop_last or remainder == 0 else full_batches + 1
    return {
        'full_batches': full_batches,
        'remainder': remainder,
        'steps': steps,
        'drop_last': drop_last,
    }


print(build_batch_plan(10, batch_size=4))
```

## Q2 verification: Are the default policy, boundaries, and invalid inputs clear?

The verification covers both a remainder and dropping the tail batch, and confirms that an invalid batch size fails immediately.

```python
assert build_batch_plan(10, batch_size=4)['steps'] == 3
assert build_batch_plan(10, batch_size=4, drop_last=True)['steps'] == 2
assert build_batch_plan(0, batch_size=4)['steps'] == 0

try:
    build_batch_plan(8, batch_size=0)
except ValueError as error:
    assert 'batch_size' in str(error)
else:
    raise AssertionError('非法 batch_size 应触发 ValueError')
print('✅ 函数契约通过')
```

## Q3: Why validate required fields at the data entry point?

Configurations, samples, and results are usually passed as dicts. The earlier you check fields and types, the easier it is to locate an error at the input entry point instead of waiting until a confusing KeyError or shape error appears inside the model. The validation function should also return a new object to avoid accidentally modifying the caller's data.

```python
def validate_record(record, required_fields):
    """Check named fields and return a shallow copy without modifying the input record."""
    if not isinstance(record, dict):
        raise TypeError('record must be a dict')
    missing = [field for field in required_fields if field not in record]
    if missing:
        raise KeyError(f'missing required fields: {missing}')
    return record.copy()


sample = {'input_ids': [1, 2, 3], 'labels': [1, 2, 3]}
validated = validate_record(sample, ('input_ids', 'labels'))
validated['source'] = 'toy'
print(validated)
```

## Q3 verification: Can missing fields be located precisely?

This checks normal input, that the input is not modified, and the failure message for a missing field.

```python
sample = {'input_ids': [1, 2], 'labels': [1, 2]}
validated = validate_record(sample, ('input_ids', 'labels'))
assert validated == sample and validated is not sample

try:
    validate_record({'input_ids': [1, 2]}, ('input_ids', 'labels'))
except KeyError as error:
    assert 'labels' in str(error)
else:
    raise AssertionError('缺失 labels 应触发 KeyError')
print('✅ 输入字段校验通过')
```

## Q4: How do you process samples one at a time without copying all data at once?

A list is suited to producing all results immediately; a generator is suited to reading a larger sample stream on demand. The generator should still reuse the entry-point validation and indicate the sample position in the failure message, so lazy execution does not hide the source of the error.

```python
def iter_valid_records(records, required_fields):
    """Yield validated shallow copies one at a time; invalid samples report their sequential position."""
    for position, record in enumerate(records):
        try:
            yield validate_record(record, required_fields)
        except (TypeError, KeyError) as error:
            raise ValueError(f'record {position}: {error}') from error


records = [
    {'input_ids': [1, 2], 'labels': [1, 2]},
    {'input_ids': [3], 'labels': [3]},
]
record_stream = iter_valid_records(records, ('input_ids', 'labels'))
print(list(record_stream))
```

## Q4 verification: Does the generator execute lazily and locate invalid samples?

Creating the generator does not immediately traverse the input; field validation runs only when the results are consumed. This also checks the output copies and the positional failure message.

```python
records = [
    {'input_ids': [1], 'labels': [1]},
    {'input_ids': [2], 'labels': [2]},
]
record_stream = iter_valid_records(records, ('input_ids', 'labels'))
assert iter(record_stream) is record_stream
validated_records = list(record_stream)
assert validated_records == records
assert all(validated is not original for validated, original in zip(validated_records, records))

invalid_records = [
    {'input_ids': [1], 'labels': [1]},
    {'input_ids': [2]},
]
try:
    list(iter_valid_records(invalid_records, ('input_ids', 'labels')))
except ValueError as error:
    assert 'record 1' in str(error) and 'labels' in str(error)
else:
    raise AssertionError('非法样本应在被消费时触发带位置的 ValueError')
print('✅ 惰性样本处理通过')
```

## Related Reading

After mastering data structures and function contracts, continue with array dimensions, object state, and experiment configuration.

- [02. NumPy and Einsum](./02_NumPy_and_Einsum.md)
- [03. Python OOP and Utility Patterns](./03_Python_OOP_and_Utility_Patterns.md)
- [04. Python Config and Data Entry](./04_Python_Config_and_Data_Entry.md)
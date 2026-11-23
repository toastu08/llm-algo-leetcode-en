# 04. Python Config and Data Entry | Python 配置与数据入口

**Difficulty:** Easy | **Environment:** CPU-first | **Tags:** `Python`, `config`, `I/O`, `reproducibility` | **Audience:** learners preparing for Parts 2–4

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/04_Python_Config_and_Data_Entry.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## Section Guide

Besides an object's internal state, a run also involves configuration, input sources, resource paths, and output artifacts. Without a unified record, even the same code may fail to reproduce because of different external conditions.

After completing this section, you should be able to distinguish internal state from external run conditions, and trace an experiment using configuration, paths, an artifact manifest, and the run identity.

**Keywords:** `config`, `path`, `artifact`, `run_id`

![Configuration, resources, and run artifacts](../public/00_Prerequisites/04_config_artifact_manifest_map.svg)

## Prerequisite Reading
**Overview:** First understand function contracts and object state, then ground them in configuration, files, and run artifacts.
- [03. Python OOP and Utility Patterns](./03_Python_OOP_and_Utility_Patterns.md)
- [0A Group Page](./0A.md)

## Q1: After overriding configuration, why still check the schema?

The override rules determine the candidate parameters, and the schema determines whether it has the fields needed for a run. The merge function should return a new object to avoid modifying the base configuration; missing fields should fail before entering the file or training workflow.

```python
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def merge_and_validate_config(base, overrides, required_fields):
    """Merge configuration and check required fields without modifying the input dicts."""
    merged = {**base, **overrides}
    missing = [field for field in required_fields if field not in merged]
    if missing:
        raise KeyError(f'missing config fields: {missing}')
    return merged


base = {'model': 'TinyLM', 'precision': 'bf16', 'batch_size': 4}
config = merge_and_validate_config(base, {'batch_size': 8}, ('model', 'precision', 'batch_size'))
print(config)
```

## Q1 verification: Can overriding, unchanged input, and missing fields be checked?

```python
base = {'model': 'TinyLM', 'precision': 'bf16', 'batch_size': 4}
merged = merge_and_validate_config(base, {'precision': 'fp16'}, ('model', 'precision', 'batch_size'))
assert merged['precision'] == 'fp16'
assert base['precision'] == 'bf16'

try:
    merge_and_validate_config({}, {}, ('model',))
except KeyError as error:
    assert 'model' in str(error)
else:
    raise AssertionError('缺失 model 应触发 KeyError')
print('✅ 配置 schema 通过')
```

## Q2: How do paths and `with` manage the resource lifecycle?

Paths are responsible for locating files, and `with` is responsible for when resources are created and released. A temporary directory is suitable for one-off checks: files inside the scope must be readable, and the directory should be cleaned up automatically after leaving the scope.

```python
def write_json(path, payload):
    """Write UTF-8 JSON and return the object read back."""
    path = Path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return json.loads(path.read_text(encoding='utf-8'))


with TemporaryDirectory() as tmpdir:
    temporary_root = Path(tmpdir)
    config_path = temporary_root / 'config.json'
    restored = write_json(config_path, {'model': 'TinyLM', 'seed': 42})
    print(config_path, restored)
```

## Q2 verification: Is the file readable and is the temporary directory really cleaned up?

```python
with TemporaryDirectory() as tmpdir:
    temporary_root = Path(tmpdir)
    payload_path = temporary_root / 'payload.json'
    assert write_json(payload_path, {'ok': True}) == {'ok': True}
    assert payload_path.exists()

assert not temporary_root.exists()
print('✅ 文件与资源生命周期通过')
```

## Q3: How do you keep the input, cache, log, and artifact paths stable?

The same `run_name` should map to a deterministic directory structure so that training, validation, and review code do not have to concatenate paths repeatedly. This only fixes the locations and does not assume the files have been generated.

```python
def build_run_paths(root, run_name):
    """Generate stable paths from the root directory and a single-segment run_name; do not create any files."""
    if (
        not isinstance(run_name, str)
        or not run_name
        or run_name in {'.', '..'}
        or '/' in run_name
        or '\\' in run_name
    ):
        raise ValueError('run_name must be one safe, non-empty path segment')
    root = Path(root)
    run_root = root / run_name
    return {
        'run_root': run_root,
        'input': root / 'inputs' / 'train.jsonl',
        'cache': run_root / 'cache',
        'log': run_root / 'logs' / 'train.jsonl',
        'artifact': run_root / 'artifacts' / 'model.pt',
    }


paths = build_run_paths('runs', 'tinylm-seed42')
print({name: str(path) for name, path in paths.items()})
```

## Q3 verification: Does the same run name produce the same paths?

```python
first = build_run_paths('runs', 'tinylm-seed42')
second = build_run_paths('runs', 'tinylm-seed42')
assert first == second
assert first['artifact'].as_posix().endswith('tinylm-seed42/artifacts/model.pt')
assert first['input'].as_posix() == 'runs/inputs/train.jsonl'

for invalid_name in ('', '.', '..', '../escape', 'nested/run', 'nested\\run'):
    try:
        build_run_paths('runs', invalid_name)
    except ValueError as error:
        assert 'path segment' in str(error)
    else:
        raise AssertionError(f'非法 run_name 应失败: {invalid_name!r}')
print('✅ 运行路径契约通过')
```

## Q4: How do you record the minimal manifest of a reproducible run?

The configuration states "how to run", the input source states "what is processed", and the artifact points to "what was produced". The `run_id` is generated from the configuration, the random seed, and the input source, so the same conditions can recompute the same identity; the artifact path is recorded separately as the output location.

```python
import hashlib


def build_run_manifest(config, seed, input_source, artifact_path):
    """Generate the minimal run manifest; run_id depends only on stable run conditions."""
    identity = {
        'config': config,
        'seed': seed,
        'input_source': str(input_source),
    }
    canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    return {
        **identity,
        'artifact_path': str(artifact_path),
        'run_id': hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:12],
    }


paths = build_run_paths('runs', 'tinylm-seed42')
manifest = build_run_manifest(
    config={'model': 'TinyLM', 'batch_size': 4},
    seed=42,
    input_source=paths['input'],
    artifact_path=paths['artifact'],
)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
```

## Q4 verification: Can identical conditions and changed conditions be distinguished?

```python
config = {'model': 'TinyLM', 'batch_size': 4}
first = build_run_manifest(config, 42, 'data/train.jsonl', 'artifacts/a.pt')
repeat = build_run_manifest(config, 42, 'data/train.jsonl', 'artifacts/a.pt')
changed_seed = build_run_manifest(config, 43, 'data/train.jsonl', 'artifacts/a.pt')

assert first['run_id'] == repeat['run_id']
assert first['run_id'] != changed_seed['run_id']
assert set(first) == {'config', 'seed', 'input_source', 'artifact_path', 'run_id'}
print('✅ 最小运行清单通过')
```

### Section Summary

- After overriding configuration, you still need to check the fields required for a run.
- Paths and `with` handle file location and resource lifecycle respectively.
- Stable paths separate inputs, caches, logs, and artifacts.
- A manifest identifies a reviewable run with the configuration, seed, input source, and `run_id`.
## Related Reading

Now that configuration and run identity are in place, you can continue with tensor inputs, checkpoints, and batch contracts.

- [05. PyTorch Tensor Fundamentals](./05_PyTorch_Tensor_Fundamentals.md)
- [10. PyTorch State, Checkpoint, and Recovery](./10_PyTorch_State_dict_and_Persistence.md)
- [12. PyTorch Data Interface and Training Lifecycle](./12_PyTorch_Minimal_Training_Interface.md)
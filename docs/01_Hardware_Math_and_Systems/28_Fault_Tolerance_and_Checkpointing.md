# 28. Fault Tolerance and Checkpointing | 容错与检查点

**Difficulty:** Medium-Hard | **Environment:** CPU-first | **Tags:** `systems engineering`, `fault-tolerant recovery`, `Checkpoint` | **Audience:** learners starting with system performance

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/28_Fault_Tolerance_and_Checkpointing.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## Section Guide

Long training runs may be interrupted by a process, device, or node failure. A checkpoint must save enough state so that after training resumes it continues along a consistent update trajectory.

This section establishes the relationship "recoverable state → save frequency → write cost → recovery validation." When you finish, you should be able to list the state required for recovery and compare the trade-off among failure loss, storage overhead, and recovery time.

**Keywords:** checkpoint, recovery, fault tolerance

![Concept relationships in this section](../public/01_Hardware_Math_and_Systems/28_checkpoint_recovery_map.svg)
## Prerequisite Reading

**Overview:** First map memory pressure, parallel decisions, and communication scheduling onto the training state, then check which state must be saved, how to validate it after recovery, and how the save policy affects training throughput and recovery time.

- [06. VRAM Calculation and ZeRO](./06_VRAM_Calculation_and_ZeRO.md)
- [26. Parallel Strategy Decision Framework](./26_Parallel_Strategy_Decision_Framework.md)
- [27. Communication Scheduling Optimization](./27_Communication_Scheduling_Optimization.md)
## Q1: What does a Checkpoint actually save?


A checkpoint does not save a single model file; it saves the set of states required to resume training together with their manifest.

It usually includes at least:
- Model weights
- Optimizer state
- Learning-rate scheduler state
- Random number seeds
- Training progress

Each state should be registered in the manifest with its role, size, and identifier. Only after the complete manifest is committed can the checkpoint be read by the recovery side; saving only the weights can load a model but cannot guarantee resuming the same training trajectory.


### Q1 mini check
Construct a checkpoint manifest and verify that it cannot be marked as recoverable when required state is missing.


```python
REQUIRED_ROLES = {'model', 'optimizer', 'scheduler', 'rng', 'progress'}

def inspect_checkpoint_manifest(entries, committed):
    """Check the roles, file sizes, and commit status required to resume training."""
    roles = {entry['role'] for entry in entries}
    missing = sorted(REQUIRED_ROLES - roles)
    invalid_files = sorted(entry['path'] for entry in entries if entry['size_mib'] <= 0)
    total_mib = sum(entry['size_mib'] for entry in entries)
    return {
        'total_mib': round(total_mib, 2),
        'missing_roles': missing,
        'invalid_files': invalid_files,
        'committed': committed,
        'recoverable': committed and not missing and not invalid_files,
    }

complete_entries = [
    {'role': 'model', 'path': 'model.bin', 'size_mib': 2400},
    {'role': 'optimizer', 'path': 'optimizer.bin', 'size_mib': 4800},
    {'role': 'scheduler', 'path': 'scheduler.json', 'size_mib': 0.01},
    {'role': 'rng', 'path': 'rng.bin', 'size_mib': 0.01},
    {'role': 'progress', 'path': 'trainer.json', 'size_mib': 0.01},
]
complete = inspect_checkpoint_manifest(complete_entries, committed=True)
weights_only = inspect_checkpoint_manifest(complete_entries[:1], committed=True)
assert complete['recoverable']
assert not weights_only['recoverable'] and 'optimizer' in weights_only['missing_roles']
print('完整检查点：', complete)
print('仅有权重：', weights_only)


```

## Q2: How do failure probability and recovery time affect the effective training time?


Long training runs must estimate the number of failures, the average rollback amount, and the reload time at the same time. If failures occur approximately uniformly between two checkpoints, the average rollback compute is about half the save interval.

- The longer the job, the higher the probability of a failure
- The loss of rerunning from scratch can be very large
- The goal of a checkpoint is to push down the recovery cost after a failure

This estimate is not a failure predictor, but a uniform cost basis for comparing checkpoint intervals.


### Q2 mini check
Estimate the expected rollback and recovery overhead for a given save interval.


```python
def expected_failure_overhead(run_hours, fault_rate_per_hour, interval_minutes, reload_minutes):
    """Estimate rollback computation and reload time under a uniform failure assumption."""
    if min(run_hours, fault_rate_per_hour, interval_minutes, reload_minutes) < 0 or interval_minutes == 0:
        raise ValueError('时间和故障率不得为负，检查点间隔必须大于 0')
    expected_faults = run_hours * fault_rate_per_hour
    rollback_minutes = expected_faults * interval_minutes / 2
    reload_total_minutes = expected_faults * reload_minutes
    return {
        'expected_faults': round(expected_faults, 3),
        'expected_rollback_minutes': round(rollback_minutes, 2),
        'expected_reload_minutes': round(reload_total_minutes, 2),
        'expected_failure_overhead_minutes': round(rollback_minutes + reload_total_minutes, 2),
    }

hourly = expected_failure_overhead(72, 0.02, 60, 8)
half_hourly = expected_failure_overhead(72, 0.02, 30, 8)
assert half_hourly['expected_rollback_minutes'] < hourly['expected_rollback_minutes']
print('每 60 分钟保存：', hourly)
print('每 30 分钟保存：', half_hourly)


```

## Q3: Why do saving and restoring also have a cost?


Checkpoint writing pauses or competes with training resources. A shorter interval reduces the average rollback amount after a failure but increases the number of writes; a longer interval does the opposite.

If you save too often, training slows down; if you save too rarely, the rollback cost becomes large once a failure happens. In distributed scenarios, saving and restoring also involve parallel I/O, consistency, and the scheduling window.

Distributed scenarios also need to record whether the write is asynchronous, whether the shards of each rank are consistent, and when the manifest is committed atomically. Here we first compare the theoretical overhead of synchronous writes.


### Q3 mini check
Scan several save intervals and observe how the write cost and the failure rollback cost change.


```python
import math

def checkpoint_interval_cost(run_minutes, interval_minutes, save_minutes, fault_rate_per_hour, reload_minutes):
    """Compare the three kinds of overhead: synchronous saving, expected rollback, and reloading."""
    if interval_minutes <= 0 or min(run_minutes, save_minutes, fault_rate_per_hour, reload_minutes) < 0:
        raise ValueError('检查点间隔必须大于 0，其余输入不得为负')
    save_count = math.floor(run_minutes / interval_minutes)
    save_overhead = save_count * save_minutes
    expected_faults = run_minutes / 60 * fault_rate_per_hour
    failure_overhead = expected_faults * (interval_minutes / 2 + reload_minutes)
    return {
        'interval_minutes': interval_minutes,
        'save_count': save_count,
        'save_overhead_minutes': round(save_overhead, 2),
        'failure_overhead_minutes': round(failure_overhead, 2),
        'total_overhead_minutes': round(save_overhead + failure_overhead, 2),
    }

intervals = [15, 30, 60, 120]
reports = [checkpoint_interval_cost(1440, interval, 2, 0.02, 8) for interval in intervals]
best = min(reports, key=lambda report: report['total_overhead_minutes'])
assert best in reports
for report in reports:
    print(report)
print('当前假设下开销最小的间隔：', best['interval_minutes'], '分钟')


```

## Q4: How do you judge whether a checkpoint plan is truly recoverable?


A distributed checkpoint is recoverable only when the following conditions are satisfied at the same time: the required state is complete, all rank shards belong to the same checkpoint ID, the number of shards matches the manifest, and the manifest has been committed.

When the world size at recovery time changes, the framework must also explicitly support resharding; you cannot assume that loading works merely because the files exist.


### Q4 mini check
Verify whether the distributed shards, the checkpoint identifier, and the commit status are consistent.


```python
def validate_sharded_checkpoint(manifest, shards):
    """Check the shard count, the checkpoint identifier, the rank coverage, and the atomic commit status."""
    issues = []
    expected_ranks = set(range(manifest['world_size']))
    actual_ranks = {shard['rank'] for shard in shards}
    if actual_ranks != expected_ranks:
        issues.append(f'rank coverage mismatch: expected {sorted(expected_ranks)}, got {sorted(actual_ranks)}')
    if any(shard['checkpoint_id'] != manifest['checkpoint_id'] for shard in shards):
        issues.append('checkpoint IDs are inconsistent')
    if any(shard['size_mib'] <= 0 for shard in shards):
        issues.append('one or more shards are empty')
    if not manifest['committed']:
        issues.append('manifest is not committed')
    return {'recoverable': not issues, 'issues': issues}

manifest = {'checkpoint_id': 'step-1200', 'world_size': 4, 'committed': True}
complete_shards = [
    {'rank': rank, 'checkpoint_id': 'step-1200', 'size_mib': 900} for rank in range(4)
]
missing_shard = complete_shards[:-1]
assert validate_sharded_checkpoint(manifest, complete_shards)['recoverable']
assert not validate_sharded_checkpoint(manifest, missing_shard)['recoverable']
print('完整分片：', validate_sharded_checkpoint(manifest, complete_shards))
print('缺少分片：', validate_sharded_checkpoint(manifest, missing_shard))


```

## Related Reading

**Overview:** Continue by putting state saving, recovery validation, and training artifact delivery into one recovery chain.

- [04 SFT Experiments, Artifact, and Delivery](../topic_discussion/post_training_optimization/04_end_to_end_experiment.md)
- [62. Instruction Fine Tuning Project](../02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.md)
- [PyTorch Distributed Checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html)
---
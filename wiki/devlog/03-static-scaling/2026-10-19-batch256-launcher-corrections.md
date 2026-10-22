# Batch-256 launcher corrections — 2026-10-19

Status: local gates PASS. No remote command ran.

## Decision and changes

The user selected global batch 256 to reduce memory risk and preparation time.
OD-7 now records this choice. The launcher skips batch 512 completely.
Local batches are 256/128/64 at W=1/2/4.
The measured epoch contains 390 updates and 99,840 images in every configuration.
Twenty discarded warmup updates and five repeats remain the campaign defaults.
The nine-configuration memory preflight remains required.

The launcher splits the dependent `local` declaration in `precompile_variant`.
The faulty batch-512 fallback branch is removed.
Any failed batch-256 process or checker blocks the campaign.

Both launcher branches call `check_full_campaign.py ready` before timing.
This check requires the selected candidate to pass all nine configurations.
It checks source identity, dataset identity, epoch, runtime, hardware capacity, and CPU policy.
It accepts another node with matching GPU models and capacity.
It rejects stale source identity and incompatible runtime versions.

Each rank writes a separate JSON file with its rank, PID, and GPU UUID before training.
The sampler records timestamps around each `nvidia-smi` query.
The checker requires at least two sample timestamps for every training PID/GPU pair.
Unrelated processes cannot substitute for missing ranks.
These measurements are sampled peaks, not continuous high-water marks.
Short spikes between samples remain a measurement limitation.

## Batch-dependent learning-rate correction

The earlier full-dataset runners used the batch-512 default learning-rate schedule at both batch sizes.
Both runners now use five warmup epochs and a 50-epoch budget from OD-4.
Batch 256 therefore uses 1,950 warmup updates and 19,500 total updates.
The diagnostic and timing paths use the same schedule and hash its Float32 values.
The fixture path remains separate.

The new regression compares all 390 Float32 values through their SHA-256 hashes.
It evaluates the actual Julia and Python schedule helpers without GPU dependencies.
The hashes match. The warmup endpoint and final schedule endpoint also pass.

## Local evidence

Exact commands:

```bash
python3 scripts/static_scaling/test_full_campaign.py
bash -n scripts/remote/leonardo/run-static-full-campaign.sh && python3 scripts/static_scaling/check_full_campaign.py --self-test
julia --startup-file=no -e 'Meta.parseall(read("scripts/static_scaling/full_dataset_timing.jl", String)); println("JULIA_PARSE_OK")'
python3 scripts/static_scaling/test_full_campaign.py && bash -n scripts/remote/leonardo/run-static-full-campaign.sh && git diff --check
```

- Final regression suite: six tests PASS.
- Checker self-test: `FULL_CAMPAIGN_SELFTEST PASS checks=21`.
- Julia parse: `JULIA_PARSE_OK`.
- Shell syntax and whitespace checks: exit 0.
- The initial regression run contained five tests and passed before the schedule regression was added.
- Synthetic checker output is not benchmark evidence.

The regression suite exercises the actual precompile and selection shell functions with CPU-only stubs.
It covers process failure, checker failure, batch-256-only selection, and the location of readiness checks.
It also covers stale decisions, runtime/resource differences, incomplete matrices, and missing or incorrect rank samples.

## Exact next action

Resolve the historical worktree identity difference before source sync.
Sync the corrected thesis files, including the untracked launcher, checker, and regression script.
Then submit the four-GPU preflight-only job with `CAMPAIGN_MODE=preflight`.
The launcher precompiles both Flux worktrees on the compute node.
Require all nine batch-256 cases and `FULL_CAMPAIGN_READY PASS` before the campaign.
GPU execution of the new rank registration and memory sampler remains unverified.

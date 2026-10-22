# Static-scaling full campaign 61493236: structured report

Date: 2026-10-20.

Evidence root:
`/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-full-campaign-61493236/`

Related adversarial review:
`reports/static-scaling-full-campaign-61493236-review.md`

## 0. Verdict and scope

The saved campaign is **PASS for qualified descriptive throughput and strong scaling**. Slurm reports `COMPLETED`, exit `0:0`, elapsed `02:46:11` on `lrdn1300`. The evidence contains 45/45 timing records (9 configurations × 5 repeats), 9/9 diagnostic records and snapshots, and 307/307 checksum entries that the review independently verified. The driver markers include `FULL_CAMPAIGN_READY PASS batch=256 updates=390`, `FULL_CAMPAIGN_timing PASS records=45`, `FULL_CAMPAIGN_diagnostics PASS snapshots=9`, `FULL_CAMPAIGN_ARTIFACT_CHECK PASS`, and `FULL_CAMPAIGN PASS`.

This report distinguishes four levels of claim:

| Claim level | Status |
|---|---|
| Campaign executed and artifacts are internally consistent | Supported |
| Timing arithmetic, medians, throughput, speedup, efficiency | Supported; independently reproduced |
| Descriptive strong-scaling comparison on this allocation | Supported with qualifications |
| Causal guard-overhead or intrinsic framework advantage | Not established |
| Numerical equivalence or equal training quality | No claim (`NUMERICAL_EQUIVALENCE_CLAIM none`) |

The campaign is a **throughput benchmark**, not a learning-curve or validation-accuracy experiment. “Accuracy” in the harness means implementation/state correctness: matched data, model, optimizer, schedule, source, finite states, exact optimizer counters, and rank-consistent final state hashes. It does not mean predictive accuracy.

## 1. Research question and design

### 1.1 Question

For a fixed global batch, how much does training time fall when moving from 1 GPU to 2 and 4 GPUs? Secondarily, how do absolute throughput and scaling efficiency compare between the current Flux DDP implementation, the historical Flux static-DDP implementation, and eager PyTorch?

The design is **strong scaling**:

- `Q_N = K B / T_N` (images per second),
- `S_N = T_1 / T_N` (speedup),
- `E_N = S_N / N` (parallel efficiency),

where `K B = 99,840` images per measured epoch and `T_N` is the median measured time at world size `N` (`reports/static-graph-gpu-scaling-proposal-detailed.md:380-384`; `experiments/static-scaling/specification.md:50-64`).

“Static scaling” here means the same layers and trainable parameters execute every step. It is not a compiled graph; PyTorch's `static_graph` flag is a separate DDP configuration (`reports/static-graph-gpu-scaling-proposal-detailed.md:33-35`).

### 1.2 Three arms

| Arm | Implementation | Distributed policy |
|---|---|---|
| `flux-current` | Flux `f7b4520c` | Guarded DDP at W2/W4: presence check, gradient averaging, gradient-validity vote; single GPU at W1 |
| `flux-historical` | Flux `8add0e42` | Static DDP at W2/W4: per-buffer gradient averaging, no presence/validity guard; single GPU at W1 |
| `pytorch` | PyTorch `2.9.1+cu126` | Native DDP; deterministic cuDNN; no equivalent timed gradient guard |

The historical arm is a within-Flux revision contrast. Current and historical share the model, data, optimizer, and schedule but differ in the distributed API and guards. The comparison is descriptive; it does not isolate guard cost from other policy differences (`reports/static-scaling-full-campaign-61493236-review.md:71-78`).

At W2/W4 the current arm issues 64 collectives per update (1 presence check + 62 gradient averages + 1 gradient-validity vote). The historical arm issues 62 (one average per buffer) and has no presence or validity vote. PyTorch uses bucketed DDP, so a raw collective count is not directly comparable (`experiments/static-scaling/specification.md:318-330`; `scripts/static_scaling/phase4_common.jl:138-240`).

### 1.3 Workload and campaign matrix

| Item | Value |
|---|---|
| Dataset | Tiny ImageNet-200, epoch 0, saved permutation/augmentation |
| Epoch size | 100,000 images; 99,840 effective; 160-image tail dropped |
| Global batch | 256 |
| Local batch | 256 / 128 / 64 at W=1 / 2 / 4 |
| Measured updates | 390 (`99,840 / 256`) |
| Warmup | 20 discarded updates, then model and optimizer reset |
| Repeats | 5 fresh-process repeats per configuration |
| Worlds | 1, 2, 4 |
| Frameworks | `flux-current`, `flux-historical`, `pytorch` |
| Timing records | 9 configurations × 5 repeats = 45 |
| Diagnostic records | 9 (one snapshot per configuration) |
| Hardware | 1 node, 4×A100-SXM-64GB, 32 CPUs, one allocation |
| Model | ResNet-18-like, 11,271,432 parameters in 62 arrays, GroupNorm, 200-way classifier |
| Optimizer | AdamW, β=(0.9, 0.999), eps=1e-8, 21 decay / 41 no-decay arrays |
| Batch-256 schedule | Warmup 1,950 updates, total 19,500 updates |

Sources: `scripts/remote/leonardo/run-static-full-campaign.sh:73-87`; `scripts/static_scaling/full_dataset_common.py:204-308`; `scripts/static_scaling/model_flux.jl:405-505`; `scripts/static_scaling/full_dataset_timing.jl:126`; `wiki/context/current.md:33-36`.

The measured epoch lies **inside the batch-256 warmup schedule** (`t=1..390` of `W=1,950`). This is a fixed-work throughput workload, not a converged-learning or equal-quality comparison (`experiments/static-scaling/specification.md:209-223`).

### 1.4 Why batch 256

The operator prescribed batch 256 before the campaign. Earlier loader dry runs measured Flux device usage at 86.34% (current) and 83.14% (historical) at batch 512/W1, above the proposed 80% criterion, so the 512 fallback was removed and the launcher tested only 256 (`wiki/devlog/03-static-scaling/2026-10-19-batch256-launcher-corrections.md:7-16`; `wiki/devlog/03-static-scaling/2026-10-19-full-dataset-loader-validation.md:106-123`).

The precise statement is: **preflight accepted the prescribed global batch 256**. The campaign did not test batch 512 and did not perform an automatic capacity search. It therefore does not show that 512 is infeasible or that 256 is optimal (`reports/static-scaling-full-campaign-61493236-review.md:80-86`).

### 1.5 Why five repeats, one allocation

Five fresh-process repeats per configuration give a median robust to a single slow run. The campaign is repeat-first: all nine configurations run in a repeat, with world-size and framework order reversed on even repeats to reduce systematic ordering effects (`scripts/remote/leonardo/run-static-full-campaign.sh:777-796`).

Running all configurations in one allocation reduces cross-allocation and hardware variance. It also limits inference: the results come from one node, one allocation, and five repeats. They are descriptive, not a statistical superiority claim (`reports/static-scaling-full-campaign-61493236-review.md:47-49`).

## 2. Execution pipeline

### 2.1 Sync and provenance

The launcher does not sync; it checks required remote paths and hashes. The thesis tree and the two Flux worktrees were synced beforehand. The current Flux arm uses `../flux-integration`; the historical arm uses the pinned `../flux-static-baseline` (`scripts/remote/sync_flux_integration.sh:68-74`; `scripts/remote/sync_flux_baseline.sh:52-108`; `scripts/remote/leonardo/run-static-full-campaign.sh:94-98`).

Both Flux worktrees were precompiled on a compute node before the workload. The launcher aborts unless `FLUX_PRECOMPILE_OK` appears (`scripts/remote/leonardo/run-static-full-campaign.sh:261-277`; `scripts/remote/leonardo/precompile-phase3.sh:24-29`). This follows the compute-only precompile policy in `wiki/decisions/ADR-0005-precompile-compute-only.md:17-45`.

### 2.2 Submission

The documented single-command invocation is `CONFIRM_CAMPAIGN=1 CAMPAIGN_MODE=all sbatch ...` with one node, one task, 32 CPUs, four GPUs, and the launcher script (`scripts/remote/leonardo/run-static-full-campaign.sh:23-29`). The saved allocation reports `time_limit=12:00:00` (`resources.json:79`); the documented example uses 4 hours. The exact submitted argv is not preserved, but the saved Slurm resources and driver log identify job `61493236`.

The launcher enforces `CONFIRM_CAMPAIGN=1`, rejects login-node execution, requires one task, and refuses a non-empty output root (`scripts/remote/leonardo/run-static-full-campaign.sh:39-48,86-102`).

### 2.3 Preflight

The preflight runs the nine framework/world configurations for 20 warmup plus 20 measured updates at batch 256. Each rank writes its PID and GPU UUID outside the timed region. An external `nvidia-smi` sampler records per-process device memory every 0.25 s (`scripts/remote/leonardo/run-static-full-campaign.sh:73,476-490,597-623`; `scripts/static_scaling/full_dataset_timing.jl:119-125`).

The memory criterion is:

- peak **per-process** device memory,
- limit = `int(0.8 * device_total)`,
- failure only if `peak > limit`,
- missing samples or counters make the candidate `INCOMPLETE` and fail (`scripts/static_scaling/check_full_campaign.py:222-265,833-873`).

Actual preflight:

- 9/9 configurations PASS,
- selected batch 256,
- acceptance limit 54,479,657,369 bytes = **51,955.8 MiB**,
- greatest sampled process peak **49,728.0 MiB** at `flux-historical` W1,
- headroom 2,227.85 MiB (4.29%),
- per-rank sample counts 77–1,645 (`preflight/decision.json`; `driver.log:151-160`).

The sampler is a sampled peak, not a continuous high-water mark; spikes between samples remain possible (`reports/static-scaling-full-campaign-61493236-review.md:84-86`).

Readiness marker: `FULL_CAMPAIGN_READY PASS batch=256 updates=390` (`scripts/static_scaling/check_full_campaign.py:942-981`).

### 2.4 Campaign

The campaign runs 45 cases. For each case:

- Flux W=1 runs directly; Flux W>1 uses `srun --overlap --mpi=pmi2 --kill-on-bad-exit=1 -n W -c 32/W`;
- PyTorch W>1 uses `srun --overlap -n1 -c32 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=W`
  (`scripts/remote/leonardo/run-static-full-campaign.sh:501-547`).

Each case runs 20 warmup updates, synchronizes, rebuilds the model from the bundle, rebuilds the optimizer state, synchronizes, then times 390 updates from batch 1 and schedule index 1 (`scripts/static_scaling/full_dataset_timing.jl:144-175`; `scripts/static_scaling/timing_torch.py:224-236`).

### 2.5 Diagnostics

A separate untimed matrix runs the same batch, epoch, 390 updates, and 20 warmup, and writes one final-state snapshot per configuration (`scripts/remote/leonardo/run-static-full-campaign.sh:801-813`; `scripts/static_scaling/full_dataset_timing.jl:197-211`; `scripts/static_scaling/timing_torch.py:248-259`).

Snapshots contain:

- Flux: `param.<name>`, `m.<name>`, `v.<name>`, `beta_power.<name>` for all 62 leaves, plus `step`;
- PyTorch: `param.<name>`, `m.<name>`, `v.<name>`, plus `step`.

### 2.6 Artifact check, summary, checksums

The driver invokes:

```text
check_full_campaign.py campaign --root OUT
```

then writes `REPORT.json`, `SUMMARY.json`, and `SHA256SUMS` (`scripts/remote/leonardo/run-static-full-campaign.sh:815-860`).

Actual final markers:

```text
FULL_CAMPAIGN_ARTIFACT_CHECK PASS
TIMING_DRIVER_EXIT=0
FULL_CAMPAIGN PASS utc=2026-10-19T23:55:23Z host=lrdn1300.leonardo.local job=61493236
```

The review independently ran `sha256sum --check --quiet SHA256SUMS` on the remote evidence: all **307** entries passed, including nine snapshots. The local metadata copy matched all **298** non-snapshot entries (`reports/static-scaling-full-campaign-61493236-review.md:15-16`).

## 3. Correctness and accuracy harness

### 3.1 What the campaign verifies

The campaign harness verifies that:

1. the intended workload ran: same dataset, epoch, permutation, augmentation, bundle, schedule, batch, updates, and source identity;
2. each timing record has the expected schema, framework, world size, batch, updates, warmup, dataset/bundle/order hashes, and finite final loss;
3. `seconds` equals the maximum rank duration;
4. throughput and update-time fields satisfy their arithmetic definitions;
5. rank replicas agree on final state hashes within each run;
6. each diagnostic snapshot is finite, has the expected array-name sets, has `step=390`, and has exact Flux `beta_power` values;
7. driver markers, runtime versions, resources, source snapshot, provenance hashes, and checksums are consistent.

Sources: `scripts/static_scaling/check_full_campaign.py:596-730,733-787,984-1035,1242-1273,1327`; `scripts/static_scaling/full_dataset_timing.jl:186-194`; `scripts/static_scaling/timing_torch.py:239-247`.

### 3.2 Upstream correctness gates

The campaign does not rerun the earlier Phase 3/4/5 acceptance suites; those are upstream prerequisites. This matters for how strong the correctness claim can be.

| Gate | What it established |
|---|---|
| Phase 2 data/fixtures | Dataset schedule, 62-array/11,271,432-parameter bundle, hashes; real-`$PREP` recheck job 60951247 passed 113 checks (`reports/static-scaling-phase2-verification.md:17-29`; `wiki/context/current.md:504-522,861-871`) |
| Phase 3 single-GPU | 3 updates at batch 12; 337-array PyTorch reference; 11 activations, logits, losses, 62 gradients, 62 params/m/v, beta powers; 32/32 `CUDNN_FMA_MATH`, zero fallback; accepted in job 60956582 (`reports/static-scaling-phase3-acceptance-review.md:16`; `reports/static-scaling-phase3-review-response.md:163-188`) |
| Phase 4 distributed | 3 updates at batch 12; 202 categories; finite strict `abs(a-b) < 1e-3`; exact structural/shard/rank/counter/hash/collective-protocol checks; 64-collective guarded vs 62-collective static Flux; accepted in job 61296074 (`artifacts/logs/static-scaling/2026-10-18-phase4/RESULTS.md:3-5,18-21`; `scripts/static_scaling/check_phase4.py:1-48,3626-3645,3760-3877`) |
| Loader validation | Job 61450014: 99,840 effective ids, 160 tail dropped, 390 updates; rank sequences match Python at W=1/2/4; update-1 transformed tensor `max_abs_diff=0.0` (`wiki/devlog/03-static-scaling/2026-10-19-full-dataset-loader-validation.md:74-104`) |
| Fixture timing pilot | Job 61314790: batch 12, 300 updates, 30/30 records; Flux flat at W2/W4 (0.9594/0.9680), PyTorch scaled (1.5602/1.8060); motivation for the full-dataset campaign (`reports/static-scaling-fixture-timing-review.md:5-10`; `wiki/devlog/03-static-scaling/2026-10-19-fixture-timing-campaign.md:30-80`) |
| 300-update diagnostic | Job 61366015: 6/6 records and snapshots; all finite; exact Flux beta powers; no exactly equal `param`/`m`/`v` arrays; descriptive only (`wiki/devlog/03-static-scaling/2026-10-19-timing-review-response.md:28-60`) |

Phase 3/4 are **short-horizon** gates: 3 updates at batch 12. Their tolerances and acceptance rules do not automatically extend to 390 updates at batch 256. OD-14 explicitly separates short-horizon correctness from long-run diagnostics (`experiments/static-scaling/OD-14-benchmark-acceptance.md:31-43`). Phase 4 CPU validation is deferred; the accepted distributed gate is GPU-only (`artifacts/logs/static-scaling/2026-10-18-phase4/RESULTS.md:5`).

A Phase 5 selected-batch checker exists, but this campaign did not invoke it. The campaign's preflight is the memory gate; the 390-update state comparisons are descriptive diagnostics.

The earlier fixture-timing review had already noted that full-length numerical agreement remained unmeasured. The full campaign adds descriptive 390-update snapshots, not a thresholded equivalence test (`reports/static-scaling-fixture-timing-review.md:18-46`).

### 3.3 Campaign record and snapshot gates

The checker requires exactly 45 timing files named `<framework>-w<world>-r<repeat>.json`, with no missing or extra files. Each record must satisfy the schema and policy fields, and `seconds` must equal `max(rank durations)` (`scripts/static_scaling/check_full_campaign.py:596-651,1242-1247`).

The snapshot checks are structural and arithmetic:

- only numeric, finite arrays;
- parameter names nonempty;
- `m.*` and `v.*` name sets equal the parameter set;
- `step == 390`;
- Flux `beta_power.*` name set equals the parameter set, and every leaf exactly equals the repeated Float32 multiplication `0.9^390` / `0.999^390`;
- no unexpected categories (`scripts/static_scaling/check_full_campaign.py:984-1035`).

All 45 timing records and all nine diagnostic records passed these checks. All 45 satisfy `seconds == max(rank durations)` (Flux `rank_seconds`, PyTorch `ranks[].seconds`); all have `batch=256`, `updates=390`, `warmup=20`, `n_eff=99,840`, `tail=160`, finite `final_loss`, and identical dataset/order/bundle/eta hashes.

### 3.4 Diagnostics: what was compared

The diagnostic report contains **15 comparisons**:

- 6 within-framework scaling comparisons: current W2/W4 vs W1, historical W2/W4 vs W1, PyTorch W2/W4 vs W1;
- 9 cross-framework comparisons: current-vs-PyTorch, historical-vs-PyTorch, and current-vs-historical at W1, W2, W4
  (`scripts/static_scaling/check_full_campaign.py:1157-1173`).

Categories are `param`, `m`, `v`, and (for Flux-only comparisons) `beta_power`. There are 62 parameter leaves per category and 11,271,432 parameters total.

The relative L2 for a leaf is `||candidate - reference||_2 / ||reference||_2`; the category and overall values use concatenated arrays. The denominator is the reference L2. A zero denominator yields 0 for an exact match and infinity otherwise (`scripts/static_scaling/check_full_campaign.py:1038-1101`).

The report explicitly sets:

```json
"threshold": null,
"NUMERICAL_EQUIVALENCE_CLAIM": "none"
```

### 3.5 Diagnostic results

| Category | Relative L2, all 15 | Relative L2, within-framework | Relative L2, cross-framework | Exact arrays |
|---|---:|---:|---:|---:|
| `param` | 1.8877–2.0819% | 1.9395–2.0819% | 1.8877–2.0447% | 0/930 |
| `m` | 18.1823–30.3619% | 20.5273–30.2335% | 18.1823–30.3619% | 0/930 |
| `v` | 5.8652–8.7686% | 6.2200–8.6635% | 5.8652–8.7686% | 0/930 |
| `beta_power` | 0 in 7 Flux-only comparisons | 0 in 4 | 0 in 3 | 434/434 |

Additional observations:

- All 15 comparisons have `exact=false`; zero parameter arrays are exactly equal.
- Worst parameter leaf: `37.1930%` relative L2 (`flux-current-vs-pytorch-w4`, `param.stage1.block0.gn1.bias`, absolute difference `0.0027637`). “About 2%” is an aggregate parameter statistic, not a per-leaf bound.
- All seven Flux-involving comparisons have exact `beta_power` for all 62 leaves. `beta_power` is optimizer exponent state, not a validation of parameter updates.
- Within each diagnostic run, all rank replicas agree on final state hashes. This is hash-level replica agreement; rank 0 writes the snapshot, so the campaign does not compare full arrays across ranks. Phase 4 provides the exact array-level replica checks at 3 updates/batch 12.
- Cross-framework differences are the same order as within-framework world-size differences. Historical-vs-PyTorch parameter relative L2 (`1.8877–1.8940%`) is smaller than historical's own W2-vs-W1 difference (`2.0015%`). The diagnostics therefore do not show a clean separation between “framework effect” and “world-size effect.”
- Flux final weights are **not repeatable across repeats**: all five repeats at every Flux world size have distinct final hash tuples. PyTorch has one unique final hash tuple per world size, i.e. its five repeats are hash-identical. The diagnostic snapshots for Flux are separate runs and do not match any timing repeat; PyTorch diagnostics match all five timing repeats. Single-snapshot Flux diagnostics therefore include run-to-run nondeterminism.

### 3.6 Self-tests and negative tests

- `check_full_campaign.py --self-test` prints `FULL_CAMPAIGN_SELFTEST PASS checks=21`. It covers one synthetic accepted campaign, two preflight selection cases, and 18 rejection mutations (missing record, wrong update count, wrong commits, rank-hash mismatch, nonfinite snapshot, wrong step, runtime mismatch, batch mismatch, claim mutation, eta mismatch, extra record, and related cases) (`scripts/static_scaling/check_full_campaign.py:1637-1778`).
- `scripts/static_scaling/test_full_campaign.py` has 8 CPU-only regression tests: cross-language batch-256 schedule hash, PyTorch GPU UUID normalization, Julia runtime-capture parse, precompile declaration, batch-256-only fail-closed selection, ready-before-timing ordering, stale/ready decision mutations, and missing/wrong-GPU/single-sample rejection (`scripts/static_scaling/test_full_campaign.py:26-190`).
- Phase 4 has a 36-check self-test, including the strict `<1e-3` boundary, nonfinite rejection, shard reconstruction, AdamW oracle, hashing, and protocol mutation cases (`artifacts/logs/static-scaling/2026-10-18-phase4/RESULTS.md:66`; `scripts/static_scaling/check_phase4.py:3012-3301`).
- The local data/fixture campaign includes independent seed/shard/transform/tail/bundle checks and negative tests (`scripts/static_scaling/selftest_local.py:33-199`; `scripts/static_scaling/negative_tests_local.py:1-70,278-480`).

### 3.7 What the harness does not verify

- **No long-horizon reference or numerical equivalence.** There is no independent 390-update oracle. The 15 diagnostics are cross-run comparisons, not correctness tests.
- **No predictive accuracy or training quality.** No validation loss, top-1/top-5, learning curves, or time-to-quality. The validation split is not in the required dataset identity.
- **No per-update loss or gradient scan.** The runner asserts the final pre-update loss and writes `pass_checks=true` / `loss_finite=true`; those flags are runner-declared. There is no loss trajectory and no independent scan of every gradient.
- **No full array-level replica comparison in the campaign.** Within-run agreement is hash-level; full array comparison is in Phase 4 at 3 updates.
- **No live cuDNN algorithm audit.** The campaign records the precision policy string and source snapshot, not per-choice `CUDNN_FMA_MATH` logs.
- **No campaign memory re-gating.** The 80% criterion is applied to the 20-update preflight, not re-applied to the 390-update campaign records.
- **No raw large-file dataset rehash.** `train_images.npy`, `train_labels.npy`, and `val_images.npy` use manifest SHA-256 values unless `verify_large=true`; the campaign uses `verify_large=false`. Sizes are checked; the 1.23 GB training-image array was not freshly hashed.
- **No git-level provenance.** Remote `.git` is absent; commits are expected constants assigned by the launcher. The evidence proves copied source content and `src/` tree hashes, not live git history or dirty/clean state.
- **No deterministic-mode sensitivity.** Flux uses strict FMA selection without a determinism filter; PyTorch uses deterministic cuDNN with autotuning disabled. OD-14 still requires a separate deterministic-mode sensitivity for the final Phase 6 cross-framework report.

## 4. Results

### 4.1 Timing results

Throughput is `99,840 / median_seconds`. W1 is the baseline for each implementation. Medians include all five repeats.

| Implementation | W1 median (s) | W2 median (s) | W4 median (s) | W1 images/s | W2 images/s | W4 images/s |
|---|---:|---:|---:|---:|---:|---:|
| `flux-current` | 91.104884 | 52.211284 | 27.923536 | 1,095.9 | 1,912.2 | 3,575.5 |
| `flux-historical` | 91.088748 | 47.457502 | 25.529042 | 1,096.1 | 2,103.8 | 3,910.8 |
| `pytorch` | 95.735126 | 49.788537 | 28.591314 | 1,042.9 | 2,005.3 | 3,492.0 |

| Implementation | W2 speedup | W4 speedup | W2 efficiency | W4 efficiency |
|---|---:|---:|---:|---:|
| `flux-current` | 1.744927 | 3.262656 | 87.2464% | 81.5664% |
| `flux-historical` | 1.919375 | 3.568044 | 95.9688% | 89.2011% |
| `pytorch` | 1.922835 | 3.348399 | 96.1417% | 83.7100% |

All nine summary rows reproduce exactly from the 45 raw records (zero difference in median, min, max, spread, sample standard deviation, throughput, speedup, and efficiency). `REPORT.timing` and `SUMMARY.timing` are identical (`scripts/static_scaling/check_full_campaign.py:1198-1228`; independent recomputation).

### 4.2 Dispersion and the first-repeat outlier

| Configuration | Median (s) | Mean (s) | Min (s) | Max (s) | Stdev (s) |
|---|---:|---:|---:|---:|---:|
| current W1 | 91.104884 | 209.024270 | 91.044960 | 680.785013 | 263.722274 |
| current W2 | 52.211284 | 52.247323 | 52.126326 | 52.475740 | 0.139358 |
| current W4 | 27.923536 | 27.924293 | 27.889720 | 27.968895 | 0.035006 |
| historical W1 | 91.088748 | 91.082342 | 91.054303 | 91.102591 | 0.020046 |
| historical W2 | 47.457502 | 47.463932 | 47.421918 | 47.518762 | 0.041495 |
| historical W4 | 25.529042 | 25.541598 | 25.498229 | 25.591086 | 0.040479 |
| PyTorch W1 | 95.735126 | 95.731952 | 95.724772 | 95.737069 | 0.005797 |
| PyTorch W2 | 49.788537 | 49.789031 | 49.734692 | 49.834033 | 0.037988 |
| PyTorch W4 | 28.591314 | 28.587325 | 28.570502 | 28.592600 | 0.009430 |

The current W1 repeats are:

```text
r1 = 680.785013230 s
r2 =  91.044959709 s
r3 =  91.073118117 s
r4 =  91.104883958 s
r5 =  91.113374216 s
```

The first repeat is **7.4725×** the median. The median includes it. Dropping r1 shifts the W1 median by only 0.0174% and shifts the W2/W4 speedups by 0.0174%. The headline scaling conclusions are robust to this outlier.

Using the mean would be wrong for the baseline: mean current W1 is 209.024 s, which would imply physically implausible W2/W4 speedups above the GPU count. The median is the appropriate robust statistic. It does not describe cold-start performance.

**Cause of the first-repeat outlier is unconfirmed.** The saved evidence has no page-fault or cache-state telemetry. The earlier cold-cache hypothesis is plausible but not established. Use “first-repeat outlier, cause unconfirmed” (`reports/static-scaling-full-campaign-61493236-review.md:53-62`).

The outlier is also **not JIT**. The timer starts after the 20-update warmup, the reset, a barrier, and a CUDA synchronize; compilation is outside the measured region (`scripts/static_scaling/full_dataset_timing.jl:163-175`). The campaign start marker (`21:32:46Z`) and the first record write time (`21:46:10Z`) imply about 13.4 minutes of wall time, of which 680.8 s (11.35 minutes) was measured; roughly 2 minutes covered startup/JIT/setup. The 680.8 s is time inside the 390-update loop, not compilation. The outlier is internally consistent: `seconds = max(rank duration)` and `update_seconds × 390 = seconds`.

### 4.3 Scaling analysis

- **Current Flux** scales from 1,095.9 to 1,912.2 images/s at W2 (1.745×, 87.2% efficiency) and to 3,575.5 images/s at W4 (3.263×, 81.6% efficiency). W2 is its weakest point.
- **Historical Flux** scales from 1,096.1 to 2,103.8 images/s at W2 (1.919×, 96.0%) and to 3,910.8 images/s at W4 (3.568×, 89.2%). It is the fastest arm at W4.
- **PyTorch** scales from 1,042.9 to 2,005.3 images/s at W2 (1.923×, 96.1%) and to 3,492.0 images/s at W4 (3.348×, 83.7%).

“Historical scales best” is supported **only for W1→W4 at W4**. At W2, PyTorch's observed speedup (1.922835) is slightly greater than historical's (1.919375), a difference of 0.003459 (0.180% relative). However, historical has higher **absolute** W2 throughput: 2,103.8 vs 2,005.3 images/s (+4.912%), because its W1 baseline is 5.101% higher.

The W2→W4 incremental efficiency is highest for current Flux:

| Implementation | W2→W4 incremental efficiency |
|---|---:|
| `flux-current` | 0.934897 |
| `flux-historical` | 0.929481 |
| `pytorch` | 0.870693 |

So historical is best on the full W1→W4 interval, but not on every sub-interval. All three arms are sublinear at W4.

### 4.4 Cross-framework comparisons

Percent difference is `(throughput_A / throughput_B − 1) × 100`; the baseline is explicit.

| World | historical vs current | current vs PyTorch | historical vs PyTorch |
|---|---:|---:|---:|
| W1 | +0.0177% | +5.0823% | +5.1009% |
| W2 | +10.0169% | −4.6403% | +4.9118% |
| W4 | +9.3795% | +2.3915% | +11.9952% |

Interpretation:

- Current and historical are descriptively tied at W1 (`−0.0177%` for current vs historical).
- Current beats PyTorch at W1 (+5.08%) and W4 (+2.39%), but loses at W2 (−4.64%).
- Historical is faster than PyTorch at every world size; the gap grows to +12.0% at W4.
- At W4, current has higher absolute throughput than PyTorch (+2.39%) but lower speedup (3.263 vs 3.348), because current's W1 baseline is 5.08% higher. Absolute throughput and speedup answer different questions; report both.

### 4.5 Evidence integrity

- Job `61493236`: `COMPLETED`, exit `0:0`, elapsed `02:46:11`, node `lrdn1300`.
- 45/45 timing records, 9/9 diagnostic records, 9/9 snapshots.
- 307/307 remote checksum entries pass (review); 298/298 local non-snapshot metadata entries match.
- All 45 records: `seconds == max(rank durations)`, finite `final_loss`, identical dataset/order/bundle/eta identity, correct batch/updates/warmup, and rank-consistent hashes.
- 9/9 diagnostics: finite snapshots, correct name sets, `step=390`, exact Flux `beta_power` for all 62 leaves, within-run replica hash agreement.

**Post-review resolution of Slurm step `61493236.9`.** The adversarial review flagged step `61493236.9` (`bash`, `FAILED 2:0`, `2026-10-19T23:45:22`–`23:46:22`) as unexplained. It was the operator's 60-second diagnostic `srun --overlap` that polled `/proc/<pid>/io` for the running Julia process, not a campaign workload. `sacct` shows the step's SubmitLine as `srun --jobid=614932+`; it exited 2 at the end of the sleep when the Julia PID disappeared and `cat /proc/<pid>/io` failed with “No such file or directory.” This matches the operator console output. It does not affect the campaign evidence or the 45 records.

## 5. Threats to validity and required qualifications

1. **First-repeat outlier cause is unconfirmed.** Do not call it cold cache without a controlled cache experiment. The median includes the outlier and does not describe cold-start.
2. **Guard overhead is not isolated.** Current Flux has presence and reduced-gradient guard work inside the timed window; historical Flux has no such guard; PyTorch has no equivalent timed guard. The results compare recorded implementations and policies, not intrinsic framework speed (`reports/static-scaling-full-campaign-61493236-review.md:71-78`).
3. **Other policy differences remain.** Flux uses strict FMA selection without a determinism filter; PyTorch uses deterministic cuDNN with autotuning disabled. PyTorch creates Adam state lazily inside the measured updates after its optimizer reset. These differences affect timing and numerics.
4. **The results are descriptive, from one allocation and five repeats.** There are no confidence intervals, hypothesis tests, or cross-allocation repeats. Do not claim statistical superiority (`reports/static-scaling-full-campaign-61493236-review.md:47-49`).
5. **Batch selection was prescribed, not searched.** Only batch 256 was tested. The campaign does not show that 512 is infeasible or that 256 is optimal.
6. **Memory is a sampled peak.** The 49,728 MiB peak is the maximum of 77–1,645 samples per rank, not a continuous high-water mark. The 80% criterion was applied to the 20-update preflight; the campaign does not re-gate the 390-update records.
7. **No numerical equivalence claim.** `threshold=null`, `NUMERICAL_EQUIVALENCE_CLAIM=none`. “About 2%” is aggregate parameter relative L2; individual leaves can differ much more (worst observed leaf 37.193%). First moments differ by 18.18–30.36% and second moments by 5.87–8.77% relative L2.
8. **No training-quality claim.** One epoch, inside the warmup schedule, no validation accuracy, no learning curves, no time-to-quality. “Same workload” does not mean “same learned model.”
9. **Flux diagnostics include run-to-run nondeterminism.** Flux has 5/5 unique final hash tuples per configuration across repeats; PyTorch is bit-identical. The single-snapshot diagnostics for Flux therefore mix world-size effects with run-to-run variation. Cross-framework differences are the same order as within-framework differences.
10. **Correctness evidence is short-horizon.** Phase 3 (3 updates, batch 12) and Phase 4 (3 updates, batch 12) are strong short-horizon gates. They do not establish numerical correctness at 390 updates/batch 256. The campaign's long-run checks are structural and descriptive.
11. **Provenance is content-based, not git-based.** Remote `.git` is absent; commits are expected constants. The current Flux state hash `e3b0c442…` is the SHA-256 of an empty diff, not a content hash. The thesis state is recorded but not pinned. Flux `ext/` is in the source snapshot but not in the `src/` tree hash. Large dataset arrays are manifest-hashed, not freshly rehashed.
12. **The measured epoch is inside the warmup schedule.** The batch-256 schedule uses `W=1,950`, `T=19,500`; the 390 measured updates are all within the warmup ramp. This is appropriate for throughput, but it limits any learning-dynamics interpretation.
13. **Deterministic-mode sensitivity is outstanding.** OD-14 requires it for the final Phase 6 cross-framework report; it was not run in this campaign (`experiments/static-scaling/OD-14-benchmark-acceptance.md:21-29`).
14. **The review did not recompute snapshot differences.** It inspected the saved diagnostic report and checksum integrity; it did not recompute the 15 comparisons from the nine `.npz` snapshots or rerun training (`reports/static-scaling-full-campaign-61493236-review.md:20-21`).

## 6. Conclusions

### 6.1 Defensible conclusions

1. On this Leonardo allocation, historical Flux is the fastest arm at W4: 25.529 s median, 3,910.8 images/s, 3.568× speedup, 89.2% efficiency.
2. Historical Flux is 9.38% faster than current Flux at W4 and 12.00% faster than PyTorch. At W2, historical is 10.02% faster than current and 4.91% faster than PyTorch.
3. Current Flux scales to 3.263× at W4 (81.6% efficiency). It beats PyTorch at W1 (+5.08%) and W4 (+2.39%), but is 4.64% slower than PyTorch at W2.
4. PyTorch has the highest observed W2 speedup by a small margin (1.922835 vs 1.919375), but not the highest absolute W2 throughput.
5. The first current-W1 repeat is a 7.47× outlier; the median is robust, and the headline speedups change by only 0.0174% if the outlier is dropped. The cause is unconfirmed.
6. The campaign artifacts are complete and internally consistent: 45/45 timing records, 9/9 diagnostics, 307/307 checksums, exact timing arithmetic, finite states, and rank-consistent hashes.

### 6.2 Explicit non-claims

- No numerical equivalence between the three implementations.
- No equal training quality or predictive accuracy.
- No causal guard-overhead estimate.
- No intrinsic framework superiority.
- No statistical significance or cross-allocation generalization.
- No claim that cold cache caused the first-repeat outlier.
- No claim that batch 256 is optimal or that batch 512 is infeasible.
- No claim that every Slurm allocation step succeeded; step `61493236.9` was an operator diagnostic `srun`, not a campaign workload.

## 7. Next steps

1. **Keep the qualified descriptive label** in the final Phase 6 report: “qualified descriptive throughput and strong scaling for the recorded workload.”
2. **Run the OD-14 deterministic-mode sensitivity** in both frameworks, with its own correctness checks and source identity. This is the main outstanding prerequisite for a stronger cross-framework claim.
3. **Run a controlled cache experiment** if a causal explanation of the first-repeat outlier is needed. Until then, keep “cause unconfirmed.”
4. **Record or revisit OD-8.** The current memory evidence is a sampled preflight peak, not a continuous high-water measurement.
5. **Archive the evidence** and preserve `SHA256SUMS`, `REPORT.json`, `SUMMARY.json`, the 45 records, the 9 diagnostics, and the source snapshot.
6. **Do not use the campaign diagnostics as an equivalence test.** Any stronger numerical or training-quality claim needs a prospective addendum with references, thresholds, seeds, repeated-run treatment, validation criteria, and budget.
7. **Update the project wiki** with the campaign outcome, the step `.9` resolution, and the remaining OD-14/OD-8 actions.

## Appendix A. Evidence and reproduction

Key evidence files:

```text
static-full-campaign-61493236/
├── driver.log
├── SUMMARY.json
├── REPORT.json
├── SHA256SUMS
├── campaign-config.json
├── provenance.json
├── source-identity.txt
├── source-sha256.txt
├── launcher-sha256.txt
├── runtime-env.txt
├── runtime.json
├── resources.txt
├── resources.json
├── dataset-identity.json
├── preflight/decision.json
├── records/*.json                  # 45 timing records
├── diagnostics/records/*.json      # 9 diagnostic records
├── diagnostics/snapshots/*.npz     # 9 state snapshots
└── logs/
```

Key markers:

```text
FULL_CAMPAIGN_READY PASS batch=256 updates=390
FULL_CAMPAIGN_preflight PASS batch=256 k=390
FULL_CAMPAIGN_timing PASS records=45
FULL_CAMPAIGN_diagnostics PASS snapshots=9
FULL_CAMPAIGN_ARTIFACT_CHECK PASS
FULL_CAMPAIGN PASS utc=2026-10-19T23:55:23Z host=lrdn1300.leonardo.local job=61493236
```

Runtime: Julia 1.12.6; Flux 0.16.11; CUDA.jl 6.2.2; NCCL 0.1.3; MPI 0.20.27; Optimisers 0.4.9; Functors 0.5.3; Python 3.11.17; torch 2.9.1+cu126; CUDA runtime 12.6; cuDNN 91002; driver 535.274.02.

Source identity:

```text
thesis        a3786d97f3263cad20b254206f09c095880f624d
current Flux  f7b4520c4bd307c67bbc5ac92a1c68493eccff6e
current src   83da2929ab7ebd66dbceb83452c8e55eae5e0444c8e96866cc2a25b513c3eb0d
current state e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  # empty-diff hash
historical    8add0e42b7d9388ff365335152126e5b1e9de657
historical src ad114ed385f7cc78e8409fa8e1f09d38133555f7fd58d5d9aca6a846bcf5e757
historical state 4af8409c2a2ed1c661e8ae7742877f1ee9789d00dc7e404365ecefd98ab0d34c
```

Documented submission form (the saved allocation used `--time=12:00:00`; the example in the launcher uses `04:00:00`):

```bash
CONFIRM_CAMPAIGN=1 CAMPAIGN_MODE=all \
sbatch --parsable \
  -A iscrc_unmasked -p boost_usr_prod --qos=boost_qos_lprod \
  -N1 -n1 --cpus-per-task=32 --gres=gpu:4 --time=04:00:00 \
  --job-name=static-full-campaign \
  --output="$WORK/.../static-full-campaign-%j.log" \
  "$HOME/projects/ddp-flux-thesis/scripts/remote/leonardo/run-static-full-campaign.sh"
```

The driver then runs `check_full_campaign.py campaign --root OUT`, writes `REPORT.json` and `SUMMARY.json`, and generates `SHA256SUMS` (`scripts/remote/leonardo/run-static-full-campaign.sh:815-860`).

Key checks:

```bash
# Remote checksum verification (review)
ssh leonardo 'cd /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-full-campaign-61493236 && sha256sum --check --quiet SHA256SUMS'

# Slurm accounting
ssh leonardo 'sacct -j 61493236 --format=JobID,JobName%28,State,ExitCode,Start,End,Elapsed,NodeList -X'
ssh leonardo 'sacct -j 61493236.9 --format=JobID,JobName%40,State,ExitCode,Start,End,Elapsed,SubmitLine'

# Local arithmetic reproduction
python3 - <<'PY'
import json, glob, os, statistics
from collections import defaultdict
recs = {os.path.basename(p)[:-5]: json.load(open(p))
        for p in glob.glob('/tmp/opencode/campaign/full/records/*.json')}
g = defaultdict(list)
for k, d in recs.items():
    g[k.rsplit('-r', 1)[0]].append(d['seconds'])
for base, v in sorted(g.items()):
    print(base, statistics.median(v), statistics.mean(v), min(v), max(v),
          max(v) - min(v), statistics.stdev(v), 99840 / statistics.median(v))
PY
```

## Appendix B. Adversarial-review claim audit

| Review claim | Status |
|---|---|
| 45 timing + 9 diagnostic records | Supported |
| All medians, throughputs, speedups, efficiencies reproduce | Supported; zero difference |
| Each duration equals max rank duration | Supported, 45/45 |
| 390 updates × batch 256 = 99,840; tail 160 | Supported |
| W1 medians 91.104884 / 91.088748 / 95.735126 | Supported |
| W2 medians 52.211284 / 47.457502 / 49.788537 | Supported |
| W4 medians 27.923536 / 25.529042 / 28.591314 | Supported |
| Speedups 1.744927/3.262656; 1.919375/3.568044; 1.922835/3.348399 | Supported |
| Historical vs current W4 +9.3795% | Supported |
| Current vs historical W1 −0.0177% | Supported |
| Current vs PyTorch W1 +5.0823% | Supported |
| Current vs PyTorch W2 −4.6403% | Supported |
| Current vs PyTorch W4 +2.3915% | Supported |
| “Historical scales best” needs “at W4” | Qualification supported |
| PyTorch W2 speedup slightly greater than historical | Supported |
| First repeat 7.47× median | Supported |
| Preflight accepted batch 256 in all nine | Supported |
| Peak 49,728 MiB vs 51,955.8 MiB limit | Supported as sampled |
| 15 comparisons; threshold null; claim none | Supported |
| Param 1.8877–2.0819%; m 18.18–30.36%; v 5.87–8.77% | Supported |
| Zero exact parameter arrays | Supported, 0/930 |
| Seven Flux-only comparisons have exact beta powers, 62/62 | Supported, 434/434 |
| Exact within-run replicas; 390 updates | Supported |
| Slurm step `61493236.9` failed, purpose unresolved | Now resolved: operator diagnostic `srun`, not campaign workload |
| Job `COMPLETED` 0:0, 02:46:11, node `lrdn1300` | Supported by remote Slurm; local copy lacks accounting |
| Remote 307 checksum entries pass | Supported by review; remote-only |

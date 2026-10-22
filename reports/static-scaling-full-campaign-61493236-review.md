# Campaign 61493236 claim review

Date: 2026-10-20.
Verdict: the arithmetic and saved campaign PASS are supported, with reporting qualifications.

## Evidence checked

The remote evidence root is:

`/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-full-campaign-61493236/`

- Slurm reports `COMPLETED`, exit `0:0`, and elapsed `02:46:11` on `lrdn1300`.
- The driver records preflight, readiness, timing, diagnostics, artifact-check, and campaign PASS markers.
- The evidence contains all 45 timing records and all nine diagnostic records.
- Remote `sha256sum --check --quiet SHA256SUMS` returned 0 for all 307 entries, including nine snapshots.
- The local metadata copy matched all 298 non-snapshot checksum entries.
- Independent arithmetic used all 45 raw timing records and reproduced every median, throughput, speedup, and efficiency.
- Each recorded duration equals the maximum rank duration.

This review inspected the saved diagnostic report and checksum integrity. It did not recompute differences from the nine snapshot arrays.
It did not run new training or repeat the earlier correctness gates.

## Timing results

Each configuration measured 390 updates at global batch 256: 99,840 images from epoch 0, with a 160-image tail excluded.
Each repeat used 20 discarded warmup updates, then reset the model and optimizer.

| Implementation | W1 median seconds | W2 median seconds | W4 median seconds | W2 speedup | W4 speedup |
|---|---:|---:|---:|---:|---:|
| flux-current | 91.104884 | 52.211284 | 27.923536 | 1.744927 | 3.262656 |
| flux-historical | 91.088748 | 47.457502 | 25.529042 | 1.919375 | 3.568044 |
| pytorch | 95.735126 | 49.788537 | 28.591314 | 1.922835 | 3.348399 |

Throughput is `99840 / median_seconds`.
Speedup is `median_seconds_W1 / median_seconds_W`.
Efficiency is `speedup / W`.
All values in the submitted table reproduce at its stated precision.

The throughput comparisons also reproduce:

- Historical versus current at W4: **+9.3795%**.
- Current versus historical at W1: **-0.0177%**, which supports “approximately equal.”
- Current versus PyTorch at W1: **+5.0823%**.
- Current versus PyTorch at W2: **-4.6403%**.
- Current versus PyTorch at W4: **+2.3915%**.

“Historical scales best” requires the qualifier **at W4**.
PyTorch has slightly greater observed speedup at W2: 1.922835 versus 1.919375.
These are descriptive comparisons from one allocation, without a statistical superiority claim.

## Required qualifications

### Outlier cause remains unproven

Current W1 times were `680.785013230`, `91.044959709`, `91.073118117`, `91.104883958`, and `91.113374216` seconds.
The five-repeat median is correct and includes the outlier.
The first repeat is 7.47 times the median.

The saved evidence does not establish a cold-cache cause.
The timer includes memory-mapped batch access, but the logs contain no page-fault or cache-state measurements that establish causation.
Use “first-repeat outlier, cause unconfirmed” until a controlled cache experiment supports the explanation.
The median does not describe cold-start performance.

### Timing scope and execution policies

Both timers include batch assembly, transformations, host-to-device transfers, forward execution, backward execution, and optimizer updates.
They exclude process startup, dataset-map opening, initial setup, warmup, and snapshot output.
They use synchronized GPU boundaries and the maximum rank duration.
PyTorch creates Adam state lazily during the measured updates after its optimizer reset.

Current Flux includes distributed presence and non-finite guards inside the timer.
Historical Flux uses the static distributed API without those guards.
PyTorch uses native DDP without an equivalent timed guard.
Flux uses strict FMA selection without a determinism filter. PyTorch uses deterministic cuDNN selection with autotuning disabled.

Therefore, these results compare the recorded implementations and policies.
They do not isolate guard overhead or establish an intrinsic framework advantage.
OD-14 still requires deterministic-mode sensitivity for the final Phase 6 cross-framework report.

### Batch selection

The preflight accepted batch 256 in all nine configurations.
The launcher tested only batch 256 because the user selected it before the campaign.
“Preflight accepted the prescribed global batch 256” is more precise than language that implies an automatic capacity search.
The greatest sampled process peak was 49,728 MiB against a 51,955.8-MiB acceptance limit.
The sampling does not establish a continuous memory high-water mark.

### Diagnostics

The report contains 15 comparisons and explicitly sets `threshold: null` and `NUMERICAL_EQUIVALENCE_CLAIM: none`.
Aggregate parameter relative L2 ranges from **1.8877% to 2.0819%**.
The denominator is the L2 norm of concatenated reference arrays within each category.
A zero reference norm produces zero for an exact match and infinity otherwise.

“About 2%” describes aggregate parameter differences, not each parameter leaf or the optimizer state.
First-moment relative L2 ranges from **18.18% to 30.36%**.
Second-moment relative L2 ranges from **5.87% to 8.77%**.
The report records zero exact parameter arrays in every comparison.
All seven Flux-only comparisons have exact beta powers for all 62 leaves.
The saved records show exact final replica agreement within each distributed diagnostic run and 390 completed updates.
These observations do not establish numerical equivalence or equal training quality.

### Allocation history

Slurm also records step `61493236.9`, named `bash`, as `FAILED 2:0`.
Its scheduler timestamps are `2026-10-19T23:45:22` through `23:46:22`.
The driver and Slurm output contain no corresponding workload failure or retry marker.
Its purpose remains unresolved. The completed allocation and complete record set remain supported.
Do not describe every allocation step as successful without an explanation of this step.

## Exact next action

Identify the command for Slurm step `61493236.9` from the operator history.
Keep the first-repeat cause unconfirmed until a controlled cache experiment supplies evidence.
Use the qualified descriptive label in the campaign summary.

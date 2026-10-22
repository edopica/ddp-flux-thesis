# OD-14 — Benchmark policy and numerical claims

Date: 2026-10-19.
Status: Recorded, following the user instruction to formalize OD-14.
Scope: Phase 5 memory measurements and Phase 6 benchmark acceptance and reporting.

## Decision

The primary benchmark measures the selected implementations under explicit execution policies.
Its accepted claim is **descriptive throughput and strong scaling for the recorded workload**.
Numerical equivalence and comparable training quality are separate claims with separate evidence requirements.
No long-run discrepancy threshold is adopted from the observed 300-update envelope.

### 1. Execution policy

- Both frameworks use FP32 with TF32 disabled and the shared model, optimizer, data, and update schedule.
- Flux retains strict FMA candidate selection without a determinism filter.
- PyTorch retains `cudnn.deterministic=True`, `cudnn.benchmark=False`, and `float32_matmul_precision="highest"`.
- Flux retains the OD-11 explicit update path and its gradient-finiteness guard inside the timer.
- PyTorch retains native DDP without an equivalent timed gradient guard.
- Algorithm IDs need not match. Different loaded cuDNN versions must appear in the report.

These policies define the baseline. They do not isolate communication cost or establish an intrinsic framework advantage.
The final Phase 6 cross-framework report must include a separate deterministic-mode sensitivity in both frameworks.
That sensitivity must retain FP32, the workload, and guard placement, and document the actual determinism controls and limitations.
It must receive its own correctness checks and source identity before measurement.
Deterministic convolution selection alone does not establish deterministic execution of every operation.
Until that sensitivity exists, cross-framework results retain the qualified descriptive label.
The sensitivity is not a prerequisite for the Phase 5 capacity pilot.

### 2. Short-horizon implementation correctness

The accepted Phase 3 and Phase 4 gates remain independent prerequisites for correctness-backed benchmark claims.
Phase 3 retains OD-10 and its three category-specific overrides.
Phase 4 GPU comparisons retain finite `abs(a-b) < 1e-3`, with exact structural and rank checks.
Neither rule extends automatically to a longer trajectory, another batch, or another execution policy.

The benchmark must identify the source and runtime covered by those gates.
Changed training dependencies require a documented comparison and relevant renewed correctness checks.
Recorded helper drift alone does not establish equivalent behavior.
The historical helper drift remains an explicit limitation until that evidence exists.

### 3. Long-run diagnostics and acceptance

Long-run snapshots are diagnostics, not numerical-equivalence tests.
The required diagnostic uses the campaign workload and update count in a separate untimed run.
It must cover all six configurations and retain canonical parameters, Adam moments, and update-state metadata.
The fixture diagnostic at 300 updates does not substitute for a different full-dataset workload or update count.

The diagnostic must establish:

- Identical input and initial-bundle identities, sample exposure, update count, and learning-rate schedule.
- Complete state structure and finite loss, final parameters, and optimizer moments.
- Exact final replica agreement within each distributed run.
- Exact completed-update counts and consistent beta-power evolution under the recorded representation.

The report must separate these comparison groups:

1. Flux W=2 and W=4 against Flux W=1.
2. PyTorch W=2 and W=4 against PyTorch W=1.
3. Flux against PyTorch at each matching W.

Each group must report `max_abs`, relative L2, exact-array counts, and the worst leaves for parameters and moments.
The report must define the relative-L2 denominator and zero-norm behavior.
Repeated-run variability must remain separate from differences across W or frameworks.
One snapshot per configuration does not estimate repeated-run numerical variability.

Finite nonzero differences receive no numerical PASS or FAIL threshold under this decision.
The report must retain `NUMERICAL_EQUIVALENCE_CLAIM none`.
Non-finite state, inconsistent replicas, missing updates, or mismatched work fail the diagnostic prerequisite for benchmark acceptance.
A successful diagnostic permits descriptive reporting. It does not replace short-horizon correctness evidence.

### 4. Stronger claims require prospective criteria

A numerical-equivalence claim requires a separate recorded addendum before its acceptance campaign.
The addendum must specify references, comparison groups, horizon, metrics, thresholds, and repeated-run treatment.
Its rationale must use numerical analysis or independent controls, not a threshold fitted solely to already observed differences.
Calibration evidence and acceptance evidence must be distinct.

A comparable-training-quality claim also requires declared validation-loss and accuracy criteria, seeds, training budget, and statistical treatment.
Those criteria must precede the learning-curve acceptance runs.
Throughput alone cannot establish comparable quality or time to a common quality target.
Until such an addendum exists, learning curves and long-run differences remain descriptive.

### 5. Timing and provenance requirements

Timing acceptance requires identical measured work, synchronized timer boundaries, maximum rank duration, complete repeats, and correct ratio-of-medians statistics.
The report must state warmup reset behavior, timing exclusions, and optimizer initialization inside the measured window.
Every accepted campaign must retain input hashes, source hashes, launcher identity, runtime paths and versions, device mapping, and effective resource policy.
The recorded environment must identify the loaded packages and libraries, not only copied manifests.
Runtime capture documents an environment. It does not make a live environment immutable.
Source or policy changes require a new campaign identity and relevant correctness checks.

## Evidence and historical disposition

Campaign `61314790` remains a qualified fixture timing PASS.
Job `61366015` remains a completed descriptive diagnostic with `NUMERICAL_EQUIVALENCE_CLAIM none`.
This decision does not retroactively certify source drift or remove missing historical runtime evidence.

The diagnostic's printed comparisons include parameter `max_abs=0.0332728997` and `rel_l2=0.00225564012`.
The latter is the Flux-versus-PyTorch W=2 value, slightly larger than the rounded summary's `0.00224`.
These values are observations, not acceptance bounds. They do not extend Phase 4's `<1e-3` rule.

Evidence:

- [Timing review response](../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/REVIEW-RESPONSE.md)
- [300-update diagnostic](../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/numerics-61366015/RESULTS.md)
- [Phase 4 acceptance](../../artifacts/logs/static-scaling/2026-10-18-phase4/RESULTS.md)

## Consequences and next action

OD-14 is closed as a policy decision. Its future evidence requirements are not marked complete.
OD-7, OD-8, OD-9, OD-12, and OD-13 remain separate decisions.
The machine-readable policy records requirements. It does not imply that every runner enforces them automatically.

**Exact next action:** run the Phase 5 larger-batch memory pilot under this baseline policy.
Record the OD-7 batch and OD-8 memory-margin decisions before the full-dataset campaign.

# Fixture timing correctness review

Date: 2026-10-19. Campaign: `61314790`, Leonardo, `lrdn3064`.

## Verdict

**Qualified timing PASS.** The saved durations, throughput, speedups, and efficiencies reproduce correctly.
Source inspection found no timing-boundary, sample-count, or gradient-averaging defect.
The results describe this batch-12 fixture and these implementation policies.
The evidence does not establish numerical equivalence through all 300 measured updates.

This review used local source and artifacts. No remote or GPU run occurred.
The timing scripts are untracked, so preserved source hashes provide stronger identity than commit history.
External documentation was unnecessary for the findings.

## Findings

### R1 — P2: The checker accepts incomplete correctness evidence

`scripts/static_scaling/check_timing.py:23–40` checks record consistency and timing arithmetic.
It does not enforce precision, schema, job identity, or required state-hash keys.
It does not require Flux state hashes.

An in-memory negative control changed every precision field, schema, and job identity.
It also deleted Flux hashes and replaced PyTorch rank hashes with empty dictionaries.
The preserved checker still printed `TIMING_ARTIFACT_CHECK PASS`.
The actual campaign records contain the expected input identities. This finding does not demonstrate false campaign metadata.

**Required correction:** enforce required fields and expected identities before this checker serves as a correctness acceptance gate.

### R2 — P2: Full-length numerical agreement remains unmeasured

The runners check the last pre-update loss and final replica hashes:

- `scripts/static_scaling/timing_flux.jl:59–68`
- `scripts/static_scaling/timing_torch.py:71–82`

Replica agreement does not establish agreement with W=1 or the other framework.
The last loss precedes the final optimizer update. Its finiteness does not establish finite final parameters and moments.
The accepted Phase 3/4 comparisons cover three updates, rather than this 300-update trajectory.

Each Flux configuration has five distinct final-state hash sets across five repeats.
Each PyTorch configuration has one final-state hash set across five repeats.
Hashes cannot quantify these differences. This observation does not demonstrate incorrect Flux results.

**Required evidence:** an untimed full-length run with finite-state checks and numerical discrepancies against declared references and tolerances.

### R3 — P2: Cross-framework policies differ

PyTorch requires deterministic cuDNN and disables benchmarking (`scripts/static_scaling/model_torch.py:350–367`).
Flux selects FMA-compatible algorithms by time and memory, without a determinism filter (`scripts/static_scaling/precision.jl:394–405`).
Both paths disable TF32, but their algorithm policies differ.

Flux DDP also includes gradient-finiteness checks and synchronization inside the timed training path.
The preserved implementation records these checks in:

- `source/src/distributed/public_api.jl:767–792`
- `source/ext/FluxMPINCCLExt/FluxMPINCCLExt.jl:87–95`

These paths are relative to the campaign directory.
The PyTorch timed loop has no equivalent gradient scan (`scripts/static_scaling/timing_torch.py:50–58`).
Flux W=1 uses ordinary optimizer state, without the distributed checks.

The measurements remain useful for these selected implementations.
They combine communication, validation, algorithm selection, and framework overhead.
They do not isolate NCCL efficiency or explain the cause of flat Flux scaling.
OD-14 already records the algorithm-policy decision for later benchmarks.

### R4 — P2: Provenance does not fully connect acceptance and timing

All 36 script hashes match the preserved snapshot and current scripts.
The preserved Flux source also matches the current integration worktree: 26 `src` files and 10 `ext` files.
All 30 records match the canonical fixture and initial-bundle hashes.

However, the launcher loads live environments rather than the preserved snapshot.
The snapshot omits the launcher and a complete record of the loaded test/helper environment.
See `scripts/remote/leonardo/run-static-timing.sh:17–25`.

The timing hashes for `phase4_common.jl` and `phase4_torch.py` differ from the earlier Phase 4 provenance.
The model, replay, and precision hashes match that record.
The changed helper hashes require a documented comparison before claiming complete source identity with accepted Phase 4.
This is an evidence gap, not evidence of source substitution.

### R5 — P3: Warmup and statistical scope need explicit labels

Both runners discard the 20-update warmup state before the measured 300 updates.
PyTorch retains warmed DDP buckets but creates a fresh optimizer.
Flux creates a fresh model and optimizer state.
See `timing_flux.jl:50–59` and `timing_torch.py:60–71`.

Thus, the measured window starts from fresh mathematical state, including any lazy optimizer initialization.
It is not a continuous 320-update trajectory or a fully warmed optimizer measurement.
The measured schedule remains within its 975-update learning-rate warmup.
The 3,600 image exposures repeat two saved batches.

W=1 uses GPU 0, whereas W=2 uses GPUs 1 and 2.
The mapping matches between frameworks by source inspection, but Flux records omit device identities.
The launcher also uses different CPU-affinity arrangements for MPI ranks and torchrun workers.

Five fresh processes and alternating configuration order provide useful pilot controls.
All repeats share one node and allocation. They do not establish variation across allocations or hardware.

## Confirmed measurement properties

- Both runners synchronize CUDA before and after the measured work.
- Distributed barriers precede the timer.
- Reported duration is the maximum rank duration.
- Input staging, H2D transfer, hashes, and output occur outside the timer.
- Local batches are 12, 6, and 3 for W=1,2,4.
- Equal-rank averaging of local mean gradients gives the global mean.
- Both paths use the canonical initial bundle, AdamW groups, and update-indexed schedule.
- All 30 records are present. Ratios use medians, rather than averages of per-repeat ratios.

| Framework | W | Median seconds | Images/s | Speedup | Efficiency |
|---|---:|---:|---:|---:|---:|
| Flux | 1 | 5.483059 | 656.568 | 1.0000 | 1.0000 |
| Flux | 2 | 5.714868 | 629.936 | 0.9594 | 0.4797 |
| Flux | 4 | 5.664201 | 635.571 | 0.9680 | 0.2420 |
| PyTorch | 1 | 8.203725 | 438.825 | 1.0000 | 1.0000 |
| PyTorch | 2 | 5.258007 | 684.670 | 1.5602 | 0.7801 |
| PyTorch | 4 | 4.542426 | 792.528 | 1.8060 | 0.4515 |

## Local commands and evidence

```bash
python3 scripts/static_scaling/check_timing.py artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790
python3 /tmp/opencode/audit61314790.py
bash -n scripts/remote/leonardo/run-static-timing.sh
git status --short
git log -5 --oneline
git diff --stat
git log --oneline -- scripts/static_scaling/timing_flux.jl scripts/static_scaling/model_flux.jl scripts/remote/leonardo/run-static-timing.sh
git -C ../flux-integration status --short
git -C ../flux-integration log -3 --oneline
```

All shell commands succeeded. The temporary audit script independently recomputed statistics and checked source and input hashes.
It used `unittest.mock.patch` on `Path.read_text` for the negative control, without altering campaign files.
The temporary script is session-local, rather than a permanent project test.

The research-agent launch failed because its configured model was unavailable.
A general agent completed the review instead. A guessed launcher filename also failed during a read-tool lookup.
These tool failures did not affect the benchmark artifacts.

## Exact next action

Harden the artifact checker and define an untimed 300-update numerical gate with pinned source/runtime identity.
Resolve the benchmark policy in OD-14 before the full-dataset comparison.
The Phase 5 memory pilot can proceed as a capacity experiment, with these correctness limitations explicit.

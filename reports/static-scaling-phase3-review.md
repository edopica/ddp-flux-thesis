# Static-scaling Phase 3 status review

Date: 2026-10-17
Status: implemented, numerical acceptance FAIL, root cause not yet proven.
Scope: local source inspection and saved Leonardo logs. This review ran no training or remote commands.

## Verdict

The agent report accurately describes the main implementation and numerical results.
Phase 3 cannot close until the declared comparisons pass under a recorded precision policy and frozen tolerances.
The cuDNN FMA experiment is the correct next diagnostic step. It is not yet an accepted fix.

## Evidence

The plan defines Phase 3 in `temp/static-graph-gpu-scaling-implementation-phases.md:131-159`.
Both model modules implement the specified architecture, bundle conversion, parameter groups, and library AdamW updates.
The Flux module uses `CrossCor`, `Flux.@layer`, Functors traversal, and `Optimisers.setup`/`update!`.

Saved run: `artifacts/logs/leonardo/2026-10-17-phase3-remote-attempt3.log`, job 60885091.
Its first line records the exact remote command and source identities.

| Category | Result | Maximum absolute difference |
|---|---|---:|
| PyTorch generator | PASS, 32 checks | — |
| Independent artifact checker | PASS, 337 arrays | — |
| Initial parameters, groups, initial moments | PASS | — |
| Learning rate, beta powers, step counter | PASS | 0 for float comparisons |
| Activations | FAIL | 7.64757395e-3 |
| Logits | FAIL | 6.50286674e-4 |
| Loss | PASS | 1.14917755e-4 |
| Gradients, update 1 | FAIL | 5.23567945e-3 |
| Gradients, update 3 | FAIL | 7.80658051e-3 |
| Parameters, update 3 | FAIL | 1.17514282e-5 |
| First moments, update 3 | FAIL | 1.31687149e-3 |
| Second moments, update 3 | PASS | 2.97278166e-6 |

The comparison uses `atol=1e-5`, `rtol=1e-4` elementwise, including the relative term.
Loss and second-moment PASS results do not establish overall correctness.
The driver correctly exits 1 and prints `PHASE3_REMOTE_CHECK FAIL`.

## Findings and acceptance gaps

### R1 — Precision requirement remains unproven

`check_phase3_flux.jl:355-376` checks `CUDA.PEDANTIC_MATH`, not the effective cuDNN convolution math type.
The local cuDNN source at `~/.julia/packages/cuDNN/3NrK1/src/cuDNN.jl:50-61` maps pedantic mode to `CUDNN_DEFAULT_MATH`.
This supports the proposed mechanism, but the exact remote package and effective descriptors still need a recorded check.
The printed `cuda_math_precision=TensorFloat32` alone does not prove that a kernel used TF32.

The stage-2 GPU diagnostic reports a `3.5725832e-3` convolution difference with identical saved input.
The projection convolution matches exactly. The CPU diagnostic reports `7.62939453e-6`, with zero tolerance violations.
These results strongly isolate a GPU convolution-path difference for this block.
They do not prove full-model CPU equivalence or uniquely identify TF32 instead of another kernel/algorithm difference.

Evidence: `2026-10-17-phase3-diagnose-stage2.log`, `2026-10-17-phase3-diagnose-stage2-cpu.log`, and `diagnose_phase3_flux_stage2.jl:43-56`.
Both diagnostics use the saved PyTorch stage-1 activation directly.

Required next evidence: a controlled default-versus-FMA comparison on identical arrays, then the complete three-update comparison.
The precision policy must cover forward convolution, both convolution gradients, and classifier matrix multiplication.
A Julia method redefinition affects process-wide dispatch. A module wrapper alone does not make that override local to one model.
Any workaround needs explicit process scope, version checks, source identity, and reuse by later training runners.

### R2 — Phase 3 decisions remain open

`experiments/static-scaling/open-decisions.md:12-21` still lists OD-2, OD-3, OD-4, OD-10, and OD-11 as open.
The code implements their proposed defaults. That implementation does not replace a decision record.
OD-10 requires calibration before acceptance evidence, followed by a separate run under the frozen configuration.
An unexplained error decrease or a relaxed threshold cannot close R1.

### R3 — Optimizer and schedule sensitivity needs stronger evidence

The three updates use learning rates from `1.0256e-6` to `3.0769e-6`.
Their sum is about `6.15e-6`, less than the parameter comparison's absolute tolerance.
The combined decay coefficient is about `6.15e-10` at `lambda=1e-4`.
Thus, final-parameter agreement alone gives weak evidence for actual updates and decay at these learning rates.
The source correctly configures library AdamW, but the numerical acceptance needs a sensitive optimizer probe.

Recommended evidence: controlled gradients, nonzero initial moments, decay/no-decay leaves, and a meaningful learning rate.
The probe must reject omitted updates, incorrect decay scaling, and incorrect epsilon placement.
This remains a test of library configuration, not a custom optimizer for training.
Schedule checks currently exercise only updates 1, 2, and 3.
Boundary checks at `W`, `W+1`, and `T` can cover the untested cosine branch without a long training run.

### R4 — Evidence retention and prerequisite closure remain incomplete

No Phase 3 reference archive or manifest exists in the local expected directory during this review.
The saved logs establish remote results, but this review could not independently rehash those artifacts.
The driver always adds `--force` (`run-phase3-check.sh:173-178`).
A rerun therefore replaces the reference archive. Preserve the failed-run reference or use a new output directory before the next experiment.

The saved Phase 3 runs establish restored Leonardo access.
They do not establish completion of the hardened Phase 2 remote re-check against the real PREP.
The Phase 3 driver does not run that prerequisite check.

Source identities appear in the log, but the driver accepts them from the caller.
`check_phase3_reference.py:377-385` checks provenance types without rejecting `unknown`, empty strings, or null values.
This is an acceptance-hardening gap, not evidence that the reported source identities are wrong.
The final evidence package needs the Julia manifests, Python freeze, source-state recipe, and retained artifact hashes.

## Update 2026-10-17 (Step 3b) — R1 root cause confirmed and corrected

The R1 mechanism is confirmed. cuDNN.jl 6.2.2's
`cudnnConvolutionAlgoPerfChoose` skips the math-type check for Float32,
so the FMA descriptor alone did not constrain the full-model algorithm
selection. `scripts/static_scaling/precision.jl` now installs the
FMA-aware `math_mode` mapping and more-specific forward/backward
algorithm-chooser methods at module load, and
`configure_precision!` clears the three cuDNN algorithm perf caches
(their keys omit the math type).

Focused regression: `PHASE3_PRECISION_REGRESSION PASS` (job 60910394);
the new autotuned NNlib-path item chose `algo=0`, `CUDNN_FMA_MATH(3)`,
`fallback=false`, `rel_l2=4.248559e-7`.

Full comparison (job 60910786, unchanged `atol=1e-5`/`rtol=1e-4`): all
32 recorded choices are `CUDNN_FMA_MATH(3)` with no fallback and every
candidate math type is FMA. The forward now passes
(`activations 1.56e-5`, `logits 1.67e-6`, `loss 4.77e-7`), as do
`params.u3` and `v.u3`. `grads.u1` (`3.03e-4`), `grads.u3` (`6.67e-4`),
and `m.u3` (`6.58e-5`) still fail. The chosen backward algorithms are
FP32 FFT-tiling/Winograd/legacy-direct variants (algo 3, 4, 5, 0), while
the PyTorch reference is deterministic cuDNN 9.10.2.21. The residual is
FP32 backward algorithm rounding, not a TF32 path.

The new reference is again 337/337 exact-equal to the preserved one and
the independent NumPy checker passes. No tolerance, model-math,
optimizer, or Flux change. Evidence:
`artifacts/logs/static-scaling/2026-10-17-phase3-step3b/`.

## Exact next action

Phase 3 acceptance remains open on the backward residual. Before Step 4,
the user must choose: (1) restrict backward selection to
deterministic/direct algorithms that match the PyTorch reference, or
(2) justify a cross-framework tolerance revision numerically. Then rerun
the full driver in a fresh evidence root with all four summaries PASS.
Do not widen tolerances or pin algorithms without that decision.

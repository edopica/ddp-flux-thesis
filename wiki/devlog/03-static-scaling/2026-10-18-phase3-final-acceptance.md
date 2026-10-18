# Phase 3 final acceptance — matched single-GPU Flux/PyTorch PASS

Date: 2026-10-18
Status: **PASS — Phase 3 accepted.** No commit.
Handoff: [static-graph-gpu-scaling-phase3-handoff.md](../../../temp/static-graph-gpu-scaling-phase3-handoff.md), Step 6.
Evidence: [RESULTS.md](../../../artifacts/logs/static-scaling/2026-10-18-phase3-final/RESULTS.md),
[COMMANDS.md](../../../artifacts/logs/static-scaling/2026-10-18-phase3-final/COMMANDS.md),
[PROVENANCE.md](../../../artifacts/logs/static-scaling/2026-10-18-phase3-final/PROVENANCE.md).

## Final source snapshot

| Item | Value |
|---|---|
| Thesis HEAD | `3088f61020137912af32d9c969e87ea9ae2a71c3` |
| Thesis state SHA-256 | `4a28adb1b031ddacb14aff3bce314d204e693c2d55910cdfe766d5ff33f131fe` |
| Flux HEAD | `f7b4520c4bd307c67bbc5ac92a1c68493eccff6e` |
| Flux state SHA-256 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (clean) |

The state recipe is `git diff HEAD` plus SHA-256 of the sorted untracked
non-artifact files, hashed together. `git diff --check` was clean in both
trees. The 93 thesis worktree entries are all Phase 2/3 work; no unintended
change or secret was found. Per-file hashes:
[`source-sha256.txt`](../../../artifacts/logs/static-scaling/2026-10-18-phase3-final/source-sha256.txt).

## Sync and precompile

The final tree was synced with an explicit-exclusion rsync (not
`sync_code.sh`), excluding the historical `2026-10-17-phase3-*` reference
archives including the two preserved attempt-3 dirs. Remote hashes of
`precision.jl`, `check_phase3_flux.jl`, `run-phase3-check.sh`, the
precision/optimizer probe drivers, both experiment configs, and the probe
sources all matched local.

Precompile job `60954207` on `lrdn3362` printed `FLUX_PRECOMPILE_OK`, exit 0.
No later log contains `Precompiling packages...`.

## Focused probes

| Probe | Job / node | Result |
|---|---|---|
| Precision regression | `60954547` / `lrdn2211` | `PHASE3_PRECISION_REGRESSION PASS`; FMA case passes, stock `DEFAULT_MATH` control fails `conv_fixed_fwd/dx/dw` at `~2.93e-4` |
| Optimizer/schedule | `60955690` / `lrdn0279` | `PHASE3_OPTIMIZER_PROBE PASS`; 69 checks, 34/34 negative controls rejected, `eta(1)/W/(W+1)/T` exact |

The precision regression confirms the accepted policy: cuBLAS
`PEDANTIC_MATH`, cuDNN `CUDNN_FMA_MATH`, FP32-level convolution
forward/data-gradient/weight-gradient and classifier matmul, and an
autotuned `Flux.CrossCor` path with `algo=0`, `CUDNN_FMA_MATH(3)`,
`fallback=false`.

## Final driver

Job `60956582` on `lrdn0165`, UTC start `2026-10-18T01:25:21Z`, duration
`5m26.4s`, exit 0. Fresh roots:
`$WORK/epicazio/static-scaling-evidence/2026-10-18-phase3-final/reference`
and `.../logs`. No `ATOL`, `RTOL`, or category override was passed.

| Category | Status | max abs diff | violations |
|---|---|---:|---:|
| eta | PASS | 0 | 0 |
| activations | PASS | 1.56164169e-5 | 0 |
| logits | PASS | 2.2649765e-6 | 0 |
| loss | PASS | 4.76837158e-7 | 0 |
| grads.u1 | PASS | 3.02568078e-4 | 0 |
| grads.u3 | PASS | 6.21065497e-4 | 0 |
| params.u3 | PASS | 5.93136065e-6 | 0 |
| m.u3 | PASS | 7.2401017e-5 | 0 |
| v.u3 | PASS | 2.32830644e-7 | 0 |
| beta_power.u3 | PASS | 0 | 0 |
| step | PASS | reference 3 = Flux 3 | 0 |

All four summaries PASS: `PHASE3_REFERENCE PASS`,
`PHASE3_REFERENCE_CHECK PASS`, `PHASE3_FLUX_CHECK PASS`,
`PHASE3_REMOTE_CHECK PASS`. The checker printed the frozen rule:

```text
PHASE3_ACCEPTANCE_RULE base="|a - b| <= 1.0e-5 + 0.0001 * max(|a|, |b|)" phase3.grads.u1="|a - b| <= 0.001 + 0.0001 * max(|a|, |b|)" phase3.grads.u3="|a - b| <= 0.002 + 0.0001 * max(|a|, |b|)" phase3.m.u3="|a - b| <= 0.0002 + 0.0001 * max(|a|, |b|)" other_categories=base
PHASE3_CUDNN_ALGO_SUMMARY choices=32 fma=32 fallback=0 enabled=true
```

Per-leaf relative L2 maxima (supplementary): `grads.u1 1.998783e-3`
(worst `grad.u1.stage1.block0.conv1.weight`), `grads.u3 4.648105e-3`
(`grad.u3.stage1.block1.conv1.weight`), `params.u3 4.420585e-3`
(`param.u3.stem.gn.bias`), `param_increment.u3 1.263869e-2`
(`delta.u3.stage1.block1.conv1.weight`), `m.u3 2.640012e-3`
(`m.u3.stage1.block0.conv1.weight`), `v.u3 2.925289e-3`
(`v.u3.stage1.block1.conv1.weight`); zero zero/nonzero disagreements.

Chosen algorithms: forward `0` (7), `6` (4); backward data `0` (7), `4` (3);
backward filter `0` (3), `3` (5), `5` (3). All are FP32 FMA families.

## Precision cause and correction

The original failure was reproduced and explained:

1. Stock cuDNN.jl maps `PEDANTIC_MATH -> CUDNN_DEFAULT_MATH`, which permits
   TF32. Isolated stage-2 convolution differed by `3.5725832e-3`
   (`conv1`) and `3.64590809e-3` (`conv2`); CPU was at `~7.6e-6`.
2. Explicit `CUDNN_FMA_MATH(3)` removed the discrepancy (`conv1 0`,
   `conv2 1.90734863e-6`, zero violations).
3. cuDNN.jl 6.2.2's Float32 autotuner skips the math-type check, so the
   descriptor alone did not constrain the full model.
   `scripts/static_scaling/precision.jl` installs an FMA-aware
   `math_mode` mapping and more-specific forward/backward chooser methods
   and clears the three algorithm caches.
4. The final run records `CUDA.math_mode()=PEDANTIC_MATH`, effective
   `CUDNN_FMA_MATH(3)`, `CUDA.allowscalar(false)`, empty caches before
   selection, and 32/32 FMA choices.

The residual backward differences are ordinary FP32 backward-algorithm
rounding, reproduced by the same-framework PyTorch CPU-vs-GPU control with
the same worst leaves. The frozen OD-10 floors retain at least 2x margin.

## Evidence retention

- Reference NPZ SHA-256
  `97e5007cbb557f4c4a476a00c7e8076e8c59858e58d597d0df88529985da0783`;
  manifest `5ba6c29f3a6167f4341e8e78b0685cb93a5b76e83575beca8db82d8585753eb7`.
- The final reference is 337/337 exact-equal to the preserved attempt-3
  reference, maximum absolute difference `0.0`.
- Independent NumPy checker locally: `PHASE3_REFERENCE_CHECK PASS
  arrays=337`.
- Remote/local SHA-256: 15/15 MATCH
  ([`SHA256-COMPARISON.txt`](../../../artifacts/logs/static-scaling/2026-10-18-phase3-final/SHA256-COMPARISON.txt));
  `sha256sum -c SHA256SUMS` passes.
- Runtime record: A100-SXM-64GB, driver `535.274.02`, Julia 1.12.6,
  CUDA.jl 6.2.2, cuDNN.jl 6.2.2 (library `92.60.0`), NNlib 0.9.45,
  Optimisers 0.4.9; Python 3.11.17, torch `2.9.1+cu126` (CUDA 12.6,
  cuDNN `9.10.2.21`), NumPy 2.4.6.
  [`runtime-versions.txt`](../../../artifacts/logs/static-scaling/2026-10-18-phase3-final/runtime-versions.txt).
- The hardened Phase 2 prerequisite remains PASS (job `60951247`,
  `lrdn0171`; 113 checks, exit 0).

## Failures during the campaign

- The first full rsync was killed by the local 900 s timeout while uploading
  historical reference archives; no source effect. The successful sync
  excluded those archives.
- Two metadata-collection jobs were superseded: `60959152` (quoting error)
  and `60959247` (header evaluated on the login node; uv venv has no `pip`).
  Acceptance evidence was unaffected.
- Leonardo SSH publickey rejection returned after the final evidence copy;
  all remote artifacts and remote SHA-256 values were already retained.

## Outcome

Phase 3 is accepted. Benchmarks are not part of Phase 3. **Exact next
action:** start Phase 4, distributed correctness at two and four ranks, per
`temp/static-graph-gpu-scaling-implementation-phases.md` section 6.

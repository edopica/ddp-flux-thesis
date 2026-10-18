# Static-scaling Phase 3 review response

Date: 2026-10-18
Review: [static-scaling-phase3-review.md](static-scaling-phase3-review.md)
Final evidence: [artifacts/logs/static-scaling/2026-10-18-phase3-final/](../artifacts/logs/static-scaling/2026-10-18-phase3-final/)
Final results: [RESULTS.md](../artifacts/logs/static-scaling/2026-10-18-phase3-final/RESULTS.md)
Final provenance: [PROVENANCE.md](../artifacts/logs/static-scaling/2026-10-18-phase3-final/PROVENANCE.md)
Status: **all four review findings addressed; Phase 3 accepted.**

## Verdict

The review's verdict (Phase 3 numerically FAIL, root cause not proven) was
correct for the evidence available on 2026-10-17. The root cause is now
proven and corrected, the open decisions are recorded, the optimizer and
schedule sensitivity gaps have a dedicated probe, and the final acceptance
run exits 0 with all four summaries PASS on a fresh evidence root.

No tolerance, model-math, optimizer, schedule, or precision-policy value was
changed during the final campaign. No commit was made.

## R1 — Precision requirement and isolated convolution

**Finding.** The checker tested `CUDA.PEDANTIC_MATH`, not the effective cuDNN
math type; the local cuDNN source mapped pedantic mode to
`CUDNN_DEFAULT_MATH`. The stage-2 GPU diagnostic differed by `3.57e-3`, but
TF32 was not proven. A precision policy had to cover forward convolution,
both convolution gradients, and classifier matrix multiplication.

**Resolution.**

1. The stock cuDNN.jl mapping is `PEDANTIC_MATH -> CUDNN_DEFAULT_MATH(0)`,
   which permits TF32. On identical saved stage-2 inputs and weights this
   reproduced `conv1 3.5725832e-3` / `conv2 3.64590809e-3` against the
   unchanged PyTorch reference; CPU was `7.63e-6` / `5.48e-6`.
2. Explicit `CUDNN_FMA_MATH(3)` removed the discrepancy
   (`conv1 0`, `conv2 1.90734863e-6`, zero violations). Requested and
   effective math types were recorded on `conv1`, `conv2`, and `proj`.
3. The descriptor alone was insufficient for the full model: cuDNN.jl 6.2.2
   `cudnnConvolutionAlgoPerfChoose` skips the math-type check for Float32, so
   the autotuner could select TF32-capable kernels. `scripts/static_scaling/precision.jl`
   now installs the FMA-aware `math_mode` mapping and more-specific
   forward/backward-data/backward-filter chooser methods at module load, and
   `configure_precision!` clears the three algorithm perf caches (their keys
   omit the math type).
4. The final focused regression (job `60954547`, `lrdn2211`) passes:
   `PHASE3_PRECISION_REGRESSION PASS`. The FMA case records `algo=0`,
   `CUDNN_FMA_MATH(3)`, `fallback=false` for the autotuned NNlib path; the
   stock `CUDNN_DEFAULT_MATH` negative control fails on
   `conv_fixed_fwd/dx/dw` at `~2.93e-4`.
5. The final full-model run records TF32 off in both frameworks,
   `CUDA.math_mode()=PEDANTIC_MATH`, effective `CUDNN_FMA_MATH(3)`,
   empty algorithm caches, `CUDA.allowscalar(false)`, and
   `PHASE3_CUDNN_ALGO_SUMMARY choices=32 fma=32 fallback=0`. All 32 choices
   are `CUDNN_FMA_MATH(3)`.

The residual backward differences (`grads.u1 3.03e-4`,
`grads.u3 6.21e-4`, `m.u3 7.24e-5`) are ordinary FP32 backward-algorithm
rounding. The same-framework PyTorch CPU-vs-GPU control reproduces the same
three categories with the same worst leaves, and the chosen families
(legacy direct, Winograd, FFT-tiling) are FP32 FMA kernels. The frozen OD-10
floors cover that envelope with at least 2x margin.

**Evidence.** Step 2:
`artifacts/logs/static-scaling/2026-10-17-phase3-step2-precision/`;
Step 3b:
`artifacts/logs/static-scaling/2026-10-17-phase3-step3b/`;
numerics:
`artifacts/logs/static-scaling/2026-10-17-phase3-numerics/`;
final:
`artifacts/logs/static-scaling/2026-10-18-phase3-final/precision/`.

## R2 — Open decisions

**Finding.** OD-2, OD-3, OD-4, OD-10, and OD-11 were still listed as open;
the code implemented proposed defaults but that did not replace a decision
record.

**Resolution.** All five decisions are recorded with user authority:

| ID | Recorded value |
|---|---|
| OD-2 | AdamW `lr=1e-3`, `wd=1e-4`, `betas=(0.9,0.999)`, `eps=1e-8` |
| OD-3 | Decay only on 21 convolution/classifier-weight leaves (`lambda=1e-4`); no decay on 41 GroupNorm/bias leaves (`lambda=0`) |
| OD-4 | 5 warm-up epochs, 50-epoch budget indexed by global update; batch 512 `W=975`, `T=9750`; batch-256 alternate `W=1950`, `T=19500` |
| OD-10 | Base `atol=1e-5`, `rtol=1e-4`; frozen floors `grads.u1 atol=1e-3`, `grads.u3 atol=2e-3`, `m.u3 atol=2e-4`, each with `rtol=1e-4` |
| OD-11 | Explicit `Zygote.withgradient` + `Optimisers.update!`; communication outside AD; 64 collectives per update |

`open-decisions.md`, `specification.md`, and `experiment.toml` agree and parse
in Python `tomllib` and Julia `TOML`. The values match the configuration used
by the accepted Steps 3c and 4. Evidence:
`experiments/static-scaling/open-decisions.md`,
`artifacts/logs/static-scaling/2026-10-18-phase2-hardened-recheck/`,
and the Step 4 probe.

## R3 — Optimizer and schedule sensitivity

**Finding.** The three training updates use rates `1.03e-6..3.08e-6`, so
final-parameter agreement alone is weak evidence for decay and the optimizer
contract. Schedule checks covered only updates 1-3.

**Resolution.** The compact cross-framework probe (job `60955690`,
`lrdn0279`) passed on the final snapshot: `PHASE3_OPTIMIZER_PROBE PASS`,
69 checks, 34/34 negative controls rejected.

- It asserts the real training-path configuration: Flux 62 leaves =
  21 decay (`lambda=1e-4`) / 41 no-decay (`lambda=0`); PyTorch two groups
  21/41; `beta=(0.9,0.999)`, `eps=1e-8`, `couple=true`.
- Three-update controlled sequences with nonzero `m`/`v` at probe
  `eta=1e-2`, compared against an independent Float64 arithmetic oracle.
  Flux/PyTorch/oracle maximum final differences are `5.38e-8..1.07e-7`.
- Negative controls reject omitted updates (ratio 1,204.8), wrong decay
  scale (44.9), omitted decay (5.10), decay on a no-decay leaf (5.36), wrong
  epsilon placement (1,204.7), wrong bias correction (996.0), and the
  warm-up formula at `W+1` (10.2).
- Schedule boundaries are exact: `eta(1)=1.0256410256410257e-6`,
  `eta(W)=1e-3`, `eta(W+1)=9.9999996795609018e-4`, `eta(T)=0`; both
  frameworks reject `t=0` and `t=T+1`.

Training still uses `Optimisers.AdamW` and `torch.optim.AdamW`; this is a
test of library configuration, not a custom optimizer. Evidence:
`artifacts/logs/static-scaling/2026-10-18-phase3-final/optimizer/`.

## R4 — Evidence retention and Phase 2 prerequisite

**Finding.** No Phase 3 reference archive or manifest was retained locally;
the driver always adds `--force`; the hardened Phase 2 remote re-check against
the real `$PREP` was not established; provenance types did not reject
`unknown`/empty/null.

**Resolution.**

- The attempt-3 reference was preserved before any rerun:
  `artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/`.
- The final run uses a fresh remote root
  `$WORK/epicazio/static-scaling-evidence/2026-10-18-phase3-final/` and a new
  local evidence root
  `artifacts/logs/static-scaling/2026-10-18-phase3-final/`.
- The final reference is byte-identical to the preserved attempt-3 reference
  (337/337 exact, maximum difference `0.0`).
- The independent NumPy checker passes locally on the copied reference and
  manifest: `PHASE3_REFERENCE_CHECK PASS arrays=337`.
- Remote/local SHA-256 comparison is 15/15 MATCH
  (`SHA256-COMPARISON.txt`); the retained `SHA256SUMS` verifies with
  `sha256sum -c`.
- The hardened Phase 2 prerequisite ran on the real `$PREP`/fixtures:
  job `60951247` (`lrdn0171`, CPU-only) printed `PHASE2_DATASET_CHECK PASS`,
  `PHASE2_JULIA_DATA_CHECK PASS`, `PHASE2_REMOTE_CHECK PASS`, 113 PASS /
  0 FAIL, driver exit 0. Evidence:
  `artifacts/logs/static-scaling/2026-10-18-phase2-hardened-recheck/`.
- The reference manifest records `source_commit=3088f610...`,
  `source_state_sha256=4a28adb1...`, `flux_commit=f7b4520c...`, and
  `flux_state_sha256=e3b0c442...`. The checker's required-field and
  string-type gate passed, and a separate local acceptance check rejects
  missing, empty, null, and `unknown` values (`PROVENANCE_CHECK PASS
  fields=4 bad=[]`, `provenance-check.log`). The checker itself still only
  enforces required fields and string types; stricter in-checker rejection
  of empty/`unknown` values remains a future hardening item, but it does not
  affect this acceptance record.
- The final evidence package includes runtime versions, Julia manifest
  hashes, the Python freeze, GPU/driver/CUDA/cuDNN versions, exact commands,
  job IDs, node names, and the failure history.

## Final acceptance

Job `60956582` on `lrdn0165`, UTC start `2026-10-18T01:25:21Z`, exit 0:

```text
PHASE3_REFERENCE PASS
PHASE3_REFERENCE_CHECK PASS
PHASE3_FLUX_CHECK PASS
PHASE3_REMOTE_CHECK PASS
```

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

Phase 3 is accepted. Benchmarks are not part of Phase 3. The next action is
Phase 4 distributed correctness at two and four ranks
(`temp/static-graph-gpu-scaling-implementation-phases.md` section 6).

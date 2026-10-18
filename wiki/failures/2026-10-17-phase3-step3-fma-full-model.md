# Phase 3 Step 3: FMA configuration does not change the full-model convolution path

Date: 2026-10-17
Status: **cause confirmed and fixed in Step 3b** (algorithm chooser now
honors the FMA descriptor); Step 3 acceptance remains blocked by a
different, backward-only FP32 algorithm-rounding residual.
Impact: the full three-update Flux comparison still fails the unchanged
`atol=1e-5`, `rtol=1e-4` rule after `configure_precision!(fma=true)`.

## Observed

Job `60904928` on `lrdn0125` ran the full driver in a new output root
`/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step3/`.
The checker printed `cudnn_requested=CUDNN_FMA_MATH(3)` and
`cudnn_effective=CUDNN_FMA_MATH(3)`, but the comparison failed:

| Category | max_abs | violations |
|---|---:|---:|
| activations | 7.64757395e-3 | 2,705,938 |
| logits | 5.64932823e-4 | 4,981 |
| grads.u1 | 5.24188578e-3 | 3,842,324 |
| grads.u3 | 7.48962909e-3 | 3,906,482 |
| params.u3 | 1.1857599e-5 | 5 |
| m.u3 | 1.47970638e-3 | 721,437 |

`eta`, `loss`, `v.u3`, `beta_power.u3`, and `step` passed.

The per-array activation violation counts are exactly the pre-FMA values
from `artifacts/logs/leonardo/2026-10-17-phase3-diagnose-gpu.log`
(job 60887065): stage2.block0 616,769; stage2.block1 864,972;
stage3.block0 346,553; stage3.block1 460,153; stage4.block0 179,821;
stage4.block1 233,331; pooled 4,339; maximum 0.00764757395. The new
run's forward output is therefore numerically unchanged from the
TF32-enabled run.

## Reference and fixtures

The driver wrote a new reference. It is array-for-array identical to the
preserved attempt-3 reference: 337/337 arrays exact-equal, zero
differing arrays, maximum absolute difference 0.0. The independent
NumPy artifact checker passes locally (`PHASE3_REFERENCE_CHECK PASS`,
exit 0). The failure is Flux-versus-reference, not a changed reference.

## Hypothesised cause

The cuDNN descriptor requests `CUDNN_FMA_MATH`, but cuDNN.jl 6.2.2's
Float32 algorithm chooser
(`cudnnConvolutionAlgoPerfChoose` in
`$WORK/epicazio/julia-depot/packages/cuDNN/0SFve/src/convolution.jl`)
sets `skipMathTypeCheck = (dtype == Float32)`. It can therefore select
an algorithm whose reported math type is TF32 even when the descriptor
requests FMA. The isolated stage-2 experiment happened to select
FMA-compatible algorithms; the full-model run selected a set that
reproduces the TF32-scale error. This matches the algorithm-selection
caveat recorded in the Step 2 devlog.

This is not ordinary FP32 reduction-order noise (the error is
TF32-scale and unchanged from the TF32 run) and not an optimizer/model
math defect (the forward already fails; loss, eta, step, beta powers,
and `v.u3` pass).

## Not done

- No tolerance or elementwise-rule change.
- No model math or optimizer change.
- No Flux source change.
- The preserved reference was not modified.

## Evidence

- Console: `artifacts/logs/leonardo/2026-10-17-phase3-step3.log`
- Remote-check log: `artifacts/logs/static-scaling/2026-10-17-phase3-step3/remote-check.log`
- Provenance/commands: `artifacts/logs/static-scaling/2026-10-17-phase3-step3/PROVENANCE.md`,
  `COMMANDS.md`
- Results and per-leaf diagnostics: `RESULTS.md`
- New reference: `artifacts/baselines/static-scaling/2026-10-17-phase3-step3/`
- Precompile job `60904055` on `lrdn3183` (`FLUX_PRECOMPILE_OK`).

## Resolution (Step 3b, 2026-10-17)

The hypothesised cause was confirmed and corrected. `precision.jl` now
installs, at module load, the FMA-aware `math_mode` mapping and
more-specific `cudnnConvolutionAlgoPerfChoose` methods that ignore
cuDNN.jl's Float32 math-type skip when the descriptor requests FMA. It
also clears the three cuDNN algorithm perf caches, whose keys omit the
math type. `check_phase3_flux.jl` records every choice and adds
`CHECK cudnn.fma_algo_selection`; `check_precision_math.jl` adds an
autotuned NNlib-path item.

Step 3b results (job `60910786`, node `lrdn0001`): precision regression
PASS (job `60910394`); all 32 full-model choices are
`CUDNN_FMA_MATH(3)` with `fallback=false`, and every candidate math type
is FMA. Activations, logits, loss, parameters, and `v.u3` now PASS.
Three backward categories remain over tolerance:
`grads.u1` `3.03e-4` (18,248 violations), `grads.u3` `6.67e-4`
(105,443), `m.u3` `6.58e-5` (1,304). Chosen backward algorithms are
FP32 FFT-tiling/Winograd/legacy-direct variants (algo 3, 4, 5, 0), while
the PyTorch reference runs deterministic cuDNN 9.10.2.21. This is an
FP32 algorithm-rounding residual, not a TF32 path.

Evidence: `artifacts/logs/static-scaling/2026-10-17-phase3-step3b/`
(`RESULTS.md`, `PROVENANCE.md`, `COMMANDS.md`, `algorithm-choices.txt`)
and `artifacts/baselines/static-scaling/2026-10-17-phase3-step3b/`.
The new reference is 337/337 exact-equal to the preserved attempt-3
reference and the independent NumPy checker passes.

The next decision belongs to the user: restrict backward selection to
deterministic/direct algorithms matching the PyTorch reference, or
justify a tolerance revision numerically. Do not start Step 4 before a
full run passes.

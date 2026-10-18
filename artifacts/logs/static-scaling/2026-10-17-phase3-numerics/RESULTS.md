# Phase 3 numerical calibration — independent-backend FP32 envelope

Date: 2026-10-17
Status: diagnostic only. No tolerance, elementwise rule, model math, optimizer,
checker, or Flux source change. No commit. The preserved PyTorch GPU reference
was not overwritten.

## Question

The Step 3b full comparison leaves `grads.u1`, `grads.u3`, and `m.u3` above the
unchanged rule `|a-b| <= 1e-5 + 1e-4*max(|a|,|b|)`. This calibration asks
whether that residual is ordinary FP32 algorithm/reduction variation between
valid backends or an outlier that suggests a Flux backward defect. It also
records, but does not apply, the smallest defensible tolerance revision if the
evidence supports one.

## Evidence identities

Source provenance used by the new CPU reference is identical to the preserved
GPU reference provenance (`source_commit=3088f610...`, `source_state_sha256=82623bbb...`,
`flux_commit=f7b4520c...`, `flux_state_sha256=e3b0c442...`).

| Item | Value |
|---|---|
| Remote evidence root | `$WORK/epicazio/static-scaling-evidence/2026-10-17-phase3-numerics/` |
| CPU reference job / node | `60912525` / `lrdn3450` |
| Flux CPU checker job / node | `60912664` / `lrdn3450` |
| PyTorch CPU reference | `torch_cpu_reference/phase3_reference.npz`, 337 arrays, SHA-256 `a7ff467b0e1d216a6d006e8ae576a97694e4a654d481bdec86b06e269a1217b3` |
| PyTorch CPU manifest | `torch_cpu_reference/phase3_reference_manifest.json`, SHA-256 `2ff9241c172a8f42fdffac81bc6fcb6907b3de9985c4d835f4ab17b52e6894ea`, `device="cpu"` |
| Preserved PyTorch GPU reference | `$WORK/.../2026-10-17-phase3-attempt3/reference/phase3_reference.npz`, SHA-256 `97e5007cbb557f4c4a476a00c7e8076e8c59858e58d597d0df88529985da0783` |
| Preserved PyTorch GPU manifest | `.../phase3_reference_manifest.json`, SHA-256 `2416e5557eb696e499a553fd1866d49c1c529a5fb778541df2f009825689f64d` |
| Generator checks | `PHASE3_REFERENCE PASS checks=32 failed=0` |
| Artifact checker | `PHASE3_REFERENCE_CHECK PASS arrays=337` |

The CPU reference uses the same fixture, bundle, three updates (`u1=[0]`,
`u2=[1]`, `u3=[0]`), seed-free frozen bundle, AdamW settings, and schedule as
the preserved GPU reference. Only `--device cpu` and the output directory
differ. `torch.set_num_threads(1)`; Julia used `JULIA_NUM_THREADS=1`.

## A. PyTorch CPU reference vs preserved PyTorch GPU reference

Same framework, same generator and inputs, different backend. Produced with
`scripts/static_scaling/compare_reference_arrays.py`.

| Category | max_abs | violations | max per-leaf rel L2 | worst leaf |
|---|---:|---:|---:|---|
| eta | 0 | 0 | 0 | — |
| loss | 4.768372e-7 | 0 | 8.584973e-8 | `loss.u2` |
| logits | 2.235174e-6 | 0 | 9.038754e-7 | `logits.u3` |
| activations | 1.764297e-5 | 0 | 1.549527e-6 | `act.stage4.block1` |
| grads.u1 | **3.021732e-4** | **14,365** | **1.839324e-3** | `grad.u1.stage1.block0.conv1.weight` |
| grads.u3 | **4.099458e-4** | **128,170** | **3.636289e-3** | `grad.u3.stage1.block1.conv1.weight` |
| params.u3 | 2.624467e-6 | 0 | 3.398208e-3 | `param.u3.stem.gn.bias` |
| m.u3 | **5.057361e-5** | **1,137** | **2.296527e-3** | `m.u3.stage1.block0.conv1.weight` |
| v.u3 | 6.893606e-8 | 0 | 2.115951e-3 | `v.u3.stage1.block0.conv1.weight` |
| param_increment.u3 | 2.624467e-6 | 0 | 1.204467e-2 | `delta.u3.stage1.block0.conv1.weight` |
| step | 0 | 0 | 0 | — |

Elementwise `max_rel` reaches 1–2 on near-zero elements; it is not used for the
verdict. The per-leaf relative L2 is the stable discriminator.

## B. Flux CPU vs preserved PyTorch GPU reference

Full checker, unchanged rule. Flux GPU (Step 3b, job `60910786`, node
`lrdn0001`) is repeated for context.

| Category | Flux CPU max_abs | Flux CPU viol. | Flux CPU max leaf rel L2 (worst leaf) | Flux GPU max_abs | Flux GPU viol. | Flux GPU max leaf rel L2 (worst leaf) |
|---|---:|---:|---|---:|---:|---|
| eta | 0 | 0 | — | 0 | 0 | — |
| activations | 1.716614e-5 | 0 | n/a | 1.561642e-5 | 0 | n/a |
| logits | 2.026558e-6 | 0 | n/a | 1.668930e-6 | 0 | n/a |
| loss | 0 | 0 | n/a | 4.768372e-7 | 0 | n/a |
| grads.u1 | **4.126653e-4** | **32,212** | **2.552219e-3** (`stage1.block0.gn1.bias`) | **3.025942e-4** | **18,248** | **1.998793e-3** (`stage1.block0.conv1.weight`) |
| grads.u3 | **7.480253e-4** | **102,364** | **4.220351e-3** (`stem.conv.weight`) | **6.672032e-4** | **105,443** | **4.536716e-3** (`stage1.block1.conv1.weight`) |
| params.u3 | 4.976988e-6 | 0 | 3.273980e-3 (`stage1.block0.gn1.bias`) | 4.775822e-6 | 0 | 4.412696e-3 (`stem.gn.bias`) |
| m.u3 | **7.226411e-5** | **946** | **2.386828e-3** (`stem.conv.weight`) | **6.584823e-5** | **1,304** | **2.615754e-3** (`stage1.block0.conv1.weight`) |
| v.u3 | 2.253801e-7 | 0 | 2.891345e-3 (`stem.conv.weight`) | 2.328306e-7 | 0 | 2.856201e-3 (`stage1.block1.conv1.weight`) |
| param_increment.u3 | 4.976988e-6 | 0 | 1.331823e-2 (`stage1.block0.conv2.weight`) | 4.775822e-6 | 0 | 1.260064e-2 (`stage1.block1.conv1.weight`) |
| beta_power.u3 | 0 | 0 | — | 0 | 0 | — |
| step | 3 = 3 | 0 | — | 3 = 3 | 0 | — |

Flux CPU vs PyTorch GPU fails exactly the same three categories as Flux GPU vs
PyTorch GPU. Forward, logits, loss, final parameters, second moments, beta
powers, and the step counter pass in both. There are no zero/nonzero gradient
disagreements in either run.

## C. Flux CPU vs PyTorch CPU (supplementary)

The checker was also run against the new CPU reference. Flux outputs were not
saved as arrays; this log is the comparison. It is supplementary, not an
acceptance input.

| Category | max_abs | violations | max per-leaf rel L2 (worst leaf) |
|---|---:|---:|---|
| eta | 0 | 0 | — |
| activations | 1.430511e-5 | **1** (`act.stage2.block1`) | n/a |
| logits | 2.443790e-6 | 0 | n/a |
| loss | 4.768372e-7 | 0 | n/a |
| grads.u1 | 2.731457e-4 | 42,530 | 2.817987e-3 (`stage1.block1.conv1.weight`) |
| grads.u3 | 6.272923e-4 | 156,257 | 4.578581e-3 (`stage1.block1.conv1.weight`) |
| params.u3 | 4.980713e-6 | 0 | 5.051078e-3 (`stage1.block1.gn1.bias`) |
| m.u3 | 9.655068e-5 | 1,438 | 2.835487e-3 (`stage1.block0.conv1.weight`) |
| v.u3 | 2.253801e-7 | 0 | 3.016001e-3 (`stem.conv.weight`) |
| param_increment.u3 | 4.980713e-6 | 0 | 1.423731e-2 (`stage1.block0.conv1.weight`) |

The single activation violation is one element at the `1e-5` absolute floor on
a near-zero activation. It shows that the declared rule is already tight at
the forward atol boundary; it is not a backward signal.

## Bounded interpretation

1. **Same order?** Yes. The same-framework PyTorch CPU-vs-GPU gradient/moment
   differences are essentially the same size as the Flux GPU-vs-PyTorch GPU
   differences:

   | Category | PyTorch CPU vs GPU max_abs | Flux GPU vs PyTorch GPU max_abs | Ratio (Flux / PyTorch backend) |
   |---|---:|---:|---:|
   | grads.u1 | 3.021732e-4 | 3.025942e-4 | 1.00 |
   | grads.u3 | 4.099458e-4 | 6.672032e-4 | 1.63 |
   | m.u3 | 5.057361e-5 | 6.584823e-5 | 1.30 |

   Per-leaf relative L2 follows the same pattern: `grads.u1` 1.84e-3 vs
   2.00e-3, `grads.u3` 3.64e-3 vs 4.54e-3, `m.u3` 2.30e-3 vs 2.62e-3. The
   worst leaves are the same stage-1 convolution/GroupNorm leaves in every
   independent-backend comparison. The unchanged rule would therefore reject
   a same-framework PyTorch CPU run against its own PyTorch GPU reference.

2. **Does Flux CPU vs PyTorch GPU pass?** The forward path passes: activations
   (0 violations), logits, loss, final parameters, `v`, beta powers, and step.
   It fails only `grads.u1`, `grads.u3`, and `m.u3`, with maxima
   `4.13e-4`, `7.48e-4`, and `7.23e-5`. The Flux CPU residual is therefore not
   removed by leaving the GPU: it is the same order as, and slightly larger
   than, the Flux GPU residual.

3. **Ordinary FP32 variation or Flux defect?** The evidence supports ordinary
   FP32 algorithm/reduction variation. The decisive control is PyTorch CPU vs
   PyTorch GPU with the same generator, fixture, bundle, updates, and
   provenance: it produces 14,365 / 128,170 / 1,137 violations and the same
   worst leaves. Flux CPU vs PyTorch GPU produces the same envelope, and the
   one framework-internal control (Flux CPU vs PyTorch CPU) is also the same
   order. The residual cannot be attributed to the cuDNN FMA policy, because
   the CPU path never uses cuDNN. It is consistent with different valid
   convolution/gradient algorithm families and reduction orders (PyTorch
   oneDNN CPU, cuDNN deterministic GPU, NNlib/Flux CPU, cuDNN FMA GPU). No
   outlier signature is present: no sign flips, no zero/nonzero
   disagreements, forward and second moments pass, and the magnitude is within
   ~1.6x of the same-framework independent-backend envelope.

4. **Not a workload or optimizer defect.** Final parameters (`4.98e-6`), `v`
   (`2.25e-7`), loss, logits, and the step counter pass in every comparison.
   The failure is confined to backward first-order quantities, as expected
   from conv backward algorithm selection.

## Proposed tolerance revision (NOT APPLIED)

The user must approve any revision. The checker, specification, experiment
configuration, and driver still contain `atol=1e-5`, `rtol=1e-4`; no file was
changed. The proposed rule keeps the declared elementwise form and keeps
`rtol=1e-4` for large-magnitude elements. Only the absolute floor changes, and
only for the three failing backward categories.

Observed independent-backend envelope (maximum `max_abs` over PyTorch
CPU-vs-GPU, Flux CPU-vs-PyTorch GPU, Flux CPU-vs-PyTorch CPU, and Flux
GPU-vs-PyTorch GPU):

| Category | Envelope max_abs | Source of maximum |
|---|---:|---|
| grads.u1 | 4.126653e-4 | Flux CPU vs PyTorch GPU |
| grads.u3 | 7.480253e-4 | Flux CPU vs PyTorch GPU |
| m.u3 | 9.655068e-5 | Flux CPU vs PyTorch CPU |

Smallest proposed category-specific floors with at least a 2x margin:

| Category | Proposed rule | Margin over envelope | What it still covers |
|---|---|---:|---|
| `grads.u1` | `atol=1e-3`, `rtol=1e-4` | 2.42x | large components still at 1e-4 relative |
| `grads.u3` | `atol=2e-3`, `rtol=1e-4` | 2.67x | large components still at 1e-4 relative |
| `m.u3` | `atol=2e-4`, `rtol=1e-4` | 2.07x | large moments still at 1e-4 relative |
| all other categories | `atol=1e-5`, `rtol=1e-4` | unchanged | forward, loss, params, v, beta, step |

Because each proposed floor exceeds the observed `max_abs` in its category,
these rules are guaranteed to produce zero violations on all four comparisons
above. As a sanity check, the loosest floor (`2e-3`) run against the
PyTorch CPU-vs-GPU reference pair gives `COMPARE_RESULT PASS` with 0
violations (`torch-cpu-vs-gpu-reference.atol2e-3.log`).

What remains covered: forward activations, logits, loss, final parameters,
second moments, beta powers, and the step counter stay at the original strict
rule. Gradient and first-moment elements with large magnitude are still held
to `1e-4` relative. Zero/nonzero disagreement detection is unchanged. What is
not covered: a backward defect confined to low-magnitude elements below the
new floors (absolute error `<1e-3` for `grads.u1`, `<2e-3` for `grads.u3`,
`<2e-4` for `m.u3`) would not be caught by the elementwise rule. The
supplementary per-leaf relative L2 remains the diagnostic for that regime; the
observed worst values are 4.58e-3 (gradients) and 2.84e-3 (`m`), so a
per-leaf ceiling of 1e-2 would be a compatible second guard if the user wants
one. This report does not propose changing the checker beyond the three
category floors.

## Recommendation and exact next action

The evidence supports a numerical tolerance revision over pinning algorithms:
the residual is reproduced by a same-framework CPU/GPU backend change, so
matching a specific cuDNN algorithm family is not required for correctness.
If the user approves, update OD-10, `experiments/static-scaling/specification.md`,
`experiment.toml`, and the checker consistently with the three category rules
above, record the approval, then run the full driver in a fresh evidence root
and require all four summaries to PASS. Do not run handoff Steps 4–6 before
that decision. If the user rejects the revision, the next bounded diagnostic is
to force a deterministic/direct backward algorithm family (for example algo 0
for backward-filter/backward-data) and rerun the full comparison without
changing tolerances.

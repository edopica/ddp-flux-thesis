# Phase 1 review response: R1-R4 corrected

Date: 2026-10-16
Review: [Phase 1 implementation review](../../../reports/static-scaling-phase1-review.md) (changes required)
Scope: `experiments/static-scaling/specification.md` and `experiments/static-scaling/experiment.toml`
Status: all four findings corrected; focused acceptance probes pass (exit 0)

## Corrections

| Finding | Correction | Locations |
|---|---|---|
| R1 classifier weight marked for transpose | Classifier conversion is now "no transpose": `Flux.Dense` stores `(out, in)` and the classifier is written `Dense(512 => 200)` with weight shape `(200, 512)`; gradients and moments keep the same axes | specification.md sections 4 and 8; experiment.toml `shared_format.flux_conversion` |
| R2 convolution orientation unspecified | The kernel rule is now explicit: canonical `(out, in, kh, kw)` -> `permutedims(w, (4, 3, 2, 1))` -> `(kw, kh, in, out)` used with `Flux.CrossCor` (cross-correlation, reproduces PyTorch `Conv2d`); `Flux.Conv` is not used (it would need both spatial axes reversed); the inverse export uses the same self-inverse permutation; gradients and moments follow their array layout with no reversal | specification.md sections 4 and 8; experiment.toml `model.conv_algorithm`, `shared_format.canonical_layout`, `shared_format.flux_conversion` |
| R3 unsupported AdamW constructor | The documented constructor is the all-keyword form `Optimisers.AdamW(; eta, beta, lambda, epsilon, couple = true)`; the text notes the positional constructor accepts only `couple` as a keyword | specification.md section 6.1; experiment.toml `optimizer.flux_config` |
| R4 beta-power state mapping off by one | The shared counter is now the int64 number of completed updates `s`; PyTorch `state['step'] == s`, compared exactly. Flux stores the beta-power pair: `beta` before the first update, `beta^(s+1)` after `s` completed updates; verification uses float tolerance against `beta .^ (s + 1)`; continuation reconstructs the pair by starting from `beta` and multiplying by `beta` exactly `s` times in float32 (the bit-exact route) | specification.md sections 8 and 11; experiment.toml `shared_format.optimizer_state_mapping` |

The status header of `specification.md` and `experiment.toml` record the review and the correction date.

## Acceptance evidence

Two probe scripts were added and run against the pinned project
(`--project=/home/kurapica/Projects/ddp_flux/flux-integration`, Optimisers 0.4.9, NNlib 0.9.45, Julia 1.12.6):

```bash
julia --startup-file=no --project=/home/kurapica/Projects/ddp_flux/flux-integration experiments/static-scaling/probes/probe_tensor_layout.jl
julia --startup-file=no --project=/home/kurapica/Projects/ddp_flux/flux-integration experiments/static-scaling/probes/probe_optimizer_contract.jl
```

Both commands exit 0. Outputs (logs: `artifacts/logs/static-scaling/2026-10-16-review-response/`):

```text
R1 dense_weight=(200, 512) grad_shape=(200, 512) no_transpose=true
R2 crosscor_output=Float32[178.0, 283.0, 388.0, 493.0, 199.0, 304.0, 409.0, 514.0, 220.0, 325.0, 430.0, 535.0, 241.0, 346.0, 451.0, 556.0]
R2 pytorch_semantics_reference=Float32[178.0, 283.0, 388.0, 493.0, 199.0, 304.0, 409.0, 514.0, 220.0, 325.0, 430.0, 535.0, 241.0, 346.0, 451.0, 556.0]
R2 crosscor_matches=true plain_conv_differs=true reversed_conv_matches=true grad_shape=(3, 2, 1, 1)
```

```text
R3 decay_rule=AdamW(eta=0.001, beta=(0.9, 0.999), lambda=0.0001, epsilon=1.0e-8, couple=true)
R3 no_decay_rule=AdamW(eta=0.001, beta=(0.9, 0.999), lambda=0.0, epsilon=1.0e-8, couple=true)
R3 mixed_constructor=MethodError
R4 s=0 beta_power=(0.9f0, 0.999f0)
R4 s=1 beta_power=(0.80999994f0, 0.99800104f0) s=2 beta_power=(0.7289999f0, 0.9970031f0)
R4 stored=beta^(s+1)=true reconstruction_repeated_multiplication_exact=true continuation_from_exchanged_state_exact=true
```

The R2 probe compares a Flux `CrossCor` forward pass against a manual
cross-correlation loop written in canonical PyTorch NCHW semantics, with an
asymmetric 2x3 kernel and a 5x6 input; the outputs match bitwise on this
fixture. It also confirms that plain `Conv` with the same permuted kernel
differs, and that `Conv` with both spatial axes reversed matches. The R1 probe
checks the classifier axes, the `W * x + b` forward identity, and the `(200,
512)` gradient shape.

No PyTorch runtime exists on the local machine (`ModuleNotFoundError: torch`),
so cross-framework numerical fixtures remain Phase 2/3 work under the frozen
format; the manual reference loop is framework-independent evidence of the
operator semantics.

## Probe development notes

Four first attempts failed and were corrected in-session; none affects the
corrected files:

1. `isapprox(Vector, Tuple)` is not defined; compare `collect.(...)` values.
2. A top-level `for` loop with `bt = bt .* b` raised `UndefVarError` under
   `julia -e` scope rules; probes now use `reduce`.
3. The first Dense probe asserted `Dense(200, 512)` has weight `(200, 512)`.
   Flux `Dense(in, out)` built weight `(512, 200)`; this caught and corrected a
   matching wording error in my own specification edit (`Dense(512 => 200)` is
   the classifier). This is the exact axis confusion of finding R1.
4. `Zygote.gradient` cannot differentiate through layer construction inside the
   closure (`Mutating arrays is not supported`); pre-construct the layer and
   differentiate `m -> sum(m(x))` instead. Recorded for Phase 2 fixture code.

## Independent verification

A fresh verification subagent checked the corrections adversarially:
R1-R4 PASS with file:line evidence; cross-file consistency PASS; both probe
scripts re-ran with exit 0 and a byte-identical diff against the saved logs.
An independent reference implementation covered seven extra kernel cases
(asymmetric shapes 3x1, 1x3, 2x5, 3x2, 5x2, padding 1, stride 2, multi-channel
and multi-batch): `CrossCor` matched 7/7, plain `Conv` differed in all cases,
reversed-kernel `Conv` matched all cases, and the export permutation is
involutive. R1 with a rectangular `Dense(7 => 3)` and R4 up to `s = 10` were
also exact; the power comparison showed 1-ulp differences against `beta .^
(s + 1)` at larger `s`, covered by the documented float tolerance. After the
verification, two wording items were tightened: the reconstruction route is
stated as "start from `beta`, multiply by `beta` `s` times" (the bit-exact
route), and the stale README "changes required" banner was replaced by the
review-response banner. No counterexample was found; residual risk is the
absence of a local PyTorch runtime, as recorded above.

## Closure review — 2026-10-16

The reviewer inspected both corrected contracts and both probe scripts.
The reviewer reran the two exact commands in the acceptance evidence section.
Both commands exited 0, and their printed outputs match the saved logs.
R1–R4 are closed in the [review report](../../../reports/static-scaling-phase1-review.md).
Actual PyTorch numerical comparisons remain Phase 2/3 work. The local tensor probe establishes forward semantics and gradient shapes.
Documentation whitespace check: `git diff --check`.

## Unchanged scope

- Open decisions OD-1..OD-13 and all proposed values are untouched.
- No GPU, distributed, or training run; no remote command; no Flux change; no
  commit.

## Exact next action

Generate the Phase 2 shared fixtures in the corrected `.npz` plus manifest
format: decoded input batches, labels, initial parameter bundles, gradients,
updated parameters, and optimizer state with the completed-update counter.
Include the rectangular-linear and asymmetric-kernel fixtures this review
requires.

# Phase 1 implementation review: static-network scaling

Date: 2026-10-16
Initial verdict: **changes required** before Phase 2 fixture implementation.
Closure verdict (2026-10-16): **R1–R4 closed**. The experiment decisions remain pending.

## Closure review — 2026-10-16

The corrected specification and TOML configuration resolve all four findings.
Classifier axes remain unchanged. Convolution conversion explicitly selects `Flux.CrossCor`.
The AdamW constructor uses keywords, and the state mapping distinguishes completed updates from the next-update beta powers.

Both probe scripts were inspected and rerun locally with the commands in the response devlog.
Both exited 0. Their printed outputs match the saved logs.

The evidence closes the specification defects. Actual PyTorch forward/backward comparisons remain Phase 2/3 acceptance work.
The tensor probe checks forward semantics and gradient shapes, not numerical cross-framework gradient equality.

Evidence: [response devlog](../wiki/devlog/03-static-scaling/2026-10-16-phase1-review-response.md).

**Exact next action:** record OD-1, OD-5, and OD-6 before generating Phase 2 fixtures.

The original findings follow for traceability.

## Scope

This review covers the specification, configuration, and decision table in `experiments/static-scaling/`.
The review compares their contracts against the implementation plan and local pinned package sources.
The Flux source is at `f7b4520c4bd307c67bbc5ac92a1c68493eccff6e`.
Local numerical probes used Optimisers 0.4.9 and NNlib 0.9.45.

The three planned deliverables exist. The open decisions distinguish proposed values from accepted choices.
However, four concrete defects prevent the documents from defining equivalent implementations.
These defects are separate from the pending supervisor decisions.

## R1 — P1: Do not transpose the classifier weight for Flux

**Locations:** `specification.md:236-237`, `experiment.toml:168-169`.

The canonical classifier weight has shape `(out, in)`, or `(200, 512)`.
The documents require a transpose for Flux, which produces `(512, 200)`.
Flux `Dense` already stores weights as `(out, in)` and computes `weight * input`.
The proposed conversion therefore fails the classifier dimension contract.

Source evidence: `../flux-integration/src/layers/basic.jl:189-199` constructs `Dense(init(out, in), ...)` and multiplies by the input.

**Required correction:** preserve the classifier weight axes during import and export.
Apply the same axis rule to its gradients and optimizer moments.
Distinguish the activation transpose from parameter conversion.

**Acceptance:** compare a rectangular linear layer across frameworks, including forward output and parameter gradients.
Assert classifier weight shape `(200, 512)` in both models.

## R2 — P1: Specify convolution kernel orientation

**Locations:** `specification.md:236-237`, `experiment.toml:168-169`.

The proposed kernel permutation changes axis order but does not reverse spatial kernel coordinates.
PyTorch `Conv2d` performs cross-correlation. Flux `Conv` uses the NNlib convolution convention.
Thus, a normal Flux `Conv` with the proposed converted weights does not reproduce the PyTorch operator.
The mismatch affects asymmetric kernels from the shared initialization bundle.

Source evidence:

- `../flux-integration/src/layers/conv.jl:204-218`: `Conv` uses the default `DenseConvDims` and `conv`.
- `../flux-integration/src/layers/conv.jl:499-501`: `CrossCor` selects `F=true` before calling `conv`.
- Local NNlib probe: default convolution gives `[23, 33, 53, 63]`; cross-correlation gives `[37, 47, 67, 77]`.
- Reversing both spatial kernel axes makes default convolution match cross-correlation exactly in this probe.

**Required correction:** explicitly choose Flux `CrossCor` with axis permutation, or Flux `Conv` with permutation and spatial reversal.
Specify inverse export rules and corresponding transformations for gradients and optimizer moments.

**Acceptance:** use an asymmetric kernel and spatially asymmetric input in a cross-framework forward/backward fixture.
Shape checks and parameter counts alone cannot expose this defect.

## R3 — P2: Use a supported AdamW constructor

**Locations:** `specification.md:170`, `experiment.toml:114`.

The documents use `Optimisers.AdamW(eta; beta, lambda, epsilon)`.
Optimisers 0.4.9 accepts only `couple` as a keyword with the positional constructor.
The other named arguments belong to its all-keyword constructor.
The documented call raises `MethodError`, before optimizer setup.

Source evidence: `Optimisers/tMaaf/src/rules.jl:537-543`.
The local package probe reproduced `mixed_constructor=MethodError` with Optimisers 0.4.9.

**Required correction:** use the all-keyword form:

```julia
Optimisers.AdamW(; eta=eta, beta=(0.9, 0.999), lambda=wd,
                epsilon=1e-8, couple=true)
```

**Acceptance:** construct both decay and no-decay rules with the documented syntax.
The AdamW equations and the `couple=true` choice are otherwise consistent with the intended learning-rate-scaled decay.

## R4 — P2: Correct the AdamW beta-power state mapping

**Locations:** `specification.md:240-241`, `experiment.toml:171`.

The contract requires Flux's stored beta-power state to equal `beta .^ step`, with the same completed-step count as PyTorch.
Optimisers initializes this state to `beta`, before any update.
Each update uses the stored power for bias correction, then multiplies it by `beta` for the next update.
After `s` completed updates, the stored state therefore represents `beta^(s+1)`, not `beta^s`.

Source evidence: `Optimisers/tMaaf/src/rules.jl:545-563`.
The local probe produced:

```text
step0_beta_power=(0.9f0, 0.999f0)
step1_beta_power=(0.80999994f0, 0.99800104f0)
```

The current mapping rejects valid initial and updated states.
Using it to reconstruct state also shifts bias correction on continuation.

**Required correction:** define the shared counter as completed optimizer updates.
Document the next-update exponent in Flux's stored state.
Account for Float32 repeated multiplication rather than requiring bitwise equality with a freshly computed power.

**Acceptance:** compare state before the first update and after multiple updates.
Include a continuation check from the exchanged state.
Compare integer counters exactly, separately from floating-point tolerances.

## Verification and limits

Two focused Julia commands passed. They covered convolution orientation, constructor rejection, and optimizer-state evolution.
The Dense finding follows directly from the pinned source's allocation and multiplication rules.
The first package-directory probe failed because its standalone environment lacked resolved dependencies.
The retry used the existing Flux project and loaded Optimisers 0.4.9 successfully.

Commands and outputs: [review devlog](../wiki/devlog/03-static-scaling/2026-10-16-phase1-review.md).
Environment failure: [failure record](../wiki/failures/2026-10-16-phase1-review-package-environment.md).
This review ran no GPU, distributed, or training campaign.

## Exact next action

Correct R1–R4 in both the readable specification and `experiment.toml`.
Then run the corresponding focused acceptance checks before generating Phase 2 fixtures.

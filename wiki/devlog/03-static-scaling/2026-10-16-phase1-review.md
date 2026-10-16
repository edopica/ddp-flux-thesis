# Phase 1 shared specification review

Date: 2026-10-16
Verdict: changes required

## Findings

The [review report](../../../reports/static-scaling-phase1-review.md) records four defects:

1. P1: the classifier weight transpose contradicts Flux Dense's `(out, in)` layout.
2. P1: convolution conversion lacks an explicit kernel-orientation rule.
3. P2: the mixed positional/keyword AdamW constructor raises `MethodError` in Optimisers 0.4.9.
4. P2: the stored beta-power mapping is off by one relative to completed optimizer updates.

The specification and implementation sources remain unchanged by this review.
The review updates the project status to changes required before fixture implementation.

## Commands and evidence

Repository inspection: `git status --short`.
Flux identity: `git rev-parse HEAD`, with working directory `/home/kurapica/Projects/ddp_flux/flux-integration`.
Result: `f7b4520c4bd307c67bbc5ac92a1c68493eccff6e`.

### Optimizer probe

```bash
julia --startup-file=no --project=/home/kurapica/Projects/ddp_flux/flux-integration -e 'using Optimisers; println("version=", pkgversion(Optimisers)); try Optimisers.AdamW(1e-3; beta=(0.9,0.999), lambda=1e-4, epsilon=1e-8); println("mixed_constructor=accepted") catch e; println("mixed_constructor=", typeof(e)); end; rule=Optimisers.AdamW(; eta=1e-3, beta=(0.9,0.999), lambda=1e-4, epsilon=1e-8); x=Float32[1,2]; s=Optimisers.init(rule,x); println("step0_beta_power=",s[3]); s,_=Optimisers.apply!(rule,s,x,Float32[0.1,0.2]); println("step1_beta_power=",s[3]); println("expected_stored_power=beta^(completed_steps+1)")'
```

Exit 0:

```text
version=0.4.9
mixed_constructor=MethodError
step0_beta_power=(0.9f0, 0.999f0)
step1_beta_power=(0.80999994f0, 0.99800104f0)
expected_stored_power=beta^(completed_steps+1)
```

### Convolution probe

```bash
julia --startup-file=no --project=/home/kurapica/Projects/ddp_flux/flux-integration -e 'using NNlib; x=reshape(Float32.(1:9),3,3,1,1); w=reshape(Float32[1,2,3,4],2,2,1,1); d=NNlib.DenseConvDims(x,w); conv_y=NNlib.conv(x,w,d); corr_y=NNlib.conv(x,w,NNlib.DenseConvDims(d; F=true)); flipped_y=NNlib.conv(x,reverse(w; dims=(1,2)),d); println("nnlib_version=",pkgversion(NNlib)); println("conv_default=",vec(conv_y)); println("cross_correlation=",vec(corr_y)); println("reversed_kernel_matches=", flipped_y == corr_y); @assert conv_y != corr_y; @assert flipped_y == corr_y'
```

Exit 0:

```text
nnlib_version=0.9.45
conv_default=Float32[23.0, 33.0, 53.0, 63.0]
cross_correlation=Float32[37.0, 47.0, 67.0, 77.0]
reversed_kernel_matches=true
```

The first optimizer probe selected the package directory as its project and failed before loading Optimisers.
The [failure record](../../failures/2026-10-16-phase1-review-package-environment.md) preserves the command and cause.
No remote execution or training occurred.

Documentation whitespace check: `git diff --check`.

**Exact next action:** correct review findings R1–R4 in the specification and configuration before generating Phase 2 fixtures.

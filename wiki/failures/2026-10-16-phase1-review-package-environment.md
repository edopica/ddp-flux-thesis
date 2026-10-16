# Phase 1 review: standalone package environment failure

Date: 2026-10-16
Scope: local review probe, before optimizer execution

## Failed command

```bash
julia --startup-file=no --project=/home/kurapica/.julia/packages/Optimisers/tMaaf -e 'using Optimisers; println("version=", pkgversion(Optimisers)); try Optimisers.AdamW(1e-3; beta=(0.9,0.999), lambda=1e-4, epsilon=1e-8); println("mixed_constructor=accepted") catch e; println("mixed_constructor=", typeof(e)); end; rule=Optimisers.AdamW(; eta=1e-3, beta=(0.9,0.999), lambda=1e-4, epsilon=1e-8); x=Float32[1,2]; s=Optimisers.init(rule,x); println("step0_beta_power=",s[3]); s,_=Optimisers.apply!(rule,s,x,Float32[0.1,0.2]); println("step1_beta_power=",s[3]); println("expected_stored_power=beta^(completed_steps+1)")'
```

## Cause and resolution

The package directory did not provide a resolved environment for its dependencies.
Precompilation stopped with:

```text
ArgumentError: Package Functors [d9f16b24-f501-4c13-a1f2-28368ffc5196] is required but does not seem to be installed
```

The retry selected the existing `/home/kurapica/Projects/ddp_flux/flux-integration` project.
It loaded Optimisers 0.4.9 and completed the probe with exit 0.
No package installation or dependency change was necessary.
The failure does not establish a product defect.

Successful command and results: [review devlog](../devlog/03-static-scaling/2026-10-16-phase1-review.md).

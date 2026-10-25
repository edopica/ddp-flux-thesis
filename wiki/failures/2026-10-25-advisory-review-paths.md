# Advisory review path lookups — 2026-10-25

A local read used `cat wiki/experiments/static-scaling/OD-14-benchmark-acceptance.md`
and searched `wiki/experiments/static-scaling` and
`scripts/static_scaling/flux*`; these paths do not exist and the shell call
returned 2. The correct decision is
`experiments/static-scaling/OD-14-benchmark-acceptance.md`; the memory probes
are `scripts/static_scaling/memory_flux.jl` and `memory_torch.py`.
Located them with `rg --files experiments scripts/static_scaling` and read
the corrected paths successfully. No training or artifact modification occurred
as a result of the failed lookups.

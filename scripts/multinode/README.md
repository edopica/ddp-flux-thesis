# Base CPU DDP acceptance

This path is independent of historical CG and GPU environments. No Flux source is
modified. The CPU c8 make targets now invoke one parent: the upstream driver owns
MPI launch and receives JULIA_MPI_TEST_NPROCS (F13). The standalone NCCL driver
still references CUDA without importing it (F14); that path is deferred.

From the thesis root:

```sh
python3 scripts/multinode/test_launcher.py
julia --startup-file=no --project=scripts/multinode/env -e 'using Pkg; Pkg.develop(path="../flux-multinode"); Pkg.instantiate(); Pkg.precompile()'
julia --startup-file=no --project=scripts/multinode/env -e 'using Flux, MPI, Optimisers, Functors, JSON, MLUtils'
julia --startup-file=no --project=scripts/multinode/env scripts/multinode/run_cpu.jl
```

`run_cpu.jl` is a single local parent using MPI.jl's selected launcher. It runs
the five upstream primitive workers without resolving the unrelated full Flux
test dependencies. Workers use `--compiled-modules=existing` after explicit precompile to avoid
cache-write churn in MPI children. Each suite worker has a 900-second deadline; acceptance has
300 seconds, followed by TERM and KILL after 15 seconds. Do not launch this parent
under mpiexec/srun. `acceptance.jl` is a direct worker for later scheduler use.

The nested Float64 fixture uses six updates for Descent and Adam. Local mean
gradients are averaged exactly once by native DistributedOptimizer, outside AD.
The ordinary reference differentiates the concatenated cyclically padded samples,
including partial batches. Bounds are fixed at atol=1e-10, rtol=1e-9. It compares
local and reduced gradients, parameters, Adam moments and beta powers, and checks
a missing-world-size-divisor negative control. Rank 0 writes JSON evidence.

This initial harness does not yet attest the native optimizer's actual collective
trace (only its diagnostic reduction trace), source tree hashes/preferences,
or process cleanup automatically. Stage 1 must not be
called fully accepted until those audit requirements have evidence. Multi-node
CPU and all GPU gates remain unverified. Remote work requires the remote skill
and matching server context.

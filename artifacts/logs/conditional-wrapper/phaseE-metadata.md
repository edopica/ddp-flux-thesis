# Phase E metadata: restore and improve the M1.1 test battery

Date: 2026-09-30
Host: omarchy (local, 14 GiB)
Flux worktree: `/home/kurapica/Projects/ddp_flux/flux-integration`
Flux branch: `ddp/integration-m1-spike`
Flux commit at test time: `d6046103` (Phase D complete, clean)
Optimisers: 0.4.9 (`/home/kurapica/.julia/packages/Optimisers/tMaaf/src/Optimisers.jl`)
MPI launcher: `/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec` (MPICH/Hydra 5.0.1)
Reference for the ported battery: `3370b910:test/ext_distributed/conditional_distributedtest.jl`
Scope: tests only. No production source change.
Thesis working tree: dirty (wiki, devlog, README, evidence logs).

## Change

`test/ext_distributed/conditional_distributedtest.jl`: +560 lines, 0 deletions
at the final revised state (the first submitted version was +411; the review
correction appended at the end added +149). No other file changed
(`git diff --stat`: 1 file).

Phase D had already replaced the following M1.1 testsets with stronger testsets
that are not duplicated here:

- BatchNorm exclusion (D5, real AD, 2 steps),
- transpose/adjoint parent space (D5, real AD),
- caller-gradient non-mutation (D5, rank-distinct + read-only),
- `update!` immutable returns (D5),
- globally absent Adam state (D5, active-then-absent snapshots).

Phase E restores the M1.1 cases with no Phase D equivalent and adds the two
explicit Phase E items:

| # | Testset | New assertions per rank |
|---|---|---|
| 1 | `real AD conditional branches average over multiple steps` | 11 |
| 2 | `Adam matches an independent global-mean reference` | 9 |
| 3 | `tied parameters through real AD` (incl. M1.1 shared-gradient subcase) | 13 |
| 4 | `tied parameter through a transpose with real AD` | 5 |
| 5 | `isbits parameters (FillArrays) sync independently` | 5 |
| 6 | `higher-order gradients rejected` | 2 |
| 7 | `collective count protocol with real AD` | 4 |
| 8 | `Flux.train! with DistributedOptimizerState` | 3 |
| 9 | `nested model (Chain of custom layer and Dense)` | 2 |
| 10 | `synchronize!! on DistributedOptimizerState` | 3 |
| 11 | `adjust / adjust! on wrapped state and rule` | 11 |
| 12 | `freeze! and thaw! preserve wrapper state and collectives` (Phase E, revised) | 31 |
| 13 | `Functor device adaptation keeps the backend` (Phase E, revised) | 26 |
| 14 | `Chain tuple-gradient diagnostic` (M1.1 API restore) | 5 |
| 15 | `Enzyme Duplicated model setup and update` (M1.1 API restore) | 5 |
| | **New total** | **135** |

Fixtures added: `ScaleLayer` (M1.1), `AdaptedArray`, `AdaptedBackend` (Phase E
device adaptation). No mock or simulated value is used: the device adaptation
test delegates reductions to the real backend on the raw storage, as the NCCL
backend does for CPU buffers. The revised device-adaptation test uses Adam and
transforms exactly the two nonzero state arrays (`mt`, `vt`), asserts their
types and preserved values, keeps the source state plain, then updates through
the adapted state and matches a native Adam reference on parameters and full
state. The revised freeze test uses Adam, retains each returned state, checks
whole-wrapper `freeze!`/`thaw!` delegation, partial freeze through the inner
tree, frozen-moment preservation, collective counts, and a native reference.
`import Enzyme` is needed for `Enzyme.make_zero`; the same test file already
depends on `Flux.EnzymeCore` dispatch.

## Commands and results

### Focused conditional file, two MPI ranks

```bash
cd /home/kurapica/Projects/ddp_flux/flux-integration
time FLUX_TEST_DISTRIBUTED_BACKEND=mpi \
  /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec \
  -n 2 julia --startup-file=no --project=test \
  test/ext_distributed/conditional_distributedtest.jl
```

- Exit 0, 1 m 13.5 s (warm; one local Flux/FluxMPIExt compile happened earlier
  in the session).
- Per-testset GREEN (from `phaseE-GREEN-mpi2-attempt6-2026-09-30.log`):
  control 5, D1 10, isbits guard 11, D2 38, D3 16, D4 20, D4-review 30,
  BatchNorm 6, transpose/adjoint 16, caller gradients 11, `update!`
  immutable 9, `update` fresh 15, absent Adam 16, empty model 9, array
  containers 20, and the 13 Phase E testsets above (83) = **315 per rank**.
- Clean aggregate total: `conditional file | 315 315 | 1m19.4s`, per rank,
  exit 0 (`phaseE-TOTALS-mpi2-2026-09-30.log`).

Iterations before GREEN (all failures were test defects, no production bugs):

1. `attempt1`: partial freeze, 7/11. `m1.frozen` was compared with the
   in-place-updated `model.frozen`; the test's own arithmetic was fixed with a
   snapshot.
2. `attempt2`: partial freeze, 7/11. Step 2 started from the original `model`
   instead of the updated `m1`, so the expected values were off by one update.
3. `attempt3`: device adaptation, 4 pass/1 error. `_avg_sum` is length-4 only;
   the test now computes the mean inline.
4. `attempt4`: device adaptation, 4 pass/1 error. `AdaptedArray` is not an MPI
   datatype (`strides` missing); added `AdaptedBackend`, which reduces the raw
   storage (same pattern as the D3 device-like backend).
5. `attempt5`: device adaptation, 6 pass/1 fail/1 error. The inner update
   returns a plain `Vector` for this parameter, so the test now asserts the
   `AbstractArray` contract and values, not the container type.

### Focused optimizer file, two MPI ranks

```bash
time FLUX_TEST_DISTRIBUTED_BACKEND=mpi <mpiexec> -n 2 \
  julia --startup-file=no --project=test \
  test/ext_distributed/optimizer_distributedtest.jl
```

- Exit 0, 10.9 s. Composition guard 6/6 per rank.
- Log: `phaseE-GREEN-optimizer-mpi2-2026-09-30.log`.

### Complete dedicated suite, two ranks

```bash
time JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test \
  test/ext_distributed/runtests.jl mpi
```

- Exit 0, 1 m 55.8 s; `Distributed | 6 6 1m52.7s`.
- Log: `phaseE-GREEN-suite-nprocs2-2026-09-30.log`.

### Static checks

```bash
julia --startup-file=no --project=test -e \
  'using Flux, Test, Optimisers; a = Test.detect_ambiguities(Flux.DistributedUtils; recursive=false); @assert isempty(a)'
git diff --check
```

- Optimisers 0.4.9, ambiguities 0
  (`phaseE-ambiguities-2026-09-30.log`).
- `git diff --check`: clean, exit 0
  (`phaseE-diff-check-2026-09-30.log`).

## Diagnostics

Tied real-AD gradients at two ranks: rank 0 and rank 1 both return
`gs.a = [0.0, 2.0]`, `gs.b = [-2.0, 0.0]` (distinct objects per occurrence).
The ported test therefore proves the wrapper's local tied sum and the native
Optimisers reference agree; rank-distinct averaging for tied parameters is
covered by the D2 mixed tied/isbits testset with synthetic gradients.

## Limits

- Two ranks only. Four-rank weighting remains Phase F.
- No four-rank run, no GPU/NCCL run, no detector update, no HPC run, no main
  suite run.
- The device adaptation test uses the real backend through a delegating
  wrapper; it does not prove native CUDA/NCCL behavior.
- No commit: the Phase E test change is uncommitted; commit/push only with
  explicit user authorization.

## Evidence files

`artifacts/logs/conditional-wrapper/`:

- `phaseE-GREEN-mpi2-2026-09-30.log` (focused conditional, per-testset),
- `phaseE-TOTALS-mpi2-2026-09-30.log` (focused conditional, aggregate totals),
- `phaseE-GREEN-optimizer-mpi2-2026-09-30.log`,
- `phaseE-GREEN-suite-nprocs2-2026-09-30.log`,
- `phaseE-ambiguities-2026-09-30.log`,
- `phaseE-diff-check-2026-09-30.log`,
- `phaseE-GREEN-mpi2-attempt1..5-*.log` (iterations, test-defect fixes).

## Review correction (2026-09-30)

The independent review (devlog section "Independent review") reopened Phase E
for two P2 coverage gaps. No production defect was established; Flux source
was not touched. The corrections below were made and the two-rank gate was
repeated.

### Correction 1: device adaptation must transform state arrays

`Descent` stores `nothing`, so the first version transformed zero arrays and
then updated a freshly constructed state. The replacement uses `Adam`, warms
`mt`/`vt` with one distributed step, adapts both state and model through
`Functors.fmap`, and asserts:

- exactly 2 arrays transformed (`mt`, `vt`), both `AdaptedArray`;
- transformed values equal the source values;
- `adapted.backend === state.backend === rec` (backend identity);
- the source state keeps `Vector{Float64}` arrays with unchanged values;
- the update runs through the adapted state and matches a native Adam
  reference on parameters, moments, and the decayed-beta tuple;
- the gradient collective saw `AdaptedArray{Float64,1}` (delegated to raw
  storage by the test backend).

`Base.convert(::Type{AdaptedArray{T,N}}, v::AbstractArray)` was added so the
Adam state, whose `Leaf.state` type is fixed after adaptation, can stay
adapted across updates (same pattern as the `IsoVec` fixture).

### Correction 2: restore the missing M1.1 API checks

- The tied-parameter testset regains the M1.1 subcase that passes one gradient
  object for both occurrences and compares against the native reference.
- `Chain tuple-gradient diagnostic` restored: the `grads[1]` warning and
  correct step for both `update!` and `update`.
- `Enzyme Duplicated model setup and update` restored: supported setup,
  `Flux.setup`, and rejection of both update paths (`import Enzyme` added for
  `Enzyme.make_zero`; the dispatch is on `EnzymeCore.Duplicated`).
- `freeze! and thaw! preserve wrapper state and collectives` replaces the
  previous partial-freeze test: Adam, retained returned states, whole-wrapper
  `freeze!`/`thaw!` delegation (`=== state`), partial freeze through the inner
  tree, frozen value and moment preservation, unchanged collective counts, and
  a native reference that freezes and thaws at the same steps.

### Correction gate results

```bash
cd /home/kurapica/Projects/ddp_flux/flux-integration
time FLUX_TEST_DISTRIBUTED_BACKEND=mpi <mpiexec> -n 2 julia --startup-file=no \
  --project=test -e 'using Test; @testset "conditional file" begin
  include("test/ext_distributed/conditional_distributedtest.jl") end'
```

- Exit 0, 1 m 22.0 s; `conditional file | 367 367 | 1m22.0s` per rank.
- Focused optimizer: exit 0, 10.3 s, composition guard 6/6 per rank.
- Complete dedicated suite: exit 0, 2 m 4.8 s, `Distributed | 6 6 2m02.6s`.
- Ambiguities: 0. `git diff --check`: clean.
- All correction runs passed on the first attempt; the previous RED iterations
  were the test defects recorded above and are not repeated.

### Correction evidence files

- `phaseE-correction-GREEN-mpi2-2026-09-30.log` (focused conditional,
  per-testset),
- `phaseE-correction-TOTALS-mpi2-2026-09-30.log` (aggregate totals),
- `phaseE-correction-GREEN-optimizer-mpi2-2026-09-30.log`,
- `phaseE-correction-GREEN-suite-nprocs2-2026-09-30.log`,
- `phaseE-correction-ambiguities-2026-09-30.log`,
- `phaseE-correction-diff-check-2026-09-30.log`.

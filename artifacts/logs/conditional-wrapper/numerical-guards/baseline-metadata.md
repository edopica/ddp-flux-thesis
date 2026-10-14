# DB-1/DB-4 numerical-guard baseline metadata

Recorded: 2026-10-13, before any source edit (plan Phase A.1).

## Flux worktree identity

| Item | Value |
|---|---|
| Repository | `../flux-integration` |
| HEAD | `55171e659b2f425da980cbc83177bf5c38f0a7a1` |
| Branch | `ddp/integration-m1-spike` |
| Dirty diff md5 | `30fe313e8ae29df18b55c607ac012b6c` |
| Dirty diff stat | 6 files changed, 278 insertions(+), 7 deletions(-) |
| `src/train.jl` md5 | `eea4e545d3eb2118c3a5b23653c7a71a` |
| `src/distributed/public_api.jl` md5 | `9bfa981384eaf88750ec3d19e8bb76bd` |

Dirty files (pre-existing, not part of this fix):

```
 M NEWS.md
 M src/distributed/public_api.jl
 M test/Project.toml
 M test/ext_distributed/distributed_setup.jl
 M test/ext_distributed/runtests.jl
 M test/ext_distributed/synchronized_distributedtest.jl
?? test/ext_distributed/cuda_conditional_distributedtest.jl
```

The pre-existing `src/distributed/public_api.jl` diff contains the conditional-wrapper
occurrence-plan work (`_SyncOccurrence`, `_buildmap` shape fix); the DB-1/DB-4 change adds
to this dirty state. No commit was made.

## Thesis workspace identity

| Item | Value |
|---|---|
| HEAD | `95e8b852f244a26667524b9a370c41f061bbc4f1` |
| Working tree | dirty (untracked reports, artifacts, devlog) |

## Local toolchain

| Item | Value |
|---|---|
| Julia | 1.12.6 |
| MPI launcher | MPICH Hydra 5.0.1 (`/opt/mpich/bin/mpiexec`) |
| Flux | 0.16.11 |
| Optimisers | 0.4.9 |
| MPI.jl | 0.20.27 |
| Project | `../flux-integration/test` |

## Remote toolchain (deathstar)

Recorded by the Phase B run metadata (`numerical-guards/metadata.md`) when execution starts.
Julia on deathstar: 1.12.6 (`~/.juliaup/bin/julia`, login shell PATH).

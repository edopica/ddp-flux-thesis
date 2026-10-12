# Phase H evidence metadata

Date: 2026-10-01

## Audited source

- Worktree: `/home/kurapica/Projects/ddp_flux/flux-integration`
- HEAD: `55171e659b2f425da980cbc83177bf5c38f0a7a1`
- Flux source and tests were clean before the audit.
- Historical source inspection: `git show fee2c310:src/distributed/public_api.jl` from that worktree, exit 0.
- Audit: `wiki/audits/001-conditional-wrapper-compatibility.md`.

## Runtime

| Component | Version |
|---|---|
| Julia | 1.12.6 |
| Flux | 0.16.11 |
| Optimisers | 0.4.9 |
| Functors | 0.5.3 |
| MPI.jl | 0.20.27 |
| JLD2 | 0.6.6 |
| MPI library | MPICH |

The active project was `../flux-integration/test`.
JLD2 was unavailable directly in that project.
The probe appended the thesis environment to `LOAD_PATH` after loading Flux, Optimisers, Functors, and MPI.
The logs record the actual package paths.
The Flux path remained the integration worktree.
No project or manifest changed.
The local write job compiled JLD2Ext within its recorded elapsed time.

Launcher: `/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec`.
The driver uses the full environment from `MPI.mpiexec()` without printing environment contents.

## Exact verification invocation

Working directory: `/home/kurapica/Projects/ddp_flux/ddp-flux-thesis`.

```bash
ls artifacts/logs/conditional-wrapper /tmp/opencode && timeout --signal=TERM --kill-after=10s 420s julia --startup-file=no --project=../flux-integration/test scripts/checks/conditional_wrapper_compatibility_check.jl
```

Exit: 0. The driver ran two MPI jobs in sequence.

| Job | Assertions | Exit | Elapsed | Log |
|---|---|---:|---:|---|
| `write` | 54/54 per rank, 108 total | 0 | 21.362 s | `phaseH-write-mpi2-2026-10-01.log` |
| `resume` | 35/35 per rank, 70 total | 0 | 17.223 s | `phaseH-resume-mpi2-2026-10-01.log` |

Each job used two ranks, CPU arrays, and disabled automatic CUDA/AMDGPU device initialization.
Each job had a 180-second timeout. Neither timeout occurred.
The logs contain the exact expanded child commands.
Temporary checkpoint directory: `/tmp/opencode/phaseH-EHgNtc`, deleted by the driver after success.
The second job initialized fresh MPI processes and a fresh runtime backend.
Both formats saved only the model and native inner tree.

The assertions deliberately record unsupported shape operations and synchronization limitations.
Expected exceptions are successful assertions, not command failures.
No failure note was necessary.

## Existing evidence reused

Phase F supplied API, traversal, freeze/thaw, and simulated device-adaptation coverage at two and four ranks.
Phase G supplied local numerical and collective-order evidence.
Neither suite was rerun for Phase H.
No remote command, native GPU command, Flux edit, or new API implementation occurred.

## Next action

Final source checks:

```bash
# Thesis repository
git diff --check && git status --short -- scripts/checks/conditional_wrapper_compatibility_check.jl scripts/checks/conditional_wrapper_compatibility_probe.jl wiki/audits/001-conditional-wrapper-compatibility.md artifacts/logs/conditional-wrapper/phaseH-metadata.md

# Working directory: /home/kurapica/Projects/ddp_flux/flux-integration
git diff --check && git status --short && git rev-parse HEAD
```

Both commands returned 0.
Flux remained clean at the audited HEAD.
The thesis check listed the four new probe, report, and metadata files.
The two runtime logs also exist under the evidence directory.

Conduct Phase I user review with the compatibility audit and the existing production diff.

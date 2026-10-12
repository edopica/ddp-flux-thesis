# Conditional-graph wrapper Phase G: success regression

Date: 2026-09-30

## Result

Phase G is complete. The local two-rank detector passed **210/210 assertions** in 1 m 7.1 s, with exit code 0.
Flux remains clean at `55171e659b2f425da980cbc83177bf5c38f0a7a1`.

## Changes

- The Makefile target uses `--project="$(FLUX_REPO_PATH)/test"` for the driver.
  The child inherits this environment. The thesis manifest still points to the frozen C8 worktree.
- Every mode requires exit code 0, no timeout, and completion markers from both ranks.
- Every rank reports a numerical verdict for all four parameter vectors, including every element.
- The driver checks each collective's index, buffer type, operation, local input, and reduced output.
  Distinct branch gradients establish order even though all buffers have the same length.
- The mean divides by all workers. A missing local gradient contributes zero.
  In `corruption` mode, the globally absent `c` receives no gradient collective and keeps its value.
- Startup and update timeouts remain separate. The driver reports child exit codes and elapsed times.
- Each output record uses one write. Collective records also have an explicit `END` terminator.

The final success message is:

```text
DETECTOR SUMMARY: conditional-graph support works (CPU/MPI, two ranks)
```

## Commands and evidence

Working directory: `/home/kurapica/Projects/ddp_flux/ddp-flux-thesis`.

```bash
set -o pipefail && make verify-conditional-deadlock FLUX_REPO_PATH=../flux-integration 2>&1 | tee artifacts/logs/conditional-wrapper/phaseG-detector-GREEN-mpi2-2026-09-30.log
git diff --check
git -C ../flux-integration status --short
git -C ../flux-integration rev-parse HEAD
julia --startup-file=no --project=../flux-integration/test -e 'using Flux, Optimisers, MPI; println("Julia ", VERSION); for m in (Flux, Optimisers, MPI); println(nameof(m), " ", pkgversion(m), " ", pathof(m)); end; println("MPI launcher: ", first(MPI.mpiexec().exec))'
```

| Mode | Assertions | Collectives per rank | Child elapsed time | Child exit |
|---|---:|---|---:|---:|
| `deadlock` | 74 | presence, trunk, a, b, c | 22.034 s | 0 |
| `corruption` | 62 | presence, trunk, a, b | 22.752 s | 0 |
| `control` | 74 | presence, trunk, a, b, c | 21.944 s | 0 |

Expected values `(trunk, a, b, c)` are `(1.0, 1.1, 1.4, 1.2)` for `deadlock`.
They are `(1.0, 1.1, 1.4, 1.0)` for `corruption` and `(1.0, 1.2, 1.8, 1.4)` for `control`.
Both ranks matched these values. No timeout occurred.

Environment: Julia 1.12.6, Flux 0.16.11, Optimisers 0.4.9, MPI 0.20.27.
Flux loaded from `/home/kurapica/Projects/ddp_flux/flux-integration/src/Flux.jl`.
Optimisers loaded from `/home/kurapica/.julia/packages/Optimisers/tMaaf/src/Optimisers.jl`.
MPI loaded from `/home/kurapica/.julia/packages/MPI/E9X9Q/src/MPI.jl`.
The launcher was `/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec`.

## First-run failure

The first run completed every child correctly but reported 206 passed assertions and four parser errors.
MPI merged some messages before their trailing newlines. The parser then received values such as `0.0TRACE`.
The correction writes each record and newline together and adds a trace terminator.
This was a detector defect, not a Flux regression.

First-run command:

```bash
ls artifacts/logs/conditional-wrapper && make verify-conditional-deadlock FLUX_REPO_PATH=../flux-integration 2>&1 | tee artifacts/logs/conditional-wrapper/phaseG-detector-mpi2-2026-09-30.log; test "${PIPESTATUS[0]}" -eq 0
```

The command returned 1. The detector returned 1 and Make returned 2.
Failure details: `wiki/failures/2026-09-30-conditional-detector-trace-framing.md`.

The historical characterization log remains at `artifacts/logs/conditional-deadlock/detector-2026-09-14.log`.
Its exit code 0 means that the old detector reproduced the bug. The new success contract applies to the updated scripts.

## Next action

Implement Phase H from `temp/conditional_graph_wrapper_implementation_plan.md`: review wrapper compatibility and document state-shape incompatibilities.

## Independent review (2026-09-30)

Result: no blocking findings. Phase G meets its CPU/MPI acceptance conditions.

The review covered the driver, probe, Makefile target, analytical expectations, and stored logs.
The probe uses real AD and checks every element of all four parameter vectors on both ranks.
The driver requires successful child exits, completion markers, and the expected collective count and order.
The trace assertions include local contributions, global means, element types, and the `avg` operation.
The globally absent `c` has no gradient collective in `corruption` mode.
The historical characterization log remains separate from the success log.

Independent commands from the thesis repository:

```bash
make verify-conditional-deadlock FLUX_REPO_PATH=../flux-integration
git -C ../flux-integration status --short && git -C ../flux-integration rev-parse HEAD && git diff --check
```

Both commands returned 0. The detector passed **210/210 assertions**, with a testset time of **1 m 5.2 s**.
Child elapsed times were 22.461 s (`deadlock`), 21.342 s (`corruption`), and 20.941 s (`control`).
Every child returned 0 without a timeout. The review output was captured in the tool transcript, not a new artifact log.
Flux remained clean at `55171e659b2f425da980cbc83177bf5c38f0a7a1`. The whitespace check passed.

Timeout termination was inspected but not exercised. This review did not rerun the detector against the old per-leaf implementation.
The review changed only this devlog and the current context. Phase H remains the next action.

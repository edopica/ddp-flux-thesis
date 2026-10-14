# DB-3 review: local test environment failure

Date: 2026-10-14

The independent two-rank MPI attempt exited 1 before test execution.
Both children failed at `using Flux` in the focused test file (line 35).
Julia 1.12.6 precompilation reported:

```text
KeyError: key Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA") not found
```

The stack starts in `Base.Precompilation.scan_deps!`.
This result does not establish a DB-3 product failure. The exact dependency-resolution cause remains undiagnosed.
No environment or manifest changed during the review.

Exact command from the thesis root:

```bash
timeout 300s julia --startup-file=no --project=../flux-integration/test -e 'using MPI; run(addenv(`$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --startup-file=no --project=../flux-integration/test ../flux-integration/test/ext_distributed/nonfinite_gradient_distributedtest.jl`, "FLUX_TEST_DISTRIBUTED_BACKEND" => "mpi"))'
```

Separately, the artifact-reader reproduction caught errors in both tutorial bundles.
Those errors establish review finding R1. The reproduction command exited 0 because it caught and printed each error.
See [the review](../../reports/conditional-graph-db3-review.md) for the exact command and output.

Next action: diagnose the local dependency graph before another local MPI acceptance run.

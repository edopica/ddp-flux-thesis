# Failure - main test suite crashes the host by instantiating Reactant/XLA

Date: 2026-09-10
Checkpoint: PR #2694 salvage phases 2-7 review fixes / local main-suite verification
Command: `FLUX_TEST_DISTRIBUTED_MPI=true FLUX_TEST_REACTANT=false JULIA_MPI_TEST_NPROCS=2 julia --project=test test/runtests.jl ext_distributed`
Backend: `MPICH_jll` (local) with `Pkg` instantiation that pulls in Reactant/XLA
Rank count: 2 (attempted; never reached a worker)
Hardware: local desktop, 14 GiB RAM

## Symptom

The main test suite could not be run locally. Two attempts crashed the desktop session (opencode + herdr) instead of producing a test result. The route log is empty (0 bytes), so the process died before emitting any output.

## Full error or log excerpt

`artifacts/logs/pr2694-salvage/review-fixes/final/main-suite-route.log` is empty (0 bytes). Its metadata records the attempted command:

```
cmd: FLUX_TEST_DISTRIBUTED_MPI=true FLUX_TEST_REACTANT=false JULIA_MPI_TEST_NPROCS=2 julia --project=test test/runtests.jl ext_distributed
cwd: /home/kurapica/Projects/ddp_flux/flux-pr2694-salvage
head: caf47145ba1ea2ca06370a2b3422df0b7ddf4a0f
manifest_before: 3b86244bc58065456acd34494db9d936953678e3824b107f8fdb4f68ecbb0046
```

A no-worker discovery run (same flags plus `--list`, `final/main-suite-route-list.meta`) was safe because it does not instantiate packages or spawn workers.

## Minimal reproduction

```bash
cd /home/kurapica/Projects/ddp_flux/flux-pr2694-salvage
FLUX_TEST_DISTRIBUTED_MPI=true FLUX_TEST_REACTANT=false JULIA_MPI_TEST_NPROCS=2 \
  julia --project=test test/runtests.jl ext_distributed
```

Do **not** run this on a memory-constrained host. It crashed the desktop twice.

## Suspected cause

1. `test/runtests.jl:33` calls `Pkg.add("MPI")` unconditionally when `FLUX_TEST_DIST_MPI || FLUX_TEST_DIST_NCCL` is set. This instantiates the test project even with `FLUX_TEST_REACTANT=false`.
2. At the time of the crash, `test/Project.toml` listed `Reactant` in `[deps]` while the resolved `Manifest.toml` did not contain it, so `Pkg` instantiation pulled in Reactant and its XLA dependencies.
3. Independently, `test/test_module.jl:25-31` runs `using` on any package where `Base.find_package(string(pkg)) !== nothing`, and `find_package("Reactant")` resolves once Reactant is installed. Every `ParallelTestRunner` worker therefore loaded Reactant.
4. On a 14 GiB host, Reactant/XLA plus the workers exhausted memory and the OOM killer took down opencode/herdr.

## Fix attempted

- `FLUX_TEST_REACTANT=false` does **NOT** help: it only skips the `Pkg.add("Reactant")` line (`test/runtests.jl:35`); the `Pkg.add("MPI")` line at `:33` still instantiates the project and resolves the `Reactant` dependency.
- The aborted first run's own `Pkg.add("Reactant")` appended `Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"` to `test/Project.toml`. The orchestrator reverted it with `git restore test/Project.toml`. No tracked environment changes remain; the gitignored root `Manifest.toml` is unchanged (sha256 `3b86244bc58065456acd34494db9d936953678e3824b107f8fdb4f68ecbb0046`).
- The real fix is a properly instantiated test environment or CI, not runtime `Pkg.add` calls: resolve/instantiate the test project once, keep `Reactant` out of the environment unless requested, and let CI (with enough memory) run the full `test/runtests.jl` route.

## Outcome

The full main-suite route was **not** verified locally. It was substituted by:

- the routing unit test `julia --project=test test/distributed_routing.jl` (18/18, exit 0);
- a real-discovery check against `ParallelTestRunner.find_tests("test")` with the new strip/add helpers (`ROUTING_OK`);
- the dedicated runner at `JULIA_MPI_TEST_NPROCS=2` and `4` (5/5 each, exit 0).

The full route remains a CI gate. See `wiki/devlog/01-pr2694/2026-09-10-pr2694-review-fixes.md`.

## Thesis lesson

Never rely on runtime `Pkg.add` to shape a test environment. It mutates `Project.toml`, triggers full resolution/instantiation, and can silently pull heavy optional packages (Reactant/XLA) into every test worker through `find_package`-based `using`. Test environments must be instantiated ahead of time, and heavy optional backends must be opt-in, not auto-detected. Also record resource limits: a route that OOM-kills the developer session produces no evidence and must be moved to CI.

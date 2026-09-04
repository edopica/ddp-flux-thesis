# ADR-0005 - Precompile only on HPC compute nodes

Date: 2026-09-03
Status: Accepted

## Context

Running distributed Flux tests on the HPC cluster (slnode01 login, gnode01/02 compute)
regularly hit very long precompilation delays and "timeout on precomp" failures after
every minor change to the dev'd Flux fork. The remote workflow scripts
(`setup_node.sh`, `precompile.sh`, `run.sh`) executed plain `julia` inside an `salloc`
shell, which on this cluster stays on the LOGIN node (slnode01); only `srun` reaches
the compute nodes.

Measured facts (2026-09-03, full protocol in `wiki/devlog/2026-09-03-precompile-experiment.md`):

1. **Caches are CPU-target-specific and cross-family caches do not load.** Julia bakes
   the host CPU feature set into pkgimage targets: login `slnode01` = `graniterapids`
   (Emerald Rapids), compute nodes = `icelake-server` (Ice Lake, gnode01 Xeon Gold 6326
   and gnode02 Gold 5317 are flag-identical and share caches). A precompile pass on the
   login node (~360 s thesis env, ~250 s Flux test env) left the compute nodes cold.
2. **A stale Flux cache silently recompiles at load, inside the test harness.**
   After a one-comment-line Flux change, the first 2-rank test run recompiled Flux on
   the compute node during the first test file (~4 min, racing across ranks under the
   Pkg file lock, silent — no "Precompiling" banner), pushing the file over the harness
   watchdog budgets (default 300 s; 120 s in some reports) -> "timeouts on precomp".
3. **Precompiling on a compute node fixes the compute-side problem**: thesis env
   200.8 s / test env 222.3 s (first post-sync pass, large stale set), steady-state cost
   per Flux source change ~2-4 min for both envs, afterwards warm 2-rank suites run in
   ~4.5 min with zero in-test compilation. gnode01 and gnode02 share the resulting caches.
4. **CPU-target pinning is not a fix.** `JULIA_CPU_TARGET` is ignored by the loading
   process (only CLI `-C` pins), and gnode-built images carry the `pconfig` feature,
   which the login VM lacks — so even `-C`-pinned gnode caches cannot load on login.
5. **`JULIA_PKG_PRECOMPILE_AUTO=0` does not stop load-time recompiles.** It only stops
   Pkg *operations* (develop/add/instantiate) from precompiling. A stale `using Flux`
   recompiled anyway (~97 s for Flux + FluxMPIExt, measured). It still has value: it
   keeps `make env` / setup on the login node compile-free.

## Decision

- Precompilation runs ONLY on compute nodes, via `srun`: `scripts/remote/precompile.sh hpc`
  now runs `make precompile-all` (thesis env + `ddp_flux/test` env) under
  `NTASKS=1 CPUS_PER_TASK=8`, executed as `srun` julia inside the salloc allocation.
- `make env` no longer calls `Pkg.precompile()` and sets `JULIA_PKG_PRECOMPILE_AUTO=0`
  (login-node setup resolves but never compiles).
- Makefile targets `precompile` / `precompile-flux-test` are `srun`-guarded on HPC
  (SLURM_JOB_ID set -> srun branch) and refuse to run on the login node (hostname
  `slnode01` guard). Locally they behave as before.
- `scripts/remote/hosts/hpc.conf` exports `JULIA_PKG_PRECOMPILE_AUTO=0` for every
  run.sh/precompile.sh session.
- Workflow rule: `sync_code.sh hpc` -> (first time or Project change: `setup_node.sh hpc`)
  -> `precompile.sh hpc` -> tests. Never run tests right after a sync without the
  precompile step; a stale cache recompiles silently during the run.
- On `hpc`, make variables are passed as make arguments
  (`make c8-mpi FLUX_REPO_PATH=../ddp_flux`), not as env prefixes before the command,
  because salloc execs argv directly.

## Consequences

- One deliberate, correctly-targeted precompile pass per code sync (~2-4 min steady
  state); warm test runs with zero in-test compilation and no login-node CPU burn.
- Login-node julia must not be used to load the project envs; accidental login
  compiles are wasted (wrong target) and produce depot clutter.
- JLL installs still force-compile at install time wherever Pkg runs them
  (measured ~22 s on login for CUDA JLLs) — small, accepted.
- If a precompile step is skipped, the symptom is a silent slow test run (not an
  error); teams should follow the documented workflow order.

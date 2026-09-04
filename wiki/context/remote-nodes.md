# Remote Execution & HPC Integration

This document outlines the workflows for syncing local development code to remote nodes and executing tests. This is particularly useful for running distributed tests (like C2 distributed skeleton and C4+ GPU work) without polluting the local environment or when needing specialized hardware like HPC clusters.

## Workflow Scripts

### 1. `scripts/remote/sync_code.sh <remote_host>`
**Purpose**: Sync local modifications of both the Flux fork (`ddp_flux`) and the thesis workspace (`ddp-flux-thesis`) to the remote node under `~/projects/`.
**Mechanism**: Uses `rsync` and explicitly excludes `.git`, `.julia`, and `Manifest.toml` to force the remote environment to resolve its own hardware-appropriate Manifest.

### 2. `scripts/remote/setup_node.sh <remote_host>`
**Purpose**: Automatically initialize the remote environment after the first sync (or whenever `Project.toml` changes).
**Mechanism**: Connects via SSH and executes `make env` on the remote side, establishing a native Julia environment.
**Note**: `make env` only resolves dependencies since 2026-09 (it never precompiles: `JULIA_PKG_PRECOMPILE_AUTO=0` and no trailing `Pkg.precompile()`). Precompilation is a separate, compute-node-only step (see below).

### 3. `scripts/remote/precompile.sh <remote_host>`
**Purpose**: Precompile both project envs (thesis env + `ddp_flux/test` env) on a **compute node**.
**Mechanism**: Runs `make precompile-all` inside `srun` (via `salloc`), so every julia process compiles on gnode01/gnode02 (icelake-server), never on the login node.
**When to run**: after every `sync_code.sh`, or after any change to the Flux fork source or to dependency manifests.

### 4. `scripts/remote/run.sh <remote_host> <command...>`
**Purpose**: Transparently execute a command (like `make check` or a specific script) on the remote node.
**HPC Integration**: If the remote host is `hpc`, it automatically intercepts the command and prefixes it with the Slurm `srun` wrapper:
`srun --gres=gpu:1 --mem=32G --cpus-per-task=8 --account=3320522 --partition=stud --qos=stud`
This ensures active testing on the cluster uses compute nodes instead of tying up the head node.

## Notes on Julia Precompilation (measured on HPC, 2026-09-03)

Precompilation on this cluster was the source of long delays and test timeouts. The measured root causes and the rules that fix them:

1. **Precompiling on the login node is wasted work.** Julia caches bake the host CPU feature set: the login node (`slnode01`, Xeon "Emerald Rapids", julia target `graniterapids`) and the compute nodes (gnode01/02, Xeon Gold Ice Lake, julia target `icelake-server`) do not share caches. Every precompile pass on the login node (5-10 min for the envs) left the compute nodes cold anyway; the first test run after a Flux source change then silently recompiled on-node for ~4 min inside the test harness's per-file watchdog (default 300 s, reports used 120 s) — that is the "timeout on precomp" symptom.
2. **Fix: precompile only on compute nodes.** `scripts/remote/precompile.sh hpc` runs julia under `srun`. Measured: full env precompile on a compute node ~200 s (thesis) / ~220 s (Flux test env, first time after sync); steady-state cost per Flux source change ~2-3 min for both envs; afterwards the warm 2-rank test suite runs in ~4.5-5 min with zero in-test compilation. gnode01 and gnode02 are flag-identical, so one precompile pass warms both.
3. **Do not pin `JULIA_CPU_TARGET`.** The env var only affects Pkg's precompile workers, not the loading process; only the CLI flag `-C` pins the target. Even `-C` cannot bridge login↔compute here because gnode-built images carry the `pconfig` feature, which the login node lacks. Keep all julia work on compute nodes instead.
4. **`JULIA_PKG_PRECOMPILE_AUTO=0` is set for every `run.sh`/`precompile.sh` session** (see `scripts/remote/hosts/hpc.conf`) and inside `make env`. Effect measured 2026-09-03: it stops Pkg *operations* (develop/add/instantiate, e.g. during `make env` on the login node) from precompiling, so setup stays compile-free. **It does NOT stop silent load-time recompiles**: `using Flux` on a stale cache recompiles anyway (~97 s for Flux + FluxMPIExt, measured). The real protection is workflow: run `scripts/remote/precompile.sh hpc` after every sync/source change so caches are never stale when tests start. If you see recompilation starting during a test run, a sync happened without a following precompile — stop, precompile, rerun.
5. **On `hpc`, pass make variables as make arguments, not env prefixes.** `scripts/remote/run.sh hpc "FLUX_REPO_PATH=../ddp_flux make c8-mpi"` fails with `salloc: error: _fork_command: Unable to find command "FLUX_REPO_PATH=../ddp_flux"` (salloc execs argv directly, no shell). Use `scripts/remote/run.sh hpc "make c8-mpi FLUX_REPO_PATH=../ddp_flux"` instead. Resource overrides (NTASKS, CPUS_PER_TASK) ARE env prefixes — but they must precede the *script*, e.g. `NTASKS=2 CPUS_PER_TASK=4 scripts/remote/run.sh hpc "make c8-mpi FLUX_REPO_PATH=../ddp_flux"`.

## Usage Example

To test the C2 distributed launch skeleton on the HPC cluster:

1. **Sync code:**
   ```bash
   ./scripts/remote/sync_code.sh hpc
   ```
2. **Setup environment (first time or on dependency change):**
   ```bash
   ./scripts/remote/setup_node.sh hpc
   ```
3. **Precompile on a compute node (after every sync / Flux source change):**
   ```bash
   ./scripts/remote/precompile.sh hpc
   ```
4. **Execute commands:**
   ```bash
   ./scripts/remote/run.sh hpc make check
   ./scripts/remote/run.sh hpc make smoke-cpu
   ```

## Deathstar Node
For instructions on using the deathstar node, see [deathstar.md](deathstar.md).

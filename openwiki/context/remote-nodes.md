# Remote Execution & HPC Integration

This document outlines the workflows for syncing local development code to remote nodes and executing tests. This is particularly useful for running distributed tests (like C2 distributed skeleton and C4+ GPU work) without polluting the local environment or when needing specialized hardware like HPC clusters.

## Workflow Scripts

### 1. `scripts/sync_code.sh <remote_host>`
**Purpose**: Sync local modifications of both the Flux fork (`ddp_flux`) and the thesis workspace (`ddp-flux-thesis`) to the remote node under `~/projects/`.
**Mechanism**: Uses `rsync` and explicitly excludes `.git`, `.julia`, and `Manifest.toml` to force the remote environment to resolve its own hardware-appropriate Manifest.

### 2. `scripts/setup_node.sh <remote_host>`
**Purpose**: Automatically initialize the remote environment after the first sync (or whenever `Project.toml` changes).
**Mechanism**: Connects via SSH and executes `make env` on the remote side, establishing a native Julia environment.

### 3. `scripts/run_remote.sh <remote_host> <command...>`
**Purpose**: Transparently execute a command (like `make check` or a specific script) on the remote node.
**HPC Integration**: If the remote host is `hpc`, it automatically intercepts the command and prefixes it with the Slurm `srun` wrapper:
`srun --gres=gpu:1 --mem=32G --cpus-per-task=8 --account=3320522 --partition=stud --qos=stud`
This ensures active testing on the cluster uses compute nodes instead of tying up the head node.

## Notes on Julia Precompilation

When using these scripts on an HPC cluster, you may encounter the following behaviors during package precompilation:

1. **Long Setup Times:** The initial run of `setup_node.sh` installs and precompiles heavy packages (like LLVM, Zygote, and Flux). This can take 5-10 minutes. If running this via an automated agent with strict timeouts, the command might get aborted. Simply re-run `setup_node.sh`—Julia will resume precompilation right where it left off.
2. **Double Precompilation (Login vs. Compute Nodes):** `setup_node.sh` executes on the cluster's **login node**. However, `run_remote.sh hpc` uses `srun` to execute your command on a **compute node**. Since compute nodes often have a different CPU architecture or instruction set than login nodes, Julia will detect the hardware change and trigger a second round of precompilation the first time you run `run_remote.sh`. This is normal and ensures the code is optimized for the actual execution hardware.

## Usage Example

To test the C2 distributed launch skeleton on the HPC cluster:

1. **Sync code:**
   ```bash
   ./scripts/sync_code.sh hpc
   ```
2. **Setup environment (first time or on dependency change):**
   ```bash
   ./scripts/setup_node.sh hpc
   ```
3. **Execute commands:**
   ```bash
   ./scripts/run_remote.sh hpc make check
   ./scripts/run_remote.sh hpc make smoke-cpu
   ```

## Deathstar Node
For instructions on using the  node, see [deathstar.md](deathstar.md).

## Deathstar Node
For instructions on using the deathstar node, see [deathstar.md](deathstar.md).

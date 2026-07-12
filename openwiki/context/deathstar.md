# Deathstar Node Interaction Guide

## Overview

The remote node `deathstar` is a non-SLURM Ubuntu machine configured for remote execution. It is accessible via SSH (alias `deathstar` in `~/.ssh/config`).

**IMPORTANT WARNING**: This node is someone else's personal computer.
- **Do NOT overuse it**: Limit your testing and execution to what is strictly necessary.
- **NO destructive commands**: Commands that delete, modify system configurations, or heavily disrupt the file system outside of the project workspace are strictly prohibited.
- **Caution with burdensome workflows**: Workflows that require heavy CPU/GPU load or excessive memory should be treated with caution to avoid freezing the system or interrupting the owner's work. Always clean up processes if they hang.

## Workflow Scripts

The same scripts used for the `hpc` cluster can be used for `deathstar`, but they will run directly on the machine without any job scheduler wrapper like `srun`.

### 1. Syncing Code
To sync the local modifications to the remote node:
```bash
./scripts/remote/sync_code.sh deathstar
```
This syncs both the Flux fork and the thesis workspace to `~/projects/` on `deathstar`.

### 2. Setting Up the Environment
To initialize the remote environment (compiling dependencies, setting up the Julia environment):
```bash
./scripts/remote/setup_node.sh deathstar
```

### 3. Running Commands
To execute a command or test on `deathstar`:
```bash
./scripts/remote/run.sh deathstar make check
```
*(Note: Unlike the `hpc` node, `run_remote.sh deathstar` executes commands directly on the host without `srun`.)*

## Current Status
- SSH configuration is present in `~/.ssh/config`.
- **Note**: The environment has not been fully initialized yet because SSH currently requires authentication (password or public key setup). Once authenticated, run the sync and setup commands above to prepare the workspace.

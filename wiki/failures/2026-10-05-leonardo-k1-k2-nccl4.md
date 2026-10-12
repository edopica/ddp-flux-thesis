# Leonardo Phase K: failed and blocked commands (K1/K2 4-rank NCCL)

Date: 2026-10-05
Host: `leonardo` (CINECA). All commands via `scripts/remote/run.sh leonardo`.
No product code changed. Both Phase K cells ultimately passed; this record
covers the failed or aborted commands on the way there, plus one benign
warning and one console-write observation.

## Failure 1: `srun ... nvidia-smi -L` cannot execute nvidia-smi

The GPU check `srun --mpi=pmi2 -n 1 -c 2 nvidia-smi -L` failed with
`srun: fatal: Can not execute nvidia-smi`.

Cause (observed): the direct `srun` exec target did not resolve in the task
environment. `/usr/bin/nvidia-smi` exists on the compute nodes and is found
once the command runs through a login shell.

Correction: use a login shell with the absolute binary:
`srun --mpi=pmi2 -n 1 -c 2 bash -lc '/usr/bin/nvidia-smi -L'`. The rerun
listed the four A100s.

## Failure 2: remote rsync path `$WORK` was not expanded

The prepared-dataset transfer used the remote path `$WORK`; the remote shell did
not expand it, so the remote `mkdir` failed with code 11.

Cause: the remote command did not inherit the interactive `$WORK` environment.

Correction: use the literal path
`/leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05/`. The
transfer then succeeded; the artifact SHA-256 was verified and no download ran
on a compute node.

## Failure 3: `uv run --with` PyPI connect timeout

The figure command `uv run --with matplotlib --with numpy` exited 2 after a
PyPI connect timeout.

Cause: no usable PyPI access at that moment.

Correction: rerun with `--offline` against the warm uv cache (matplotlib
3.11.2, numpy 2.5.3). The figure command then exited 0 and produced 41 PNGs
plus `figures-metadata.txt`.

## Failure 4: full-tree rsync aborted

A full-tree rsync of the thesis repo was aborted because it would have uploaded
about 650 MB of prior artifacts.

Cause: the local `artifacts/` tree is much larger than the demo sources that
changed.

Correction: targeted rsync of only
`scripts/checks/conditional_graph_demo_check.jl`,
`scripts/checks/conditional_graph_checkpoint_roundtrip.jl`,
`scripts/examples/conditional_graph_demo.jl`, and
`scripts/examples/conditional_graph/` (no `--delete`).

## Not a failure: benign Slurm warning

`srun: Job step's --cpus-per-task value exceeds that of job (8 > 4). Job step
may never run.` The allocation requested `--cpus-per-task=4` while the step
asked for `-c 8`. The step ran and the jobs passed; listed for completeness.

## Note: four ranks concatenate records on one line

All four ranks write to one `console.log`, so some `DEMO_STEP` records share a
line. Counts were verified with `grep -o` rather than by line count: K2 480
train / 472 resume records on 362/352 lines; K1 64/56 records. This is a
console-write observation, not a failure.

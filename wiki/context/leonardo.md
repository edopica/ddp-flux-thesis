# Leonardo Node (planned)

`leonardo` is the planned CINECA GPU cluster (SLURM) for the GPU/NCCL phase. An SSH alias exists in `~/.ssh/config` (`HostName login.leonardo.cineca.it`, user `epicazio`), but no workflow has been validated yet.

**Do not run sync/setup/precompile/tests on `leonardo` until this page documents:**

- login and compute node CPU targets (the precompile rule),
- Slurm account, partition, QoS, and GPU GRES flags,
- the module/MPI/NCCL stack and PMI protocol,
- a validated smoke-test command.

`scripts/remote/run.sh` rejects `leonardo` until a `scripts/remote/hosts/leonardo.conf` exists.

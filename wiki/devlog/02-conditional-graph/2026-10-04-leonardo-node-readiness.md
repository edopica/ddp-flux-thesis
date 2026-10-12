# Leonardo node readiness check

Date: 2026-10-04
Host: `leonardo` (CINECA), user `epicazio`, alias in `~/.ssh/config`
Evidence: `artifacts/logs/leonardo/2026-10-04-node-readiness.md`
Context page: `wiki/context/leonardo.md` (rewritten this session)

## Outcome

SSH access to Leonardo works. The Slurm configuration, the module/PMI stack,
the CPU targets, and single-node 4-GPU use are now verified and documented.
No Julia workflow ran: the cluster has no `julia` module and no `julia` binary,
and `scripts/remote/hosts/leonardo.conf` does not exist yet (so `run.sh`
still rejects `leonardo`). No repository code, test, or remote environment
changed. Nothing was synced.

## Verified facts

- SSH alias `leonardo` -> `login.leonardo.cineca.it`, user `epicazio`;
  sessions land on `login01`…`login05`.
- Account `iscrc_unmasked`; partition `boost_usr_prod`; QoS `boost_qos_dbg`
  (30 min, 8 nodes / 32 GPUs), `boost_qos_bprod` (24 h, 256 nodes),
  `boost_qos_lprod` (4 days). Boost nodes: 32 CPUs, 514 GB RAM,
  `gres/gpu=a100:4`; 3256 nodes / 13024 GPUs.
- Login and compute CPUs are both Xeon Platinum 8358 (Ice Lake), and their
  `/proc/cpuinfo` flags are byte-identical (md5
  `cdd007a665abdd169b9384296f33ffec`). The Bocconi login/compute cache
  incompatibility does **not** exist here.
- Modules: `cuda/12.2` (default) + `12.3`/`12.6`, `cudnn/8.9.7.29-12`,
  `nccl/2.22.3-1--gcc--12.2.0-cuda-12.2`, `openmpi/4.1.6` (default) and `5.0.9`
  with CUDA, `hpcx-mpi/2.19`, `intel-oneapi-mpi/2021.12.1`.
- `srun --mpi=list` -> `none`, `pmix` (v3, v5), `pmi2`, `cray_shasta`.
  Julia's `MPICH_jll` needs `srun --mpi=pmi2` / `SLURM_MPI_TYPE=pmi2`.
- Login nodes reach github.com, julialang-s3, and pkg.julialang.org over HTTPS
  (no proxy).
- `$HOME` is 50 GB; `$WORK` (`/leonardo_work/IscrC_UNMASKED`) is 1 TB with 5%
  used, so the Julia depot/artifacts should live on `$WORK` or `$SCRATCH`.

## Single-node 4-GPU check (the user-requested gate)

Command (node `lrdn0176`):

```
srun -A iscrc_unmasked -p boost_usr_prod --qos=boost_qos_dbg \
  -N1 -n4 --cpus-per-task=4 --gpus-per-task=1 --time=00:10:00 \
  bash -lc 'echo "task on $(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; \
            nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader; \
            ~/leonardo-gpu-smoke/gpu_smoke'
```

Result: four ranks on one node, four **distinct** A100-SXM-64GB UUIDs, each rank
one GPU (`CUDA_VISIBLE_DEVICES` remapped to `0` per task), and the CUDA kernel
of `gpu_smoke` returned `h[0]=1.0` on every rank. With `--gres=gpu:4` instead,
every rank sees all four GPUs (`0,1,2,3`) — use `--gpus-per-task=1` for DDP.
A second check (`--gres=gpu:4`, node `lrdn0981`) showed
`cudaDeviceCanAccessPeer = YES` for all 12 ordered GPU pairs, with 256 MiB
`cudaMemcpyPeer` at 43.9–89.8 GB/s.

Anomaly recorded, not resolved: `nvidia-smi nvlink -s` reports 12 links up at
25 GB/s, `nvidia-fabricmanager` is inactive, and `nvidia-smi topo -m` prints no
`NV#` entries even though P2P bandwidth is above PCIe. Measure NCCL behavior
during GPU bring-up; do not assume a topology from `topo -m`.

## Wrapper shape

`salloc` control commands stay on the login node, exactly like Bocconi; only
the inner `srun` reaches compute nodes. Verified:

```
salloc -A iscrc_unmasked -p boost_usr_prod --qos=boost_qos_dbg \
  -N1 -n4 --gres=gpu:4 --cpus-per-task=4 --time=00:05:00 \
  bash -c "hostname; srun --mpi=pmi2 bash -c 'echo step-host=$(hostname) rank=$SLURM_PROCID'"
-> login02 (control), then lrdn0143 rank 0..3, srun --mpi=pmi2 clean.
```

## Documentation changes

- `wiki/context/leonardo.md`: replaced the planned-only stub with the verified
  environment, the 4-GPU pattern, the precompile rule, the storage/network
  facts, and an explicit open checklist. The "do not run until documented"
  block is gone; the new gate is the open checklist (Julia install,
  `hosts/leonardo.conf`, sync/env/precompile, MPI + NCCL smoke).
- `wiki/context/remote-nodes.md`: index row updated to
  "SSH/Slurm/4-GPU verified, Julia workflow not yet up".
- `artifacts/logs/leonardo/2026-10-04-node-readiness.md`: full command/output
  evidence.

## Next action

Install Julia 1.12.x on Leonardo with a `$WORK`/`$SCRATCH` depot, then add
`scripts/remote/hosts/leonardo.conf` mirroring `bocconi.conf` (`salloc` wrapper,
`SLURM_MPI_TYPE=pmi2`, `JULIA_PKG_PRECOMPILE_AUTO=0`, `boost_qos_dbg`), sync,
run `setup_node.sh leonardo`, precompile on a compute node, and run the first
MPI smoke. This is the prerequisite for using Leonardo for the Phase K
4-rank GPU/NCCL runs.

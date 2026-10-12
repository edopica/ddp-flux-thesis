# Leonardo node readiness check — metadata

Date: 2026-10-04
Host: `leonardo` (CINECA). SSH alias in `~/.ssh/config`
(`HostName login.leonardo.cineca.it`, user `epicazio`).
Access verified on `login01`, `login02`, `login05.leonardo.local`.
Scope: SSH access, Slurm configuration, module stack, CPU targets, and
single-node 4-GPU use. No Julia workflow was run: Julia is not installed on the
cluster (no `julia` module, no `julia`/`juliaup` binary).

All commands below were run from the local thesis repo root via `ssh leonardo`
(or directly inside an `srun`/`salloc` allocation, as noted). No repository
files were synced and no test was executed.

## 1. SSH access

```
$ ssh leonardo 'hostname; whoami; echo OK'
login05.leonardo.local
epicazio
OK
```

Session hostname varies between `login01` … `login05`.

## 2. Slurm configuration

```
$ sinfo -o "%20P %6D %14F %12m %6c %12G %20f %15l"
PARTITION            NODES  NODES(A/I/O/T) MEMORY       CPUS   GRES         AVAIL_FEATURES       TIMELIMIT
boost_usr_prod       3256   3190/66/0/3256 514000       32     gpu:a100:4   xorg                 1-00:00:00
dcgp_usr_prod        1459   1221/238/0/145 514000       112    tmpfs:3T     kvmcpu               1-00:00:00
lrd_all_viz          15     2/2/11/15      514000       128    gpu:quadro:2 xorg                 12:00:00
lrd_all_serial*      2      2/0/0/2        514000       128    tmpfs:14T    (null)               4:00:00

$ scontrol show partition boost_usr_prod
   MaxNodes=64 MaxTime=1-00:00:00 MinNodes=0
   State=UP TotalCPUs=104192 TotalNodes=3256
   TRES=cpu=104192,mem=1673584000M,node=3256,billing=104192,gres/gpu=13024

$ sacctmgr -n -P show assoc user=$USER format=Account,Partition,Qos,DefaultQOS
iscrc_unmasked||boost_qos_bprod,boost_qos_dbg,boost_qos_lprod,normal|

$ sacctmgr -n -P show user $USER format=DefaultAccount
iscrc_unmasked

$ sacctmgr -n -P show qos format=Name,MaxWall,MaxTRESPerUser,Flags | grep boost
boost_qos_bprod|1-00:00:00|node=256|DenyOnLimit,PartitionMaxNodes
boost_qos_lprod|4-00:00:00||DenyOnLimit,PartitionTimeLimit
boost_qos_dbg|00:30:00|cpu=256,gres/gpu=32,mem=3952000M,node=8|DenyOnLimit,OverPartQOS
boost_usr_prod||node=256|DenyOnLimit
```

Usable by `epicazio`:

| Account | Partition | QoS | Max wall | Limits |
|---|---|---|---|---|
| `iscrc_unmasked` | `boost_usr_prod` | `boost_qos_dbg` | 30 min | 8 nodes, 32 GPUs, cpu=256, mem≈3.95 TB per user |
| `iscrc_unmasked` | `boost_usr_prod` | `boost_qos_bprod` | 24 h | max 256 nodes (DenyOnLimit) |
| `iscrc_unmasked` | `boost_usr_prod` | `boost_qos_lprod` | 4 days | partition time limit |

Node TRES (any boost node, e.g. `lrdn3476`): `cpu=32,mem=514000M,billing=32,gres/gpu=4`
(`CoresPerSocket=32`, `Sockets=1`, 514 GB RAM, 4 GPUs).

## 3. CPU targets (login vs compute)

```
login (login01) : Intel(R) Xeon(R) Platinum 8358 @ 2.60GHz
                  Thread(s) per core: 2, Core(s) per socket: 32, Socket(s): 2
                  RHEL 8.8

compute (srun)  : Intel(R) Xeon(R) Platinum 8358 @ 2.60GHz
                  Thread(s) per core: 1, Core(s) per socket: 32, Socket(s): 1
                  NUMA node0: 0-15, NUMA node1: 16-31; CPU(s): 32
```

Feature-flag fingerprints:

```
login   : grep -m1 '^flags' /proc/cpuinfo | md5sum -> cdd007a665abdd169b9384296f33ffec
compute : grep -m1 '^flags' /proc/cpuinfo | md5sum -> cdd007a665abdd169b9384296f33ffec
```

The flag sets are byte-identical, so the Julia CPU target is the same on login
and compute nodes (unlike Bocconi, where login=`graniterapids` and
compute=`icelake-server`). Precompiling through a compute node remains the
project rule, because that is where the tests run.

## 4. Modules, MPI, NCCL, PMI

```
$ module --version
Modules Release 5.2.0 (2022-11-08)

$ module avail | grep -Ei 'cuda|nccl|mpi'
cuda/12.2 (default), cuda/12.3, cuda/12.6
cudnn/8.9.7.29-12--gcc--12.2.0-cuda-12.2 (default)
nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22 (default)
openmpi/4.1.6--gcc--12.2.0-cuda-12.2 (default), openmpi/5.0.9--gcc--12.2.0-cuda-12.2
hpcx-mpi/2.19 (default), hpcx-mpi/2.25.1
intel-oneapi-mpi/2021.12.1 (default)
gcc/12.2.0; system gcc 8.5.0; glibc 2.28; system python3 3.6.8

$ srun --mpi=list
MPI plugin types are...
        none
        pmix
        pmi2
        cray_shasta
specific pmix plugin versions available: pmix_v3,pmix_v5
```

Julia's default `MPICH_jll` expects the PMI2 wire protocol, so
`srun --mpi=pmi2` / `SLURM_MPI_TYPE=pmi2` applies here exactly as on Bocconi.
No `julia` module exists; the toolchain must be installed by the user.

## 5. GPUs

```
driver: 535.274.02
4x NVIDIA A100-SXM-64GB, compute capability 8.0, 65536 MiB per boost node
nvidia-smi nvlink -s: 12 links up, 25 GB/s each
nvidia-smi topo -m: does NOT print NV# entries on the tested nodes
nvidia-fabricmanager: inactive (Fabric State N/A in nvidia-smi -q)
4x InfiniBand HCA mlx5_0..3, OFED-internal-23.04-1.1.3
```

CUDA peer-access check (single process, `--gres=gpu:4`, node `lrdn0981`,
256 MiB `cudaMemcpyPeer` per ordered pair):

```
peer i->j: YES for all 12 ordered pairs
copy 0<-1: 89.8 GB/s   copy 1<-0: 44.2 GB/s
copy 0<-2: 87.7 GB/s   copy 2<-0: 43.9 GB/s
copy 0<-3: 87.3 GB/s   copy 3<-0: 43.9 GB/s
copy 1<-2: 87.4 GB/s   copy 2<-1: 43.9 GB/s
copy 1<-3: 87.7 GB/s   copy 3<-1: 43.9 GB/s
copy 2<-3: 87.9 GB/s   copy 3<-2: 43.9 GB/s
```

P2P works on every pair, well above PCIe bandwidth, even though
`nvidia-smi topo -m` omits NV# and the fabric manager is inactive. Bandwidth
is asymmetric (≈44 vs ≈88 GB/s) and far below the physical 12×25 GB/s; NCCL
behavior should be measured during GPU bring-up rather than assumed.

## 6. Verified single-node 4-GPU run

Test program: `~/leonardo-gpu-smoke/gpu_smoke.cu` (compiled on the login node
with `module load cuda/12.2 && nvcc -O2`, allocates a 32-float buffer, adds 1
in a kernel, copies back and prints the result).

```
$ ssh leonardo
$ srun -A iscrc_unmasked -p boost_usr_prod --qos=boost_qos_dbg \
    -N1 -n4 --cpus-per-task=4 --gpus-per-task=1 --time=00:10:00 \
    bash -lc 'echo "task on $(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; \
              nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader; \
              ~/leonardo-gpu-smoke/gpu_smoke'
```

Output (node `lrdn0176`, 4 ranks, each rank one distinct A100):

```
task on lrdn0176.leonardo.local CUDA_VISIBLE_DEVICES=0   (x4)
0, NVIDIA A100-SXM-64GB, GPU-46744d63-a516-0b2a-692c-5abcd836fc41, 65536 MiB, 535.274.02
0, NVIDIA A100-SXM-64GB, GPU-499ffcf0-c9dd-3f2d-ee69-34a7f9b6827e, 65536 MiB, 535.274.02
0, NVIDIA A100-SXM-64GB, GPU-eec2ba8b-9e17-903b-77d1-1327b05410f9, 65536 MiB, 535.274.02
0, NVIDIA A100-SXM-64GB, GPU-3959c0ed-98f1-fc85-82d2-b4f46be699e3, 65536 MiB, 535.274.02
visible_devices=1 device0=NVIDIA A100-SXM-64GB cc=8.0 h[0]=1.0   (x4)
```

Four distinct UUIDs, one per rank, and a successful CUDA kernel on each.
With `--gpus-per-task=1`, Slurm remaps each rank's `CUDA_VISIBLE_DEVICES` to a
single GPU (index 0). With `--gres=gpu:4` instead, every rank sees all four
GPUs (`CUDA_VISIBLE_DEVICES=0,1,2,3`); use `--gpus-per-task=1` for DDP.

## 7. `salloc` wrapper shape (mirrors bocconi)

```
$ salloc -A iscrc_unmasked -p boost_usr_prod --qos=boost_qos_dbg \
    -N1 -n4 --gres=gpu:4 --cpus-per-task=4 --time=00:05:00 \
    bash -c "hostname; srun --mpi=pmi2 bash -c 'echo step-host=$(hostname) rank=$SLURM_PROCID gpus=$CUDA_VISIBLE_DEVICES'"
salloc: Granted job allocation 59334645
login02.leonardo.local          <- salloc control command runs on the LOGIN node
salloc: Nodes lrdn0143 are ready for job
step-host=lrdn0143.leonardo.local rank=0 gpus=0,1,2,3
step-host=lrdn0143.leonardo.local rank=1 gpus=0,1,2,3
step-host=lrdn0143.leonardo.local rank=2 gpus=0,1,2,3
step-host=lrdn0143.leonardo.local rank=3 gpus=0,1,2,3
```

Only the inner `srun` reaches compute nodes; `--mpi=pmi2` loads the plugin
without error. This is the same wrapper pattern as `hosts/bocconi.conf`.

## 8. Network access from the login node

```
$ curl -sI https://github.com | head -1                    -> HTTP/2 200
$ curl -sI https://raw.githubusercontent.com | head -1      -> HTTP/2 301
$ curl -sI https://julialang-s3.julialang.org | head -1     -> HTTP/2 200
$ curl -sI https://pkg.julialang.org | head -1              -> HTTP/2 301
```

No proxy is needed: package downloads and Julia artifacts can be fetched from
the login node.

## 9. Not verified / open items

- Julia installation and depot layout (no Julia on the cluster; local and
  deathstar use 1.12.6; `$HOME` is a 50 GB filesystem, `$WORK`
  (`/leonardo_work/IscrC_UNMASKED`) has 1 TB with 5% used).
- `scripts/remote/hosts/leonardo.conf` (does not exist; `run.sh` rejects
  `leonardo` until it does).
- First sync, `make env`, `MPI.install_mpiexecjl`, compute-node precompile.
- MPI (`srun --mpi=pmi2` with MPICH_jll) and NCCL smoke tests with Julia.
- Repo target directory on Leonardo (`~/projects/`, as `sync_code.sh`
  expects) and whether the depot lives in `$HOME` or `$WORK`.

No failure occurred in this check. No repository code or test changed.

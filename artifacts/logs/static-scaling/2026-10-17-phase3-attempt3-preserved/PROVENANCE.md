# Provenance — static-scaling Phase 3 attempt-3 evidence

Status: **preserved and verified**. Date: **2026-10-17** (UTC preservation window
17:45–18:00). Historical run: **job 60885091**, node **lrdn0084.leonardo.local**.

This file records the identity of the evidence for the failed Phase 3 numerical
acceptance run (`PHASE3_REMOTE_CHECK FAIL`, exit 1) and of the environment that
produced it. No training was rerun; no reference generator was rerun; no remote
source file was modified.

## 1. Locations

Remote immutable evidence root (created on Leonardo `$WORK`):

```text
/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-attempt3/
  reference/   phase3_reference.npz, phase3_reference_manifest.json
  diag/        stage2_diag.npz
  logs/        remote-check.log, attempt-3 driver log, three diagnose logs
  provenance/  source hashes, manifests, probes, flags, commands
  SHA256SUMS   SHA-256 of every file in the bundle
```

Local preserved copies (this repository):

```text
artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/
  phase3_reference.npz
  phase3_reference_manifest.json

artifacts/logs/static-scaling/2026-10-17-phase3-attempt3-preserved/
  PROVENANCE.md, PROVENANCE.json
  SHA256SUMS                 (remote layout; authoritative bundle hashes)
  diag/stage2_diag.npz
  logs/                      remote-check.log, attempt-3 driver log, diagnose logs
  provenance/                raw provenance files copied from the remote bundle
```

`chmod -R a-w` was applied to the remote evidence root. This is best-effort
write protection, not filesystem immutability; the owner can reverse it.

## 2. Historical run

| Field | Value |
|---|---|
| Slurm job | `60885091` |
| Compute node | `lrdn0084.leonardo.local` |
| Driver start | `utc=2026-10-17T16:59:54Z` (log line 9) |
| Reference created | `2026-10-17T17:02:14Z` (manifest `created`) |
| `remote-check.log` end | `2026-10-17T17:07:58Z` (mtime 19:07:58 CEST) |
| Stage-2 diagnostic | job `60887764` on `lrdn0339`, `stage2_diag.npz` mtime 19:17:30 CEST |
| Reference generator | PASS, 32 checks |
| Independent artifact check | PASS, 337 arrays |
| Flux comparison | FAIL, exit 1 (`PHASE3_REMOTE_CHECK FAIL`) |
| Allocation / QoS | `boost_qos_dbg`, 1 node, 1 task, 8 CPUs, 1 GPU, 30 min |
| Tolerances | `ATOL=1e-5`, `RTOL=1e-4` |

The failing Flux categories were `phase3.activations`, `phase3.logits`,
`phase3.grads.u1`, `phase3.grads.u3`, `phase3.params.u3`, and `phase3.m.u3`
(see `logs/remote-check.log` lines 185–204). `phase3.loss` and `phase3.v.u3`
passed.

## 3. Source identity

### 3.1 Historical values supplied to the run

Taken verbatim from line 1 of the attempt-3 driver log
(`provenance/05_attempt3_launch_command.txt`; log SHA-256
`3325a3575c9be7ec08748e76b29a48c3328b538c3ede8630b88535276751a4bf`):

| Variable | Value |
|---|---|
| `SOURCE_COMMIT` | `3088f61020137912af32d9c969e87ea9ae2a71c3` |
| `SOURCE_STATE_SHA256` | `82623bbb516a981d6c795792d1433521d25a8ee3bbcc44a2b66f6f4631a452b9` |
| `FLUX_COMMIT` | `f7b4520c4bd307c67bbc5ac92a1c68493eccff6e` |
| `FLUX_STATE_SHA256` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

`FLUX_STATE_SHA256` is the SHA-256 of empty input, i.e. the Flux worktree was
clean relative to `FLUX_COMMIT`. The run log itself prints
`THESIS_GIT=<no-git>` and `FLUX_GIT=<no-git>` (lines 21–22).

### 3.2 Remote git inspection: unavailable

Leonardo `~/projects/ddp-flux-thesis` and `~/projects/flux-integration` are
rsynced trees **without `.git`**. The requested commands fail:

```text
$ git -C ~/projects/ddp-flux-thesis rev-parse HEAD
fatal: not a git repository (or any parent up to mount point /)
exit=128
$ git -C ~/projects/flux-integration rev-parse HEAD
fatal: not a git repository (or any parent up to mount point /)
exit=128
```

Consequently no valid remote `git diff HEAD | sha256sum` or untracked-list hash
exists. The preservation probe printed the empty-input hash
`e3b0c442…b855` for those pipelines **with git exit 128/129**; that is a
false-positive and is flagged in `provenance/50_missing_and_flags.txt`. See
`provenance/00_git_probe.txt` for the raw output.

### 3.3 Local corroboration (computed 2026-10-17, UTC 17:45–17:58)

Thesis repository (`/home/kurapica/Projects/ddp_flux/ddp-flux-thesis`):

| Quantity | Value |
|---|---|
| `git rev-parse HEAD` | `3088f61020137912af32d9c969e87ea9ae2a71c3` (matches `SOURCE_COMMIT`) |
| `git diff HEAD \| sha256sum` | `a96456534d1d467c649bc813291f8d3070a924f035743db546321b05ea0df8bc` |
| `git ls-files --others --exclude-standard \| sort \| sha256sum` | `31a7967fc7c63b1820e43e6b7c2891a2751b0c6ab1e890b1a4d20f02ab484815` |
| Full state recipe (below) | `fe176b0c6e444fd32324603ed223430b2e60df143aebc49e483b770b39d219a6` |

Full state recipe (from `artifacts/logs/static-scaling/2026-10-16-phase2/COMMANDS.md`):

```bash
STATE=$( { git diff HEAD; git ls-files --others --exclude-standard \
  | grep -v '^artifacts/' | sort | xargs -r sha256sum; } | sha256sum | cut -d' ' -f1 )
```

The current state hash differs from the historical `82623bbb…` because reports,
the handoff, and devlog entries were added after job `60885091`. The historical
state is **not reconstructible from the current tree**; use the recorded value
only to identify that run.

Flux worktree (`/home/kurapica/Projects/ddp_flux/flux-integration`):

| Quantity | Value |
|---|---|
| `git rev-parse HEAD` | `f7b4520c4bd307c67bbc5ac92a1c68493eccff6e` (matches `FLUX_COMMIT`) |
| `git status --porcelain` | empty (0 lines) |
| `git diff HEAD \| sha256sum` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `git ls-files --others --exclude-standard \| sort \| sha256sum` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

These local values corroborate the historical empty-diff `FLUX_STATE_SHA256`.

### 3.4 Executed source files

The remote Phase 3 source files are byte-identical to the local files (all 11
checked; full table in `provenance/10_source_files_sha256.txt`):

| File | SHA-256 |
|---|---|
| `scripts/remote/leonardo/run-phase3-check.sh` | `f52c454de1ecbadaa45a7c7be9556ab59ab0628d99269e3d0a0ad2abc5432f6b` |
| `scripts/static_scaling/make_phase3_reference.py` | `0a9a2645f2bf0bd66e94e1351d3538e050a8261dee451c77dd4b7b282cdab21d` |
| `scripts/static_scaling/check_phase3_reference.py` | `85297025dfd9b2222e2ecbae6c5bdca80f0e724f80b680fcdc67fa6653723515` |
| `scripts/static_scaling/check_phase3_flux.jl` | `c45b8a2e6f4bfbcb1c69c6e6debd60e4f5333a8675b884ffaf4d5d299d1c5e65` |
| `scripts/static_scaling/model_flux.jl` | `f187ba8929246bbf4c56d8fad726ed23668d4e4f4ed06f107f1b8cf4d96f8ea5` |
| `scripts/static_scaling/model_torch.py` | `77ac48e815b58948ab64b16938d988cae35f4719f084c25432703f6ffadfdf76` |
| `scripts/static_scaling/diagnose_phase3_flux_stage2.jl` | `57c32f42041daab933495b12aa90fa3d6852e350497160b7c468ea7640a96d3b` |
| `scripts/static_scaling/diagnose_phase3_torch_stage2.py` | `cb37fc124b7dab1ffcd43ed0f02fa56dc2f425333980c5f30ea67fd89ea26e97` |
| `scripts/remote/leonardo/run-julia-phase3.sh` | `43841338e8fad8cb00dc192bed30860c05e9380ba91ba2ab9a8751f643ef6314` |
| `scripts/remote/leonardo/diagnose-phase3.sh` | `f9bf5485992e4fae4408922ddc6d635145796b86a8bc38823c897b0c31f97211` |
| `scripts/remote/leonardo/diagnose-phase3-stage2.sh` | `e247b0f17b834e85bac1127e9c848331f5c40128194e5e3b4822b2b5aeeea7c9` |

Because the remote tree has no `.git`, identity is anchored by these content
hashes plus the caller-supplied commit/state values, not by remote git output.
A preservation-time tree-content hash (not the historical run-time state) is in
`provenance/15_tree_hashes.txt`:

- thesis source tree (excludes `artifacts/` and `.git`): `e6bdf241324ae3334ce398c80e0b60b85951a7fcd952eb379e2b2739ff650994`
- Flux worktree (excludes `.git`): `44b3a5b51fdb08c5c22815e0121a923d4468817c44ab7c828c839ba13ac1effd`

## 4. Environment

### 4.1 Julia

| Item | Value |
|---|---|
| Julia | `1.12.6` (`$WORK/epicazio/julia-1.12.6/bin/julia`) |
| `scripts/static_scaling/julia-env/Manifest.toml` | `c8077e0c6504adedfdce5a35dd1a2ba6bde057fc45ee9bd7ba1edbd35ee23874` (4669 bytes; NPZ-only environment) |
| `~/projects/flux-integration/Manifest.toml` | `4c4eaf77633c493d8463a1a57236197423a8995f92f92eff00879475fdce5837` (93222 bytes; remote machine-resolved) |
| `~/projects/flux-integration/test/Manifest.toml` | **MISSING** (remote and local) |
| `~/projects/flux-integration/test/Project.toml` | `13970e0a9b8595595ce63a9e05d69de0b381fff86d16f774c4e0d04d8bb38853` |
| CUDA.jl | `6.2.2` |
| cuDNN.jl | `6.2.2` (`$WORK/epicazio/julia-depot/packages/cuDNN/0SFve`) |
| CUDNN_jll | `9.26.0+0` (cuDNN 9.26.0) |
| NCCL.jl / NCCL_jll | `0.1.3` / `2.30.7+0` |
| CUDA.jl `runtime_version()` | `12.9.0` |
| CUDA.jl `driver_version()` | `13.3.0` |
| `cuDNN.cudnnGetVersion()` | `92600` (cuDNN 9.26.0) |
| `cuDNN.version()` | prints `92.60.0` (cuDNN.jl parser divides by 1000; see flags) |

Probe job `60893401` (node `lrdn0042`) confirmed the mapping directly:

```text
cuDNN.math_mode(PEDANTIC_MATH) = CUDNN_DEFAULT_MATH
cuDNN.math_mode(DEFAULT_MATH)  = CUDNN_TENSOR_OP_MATH
cuDNN.math_mode(FAST_MATH)     = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION
after CUDA.math_mode!(PEDANTIC_MATH):
  CUDA.math_mode()      = PEDANTIC_MATH
  cuDNN.math_mode()     = CUDNN_DEFAULT_MATH
  CUDA.math_precision() = TensorFloat32
```

The checker calls `CUDA.math_mode!(CUDA.PEDANTIC_MATH)`
(`scripts/static_scaling/check_phase3_flux.jl:368`), so the historical run
requested `PEDANTIC_MATH`, which cuDNN.jl maps to `CUDNN_DEFAULT_MATH`.
The mapping source is
`$WORK/epicazio/julia-depot/packages/cuDNN/0SFve/src/cuDNN.jl:50-61`
(package version `6.2.2`, file SHA-256
`86a54c6d299e68382b2b68e462f9e7fcbae504ba9a91db7d82d9caf383351119`;
full text in `provenance/30_cudnn-math-mode.txt`). Note that
`CUDA.math_precision() == TensorFloat32` after setting `PEDANTIC_MATH`; the
printed precision label alone does not prove TF32 execution.

### 4.2 Python / PyTorch

Freeze file: `$WORK/epicazio/venvs/pytorch-scaling/requirements-freeze.txt`,
SHA-256 `cd39f2829c56037a3c68d05299f6d823f4ec603d7422c5d377104fc48d8c7532`
(668 bytes; copy in `provenance/venv-requirements-freeze.txt`).

| Item | Value |
|---|---|
| Python | `3.11.17` |
| numpy | `2.4.6` |
| torch | `2.9.1+cu126` |
| torchvision | `0.24.1+cu126` |
| CUDA runtime (torch) | `12.6` |
| cuDNN (torch) | `9.10.2.21` (`torch.backends.cudnn.version()=91002`) |
| NCCL (torch) | `2.27.5` |
| TF32 flags in the generator | `cudnn_allow_tf32=False`, `cuda_matmul_allow_tf32=False`, `float32_matmul_precision='highest'` |

Julia and PyTorch use **different cuDNN libraries** (9.26.0 vs 9.10.2.21).

### 4.3 GPU and driver (compute-node probe)

Probe job `60892511`, node `lrdn0252`, `CUDA_VISIBLE_DEVICES=0`:

| Item | Value |
|---|---|
| GPU | `NVIDIA A100-SXM-64GB` |
| Memory | `65536 MiB` |
| Compute capability | `8.0` |
| Driver (`nvidia-smi`) | `535.274.02` |
| CUDA availability | `True` |

Full output: `provenance/environment_probe.txt`; the raw probe script is
`provenance/compute_probe.sh`.

## 5. Preserved files and SHA-256

All files below were hashed remotely, copied with `rsync`/`scp` (no `--delete`),
and re-hashed locally. **All 24 evidence files MATCH.** The authoritative list
is the remote-layout `SHA256SUMS` copied into the local logs directory; the
table below reproduces it.

| Remote path | Bytes | SHA-256 | Local status |
|---|---:|---|---|
| `reference/phase3_reference.npz` | 299701312 | `97e5007cbb557f4c4a476a00c7e8076e8c59858e58d597d0df88529985da0783` | MATCH |
| `reference/phase3_reference_manifest.json` | 95759 | `2416e5557eb696e499a553fd1866d49c1c529a5fb778541df2f009825689f64d` | MATCH |
| `diag/stage2_diag.npz` | 56625286 | `a5dd592d7427ccdfba6a68dd3e8c9985a4466e8f7eca067095e5fc96d11a9b7e` | MATCH |
| `logs/remote-check.log` | 44677 | `c8fe2088dcb8746bf0640e720d72677369f7556435f643ee85d2211ca031d06b` | MATCH |
| `logs/2026-10-17-phase3-remote-attempt3.log` | 19845 | `3325a3575c9be7ec08748e76b29a48c3328b538c3ede8630b88535276751a4bf` | MATCH |
| `logs/2026-10-17-phase3-diagnose-gpu.log` | 1790 | `ae17a23e18f7665209342045b041f0745b452f96f718dc3072e37c796b831d72` | MATCH |
| `logs/2026-10-17-phase3-diagnose-stage2.log` | 2376 | `65b2ff78b1ac966342b9b2afc337accc8e18654a13587c72d7bbe472e4636bf0` | MATCH |
| `logs/2026-10-17-phase3-diagnose-stage2-cpu.log` | 2003 | `6cbbc3d99072c9aef8c9350607e9c0ec984f952fa24b4836a4c488e2aea5ab6d` | MATCH |
| `provenance/00_git_probe.txt` | 2370 | `b774db1c03d28717b3c469a77612c65777665a51333b3756aab072c98d468cc7` | MATCH |
| `provenance/05_attempt3_launch_command.txt` | 1100 | `3f002a04fdd4f504aabbf3c8da112ba9a7b0bb762bbc3f477c721440ec51a9ad` | MATCH |
| `provenance/10_source_files_sha256.txt` | 2485 | `8086f6c7ea238546312bdf87a383aed7e17528ad041bbe6c5cc6fad5cb616d8a` | MATCH |
| `provenance/15_tree_hashes.txt` | 757 | `c3c0c34770be71d650f870a2b2f21d790a935b0f37edd856da793303e0bcb8af` | MATCH |
| `provenance/20_julia_package_versions.txt` | 1502 | `4eb104ba91c61e3f1ab3109a16e67515ece9f4e08aa7648b18957518ff14a056` | MATCH |
| `provenance/30_cudnn-math-mode.txt` | 1423 | `1848a0b8964cba70ad22b175ae785ac7cd3be4018b4258eb2444a088b60a109c` | MATCH |
| `provenance/40_preservation_commands.txt` | 3010 | `c58966f6f6a0c13061b60612710bb7e532a49081b98b618580bfe97a56bb9e33` | MATCH |
| `provenance/50_missing_and_flags.txt` | 3190 | `c3b029fc1d98d47311f4d84c552ae5f0e4babfe2756af06cd6d268480486ce7a` | MATCH |
| `provenance/compute_probe.sh` | 2144 | `6c3a314991fd9baad126e9c793b14b292543614812c84d84f9aa90d6372cb318` | MATCH |
| `provenance/environment_probe.txt` | 861 | `10f54a074cd3609af4085840d7d4afcf2997e613ac95f91579e56dc6503aa59e` | MATCH |
| `provenance/environment_probe_math_mode.txt` | 576 | `38118e64501d668e5b662ae2e6d498273f34ada016510e57d307e1d0fb0273aa` | MATCH |
| `provenance/flux-root-Manifest.toml` | 93222 | `4c4eaf77633c493d8463a1a57236197423a8995f92f92eff00879475fdce5837` | MATCH |
| `provenance/flux-test-Project.toml` | 1509 | `13970e0a9b8595595ce63a9e05d69de0b381fff86d16f774c4e0d04d8bb38853` | MATCH |
| `provenance/julia-env-Manifest.toml` | 4669 | `c8077e0c6504adedfdce5a35dd1a2ba6bde057fc45ee9bd7ba1edbd35ee23874` | MATCH |
| `provenance/math_mode_probe.sh` | 1171 | `1c8a36463b9a98fcb416681b63ee6a67861415204d31a0cf25d908b315f41959` | MATCH |
| `provenance/venv-requirements-freeze.txt` | 668 | `cd39f2829c56037a3c68d05299f6d823f4ec603d7422c5d377104fc48d8c7532` | MATCH |

Remote `sha256sum -c SHA256SUMS` reported `OK` for every file before
`chmod -R a-w`. The local comparison recomputed each local file with
`sha256sum` and compared it with a fresh remote `sha256sum`; all 24 were
`MATCH`. `PROVENANCE.md`, `PROVENANCE.json`, and the final `SHA256SUMS` were
added after that comparison; their SHA-256 values are recorded in the final
remote `SHA256SUMS`.

## 6. Exact commands

### 6.1 Historical outer launch (verbatim, log line 1)

```bash
Executing on leonardo: JULIA_DEPOT_PATH=/leonardo_work/IscrC_UNMASKED/epicazio/julia-depot \
PATH=/leonardo_work/IscrC_UNMASKED/epicazio/julia-1.12.6/bin:$PATH SLURM_MPI_TYPE=pmi2 \
JULIA_PKG_PRECOMPILE_AUTO=0 salloc --account=iscrc_unmasked --partition=boost_usr_prod \
--qos=boost_qos_dbg -N1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=00:30:00 \
srun --mpi=pmi2 -n1 -c8 --chdir=$HOME/projects/ddp-flux-thesis bash -c \
'SOURCE_COMMIT=3088f61020137912af32d9c969e87ea9ae2a71c3 \
SOURCE_STATE_SHA256=82623bbb516a981d6c795792d1433521d25a8ee3bbcc44a2b66f6f4631a452b9 \
FLUX_COMMIT=f7b4520c4bd307c67bbc5ac92a1c68493eccff6e \
FLUX_STATE_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 \
bash scripts/remote/leonardo/run-phase3-check.sh'
```

### 6.2 Diagnostics (from the local driver logs)

```bash
# forward-only GPU diagnostic, job 60887065, node lrdn0339
scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 bash scripts/remote/leonardo/diagnose-phase3.sh --device cuda"

# stage-2 torch+Flux diagnostic, job 60887764, node lrdn0339 (wrote stage2_diag.npz)
scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 bash scripts/remote/leonardo/diagnose-phase3-stage2.sh"

# stage-2 CPU Flux diagnostic, job 60888250, node lrdn2086
scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 bash scripts/remote/leonardo/run-julia-phase3.sh \
  scripts/static_scaling/diagnose_phase3_flux_stage2.jl \
  --fixture-dir \$HOME/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 \
  --bundle \$HOME/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz \
  --reference \$HOME/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-17-phase3/phase3_reference.npz \
  --diag \$HOME/projects/ddp-flux-thesis/artifacts/logs/static-scaling/2026-10-17-phase3/stage2_diag.npz \
  --device cpu"
```

### 6.3 Preservation and probes

See `provenance/40_preservation_commands.txt` for the full command list.
Probe jobs: `60892511` (node `lrdn0252`, environment) and `60893401`
(node `lrdn0042`, math-mode mapping). Both used
`scripts/remote/run.sh leonardo "srun --mpi=pmi2 -n1 -c8 bash <probe>"`.

### 6.4 Local reopening check

```bash
python3 - <<'PY'
import numpy as np
for p in (".../phase3_reference.npz", ".../stage2_diag.npz"):
    z = np.load(p, allow_pickle=False)
    print(p, len(z.files))
PY
```

Result: `phase3_reference.npz` reopened with **337 arrays** (74,900,851
elements; dtypes float32/float64/int64); `stage2_diag.npz` reopened with
**9 arrays**, each `(12, 128, 32, 32)` float32
(`conv1, gn1, relu1, conv2, gn2, proj, pgn, total, out`).

## 7. Missing, blocked, and flagged provenance

- **Remote git identity unavailable** — both synced trees lack `.git`; all
  remote git commands fail with exit 128. Historical commit/state values are
  caller-supplied and cannot be re-derived from the remote copy.
- **False-positive empty-diff hash** — the remote pipeline printed
  `e3b0c44…b855` for `git diff HEAD | sha256sum` with a failing git exit code.
  It is not a valid clean-diff proof.
- **`~/projects/flux-integration/test/Manifest.toml` missing** remotely and
  locally. The flux test project resolves through
  `~/projects/flux-integration/Manifest.toml`
  (`4c4eaf77…5837`), which is preserved.
- **Historical source state not reproducible** — current local state
  `fe176b0c…` differs from `82623bbb…` because documentation was added after
  job `60885091`. The historical value identifies that run only.
- **cuDNN.jl `version()` parser** — raw `92600` prints as `92.60.0`; the
  authoritative library version is `CUDNN_jll 9.26.0+0`.
- **Julia vs PyTorch cuDNN** — `9.26.0` vs `9.10.2.21`; not one shared library.
- **CUDA.jl runtime/driver view** — `12.9.0` / `13.3.0` as reported, versus
  host driver `535.274.02` from `nvidia-smi`.
- **One intentional null** — `optimizer.gradient_clipping = null` in the
  manifest means no clipping. No other null, empty, or `unknown` provenance
  values were found (recursive scan of the manifest).
- **Immutability is best-effort** — `chmod -R a-w` on the remote root.

## 8. Exact next action

Proceed to **Step 2** of
`temp/static-graph-gpu-scaling-phase3-handoff.md`: on a Leonardo compute node,
run the isolated stage-2 convolution with the current math type and with
explicit `CUDNN_FMA_MATH`, using identical saved inputs/weights and fresh
processes, recording requested and effective math types and the discrepancy
against the unchanged PyTorch reference. Do not rerun the full Phase 3 driver
until that comparison is recorded.

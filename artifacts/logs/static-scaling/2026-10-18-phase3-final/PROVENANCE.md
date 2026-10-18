# Provenance — static-scaling Phase 3 final acceptance (2026-10-18)

All values in this file were recorded before or during the acceptance run.
No value is estimated. The final reference manifest carries the same
acceptance-time source identities.

## 1. Source identity (thesis)

| Item | Value |
|---|---|
| Thesis HEAD | `3088f61020137912af32d9c969e87ea9ae2a71c3` |
| Worktree state at acceptance | dirty: 93 `git status --short` entries, all Phase 2/3 work; no unintended changes and no secrets found (tracked diff plus untracked-file scan) |
| `git diff --check` | clean |
| Source-state SHA-256 at acceptance | `4a28adb1b031ddacb14aff3bce314d204e693c2d55910cdfe766d5ff33f131fe` |

State recipe (exactly as executed):

```bash
STATE=$( { git diff HEAD; git ls-files --others --exclude-standard \
  | grep -v '^artifacts/' | sort | xargs -r sha256sum; } \
  | sha256sum | cut -d' ' -f1 )
```

The reference manifest records `source_state_sha256=4a28adb1...`. Documentation
edits made after acceptance (this evidence package and the status notes) move
the live worktree hash; that does not change the acceptance-time identity.

A separate local acceptance check rejected missing, empty, null, and
`unknown` provenance values. All four manifest identities are concrete
([`provenance-check.log`](provenance-check.log)):

```text
source_commit='3088f61020137912af32d9c969e87ea9ae2a71c3' ok=True
source_state_sha256='4a28adb1b031ddacb14aff3bce314d204e693c2d55910cdfe766d5ff33f131fe' ok=True
flux_commit='f7b4520c4bd307c67bbc5ac92a1c68493eccff6e' ok=True
flux_state_sha256='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855' ok=True
PROVENANCE_CHECK PASS fields=4 bad=[]
```

## 2. Source identity (Flux)

| Item | Value |
|---|---|
| Worktree | `/home/kurapica/Projects/ddp_flux/flux-integration` |
| Branch | `ddp/integration-m1-spike` |
| HEAD | `f7b4520c4bd307c67bbc5ac92a1c68493eccff6e` |
| State | clean: no tracked modifications, no untracked files |
| Diff SHA-256 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty) |
| Untracked-list state SHA-256 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty) |
| `git diff --check` | clean |

Flux state recipe:

```bash
FLUX_STATE=$( { git -C ../flux-integration diff HEAD; \
  git -C ../flux-integration ls-files --others --exclude-standard \
  | sort | xargs -r sha256sum; } | sha256sum | cut -d' ' -f1 )
```

The empty hash `e3b0c442...` is the SHA-256 of empty input, so it proves the
clean worktree. The remote Leonardo copy is an rsync copy without `.git`;
remote source files were verified by SHA-256 instead (section 4).

## 3. Phase 3 source hashes

The complete list is in [`source-sha256.txt`](source-sha256.txt). Key files:

```text
60e003ced5cb50fd546edddc2976bc403ee3d855977274f0bb09d7f34946916d  scripts/static_scaling/precision.jl
16a9a4c51f78e8d2f659a06c4fc575ee211bcdc4bc6cd69c5f979967c7a71c44  scripts/static_scaling/check_phase3_flux.jl
0a9a2645f2bf0bd66e94e1351d3538e050a8261dee451c77dd4b7b282cdab21d  scripts/static_scaling/make_phase3_reference.py
85297025dfd9b2222e2ecbae6c5bdca80f0e724f80b680fcdc67fa6653723515  scripts/static_scaling/check_phase3_reference.py
f52c454de1ecbadaa45a7c7be9556ab59ab0628d99269e3d0a0ad2abc5432f6b  scripts/remote/leonardo/run-phase3-check.sh
ddb4e3b8a107341ccb8f1004248450836dc1bad9a88b0b6837169adfd65212b1  scripts/remote/leonardo/precompile-phase3.sh
8bfccb66395f13542330cb4c975c8419c5085a83f3d4d25b8fcc58bd3b47c90d  scripts/remote/leonardo/run-phase3-precision-regression.sh
4a70f3acc0205c975e242370ef36895d76fbaf0f5d20bb81ce2ac0b0078c4e6d  scripts/remote/leonardo/run-phase3-optimizer-probe.sh
3330437c1db470e6d4680ced961bcc99b2d1342f1157e5f1270fb7ab2a26e4c9  experiments/static-scaling/experiment.toml
a542988a0c99f111b4211e53209fe8e94613c7e8356d1474da9a0a127e610f13  experiments/static-scaling/specification.md
631ce1e00573eb84e53ed2b8e62186e9c606f88b95549961cbae16da8926fb06  experiments/static-scaling/open-decisions.md
```

## 4. Remote source verification

Before any run, the remote hashes of the key Phase 3 files were compared with
the local values. All matched:

```text
MATCH scripts/static_scaling/precision.jl                         60e003ce...
MATCH scripts/static_scaling/check_phase3_flux.jl                 16a9a4c5...
MATCH scripts/remote/leonardo/run-phase3-check.sh                 f52c454d...
MATCH scripts/remote/leonardo/precompile-phase3.sh                ddb4e3b8...
MATCH scripts/remote/leonardo/run-phase3-precision-regression.sh  8bfccb66...
MATCH scripts/remote/leonardo/run-phase3-optimizer-probe.sh       4a70f3ac...
MATCH experiments/static-scaling/probes/probe_optimizer_contract_cross.jl 243e69e8...
MATCH experiments/static-scaling/probes/probe_optimizer_contract_cross.py 7e3a58f9...
MATCH experiments/static-scaling/probes/compare_optimizer_contract.py     56c7e043...
MATCH experiments/static-scaling/probes/probe_optimizer_contract.jl       e51b8ea6...
MATCH experiments/static-scaling/experiment.toml                  3330437c...
MATCH experiments/static-scaling/specification.md                 a542988a...
MATCH experiments/static-scaling/open-decisions.md                631ce1e0...
```

## 5. Reference artifact identity

| Item | Value |
|---|---|
| Reference NPZ | `phase3_reference.npz` |
| Reference NPZ SHA-256 | `97e5007cbb557f4c4a476a00c7e8076e8c59858e58d597d0df88529985da0783` |
| Manifest | `phase3_reference_manifest.json` |
| Manifest SHA-256 | `5ba6c29f3a6167f4341e8e78b0685cb93a5b76e83575beca8db82d8585753eb7` |
| Arrays | 337 |
| Reference device | `cuda:0`, `NVIDIA A100-SXM-64GB` |
| Reference generator | `scripts/static_scaling/make_phase3_reference.py` |
| Reference created (UTC) | `2026-10-18T01:27:26Z` |
| Training fixture | batch 12, three updates, W=975, T=9750 |

The final NPZ is byte-identical to the preserved attempt-3 reference
(`artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/phase3_reference.npz`,
SHA-256 `97e5007c...`): 337/337 arrays exact-equal, maximum absolute
difference `0.0` (`reference-comparison.txt`).

## 6. Runtime environment

Recorded on Leonardo compute node `lrdn0125` in job `60959396`
([`runtime-versions.txt`](runtime-versions.txt)):

| Item | Value |
|---|---|
| GPU | `NVIDIA A100-SXM-64GB`, 65536 MiB, compute capability 8.0 |
| Driver | `535.274.02` (nvidia-smi reports CUDA 12.2 driver API) |
| Julia | `1.12.6`; depot `$WORK/epicazio/julia-depot` |
| Flux | `0.16.11` (worktree `f7b4520c`) |
| CUDA.jl | `6.2.2` |
| cuDNN.jl | `6.2.2`; loaded cuDNN library version reported as `92.60.0` |
| NNlib | `0.9.45` |
| Optimisers | `0.4.9` |
| Python | CPython `3.11.17` |
| PyTorch | `2.9.1+cu126`, CUDA 12.6, cuDNN `9.10.2.21` |
| NumPy | `2.4.6` |
| NCCL (PyTorch wheel) | `2.27.5` |
| Flux root Manifest SHA-256 | `4c4eaf77633c493d8463a1a57236197423a8995f92f92eff00879475fdce5837` |
| Flux root Project.toml SHA-256 | `e950f5d92f42024f8c54faf484d01a56a06d4647e1ab1f6782c4aeef62e3104b` |
| Flux test Project.toml SHA-256 | `13970e0a9b8595595ce63a9e05d69de0b381fff86d16f774c4e0d04d8bb38853` |
| Thesis `julia-env` Manifest SHA-256 | `c8077e0c6504adedfdce5a35dd1a2ba6bde057fc45ee9bd7ba1edbd35ee23874` |
| Thesis `julia-env` Project.toml SHA-256 | `6e75702e0e98a6fe3a3c253a278be827f771295d950dee67eae38367bb8f032d` |
| Python freeze SHA-256 | `cd39f2829c56037a3c68d05299f6d823f4ec603d7422c5d377104fc48d8c7532` (28 lines) |

The manifest hashes and the Python freeze are extracted unchanged into
[`julia-manifest-hashes.txt`](julia-manifest-hashes.txt) and
[`python-freeze.txt`](python-freeze.txt); both are excerpts of
[`runtime-versions.txt`](runtime-versions.txt).

The acceptance command verified TF32 is disabled on both sides:
`cuda_matmul_allow_tf32=False`, `cudnn_allow_tf32=False`,
`float32_matmul_precision='highest'`, `cudnn_deterministic=True`.
The Flux checker verified `CUDA.math_mode()=PEDANTIC_MATH`,
`cudnn_requested=CUDNN_FMA_MATH(3)`, `cudnn_effective=CUDNN_FMA_MATH(3)`,
`CUDA.allowscalar(false)`, and empty algorithm caches before selection.
The printed `cuda_math_precision=TensorFloat32` is the CUDA.jl preference
label; it is not used as evidence of TF32 execution. The effective controls
are the pedantic cuBLAS mode, the FMA cuDNN math type, and the recorded
FMA-only algorithm choices (32/32, no fallback).

## 7. Evidence locations

| Item | Path |
|---|---|
| Remote evidence root | `$WORK/epicazio/static-scaling-evidence/2026-10-18-phase3-final/` |
| Local reference | `artifacts/baselines/static-scaling/2026-10-18-phase3-final/` |
| Local logs | `artifacts/logs/static-scaling/2026-10-18-phase3-final/` |
| Preserved attempt-3 reference | `artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/` |

The remote-to-local SHA-256 comparison is in
[`SHA256-COMPARISON.txt`](SHA256-COMPARISON.txt): 15/15 files MATCH. The
remote values are the transcript in [`remote-sha256.txt`](remote-sha256.txt).
The acceptance-record provenance check is
[`provenance-check.log`](provenance-check.log). Leonardo SSH returned the
documented publickey rejection again immediately after the evidence copy; no
remote artifact was needed after that point.

# Phase 1 shared specification drafted (static scaling)

Date: 2026-10-15
Scope: static-network GPU scaling (scope 3), Phase 1 only
Plan: [implementation phases](../../../temp/static-graph-gpu-scaling-implementation-phases.md) (section 3)
Status: Phase 1 draft complete; 13 open decisions recorded with proposed defaults

## Goal

Implement Phase 1 of the static scaling plan: freeze one shared experiment
specification that both implementations can reproduce. Deliverables: a shared
machine-readable configuration, a readable specification, and an open-decision
table.

## What was created

All files are new under `experiments/static-scaling/`:

| File | Content |
|---|---|
| `specification.md` | Readable contract: matrix, data schedule, input pipeline, architecture, loss/labels, optimizer and schedule, gradient averaging, resources, memory policy, source pins, guard behavior, tolerances; a task map covers all 12 Phase 1 items |
| `experiment.toml` | Machine-readable config; readable by the Julia standard library TOML and Python 3.11 `tomllib`; every value carries a status (`recorded`/`proposed`/`pending_phase_N`) |
| `open-decisions.md` | 13 pending decisions with proposed defaults, rationale, and blocking phase |

Key recorded rules:

- Architecture: small-image ResNet-18, 3x3 stride-1 bias-free stem, no max-pool,
  two BasicBlocks per stage at widths 64/128/256/512, 1x1 projections with
  GroupNorm on stages 2-4, GroupNorm-32 eps 1e-5 everywhere, global average pool,
  biased 200-class classifier. Trainable parameters: **11,271,432**
  (convolutions 11,159,232 + GroupNorm affine 9,600 + classifier 102,600);
  62 unique trainable arrays.
- Data: 99,840 samples per epoch, 160-image tail dropped everywhere, 195 updates
  per epoch at batch 512 and 390 at 256; one saved permutation per epoch; one
  saved augmentation decision per (epoch, sample) replayed by both frameworks.
- Optimizer: Optimisers.jl `AdamW` with the default `couple = true` matches
  `torch.optim.AdamW` exactly (`x = x - eta*(dx + lambda*x)`); proposed
  `lr = 1e-3`, `wd = 1e-4`, `betas = (0.9, 0.999)`, `eps = 1e-8`; no decay for
  GroupNorm affine arrays and the classifier bias.
- Schedule: linear 5-epoch warm-up and cosine decay, functions of the global
  update index; proposed 50-epoch learning-curve budget (`T` = 9,750 at batch
  512).
- Shared format: `.npz` plus a JSON manifest; PyTorch-native canonical layout;
  documented Flux permutations; per-array SHA-256 of raw C-order bytes.
- Resources: fixed total CPU budget of 32 cores per job (32/16/8 CPUs per task
  at one/two/four GPUs), 1 compute thread and 4 loader workers per rank, Slurm
  default affinity, `--gres=gpu:4`, Flux device mapping `mod1(r + 2, n)`.
- Memory: proposed margin of 20% free (peak at most 80% of 64 GiB), counters
  recorded per category, unavailable counters recorded as unavailable.
- Tolerances: same-framework 2e-6 / 1e-5 (project standard); cross-framework
  provisional 1e-5 / 1e-4, calibrated on frozen fixtures before Phase 3
  acceptance.
- Entry point: explicit `Zygote.withgradient` + `Optimisers.update!`; the
  ADR-0009 gradient guard runs inside each update (1 presence + 62 gradient
  allreduces + 1 vote = 64 collectives); the eager `Flux.train!` path would add
  the ADR-0008 loss vote (65).

## Source pinning

- Flux worktree `../flux-integration` is **clean** at
  `f7b4520c4bd307c67bbc5ac92a1c68493eccff6e` (branch `ddp/integration-m1-spike`),
  which contains both committed guards (DB-1/DB-4 `314aef20`, DB-3 `f7b4520c`).
  This supersedes the older wiki statements that pin the worktree dirty at
  `55171e65` or `314aef20`.
- Julia 1.12.6; remote resolved manifest sha256 `4c4eaf77...`; the local
  `flux-integration` manifest is stale (no CUDA/NCCL/cuDNN) and must not be used.
- Python freeze record `artifacts/logs/leonardo/2026-10-15-pytorch-env-freeze.txt`
  (28 lines, sha256 `cd39f282...`); torch 2.9.1+cu126, torchvision 0.24.1+cu126.
- **Prerequisite:** the Leonardo copy of `flux-integration` is a pre-guard
  snapshot (verified 2026-10-15). Before any Phase 3+ run: sync `f7b4520c`,
  resolve/instantiate, precompile on a compute node, re-record hashes.
  `scripts/remote/sync_code.sh` still points at the frozen `ddp_flux` and must
  not be used unchanged for this benchmark.

## Verification

- Parameter count recomputed twice (author and an independent verification
  subagent) with different scripts; both give 11,271,432 and match the
  proposal's TorchVision formula.
- Independent verification (fresh `explore` subagent, adversarial checklist):
  all 9 checks PASS, including source-pin hashes, guard line references,
  collective arithmetic, data arithmetic, cross-file consistency, link targets,
  and TOML parsing in Julia and Python.
- One wording defect found by the verifier and fixed: the config and spec
  claimed ADR-0009 records hashes for `train.jl` and `FluxMPIExt.jl`; ADR-0009
  records hashes only for `public_api.jl`, `FluxMPINCCLExt.jl`, NEWS.md, and the
  DB-3 test. The wording now states which files ADR-0009 covers.
- TOML parsed by `python3` `tomllib` and Julia `TOML`; all 19 relative links
  resolve.

## Failures and deviations

- The `research-docs` and `research-web` subagent types failed again with HTTP
  403 `SUBSCRIPTION_REQUIRED` (same failure as the environment-setup session).
  The work was completed with `explore` subagents, pinned local package sources
  (Optimisers 0.4.9, Flux, NNlib, CUDA.jl 6.x), and direct web search.
  The precedent record is
  [wiki/failures/2026-10-15-research-agent-model-unavailable.md](../../failures/2026-10-15-research-agent-model-unavailable.md).
- No remote command, no training, no Flux source change, no commit.

## Exact next action

Record the open decisions in `open-decisions.md` (at minimum OD-3 groups, OD-5
normalization constants, OD-6 augmentation set, OD-10 cross-framework tolerance,
OD-11 entry point), then start Phase 2: data adapters, saved epoch permutations,
saved augmentation records, and the shared correctness fixtures in the `.npz`
plus manifest format.

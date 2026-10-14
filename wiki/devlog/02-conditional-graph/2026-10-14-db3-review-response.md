# DB-3 review response — protocol pin, healthy reference, contract docs

Date: 2026-10-14

Responds to all three findings of the
[DB-3 implementation review](../../../reports/conditional-graph-db3-review.md)
(R1/R2 P2, R3 P3).
R1 is a thesis change, R2 is test-only, and R3 is documentation-only; the
production guard code did not change. No commit was made; the worktrees remain
dirty by project convention. All gates below ran on `deathstar` with 0
`Precompiling packages...` lines.

Related records:
[review devlog](2026-10-14-db3-review.md),
[gradient-rejection devlog](2026-10-14-db3-gradient-rejection.md),
[ADR-0009](../../decisions/ADR-0009-post-reduction-nonfinite-gradient-rejection.md),
[ADR-0008](../../decisions/ADR-0008-coordinated-nonfinite-loss-rejection.md),
and the
[R1 failure note](../../failures/2026-10-14-db3-review-response-r1-shape-and-artifacts.md).

## R1 — protocol-aware artifact reader (thesis repository)

Before: `collective_summary` required `4 + 2 * present` collectives for every
record, so both saved pre-DB-3 tutorial bundles failed at step 1.

Changed files:

- `notebook/phase_k_workflow_helpers.jl`
  (md5 `b570d0777109a41eee5bb1be3b4d53d5`): `COLLECTIVE_PROTOCOLS` (`legacy-5/7`:
  base 3, `vote_required=false`; `guarded-6/8`: base 4, `vote_required=true`);
  both `PHASE_K_BUNDLES` tutorials pin `legacy-5/7`;
  `collective_protocol(bundle)` errors on a missing or unknown pin;
  `expected_collectives(record, protocol)` = `protocol.base + 2 * present`;
  `collective_summary` requires, for guarded bundles, that the final collective
  of every (mode, step, rank) is the host `Vector{Float32}` shape `(1,)` `avg`
  gradient-validity vote; new `shape_dims(text)` parses the console shape text
  (`(1,)` keeps a trailing comma, so a one-dimensional shape parses as `"1,"`).
- `scripts/checks/phase_k_collective_protocol_check.jl` (new, stdlib only, md5
  `923b7f49c20fa763a0e4727d27191331`): passes both tutorials, runs the guarded
  DB-3 toy bundle, and requires wrong or missing pins to throw.

The first deathstar run failed for two reasons: the remote 4-rank tutorial
bundle was incomplete, and the vote check compared the raw `(1,)` shape capture
against `"1"`. Both causes, the fixes, and the transcribed failed console are in
the
[failure note](../../failures/2026-10-14-db3-review-response-r1-shape-and-artifacts.md).
The two missing files (local sizes `trajectory.jls` 148521848 bytes,
`validation-train.tsv` 108911 bytes) were uploaded without `--delete`.

Final run, exit 0:

```
scripts/remote/run.sh deathstar "julia --startup-file=no scripts/checks/phase_k_collective_protocol_check.jl"

PHASE_K_PROTOCOL tutorial-2rank PASS
PHASE_K_PROTOCOL tutorial-4rank PASS
PHASE_K_PROTOCOL db3-toy-mpi2 PASS
PHASE_K_PROTOCOL guarded-pinned-legacy-5/7 PASS
PHASE_K_PROTOCOL guarded-without-protocol PASS
PHASE_K_PROTOCOL tutorial-2rank-pinned-guarded-6/8 PASS
PHASE_K_PROTOCOL_SUMMARY PASS
```

The guarded check ran against the real DB-3 guarded artifacts
`artifacts/logs/conditional-wrapper/demo/db3-toy-2026-10-14-mpi` (mpi2, 16
steps, 6/8 traces with the trailing vote); the negatives prove a guarded trace
pinned legacy, a missing pin, and a legacy trace pinned guarded all throw.
Log:
`artifacts/logs/conditional-wrapper/numerical-guards/db3-review-response/r1-protocol-check.log`.

## R2 — corrected healthy reference (Flux repository, test only)

`../flux-integration/test/ext_distributed/nonfinite_gradient_distributedtest.jl`
is now md5 `b0566261df12deb00cd69fc6d99220e9` (was
`06eef21d92c78ca85f718b325ddbe8c6`; 862 lines). Changes in testset 13 only:

- `ta = dev_fill(1.5, W_LEN)` (was `dev_ones`), so the `a` branch has nonzero
  gradients and moments.
- Step-dependent reference gradient:
  `local_ga = 2 .* (ref.a .- to_host(ta)); ga = step == 3 ? local_ga : local_ga ./ tworkers`
  (step 3 trains `a` on every rank, so there is no `/tworkers`).
- At `step == 2`, assert `any(!iszero, to_host(state.tree.a.state[1]))` and the
  same for `state[2]` before the step-3 absence comparison.

GREEN on deathstar, all exit 0; the healthy testset is 59/59 per rank on each
run:

```
scripts/remote/run.sh deathstar "FLUX_TEST_DISTRIBUTED_BACKEND=mpi ~/.julia/bin/mpiexecjl -n 2 julia --project=../flux-integration/test --startup-file=no ../flux-integration/test/ext_distributed/nonfinite_gradient_distributedtest.jl"

scripts/remote/run.sh deathstar "FLUX_TEST_DISTRIBUTED_BACKEND=mpi ~/.julia/bin/mpiexecjl -n 4 julia --project=../flux-integration/test --startup-file=no ../flux-integration/test/ext_distributed/nonfinite_gradient_distributedtest.jl"

scripts/remote/run.sh deathstar "FLUX_TEST_DISTRIBUTED_BACKEND=nccl ~/.julia/bin/mpiexecjl -n 2 julia --project=../flux-integration/test --startup-file=no ../flux-integration/test/ext_distributed/nonfinite_gradient_distributedtest.jl"
```

Logs:
`artifacts/logs/conditional-wrapper/numerical-guards/db3-review-response/focused-mpi2.log`,
`.../focused-mpi4.log`, and `.../focused-nccl2.log` (the nccl2 run also sets
`allowscalar(false)`).

RED control: a temporary copy `red_weighting.jl` with the old
`ga = local_ga ./ tworkers` fails the healthy testset: 53 pass / 6 fail / 59
per rank, exit 1, at the model-`a` comparison and both `a`-moment comparisons.
The temporary copy was deleted; the real file md5 matches remote and local
(`b0566261df12deb00cd69fc6d99220e9`).
Log: `.../red-mpi2-old-weighting.log`.

## R3 — public contract documentation (Flux repository, docs only)

Corrected hashes (old → new):

| File | Old | New |
|---|---|---|
| `src/distributed/public_api.jl` | `d21c73f8885b35ae8558a0f0d4e43196` | `48ea1fcd3c0629d1773b91ae9d6f1677` |
| `ext/FluxMPINCCLExt/FluxMPINCCLExt.jl` | `841c662047cbe86cd082ceeb226b6231` | `3bca62aa63f2b1d18e2f7f4eb630d221` |
| `NEWS.md` | `83e69bb91cc5edd110c7de9af2b0b87a` | `f9cd8bf52caf18a99d0b2b76cb13ee2b` |

All changes are docstrings/comments/prose only; no executable line changed
(verified by reverting the prose edits and reproducing the campaign md5s).
Both wrapper docstrings, `NEWS.md`, the NCCL extension comment, and the
[ADR-0009](../../decisions/ADR-0009-post-reduction-nonfinite-gradient-rejection.md)
implementation summary now state consistently: the all-rank
gradient-validity vote; coordinated `DomainError` on every rank before any
optimizer update; preservation of parameters/optimizer state/caller gradients;
the nonempty-plan rule (the vote runs for every update with at least one
trainable parameter, including all-gradients-globally-absent; only models with
no trainable parameters issue no collectives); and the CUDA scalar
device-to-host validity transfer (device reduction, only the scalar `Bool`
transfers, no full-buffer transfer, no scalar indexing).

Evidence: `git -C ../flux-integration diff --check` clean; the synced sources
precompiled successfully on deathstar (`make precompile-all` after sync; Flux
and FluxMPINCCLExt precompiled). No runtime test was needed for docs.

## Campaign mechanics

```
rsync -az --delete --exclude-from='.rsyncignore' --exclude='artifacts/' . deathstar:~/projects/ddp-flux-thesis/
rsync -az --delete --exclude='.git' --exclude='Manifest.toml' --exclude='test/Manifest.toml' ../flux-integration deathstar:~/projects/
scripts/remote/run.sh deathstar "make precompile-all FLUX_REPO_PATH=../flux-integration"
```

The thesis sync excludes `artifacts/`, so remote-only artifacts survive.
0 `Precompiling packages...` lines in every gate log. Deathstar was left clean
(no `julia`/`mpiexec`, GPUs idle at 6/6/16 MiB) and the RED temp file was
removed.

## Evidence map

All under
`artifacts/logs/conditional-wrapper/numerical-guards/db3-review-response/`:

- `r1-protocol-check.log`: final R1 acceptance run (the local log was
  overwritten by the successful rerun).
- `focused-mpi2.log`, `focused-mpi4.log`, `focused-nccl2.log`: R2 GREEN runs.
- `red-mpi2-old-weighting.log`: R2 RED control.

Commands, source identity, and the gate table:
[COMMANDS.md, DB-3 review response campaign](../../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md#db-3-review-response-campaign-2026-10-14-deathstar).

## Exact next action

Review the combined DB-3 diff against the corrected hashes, then decide on
committing the DB-3 change set and updating the upstream PR/review.

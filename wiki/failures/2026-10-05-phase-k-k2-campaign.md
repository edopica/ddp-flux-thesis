# Phase K K2 campaign: smoke resume checkpoint-identity failure

Date: 2026-10-05
Host: deathstar, 2-rank CPU/MPI smoke cell
Remote root: `/home/picazio/projects/ddp-flux-thesis/artifacts/logs/conditional-wrapper/demo/k2-smoke-2026-10-05-deathstar-mpi2/`
Local preserved copy: `artifacts/logs/conditional-wrapper/demo/k2-smoke-2026-10-05-deathstar-mpi2/`
Status: campaign aborted at the smoke cell. K1 regression, full 2/4-rank campaign, and the NCCL cell did not run.

## Summary

The smoke cell ran `prepared` (PASS, exit 0, 17.081 s) and 2-rank `train` (PASS, exit 0, 49.172 s), then the fresh-process `resume` cell failed with exit 1 after 18.064 s.
The failure is deterministic and aborts every resume: the checkpoint identity digest computed by `CGCheckpoint.checkpoint_digest` is not stable across a `Serialization` roundtrip.
The checkpoint payload is correct: all parameter, moment, and reference values match the saved uninterrupted step-2 snapshots exactly (`DDP_SNAP_MATCH=true`, `REF_SNAP_MATCH=true`).
No tolerance was changed. The smoke cell was stopped per the campaign failure rule.

## Verbatim failure

Driver assertion at `scripts/checks/conditional_graph_demo_check.jl:197`:

```text
Expression: occursin("DEMO_RESUME_DONE", resume.output)
```

Child error, both ranks (`mpi2-resume/console.log`):

```text
ERROR: ERROR: LoadError: LoadError: checkpoint payload does not match the one the trajectory was saved from
Stacktrace:
 [1] error(s::String)
   @ Base ./error.jl:44
 [2] ensure
   @ ~/projects/ddp-flux-thesis/scripts/examples/conditional_graph/common.jl:36 [inlined]
 [3] main_resume(opts::Dict{String, String}, flags::Set{String})
   @ Main ~/projects/ddp-flux-thesis/scripts/examples/conditional_graph_demo.jl:894
```

`DEMO_SCHEMA_OK` and `DEMO_TRAIN_DONE` markers are present for the train cell; `DEMO_RESUME_DONE` is absent and no resume output directory was created.
The train console printed the conflicting digest at save time:

```text
DEMO_CHECKPOINT step_saved=2 step_next=3 ... digest=7258b97e2420051df7a6ddac9c55eb604ef3e4c94013b9273e84fd093e41abb3
```

## Root cause

`CGCheckpoint.checkpoint_digest` (`scripts/examples/conditional_graph/checkpoint.jl:40-49`) serializes each digest field with Julia `Serialization` and hashes the bytes.
One field, `source_hashes`, is a `Dict{String,String}` produced by `demo_source_hashes()` (`conditional_graph_demo.jl:165-183`).
`Serialization` writes `Dict` entries in hash-table iteration order, and that order is not preserved across a serialize/deserialize roundtrip:

- The training process builds the `Dict` by insertion in a fixed order, then hashes it.
- The resuming process rebuilds the `Dict` by inserting the serialized entries, producing a different rehash history and iteration order.
- The two `Dict`s compare equal (`==` is true), but their serialized bytes differ, so the SHA-256 digest differs.

The first divergent item is `source_hashes`, the only `Dict` field in the digest. The payload values, schedule, dataset, optimizer, and snapshot data are unchanged.

## Evidence

Read-only diagnostics are preserved next to the failed run:

- `diagnostics/k2_digest_diag.jl` (whole-payload and per-field digests, roundtrip check, snapshot comparison)
- `diagnostics/k2_sourcehash_diag.jl` (insertion-order reconstruction of `source_hashes`)

Exact invocation used (standard input, test project, no MPI backend):

```bash
ssh -o BatchMode=yes deathstar 'env JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0 /home/picazio/.juliaup/bin/julia --startup-file=no --project=/home/picazio/projects/flux-integration/test -e "include_string(Main, read(stdin, String), \"digest-diag\")"' < diagnostics/k2_digest_diag.jl
ssh -o BatchMode=yes deathstar 'env JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0 /home/picazio/.juliaup/bin/julia --startup-file=no --project=/home/picazio/projects/flux-integration/test -e "include_string(Main, read(stdin, String), \"src-diag\")"' < diagnostics/k2_sourcehash_diag.jl
```

Key outputs:

```text
STORED_DIGEST=7258b97e2420051df7a6ddac9c55eb604ef3e4c94013b9273e84fd093e41abb3
FRESH_DIGEST=973d082bfe3bb0e2719c955b896b6fa21261b86f8fd3ab16131d20eae7bfea34
ROUNDTRIP_DIGEST=973d082bfe3bb0e2719c955b896b6fa21261b86f8fd3ab16131d20eae7bfea34
DDP_SNAP_MATCH=true
REF_SNAP_MATCH=true
DICT_EQUAL=true
DICT_BYTES_EQUAL=false
DIGEST_LOADED=973d082bfe3bb0e2719c955b896b6fa21261b86f8fd3ab16131d20eae7bfea34
DIGEST_REBUILT_SOURCE_HASHES=7258b97e2420051df7a6ddac9c55eb604ef3e4c94013b9273e84fd093e41abb3
DIGEST_REBUILT_BOTH=7258b97e2420051df7a6ddac9c55eb604ef3e4c94013b9273e84fd093e41abb3
```

Reconstructed insertion order (training process):

```text
common.jl, model.jl, dataset.jl, schedule.jl, reference.jl, checkpoint.jl,
metrics.jl, conditional_graph_demo.jl, conditional_graph_demo_check.jl,
flux-integration/src/Flux.jl, flux-integration/src/distributed/public_api.jl
```

Deserialized iteration order (resume process):

```text
model.jl, schedule.jl, Flux.jl, checkpoint.jl, common.jl, public_api.jl,
dataset.jl, metrics.jl, conditional_graph_demo.jl, reference.jl, checker
```

## Impact and scope

- Same-digest failure affects the toy K1 path and every K2 resume under schema 2, including CPU/MPI and CUDA/NCCL.
- The failure is in the K2 demo checkpoint identity check, not in Flux, the wrapper, the distributed reduction, or any numerical path.
- No numerical tolerance failure occurred; the train cell matched the independent reference with `peer_param_maxerr=0.000e+00`, `peer_mom1_maxerr=0.000e+00`, `peer_mom2_maxerr=0.000e+00`, and `beta_exact=true` on all 12 train rank-rows.

## Recommended correction (implementation owner)

Canonicalize the hashed representation before hashing instead of relying on `Dict` iteration order. For example, hash `sort!(collect(d))` for every `Dict`-typed digest field, or store `source_hashes` as a sorted vector of pairs in the payload.
Add a regression that builds the 11-key `source_hashes` Dict in insertion order, saves, loads, and asserts `checkpoint_digest(loaded) == checkpoint_digest(original)`; a small or single-key Dict does not expose this defect.

## Preserved evidence

- `summary.txt` (prepared exit 0; train exit 0; resume exit 1; `status FAIL`)
- `prepared/` (console, prepared artifact, metadata, sources, subset indices, schedule)
- `mpi2-train/` and `mpi2-resume/` consoles and `command.txt`
- `mpi2/train/` (checkpoint, trajectory, metrics, validation, learning rate, predictions, metadata)
- `diagnostics/` (both diagnostic scripts)

No orphan Julia or `mpiexec` processes remained after the failure, and all three GPUs were idle (0%, 6/6/16 MiB).

## Resolution (2026-10-05)

The implementation owner fixed the digest defect in `scripts/examples/conditional_graph/checkpoint.jl`: `canonical_digest_value` now canonicalizes `AbstractDict` (sorted by `string(key)`), `NamedTuple`, `Tuple`, and `AbstractVector` values before hashing, so iteration order cannot change the digest.
New pure-stdlib regression `scripts/checks/conditional_graph_checkpoint_roundtrip.jl` was RED pre-fix and GREEN post-fix; it covers three roundtrips, Dict insertion order, a nested Dict, and mutation sensitivity.
Premises of the fix: pre-fix checkpoints are intentionally incompatible, and the failed smoke root is evidence only.

Remote verification after sync (`checkpoint.jl` SHA-256 `d494550f...`, roundtrip script `a70bcf9e...`):

```text
CHECKPOINT_DIGEST 51cc18f34c48e59bee3538c5f404b8bea80c9e96437e80949c6291f9cdd47429
CHECKPOINT_ROUNDTRIP_STABLE roundtrips=3
CHECKPOINT_DICT_ORDER_CANONICAL entries=11
CHECKPOINT_NESTED_DICT_CANONICAL
CHECKPOINT_MUTATION_SENSITIVE
CHECKPOINT_ROUNDTRIP_OK
exit 0
```

Campaign resumed from fresh roots and completed on deathstar:

- `k2-smoke2-2026-10-05-deathstar-mpi2`: prepared/train/resume PASS (8/8 driver tests).
- `k1-regression-2026-10-05-deathstar-mpi2`: toy 2-rank prepared/train/resume PASS (8/8).
- `k2-2026-10-05-deathstar-mpi`: mpi2 and mpi4 prepared/train/resume PASS (13/13).
- `k2-2026-10-05-deathstar-nccl2`: CUDA/NCCL train/resume PASS (6/6), distinct devices `[1, 2]`.

Final per-cell evidence: `artifacts/logs/conditional-wrapper/demo/CAMPAIGN-2026-10-05-deathstar-k2.md`.

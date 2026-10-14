# DB-3 implementation review

Date: 2026-10-14

## Verdict

**Final verdict (2026-10-14): R1, R2, and R3 closed. No blocking findings remain within the reviewed scope.**
The initial findings and evidence follow as historical records.
The inspected production guard follows the central design in the DB-3 plan.
No production rejection defect was found in the inspected path.

Scope: the uncommitted DB-3 diff in `../flux-integration`, its regression tests, the thesis harness, and saved acceptance evidence.
Flux HEAD is recorded as `314aef2022333fa8f37035ed10cdc8e2360fa475` in the campaign.
The production-file hashes and focused-test hash match that campaign.

## R1 — P2: the protocol change breaks both artifact-mode tutorials

Location: `notebook/phase_k_workflow_helpers.jl:627-631`.

`expected_collectives` now requires `4 + 2 * present` for every saved record.
The two configured tutorial bundles contain pre-DB-3 evidence with `3 + 2 * present` collectives.
As a result, both tutorial configurations fail in `collective_summary` at the first training step.
This is a reproduced regression in the delivered artifact-mode notebook.

Reproduction from the thesis root:

```bash
julia --startup-file=no -e 'include("notebook/phase_k_workflow_helpers.jl"); using .PhaseKWorkflow; for key in ("tutorial-2rank", "tutorial-4rank"); try collective_summary(load_bundle(resolve_bundle(key))); println(key, " PASS"); catch e; println(key, " REPRODUCED: ", sprint(showerror, e)); end; end'
```

Output:

```text
tutorial-2rank REPRODUCED: train step 1 rank 1: expected 8 collectives, metrics record 7, console 7
tutorial-4rank REPRODUCED: train step 1 rank 0: expected 8 collectives, metrics record 7, console 7
```

Required correction: select the protocol from explicit campaign metadata or a pinned bundle version.
Historical bundles must retain the 5/7 protocol. New guarded bundles must use 6/8 and require the final host validity vote.
Accepting either count without a protocol identity can hide a missing vote in a new campaign.

## R2 — P2: a zero gradient hides an incorrect healthy reference

Location: `../flux-integration/test/ext_distributed/nonfinite_gradient_distributedtest.jl:732-734`.

At step 3, every rank computes the `a` loss (`step == 3 || rank == 0`).
Its synchronized gradient therefore equals `2 .* (ref.a .- ta)`, without division by `tworkers`.
The reference instead divides this gradient by `tworkers` at every step.

The fixture initializes both `model.a` and `ta` to ones (lines 706 and 712).
Consequently, `a` has zero gradients and zero moments throughout the test.
Both formulas return zero, so the test passes despite its incorrect reference.
It cannot detect incorrect weighting on the active branch during the globally absent step.

Required correction: initialize `a` away from its target and use the correct step-dependent weighting.
Assert nonzero moments before the absence step, then retain the complete per-step state comparisons at two and four ranks.
The separate healthy harness provides additional coverage, but does not correct this focused regression.

## R3 — P3: public documentation does not describe the implemented contract

Locations:

- `../flux-integration/src/distributed/public_api.jl:375-378,414-417`: both wrapper docstrings omit the gradient vote and rejection contract.
- `../flux-integration/NEWS.md:24-25`: the entry limits the extra collective to updates that reduce a gradient buffer.
- `../flux-integration/ext/FluxMPINCCLExt/FluxMPINCCLExt.jl:87-90`: the comment says that no host transfer occurs.

The vote also occurs when trainable buffers exist but all gradients are globally absent.
CUDA reduces finiteness on the device, then transfers the scalar result for host control flow.
The comment must distinguish that scalar transfer from a full-gradient transfer.
ADR-0009 describes the scalar transfer correctly in its device rules, but contradicts itself in its implementation summary.

Required correction: document the nonempty-plan rule, coordinated `DomainError`, preservation guarantee, and scalar device-to-host transfer consistently.

## Production design and evidence

- Both update APIs call `_sync_gradients` before the inner optimizer.
- The guard follows the complete reduction loop and precedes `_build!`.
- The scan uses canonical buffers and checks every globally present buffer.
- `_any_rank_invalid` uses the required any-positive vote.
- All-absent gradients still vote. A plan without trainable buffers returns before gradient collectives.
- The CUDA extension uses a stream synchronization and a device reduction.
- The local NCCL source uses the communicator device's task-local stream by default.
- Saved native NCCL results cover NaN, both Inf signs, real AD, asymmetric injection, and continuation with scalar indexing disabled.

The inspected combined summaries report MPI two ranks 32/32, MPI four ranks 16/16, and NCCL two ranks 29 PASS plus 3 UNTESTED.
The UNTESTED cases are explicit: Duplicated eager training and the two native overflow fixtures.
The saved focused NCCL console reports passing testsets.
These are inspected campaign results, not independent runtime reruns from this review.
GPU overhead remains unmeasured.

## Independent checks

From `../flux-integration`:

```bash
git diff --check && md5sum src/distributed/public_api.jl ext/FluxMPINCCLExt/FluxMPINCCLExt.jl test/ext_distributed/nonfinite_gradient_distributedtest.jl
```

Result: exit 0. Hashes match the campaign:

```text
d21c73f8885b35ae8558a0f0d4e43196  src/distributed/public_api.jl
841c662047cbe86cd082ceeb226b6231  ext/FluxMPINCCLExt/FluxMPINCCLExt.jl
06eef21d92c78ca85f718b325ddbe8c6  test/ext_distributed/nonfinite_gradient_distributedtest.jl
```

The independent two-rank MPI attempt failed during `using Flux`, before test execution.
Julia precompilation reported a CUDA dependency `KeyError`.
The exact command and failure are in [the failure record](../wiki/failures/2026-10-14-db3-review-local-environment.md).
No local MPI acceptance claim follows from that attempt. No remote commands ran during this review.

## Exact next action

Decide whether to commit the reviewed DB-3 change set and prepare the upstream review.

## Independent closure review — 2026-10-14

All six corrected file hashes match the [response devlog](../wiki/devlog/02-conditional-graph/2026-10-14-db3-review-response.md).
Flux `git diff --check` passed.

- **R1 closed:** each bundle pins its protocol. Guarded traces require a final host Float32 length-1 average vote.
  Both historical tutorial summaries passed an independent local rerun.
  The saved remote log also records the real guarded toy trace and three negative controls as PASS.
- **R2 closed:** the target differs from the initial parameter, step 3 uses the full local gradient, and assertions require nonzero Adam moments.
  Saved focused logs show 59/59 healthy assertions per rank at mpi2, mpi4, and nccl2.
  The old-weighting control shows 53 pass and 6 fail per rank, which establishes that the corrected fixture detects the error.
- **R3 closed:** the public docs describe the nonempty-plan vote and coordinated rejection.
  The CUDA comment and ADR now distinguish a scalar validity transfer from a gradient-buffer transfer.

Independent local command (exit 0):

```bash
julia --startup-file=no -e 'include("notebook/phase_k_workflow_helpers.jl"); using .PhaseKWorkflow; for key in ("tutorial-2rank", "tutorial-4rank"); collective_summary(load_bundle(resolve_bundle(key))); println(key, " PASS"); end'
```

Output: `tutorial-2rank PASS` and `tutorial-4rank PASS`.
MPI and NCCL results in this closure are inspected remote evidence, not independent reruns.
The earlier local environment failure remains separate from product acceptance.
No production source changed during this closure review, and no commit was made.
